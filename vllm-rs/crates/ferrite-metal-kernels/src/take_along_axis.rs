// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! TakeAlongAxis (axis=-1, 2D contiguous) kernel dispatcher.
//!
//! Faithful port of mlx `GatherAxis::eval_gpu`
//! (`mlx/backend/metal/indexing.cpp:440`) specialized for the MoE
//! router shape: `src=[N, num_experts]` contiguous, `idx=[N, top_k]`
//! contiguous, axis=-1, output `[N, top_k]` contiguous.
//!
//! Used immediately after `argpartition` to read the per-row top-k
//! expert scores (`qwen3_next.py:342 mx.take_along_axis(gates, inds,
//! axis=-1)`).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

use crate::argpartition::ArgSortDtype;
use crate::shader_cache::load_library_from_bytes;
use crate::stream::MetalStreamError;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

pub struct TakeAlongAxisKernels {
    pub f32: ComputePipelineState,
    pub f16: ComputePipelineState,
    pub bf16: ComputePipelineState,
    _library: Library,
}

impl TakeAlongAxisKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library =
            load_library_from_bytes(device, crate::embedded_metallib!("take_along_axis")).map_err(
                |e| {
                    MetalStreamError::ShaderCompilationFailed(format!(
                        "load `take_along_axis.metallib`: {e}"
                    ))
                },
            )?;
        let bp = |n: &str| build_pipeline(device, &library, n);
        Ok(Self {
            f32: bp("gather_axis_cc_float32_uint32")?,
            f16: bp("gather_axis_cc_float16_uint32")?,
            bf16: bp("gather_axis_cc_bfloat16_uint32")?,
            _library: library,
        })
    }

    pub fn pipeline_for(&self, dtype: ArgSortDtype) -> &ComputePipelineState {
        match dtype {
            ArgSortDtype::F32 => &self.f32,
            ArgSortDtype::F16 => &self.f16,
            ArgSortDtype::BF16 => &self.bf16,
            // take_along_axis on integer values isn't a router need;
            // mlx's gather_axis instantiates for the value dtype, not
            // the index dtype, so a u32 value gather would need its
            // own shader instantiation. Surface clearly.
            ArgSortDtype::U32 => panic!(
                "take_along_axis: u32 value dtype not instantiated (router uses f32/f16/bf16)"
            ),
        }
    }
}

fn build_pipeline(
    device: &Device,
    library: &Library,
    name: &str,
) -> Result<ComputePipelineState, MetalStreamError> {
    let ns_name = NSString::from_str(name);
    let function = library
        .newFunctionWithName(&ns_name)
        .ok_or_else(|| MetalStreamError::ShaderCompilationFailed(format!("{name} fn missing")))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| MetalStreamError::ShaderCompilationFailed(format!("{name} pipeline: {e:?}")))
}

/// Dispatch `out[n, k] = src[n, indices[n, k]]`.
///
/// `src` is `[batch, axis_size]`, `indices` is `[batch, top_k]`
/// (uint32), `out` is `[batch, top_k]`. All contiguous, row-major.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_take_along_axis(
    kernels: &TakeAlongAxisKernels,
    dtype: ArgSortDtype,
    queue: &CommandQueue,
    src: &Buffer,
    indices: &Buffer,
    out: &Buffer,
    batch: u32,
    axis_size: u32,
    top_k: u32,
    dtype_bytes: usize,
) -> Result<(), MetalStreamError> {
    if batch == 0 || axis_size == 0 || top_k == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "take_along_axis: batch={batch} axis_size={axis_size} top_k={top_k}; all must be > 0"
        )));
    }
    let src_bytes = (batch as usize) * (axis_size as usize) * dtype_bytes;
    let idx_bytes = (batch as usize) * (top_k as usize) * 4;
    let out_bytes = (batch as usize) * (top_k as usize) * dtype_bytes;
    if src.length() < src_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "take_along_axis: src buffer too small: {} < {src_bytes}",
            src.length()
        )));
    }
    if indices.length() < idx_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "take_along_axis: indices buffer too small: {} < {idx_bytes}",
            indices.length()
        )));
    }
    if out.length() < out_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "take_along_axis: out buffer too small: {} < {out_bytes}",
            out.length()
        )));
    }

    let pipeline = kernels.pipeline_for(dtype);

    let cmdbuf = queue.commandBuffer().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("queue.commandBuffer returned nil".into())
    })?;
    let enc = cmdbuf.computeCommandEncoder().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("computeCommandEncoder returned nil".into())
    })?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(src), 0, 0);
        enc.setBuffer_offset_atIndex(Some(indices), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out), 0, 2);
        let axis_i32: i32 = axis_size as i32;
        let src_ax_stride: i32 = 1; // contiguous, axis=-1
        let idx_ax_stride: i32 = 1;
        enc.setBytes_length_atIndex(
            NonNull::new(&axis_i32 as *const i32 as *mut c_void).unwrap(),
            std::mem::size_of::<i32>(),
            8,
        );
        enc.setBytes_length_atIndex(
            NonNull::new(&src_ax_stride as *const i32 as *mut c_void).unwrap(),
            std::mem::size_of::<i32>(),
            9,
        );
        enc.setBytes_length_atIndex(
            NonNull::new(&idx_ax_stride as *const i32 as *mut c_void).unwrap(),
            std::mem::size_of::<i32>(),
            10,
        );
    }
    // Grid (size_post=1, idx_ax_size=top_k, size_pre=batch) per
    // mlx indexing.cpp:501.
    let grid = MTLSize {
        width: 1,
        height: top_k as usize,
        depth: batch as usize,
    };
    // Mlx uses `get_block_dims(...)` which picks a threadgroup shape
    // ≤ kernel.maxTotalThreadsPerThreadgroup. Our grid is tiny
    // (1 × top_k × batch ≤ 1 × 10 × 32 = 320) so a single threadgroup
    // covers everything.
    let threads_per_tg = MTLSize {
        width: 1,
        height: top_k as usize,
        depth: batch as usize,
    };
    enc.dispatchThreads_threadsPerThreadgroup(grid, threads_per_tg);
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();
    if cmdbuf.status() != MTLCommandBufferStatus::Completed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "take_along_axis dispatch finished with status {:?}",
            cmdbuf.status()
        )));
    }
    Ok(())
}

// ─── CPU reference for tests ─────────────────────────────────────────

pub fn take_along_axis_cpu_f32(
    src: &[f32],
    indices: &[u32],
    batch: usize,
    axis_size: usize,
    top_k: usize,
) -> Vec<f32> {
    assert_eq!(src.len(), batch * axis_size);
    assert_eq!(indices.len(), batch * top_k);
    let mut out = vec![0.0f32; batch * top_k];
    for b in 0..batch {
        for k in 0..top_k {
            let i = indices[b * top_k + k] as usize;
            assert!(
                i < axis_size,
                "index {i} out of range [0, {axis_size})"
            );
            out[b * top_k + k] = src[b * axis_size + i];
        }
    }
    out
}
