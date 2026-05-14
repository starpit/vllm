// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! ArgPartition (top-k) kernel dispatcher.
//!
//! Faithful port of mlx `gpu_merge_sort` single-block path
//! (`mlx/backend/metal/sort.cpp:15 single_block_sort`) in ARG_SORT
//! mode, contiguous variant only. Used by the MoE router to pick the
//! top-`k` expert indices per token:
//!
//! ```python
//! inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
//! # qwen3_next.py:338 / qwen3_moe.py:131
//! ```
//!
//! mlx implements `argpartition` by dispatching the full argsort and
//! letting the user slice the trailing `k` indices — see
//! `mlx/backend/metal/sort.cpp:342 ArgPartition::eval_gpu`. We do the
//! same: the shader writes the full `[batch, axis_size]` argsort, and
//! callers consume the trailing `k` columns via offset+stride.
//!
//! Layout: input/output rows are contiguous; one threadgroup per row;
//! `axis_size ≤ BLOCK_THREADS * 4`. MoE router `axis_size`
//! (= num_experts) is at most 512 for any current model, fitting in
//! the bn=128, tn=4 instantiation.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

use crate::shader_cache::load_library_from_bytes;
use crate::stream::MetalStreamError;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

/// `tn=4` matches mlx's instantiation across all dtypes for argsort.
pub const ARG_BLOCK_SORT_N_PER_THREAD: usize = 4;

/// Pipelines for one `(dtype, bn)` triple. mlx picks `bn` from
/// `axis_size`:
///
/// | axis_size  | potential_bn | bn |
/// |------------|--------------|----|
/// | ≤ 128      | ≤ 32         | 32 |
/// | ≤ 256      | ≤ 64         | 64 |
/// | ≤ 512      | ≤ 128        | 128|
///
/// (See `mlx/backend/metal/sort.cpp:286-299`.) We match this exactly.
pub struct ArgPartitionKernels {
    pub f32_bn32: ComputePipelineState,
    pub f32_bn64: ComputePipelineState,
    pub f32_bn128: ComputePipelineState,
    pub f16_bn32: ComputePipelineState,
    pub f16_bn64: ComputePipelineState,
    pub f16_bn128: ComputePipelineState,
    pub bf16_bn32: ComputePipelineState,
    pub bf16_bn64: ComputePipelineState,
    pub bf16_bn128: ComputePipelineState,
    pub u32_bn32: ComputePipelineState,
    pub u32_bn64: ComputePipelineState,
    pub u32_bn128: ComputePipelineState,
    _library: Library,
}

impl ArgPartitionKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library =
            load_library_from_bytes(device, crate::embedded_metallib!("argpartition")).map_err(
                |e| {
                    MetalStreamError::ShaderCompilationFailed(format!(
                        "load `argpartition.metallib`: {e}"
                    ))
                },
            )?;
        let bp = |n: &str| build_pipeline(device, &library, n);
        Ok(Self {
            f32_bn32: bp("c_arg_block_sort_float32_uint32_bn32_tn4")?,
            f32_bn64: bp("c_arg_block_sort_float32_uint32_bn64_tn4")?,
            f32_bn128: bp("c_arg_block_sort_float32_uint32_bn128_tn4")?,
            f16_bn32: bp("c_arg_block_sort_float16_uint32_bn32_tn4")?,
            f16_bn64: bp("c_arg_block_sort_float16_uint32_bn64_tn4")?,
            f16_bn128: bp("c_arg_block_sort_float16_uint32_bn128_tn4")?,
            bf16_bn32: bp("c_arg_block_sort_bfloat16_uint32_bn32_tn4")?,
            bf16_bn64: bp("c_arg_block_sort_bfloat16_uint32_bn64_tn4")?,
            bf16_bn128: bp("c_arg_block_sort_bfloat16_uint32_bn128_tn4")?,
            u32_bn32: bp("c_arg_block_sort_uint32_uint32_bn32_tn4")?,
            u32_bn64: bp("c_arg_block_sort_uint32_uint32_bn64_tn4")?,
            u32_bn128: bp("c_arg_block_sort_uint32_uint32_bn128_tn4")?,
            _library: library,
        })
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

/// Dtype tag for picking the right pipeline.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ArgSortDtype {
    F32,
    F16,
    BF16,
    U32,
}

/// Mirror mlx's `bn` picker (`sort.cpp:286-299`) but bounded to bn ≤
/// 128 — beyond that, callers must fall back to multi-block sort
/// (not yet ported; MoE router doesn't need it).
pub fn arg_sort_bn(axis_size: usize) -> usize {
    let potential_bn = axis_size.div_ceil(ARG_BLOCK_SORT_N_PER_THREAD);
    if potential_bn > 128 {
        // Caller violated single-block bound; pipeline construction
        // will fail at dispatch time.
        panic!(
            "arg_sort_bn: axis_size={axis_size} requires multi-block sort, not ported yet"
        );
    } else if potential_bn > 64 {
        128
    } else if potential_bn > 32 {
        64
    } else {
        32
    }
}

impl ArgPartitionKernels {
    pub fn pipeline_for(&self, dtype: ArgSortDtype, axis_size: usize) -> &ComputePipelineState {
        let bn = arg_sort_bn(axis_size);
        match (dtype, bn) {
            (ArgSortDtype::F32, 32) => &self.f32_bn32,
            (ArgSortDtype::F32, 64) => &self.f32_bn64,
            (ArgSortDtype::F32, 128) => &self.f32_bn128,
            (ArgSortDtype::F16, 32) => &self.f16_bn32,
            (ArgSortDtype::F16, 64) => &self.f16_bn64,
            (ArgSortDtype::F16, 128) => &self.f16_bn128,
            (ArgSortDtype::BF16, 32) => &self.bf16_bn32,
            (ArgSortDtype::BF16, 64) => &self.bf16_bn64,
            (ArgSortDtype::BF16, 128) => &self.bf16_bn128,
            (ArgSortDtype::U32, 32) => &self.u32_bn32,
            (ArgSortDtype::U32, 64) => &self.u32_bn64,
            (ArgSortDtype::U32, 128) => &self.u32_bn128,
            _ => unreachable!("arg_sort_bn returned unexpected bn={bn}"),
        }
    }
}

/// Dispatch one full argsort over `[batch, axis_size]`. Output is
/// `[batch, axis_size]` `uint32` indices in ascending order; the
/// caller takes the trailing `k` columns for top-k.
///
/// `dtype_bytes` must match `dtype` (4 for F32, 2 for F16/BF16).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_argpartition(
    kernels: &ArgPartitionKernels,
    dtype: ArgSortDtype,
    queue: &CommandQueue,
    input: &Buffer,
    output: &Buffer,
    batch: u32,
    axis_size: u32,
    dtype_bytes: usize,
) -> Result<(), MetalStreamError> {
    if batch == 0 || axis_size == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "dispatch_argpartition: batch={batch} axis_size={axis_size}; both must be > 0"
        )));
    }
    let elems = (batch as usize) * (axis_size as usize);
    let in_bytes_needed = elems * dtype_bytes;
    let out_bytes_needed = elems * 4; // uint32
    if input.length() < in_bytes_needed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "argpartition: input buffer too small: have {} bytes, need {in_bytes_needed}",
            input.length()
        )));
    }
    if output.length() < out_bytes_needed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "argpartition: output buffer too small: have {} bytes, need {out_bytes_needed}",
            output.length()
        )));
    }

    let pipeline = kernels.pipeline_for(dtype, axis_size as usize);
    let bn = arg_sort_bn(axis_size as usize);

    let cmdbuf = queue.commandBuffer().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("queue.commandBuffer returned nil".into())
    })?;
    let enc = cmdbuf.computeCommandEncoder().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("computeCommandEncoder returned nil".into())
    })?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 0);
        enc.setBuffer_offset_atIndex(Some(output), 0, 1);
        let axis_i32: i32 = axis_size as i32;
        enc.setBytes_length_atIndex(
            NonNull::new(&axis_i32 as *const i32 as *mut c_void).unwrap(),
            std::mem::size_of::<i32>(),
            2,
        );
    }
    let threadgroups = MTLSize {
        width: 1,
        height: batch as usize,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: bn,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();
    if cmdbuf.status() != MTLCommandBufferStatus::Completed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "argpartition dispatch finished with status {:?}",
            cmdbuf.status()
        )));
    }
    Ok(())
}

// ─── CPU reference for tests ─────────────────────────────────────────

/// Stable argsort reference, row by row, ascending order.
///
/// Matches mlx's argsort tie-break: equal values keep their original
/// relative order (stable sort). The Metal kernel is not strictly
/// stable but test cases use distinct values, so this is fine for
/// parity testing.
pub fn argsort_cpu_u32(input: &[u32], batch: usize, axis_size: usize) -> Vec<u32> {
    assert_eq!(input.len(), batch * axis_size);
    let mut out = vec![0u32; input.len()];
    for b in 0..batch {
        let row = &input[b * axis_size..(b + 1) * axis_size];
        let mut idx: Vec<u32> = (0..axis_size as u32).collect();
        idx.sort_by_key(|&a| row[a as usize]);
        out[b * axis_size..(b + 1) * axis_size].copy_from_slice(&idx);
    }
    out
}

pub fn argsort_cpu_f32(input: &[f32], batch: usize, axis_size: usize) -> Vec<u32> {
    assert_eq!(input.len(), batch * axis_size);
    let mut out = vec![0u32; input.len()];
    for b in 0..batch {
        let row = &input[b * axis_size..(b + 1) * axis_size];
        let mut idx: Vec<u32> = (0..axis_size as u32).collect();
        // Stable ascending sort by row value.
        idx.sort_by(|&a, &c| {
            row[a as usize]
                .partial_cmp(&row[c as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out[b * axis_size..(b + 1) * axis_size].copy_from_slice(&idx);
    }
    out
}
