// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Row-gather kernel dispatcher.
//!
//! Implements `out[m, d] = src[idx[m] / idx_divisor, d]` for 2D
//! contiguous `src` and `out`, 1D contiguous `idx` (uint32). Used by
//! the MoE `_gather_sort` / `_scatter_unsort` helpers
//! (`mlx-lm/mlx_lm/models/switch_layers.py:12-23`):
//!
//! * `idx_divisor == 1` — plain row gather. Used for
//!   `indices_flat[order]` and `_scatter_unsort: x[inv_order]`.
//! * `idx_divisor == top_k` — fuses the `order // K` arithmetic with
//!   the gather (`x.flatten(0,-3)[order // M]`,
//!   `switch_layers.py:17`).

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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RowGatherDtype {
    F32,
    F16,
    BF16,
    U32,
}

pub struct RowGatherKernels {
    pub f32: ComputePipelineState,
    pub f16: ComputePipelineState,
    pub bf16: ComputePipelineState,
    pub u32: ComputePipelineState,
    _library: Library,
}

impl RowGatherKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library =
            load_library_from_bytes(device, crate::embedded_metallib!("row_gather")).map_err(
                |e| {
                    MetalStreamError::ShaderCompilationFailed(format!(
                        "load `row_gather.metallib`: {e}"
                    ))
                },
            )?;
        let bp = |n: &str| build_pipeline(device, &library, n);
        Ok(Self {
            f32: bp("row_gather_float32")?,
            f16: bp("row_gather_float16")?,
            bf16: bp("row_gather_bfloat16")?,
            u32: bp("row_gather_uint32")?,
            _library: library,
        })
    }

    pub fn pipeline_for(&self, dtype: RowGatherDtype) -> &ComputePipelineState {
        match dtype {
            RowGatherDtype::F32 => &self.f32,
            RowGatherDtype::F16 => &self.f16,
            RowGatherDtype::BF16 => &self.bf16,
            RowGatherDtype::U32 => &self.u32,
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

/// Dispatch `out[m, d] = src[idx[m] / idx_divisor, d]`.
///
/// `dtype_bytes` must match `dtype` (4 for F32/U32, 2 for F16/BF16).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_row_gather(
    kernels: &RowGatherKernels,
    dtype: RowGatherDtype,
    queue: &CommandQueue,
    src: &Buffer,
    idx: &Buffer,
    out: &Buffer,
    src_rows: u32,
    m: u32,
    d: u32,
    idx_divisor: u32,
    dtype_bytes: usize,
) -> Result<(), MetalStreamError> {
    if m == 0 || d == 0 || idx_divisor == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "row_gather: m={m} d={d} idx_divisor={idx_divisor}; all must be > 0"
        )));
    }
    let src_bytes = (src_rows as usize) * (d as usize) * dtype_bytes;
    let idx_bytes = (m as usize) * 4;
    let out_bytes = (m as usize) * (d as usize) * dtype_bytes;
    if src.length() < src_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "row_gather: src buffer too small: {} < {src_bytes}",
            src.length()
        )));
    }
    if idx.length() < idx_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "row_gather: idx buffer too small: {} < {idx_bytes}",
            idx.length()
        )));
    }
    if out.length() < out_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "row_gather: out buffer too small: {} < {out_bytes}",
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
        enc.setBuffer_offset_atIndex(Some(idx), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out), 0, 2);
        let d_i32: i32 = d as i32;
        let div_i32: i32 = idx_divisor as i32;
        enc.setBytes_length_atIndex(
            NonNull::new(&d_i32 as *const i32 as *mut c_void).unwrap(),
            std::mem::size_of::<i32>(),
            3,
        );
        enc.setBytes_length_atIndex(
            NonNull::new(&div_i32 as *const i32 as *mut c_void).unwrap(),
            std::mem::size_of::<i32>(),
            4,
        );
    }

    // Choose threadgroup width along D up to maxTotalThreadsPerThreadgroup
    // (1024 on Apple Silicon) and the inner-D dimension.
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
    let tg_d = (d as usize).min(max_threads).max(1);
    let grid = MTLSize {
        width: d as usize,
        height: m as usize,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: tg_d,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreads_threadsPerThreadgroup(grid, threads_per_tg);
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();
    if cmdbuf.status() != MTLCommandBufferStatus::Completed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "row_gather dispatch finished with status {:?}",
            cmdbuf.status()
        )));
    }
    Ok(())
}

// ─── CPU reference for tests ─────────────────────────────────────────

pub fn row_gather_cpu_f32(
    src: &[f32],
    idx: &[u32],
    src_rows: usize,
    d: usize,
    idx_divisor: u32,
) -> Vec<f32> {
    assert_eq!(src.len(), src_rows * d);
    let m = idx.len();
    let mut out = vec![0.0f32; m * d];
    for mi in 0..m {
        let src_row = (idx[mi] / idx_divisor) as usize;
        assert!(
            src_row < src_rows,
            "src_row {src_row} out of range [0, {src_rows})"
        );
        for di in 0..d {
            out[mi * d + di] = src[src_row * d + di];
        }
    }
    out
}

pub fn row_gather_cpu_u32(
    src: &[u32],
    idx: &[u32],
    src_rows: usize,
    d: usize,
    idx_divisor: u32,
) -> Vec<u32> {
    assert_eq!(src.len(), src_rows * d);
    let m = idx.len();
    let mut out = vec![0u32; m * d];
    for mi in 0..m {
        let src_row = (idx[mi] / idx_divisor) as usize;
        for di in 0..d {
            out[mi * d + di] = src[src_row * d + di];
        }
    }
    out
}
