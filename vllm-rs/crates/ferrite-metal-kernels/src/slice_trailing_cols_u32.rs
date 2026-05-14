// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Slice trailing columns of a 2D uint32 buffer:
//!
//!   `dst[n, k] = src[n, src_cols - dst_cols + k]`
//!
//! Used by the MoE `I::SharedFusedMoe` lowering arm to take the
//! trailing `top_k` columns of `argpartition`'s sorted `[N, num_experts]`
//! output into a contiguous `[N, top_k]` indices buffer.
//!
//! MLX represents the `[..., -k:]` slice as a strided view at graph
//! level (`qwen3_moe.py:131`); ferrite-metal materializes it because our
//! MoE scratch regions are flat byte ranges, not array views.

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

pub struct SliceTrailingColsU32Kernels {
    pub u32: ComputePipelineState,
    _library: Library,
}

impl SliceTrailingColsU32Kernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(
            device,
            crate::embedded_metallib!("slice_trailing_cols_u32"),
        )
        .map_err(|e| {
            MetalStreamError::ShaderCompilationFailed(format!(
                "load `slice_trailing_cols_u32.metallib`: {e}"
            ))
        })?;
        let ns_name = NSString::from_str("slice_trailing_cols_uint32");
        let function = library.newFunctionWithName(&ns_name).ok_or_else(|| {
            MetalStreamError::ShaderCompilationFailed(
                "slice_trailing_cols_uint32 fn missing".into(),
            )
        })?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "slice_trailing_cols_uint32 pipeline: {e:?}"
                ))
            })?;
        Ok(Self {
            u32: pipeline,
            _library: library,
        })
    }
}

/// Dispatch `dst[n, k] = src[n, src_cols - dst_cols + k]`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_slice_trailing_cols_u32(
    kernels: &SliceTrailingColsU32Kernels,
    queue: &CommandQueue,
    src: &Buffer,
    dst: &Buffer,
    rows: u32,
    src_cols: u32,
    dst_cols: u32,
) -> Result<(), MetalStreamError> {
    if rows == 0 || src_cols == 0 || dst_cols == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "slice_trailing_cols_u32: rows={rows} src_cols={src_cols} dst_cols={dst_cols}; \
             all must be > 0"
        )));
    }
    if dst_cols > src_cols {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "slice_trailing_cols_u32: dst_cols={dst_cols} > src_cols={src_cols}"
        )));
    }
    let src_bytes = (rows as usize) * (src_cols as usize) * 4;
    let dst_bytes = (rows as usize) * (dst_cols as usize) * 4;
    if src.length() < src_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "slice_trailing_cols_u32: src too small: {} < {src_bytes}",
            src.length()
        )));
    }
    if dst.length() < dst_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "slice_trailing_cols_u32: dst too small: {} < {dst_bytes}",
            dst.length()
        )));
    }

    let pipeline = &kernels.u32;
    let cmdbuf = queue.commandBuffer().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("queue.commandBuffer returned nil".into())
    })?;
    let enc = cmdbuf.computeCommandEncoder().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("computeCommandEncoder returned nil".into())
    })?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(src), 0, 0);
        enc.setBuffer_offset_atIndex(Some(dst), 0, 1);
        let sc: i32 = src_cols as i32;
        let dc: i32 = dst_cols as i32;
        enc.setBytes_length_atIndex(
            NonNull::new(&sc as *const i32 as *mut c_void).unwrap(),
            std::mem::size_of::<i32>(),
            2,
        );
        enc.setBytes_length_atIndex(
            NonNull::new(&dc as *const i32 as *mut c_void).unwrap(),
            std::mem::size_of::<i32>(),
            3,
        );
    }
    let grid = MTLSize {
        width: dst_cols as usize,
        height: rows as usize,
        depth: 1,
    };
    // Tiny grid (≤ top_k × bucket_m, both small); fit in one threadgroup.
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
    let tg_w = (dst_cols as usize).min(max_threads).max(1);
    let threads_per_tg = MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreads_threadsPerThreadgroup(grid, threads_per_tg);
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();
    if cmdbuf.status() != MTLCommandBufferStatus::Completed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "slice_trailing_cols_u32 dispatch finished with status {:?}",
            cmdbuf.status()
        )));
    }
    Ok(())
}

// ─── CPU reference ─────────────────────────────────────────────────────

pub fn slice_trailing_cols_u32_cpu(
    src: &[u32],
    rows: usize,
    src_cols: usize,
    dst_cols: usize,
) -> Vec<u32> {
    assert_eq!(src.len(), rows * src_cols);
    assert!(dst_cols <= src_cols);
    let mut out = vec![0u32; rows * dst_cols];
    for n in 0..rows {
        for k in 0..dst_cols {
            let sc = src_cols - dst_cols + k;
            out[n * dst_cols + k] = src[n * src_cols + sc];
        }
    }
    out
}
