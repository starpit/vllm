// SPDX-License-Identifier: Apache-2.0
//! `out[n, k] = in[n, axis_size - top_k + k]` for u32 buffers.
//! Used in the MoE router lowering: argpartition produces the full
//! sorted-ascending [N, num_experts] index tensor, and the top-k
//! indices live in the trailing `top_k` columns. MTLBuffer offsets
//! are per-binding (not per-row) so a row-wise trailing slice
//! needs an explicit kernel.

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

pub struct SliceTrailingColsKernels {
    pub u32_pipeline: ComputePipelineState,
    _library: Library,
}

impl SliceTrailingColsKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(
            device,
            crate::embedded_metallib!("slice_trailing_cols"),
        )
        .map_err(|e| {
            MetalStreamError::ShaderCompilationFailed(format!(
                "load `slice_trailing_cols.metallib`: {e}"
            ))
        })?;
        let ns = NSString::from_str("slice_trailing_cols_u32");
        let func = library
            .newFunctionWithName(&ns)
            .ok_or_else(|| {
                MetalStreamError::ShaderCompilationFailed("slice_trailing_cols_u32 fn missing".into())
            })?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "slice_trailing_cols_u32 pipeline: {e:?}"
                ))
            })?;
        Ok(Self {
            u32_pipeline: pipeline,
            _library: library,
        })
    }
}

pub fn dispatch_slice_trailing_cols_u32(
    kernels: &SliceTrailingColsKernels,
    queue: &CommandQueue,
    src: &Buffer,
    dst: &Buffer,
    rows: u32,
    axis_size: u32,
    top_k: u32,
) -> Result<(), MetalStreamError> {
    if rows == 0 || top_k == 0 || axis_size < top_k {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "dispatch_slice_trailing_cols_u32: rows={rows} axis={axis_size} top_k={top_k}; rows > 0, top_k > 0, axis >= top_k"
        )));
    }
    let src_bytes = (rows as usize) * (axis_size as usize) * 4;
    let dst_bytes = (rows as usize) * (top_k as usize) * 4;
    if src.length() < src_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "src too small: have {} need {src_bytes}",
            src.length()
        )));
    }
    if dst.length() < dst_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "dst too small: have {} need {dst_bytes}",
            dst.length()
        )));
    }

    let axis = axis_size as i32;
    let k = top_k as i32;
    let cmdbuf = queue.commandBuffer().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("commandBuffer nil".into())
    })?;
    let enc = cmdbuf.computeCommandEncoder().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("computeCommandEncoder nil".into())
    })?;
    enc.setComputePipelineState(&kernels.u32_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(src), 0, 0);
        enc.setBuffer_offset_atIndex(Some(dst), 0, 1);
        enc.setBytes_length_atIndex(
            NonNull::new(&axis as *const i32 as *mut c_void).unwrap(),
            std::mem::size_of::<i32>(),
            2,
        );
        enc.setBytes_length_atIndex(
            NonNull::new(&k as *const i32 as *mut c_void).unwrap(),
            std::mem::size_of::<i32>(),
            3,
        );
    }
    let grid = MTLSize {
        width: top_k as usize,
        height: rows as usize,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: top_k.min(32) as usize,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreads_threadsPerThreadgroup(grid, threads_per_tg);
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();
    if cmdbuf.status() != MTLCommandBufferStatus::Completed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "slice_trailing_cols_u32 status {:?}",
            cmdbuf.status()
        )));
    }
    Ok(())
}
