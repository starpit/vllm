// SPDX-License-Identifier: Apache-2.0
//! Rust dispatcher for the 2D-contiguous take_along_axis kernel.
//! Used in the MoE router to pull per-token scores from the
//! softmax'd router probabilities via the top-k indices.

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TakeAlongDType {
    F32,
    F16,
    BF16,
}

impl TakeAlongDType {
    fn symbol(self) -> &'static str {
        match self {
            Self::F32 => "take_along_axis_2d_contig_float32",
            Self::F16 => "take_along_axis_2d_contig_float16",
            Self::BF16 => "take_along_axis_2d_contig_bfloat16",
        }
    }

    fn element_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

pub struct TakeAlongAxisKernels {
    pub f32: ComputePipelineState,
    pub f16: ComputePipelineState,
    pub bf16: ComputePipelineState,
    _library: Library,
}

impl TakeAlongAxisKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(device, crate::embedded_metallib!("take_along_axis"))
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "load `take_along_axis.metallib`: {e}"
                ))
            })?;
        let f32 = build(device, &library, TakeAlongDType::F32.symbol())?;
        let f16 = build(device, &library, TakeAlongDType::F16.symbol())?;
        let bf16 = build(device, &library, TakeAlongDType::BF16.symbol())?;
        Ok(Self {
            f32,
            f16,
            bf16,
            _library: library,
        })
    }

    pub fn pipeline_for(&self, dtype: TakeAlongDType) -> &ComputePipelineState {
        match dtype {
            TakeAlongDType::F32 => &self.f32,
            TakeAlongDType::F16 => &self.f16,
            TakeAlongDType::BF16 => &self.bf16,
        }
    }
}

fn build(
    device: &Device,
    library: &Library,
    name: &str,
) -> Result<ComputePipelineState, MetalStreamError> {
    let ns = NSString::from_str(name);
    let func = library
        .newFunctionWithName(&ns)
        .ok_or_else(|| MetalStreamError::ShaderCompilationFailed(format!("{name} fn missing")))?;
    device
        .newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| MetalStreamError::ShaderCompilationFailed(format!("{name} pipeline: {e:?}")))
}

pub fn dispatch_take_along_axis(
    kernels: &TakeAlongAxisKernels,
    queue: &CommandQueue,
    src: &Buffer,
    indices: &Buffer,
    out: &Buffer,
    rows: u32,
    src_axis_size: u32,
    idx_axis_size: u32,
    dtype: TakeAlongDType,
) -> Result<(), MetalStreamError> {
    if rows == 0 || src_axis_size == 0 || idx_axis_size == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "dispatch_take_along_axis: rows={rows} src_axis={src_axis_size} idx_axis={idx_axis_size}; all > 0"
        )));
    }
    let src_bytes = (rows as usize) * (src_axis_size as usize) * dtype.element_size();
    let idx_bytes = (rows as usize) * (idx_axis_size as usize) * 4;
    let out_bytes = idx_bytes / 4 * dtype.element_size();
    if src.length() < src_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "src buffer too small: have {} need {src_bytes}",
            src.length()
        )));
    }
    if indices.length() < idx_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "indices buffer too small: have {} need {idx_bytes}",
            indices.length()
        )));
    }
    if out.length() < out_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "out buffer too small: have {} need {out_bytes}",
            out.length()
        )));
    }

    let pipeline = kernels.pipeline_for(dtype);
    let src_axis_i = src_axis_size as i32;
    let idx_axis_i = idx_axis_size as i32;

    let cmdbuf = queue
        .commandBuffer()
        .ok_or_else(|| MetalStreamError::ShaderCompilationFailed("commandBuffer nil".into()))?;
    let enc = cmdbuf.computeCommandEncoder().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("computeCommandEncoder nil".into())
    })?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(src), 0, 0);
        enc.setBuffer_offset_atIndex(Some(indices), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out), 0, 2);
        enc.setBytes_length_atIndex(
            NonNull::new(&src_axis_i as *const i32 as *mut c_void).unwrap(),
            std::mem::size_of::<i32>(),
            3,
        );
        enc.setBytes_length_atIndex(
            NonNull::new(&idx_axis_i as *const i32 as *mut c_void).unwrap(),
            std::mem::size_of::<i32>(),
            4,
        );
    }
    let grid = MTLSize {
        width: idx_axis_size as usize,
        height: rows as usize,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: idx_axis_size.min(32) as usize,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreads_threadsPerThreadgroup(grid, threads_per_tg);
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();
    if cmdbuf.status() != MTLCommandBufferStatus::Completed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "take_along_axis status {:?}",
            cmdbuf.status()
        )));
    }
    Ok(())
}
