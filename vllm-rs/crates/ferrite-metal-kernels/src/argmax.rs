// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `argmax_f16` — greedy-sample MSL kernel + Rust dispatcher.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ComputeCommandEncoder, MTLBuffer, MTLCommandBuffer,
    MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
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

pub const ARGMAX_DEFAULT_TG_SIZE: usize = 256;

pub struct ArgmaxKernels {
    pub f16: ComputePipelineState,
    pub bf16: ComputePipelineState,
    _library: Library,
}

impl ArgmaxKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(device, crate::embedded_metallib!("argmax"))
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("load `argmax.metallib`: {e}"))
            })?;
        let f16 = build_pipeline(device, &library, "argmax_f16")?;
        let bf16 = build_pipeline(device, &library, "argmax_bf16")?;
        Ok(Self {
            f16,
            bf16,
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

pub fn dispatch_argmax_f16(
    kernels: &ArgmaxKernels,
    queue: &CommandQueue,
    logits: &Buffer,
    output: &Buffer,
    batch: u32,
    vocab: u32,
) -> Result<(), MetalStreamError> {
    dispatch_argmax_f16_with_tg_size(
        kernels,
        queue,
        logits,
        output,
        batch,
        vocab,
        ARGMAX_DEFAULT_TG_SIZE,
    )
}

pub fn dispatch_argmax_f16_with_tg_size(
    kernels: &ArgmaxKernels,
    queue: &CommandQueue,
    logits: &Buffer,
    output: &Buffer,
    batch: u32,
    vocab: u32,
    tg_size: usize,
) -> Result<(), MetalStreamError> {
    if batch == 0 || vocab == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "dispatch_argmax_f16: batch={batch} vocab={vocab}; both must be > 0"
        )));
    }
    if !tg_size.is_power_of_two() || tg_size == 0 || tg_size > 1024 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "dispatch_argmax_f16: tg_size={tg_size} must be a power of two in [1, 1024]"
        )));
    }
    let logits_bytes_needed = (batch as usize) * (vocab as usize) * 2;
    if logits.length() < logits_bytes_needed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "logits buffer too small: have {} bytes, need {logits_bytes_needed}",
            logits.length()
        )));
    }
    let output_bytes_needed = (batch as usize) * 4;
    if output.length() < output_bytes_needed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "output buffer too small: have {} bytes, need {output_bytes_needed}",
            output.length()
        )));
    }

    let cmdbuf = queue.commandBuffer().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("queue.commandBuffer returned nil".into())
    })?;
    let enc = cmdbuf.computeCommandEncoder().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("computeCommandEncoder returned nil".into())
    })?;
    enc.setComputePipelineState(&kernels.f16);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(logits), 0, 0);
    }
    unsafe {
        enc.setBuffer_offset_atIndex(Some(output), 0, 1);
    }
    unsafe {
        enc.setBytes_length_atIndex(
            NonNull::new(&batch as *const u32 as *mut c_void).unwrap(),
            std::mem::size_of::<u32>(),
            2,
        );
        enc.setBytes_length_atIndex(
            NonNull::new(&vocab as *const u32 as *mut c_void).unwrap(),
            std::mem::size_of::<u32>(),
            3,
        );
    }
    let threadgroups = MTLSize {
        width: batch as usize,
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: tg_size,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();
    if cmdbuf.status() != MTLCommandBufferStatus::Completed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "argmax_f16 dispatch finished with status {:?}",
            cmdbuf.status()
        )));
    }
    Ok(())
}

pub fn dispatch_argmax_bf16(
    kernels: &ArgmaxKernels,
    queue: &CommandQueue,
    logits: &Buffer,
    output: &Buffer,
    batch: u32,
    vocab: u32,
) -> Result<(), MetalStreamError> {
    if batch == 0 || vocab == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "dispatch_argmax_bf16: batch={batch} vocab={vocab}; both must be > 0"
        )));
    }
    let tg_size: usize = ARGMAX_DEFAULT_TG_SIZE;
    let logits_bytes_needed = (batch as usize) * (vocab as usize) * 2;
    if logits.length() < logits_bytes_needed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "logits buffer too small: have {} bytes, need {logits_bytes_needed}",
            logits.length()
        )));
    }
    let output_bytes_needed = (batch as usize) * 4;
    if output.length() < output_bytes_needed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "output buffer too small: have {} bytes, need {output_bytes_needed}",
            output.length()
        )));
    }
    let cmdbuf = queue.commandBuffer().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("queue.commandBuffer returned nil".into())
    })?;
    let enc = cmdbuf.computeCommandEncoder().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("computeCommandEncoder returned nil".into())
    })?;
    enc.setComputePipelineState(&kernels.bf16);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(logits), 0, 0);
    }
    unsafe {
        enc.setBuffer_offset_atIndex(Some(output), 0, 1);
    }
    unsafe {
        enc.setBytes_length_atIndex(
            NonNull::new(&batch as *const u32 as *mut c_void).unwrap(),
            std::mem::size_of::<u32>(),
            2,
        );
        enc.setBytes_length_atIndex(
            NonNull::new(&vocab as *const u32 as *mut c_void).unwrap(),
            std::mem::size_of::<u32>(),
            3,
        );
    }
    let threadgroups = MTLSize {
        width: batch as usize,
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: tg_size,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();
    if cmdbuf.status() != MTLCommandBufferStatus::Completed {
        // Surface the underlying `NSError` (e.g.
        // `kIOGPUCommandBufferCallbackErrorOutOfMemory`) — the bare
        // status code makes status=5 indistinguishable between a
        // memory error and a kernel fault.
        let err_desc = cmdbuf
            .error()
            .map(|e| format!("{:?}", e))
            .unwrap_or_else(|| "(no NSError)".into());
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "argmax_bf16 dispatch finished with status {:?} — error={}",
            cmdbuf.status(),
            err_desc,
        )));
    }
    Ok(())
}

/// MTL4 encoder-tail argmax dispatcher.
///
/// Encodes `setComputePipelineState` + `setArgumentTable` +
/// `dispatchThreadgroups` onto a caller-supplied
/// [`MTL4ComputeCommandEncoder`]. The caller is responsible for
/// encoder/CB/queue lifecycle (typically the same encoder being used
/// to encode the model forward — append argmax as the tail of the
/// forward CB so they share one commit and one host wait).
///
/// The argument table must be pre-built with:
/// - index 0: logits GPU address
/// - index 1: output GPU address
/// - index 2: 4-byte address holding `batch` (u32)
/// - index 3: 4-byte address holding `vocab` (u32)
pub fn encode_argmax_f16_into_mtl4(
    kernels: &ArgmaxKernels,
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    arg_table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    batch: u32,
) -> Result<(), MetalStreamError> {
    encode_argmax_into_mtl4_inner(&kernels.f16, encoder, arg_table, batch, "argmax_f16")
}

pub fn encode_argmax_bf16_into_mtl4(
    kernels: &ArgmaxKernels,
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    arg_table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    batch: u32,
) -> Result<(), MetalStreamError> {
    encode_argmax_into_mtl4_inner(&kernels.bf16, encoder, arg_table, batch, "argmax_bf16")
}

fn encode_argmax_into_mtl4_inner(
    pipeline: &ComputePipelineState,
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    arg_table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    batch: u32,
    name: &'static str,
) -> Result<(), MetalStreamError> {
    if batch == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "{name} encode: batch=0"
        )));
    }
    encoder.setComputePipelineState(pipeline);
    encoder.setArgumentTable(Some(arg_table));
    let threadgroups = MTLSize { width: batch as usize, height: 1, depth: 1 };
    let threads_per_tg = MTLSize { width: ARGMAX_DEFAULT_TG_SIZE, height: 1, depth: 1 };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
    Ok(())
}

/// Tiny helper: stage `data` into a fresh `StorageModeShared` buffer.
pub fn upload_shared_buffer<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let nbytes = std::mem::size_of_val(data);
    let buffer = device
        .newBufferWithLength_options(nbytes.max(1), MTLResourceOptions::StorageModeShared)
        .expect("upload_shared_buffer");
    if nbytes > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buffer.contents().as_ptr() as *mut u8,
                nbytes,
            );
        }
    }
    buffer
}
