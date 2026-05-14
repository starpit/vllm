// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! In-place L1 row-renormalize for MoE router top-k scores.
//!
//! Implements `norm_topk_prob` from qwen3_moe.py:134.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLFunctionConstantValues,
    MTLLibrary, MTLSize,
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
pub enum TopKRenormalizeDtype {
    F16,
    BF16,
}

pub struct TopKRenormalizeKernels {
    library: Library,
}

impl TopKRenormalizeKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(
            device,
            crate::embedded_metallib!("top_k_renormalize"),
        )
        .map_err(|e| {
            MetalStreamError::ShaderCompilationFailed(format!(
                "load `top_k_renormalize.metallib`: {e}"
            ))
        })?;
        Ok(Self { library })
    }

    pub fn pipeline_for(
        &self,
        device: &Device,
        dtype: TopKRenormalizeDtype,
        top_k: u32,
    ) -> Result<ComputePipelineState, MetalStreamError> {
        let symbol = match dtype {
            TopKRenormalizeDtype::F16 => "top_k_renormalize_f16",
            TopKRenormalizeDtype::BF16 => "top_k_renormalize_bf16",
        };
        let constants = MTLFunctionConstantValues::new();
        unsafe {
            constants.setConstantValue_type_atIndex(
                NonNull::new(&top_k as *const u32 as *mut c_void).unwrap(),
                ::objc2_metal::MTLDataType::UInt,
                0,
            );
        }
        let ns_name = NSString::from_str(symbol);
        let function = self
            .library
            .newFunctionWithName_constantValues_error(&ns_name, &constants)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("{symbol} fn missing: {e:?}"))
            })?;
        device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("{symbol} pipeline: {e:?}"))
            })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_top_k_renormalize(
    pipeline: &ComputePipelineState,
    queue: &CommandQueue,
    scores: &Buffer,
    n: u32,
    top_k: u32,
    dtype_bytes: usize,
) -> Result<(), MetalStreamError> {
    if n == 0 || top_k == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "top_k_renormalize: n={n} top_k={top_k}; both must be > 0"
        )));
    }
    let scores_bytes = (n as usize) * (top_k as usize) * dtype_bytes;
    if scores.length() < scores_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "top_k_renormalize: scores too small: {} < {scores_bytes}",
            scores.length()
        )));
    }

    let cmdbuf = queue.commandBuffer().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("queue.commandBuffer returned nil".into())
    })?;
    let enc = cmdbuf.computeCommandEncoder().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("computeCommandEncoder returned nil".into())
    })?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(scores), 0, 0);
    }
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
    let tg_w = (n as usize).min(max_threads).max(1);
    let grid = MTLSize {
        width: n as usize,
        height: 1,
        depth: 1,
    };
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
            "top_k_renormalize dispatch finished with status {:?}",
            cmdbuf.status()
        )));
    }
    Ok(())
}

// ─── CPU reference ─────────────────────────────────────────────────────

pub fn top_k_renormalize_cpu_f32(scores: &mut [f32], n: usize, top_k: usize) {
    assert_eq!(scores.len(), n * top_k);
    for ni in 0..n {
        let base = ni * top_k;
        let sum: f32 = scores[base..base + top_k].iter().sum();
        if sum <= 0.0 {
            continue;
        }
        let inv = 1.0 / sum;
        for v in &mut scores[base..base + top_k] {
            *v *= inv;
        }
    }
}
