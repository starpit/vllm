// SPDX-License-Identifier: Apache-2.0
//! MoE weighted-sum reduction dispatcher. Mirrors the trailing
//! Python expression `y = (y * scores[..., None]).sum(axis=-2)`
//! from qwen3_moe.py:137 / qwen2_moe.py:138 / mixtral.py:119.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLLibrary, MTLSize,
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
pub enum MoeSumDType {
    F32,
    F16,
    BF16,
}

impl MoeSumDType {
    fn symbol(self) -> &'static str {
        match self {
            Self::F32 => "moe_weighted_sum_float32",
            Self::F16 => "moe_weighted_sum_float16",
            Self::BF16 => "moe_weighted_sum_bfloat16",
        }
    }

    fn element_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

pub struct MoeWeightedSumKernels {
    library: Library,
    device: Device,
}

impl MoeWeightedSumKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(device, crate::embedded_metallib!("moe_weighted_sum"))
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "load `moe_weighted_sum.metallib`: {e}"
                ))
            })?;
        Ok(Self {
            library,
            device: device.clone(),
        })
    }

    pub fn build_pipeline(
        &self,
        dtype: MoeSumDType,
        top_k: u32,
        hidden: u32,
    ) -> Result<ComputePipelineState, MetalStreamError> {
        let constants = MTLFunctionConstantValues::new();
        let top_k_i = top_k as i32;
        let hidden_i = hidden as i32;
        unsafe {
            constants.setConstantValue_type_atIndex(
                NonNull::new(&top_k_i as *const i32 as *mut c_void).unwrap(),
                MTLDataType::Int,
                0,
            );
            constants.setConstantValue_type_atIndex(
                NonNull::new(&hidden_i as *const i32 as *mut c_void).unwrap(),
                MTLDataType::Int,
                1,
            );
        }
        let name = NSString::from_str(dtype.symbol());
        let func = self
            .library
            .newFunctionWithName_constantValues_error(&name, &constants)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "{}: {e:?}",
                    dtype.symbol()
                ))
            })?;
        self.device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "{} pipeline: {e:?}",
                    dtype.symbol()
                ))
            })
    }
}

pub fn dispatch_moe_weighted_sum(
    kernels: &MoeWeightedSumKernels,
    queue: &CommandQueue,
    expert_out: &Buffer,
    scores: &Buffer,
    out: &Buffer,
    rows: u32,
    top_k: u32,
    hidden: u32,
    dtype: MoeSumDType,
) -> Result<(), MetalStreamError> {
    if rows == 0 || top_k == 0 || hidden == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "dispatch_moe_weighted_sum: rows={rows} top_k={top_k} hidden={hidden}; all > 0"
        )));
    }
    let expert_bytes = (rows as usize) * (top_k as usize) * (hidden as usize) * dtype.element_size();
    let scores_bytes = (rows as usize) * (top_k as usize) * dtype.element_size();
    let out_bytes = (rows as usize) * (hidden as usize) * dtype.element_size();
    if expert_out.length() < expert_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "expert_out too small: have {} need {expert_bytes}",
            expert_out.length()
        )));
    }
    if scores.length() < scores_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "scores too small: have {} need {scores_bytes}",
            scores.length()
        )));
    }
    if out.length() < out_bytes {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "out too small: have {} need {out_bytes}",
            out.length()
        )));
    }

    let pipeline = kernels.build_pipeline(dtype, top_k, hidden)?;
    let cmdbuf = queue.commandBuffer().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("commandBuffer nil".into())
    })?;
    let enc = cmdbuf.computeCommandEncoder().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("computeCommandEncoder nil".into())
    })?;
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(expert_out), 0, 0);
        enc.setBuffer_offset_atIndex(Some(scores), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out), 0, 2);
    }
    let grid = MTLSize {
        width: hidden as usize,
        height: rows as usize,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: hidden.min(64) as usize,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreads_threadsPerThreadgroup(grid, threads_per_tg);
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();
    if cmdbuf.status() != MTLCommandBufferStatus::Completed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "moe_weighted_sum status {:?}",
            cmdbuf.status()
        )));
    }
    Ok(())
}

/// CPU reference: `out[n, d] = Σ_k expert[n, k, d] * scores[n, k]`.
pub fn moe_weighted_sum_cpu_f32(
    expert: &[f32],
    scores: &[f32],
    out: &mut [f32],
    rows: usize,
    top_k: usize,
    hidden: usize,
) {
    assert_eq!(expert.len(), rows * top_k * hidden);
    assert_eq!(scores.len(), rows * top_k);
    assert_eq!(out.len(), rows * hidden);
    for n in 0..rows {
        for d in 0..hidden {
            let mut acc = 0.0_f32;
            for k in 0..top_k {
                acc += expert[n * top_k * hidden + k * hidden + d] * scores[n * top_k + k];
            }
            out[n * hidden + d] = acc;
        }
    }
}
