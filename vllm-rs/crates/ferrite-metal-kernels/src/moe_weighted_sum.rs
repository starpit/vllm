// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! MoE final-reduction kernel dispatcher.
//!
//! Fuses the broadcast-multiply and the sum-reduce that MLX expresses as
//! two separate ops at qwen3_moe.py:137:
//!
//! ```python
//! y = (y * scores[..., None]).sum(axis=-2)
//! ```
//!
//! `y` is `[N, top_k, hidden]`, `scores` is `[N, top_k]`. Output is
//! `[N, hidden]`. One thread per output element, accumulating in float
//! across the top_k axis. Used by the `I::SharedFusedMoe` Metal
//! lowering as the final step of the routed-experts decomposition.

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
pub enum MoeWeightedSumDtype {
    F16,
    BF16,
}

/// Standalone kernel handle. Independent of the worker's
/// `SpecializedPipelineCache` — used by direct callers (tests,
/// benches). The pool's worker resolves the pipeline through
/// `pipelines.pipeline_for_command(cmd)` instead.
pub struct MoeWeightedSumKernels {
    library: Library,
}

impl MoeWeightedSumKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library =
            load_library_from_bytes(device, crate::embedded_metallib!("moe_weighted_sum")).map_err(
                |e| {
                    MetalStreamError::ShaderCompilationFailed(format!(
                        "load `moe_weighted_sum.metallib`: {e}"
                    ))
                },
            )?;
        Ok(Self { library })
    }

    /// Build a `(top_k, hidden)`-specialized pipeline. Both dims ride as
    /// function constants; lowering bakes them per-bucket so the
    /// compiled code unrolls the top_k loop and folds the hidden
    /// stride into addressing math.
    pub fn pipeline_for(
        &self,
        device: &Device,
        dtype: MoeWeightedSumDtype,
        top_k: u32,
        hidden: u32,
    ) -> Result<ComputePipelineState, MetalStreamError> {
        let symbol = match dtype {
            MoeWeightedSumDtype::F16 => "moe_weighted_sum_f16",
            MoeWeightedSumDtype::BF16 => "moe_weighted_sum_bf16",
        };
        let constants = MTLFunctionConstantValues::new();
        unsafe {
            constants.setConstantValue_type_atIndex(
                NonNull::new(&top_k as *const u32 as *mut c_void).unwrap(),
                ::objc2_metal::MTLDataType::UInt,
                0,
            );
            constants.setConstantValue_type_atIndex(
                NonNull::new(&hidden as *const u32 as *mut c_void).unwrap(),
                ::objc2_metal::MTLDataType::UInt,
                1,
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

/// One-shot dispatch. Synchronous; intended for tests and benches.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_moe_weighted_sum(
    pipeline: &ComputePipelineState,
    queue: &CommandQueue,
    out: &Buffer,
    expert_out: &Buffer,
    scores: &Buffer,
    n: u32,
    top_k: u32,
    hidden: u32,
    dtype_bytes: usize,
) -> Result<(), MetalStreamError> {
    if n == 0 || top_k == 0 || hidden == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "moe_weighted_sum: n={n} top_k={top_k} hidden={hidden}; all must be > 0"
        )));
    }
    let expert_bytes =
        (n as usize) * (top_k as usize) * (hidden as usize) * dtype_bytes;
    let scores_bytes = (n as usize) * (top_k as usize) * dtype_bytes;
    let out_bytes = (n as usize) * (hidden as usize) * dtype_bytes;
    if out.length() < out_bytes
        || expert_out.length() < expert_bytes
        || scores.length() < scores_bytes
    {
        return Err(MetalStreamError::ShaderCompilationFailed(
            "moe_weighted_sum: buffer too small".into(),
        ));
    }

    let cmdbuf = queue.commandBuffer().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("queue.commandBuffer returned nil".into())
    })?;
    let enc = cmdbuf.computeCommandEncoder().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("computeCommandEncoder returned nil".into())
    })?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(out), 0, 0);
        enc.setBuffer_offset_atIndex(Some(expert_out), 0, 1);
        enc.setBuffer_offset_atIndex(Some(scores), 0, 2);
    }

    let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
    let tg_d = (hidden as usize).min(max_threads).max(1);
    let grid = MTLSize {
        width: hidden as usize,
        height: n as usize,
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
            "moe_weighted_sum dispatch finished with status {:?}",
            cmdbuf.status()
        )));
    }
    Ok(())
}

// ─── CPU reference for tests ─────────────────────────────────────────

/// Float-accumulator reference matching the kernel exactly.
pub fn moe_weighted_sum_cpu_f32(
    expert_out: &[f32],
    scores: &[f32],
    n: usize,
    top_k: usize,
    hidden: usize,
) -> Vec<f32> {
    assert_eq!(expert_out.len(), n * top_k * hidden);
    assert_eq!(scores.len(), n * top_k);
    let mut out = vec![0.0f32; n * hidden];
    for ni in 0..n {
        for di in 0..hidden {
            let mut acc = 0.0f32;
            for k in 0..top_k {
                let s = scores[ni * top_k + k];
                let e = expert_out[(ni * top_k + k) * hidden + di];
                acc = s.mul_add(e, acc);
            }
            out[ni * hidden + di] = acc;
        }
    }
    out
}
