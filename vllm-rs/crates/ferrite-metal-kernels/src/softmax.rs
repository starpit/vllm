// SPDX-License-Identifier: Apache-2.0
//
// Rust dispatcher for the precise softmax kernel (port of MLX
// `softmax_single_row` from `mlx/backend/metal/kernels/softmax.h`).
//
// Used by the MoE router lowering arm: `router_logits` →
// `router_probs` over axis=-1 with shape `[N, num_experts]`.
// `precise=True` is the only mode the Qwen / Mixtral MoE blocks
// use (qwen3_moe.py:128, qwen2_moe.py:131, mixtral.py:116) — they
// route through float accumulation regardless of input dtype.

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

/// Block size used to dispatch `softmax_single_row`. MLX picks 256
/// for small axes; we hold to that until we have benchmark numbers
/// from a real MoE forward pass.
pub const SOFTMAX_BLOCK_THREADS: usize = 256;

/// `N_READS` baked into the shader template arg — must match
/// `MLX_N_READS` in shaders/softmax.metal.
pub const SOFTMAX_N_READS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxDType {
    F32,
    F16,
    BF16,
}

impl SoftmaxDType {
    fn precise_symbol(self) -> &'static str {
        match self {
            // F32 has no precise variant in MLX — the precise mode
            // exists to lift half/bfloat → float accum, which is
            // a no-op when T is already float. Callers that ask
            // for F32 precise route to the non-precise symbol.
            SoftmaxDType::F32 => "block_softmax_float32",
            SoftmaxDType::F16 => "block_softmax_precise_float16",
            SoftmaxDType::BF16 => "block_softmax_precise_bfloat16",
        }
    }

    fn element_size(self) -> usize {
        match self {
            SoftmaxDType::F32 => 4,
            SoftmaxDType::F16 | SoftmaxDType::BF16 => 2,
        }
    }
}

pub struct SoftmaxKernels {
    pub precise_f16: ComputePipelineState,
    pub precise_bf16: ComputePipelineState,
    pub f32: ComputePipelineState,
    _library: Library,
}

impl SoftmaxKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(device, crate::embedded_metallib!("softmax"))
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("load `softmax.metallib`: {e}"))
            })?;
        let precise_f16 = build_pipeline(device, &library, "block_softmax_precise_float16")?;
        let precise_bf16 = build_pipeline(device, &library, "block_softmax_precise_bfloat16")?;
        let f32 = build_pipeline(device, &library, "block_softmax_float32")?;
        Ok(Self {
            precise_f16,
            precise_bf16,
            f32,
            _library: library,
        })
    }

    pub fn pipeline_for(&self, dtype: SoftmaxDType) -> &ComputePipelineState {
        match dtype {
            SoftmaxDType::F16 => &self.precise_f16,
            SoftmaxDType::BF16 => &self.precise_bf16,
            SoftmaxDType::F32 => &self.f32,
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

/// Dispatch precise row-wise softmax over `[rows, axis_size]`. The
/// in/out buffers may alias (MLX in-place softmax pattern). Used by
/// the MoE router lowering arm for `mx.softmax(gates, axis=-1,
/// precise=True)`.
pub fn dispatch_softmax(
    kernels: &SoftmaxKernels,
    queue: &CommandQueue,
    input: &Buffer,
    output: &Buffer,
    rows: u32,
    axis_size: u32,
    dtype: SoftmaxDType,
) -> Result<(), MetalStreamError> {
    if rows == 0 || axis_size == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "dispatch_softmax: rows={rows} axis_size={axis_size}; both must be > 0"
        )));
    }
    let bytes_needed = (rows as usize) * (axis_size as usize) * dtype.element_size();
    if input.length() < bytes_needed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "input buffer too small: have {} bytes, need {bytes_needed}",
            input.length()
        )));
    }
    if output.length() < bytes_needed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "output buffer too small: have {} bytes, need {bytes_needed}",
            output.length()
        )));
    }
    if (axis_size as usize) > SOFTMAX_BLOCK_THREADS * SOFTMAX_N_READS {
        // Single-row variant requires BLOCK_THREADS * N_READS to
        // cover axis_size; the looped variant handles larger axes.
        // For MoE router widths (Mixtral=8, Qwen2-MoE=60, Qwen3-MoE=128)
        // the single-row form always fits.
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "dispatch_softmax: axis_size={axis_size} exceeds single-row capacity {}",
            SOFTMAX_BLOCK_THREADS * SOFTMAX_N_READS
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
        enc.setBuffer_offset_atIndex(Some(input), 0, 0);
        enc.setBuffer_offset_atIndex(Some(output), 0, 1);
        enc.setBytes_length_atIndex(
            NonNull::new(&axis_size as *const u32 as *mut c_void).unwrap(),
            std::mem::size_of::<u32>(),
            2,
        );
    }
    let threadgroups = MTLSize {
        width: rows as usize,
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: SOFTMAX_BLOCK_THREADS,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();
    if cmdbuf.status() != MTLCommandBufferStatus::Completed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "softmax dispatch finished with status {:?}",
            cmdbuf.status()
        )));
    }
    Ok(())
}

/// CPU reference implementation of precise softmax for parity tests.
/// Matches MLX's `mx.softmax(x, axis=-1, precise=True)` — float accum
/// then cast back to input dtype.
pub fn softmax_cpu_f32(input: &[f32], rows: usize, axis_size: usize, out: &mut [f32]) {
    assert_eq!(input.len(), rows * axis_size);
    assert_eq!(out.len(), rows * axis_size);
    for r in 0..rows {
        let row = &input[r * axis_size..(r + 1) * axis_size];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        let mut exps = vec![0.0_f32; axis_size];
        for (i, &v) in row.iter().enumerate() {
            let e = (v - m).exp();
            exps[i] = e;
            sum += e;
        }
        let inv = 1.0_f32 / sum;
        for (i, e) in exps.iter().enumerate() {
            out[r * axis_size + i] = *e * inv;
        }
    }
}

pub fn symbol_for(dtype: SoftmaxDType) -> &'static str {
    dtype.precise_symbol()
}
