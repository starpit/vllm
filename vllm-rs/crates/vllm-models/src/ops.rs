// SPDX-License-Identifier: Apache-2.0
//! Device-aware kernel dispatch for model operations.
//!
//! These functions transparently dispatch to fused CUDA kernels when tensors
//! are on GPU, falling back to candle ops on CPU. Model layers call these
//! instead of raw candle operations to automatically benefit from CUDA
//! acceleration without any structural changes.
//!
//! ## Pattern
//!
//! ```ignore
//! // Before (candle ops):
//! let activated = gate.silu()?.mul(&up)?;
//!
//! // After (auto-dispatched):
//! let activated = ops::silu_and_mul(&gate, &up)?;
//! ```

use candle_core::Tensor;
use vllm_model::layers::norm::{GemmaRmsNorm, RmsNorm};

/// Convert a kernel error to a candle error.
#[cfg(feature = "cuda")]
fn kernel_err(e: vllm_kernels::KernelError) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

// ---------------------------------------------------------------------------
// Activation dispatch
// ---------------------------------------------------------------------------

/// Fused SiLU(gate) * up — dispatches to CUDA kernel on GPU.
pub fn silu_and_mul(gate: &Tensor, up: &Tensor) -> candle_core::Result<Tensor> {
    #[cfg(feature = "cuda")]
    if gate.device().is_cuda() {
        use vllm_kernels::activation::{ActivationKernels, CudaActivationKernels};
        return CudaActivationKernels
            .silu_and_mul(gate, up)
            .map_err(kernel_err);
    }
    gate.silu()?.mul(up)
}

/// Fused GELU(gate) * up (tanh approximation) — dispatches to CUDA kernel on GPU.
pub fn gelu_and_mul(gate: &Tensor, up: &Tensor) -> candle_core::Result<Tensor> {
    #[cfg(feature = "cuda")]
    if gate.device().is_cuda() {
        use vllm_kernels::activation::{ActivationKernels, CudaActivationKernels};
        return CudaActivationKernels
            .gelu_and_mul(gate, up)
            .map_err(kernel_err);
    }
    gate.gelu()?.mul(up)
}

// ---------------------------------------------------------------------------
// Norm dispatch
// ---------------------------------------------------------------------------

/// RMS normalization — dispatches to CUDA fused kernel on GPU,
/// falls back to RmsNorm::forward() on CPU (preserves f16/bf16 upcast logic).
pub fn rms_norm(input: &Tensor, norm: &RmsNorm) -> candle_core::Result<Tensor> {
    #[cfg(feature = "cuda")]
    if input.device().is_cuda() {
        use vllm_kernels::norm::{CudaNormKernels, NormKernels};
        return CudaNormKernels
            .rms_norm(input, norm.weight(), norm.eps())
            .map_err(kernel_err);
    }
    candle_core::Module::forward(norm, input)
}

/// Gemma RMS normalization — dispatches to CUDA fused kernel on GPU.
///
/// GemmaRmsNorm stores the effective weight (weight + 1) already, so the
/// standard rms_norm kernel works directly with the effective weight.
pub fn gemma_rms_norm(input: &Tensor, norm: &GemmaRmsNorm) -> candle_core::Result<Tensor> {
    #[cfg(feature = "cuda")]
    if input.device().is_cuda() {
        use vllm_kernels::norm::{CudaNormKernels, NormKernels};
        return CudaNormKernels
            .rms_norm(input, norm.weight(), norm.eps())
            .map_err(kernel_err);
    }
    candle_core::Module::forward(norm, input)
}

// ---------------------------------------------------------------------------
// Fused QK-norm + RoPE dispatch
// ---------------------------------------------------------------------------

/// Fused per-head QK RMS normalization + NeoX RoPE rotation.
///
/// On CUDA: single fused kernel (no intermediate global memory round-trip).
/// On CPU: separate RMS norm + RoPE via candle ops.
///
/// * `query` — `[num_tokens, num_q_heads, head_dim]`
/// * `key` — `[num_tokens, num_kv_heads, head_dim]`
/// * `q_weight`, `k_weight` — `[head_dim]` (effective weight, already +1 for Gemma)
/// * `epsilon` — norm epsilon
/// * `cos_cache`, `sin_cache` — `[max_pos, head_dim]`
/// * `positions` — `[num_tokens]` (u32)
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_and_rope(
    query: &Tensor,
    key: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    epsilon: f64,
    cos_cache: &Tensor,
    sin_cache: &Tensor,
    positions: &Tensor,
) -> candle_core::Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if query.device().is_cuda() {
        use vllm_kernels::norm::CudaNormKernels;
        return CudaNormKernels
            .qk_norm_and_rope(
                query, key, q_weight, k_weight, epsilon, cos_cache, sin_cache, positions,
            )
            .map_err(kernel_err);
    }

    // CPU fallback: separate norm + RoPE.
    use vllm_kernels::norm::{CpuNormKernels, NormKernels};
    let q_normed = CpuNormKernels
        .rms_norm(query, q_weight, epsilon)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let k_normed = CpuNormKernels
        .rms_norm(key, k_weight, epsilon)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let cos = cos_cache.index_select(positions, 0)?;
    let sin = sin_cache.index_select(positions, 0)?;
    let q_rot = vllm_model::layers::apply_rotary_to_tensor(&q_normed, &cos, &sin)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let k_rot = vllm_model::layers::apply_rotary_to_tensor(&k_normed, &cos, &sin)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    Ok((q_rot, k_rot))
}
