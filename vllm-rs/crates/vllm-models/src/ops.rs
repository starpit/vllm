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

/// Fused SiLU(gate) * up from combined gate_up tensor `[num_tokens, 2*d]`.
///
/// On CUDA: single kernel reads both halves from the combined tensor,
/// eliminating 2 contiguous copy kernels + 2 allocations per layer.
/// On CPU: falls back to split + decomposed ops.
pub fn silu_and_mul_fused(gate_up: &Tensor, d: usize) -> candle_core::Result<Tensor> {
    #[cfg(feature = "cuda")]
    if gate_up.device().is_cuda() {
        return vllm_kernels::activation::silu_and_mul_fused(gate_up, d).map_err(kernel_err);
    }
    // CPU fallback: split and use candle ops
    let gate = gate_up.narrow(candle_core::D::Minus1, 0, d)?.contiguous()?;
    let up = gate_up.narrow(candle_core::D::Minus1, d, d)?.contiguous()?;
    gate.silu()?.mul(&up)
}

/// Fused GELU(gate) * up from combined gate_up tensor `[num_tokens, 2*d]`.
pub fn gelu_and_mul_fused(gate_up: &Tensor, d: usize) -> candle_core::Result<Tensor> {
    #[cfg(feature = "cuda")]
    if gate_up.device().is_cuda() {
        return vllm_kernels::activation::gelu_and_mul_fused(gate_up, d).map_err(kernel_err);
    }
    let gate = gate_up.narrow(candle_core::D::Minus1, 0, d)?.contiguous()?;
    let up = gate_up.narrow(candle_core::D::Minus1, d, d)?.contiguous()?;
    gate.gelu()?.mul(&up)
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

/// Fused add + RMS normalization — dispatches to CUDA fused kernel on GPU.
///
/// Computes `residual += input`, then `normed = rms_norm(residual) * weight`.
/// Returns `(normed, updated_residual)`.
///
/// On CPU: falls back to separate add + rms_norm.
pub fn fused_add_rms_norm(
    input: &Tensor,
    residual: &Tensor,
    norm: &RmsNorm,
) -> candle_core::Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if input.device().is_cuda() {
        use vllm_kernels::norm::{CudaNormKernels, NormKernels};
        return CudaNormKernels
            .fused_add_rms_norm(input, residual, norm.weight(), norm.eps())
            .map_err(kernel_err);
    }
    let updated = (input + residual)?;
    let normed = candle_core::Module::forward(norm, &updated)?;
    Ok((normed, updated))
}

/// Fused add + Gemma RMS normalization — dispatches to CUDA fused kernel on GPU.
///
/// Same as `fused_add_rms_norm` but uses GemmaRmsNorm (effective weight = weight + 1).
pub fn fused_add_gemma_rms_norm(
    input: &Tensor,
    residual: &Tensor,
    norm: &GemmaRmsNorm,
) -> candle_core::Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if input.device().is_cuda() {
        use vllm_kernels::norm::{CudaNormKernels, NormKernels};
        return CudaNormKernels
            .fused_add_rms_norm(input, residual, norm.weight(), norm.eps())
            .map_err(kernel_err);
    }
    let updated = (input + residual)?;
    let normed = candle_core::Module::forward(norm, &updated)?;
    Ok((normed, updated))
}

// ---------------------------------------------------------------------------
// Fused RoPE dispatch
// ---------------------------------------------------------------------------

/// Fused rotary position embedding — dispatches to CUDA fused kernel on GPU.
///
/// On CUDA: single kernel launch per tensor (vs 5-7 from candle decomposition).
/// On CPU: falls back to the standard `RotaryEmbedding::apply()` decomposition.
///
/// * `q` — query tensor `[num_tokens, num_q_heads, head_dim]`
/// * `k` — key tensor `[num_tokens, num_kv_heads, head_dim]`
/// * `positions` — `[num_tokens]` u32
/// * `cos_sin_cache` — `[max_pos, head_dim]` combined `[cos|sin]` cache
/// * `head_size` — dimension per attention head
pub fn rotary_embedding(
    q: &Tensor,
    k: &Tensor,
    positions: &Tensor,
    #[allow(unused_variables)] cos_sin_cache: &Tensor,
    #[allow(unused_variables)] head_size: usize,
) -> candle_core::Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() {
        // Fused Q+K RoPE: single kernel launch for both tensors.
        return vllm_kernels::rotary::fused_rotary_apply_qk(
            q, k, positions, cos_sin_cache, head_size,
        );
    }
    let _ = (q, k, positions);
    candle_core::bail!("ops::rotary_embedding requires CUDA; use RotaryEmbedding::apply() on CPU")
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

// ---------------------------------------------------------------------------
// GPU sampling dispatch
// ---------------------------------------------------------------------------

/// Fused top-k / top-p / min-p sampling entirely on GPU.
///
/// Avoids transferring full vocab logits to CPU. Only the 4-byte token ID
/// comes back. Dispatches to CUDA kernel on GPU, returns error on CPU.
#[cfg(feature = "cuda")]
pub fn gpu_sample_top_k_top_p(
    logits: &Tensor,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    uniform: f32,
) -> candle_core::Result<u32> {
    vllm_kernels::sampling::cuda_sample_top_k_top_p(
        logits,
        temperature,
        top_k,
        top_p,
        min_p,
        uniform,
    )
    .map_err(kernel_err)
}

/// Batched fused top-k / top-p / min-p sampling on GPU.
///
/// Samples all requests in a single kernel launch + single GPU sync,
/// eliminating per-request sync overhead. Falls back to error on CPU.
///
/// * `logits` — 2-D `[batch_size, vocab_size]` CUDA tensor
/// * Per-request parameter slices of length `batch_size`
///
/// Returns `Vec<u32>` of sampled token IDs.
#[cfg(feature = "cuda")]
pub fn gpu_sample_batched(
    logits: &Tensor,
    temperatures: &[f32],
    top_ks: &[i32],
    top_ps: &[f32],
    min_ps: &[f32],
    uniform_randoms: &[f32],
) -> candle_core::Result<Vec<u32>> {
    vllm_kernels::sampling::cuda_sample_batched(
        logits,
        temperatures,
        top_ks,
        top_ps,
        min_ps,
        uniform_randoms,
    )
    .map_err(kernel_err)
}

/// Re-export pre-allocated sampling buffers for the worker to hold.
#[cfg(feature = "cuda")]
pub use vllm_kernels::sampling::SamplingBuffers;

// ---------------------------------------------------------------------------
// MoE dispatch
// ---------------------------------------------------------------------------

/// Top-k softmax gating — dispatches to CUDA kernel on GPU, CPU fallback otherwise.
///
/// * `router_logits` — `[num_tokens, num_experts]`
/// * Returns `(topk_weights, topk_ids)` — `[num_tokens, top_k]` each (f32 and u32)
pub fn topk_softmax(
    router_logits: &Tensor,
    top_k: usize,
    renormalize: bool,
) -> candle_core::Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if router_logits.device().is_cuda() {
        use vllm_kernels::moe::{CudaMoeKernels, MoeKernels};
        return CudaMoeKernels
            .topk_softmax(router_logits, top_k, renormalize)
            .map_err(kernel_err);
    }
    use vllm_kernels::moe::{CpuMoeKernels, MoeKernels};
    CpuMoeKernels
        .topk_softmax(router_logits, top_k, renormalize)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))
}

/// Weighted sum across top-k expert outputs.
///
/// * `input` — `[num_tokens, top_k, hidden_size]` (already weighted by routing probs)
///
/// Returns `[num_tokens, hidden_size]`
pub fn moe_sum(input: &Tensor, top_k: usize) -> candle_core::Result<Tensor> {
    #[cfg(feature = "cuda")]
    if input.device().is_cuda() {
        use vllm_kernels::moe::{CudaMoeKernels, MoeKernels};
        return CudaMoeKernels.moe_sum(input, top_k).map_err(kernel_err);
    }
    use vllm_kernels::moe::{CpuMoeKernels, MoeKernels};
    CpuMoeKernels
        .moe_sum(input, top_k)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))
}
