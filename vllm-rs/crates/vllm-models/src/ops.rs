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
