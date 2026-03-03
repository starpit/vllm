// SPDX-License-Identifier: Apache-2.0
//! FFI bindings to CUDA/C++ kernels and CPU fallback implementations.
//!
//! This crate provides trait-based abstractions for GPU kernels used in
//! vLLM inference. Each trait defines the kernel interface, with:
//! - **CPU implementations** for testing without GPU hardware
//! - **CUDA FFI bindings** (behind `cuda` feature) for production use
//!
//! Port of: kernel functions declared in `csrc/ops.h` and `csrc/cache.h`

pub mod activation;
pub mod attention;
pub mod cache;
#[cfg(feature = "cuda")]
pub mod cuda_graph;
pub mod error;
pub mod moe;
#[cfg(feature = "nccl")]
pub mod nccl;
pub mod norm;
pub mod rotary;

pub use error::{KernelError, KernelResult};

use activation::ActivationKernels;
use moe::MoeKernels;
use norm::NormKernels;
use rotary::RotaryKernels;

// ---------------------------------------------------------------------------
// KernelSet: composite kernel dispatch
// ---------------------------------------------------------------------------

/// Composite kernel set for device-specific dispatch.
///
/// CandleWorker holds a `Box<dyn KernelSet>` selected at init based on
/// the target device (CPU vs CUDA). Model layers access individual kernel
/// traits through this interface.
pub trait KernelSet: Send + Sync {
    fn norm(&self) -> &dyn NormKernels;
    fn activation(&self) -> &dyn ActivationKernels;
    fn rotary(&self) -> &dyn RotaryKernels;
    fn moe(&self) -> &dyn MoeKernels;
}

/// CPU kernel set (always available).
pub struct CpuKernelSet;

impl KernelSet for CpuKernelSet {
    fn norm(&self) -> &dyn NormKernels {
        &norm::CpuNormKernels
    }
    fn activation(&self) -> &dyn ActivationKernels {
        &activation::CpuActivationKernels
    }
    fn rotary(&self) -> &dyn RotaryKernels {
        &rotary::CpuRotaryKernels
    }
    fn moe(&self) -> &dyn MoeKernels {
        &moe::CpuMoeKernels
    }
}

/// CUDA kernel set (fused kernels via FFI).
#[cfg(feature = "cuda")]
pub struct CudaKernelSet;

#[cfg(feature = "cuda")]
impl KernelSet for CudaKernelSet {
    fn norm(&self) -> &dyn NormKernels {
        &norm::CudaNormKernels
    }
    fn activation(&self) -> &dyn ActivationKernels {
        &activation::CudaActivationKernels
    }
    fn rotary(&self) -> &dyn RotaryKernels {
        &rotary::CudaRotaryKernels
    }
    fn moe(&self) -> &dyn MoeKernels {
        &moe::CudaMoeKernels
    }
}

/// Create the appropriate kernel set for the given device.
pub fn create_kernel_set(device: &candle_core::Device) -> Box<dyn KernelSet> {
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        return Box::new(CudaKernelSet);
    }
    let _ = device;
    Box::new(CpuKernelSet)
}
