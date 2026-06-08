// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
// Low-level CUDA FFI crate: raw pointers are pervasive and intentional.
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Purpose-built CUDA tensor runtime for LLM inference.
//!
//! Uses a caching allocator (like PyTorch's CUDACachingAllocator) for all GPU
//! memory management. Tensors are freed on drop and their blocks reused from
//! a free list — zero `cudaMalloc` calls on the hot path after warmup.
//!
//! ## Key design principles
//!
//! 1. **Caching allocator**: Like PyTorch — free-list based, tensors freed on
//!    drop. Zero D2D copies between layers.
//! 2. **Always contiguous**: `GpuTensor` has no strides — eliminates
//!    `.contiguous()` copies.
//! 3. **Own streams**: Non-default compute and transfer streams enable
//!    CUDA graph capture and async scheduling overlap.
//! 4. **Minimal tensor type**: `GpuTensor` is 32 bytes, `Copy`, no `Drop`.
//!    `OwnedTensor` wraps it with automatic memory management.

// ---------------------------------------------------------------------------
// Core types re-exported from ferrite-cuda-core
// ---------------------------------------------------------------------------

// Always-available types (pure metadata, no CUDA calls).
pub use ferrite_cuda_core::dtype;
pub use ferrite_cuda_core::tensor;

pub use ferrite_cuda_core::{DType, GpuTensor, TensorView};

// CUDA runtime (requires CUDA toolkit).
#[cfg(feature = "cuda")]
pub use ferrite_cuda_core::alloc;
#[cfg(feature = "cuda")]
pub use ferrite_cuda_core::arena;
#[cfg(feature = "cuda")]
pub use ferrite_cuda_core::cpu_gpu_buf;
#[cfg(feature = "cuda")]
pub use ferrite_cuda_core::cublas;
#[cfg(feature = "cuda")]
pub use ferrite_cuda_core::device;
#[cfg(feature = "cuda")]
pub use ferrite_cuda_core::driver;
// weights: re-export core GpuWeights + local quantized loaders under one module.
#[cfg(feature = "cuda")]
pub mod weights {
    pub use crate::weights_quant::*;
    pub use ferrite_cuda_core::weights::*;
}

// ---------------------------------------------------------------------------
// Kernel/layer types re-exported from ferrite-kernels
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub use ferrite_kernels::attention_helpers;
#[cfg(feature = "cuda")]
pub use ferrite_kernels::forward_output;
#[cfg(feature = "cuda")]
pub use ferrite_kernels::ggml;
#[cfg(feature = "cuda")]
pub use ferrite_kernels::kernels;
#[cfg(feature = "cuda")]
pub use ferrite_kernels::kv_cache;
#[cfg(feature = "cuda")]
pub use ferrite_kernels::layers;
#[cfg(feature = "cuda")]
pub use ferrite_kernels::layers_moe;
#[cfg(feature = "cuda")]
pub use ferrite_kernels::rotary;

// ---------------------------------------------------------------------------
// vllm-cuda local modules
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub mod graph;
#[cfg(feature = "cuda")]
pub mod graph_piece;
#[cfg(feature = "cuda")]
pub mod logits_processor;
#[cfg(feature = "cuda")]
pub mod model;
#[cfg(feature = "nccl")]
pub use ferrite_cuda_core::nccl;
pub mod pp;
pub mod quant;
pub mod tcp_store;
pub use tcp_store::TcpControlChannel;
#[cfg(feature = "cuda")]
pub mod weights_quant;

// Layers test code (depends on vllm-cuda modules like quant, weights_quant).
#[cfg(all(feature = "cuda", test))]
#[path = "layers_tests.rs"]
mod layers_tests;

/// Total memory (bytes) and name of the current CUDA device.
///
/// Used to pick device-aware scheduler batch defaults (mirroring Python vLLM's
/// `EngineArgs.get_batch_defaults`). Requires a current CUDA context; returns
/// `None` if the device cannot be queried (caller falls back to base defaults).
#[cfg(feature = "cuda")]
pub fn current_device_total_bytes_and_name() -> Option<(u64, String)> {
    let (_free, total) = cudarc::driver::result::mem_get_info().ok()?;
    let dev = cudarc::driver::result::device::get(0).ok()?;
    let name = cudarc::driver::result::device::get_name(dev).ok()?;
    Some((total as u64, name))
}

// ---------------------------------------------------------------------------
// Flat re-exports for convenience
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub use ferrite_cuda_core::{
    CachingAllocator, CublasHandle, GpuDevice, OwnedTensor, RawGpuAlloc, RawGpuMem,
};
#[cfg(feature = "cuda")]
pub use ferrite_cuda_core::{CpuGpuBuf, PinnedBuf};
#[cfg(feature = "cuda")]
pub use weights::GpuWeights;

#[cfg(feature = "nccl")]
pub use ferrite_cuda_core::nccl::{NcclGroup, NcclId};
#[cfg(feature = "cuda")]
pub use ferrite_kernels::{
    Bnb4bitLinear, ColumnParallelLinear, Embedding, GgmlLinear, Linear, LinearLayer, MarlinLinear,
    RmsNorm, RowParallelLinear, VocabParallelEmbedding,
};
#[cfg(feature = "cuda")]
pub use ferrite_kernels::{ForwardOutput, KvCachePool};
#[cfg(feature = "cuda")]
pub use ferrite_kernels::{Llama3RopeScaling, LlamaConfig, RotaryCache};
#[cfg(feature = "cuda")]
pub use ferrite_kernels::{MarlinFusedMoELayer, MarlinSharedFusedMoELayer};
pub use pp::PpConfig;
