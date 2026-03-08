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
//!    `.contiguous()` copies that plague the candle backend.
//! 3. **Own streams**: Non-default compute and transfer streams enable
//!    CUDA graph capture and async scheduling overlap.
//! 4. **Minimal tensor type**: `GpuTensor` is 32 bytes, `Copy`, no `Drop`.
//!    `OwnedTensor` wraps it with automatic memory management.

// Always-available types (pure metadata, no CUDA calls).
pub mod dtype;
pub mod tensor;

pub use dtype::DType;
pub use tensor::GpuTensor;

// CUDA runtime (requires CUDA toolkit).
#[cfg(feature = "cuda")]
pub mod alloc;
#[cfg(feature = "cuda")]
pub mod arena; // kept for backwards compat; not used in forward path
#[cfg(feature = "cuda")]
pub mod cpu_gpu_buf;
#[cfg(feature = "cuda")]
pub mod cublas;
#[cfg(feature = "cuda")]
pub mod device;
#[cfg(feature = "cuda")]
pub mod driver;
#[cfg(feature = "cuda")]
pub mod graph;
#[cfg(feature = "cuda")]
pub mod kernels;
#[cfg(feature = "cuda")]
pub mod kv_cache;
#[cfg(feature = "cuda")]
pub mod layers;
#[cfg(feature = "cuda")]
pub mod model;
#[cfg(feature = "cuda")]
pub mod weights;

#[cfg(feature = "cuda")]
pub use alloc::{CachingAllocator, OwnedTensor};
#[cfg(feature = "cuda")]
pub use cpu_gpu_buf::{CpuGpuBuf, PinnedBuf};
#[cfg(feature = "cuda")]
pub use cublas::CublasHandle;
#[cfg(feature = "cuda")]
pub use device::GpuDevice;
#[cfg(feature = "cuda")]
pub use kv_cache::KvCachePool;
#[cfg(feature = "cuda")]
pub use layers::{Embedding, Linear, RmsNorm};
#[cfg(feature = "cuda")]
pub use weights::GpuWeights;
