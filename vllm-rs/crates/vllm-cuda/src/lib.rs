// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
// Low-level CUDA FFI crate: raw pointers are pervasive and intentional.
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Purpose-built CUDA tensor runtime for LLM inference.
//!
//! Replaces candle as the GPU backend with a zero-allocation, arena-based
//! design optimized for CUDA graph capture and maximum throughput.
//!
//! ## Key design principles
//!
//! 1. **Zero hot-path allocation**: All intermediate tensors come from a
//!    bump-allocated `ScratchArena` that resets each engine step.
//! 2. **Always contiguous**: `GpuTensor` has no strides — eliminates
//!    `.contiguous()` copies that plague the candle backend.
//! 3. **Own streams**: Non-default compute and transfer streams enable
//!    CUDA graph capture and async scheduling overlap.
//! 4. **Minimal tensor type**: `GpuTensor` is 32 bytes, `Copy`, no `Drop`.
//!    Pointer extraction is `t.as_ptr::<T>()`, not 10 lines of match/slice.

// Always-available types (pure metadata, no CUDA calls).
pub mod dtype;
pub mod tensor;

pub use dtype::DType;
pub use tensor::GpuTensor;

// CUDA runtime (requires CUDA toolkit).
#[cfg(feature = "cuda")]
pub mod arena;
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
pub use arena::ScratchArena;
#[cfg(feature = "cuda")]
pub use cpu_gpu_buf::CpuGpuBuf;
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
