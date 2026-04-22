// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Core GPU types for the ferrite inference framework.
//!
//! This crate provides the foundational types for GPU tensor management:
//! `GpuTensor` (32-byte descriptor), `TensorView` (lifetime-checked borrow),
//! `OwnedTensor` (RAII allocation), `GpuDevice` (streams + cublas + allocator),
//! and the caching allocator.

// Always-available types (pure metadata, no CUDA calls).
pub mod dtype;
pub mod tensor;

pub use dtype::DType;
pub use tensor::{GpuTensor, TensorView};

// CUDA runtime (requires CUDA toolkit).
#[cfg(feature = "cuda")]
pub mod alloc;
#[cfg(feature = "cuda")]
pub mod arena;
#[cfg(feature = "cuda")]
pub mod cpu_gpu_buf;
#[cfg(feature = "cublas")]
pub mod cublas;
#[cfg(feature = "cuda")]
pub mod device;
#[cfg(feature = "cuda")]
pub mod driver;
#[cfg(feature = "cuda")]
pub mod weights;

#[cfg(feature = "cuda")]
pub use alloc::{CachingAllocator, OwnedTensor, RawGpuAlloc, RawGpuMem};
#[cfg(feature = "cuda")]
pub use cpu_gpu_buf::{CpuGpuBuf, PinnedBuf};
#[cfg(feature = "cublas")]
pub use cublas::CublasHandle;
#[cfg(feature = "cuda")]
pub use device::GpuDevice;
#[cfg(feature = "cuda")]
pub use weights::GpuWeights;

/// Re-export `cudarc::driver::sys::CUstream` at a stable path so
/// generated code (ferrite-forward, ferrite-models) doesn't have to
/// pull cudarc into its own Cargo.toml.
#[cfg(feature = "cuda")]
pub use cudarc::driver::sys::CUstream;

#[cfg(feature = "nccl")]
pub mod nccl;
#[cfg(feature = "nccl")]
pub use nccl::{NcclGroup, NcclId};
