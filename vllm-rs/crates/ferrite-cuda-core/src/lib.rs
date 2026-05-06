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
pub mod device_allocator;
pub mod dtype;
pub mod ggml_quant;
pub mod tensor;

#[cfg(feature = "metal")]
pub mod metal_allocator;

pub use device_allocator::DeviceAllocator;
pub use dtype::DType;
pub use ggml_quant::{GgmlDType, GgmlStorage};
#[cfg(feature = "metal")]
pub use metal_allocator::MetalAllocator;
pub use tensor::{GpuTensor, TensorView};

// CUDA runtime (requires CUDA toolkit).
#[cfg(feature = "cuda")]
pub mod alloc;
#[cfg(feature = "cuda")]
pub mod arena;
#[cfg(feature = "cuda")]
pub mod cpu_gpu_buf;
#[cfg(feature = "cuda")]
pub mod cublas;
#[cfg(feature = "cuda")]
pub mod cuda_allocator;
#[cfg(feature = "cuda")]
pub mod device;
#[cfg(feature = "cuda")]
pub mod driver;
#[cfg(feature = "cuda")]
pub mod gguf_loader;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod weights;

#[cfg(feature = "cuda")]
pub use alloc::{CachingAllocator, OwnedTensor, RawGpuAlloc, RawGpuMem};
#[cfg(feature = "cuda")]
pub use cuda_allocator::CudaAllocator;

/// The concrete allocator type for the active backend. CUDA xor
/// Metal — features are mutually exclusive — so this is statically
/// determined at build time. `GpuWeights` holds one as a field, and
/// backend-specific accessors (e.g. `take_gpu_allocs` returning
/// `Vec<RawGpuMem>`) live on `impl GpuWeights` blocks gated to the
/// matching feature.
#[cfg(feature = "cuda")]
pub type BackendAllocator = CudaAllocator;
#[cfg(feature = "metal")]
pub type BackendAllocator = MetalAllocator;
#[cfg(feature = "cuda")]
pub use cpu_gpu_buf::{CpuGpuBuf, PinnedBuf};
#[cfg(feature = "cuda")]
pub use cublas::CublasHandle;
#[cfg(feature = "cuda")]
pub use device::GpuDevice;
#[cfg(any(feature = "cuda", feature = "metal"))]
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
