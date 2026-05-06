// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! GPU kernel wrappers and layer types for the ferrite inference framework.
//!
//! This crate provides:
//! - Kernel FFI wrappers (`kernels`) with safe Rust interfaces
//! - Layer types (`LinearLayer`, `RmsNorm`, `Embedding`, etc.)
//! - KV cache management (`KvCachePool`)
//! - Attention helpers
//! - Rotary positional embedding cache

#[cfg(feature = "cuda")]
pub mod attention_helpers;
#[cfg(feature = "cuda")]
pub mod cutlass;
#[cfg(feature = "cuda")]
pub mod flashinfer;
#[cfg(feature = "cuda")]
pub mod forward_output;
#[cfg(feature = "cuda")]
pub mod ggml;
#[cfg(feature = "cuda")]
pub mod kernels;
#[cfg(feature = "cuda")]
pub mod kv_cache;
// `layers` and `layers_moe` are dual-mode: the struct *definitions* compile
// without the `cuda` feature (they reference only `GpuTensor`, which lives in
// the always-available `ferrite_cuda_core::tensor` module), so the
// `Instruction<W>` enum the frontend produces resolves on Apple Silicon Metal
// builds. The `impl` blocks that use cudarc / `CachingAllocator` / etc. are
// individually `#[cfg(feature = "cuda")]`-gated inside each file.
pub mod layers;
pub mod layers_moe;
#[cfg(feature = "cuda")]
pub mod layers_quant;
#[cfg(feature = "cuda")]
pub mod rotary;

#[cfg(feature = "cuda")]
pub use forward_output::ForwardOutput;
#[cfg(feature = "cuda")]
pub use kv_cache::KvCachePool;
// Layer struct types compile without `cuda` (see comment above the module
// declarations). Re-export them ungated so consumers (notably the
// `Instruction<W>` enum in ferrite-forward) can name them without the cuda
// feature.
pub use layers::{
    Bnb4bitLinear, ColumnParallelLinear, Embedding, GgmlLinear, Linear, LinearLayer, MarlinLinear,
    RmsNorm, RowParallelLinear, VocabParallelEmbedding,
};
pub use layers_moe::{DeepSeekV2MoELayer, MarlinFusedMoELayer, MarlinSharedFusedMoELayer};
#[cfg(feature = "cuda")]
pub use rotary::{Llama3RopeScaling, LlamaConfig, LongRopeScaling, RotaryCache, YarnRopeScaling};
