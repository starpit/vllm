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
#[cfg(all(feature = "cuda", fa3_built))]
pub mod flash_attn_3;
#[cfg(feature = "cuda")]
pub mod flashinfer;
#[cfg(feature = "cuda")]
pub mod forward_output;
#[cfg(feature = "cuda")]
pub mod ggml;
#[cfg(feature = "cuda")]
pub mod kernels;
// `kv_cache` is dual-mode: storage layout, sizing, and span
// bookkeeping live in one place. The buffer-allocation closure
// that callers pass into `KvCachePool::new` is what differs per
// backend; FP8 scale machinery + the D2D gather + GPU mirror flags
// stay `cfg(feature = "cuda")` *inside* the unified type.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod kv_cache;
// `gdn_state` is the non-paged recurrent-state sibling of `kv_cache`, for the
// linear-attention (Gated-DeltaNet) layers of hybrid models. Same dual-mode
// shape: backend-neutral layout/sizing/accessors, caller-supplied alloc closure.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod gdn_state;
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
// `rotary` is dual-mode like `layers`: the struct *definitions* and CPU-side
// math helpers compile without `cuda`; the cudarc-using stream constructors
// are individually `#[cfg(feature = "cuda")]`-gated inside the file. Under
// metal the [`new_from_gpuweights`] constructor builds the same cache via
// the active `DeviceAllocator`.
//
// [`new_from_gpuweights`]: rotary::RotaryCache::new_from_gpuweights
pub mod rotary;

#[cfg(feature = "cuda")]
pub use forward_output::ForwardOutput;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use gdn_state::GdnStatePool;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use kv_cache::KvCachePool;
// Layer struct types compile without `cuda` (see comment above the module
// declarations). Re-export them ungated so consumers (notably the
// `Instruction<W>` enum in ferrite-forward) can name them without the cuda
// feature.
pub use layers::{
    Bnb4bitLinear, ColumnParallelLinear, Embedding, GatedDeltaNetLayer, GgmlLinear, Linear,
    LinearLayer, MarlinLinear, RmsNorm, RowParallelLinear, VocabParallelEmbedding,
};
pub use layers_moe::{DeepSeekV2MoELayer, MarlinFusedMoELayer, MarlinSharedFusedMoELayer};
pub use rotary::{Llama3RopeScaling, LlamaConfig, LongRopeScaling, RotaryCache, YarnRopeScaling};
