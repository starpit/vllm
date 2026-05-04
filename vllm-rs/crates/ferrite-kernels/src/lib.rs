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
#[cfg(feature = "cuda")]
pub mod layers;
#[cfg(feature = "cuda")]
pub mod layers_attn_gated;
#[cfg(feature = "cuda")]
pub mod layers_gdn;
#[cfg(feature = "cuda")]
pub mod layers_moe;
#[cfg(feature = "cuda")]
pub mod layers_quant;
#[cfg(feature = "cuda")]
pub mod rotary;

#[cfg(feature = "cuda")]
pub use forward_output::ForwardOutput;
#[cfg(feature = "cuda")]
pub use kv_cache::KvCachePool;
#[cfg(feature = "cuda")]
pub use layers::{
    Bnb4bitLinear, ColumnParallelLinear, Embedding, GgmlLinear, Linear, LinearLayer, MarlinLinear,
    RmsNorm, RowParallelLinear, VocabParallelEmbedding,
};
#[cfg(feature = "cuda")]
pub use layers_attn_gated::Qwen3NextGatedAttentionLayer;
#[cfg(feature = "cuda")]
pub use layers_gdn::{GdnStatePool, Qwen3NextGdnLayer};
#[cfg(feature = "cuda")]
pub use layers_moe::{DeepSeekV2MoELayer, MarlinFusedMoELayer, MarlinSharedFusedMoELayer};
#[cfg(feature = "cuda")]
pub use rotary::{Llama3RopeScaling, LlamaConfig, LongRopeScaling, RotaryCache, YarnRopeScaling};
