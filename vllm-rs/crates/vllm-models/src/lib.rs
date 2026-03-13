// SPDX-License-Identifier: Apache-2.0
//! Shared model infrastructure: sampler, attention metadata, embeddings.
//!
//! Model architecture implementations have moved to the vllm-cuda backend.
//! This crate provides types shared across backends (CUDA, MLX).

pub mod attention_metadata;
pub mod embedding;
#[cfg(feature = "guided-decoding")]
pub mod grammar;
pub mod sampler;

// Re-export for convenience.
pub use attention_metadata::AttentionMetadata;
pub use sampler::Sampler;
