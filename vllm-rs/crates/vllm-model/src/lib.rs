// SPDX-License-Identifier: Apache-2.0
//! Model configuration, weight index, and quantization config types.
//!
//! This crate provides:
//! - **HfModelConfig** for parsing HuggingFace `config.json`
//! - **SafeTensorsIndex** for sharded weight map lookups
//! - **Quantization configs** (AWQ, BnB, GPTQ)
//! - **LoRA adapter config** parsing
//!
//! GGUF format support lives in the `ferrite-gguf` crate — adding GGUF
//! support to a new model arch should not touch this crate.

pub mod attention_metadata;
pub mod awq_config;
pub mod bnb_config;
pub mod embedding;
pub mod gptq_config;
#[cfg(feature = "guided-decoding")]
pub mod grammar;
#[cfg(feature = "multimodal")]
pub mod image;
pub mod layers;
pub mod lora;
pub mod process_group;
pub mod sampler;
pub mod tensor;
pub mod weight;

// Re-export for convenience.
pub use attention_metadata::AttentionMetadata;
pub use sampler::Sampler;
pub use tensor::error;
pub use tensor::error::{ModelError, ModelResult};
