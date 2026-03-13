// SPDX-License-Identifier: Apache-2.0
//! Model configuration, weight index, and quantization config types.
//!
//! This crate provides:
//! - **HfModelConfig** for parsing HuggingFace `config.json`
//! - **SafeTensorsIndex** for sharded weight map lookups
//! - **GGUF format** parsing for quantized models
//! - **Quantization configs** (AWQ, BnB, GPTQ)
//! - **LoRA adapter config** parsing

pub mod attention_metadata;
pub mod awq_config;
pub mod bnb_config;
pub mod embedding;
pub mod gguf;
pub mod gguf_format;
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
