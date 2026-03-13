// SPDX-License-Identifier: Apache-2.0
//! Quantization config types for AWQ, BitsAndBytes, and GPTQ.
//!
//! The actual model layers (Linear, RmsNorm, Embedding, RoPE, activations)
//! live in each backend crate (vllm-cuda, vllm-mlx).

pub mod awq;
pub mod bnb;
pub mod gptq;

pub use awq::AwqConfig;
pub use bnb::{BnbLayerConfig, BnbNf4Config, BnbQuantType};
pub use gptq::GptqConfig;
