// SPDX-License-Identifier: Apache-2.0
//! Quantization config types for AWQ, BitsAndBytes, and GPTQ.
//!
//! The actual model layers (Linear, RmsNorm, Embedding, RoPE, activations)
//! live in `vllm-cuda` for CUDA and `ferrite-metal-kernels` /
//! `ferrite-model-*` for Metal (`vllm-mlx` was removed — see
//! `project_vllm_mlx_nuke_plan`).

pub mod awq;
pub mod bnb;
pub mod gptq;

pub use awq::AwqConfig;
pub use bnb::{BnbLayerConfig, BnbNf4Config, BnbQuantType};
pub use gptq::GptqConfig;
