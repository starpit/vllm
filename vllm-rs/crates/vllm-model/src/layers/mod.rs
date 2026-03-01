// SPDX-License-Identifier: Apache-2.0
//! Layer abstractions for building model architectures.
//!
//! Provides traits and implementations for common neural network layers:
//! - `Module`: base trait for forward pass
//! - `Linear`: dense linear layer (with column/row parallel variants)
//! - `RmsNorm`: RMS normalization
//! - `Embedding`: token embedding lookup
//! - Activation functions (SiLU, GELU, etc.)
//! - `RotaryEmbedding`: Rotary Position Embedding (RoPE)
//!
//! Port of: `vllm/model_executor/layers/`

pub mod activation;
pub mod embedding;
pub mod gptq;
pub mod linear;
pub mod norm;
pub mod quantized_linear;
pub mod rotary;

pub use activation::{Activation, gelu, relu, silu};
pub use embedding::Embedding;
pub use gptq::{GptqConfig, GptqLinear};
pub use linear::{ColumnParallelLinear, Linear, RowParallelLinear};
pub use norm::{CohereLayerNorm, GemmaRmsNorm, LayerNorm, RmsNorm};
pub use quantized_linear::QuantizedLinear;
pub use rotary::RotaryEmbedding;
