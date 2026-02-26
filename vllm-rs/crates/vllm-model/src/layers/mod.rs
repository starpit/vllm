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

pub mod linear;
pub mod norm;
pub mod activation;
pub mod embedding;
pub mod rotary;

pub use linear::{Linear, ColumnParallelLinear, RowParallelLinear};
pub use norm::{RmsNorm, GemmaRmsNorm};
pub use activation::{Activation, silu, gelu, relu};
pub use embedding::Embedding;
pub use rotary::RotaryEmbedding;
