// SPDX-License-Identifier: Apache-2.0
//! Model loading, weight management, and tensor abstractions.
//!
//! This crate provides:
//! - **Tensor abstraction** wrapping `candle-core` for CPU and CUDA tensor ops
//! - **SafeTensors weight loading** for reading model weights from disk
//! - **Layer abstractions** for building model architectures
//!
//! Port of: `vllm/model_executor/model_loader/` and `vllm/model_executor/layers/`

pub mod layers;
pub mod tensor;
pub mod weight;

// The error module is defined in tensor.rs and re-exported here for convenience.
pub use tensor::error;
pub use tensor::error::{ModelError, ModelResult};
