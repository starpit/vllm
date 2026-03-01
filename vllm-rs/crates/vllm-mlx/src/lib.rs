// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! MLX backend for Apple Silicon GPU acceleration.
//!
//! This crate provides an alternative worker implementation backed by Apple's
//! MLX framework (via `mlx-rs`). Unlike candle's eager execution model which
//! creates ~1200 individual Metal kernel dispatches per forward pass, MLX uses
//! lazy evaluation with graph fusion — operations build a compute graph, then
//! `eval()` submits the entire graph as minimal Metal command buffers.
//!
//! Key components:
//! - [`worker::MlxWorker`] — implements the `Worker` trait from `vllm-executor`
//! - [`models`] — model architecture implementations using mlx-rs primitives
//! - [`cache`] — MLX KV cache (Array-based)

pub mod awq;
pub mod cache;
pub mod gptq;
pub mod lora;
pub mod models;
pub mod worker;
