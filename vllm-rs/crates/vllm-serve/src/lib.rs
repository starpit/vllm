// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! HTTP serving layer for vLLM Rust port.
//!
//! Provides an OpenAI-compatible HTTP API server using axum, with endpoints
//! for chat completions, text completions, model listing, health checks,
//! and version information.
//!
//! Port of: `vllm/entrypoints/openai/` (subset)

pub mod chat_template;
pub mod detokenizer;
pub mod engine;
pub mod error;
pub mod metrics;
pub mod protocol;
pub mod server;
pub mod tokenizer;
