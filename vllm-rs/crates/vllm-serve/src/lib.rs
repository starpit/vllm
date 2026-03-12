// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! HTTP serving layer for vLLM Rust port.
//!
//! Provides an OpenAI-compatible HTTP API server using axum, with endpoints
//! for chat completions, text completions, model listing, health checks,
//! and version information.
//!
//! Port of: `vllm/entrypoints/openai/` (subset)

pub mod anthropic;
#[cfg(feature = "chat-template")]
pub mod chat_template;
pub mod detokenizer;
pub mod engine;
pub mod error;
pub mod headless;
pub mod init;
pub mod llm;
#[cfg(feature = "metrics")]
pub mod metrics;
#[cfg(feature = "metrics")]
pub mod orca;
pub mod protocol;
pub mod reasoning_parser;
pub mod responses;
pub mod server;
pub mod tokenizer;
pub mod tool_parser;
