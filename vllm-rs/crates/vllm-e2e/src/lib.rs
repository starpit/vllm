// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! End-to-end test infrastructure for the vLLM Rust inference engine.
//!
//! Provides [`TestServer`] for spawning a real `vllm serve` process,
//! [`Client`] for sending HTTP requests, and assertion helpers for
//! validating OpenAI-compatible responses.

pub mod assertions;
pub mod client;
pub mod server;

pub use client::Client;
pub use server::TestServer;

/// Test models by architecture (smallest available for CI).
pub struct TestModels;

impl TestModels {
    // Tier 1: Tiny (<500 MB) — run on every PR
    pub const SMOLLM_135M_4BIT: &str = "mlx-community/SmolLM-135M-Instruct-4bit";
    pub const QWEN2_0_5B_4BIT: &str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit";
    pub const QWEN3_0_6B_4BIT: &str = "mlx-community/Qwen3-0.6B-4bit";

    // Tier 2: Small (<1 GB) — run on every PR
    pub const LLAMA_3_2_1B_4BIT: &str = "mlx-community/Llama-3.2-1B-Instruct-4bit";

    // Tier 3: Medium (1–3 GB) — nightly only
    pub const GEMMA2_2B_4BIT: &str = "mlx-community/gemma-2-2b-it-4bit";
    pub const PHI3_5_MINI_4BIT: &str = "mlx-community/Phi-3.5-mini-instruct-4bit";

    // Tier 4: Large (3+ GB) — weekly/manual only
    pub const MISTRAL_7B_4BIT: &str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit";
    pub const DEEPSEEK_V2_LITE_4BIT: &str =
        "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx";

    // Float16 variants for non-quantized testing
    pub const SMOLLM_135M_F16: &str = "mlx-community/SmolLM2-135M-Instruct";
}
