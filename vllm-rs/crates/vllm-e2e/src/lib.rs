// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! End-to-end test infrastructure for the vLLM Rust inference engine.
//!
//! Provides [`TestServer`] for starting the server in-process,
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

    // Tier 2: Small (<1 GB) — run on every PR
    pub const GEMMA3_270M_4BIT: &str = "mlx-community/gemma-3-270m-it-qat-4bit";

    // Tier 3: Medium (1–3 GB) — nightly only
    pub const GEMMA2_2B_4BIT: &str = "mlx-community/gemma-2-2b-it-4bit";
    pub const PHI3_5_MINI_4BIT: &str = "mlx-community/Phi-3.5-mini-instruct-4bit";
    pub const PHI4_MINI_4BIT: &str = "mlx-community/Unsloth-Phi-4-mini-instruct-4bit";

    // Tier 4: Large (3+ GB) — weekly/manual only
    pub const MISTRAL_7B_4BIT: &str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit";
    pub const DEEPSEEK_V2_LITE_4BIT: &str =
        "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx";

    // MoE models
    pub const QWEN3_MOE_4X06B_4BIT: &str =
        "justneedsomeavailableusername/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2-mlx-4Bit";

    // Float16 variants for non-quantized testing
    pub const SMOLLM_135M_F16: &str = "mlx-community/SmolLM2-135M-Instruct";

    // GPTQ quantized models (candle backend, not MLX)
    pub const QWEN2_0_5B_GPTQ_INT4: &str = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4";

    // AWQ quantized models (candle backend, not MLX)
    pub const QWEN2_0_5B_AWQ: &str = "Qwen/Qwen2.5-0.5B-Instruct-AWQ";

    // Gemma2 GPTQ quantized models (ungated)
    pub const GEMMA2_2B_GPTQ_INT4: &str = "qilowoq/gemma-2-2B-it-4Bit-GPTQ";

    // BitsAndBytes quantized models (MLX dequant-at-load or candle)
    pub const LLAMA_3_2_1B_BNB_4BIT: &str = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit";
    pub const TINYLLAMA_1B_BNB_8BIT: &str = "Jiqing/TinyLlama-1.1B-Chat-v1.0-bnb-8bit";

    // GGUF quantized models (candle backend, not MLX)
    pub const GEMMA3_270M_GGUF: &str = "unsloth/gemma-3-270m-it-qat-GGUF";
    pub const GEMMA3_1B_GGUF: &str = "unsloth/gemma-3-1b-it-GGUF";
    pub const QWEN2_0_5B_GGUF: &str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
    pub const QWEN3_0_6B_GGUF: &str = "unsloth/Qwen3-0.6B-GGUF";
    pub const QWEN3_NEXT_0_8B_GGUF: &str = "unsloth/Qwen3.5-0.8B-GGUF";

    // CUDA-compatible safetensors models (non-quantized, run on GPU)
    pub const SMOLLM_135M_CUDA: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    pub const QWEN2_0_5B_CUDA: &str = "Qwen/Qwen2.5-0.5B";
    // MoE models for CUDA — safetensors BF16
    // Mixtral: ~0.8B total params (~1.5GB BF16), MixtralForCausalLM
    pub const MIXTRAL_SMALL_CUDA: &str = "if001/small_mixtral_ja_llm_jp_tk";
    // Qwen2 MoE: ~14.3B total params (~29GB BF16), Qwen2MoeForCausalLM — fits on L40S (48GB)
    pub const QWEN2_MOE_A2_7B_CUDA: &str = "Qwen/Qwen1.5-MoE-A2.7B-Chat";

    // Qwen3 — safetensors BF16 for cuda-backend (~1.2GB)
    pub const QWEN3_0_6B_CUDA: &str = "Qwen/Qwen3-0.6B";

    // Gemma2 — safetensors BF16 for cuda-backend (~5GB, Gemma2ForCausalLM)
    pub const GEMMA2_2B_IT_CUDA: &str = "unsloth/gemma-2-2b-it";

    // Gemma3 — safetensors BF16 for cuda-backend (~2GB, Gemma3ForCausalLM)
    pub const GEMMA3_1B_IT_CUDA: &str = "unsloth/gemma-3-1b-it";

    // DeepSeek V2 — safetensors BF16 for TP testing (2x L40S)
    pub const DEEPSEEK_V2_LITE_CUDA: &str = "deepseek-ai/DeepSeek-V2-Lite";

    // Granite (IBM) — MLX 4-bit quantized
    pub const GRANITE_3_3_2B_4BIT: &str = "mlx-community/granite-3.3-2b-instruct-4bit";
    // Granite (IBM) — safetensors BF16, candle backend (CUDA)
    pub const GRANITE_3_3_2B_INSTRUCT: &str = "ibm-granite/granite-3.3-2b-instruct";
    // Granite GGUF — quantized, candle backend (CUDA)
    pub const GRANITE_3_3_2B_INSTRUCT_GGUF: &str = "ibm-granite/granite-3.3-2b-instruct-GGUF";

    // Multimodal (vision-language) models
    // Tier 3: ~2.8 GB — nightly only (QAT = quantization-aware training)
    pub const GEMMA3_4B_IT_QAT_3BIT: &str = "mlx-community/gemma-3-4b-it-qat-3bit";
    // Tier 4: ~8 GB BF16 SafeTensors — Gemma3ForConditionalGeneration (Candle path)
    pub const GEMMA3_4B_IT: &str = "google/gemma-3-4b-it";

    // Qwen2-VL multimodal (vision-language) models
    // Tier 3: ~3.8 GB BF16 SafeTensors — Candle path
    pub const QWEN2_VL_2B_INSTRUCT: &str = "unsloth/Qwen2-VL-2B-Instruct";
    // Tier 4: ~4.6 GB 4-bit quantized — MLX path
    pub const QWEN2_VL_7B_4BIT: &str = "mlx-community/Qwen2-VL-7B-4bit";
}
