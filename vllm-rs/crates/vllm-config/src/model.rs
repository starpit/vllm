// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Model configuration types, ported from `vllm/config/model.py`.
//!
//! Only the subset of fields that are needed by the scheduler, KV cache
//! manager, and related Rust components are included here.  The full Python
//! `ModelConfig` is extremely large and depends on HuggingFace / PyTorch types
//! that have no direct Rust equivalent.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// Model data type selection (mirrors the Python `ModelDType` literal).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelDType {
    #[default]
    Auto,
    Half,
    #[serde(rename = "float16")]
    Float16,
    #[serde(rename = "bfloat16")]
    BFloat16,
    Float,
    #[serde(rename = "float32")]
    Float32,
}

/// Attention architecture type.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttnType {
    #[default]
    Decoder,
    Encoder,
    EncoderOnly,
    EncoderDecoder,
    AttentionFree,
    Hybrid,
}

/// Configuration for the model.
///
/// This is a simplified Rust port of `vllm.config.model.ModelConfig`,
/// containing only the fields required by the scheduler, KV cache manager,
/// and related Rust-side components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Name or path of the Hugging Face model.
    #[serde(default = "default_model_name")]
    pub model: Cow<'static, str>,

    /// Model context length (prompt + output).
    pub max_model_len: usize,

    /// Maximum number of log probabilities to return.  -1 means no cap.
    #[serde(default = "default_max_logprobs")]
    pub max_logprobs: i32,

    /// Whether the model is an encoder-decoder architecture.
    #[serde(default)]
    pub is_encoder_decoder: bool,

    /// Whether to return routed experts information.
    #[serde(default)]
    pub enable_return_routed_experts: bool,

    /// Data type for model weights and activations.
    #[serde(default)]
    pub dtype: ModelDType,

    /// Random seed for reproducibility.
    #[serde(default)]
    pub seed: u64,

    /// Whether to always use eager-mode execution (no CUDA graph).
    #[serde(default)]
    pub enforce_eager: bool,

    /// Whether the model is attention-free (e.g. pure Mamba).
    #[serde(default)]
    pub is_attention_free: bool,

    /// Whether the model is multimodal.
    #[serde(default)]
    pub is_multimodal_model: bool,

    /// Whether sliding window is disabled.
    #[serde(default)]
    pub disable_sliding_window: bool,

    /// Whether cascade attention is disabled.
    #[serde(default)]
    pub disable_cascade_attn: bool,

    /// Whether tokenizer initialization is skipped.
    #[serde(default)]
    pub skip_tokenizer_init: bool,

    /// Whether passing text embeddings as inputs is enabled.
    #[serde(default)]
    pub enable_prompt_embeds: bool,

    /// Attention architecture type.
    #[serde(default)]
    pub attn_type: AttnType,
}

fn default_model_name() -> Cow<'static, str> {
    Cow::Borrowed("Qwen/Qwen3-0.6B")
}

fn default_max_logprobs() -> i32 {
    20
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model: Cow::Borrowed("Qwen/Qwen3-0.6B"),
            max_model_len: 8192,
            max_logprobs: 20,
            is_encoder_decoder: false,
            enable_return_routed_experts: false,
            dtype: ModelDType::default(),
            seed: 0,
            enforce_eager: false,
            is_attention_free: false,
            is_multimodal_model: false,
            disable_sliding_window: false,
            disable_cascade_attn: false,
            skip_tokenizer_init: false,
            enable_prompt_embeds: false,
            attn_type: AttnType::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model_config() {
        let cfg = ModelConfig::default();
        assert_eq!(cfg.max_model_len, 8192);
        assert_eq!(cfg.max_logprobs, 20);
        assert!(!cfg.is_encoder_decoder);
        assert!(!cfg.enable_return_routed_experts);
    }

    #[test]
    fn test_model_config_roundtrip() {
        let cfg = ModelConfig {
            model: Cow::Borrowed("meta-llama/Llama-3-8B"),
            max_model_len: 4096,
            is_encoder_decoder: true,
            max_logprobs: -1,
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.model, cfg2.model);
        assert_eq!(cfg.max_model_len, cfg2.max_model_len);
        assert!(cfg2.is_encoder_decoder);
        assert_eq!(cfg2.max_logprobs, -1);
    }
}
