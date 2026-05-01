// SPDX-License-Identifier: Apache-2.0
//! SafeTensors index and HuggingFace model config.
//!
//! Provides:
//! - `SafeTensorsIndex`: parse `model.safetensors.index.json` for shard lookups
//! - `HfModelConfig`: parse HuggingFace `config.json` for architecture info

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::ModelResult;

// ---------------------------------------------------------------------------
// Sharded weight index (model.safetensors.index.json)
// ---------------------------------------------------------------------------

/// The index file that maps tensor names to shard files.
///
/// Format: `model.safetensors.index.json`
/// ```json
/// {
///   "metadata": { "total_size": 12345 },
///   "weight_map": {
///     "model.layers.0.weight": "model-00001-of-00002.safetensors",
///     ...
///   }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeTensorsIndex {
    /// Metadata (usually just total_size).
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,

    /// Maps tensor name → shard filename.
    pub weight_map: HashMap<String, String>,
}

impl SafeTensorsIndex {
    /// Load from a JSON file.
    pub fn from_file(path: impl AsRef<Path>) -> ModelResult<Self> {
        let data = std::fs::read_to_string(path)?;
        let index: Self = serde_json::from_str(&data)?;
        Ok(index)
    }

    /// Get the set of unique shard filenames.
    pub fn shard_files(&self) -> Vec<String> {
        let mut files: Vec<String> = self.weight_map.values().cloned().collect();
        files.sort();
        files.dedup();
        files
    }

    /// Look up which shard file contains a given tensor.
    pub fn get_shard(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(|s| s.as_str())
    }
}

// ---------------------------------------------------------------------------
// HuggingFace config.json parser
// ---------------------------------------------------------------------------

/// Minimal HuggingFace model configuration parsed from `config.json`.
///
/// Only includes fields commonly needed by vLLM for model loading.
/// The full config can be accessed via the raw JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HfModelConfig {
    /// Model architecture identifiers (e.g., ["LlamaForCausalLM"]).
    #[serde(default, deserialize_with = "deserialize_null_as_default")]
    pub architectures: Vec<String>,

    /// Model type (e.g., "llama", "mistral", "qwen2").
    #[serde(default)]
    pub model_type: Option<String>,

    /// Hidden size / model dimension.
    #[serde(default)]
    pub hidden_size: Option<usize>,

    /// Number of attention heads.
    #[serde(default)]
    pub num_attention_heads: Option<usize>,

    /// Number of key-value heads (for GQA).
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Number of hidden layers.
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,

    /// Intermediate size (FFN dimension).
    #[serde(default)]
    pub intermediate_size: Option<usize>,

    /// Vocabulary size.
    #[serde(default)]
    pub vocab_size: Option<usize>,

    /// Maximum sequence length.
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    /// RMS norm epsilon.
    #[serde(default)]
    pub rms_norm_eps: Option<f64>,

    /// Layer norm epsilon.
    #[serde(default)]
    pub layer_norm_eps: Option<f64>,

    /// RoPE theta.
    #[serde(default)]
    pub rope_theta: Option<f64>,

    /// Torch dtype string (e.g., "float16", "bfloat16").
    #[serde(default)]
    pub torch_dtype: Option<String>,

    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,

    /// Head dimension (if explicitly specified).
    #[serde(default)]
    pub head_dim: Option<usize>,

    /// Raw JSON for accessing any field not in this struct.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl HfModelConfig {
    /// Load from a `config.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> ModelResult<Self> {
        let data = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&data)?;
        Ok(config)
    }

    /// Load from a model directory (reads `config.json` inside it).
    pub fn from_dir(dir: impl AsRef<Path>) -> ModelResult<Self> {
        let path = dir.as_ref().join("config.json");
        Self::from_file(path)
    }

    /// Load from a model directory (reads `config.json`).
    ///
    /// For `.gguf` paths, callers must use `ferrite_gguf::gguf_model_config`
    /// directly — GGUF format support lives in ferrite, not vllm-model.
    pub fn from_path(path: impl AsRef<Path>) -> ModelResult<Self> {
        Self::from_dir(path)
    }

    /// Effective head dimension.
    ///
    /// For MLA models (DeepSeek V2/V3) this returns `qk_nope_head_dim + qk_rope_head_dim`
    /// so that the KV cache is allocated with the correct dimension.
    pub fn head_dim(&self) -> Option<usize> {
        if let (Some(nope), Some(rope)) = (
            self.extra.get("qk_nope_head_dim").and_then(|v| v.as_u64()),
            self.extra.get("qk_rope_head_dim").and_then(|v| v.as_u64()),
        ) {
            return Some((nope + rope) as usize);
        }
        self.head_dim
            .or_else(|| match (self.hidden_size, self.num_attention_heads) {
                (Some(h), Some(n)) if n > 0 => Some(h / n),
                _ => None,
            })
    }

    /// Effective number of KV heads (defaults to num_attention_heads for MHA).
    pub fn num_kv_heads(&self) -> Option<usize> {
        self.num_key_value_heads.or(self.num_attention_heads)
    }

    /// Effective norm epsilon.
    pub fn norm_eps(&self) -> f64 {
        self.rms_norm_eps.or(self.layer_norm_eps).unwrap_or(1e-5)
    }

    /// For composite models (e.g. vision-language models like Kimi K2.5),
    /// extract the text sub-config.
    ///
    /// Returns `Some((text_config, weight_prefix_to_strip))` if this is a
    /// composite model whose text backbone is a supported architecture.
    /// Returns `None` for non-composite models.
    pub fn resolve_text_config(&self) -> Option<(Self, &'static str)> {
        match self.model_type.as_deref() {
            Some("kimi_k25") => {
                let text_config_val = self.extra.get("text_config")?;
                let mut cfg: Self = serde_json::from_value(text_config_val.clone()).ok()?;
                if cfg.architectures.is_empty() {
                    cfg.architectures = self.architectures.clone();
                }
                Some((cfg, "language_model."))
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deserialize a value that may be `null` as the type's `Default`.
fn deserialize_null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + serde::Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_index_parse() {
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("model.safetensors.index.json");

        let index_json = r#"{
            "metadata": {"total_size": 1000},
            "weight_map": {
                "model.embed.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.weight": "model-00002-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors"
            }
        }"#;
        std::fs::write(&index_path, index_json).unwrap();

        let index = SafeTensorsIndex::from_file(&index_path).unwrap();
        assert_eq!(index.weight_map.len(), 4);

        let shards = index.shard_files();
        assert_eq!(shards.len(), 2);

        assert_eq!(
            index.get_shard("model.embed.weight"),
            Some("model-00001-of-00002.safetensors")
        );
        assert_eq!(
            index.get_shard("model.layers.1.weight"),
            Some("model-00002-of-00002.safetensors")
        );
        assert_eq!(index.get_shard("nonexistent"), None);
    }

    #[test]
    fn test_hf_model_config_parse() {
        let config_json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "torch_dtype": "float16",
            "tie_word_embeddings": false
        }"#;

        let config: HfModelConfig = serde_json::from_str(config_json).unwrap();
        assert_eq!(config.architectures, vec!["LlamaForCausalLM"]);
        assert_eq!(config.model_type, Some("llama".to_string()));
        assert_eq!(config.hidden_size, Some(4096));
        assert_eq!(config.num_attention_heads, Some(32));
        assert_eq!(config.num_key_value_heads, Some(8));
        assert_eq!(config.num_hidden_layers, Some(32));
        assert_eq!(config.intermediate_size, Some(11008));
        assert_eq!(config.vocab_size, Some(32000));
        assert_eq!(config.head_dim(), Some(128));
        assert_eq!(config.num_kv_heads(), Some(8));
        assert!((config.norm_eps() - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_hf_model_config_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let config_json = r#"{
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_attention_heads": 32
        }"#;
        std::fs::write(dir.path().join("config.json"), config_json).unwrap();

        let config = HfModelConfig::from_dir(dir.path()).unwrap();
        assert_eq!(config.model_type, Some("mistral".to_string()));
    }

    #[test]
    fn test_hf_model_config_defaults() {
        let config: HfModelConfig = serde_json::from_str("{}").unwrap();
        assert!(config.architectures.is_empty());
        assert_eq!(config.model_type, None);
        assert_eq!(config.hidden_size, None);
        assert_eq!(config.head_dim(), None);
        assert_eq!(config.num_kv_heads(), None);
        assert!((config.norm_eps() - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_hf_model_config_null_architectures() {
        let config: HfModelConfig =
            serde_json::from_str(r#"{"architectures": null, "hidden_size": 2560}"#).unwrap();
        assert!(config.architectures.is_empty());
        assert_eq!(config.hidden_size, Some(2560));
    }

    #[test]
    fn test_resolve_text_config_kimi_k25() {
        let config_json = r#"{
            "architectures": ["KimiK25ForCausalLM"],
            "model_type": "kimi_k25",
            "text_config": {
                "model_type": "deepseek_v2",
                "hidden_size": 7168,
                "num_attention_heads": 128,
                "num_hidden_layers": 61,
                "vocab_size": 129280,
                "rms_norm_eps": 1e-6
            },
            "vision_config": {
                "image_size": 384
            }
        }"#;
        let config: HfModelConfig = serde_json::from_str(config_json).unwrap();
        let result = config.resolve_text_config();
        assert!(result.is_some());

        let (text_cfg, prefix) = result.unwrap();
        assert_eq!(prefix, "language_model.");
        assert_eq!(text_cfg.model_type, Some("deepseek_v2".to_string()));
        assert_eq!(text_cfg.hidden_size, Some(7168));
        assert_eq!(text_cfg.num_hidden_layers, Some(61));
        assert_eq!(text_cfg.architectures, vec!["KimiK25ForCausalLM"]);
    }

    #[test]
    fn test_resolve_text_config_non_composite() {
        let config_json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096
        }"#;
        let config: HfModelConfig = serde_json::from_str(config_json).unwrap();
        assert!(config.resolve_text_config().is_none());
    }

    #[test]
    fn test_hf_model_config_extra_fields() {
        let config_json = r#"{
            "model_type": "qwen2",
            "sliding_window": 4096,
            "use_cache": true
        }"#;
        let config: HfModelConfig = serde_json::from_str(config_json).unwrap();
        assert_eq!(config.model_type, Some("qwen2".to_string()));
        assert!(config.extra.contains_key("sliding_window"));
        assert_eq!(config.extra["sliding_window"], 4096);
    }
}
