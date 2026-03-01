// SPDX-License-Identifier: Apache-2.0
//! GPTQ `quantize_config.json` parsing.
//!
//! GPTQ models on HuggingFace include a `quantize_config.json` alongside
//! the standard `config.json`. This file specifies the quantization parameters
//! (bits, group_size, desc_act, sym, etc.).

use std::path::Path;

use serde::Deserialize;

use crate::error::{ModelError, ModelResult};
use crate::layers::gptq::GptqConfig;

// ---------------------------------------------------------------------------
// GptqQuantizeConfig
// ---------------------------------------------------------------------------

/// Parsed contents of `quantize_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct GptqQuantizeConfig {
    pub bits: usize,
    pub group_size: usize,
    #[serde(default)]
    pub desc_act: bool,
    #[serde(default = "default_true")]
    pub sym: bool,
    #[serde(default)]
    pub quant_method: Option<String>,
    #[serde(default)]
    pub model_name_or_path: Option<String>,
}

fn default_true() -> bool {
    true
}

impl GptqQuantizeConfig {
    /// Parse from a JSON file path.
    pub fn from_file(path: impl AsRef<Path>) -> ModelResult<Self> {
        let path = path.as_ref();
        let data = std::fs::read_to_string(path).map_err(|e| {
            ModelError::Other(format!(
                "failed to read {}: {e}",
                path.display()
            ))
        })?;
        let config: Self = serde_json::from_str(&data).map_err(|e| {
            ModelError::Other(format!(
                "failed to parse {}: {e}",
                path.display()
            ))
        })?;
        Ok(config)
    }

    /// Parse from a directory containing `quantize_config.json`.
    pub fn from_dir(dir: impl AsRef<Path>) -> ModelResult<Self> {
        Self::from_file(dir.as_ref().join("quantize_config.json"))
    }

    /// Parse from a serde_json Value (e.g. from config.json `quantization_config`).
    pub fn from_json_value(value: &serde_json::Value) -> ModelResult<Self> {
        serde_json::from_value(value.clone()).map_err(|e| {
            ModelError::Other(format!("failed to parse quantization_config: {e}"))
        })
    }

    /// Convert to the layer-level `GptqConfig`.
    pub fn to_gptq_config(&self) -> GptqConfig {
        GptqConfig {
            bits: self.bits,
            group_size: self.group_size,
            desc_act: self.desc_act,
            sym: self.sym,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gptq_config_parse_full() {
        let json = r#"{
            "bits": 4,
            "group_size": 128,
            "desc_act": true,
            "sym": true,
            "quant_method": "gptq",
            "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"
        }"#;
        let config: GptqQuantizeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 128);
        assert!(config.desc_act);
        assert!(config.sym);
        assert_eq!(config.quant_method.as_deref(), Some("gptq"));
    }

    #[test]
    fn test_gptq_config_parse_minimal() {
        let json = r#"{
            "bits": 4,
            "group_size": 128
        }"#;
        let config: GptqQuantizeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 128);
        assert!(!config.desc_act); // default false
        assert!(config.sym); // default true
    }

    #[test]
    fn test_gptq_config_to_layer_config() {
        let json = r#"{
            "bits": 4,
            "group_size": 128,
            "desc_act": true,
            "sym": false
        }"#;
        let config: GptqQuantizeConfig = serde_json::from_str(json).unwrap();
        let layer_config = config.to_gptq_config();
        assert_eq!(layer_config.bits, 4);
        assert_eq!(layer_config.group_size, 128);
        assert!(layer_config.desc_act);
        assert!(!layer_config.sym);
    }

    #[test]
    fn test_gptq_config_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("quantize_config.json");
        std::fs::write(
            &path,
            r#"{"bits": 4, "group_size": 128, "desc_act": true, "sym": true}"#,
        )
        .unwrap();

        let config = GptqQuantizeConfig::from_file(&path).unwrap();
        assert_eq!(config.bits, 4);
        assert!(config.desc_act);
    }

    #[test]
    fn test_gptq_config_from_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("quantize_config.json"),
            r#"{"bits": 4, "group_size": 64}"#,
        )
        .unwrap();

        let config = GptqQuantizeConfig::from_dir(dir.path()).unwrap();
        assert_eq!(config.group_size, 64);
    }

    #[test]
    fn test_gptq_config_missing_file() {
        let result = GptqQuantizeConfig::from_file("/nonexistent/path/quantize_config.json");
        assert!(result.is_err());
    }
}
