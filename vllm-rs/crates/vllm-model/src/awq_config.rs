// SPDX-License-Identifier: Apache-2.0
//! AWQ `quant_config.json` parsing.
//!
//! AWQ models on HuggingFace include a `quant_config.json` alongside
//! the standard `config.json`. This file specifies the quantization parameters
//! (w_bit/bits, q_group_size/group_size, zero_point, etc.).

use std::path::Path;

use serde::Deserialize;

use crate::error::{ModelError, ModelResult};
use crate::layers::awq::AwqConfig;

// ---------------------------------------------------------------------------
// AwqQuantizeConfig
// ---------------------------------------------------------------------------

/// Parsed contents of `quant_config.json` for AWQ.
#[derive(Debug, Clone, Deserialize)]
pub struct AwqQuantizeConfig {
    /// Quantization bit-width (AWQ calls this `w_bit`, some configs use `bits`).
    #[serde(alias = "w_bit")]
    pub bits: usize,
    /// Group size (AWQ calls this `q_group_size`, some configs use `group_size`).
    #[serde(alias = "q_group_size")]
    pub group_size: usize,
    /// Whether to use zero-point (most AWQ models default to true).
    #[serde(default = "default_true")]
    pub zero_point: bool,
    #[serde(default)]
    pub quant_method: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
}

fn default_true() -> bool {
    true
}

impl AwqQuantizeConfig {
    /// Parse from a JSON file path.
    pub fn from_file(path: impl AsRef<Path>) -> ModelResult<Self> {
        let path = path.as_ref();
        let data = std::fs::read_to_string(path)
            .map_err(|e| ModelError::Other(format!("failed to read {}: {e}", path.display())))?;
        let config: Self = serde_json::from_str(&data)
            .map_err(|e| ModelError::Other(format!("failed to parse {}: {e}", path.display())))?;
        Ok(config)
    }

    /// Parse from a directory containing `quant_config.json`.
    pub fn from_dir(dir: impl AsRef<Path>) -> ModelResult<Self> {
        Self::from_file(dir.as_ref().join("quant_config.json"))
    }

    /// Parse from a serde_json Value (e.g. from config.json `quantization_config`).
    pub fn from_json_value(value: &serde_json::Value) -> ModelResult<Self> {
        serde_json::from_value(value.clone())
            .map_err(|e| ModelError::Other(format!("failed to parse quantization_config: {e}")))
    }

    /// Convert to the layer-level `AwqConfig`.
    pub fn to_awq_config(&self) -> AwqConfig {
        AwqConfig {
            bits: self.bits,
            group_size: self.group_size,
            zero_point: self.zero_point,
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
    fn test_awq_config_parse_full() {
        let json = r#"{
            "zero_point": true,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
            "quant_method": "awq"
        }"#;
        let config: AwqQuantizeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 128);
        assert!(config.zero_point);
        assert_eq!(config.quant_method.as_deref(), Some("awq"));
    }

    #[test]
    fn test_awq_config_parse_alt_names() {
        // Some AWQ configs use `bits` and `group_size` instead of `w_bit` and `q_group_size`.
        let json = r#"{
            "bits": 4,
            "group_size": 64,
            "zero_point": true
        }"#;
        let config: AwqQuantizeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 64);
        assert!(config.zero_point);
    }

    #[test]
    fn test_awq_config_parse_minimal() {
        let json = r#"{
            "w_bit": 4,
            "q_group_size": 128
        }"#;
        let config: AwqQuantizeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 128);
        assert!(config.zero_point); // default true
    }

    #[test]
    fn test_awq_config_to_layer_config() {
        let json = r#"{
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": true
        }"#;
        let config: AwqQuantizeConfig = serde_json::from_str(json).unwrap();
        let layer_config = config.to_awq_config();
        assert_eq!(layer_config.bits, 4);
        assert_eq!(layer_config.group_size, 128);
        assert!(layer_config.zero_point);
    }

    #[test]
    fn test_awq_config_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("quant_config.json");
        std::fs::write(
            &path,
            r#"{"w_bit": 4, "q_group_size": 128, "zero_point": true}"#,
        )
        .unwrap();

        let config = AwqQuantizeConfig::from_file(&path).unwrap();
        assert_eq!(config.bits, 4);
        assert!(config.zero_point);
    }

    #[test]
    fn test_awq_config_from_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("quant_config.json"),
            r#"{"w_bit": 4, "q_group_size": 64}"#,
        )
        .unwrap();

        let config = AwqQuantizeConfig::from_dir(dir.path()).unwrap();
        assert_eq!(config.group_size, 64);
    }

    #[test]
    fn test_awq_config_missing_file() {
        let result = AwqQuantizeConfig::from_file("/nonexistent/path/quant_config.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_awq_config_from_json_value() {
        let value = serde_json::json!({
            "quant_method": "awq",
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": true,
            "version": "GEMM"
        });
        let config = AwqQuantizeConfig::from_json_value(&value).unwrap();
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 128);
    }
}
