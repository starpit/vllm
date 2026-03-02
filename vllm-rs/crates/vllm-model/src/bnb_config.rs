// SPDX-License-Identifier: Apache-2.0
//! BitsAndBytes `quantization_config` parsing.
//!
//! BitsAndBytes NF4 models on HuggingFace embed the quantization parameters
//! inside `config.json` under the `quantization_config` key. The key fields
//! are `quant_method: "bitsandbytes"`, `load_in_4bit: true`, and optionally
//! `bnb_4bit_quant_type`, `blocksize`, and `bnb_4bit_use_double_quant`.

use serde::Deserialize;

use crate::error::{ModelError, ModelResult};
use crate::layers::bnb::BnbNf4Config;

// ---------------------------------------------------------------------------
// BnbQuantizeConfig
// ---------------------------------------------------------------------------

/// Parsed `quantization_config` for BitsAndBytes from `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct BnbQuantizeConfig {
    #[serde(default)]
    pub load_in_4bit: bool,
    #[serde(default)]
    pub load_in_8bit: bool,
    #[serde(default = "default_nf4")]
    pub bnb_4bit_quant_type: String,
    #[serde(alias = "blocksize", default = "default_blocksize")]
    pub bnb_4bit_blocksize: usize,
    #[serde(default)]
    pub bnb_4bit_use_double_quant: bool,
    #[serde(default)]
    pub bnb_4bit_compute_dtype: Option<String>,
    #[serde(default)]
    pub quant_method: Option<String>,
}

fn default_nf4() -> String {
    "nf4".to_string()
}

fn default_blocksize() -> usize {
    64
}

impl BnbQuantizeConfig {
    /// Parse from a serde_json Value (e.g. from config.json `quantization_config`).
    pub fn from_json_value(value: &serde_json::Value) -> ModelResult<Self> {
        serde_json::from_value(value.clone())
            .map_err(|e| ModelError::Other(format!("failed to parse BnB quantization_config: {e}")))
    }

    /// Convert to the layer-level `BnbNf4Config`.
    pub fn to_bnb_config(&self) -> BnbNf4Config {
        let quant_type = match self.bnb_4bit_quant_type.as_str() {
            "fp4" => crate::layers::bnb::BnbQuantType::FP4,
            _ => crate::layers::bnb::BnbQuantType::NF4,
        };
        BnbNf4Config {
            quant_type,
            blocksize: self.bnb_4bit_blocksize,
            double_quant: self.bnb_4bit_use_double_quant,
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
    fn test_bnb_config_parse_full() {
        let json = r#"{
            "quant_method": "bitsandbytes",
            "load_in_4bit": true,
            "load_in_8bit": false,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_blocksize": 64,
            "bnb_4bit_use_double_quant": false,
            "bnb_4bit_compute_dtype": "bfloat16"
        }"#;
        let config: BnbQuantizeConfig = serde_json::from_str(json).unwrap();
        assert!(config.load_in_4bit);
        assert!(!config.load_in_8bit);
        assert_eq!(config.bnb_4bit_quant_type, "nf4");
        assert_eq!(config.bnb_4bit_blocksize, 64);
        assert!(!config.bnb_4bit_use_double_quant);
        assert_eq!(config.bnb_4bit_compute_dtype.as_deref(), Some("bfloat16"));
        assert_eq!(config.quant_method.as_deref(), Some("bitsandbytes"));
    }

    #[test]
    fn test_bnb_config_parse_minimal() {
        let json = r#"{
            "quant_method": "bitsandbytes",
            "load_in_4bit": true
        }"#;
        let config: BnbQuantizeConfig = serde_json::from_str(json).unwrap();
        assert!(config.load_in_4bit);
        assert_eq!(config.bnb_4bit_quant_type, "nf4"); // default
        assert_eq!(config.bnb_4bit_blocksize, 64); // default
        assert!(!config.bnb_4bit_use_double_quant); // default
    }

    #[test]
    fn test_bnb_config_fp4_variant() {
        let json = r#"{
            "quant_method": "bitsandbytes",
            "load_in_4bit": true,
            "bnb_4bit_quant_type": "fp4"
        }"#;
        let config: BnbQuantizeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.bnb_4bit_quant_type, "fp4");
        let layer_cfg = config.to_bnb_config();
        assert!(matches!(
            layer_cfg.quant_type,
            crate::layers::bnb::BnbQuantType::FP4
        ));
    }

    #[test]
    fn test_bnb_config_to_layer_config() {
        let json = r#"{
            "quant_method": "bitsandbytes",
            "load_in_4bit": true,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_blocksize": 64,
            "bnb_4bit_use_double_quant": true
        }"#;
        let config: BnbQuantizeConfig = serde_json::from_str(json).unwrap();
        let layer_cfg = config.to_bnb_config();
        assert!(matches!(
            layer_cfg.quant_type,
            crate::layers::bnb::BnbQuantType::NF4
        ));
        assert_eq!(layer_cfg.blocksize, 64);
        assert!(layer_cfg.double_quant);
    }

    #[test]
    fn test_bnb_config_from_json_value() {
        let value = serde_json::json!({
            "quant_method": "bitsandbytes",
            "load_in_4bit": true,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_blocksize": 64
        });
        let config = BnbQuantizeConfig::from_json_value(&value).unwrap();
        assert!(config.load_in_4bit);
        assert_eq!(config.bnb_4bit_blocksize, 64);
    }

    #[test]
    fn test_bnb_config_blocksize_alias() {
        // Some models use "blocksize" instead of "bnb_4bit_blocksize".
        let json = r#"{
            "quant_method": "bitsandbytes",
            "load_in_4bit": true,
            "blocksize": 128
        }"#;
        let config: BnbQuantizeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.bnb_4bit_blocksize, 128);
    }
}
