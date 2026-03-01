// SPDX-License-Identifier: Apache-2.0
//! LoRA (Low-Rank Adaptation) adapter loading.
//!
//! Supports loading PEFT-format LoRA adapters from local directories
//! containing `adapter_config.json` and `adapter_model.safetensors`.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

use crate::error::{ModelError, ModelResult};
use crate::weight::SafeTensorsFile;

// ---------------------------------------------------------------------------
// LoraAdapterConfig — parsed adapter_config.json
// ---------------------------------------------------------------------------

/// Parsed PEFT adapter_config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct LoraAdapterConfig {
    /// LoRA rank (dimension of the low-rank matrices).
    pub r: usize,
    /// Scaling factor: effective scaling = lora_alpha / r.
    pub lora_alpha: f64,
    /// Which modules to apply LoRA to (e.g. ["q_proj", "k_proj", "v_proj", "o_proj"]).
    pub target_modules: Vec<String>,
    /// Whether to use rsLoRA scaling (alpha / sqrt(r) instead of alpha / r).
    #[serde(default)]
    pub use_rslora: bool,
}

impl LoraAdapterConfig {
    /// Parse from an adapter_config.json file.
    pub fn from_file(path: impl AsRef<Path>) -> ModelResult<Self> {
        let data = std::fs::read_to_string(path.as_ref())?;
        let config: Self = serde_json::from_str(&data)
            .map_err(|e| ModelError::Other(format!("failed to parse adapter_config.json: {e}")))?;
        Ok(config)
    }

    /// Compute the LoRA scaling factor.
    pub fn scaling(&self) -> f64 {
        if self.use_rslora {
            self.lora_alpha / (self.r as f64).sqrt()
        } else {
            self.lora_alpha / self.r as f64
        }
    }
}

// ---------------------------------------------------------------------------
// LoraAdapter — loaded adapter weights
// ---------------------------------------------------------------------------

/// A loaded LoRA adapter: config + weight tensors.
pub struct LoraAdapter {
    /// Adapter name (for logging).
    pub name: String,
    /// Parsed adapter configuration.
    pub config: LoraAdapterConfig,
    /// Pre-computed scaling factor (alpha / rank or alpha / sqrt(rank)).
    pub scaling: f64,
    /// Weight pairs keyed by layer prefix.
    ///
    /// Key: layer prefix (e.g. "model.layers.0.self_attn.q_proj")
    /// Value: (lora_A [rank, in_features], lora_B [out_features, rank]) — raw from safetensors
    pub weights: HashMap<String, (Tensor, Tensor)>,
}

impl LoraAdapter {
    /// Load a LoRA adapter from a directory containing adapter_config.json
    /// and adapter_model.safetensors.
    pub fn from_dir(
        dir: impl AsRef<Path>,
        name: &str,
        device: &Device,
        dtype: DType,
    ) -> ModelResult<Self> {
        let dir = dir.as_ref();
        let config = LoraAdapterConfig::from_file(dir.join("adapter_config.json"))?;
        let scaling = config.scaling();

        let safetensors_path = dir.join("adapter_model.safetensors");
        if !safetensors_path.exists() {
            return Err(ModelError::Other(format!(
                "adapter_model.safetensors not found in {}",
                dir.display()
            )));
        }

        let file = SafeTensorsFile::open(&safetensors_path)?;
        let all_tensors = file.load_all(device)?;

        let mut weights: HashMap<String, (Tensor, Tensor)> = HashMap::new();

        // Group tensors by layer prefix.
        // PEFT names look like: base_model.model.{prefix}.lora_A.weight
        //                    or: base_model.model.{prefix}.lora_B.weight
        for (name, tensor) in &all_tensors {
            let (prefix, is_a) = if let Some(p) = name.strip_suffix(".lora_A.weight") {
                (p, true)
            } else if let Some(p) = name.strip_suffix(".lora_B.weight") {
                (p, false)
            } else {
                continue;
            };

            // Strip the "base_model.model." prefix that PEFT adds.
            let clean_prefix = prefix.strip_prefix("base_model.model.").unwrap_or(prefix);

            let tensor = tensor.to_dtype(dtype)?;

            let entry = weights.entry(clean_prefix.to_string()).or_insert_with(|| {
                // Placeholders — will be replaced.
                let z = Tensor::zeros(&[1], dtype, device).unwrap();
                (z.clone(), z)
            });

            if is_a {
                entry.0 = tensor;
            } else {
                entry.1 = tensor;
            }
        }

        // Remove entries where we only got one of A/B (shouldn't happen with valid adapters).
        weights.retain(|_, (a, b)| a.dims().len() == 2 && b.dims().len() == 2);

        tracing::debug!(
            "Loaded LoRA adapter '{}': rank={}, alpha={}, scaling={:.4}, {} target layers",
            name,
            config.r,
            config.lora_alpha,
            scaling,
            weights.len(),
        );

        Ok(Self {
            name: name.to_string(),
            config,
            scaling,
            weights,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_config_parse() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("adapter_config.json");
        std::fs::write(
            &config_path,
            r#"{
                "r": 16,
                "lora_alpha": 32.0,
                "target_modules": ["q_proj", "v_proj"],
                "use_rslora": false
            }"#,
        )
        .unwrap();

        let config = LoraAdapterConfig::from_file(&config_path).unwrap();
        assert_eq!(config.r, 16);
        assert!((config.lora_alpha - 32.0).abs() < 1e-6);
        assert_eq!(config.target_modules, vec!["q_proj", "v_proj"]);
        assert!(!config.use_rslora);
        assert!((config.scaling() - 2.0).abs() < 1e-6); // 32 / 16 = 2
    }

    #[test]
    fn test_adapter_config_rslora() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("adapter_config.json");
        std::fs::write(
            &config_path,
            r#"{
                "r": 16,
                "lora_alpha": 32.0,
                "target_modules": ["q_proj"],
                "use_rslora": true
            }"#,
        )
        .unwrap();

        let config = LoraAdapterConfig::from_file(&config_path).unwrap();
        assert!(config.use_rslora);
        // 32 / sqrt(16) = 32 / 4 = 8
        assert!((config.scaling() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_adapter_weight_load() {
        let dir = tempfile::tempdir().unwrap();

        // Write adapter_config.json.
        std::fs::write(
            dir.path().join("adapter_config.json"),
            r#"{"r": 4, "lora_alpha": 8.0, "target_modules": ["q_proj", "v_proj"]}"#,
        )
        .unwrap();

        // Create synthetic adapter_model.safetensors with LoRA A/B pairs.
        let safetensors_path = dir.path().join("adapter_model.safetensors");
        let rank = 4usize;
        let in_features = 8usize;
        let out_features = 8usize;

        // lora_A: [rank, in_features], lora_B: [out_features, rank]
        let a_data: Vec<u8> = vec![0.1f32; rank * in_features]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let b_data: Vec<u8> = vec![0.2f32; out_features * rank]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        crate::weight::tests_helper::create_safetensors_file(
            &safetensors_path,
            &[
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                    vec![rank, in_features],
                    DType::F32,
                    &a_data,
                ),
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                    vec![out_features, rank],
                    DType::F32,
                    &b_data,
                ),
                (
                    "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight",
                    vec![rank, in_features],
                    DType::F32,
                    &a_data,
                ),
                (
                    "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight",
                    vec![out_features, rank],
                    DType::F32,
                    &b_data,
                ),
            ],
        );

        let adapter =
            LoraAdapter::from_dir(dir.path(), "test_adapter", &Device::Cpu, DType::F32).unwrap();

        assert_eq!(adapter.name, "test_adapter");
        assert_eq!(adapter.config.r, 4);
        assert!((adapter.scaling - 2.0).abs() < 1e-6); // 8 / 4
        assert_eq!(adapter.weights.len(), 2);
        assert!(
            adapter
                .weights
                .contains_key("model.layers.0.self_attn.q_proj")
        );
        assert!(
            adapter
                .weights
                .contains_key("model.layers.0.self_attn.v_proj")
        );

        // Verify tensor shapes.
        let (a, b) = &adapter.weights["model.layers.0.self_attn.q_proj"];
        assert_eq!(a.dims(), &[rank, in_features]);
        assert_eq!(b.dims(), &[out_features, rank]);
    }

    #[test]
    fn test_weight_name_parsing() {
        // Verify that PEFT tensor names are parsed correctly.
        let name = "base_model.model.model.layers.5.self_attn.o_proj.lora_A.weight";
        let stripped = name.strip_suffix(".lora_A.weight").unwrap();
        let clean = stripped.strip_prefix("base_model.model.").unwrap();
        assert_eq!(clean, "model.layers.5.self_attn.o_proj");
    }
}
