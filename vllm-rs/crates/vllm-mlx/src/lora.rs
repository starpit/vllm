// SPDX-License-Identifier: Apache-2.0
//! MLX LoRA adapter loading and weight merging.
//!
//! For single-adapter serving, LoRA weights are merged into the base model
//! at load time: `W_merged = W + scaling * B @ A`. This avoids runtime
//! overhead — the merged model runs at the same speed as the base model.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::module::Param;
use mlx_rs::ops;
use mlx_rs::{Array, Dtype};
use vllm_model::lora::LoraAdapterConfig;

/// A loaded LoRA adapter with MLX arrays.
pub struct MlxLoraAdapter {
    /// Adapter name (for logging).
    pub name: String,
    /// Parsed adapter configuration.
    pub config: LoraAdapterConfig,
    /// Pre-computed scaling factor (alpha / rank or alpha / sqrt(rank)).
    pub scaling: f64,
    /// Weight pairs keyed by layer prefix.
    ///
    /// Key: layer prefix (e.g. "model.layers.0.self_attn.q_proj")
    /// Value: (lora_A [rank, in_features], lora_B [out_features, rank])
    pub weights: HashMap<String, (Array, Array)>,
}

impl MlxLoraAdapter {
    /// Load a LoRA adapter from a directory containing adapter_config.json
    /// and adapter_model.safetensors.
    pub fn from_dir(
        dir: &Path,
        name: &str,
        dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config = LoraAdapterConfig::from_file(dir.join("adapter_config.json"))?;
        let scaling = config.scaling();

        let safetensors_path = dir.join("adapter_model.safetensors");
        if !safetensors_path.exists() {
            return Err(format!("adapter_model.safetensors not found in {}", dir.display()).into());
        }

        let all_tensors = Array::load_safetensors(&safetensors_path)?;

        let mut weights: HashMap<String, (Array, Array)> = HashMap::new();

        for (tensor_name, tensor) in &all_tensors {
            let (prefix, is_a) = if let Some(p) = tensor_name.strip_suffix(".lora_A.weight") {
                (p, true)
            } else if let Some(p) = tensor_name.strip_suffix(".lora_B.weight") {
                (p, false)
            } else {
                continue;
            };

            // Strip PEFT prefix.
            let clean_prefix = prefix.strip_prefix("base_model.model.").unwrap_or(prefix);

            let tensor = tensor.as_dtype(dtype)?;

            let entry = weights.entry(clean_prefix.to_string()).or_insert_with(|| {
                let z = Array::zeros::<f32>(&[1]).unwrap();
                (z.clone(), z)
            });

            if is_a {
                entry.0 = tensor;
            } else {
                entry.1 = tensor;
            }
        }

        // Remove incomplete entries.
        weights.retain(|_, (a, b)| a.ndim() == 2 && b.ndim() == 2);

        tracing::debug!(
            "Loaded MLX LoRA adapter '{}': rank={}, alpha={}, scaling={:.4}, {} target layers",
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

/// Merge a LoRA A/B pair into a base weight in-place.
///
/// Computes: `W_merged = W + scaling * B @ A`
/// where W is `[out, in]`, A is `[rank, in]`, B is `[out, rank]`.
pub fn merge_lora_into_weight(
    weight: &mut Param<Array>,
    lora_a: &Array,
    lora_b: &Array,
    scaling: f64,
) -> Result<(), mlx_rs::error::Exception> {
    // B @ A → [out, in]
    let delta = ops::matmul(lora_b, lora_a)?;
    // Scale the delta.
    let scale_arr = Array::from_f32(scaling as f32);
    let scaled_delta = ops::multiply(&delta, &scale_arr)?;
    // Merge: W + scaled_delta
    let merged = ops::add(&weight.value, &scaled_delta)?;
    weight.value = merged;
    Ok(())
}
