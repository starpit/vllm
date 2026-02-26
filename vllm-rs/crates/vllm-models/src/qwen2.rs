// SPDX-License-Identifier: Apache-2.0
//! Qwen2 model architecture.
//!
//! Qwen2 is architecturally identical to LLaMA — same components
//! (RMSNorm, SiLU-gated MLP, RoPE, GQA) and same HuggingFace weight
//! naming convention. The only notable differences are:
//!
//! - QKV projections include bias terms (our layer loading handles this
//!   automatically since `ColumnParallelLinear::load` checks for `.bias`).
//! - Default `rope_theta` is 1,000,000 (vs LLaMA's 10,000), though this
//!   is always specified explicitly in `config.json`.
//!
//! Because of this, Qwen2 reuses `LlamaForCausalLM` internally.
//!
//! Port of: `vllm/model_executor/models/qwen2.py`

use candle_core::{DType, Device};

use vllm_model::error::ModelResult;
use vllm_model::weight::{HfModelConfig, ModelWeights};

use crate::llama::{LlamaConfig, LlamaForCausalLM};

// ---------------------------------------------------------------------------
// Qwen2Config
// ---------------------------------------------------------------------------

/// Parsed configuration for a Qwen2 model.
///
/// Identical fields to `LlamaConfig`, with Qwen2-specific defaults
/// (e.g., `rope_theta = 1_000_000`).
#[derive(Debug, Clone)]
pub struct Qwen2Config(pub LlamaConfig);

impl Qwen2Config {
    /// Parse from a HuggingFace config.json, using Qwen2 defaults.
    pub fn from_hf_config(config: &HfModelConfig) -> ModelResult<Self> {
        let mut llama_config = LlamaConfig::from_hf_config(config)?;
        // Qwen2 defaults to rope_theta = 1M if not specified.
        if config.rope_theta.is_none() {
            llama_config.rope_theta = 1_000_000.0;
        }
        Ok(Self(llama_config))
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Factory function for the model registry.
pub fn create_qwen2(
    weights: &ModelWeights,
    config: &HfModelConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let qwen2_config = Qwen2Config::from_hf_config(config)?;
    let model = LlamaForCausalLM::load(weights, &qwen2_config.0, dtype, device, 0, 1)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    fn test_hf_config() -> HfModelConfig {
        serde_json::from_str(
            r#"{
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "vocab_size": 100,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "tie_word_embeddings": false
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn test_qwen2_config_from_hf() {
        let hf_config = test_hf_config();
        let config = Qwen2Config::from_hf_config(&hf_config).unwrap();

        assert_eq!(config.0.hidden_size, 32);
        assert_eq!(config.0.num_attention_heads, 4);
        assert_eq!(config.0.num_kv_heads, 2);
        assert_eq!(config.0.num_hidden_layers, 2);
        assert_eq!(config.0.intermediate_size, 64);
        assert_eq!(config.0.vocab_size, 100);
        assert!((config.0.rope_theta - 1_000_000.0).abs() < 1.0);
        assert!(!config.0.tie_word_embeddings);
    }

    #[test]
    fn test_qwen2_config_default_rope_theta() {
        // When rope_theta is not specified, Qwen2 defaults to 1M.
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "vocab_size": 100
            }"#,
        )
        .unwrap();

        let config = Qwen2Config::from_hf_config(&hf_config).unwrap();
        assert!((config.0.rope_theta - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_qwen2_model_forward() {
        let hf_config = test_hf_config();
        let config = Qwen2Config::from_hf_config(&hf_config).unwrap();
        let c = &config.0;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let device = Device::Cpu;

        let mut tensor_specs: Vec<(String, Vec<usize>)> = Vec::new();
        tensor_specs.push((
            "model.embed_tokens.weight".to_string(),
            vec![c.vocab_size, c.hidden_size],
        ));

        for i in 0..c.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            let q_size = c.num_attention_heads * c.head_dim;
            let kv_size = c.num_kv_heads * c.head_dim;

            // Qwen2 has bias on Q/K/V projections.
            tensor_specs.push((
                format!("{}.self_attn.q_proj.weight", prefix),
                vec![q_size, c.hidden_size],
            ));
            tensor_specs.push((format!("{}.self_attn.q_proj.bias", prefix), vec![q_size]));
            tensor_specs.push((
                format!("{}.self_attn.k_proj.weight", prefix),
                vec![kv_size, c.hidden_size],
            ));
            tensor_specs.push((format!("{}.self_attn.k_proj.bias", prefix), vec![kv_size]));
            tensor_specs.push((
                format!("{}.self_attn.v_proj.weight", prefix),
                vec![kv_size, c.hidden_size],
            ));
            tensor_specs.push((format!("{}.self_attn.v_proj.bias", prefix), vec![kv_size]));
            tensor_specs.push((
                format!("{}.self_attn.o_proj.weight", prefix),
                vec![c.hidden_size, q_size],
            ));

            tensor_specs.push((
                format!("{}.mlp.gate_proj.weight", prefix),
                vec![c.intermediate_size, c.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.mlp.up_proj.weight", prefix),
                vec![c.intermediate_size, c.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.mlp.down_proj.weight", prefix),
                vec![c.hidden_size, c.intermediate_size],
            ));

            tensor_specs.push((
                format!("{}.input_layernorm.weight", prefix),
                vec![c.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.post_attention_layernorm.weight", prefix),
                vec![c.hidden_size],
            ));
        }

        tensor_specs.push(("model.norm.weight".to_string(), vec![c.hidden_size]));
        tensor_specs.push((
            "lm_head.weight".to_string(),
            vec![c.vocab_size, c.hidden_size],
        ));

        create_test_weights(&path, &tensor_specs);

        let weights = ModelWeights::from_single_file(&path, &device).unwrap();
        let model = create_qwen2(&weights, &hf_config, DType::F32, &device).unwrap();

        let input_ids = Tensor::new(&[1u32, 5, 10], &device).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &device).unwrap();

        let logits = model.forward(&input_ids, &positions, None).unwrap();
        assert_eq!(logits.dims(), &[3, c.vocab_size]);
    }

    #[test]
    fn test_qwen2_registry() {
        let registry = crate::ModelRegistry::default_registry();
        assert!(registry.contains("Qwen2ForCausalLM"));
    }

    // -----------------------------------------------------------------------
    // Test helper
    // -----------------------------------------------------------------------

    fn create_test_weights(path: &std::path::Path, specs: &[(String, Vec<usize>)]) {
        use safetensors::tensor::TensorView;

        let mut all_data: Vec<Vec<u8>> = Vec::new();
        for (_, shape) in specs {
            let num_elements: usize = shape.iter().product();
            let data: Vec<u8> = (0..num_elements)
                .flat_map(|_| 0.01f32.to_le_bytes())
                .collect();
            all_data.push(data);
        }

        // Override norm weights with 1.0.
        for (i, (name, shape)) in specs.iter().enumerate() {
            if name.contains("layernorm") || (name.as_str() == "model.norm.weight") {
                let num_elements: usize = shape.iter().product();
                all_data[i] = (0..num_elements)
                    .flat_map(|_| 1.0f32.to_le_bytes())
                    .collect();
            }
        }

        let views: Vec<(&str, TensorView<'_>)> = specs
            .iter()
            .zip(all_data.iter())
            .map(|((name, shape), data)| {
                (
                    name.as_str(),
                    TensorView::new(safetensors::Dtype::F32, shape.clone(), data).unwrap(),
                )
            })
            .collect();

        let bytes = safetensors::tensor::serialize(views, None).unwrap();
        std::fs::write(path, bytes).unwrap();
    }
}
