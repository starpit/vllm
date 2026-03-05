// SPDX-License-Identifier: Apache-2.0
//! Granite (IBM) model architecture.
//!
//! Architecturally identical to LLaMA with 4 scalar multipliers:
//! - `embedding_multiplier` — scales embeddings after lookup
//! - `residual_multiplier` — scales attention/MLP outputs before residual add
//! - `attention_multiplier` — replaces the standard 1/sqrt(head_dim) scaling
//! - `logits_scaling` — divides logits before softmax
//!
//! Port of: `vllm/model_executor/models/granite.py`

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::error::{ModelError, ModelResult};
use vllm_model::layers::{Embedding, Linear, RmsNorm};
use vllm_model::weight::{HfModelConfig, ModelWeights};

use crate::llama::{LlamaAttention, LlamaConfig, LlamaMLP};

// ---------------------------------------------------------------------------
// GraniteConfig
// ---------------------------------------------------------------------------

/// Parsed configuration for a Granite model.
#[derive(Debug, Clone)]
pub struct GraniteConfig {
    pub llama: LlamaConfig,
    pub embedding_multiplier: f64,
    pub residual_multiplier: f64,
    pub attention_multiplier: f64,
    pub logits_scaling: f64,
}

impl GraniteConfig {
    /// Parse from a HuggingFace config.json.
    pub fn from_hf_config(config: &HfModelConfig) -> ModelResult<Self> {
        let llama = LlamaConfig::from_hf_config(config)?;

        let embedding_multiplier = config
            .extra
            .get("embedding_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let residual_multiplier = config
            .extra
            .get("residual_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let attention_multiplier = config
            .extra
            .get("attention_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0 / (llama.head_dim as f64).sqrt());
        let logits_scaling = config
            .extra
            .get("logits_scaling")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        Ok(Self {
            llama,
            embedding_multiplier,
            residual_multiplier,
            attention_multiplier,
            logits_scaling,
        })
    }
}

// ---------------------------------------------------------------------------
// GraniteDecoderLayer
// ---------------------------------------------------------------------------

/// A single Granite decoder layer.
///
/// Same sub-components as LLaMA, but scales attention/MLP outputs by
/// `residual_multiplier` before the residual add.
struct GraniteDecoderLayer {
    self_attn: LlamaAttention,
    mlp: LlamaMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    residual_multiplier: f64,
}

impl GraniteDecoderLayer {
    /// Load a decoder layer.
    #[allow(clippy::too_many_arguments)]
    fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        config: &GraniteConfig,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
        layer_idx: usize,
    ) -> ModelResult<Self> {
        let mut self_attn = LlamaAttention::load(
            weights,
            &format!("{}.self_attn", prefix),
            &config.llama,
            dtype,
            device,
            rank,
            world_size,
            layer_idx,
        )?;
        // Override attention scaling with Granite's multiplier.
        self_attn.scale = config.attention_multiplier;

        let mlp = LlamaMLP::load(weights, &format!("{}.mlp", prefix), dtype, rank, world_size)?;
        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{}.input_layernorm", prefix),
            config.llama.rms_norm_eps,
            dtype,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{}.post_attention_layernorm", prefix),
            config.llama.rms_norm_eps,
            dtype,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            residual_multiplier: config.residual_multiplier,
        })
    }

    /// Forward pass.
    fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<Tensor> {
        // Pre-attention norm.
        let normed = crate::ops::rms_norm(hidden_states, &self.input_layernorm)
            .map_err(ModelError::Candle)?;

        // Attention.
        let attn_output = self.self_attn.forward(&normed, positions, kv_cache)?;

        // Residual with scaled attention output.
        let hidden_states = (hidden_states
            + attn_output
                .affine(self.residual_multiplier, 0.0)
                .map_err(ModelError::Candle)?)
        .map_err(ModelError::Candle)?;

        // Post-attention norm.
        let normed = crate::ops::rms_norm(&hidden_states, &self.post_attention_layernorm)
            .map_err(ModelError::Candle)?;

        // MLP with scaled output + residual.
        let mlp_output = self.mlp.forward(&normed).map_err(ModelError::Candle)?;
        let hidden_states = (hidden_states
            + mlp_output
                .affine(self.residual_multiplier, 0.0)
                .map_err(ModelError::Candle)?)
        .map_err(ModelError::Candle)?;

        Ok(hidden_states)
    }
}

// ---------------------------------------------------------------------------
// GraniteModel
// ---------------------------------------------------------------------------

/// Granite transformer backbone.
///
/// Like LLaMA but scales embeddings by `embedding_multiplier`.
struct GraniteModel {
    embed_tokens: Embedding,
    layers: Vec<GraniteDecoderLayer>,
    norm: RmsNorm,
    embedding_multiplier: f64,
}

impl GraniteModel {
    /// Load the model backbone.
    fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        config: &GraniteConfig,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let embed_tokens = Embedding::load(weights, &format!("{}.embed_tokens", prefix), dtype)?;

        let mut layers = Vec::with_capacity(config.llama.num_hidden_layers);
        for i in 0..config.llama.num_hidden_layers {
            let layer = GraniteDecoderLayer::load(
                weights,
                &format!("{}.layers.{}", prefix, i),
                config,
                dtype,
                device,
                rank,
                world_size,
                i,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(
            weights,
            &format!("{}.norm", prefix),
            config.llama.rms_norm_eps,
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            embedding_multiplier: config.embedding_multiplier,
        })
    }

    /// Forward pass.
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        mut kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let mut hidden_states = self
            .embed_tokens
            .forward(input_ids)
            .map_err(ModelError::Candle)?;

        // Scale embeddings.
        hidden_states = hidden_states
            .affine(self.embedding_multiplier, 0.0)
            .map_err(ModelError::Candle)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_handle = kv_cache.as_mut().map(|s| s.layer_handle(i));
            hidden_states = layer.forward(&hidden_states, positions, layer_handle)?;
        }

        crate::ops::rms_norm(&hidden_states, &self.norm).map_err(ModelError::Candle)
    }

    /// Run the transformer backbone on pre-computed embeddings.
    fn backbone(
        &self,
        mut hidden_states: Tensor,
        positions: &Tensor,
        mut kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        // Scale embeddings (caller already looked up embed_tokens).
        hidden_states = hidden_states
            .affine(self.embedding_multiplier, 0.0)
            .map_err(ModelError::Candle)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_handle = kv_cache.as_mut().map(|s| s.layer_handle(i));
            hidden_states = layer.forward(&hidden_states, positions, layer_handle)?;
        }

        crate::ops::rms_norm(&hidden_states, &self.norm).map_err(ModelError::Candle)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// GraniteForCausalLM
// ---------------------------------------------------------------------------

/// Granite for causal language modeling.
///
/// Wraps `GraniteModel` with a language model head and `logits_scaling`.
pub struct GraniteForCausalLM {
    model: GraniteModel,
    lm_head: Linear,
    logits_scaling: f64,
}

impl GraniteForCausalLM {
    /// Load the full model from weights.
    pub fn load(
        weights: &mut ModelWeights,
        config: &GraniteConfig,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let model = GraniteModel::load(weights, "model", config, dtype, device, rank, world_size)?;

        let lm_head = if config.llama.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight().clone(), None)
        } else {
            Linear::load(weights, "lm_head", dtype)?
        };

        Ok(Self {
            model,
            lm_head,
            logits_scaling: config.logits_scaling,
        })
    }
}

impl crate::Model for GraniteForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let hidden_states = self.model.forward(input_ids, positions, kv_cache)?;
        let logits = self
            .lm_head
            .forward(&hidden_states)
            .map_err(ModelError::Candle)?;
        // Divide by logits_scaling then cast to f32 for sampling.
        let logits = logits
            .affine(1.0 / self.logits_scaling, 0.0)
            .map_err(ModelError::Candle)?;
        logits.to_dtype(DType::F32).map_err(ModelError::Candle)
    }

    fn forward_embeds(
        &self,
        inputs_embeds: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let hidden_states = self
            .model
            .backbone(inputs_embeds.clone(), positions, kv_cache)?;
        let logits = self
            .lm_head
            .forward(&hidden_states)
            .map_err(ModelError::Candle)?;
        let logits = logits
            .affine(1.0 / self.logits_scaling, 0.0)
            .map_err(ModelError::Candle)?;
        logits.to_dtype(DType::F32).map_err(ModelError::Candle)
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn hidden_states(&self, input_ids: &Tensor, positions: &Tensor) -> ModelResult<Tensor> {
        self.model.forward(input_ids, positions, None)
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Factory function for the model registry.
pub fn create_granite(
    weights: &mut ModelWeights,
    config: &HfModelConfig,
    dtype: DType,
    device: &Device,
    rank: usize,
    world_size: usize,
) -> ModelResult<Box<dyn crate::Model>> {
    let granite_config = GraniteConfig::from_hf_config(config)?;
    let model =
        GraniteForCausalLM::load(weights, &granite_config, dtype, device, rank, world_size)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_granite_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["GraniteForCausalLM"],
                "model_type": "granite",
                "hidden_size": 2048,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 40,
                "intermediate_size": 8192,
                "vocab_size": 49152,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "tie_word_embeddings": true,
                "embedding_multiplier": 12.0,
                "residual_multiplier": 0.22,
                "attention_multiplier": 0.0625,
                "logits_scaling": 13.0
            }"#,
        )
        .unwrap();

        let config = GraniteConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.llama.hidden_size, 2048);
        assert_eq!(config.llama.num_attention_heads, 32);
        assert_eq!(config.llama.num_kv_heads, 8);
        assert_eq!(config.llama.num_hidden_layers, 40);
        assert!(config.llama.tie_word_embeddings);
        assert!((config.embedding_multiplier - 12.0).abs() < 1e-6);
        assert!((config.residual_multiplier - 0.22).abs() < 1e-6);
        assert!((config.attention_multiplier - 0.0625).abs() < 1e-6);
        assert!((config.logits_scaling - 13.0).abs() < 1e-6);
    }

    #[test]
    fn test_granite_config_defaults() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["GraniteForCausalLM"],
                "model_type": "granite",
                "hidden_size": 2048,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 40,
                "intermediate_size": 8192,
                "vocab_size": 49152
            }"#,
        )
        .unwrap();

        let config = GraniteConfig::from_hf_config(&hf_config).unwrap();
        // Defaults when not specified.
        assert!((config.embedding_multiplier - 1.0).abs() < 1e-6);
        assert!((config.residual_multiplier - 1.0).abs() < 1e-6);
        // attention_multiplier defaults to 1/sqrt(head_dim) = 1/sqrt(64) = 0.125
        let expected_attn = 1.0 / (config.llama.head_dim as f64).sqrt();
        assert!((config.attention_multiplier - expected_attn).abs() < 1e-6);
        assert!((config.logits_scaling - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_granite_registry() {
        let registry = crate::ModelRegistry::default_registry();
        assert!(registry.contains("GraniteForCausalLM"));
    }
}
