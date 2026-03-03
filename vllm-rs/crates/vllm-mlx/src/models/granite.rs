// SPDX-License-Identifier: Apache-2.0
//! Granite (IBM) model architecture for MLX.
//!
//! Reuses quantized LLaMA MLX components with 4 scalar multipliers:
//! - `embedding_multiplier` — scales embeddings after lookup
//! - `residual_multiplier` — scales attention/MLP outputs before residual add
//! - `attention_multiplier` — replaces the standard 1/sqrt(head_dim) scaling
//! - `logits_scaling` — divides logits before softmax

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::{Array, Dtype};

use crate::cache::MlxKvCache;
use crate::models::llama::{LlamaConfig, load_safetensors_weights};
use crate::models::quantized_llama::{
    MlxEmbedTokens, MlxLmHead, MlxQuantizedLlamaAttention, MlxQuantizedLlamaMLP, QuantConfig,
    assign_weight,
};
use vllm_model::weight::HfModelConfig;

// ---------------------------------------------------------------------------
// GraniteConfig
// ---------------------------------------------------------------------------

/// Parsed configuration for a Granite model (MLX).
#[derive(Debug, Clone)]
struct GraniteConfig {
    llama: LlamaConfig,
    embedding_multiplier: f32,
    residual_multiplier: f32,
    attention_multiplier: f32,
    logits_scaling: f32,
}

impl GraniteConfig {
    fn from_hf_config(config: &HfModelConfig) -> Result<Self, String> {
        let llama = LlamaConfig::from_hf_config(config)?;

        let embedding_multiplier = config
            .extra
            .get("embedding_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;
        let residual_multiplier = config
            .extra
            .get("residual_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;
        let attention_multiplier = config
            .extra
            .get("attention_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0 / (llama.head_dim as f64).sqrt())
            as f32;
        let logits_scaling = config
            .extra
            .get("logits_scaling")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

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
// MlxQuantizedGraniteDecoderLayer
// ---------------------------------------------------------------------------

/// A single quantized Granite decoder layer.
struct MlxQuantizedGraniteDecoderLayer {
    self_attn: MlxQuantizedLlamaAttention,
    mlp: MlxQuantizedLlamaMLP,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
    residual_multiplier: f32,
}

impl MlxQuantizedGraniteDecoderLayer {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &GraniteConfig,
        qc: &QuantConfig,
    ) -> Result<Self, Exception> {
        let mut self_attn = MlxQuantizedLlamaAttention::from_weights(
            weights,
            &format!("{prefix}.self_attn"),
            &config.llama,
            qc,
        );
        self_attn.scale = config.attention_multiplier;

        let mlp = MlxQuantizedLlamaMLP::from_weights(weights, &format!("{prefix}.mlp"), qc);

        let mut input_layernorm = nn::RmsNormBuilder::new(config.llama.hidden_size as i32)
            .eps(config.llama.rms_norm_eps)
            .build()?;
        let mut post_attention_layernorm = nn::RmsNormBuilder::new(config.llama.hidden_size as i32)
            .eps(config.llama.rms_norm_eps)
            .build()?;
        assign_weight(
            &mut input_layernorm.weight,
            weights,
            &format!("{prefix}.input_layernorm.weight"),
        );
        assign_weight(
            &mut post_attention_layernorm.weight,
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        );

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            residual_multiplier: config.residual_multiplier,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        // Pre-attention norm.
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed, positions, cache)?;
        // Scale attention output before residual add.
        let hidden_states =
            hidden_states.add(&attn_output.multiply(Array::from(self.residual_multiplier))?)?;

        // Post-attention norm.
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        // Scale MLP output before residual add.
        hidden_states.add(&mlp_output.multiply(Array::from(self.residual_multiplier))?)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedGraniteForCausalLM
// ---------------------------------------------------------------------------

/// Quantized Granite for causal language modeling using MLX.
struct MlxQuantizedGraniteForCausalLM {
    embed_tokens: MlxEmbedTokens,
    layers: Vec<MlxQuantizedGraniteDecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: Option<MlxLmHead>,
    tie_word_embeddings: bool,
    embedding_multiplier: f32,
    logits_scaling: f32,
}

impl MlxQuantizedGraniteForCausalLM {
    fn load(
        model_dir: &Path,
        config: &GraniteConfig,
        qc: &QuantConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let weights = load_safetensors_weights(model_dir)?;

        let embed_tokens =
            MlxEmbedTokens::from_weights(&weights, "model.embed_tokens", qc.group_size, qc.bits);

        let mut layers = Vec::with_capacity(config.llama.num_hidden_layers);
        for i in 0..config.llama.num_hidden_layers {
            layers.push(MlxQuantizedGraniteDecoderLayer::from_weights(
                &weights,
                &format!("model.layers.{i}"),
                config,
                qc,
            )?);
        }

        let mut norm = nn::RmsNormBuilder::new(config.llama.hidden_size as i32)
            .eps(config.llama.rms_norm_eps)
            .build()?;
        assign_weight(&mut norm.weight, &weights, "model.norm.weight");

        let lm_head = if config.llama.tie_word_embeddings {
            None
        } else {
            Some(MlxLmHead::from_weights(
                &weights,
                "lm_head",
                qc.group_size,
                qc.bits,
            ))
        };

        mlx_rs::transforms::eval(weights.values())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            tie_word_embeddings: config.llama.tie_word_embeddings,
            embedding_multiplier: config.embedding_multiplier,
            logits_scaling: config.logits_scaling,
        })
    }
}

impl super::MlxModel for MlxQuantizedGraniteForCausalLM {
    fn forward(
        &mut self,
        input_ids: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
    ) -> mlx_rs::error::Result<Array> {
        // Embed + scale.
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = hidden_states.multiply(Array::from(self.embedding_multiplier))?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i])?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        let logits = if self.tie_word_embeddings {
            self.embed_tokens.as_linear(&hidden_states)?
        } else {
            self.lm_head.as_mut().unwrap().forward(&hidden_states)?
        };

        // Divide by logits_scaling, cast to f32.
        let logits = logits.multiply(Array::from(1.0f32 / self.logits_scaling))?;
        logits.as_dtype(Dtype::Float32)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut kv_cache: MlxKvCache = (0..self.layers.len()).map(|_| None).collect();
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = hidden_states.multiply(Array::from(self.embedding_multiplier))?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i])?;
        }
        self.norm.forward(&hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Factory function for creating a quantized MLX Granite model.
pub fn create_mlx_quantized_granite(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let granite_config = GraniteConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();
    tracing::info!(
        "Loading quantized MLX Granite (group_size={}, bits={})",
        qc.group_size,
        qc.bits
    );

    let model = MlxQuantizedGraniteForCausalLM::load(model_dir, &granite_config, &qc, dtype)?;
    Ok(Box::new(model))
}
