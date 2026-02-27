// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Command R (CohereForCausalLM) model architecture for MLX.
//!
//! Key differences from LLaMA:
//! - LayerNorm (with mean subtraction, weight only, no bias) instead of RMSNorm
//! - Parallel attention + MLP: one norm per layer, both branches read same normed input
//! - Logit scaling: `logits *= logit_scale` (e.g. 0.0625 = 1/16)
//! - Interleaved RoPE: `nn::Rope` with `traditional = true`
//! - Optional QK norm (LayerNorm on Q/K after projection, before RoPE)
//!
//! Also provides a quantized variant using `nn::QuantizedLinear` / `nn::QuantizedEmbedding`.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

use crate::cache::MlxKvCache;
use crate::models::llama::{LlamaConfig, assign_weight, load_safetensors_weights};
use crate::models::quantized_llama::{
    MlxEmbedTokens, MlxLmHead, MlxQuantizedLlamaMLP, QuantConfig, make_quantized_linear,
};
use vllm_model::weight::HfModelConfig;

// ---------------------------------------------------------------------------
// CommandRConfig (extends LlamaConfig with Cohere-specific fields)
// ---------------------------------------------------------------------------

/// Parsed configuration for a Command R model.
#[derive(Debug, Clone)]
pub struct CommandRConfig {
    pub base: LlamaConfig,
    pub logit_scale: f32,
    pub use_qk_norm: bool,
}

impl CommandRConfig {
    /// Parse from a HuggingFace config.json.
    pub fn from_hf_config(config: &HfModelConfig) -> Result<Self, String> {
        let mut base = LlamaConfig::from_hf_config(config)?;
        // Command R defaults.
        if config.rope_theta.is_none() {
            base.rope_theta = 8000000.0;
        }
        if config.tie_word_embeddings.is_none() {
            base.tie_word_embeddings = true;
        }

        let logit_scale = config
            .extra
            .get("logit_scale")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

        let use_qk_norm = config
            .extra
            .get("use_qk_norm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(Self {
            base,
            logit_scale,
            use_qk_norm,
        })
    }
}

// ---------------------------------------------------------------------------
// Helper: create a LayerNorm without bias
// ---------------------------------------------------------------------------

/// Create a `nn::LayerNorm` with weight but no bias (Cohere convention).
fn cohere_layer_norm(hidden_size: i32, eps: f32) -> Result<nn::LayerNorm, Exception> {
    let mut norm = nn::LayerNormBuilder::new(hidden_size).eps(eps).build()?;
    // Remove bias — CohereLayerNorm has weight only.
    norm.bias.value = None;
    Ok(norm)
}

// ---------------------------------------------------------------------------
// MlxCommandRMLP
// ---------------------------------------------------------------------------

/// Command R MLP (SiLU-gated feed-forward network) using MLX.
/// Identical to LLaMA MLP.
struct MlxCommandRMLP {
    gate_proj: nn::Linear,
    up_proj: nn::Linear,
    down_proj: nn::Linear,
}

impl MlxCommandRMLP {
    fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(false)
                .build()?,
            up_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(false)
                .build()?,
            down_proj: nn::LinearBuilder::new(intermediate_size, hidden_size)
                .bias(false)
                .build()?,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(
            &mut self.gate_proj.weight,
            weights,
            &format!("{prefix}.gate_proj.weight"),
        );
        assign_weight(
            &mut self.up_proj.weight,
            weights,
            &format!("{prefix}.up_proj.weight"),
        );
        assign_weight(
            &mut self.down_proj.weight,
            weights,
            &format!("{prefix}.down_proj.weight"),
        );
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let gate = nn::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = gate.multiply(&up)?;
        self.down_proj.forward(&hidden)
    }
}

// ---------------------------------------------------------------------------
// MlxCommandRAttention
// ---------------------------------------------------------------------------

/// Command R attention with interleaved RoPE and optional QK norm.
struct MlxCommandRAttention {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    o_proj: nn::Linear,
    /// Optional QK norms (LayerNorm without bias).
    q_norm: Option<nn::LayerNorm>,
    k_norm: Option<nn::LayerNorm>,
    rope: nn::Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl MlxCommandRAttention {
    fn new(config: &CommandRConfig) -> Result<Self, Exception> {
        let c = &config.base;
        let hidden = c.hidden_size as i32;
        let q_size = (c.num_attention_heads * c.head_dim) as i32;
        let kv_size = (c.num_kv_heads * c.head_dim) as i32;

        let (q_norm, k_norm) = if config.use_qk_norm {
            (
                Some(cohere_layer_norm(c.head_dim as i32, c.rms_norm_eps)?),
                Some(cohere_layer_norm(c.head_dim as i32, c.rms_norm_eps)?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj: nn::LinearBuilder::new(hidden, q_size).bias(false).build()?,
            k_proj: nn::LinearBuilder::new(hidden, kv_size)
                .bias(false)
                .build()?,
            v_proj: nn::LinearBuilder::new(hidden, kv_size)
                .bias(false)
                .build()?,
            o_proj: nn::LinearBuilder::new(q_size, hidden).bias(false).build()?,
            q_norm,
            k_norm,
            rope: {
                let mut r = nn::Rope::new(c.head_dim as i32);
                r.base = c.rope_theta;
                r.traditional = true; // Interleaved RoPE (Cohere convention)
                r
            },
            num_heads: c.num_attention_heads,
            num_kv_heads: c.num_kv_heads,
            head_dim: c.head_dim,
            scale: 1.0 / (c.head_dim as f32).sqrt(),
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(
            &mut self.q_proj.weight,
            weights,
            &format!("{prefix}.q_proj.weight"),
        );
        assign_weight(
            &mut self.k_proj.weight,
            weights,
            &format!("{prefix}.k_proj.weight"),
        );
        assign_weight(
            &mut self.v_proj.weight,
            weights,
            &format!("{prefix}.v_proj.weight"),
        );
        assign_weight(
            &mut self.o_proj.weight,
            weights,
            &format!("{prefix}.o_proj.weight"),
        );

        // Optional QK norms.
        if let Some(ref mut norm) = self.q_norm
            && let Some(w) = weights.get(&format!("{prefix}.q_norm.weight"))
        {
            norm.weight.value = Some(w.clone());
        }
        if let Some(ref mut norm) = self.k_norm
            && let Some(w) = weights.get(&format!("{prefix}.k_norm.weight"))
        {
            norm.weight.value = Some(w.clone());
        }
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape: [seq, hidden] -> [seq, heads, head_dim]
        let mut q = q.reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?;
        let mut k = k.reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?;

        // Optional QK norms (applied per-head before RoPE).
        if let Some(ref mut norm) = self.q_norm {
            q = norm.forward(&q)?;
        }
        if let Some(ref mut norm) = self.k_norm {
            k = norm.forward(&k)?;
        }

        // [seq, heads, head_dim] -> [1, heads, seq, head_dim]
        let q = q.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut v = v
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        // RoPE (interleaved via traditional=true).
        let offset = if positions.size() > 0 {
            positions.reshape(&[-1])?.min(None)?.item::<i32>()
        } else {
            0
        };
        let q = self.rope.forward((&q, offset))?;
        k = self.rope.forward((&k, offset))?;

        // KV cache update.
        if let Some((ck, cv)) = cache.take() {
            k = concatenate_axis(&[ck, k], 2)?;
            v = concatenate_axis(&[cv, v], 2)?;
        }
        *cache = Some((k.clone(), v.clone()));

        // Fused SDPA.
        let mask = if seq_len > 1 {
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let out = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;

        // out: [1, heads, seq, head_dim] -> [seq, hidden]
        let hidden = (self.num_heads * self.head_dim) as i32;
        let out = out
            .squeeze_axes(&[0])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[seq_len, hidden])?;

        self.o_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// MlxCommandRDecoderLayer
// ---------------------------------------------------------------------------

/// Command R decoder layer: parallel attention + MLP, one LayerNorm.
struct MlxCommandRDecoderLayer {
    self_attn: MlxCommandRAttention,
    mlp: MlxCommandRMLP,
    input_layernorm: nn::LayerNorm,
}

impl MlxCommandRDecoderLayer {
    fn new(config: &CommandRConfig) -> Result<Self, Exception> {
        let c = &config.base;
        Ok(Self {
            self_attn: MlxCommandRAttention::new(config)?,
            mlp: MlxCommandRMLP::new(c.hidden_size as i32, c.intermediate_size as i32)?,
            input_layernorm: cohere_layer_norm(c.hidden_size as i32, c.rms_norm_eps)?,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        self.self_attn
            .load_weights(weights, &format!("{prefix}.self_attn"));
        self.mlp.load_weights(weights, &format!("{prefix}.mlp"));
        if let Some(w) = weights.get(&format!("{prefix}.input_layernorm.weight")) {
            self.input_layernorm.weight.value = Some(w.clone());
        }
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        let residual = hidden_states;

        // Single norm (shared by attn and mlp).
        let normed = self.input_layernorm.forward(hidden_states)?;

        // Parallel attention + MLP.
        let attn_output = self.self_attn.forward(&normed, positions, cache)?;
        let mlp_output = self.mlp.forward(&normed)?;

        // residual + attn_output + mlp_output
        residual.add(&attn_output)?.add(&mlp_output)
    }
}

// ---------------------------------------------------------------------------
// MlxCommandRForCausalLM (float)
// ---------------------------------------------------------------------------

/// Command R for causal language modeling using MLX.
pub struct MlxCommandRForCausalLM {
    embed_tokens: nn::Embedding,
    layers: Vec<MlxCommandRDecoderLayer>,
    norm: nn::LayerNorm,
    lm_head: Option<nn::Linear>,
    tie_word_embeddings: bool,
    logit_scale: f32,
    #[allow(dead_code)]
    config: CommandRConfig,
}

impl MlxCommandRForCausalLM {
    fn new(config: &CommandRConfig) -> Result<Self, Exception> {
        let c = &config.base;
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for _ in 0..c.num_hidden_layers {
            layers.push(MlxCommandRDecoderLayer::new(config)?);
        }

        let lm_head = if c.tie_word_embeddings {
            None
        } else {
            Some(
                nn::LinearBuilder::new(c.hidden_size as i32, c.vocab_size as i32)
                    .bias(false)
                    .build()?,
            )
        };

        Ok(Self {
            embed_tokens: nn::Embedding::new(c.vocab_size as i32, c.hidden_size as i32)?,
            layers,
            norm: cohere_layer_norm(c.hidden_size as i32, c.rms_norm_eps)?,
            lm_head,
            tie_word_embeddings: c.tie_word_embeddings,
            logit_scale: config.logit_scale,
            config: config.clone(),
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>) {
        assign_weight(
            &mut self.embed_tokens.weight,
            weights,
            "model.embed_tokens.weight",
        );
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, &format!("model.layers.{i}"));
        }
        if let Some(w) = weights.get("model.norm.weight") {
            self.norm.weight.value = Some(w.clone());
        }
        if let Some(ref mut lm_head) = self.lm_head {
            assign_weight(&mut lm_head.weight, weights, "lm_head.weight");
        }
    }

    pub fn load(
        model_dir: &Path,
        config: &CommandRConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut model = Self::new(config)?;
        let weights = load_safetensors_weights(model_dir)?;
        model.load_weights(&weights);
        mlx_rs::transforms::eval(weights.values())?;
        Ok(model)
    }
}

impl super::MlxModel for MlxCommandRForCausalLM {
    fn forward(
        &mut self,
        input_ids: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
    ) -> mlx_rs::error::Result<Array> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i])?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        // Compute logits.
        let logits = if self.tie_word_embeddings {
            self.embed_tokens.as_linear(&hidden_states)?
        } else {
            self.lm_head.as_mut().unwrap().forward(&hidden_states)?
        };

        // Apply logit scaling.
        let logits = logits.multiply(Array::from_f32(self.logit_scale))?;

        // Cast logits to f32 for sampling.
        logits.as_dtype(Dtype::Float32)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// Quantized variant
// ---------------------------------------------------------------------------

/// Quantized Command R attention.
struct MlxQuantizedCommandRAttention {
    q_proj: nn::QuantizedLinear,
    k_proj: nn::QuantizedLinear,
    v_proj: nn::QuantizedLinear,
    o_proj: nn::QuantizedLinear,
    q_norm: Option<nn::LayerNorm>,
    k_norm: Option<nn::LayerNorm>,
    rope: nn::Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl MlxQuantizedCommandRAttention {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &CommandRConfig,
        qc: &QuantConfig,
    ) -> Result<Self, Exception> {
        let c = &config.base;

        let (q_norm, k_norm) = if config.use_qk_norm {
            let mut qn = cohere_layer_norm(c.head_dim as i32, c.rms_norm_eps)?;
            let mut kn = cohere_layer_norm(c.head_dim as i32, c.rms_norm_eps)?;
            if let Some(w) = weights.get(&format!("{prefix}.q_norm.weight")) {
                qn.weight.value = Some(w.clone());
            }
            if let Some(w) = weights.get(&format!("{prefix}.k_norm.weight")) {
                kn.weight.value = Some(w.clone());
            }
            (Some(qn), Some(kn))
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.q_proj"),
                qc.group_size,
                qc.bits,
            ),
            k_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.k_proj"),
                qc.group_size,
                qc.bits,
            ),
            v_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.v_proj"),
                qc.group_size,
                qc.bits,
            ),
            o_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.o_proj"),
                qc.group_size,
                qc.bits,
            ),
            q_norm,
            k_norm,
            rope: {
                let mut r = nn::Rope::new(c.head_dim as i32);
                r.base = c.rope_theta;
                r.traditional = true;
                r
            },
            num_heads: c.num_attention_heads,
            num_kv_heads: c.num_kv_heads,
            head_dim: c.head_dim,
            scale: 1.0 / (c.head_dim as f32).sqrt(),
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        let mut q = q.reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?;
        let mut k = k.reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?;

        if let Some(ref mut norm) = self.q_norm {
            q = norm.forward(&q)?;
        }
        if let Some(ref mut norm) = self.k_norm {
            k = norm.forward(&k)?;
        }

        let q = q.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut v = v
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        let offset = if positions.size() > 0 {
            positions.reshape(&[-1])?.min(None)?.item::<i32>()
        } else {
            0
        };
        let q = self.rope.forward((&q, offset))?;
        k = self.rope.forward((&k, offset))?;

        if let Some((ck, cv)) = cache.take() {
            k = concatenate_axis(&[ck, k], 2)?;
            v = concatenate_axis(&[cv, v], 2)?;
        }
        *cache = Some((k.clone(), v.clone()));

        let mask = if seq_len > 1 {
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let out = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;

        let hidden = (self.num_heads * self.head_dim) as i32;
        let out = out
            .squeeze_axes(&[0])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[seq_len, hidden])?;

        self.o_proj.forward(&out)
    }
}

/// Quantized Command R decoder layer.
struct MlxQuantizedCommandRDecoderLayer {
    self_attn: MlxQuantizedCommandRAttention,
    mlp: MlxQuantizedLlamaMLP,
    input_layernorm: nn::LayerNorm,
}

impl MlxQuantizedCommandRDecoderLayer {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &CommandRConfig,
        qc: &QuantConfig,
    ) -> Result<Self, Exception> {
        let c = &config.base;
        let mut input_layernorm = cohere_layer_norm(c.hidden_size as i32, c.rms_norm_eps)?;
        if let Some(w) = weights.get(&format!("{prefix}.input_layernorm.weight")) {
            input_layernorm.weight.value = Some(w.clone());
        }

        Ok(Self {
            self_attn: MlxQuantizedCommandRAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config,
                qc,
            )?,
            mlp: MlxQuantizedLlamaMLP::from_weights(weights, &format!("{prefix}.mlp"), qc),
            input_layernorm,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        let residual = hidden_states;
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed, positions, cache)?;
        let mlp_output = self.mlp.forward(&normed)?;
        residual.add(&attn_output)?.add(&mlp_output)
    }
}

/// Quantized Command R for causal language modeling.
pub struct MlxQuantizedCommandRForCausalLM {
    embed_tokens: MlxEmbedTokens,
    layers: Vec<MlxQuantizedCommandRDecoderLayer>,
    norm: nn::LayerNorm,
    lm_head: Option<MlxLmHead>,
    tie_word_embeddings: bool,
    logit_scale: f32,
}

impl MlxQuantizedCommandRForCausalLM {
    pub fn load(
        model_dir: &Path,
        config: &CommandRConfig,
        qc: &QuantConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let c = &config.base;
        let weights = load_safetensors_weights(model_dir)?;

        let embed_tokens =
            MlxEmbedTokens::from_weights(&weights, "model.embed_tokens", qc.group_size, qc.bits);

        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for i in 0..c.num_hidden_layers {
            layers.push(MlxQuantizedCommandRDecoderLayer::from_weights(
                &weights,
                &format!("model.layers.{i}"),
                config,
                qc,
            )?);
        }

        let mut norm = cohere_layer_norm(c.hidden_size as i32, c.rms_norm_eps)?;
        if let Some(w) = weights.get("model.norm.weight") {
            norm.weight.value = Some(w.clone());
        }

        let lm_head = if c.tie_word_embeddings {
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
            tie_word_embeddings: c.tie_word_embeddings,
            logit_scale: config.logit_scale,
        })
    }
}

impl super::MlxModel for MlxQuantizedCommandRForCausalLM {
    fn forward(
        &mut self,
        input_ids: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
    ) -> mlx_rs::error::Result<Array> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i])?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        let logits = if self.tie_word_embeddings {
            self.embed_tokens.as_linear(&hidden_states)?
        } else {
            self.lm_head.as_mut().unwrap().forward(&hidden_states)?
        };

        let logits = logits.multiply(Array::from_f32(self.logit_scale))?;
        logits.as_dtype(Dtype::Float32)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/// Create an MLX Command R model (float).
pub fn create_mlx_commandr(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let commandr_config = CommandRConfig::from_hf_config(config)?;
    let model = MlxCommandRForCausalLM::load(model_dir, &commandr_config, dtype)?;
    Ok(Box::new(model))
}

/// Create a quantized MLX Command R model.
pub fn create_mlx_quantized_commandr(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let commandr_config = CommandRConfig::from_hf_config(config)?;
    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();
    let model = MlxQuantizedCommandRForCausalLM::load(model_dir, &commandr_config, &qc, dtype)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commandr_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["CohereForCausalLM"],
                "model_type": "cohere",
                "hidden_size": 8192,
                "num_attention_heads": 64,
                "num_key_value_heads": 64,
                "num_hidden_layers": 40,
                "intermediate_size": 22528,
                "vocab_size": 256000,
                "max_position_embeddings": 8192,
                "layer_norm_eps": 1e-5,
                "rope_theta": 8000000.0,
                "logit_scale": 0.0625,
                "tie_word_embeddings": true
            }"#,
        )
        .unwrap();

        let config = CommandRConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.base.hidden_size, 8192);
        assert_eq!(config.base.num_attention_heads, 64);
        assert!((config.logit_scale - 0.0625).abs() < 1e-6);
        assert!(!config.use_qk_norm);
        assert!(config.base.tie_word_embeddings);
    }

    #[test]
    fn test_commandr_config_defaults() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["CohereForCausalLM"],
                "hidden_size": 256,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "intermediate_size": 512,
                "vocab_size": 1000
            }"#,
        )
        .unwrap();

        let config = CommandRConfig::from_hf_config(&hf_config).unwrap();
        assert!((config.logit_scale - 1.0).abs() < 1e-6);
        assert!(!config.use_qk_norm);
        assert!(config.base.tie_word_embeddings);
        assert!((config.base.rope_theta - 8000000.0).abs() < 1.0);
    }

    #[test]
    fn test_cohere_layer_norm_no_bias() {
        let norm = cohere_layer_norm(32, 1e-5).unwrap();
        assert!(norm.bias.value.is_none());
        assert!(norm.weight.value.is_some());
    }

    #[test]
    fn test_mlx_registry_commandr() {
        let registry = super::super::MlxModelRegistry::default_registry();
        assert!(registry.contains("CohereForCausalLM"));
        assert!(registry.contains_quantized("CohereForCausalLM"));
    }
}
