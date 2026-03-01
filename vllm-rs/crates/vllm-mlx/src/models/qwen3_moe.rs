// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Qwen3 MoE / Qwen2 MoE model architecture for MLX.
//!
//! Both float and quantized variants. Reuses LLaMA attention (handles QK
//! norms for Qwen3, optional QKV bias for Qwen2) and LLaMA MLP for experts.
//! Novel: sigmoid-gated shared expert + decoder_sparse_step layer selection.
//!
//! Port of: `vllm/model_executor/models/qwen3_moe.py`

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use crate::cache::MlxKvCache;
use crate::models::deepseek_v2::slice_quantized_linear;
use crate::models::llama::{
    LlamaConfig, MlxLlamaAttention, MlxLlamaMLP, assign_weight, load_safetensors_weights,
};
use crate::models::quantized_llama::{
    MlxEmbedTokens, MlxLmHead, MlxQuantizedLlamaAttention, MlxQuantizedLlamaMLP, QuantConfig,
    make_quantized_linear,
};
use vllm_model::weight::HfModelConfig;

// ---------------------------------------------------------------------------
// MlxQwen3MoeConfig
// ---------------------------------------------------------------------------

/// Parsed configuration for a Qwen3 MoE / Qwen2 MoE model (MLX backend).
#[derive(Debug, Clone)]
pub struct MlxQwen3MoeConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub head_dim: usize,
    pub tie_word_embeddings: bool,

    // MoE-specific.
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub decoder_sparse_step: usize,
    pub mlp_only_layers: Vec<usize>,
}

impl MlxQwen3MoeConfig {
    /// Parse from a HuggingFace config.json.
    pub fn from_hf_config(config: &HfModelConfig) -> Result<Self, String> {
        let llama = LlamaConfig::from_hf_config(config)?;
        let extra = &config.extra;

        let get_usize = |key: &str| -> Option<usize> {
            extra.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
        };
        let get_bool = |key: &str| -> Option<bool> { extra.get(key).and_then(|v| v.as_bool()) };

        Ok(Self {
            hidden_size: llama.hidden_size,
            num_attention_heads: llama.num_attention_heads,
            num_kv_heads: llama.num_kv_heads,
            num_hidden_layers: llama.num_hidden_layers,
            intermediate_size: llama.intermediate_size,
            vocab_size: llama.vocab_size,
            max_position_embeddings: llama.max_position_embeddings,
            rms_norm_eps: llama.rms_norm_eps,
            rope_theta: llama.rope_theta,
            head_dim: llama.head_dim,
            tie_word_embeddings: llama.tie_word_embeddings,
            num_experts: get_usize("num_experts").unwrap_or(0),
            num_experts_per_tok: get_usize("num_experts_per_tok").unwrap_or(4),
            moe_intermediate_size: get_usize("moe_intermediate_size")
                .unwrap_or(llama.intermediate_size),
            shared_expert_intermediate_size: get_usize("shared_expert_intermediate_size")
                .unwrap_or(0),
            norm_topk_prob: get_bool("norm_topk_prob").unwrap_or(true),
            decoder_sparse_step: get_usize("decoder_sparse_step").unwrap_or(1),
            mlp_only_layers: extra
                .get("mlp_only_layers")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default(),
        })
    }

    /// Produce a `LlamaConfig` for constructing attention layers.
    pub fn llama_config(&self) -> LlamaConfig {
        LlamaConfig {
            hidden_size: self.hidden_size,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            head_dim: self.head_dim,
            tie_word_embeddings: self.tie_word_embeddings,
            sliding_window: None,
            partial_rotary_factor: 1.0,
            long_rope_scaling: None,
        }
    }

    /// Whether a given layer index is a MoE layer.
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        !self.mlp_only_layers.contains(&layer_idx)
            && self.num_experts > 0
            && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }
}

// ===========================================================================
// Float MoE
// ===========================================================================

// ---------------------------------------------------------------------------
// MlxQwen3MoeMoE
// ---------------------------------------------------------------------------

/// Mixture of Experts layer with sigmoid-gated shared expert (MLX float).
struct MlxQwen3MoeMoE {
    gate: nn::Linear,
    experts: Vec<MlxLlamaMLP>,
    shared_expert: Option<MlxLlamaMLP>,
    shared_expert_gate: Option<nn::Linear>,
    top_k: usize,
    norm_topk_prob: bool,
}

impl MlxQwen3MoeMoE {
    fn new(config: &MlxQwen3MoeConfig) -> Result<Self, Exception> {
        let hidden = config.hidden_size as i32;
        let n = config.num_experts;

        let gate = nn::LinearBuilder::new(hidden, n as i32)
            .bias(false)
            .build()?;

        let mut experts = Vec::with_capacity(n);
        for _ in 0..n {
            experts.push(MlxLlamaMLP::new(
                hidden,
                config.moe_intermediate_size as i32,
            )?);
        }

        let (shared_expert, shared_expert_gate) = if config.shared_expert_intermediate_size > 0 {
            let se = MlxLlamaMLP::new(hidden, config.shared_expert_intermediate_size as i32)?;
            let seg = nn::LinearBuilder::new(hidden, 1).bias(false).build()?;
            (Some(se), Some(seg))
        } else {
            (None, None)
        };

        Ok(Self {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
            top_k: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(
            &mut self.gate.weight,
            weights,
            &format!("{prefix}.gate.weight"),
        );

        for (i, expert) in self.experts.iter_mut().enumerate() {
            expert.load_weights(weights, &format!("{prefix}.experts.{i}"));
        }

        if let Some(ref mut se) = self.shared_expert {
            se.load_weights(weights, &format!("{prefix}.shared_expert"));
        }
        if let Some(ref mut seg) = self.shared_expert_gate {
            assign_weight(
                &mut seg.weight,
                weights,
                &format!("{prefix}.shared_expert_gate.weight"),
            );
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // Evaluate x for routing decisions.
        mlx_rs::transforms::eval(std::iter::once(x))?;

        let seq_len = x.dim(0);
        let hidden = x.dim(1);

        // Compute router logits and softmax.
        let router_logits = self.gate.forward(x)?;
        let probs = mlx_rs::ops::softmax_axis(&router_logits, -1, None)?;
        mlx_rs::transforms::eval(std::iter::once(&probs))?;

        let probs_flat: Vec<f32> = probs.as_dtype(Dtype::Float32)?.as_slice().to_vec();
        let n_experts = self.experts.len();

        // Route each token.
        let mut output_data = vec![0.0f32; (seq_len * hidden) as usize];

        for tok in 0..(seq_len as usize) {
            let tok_probs = &probs_flat[tok * n_experts..(tok + 1) * n_experts];

            let mut indexed: Vec<(usize, f32)> = tok_probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(self.top_k);

            let total: f32 = indexed.iter().map(|(_, p)| p).sum();
            let scale = if self.norm_topk_prob && total > 0.0 {
                1.0 / total
            } else {
                1.0
            };

            let token_x = x.try_index(tok as i32)?;

            for &(expert_idx, prob) in &indexed {
                let token_x_2d = token_x.reshape(&[1, hidden])?;
                let expert_out = self.experts[expert_idx].forward(&token_x_2d)?;
                mlx_rs::transforms::eval(std::iter::once(&expert_out))?;

                let weight = prob * scale;
                let vals: Vec<f32> = expert_out.as_dtype(Dtype::Float32)?.as_slice().to_vec();
                for (j, &v) in vals.iter().enumerate() {
                    output_data[tok * (hidden as usize) + j] += v * weight;
                }
            }
        }

        let mut output = Array::from_slice(&output_data, &[seq_len, hidden]);
        output = output.as_dtype(x.dtype())?;

        // Shared expert with sigmoid gate.
        if let (Some(se), Some(seg)) = (&mut self.shared_expert, &mut self.shared_expert_gate) {
            let shared_out = se.forward(x)?;
            let gate_val = seg.forward(x)?; // [seq, 1]
            let gate_sigmoid = nn::sigmoid(&gate_val)?;
            let shared_gated = shared_out.multiply(&gate_sigmoid)?;
            output = output.add(&shared_gated)?;
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// MlxQwen3MoeDecoderLayer (float)
// ---------------------------------------------------------------------------

/// A single float decoder layer with dense MLP or MoE dispatch.
struct MlxQwen3MoeDecoderLayer {
    self_attn: MlxLlamaAttention,
    mlp: MlxQwen3MoeMlp,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
}

enum MlxQwen3MoeMlp {
    Dense(MlxLlamaMLP),
    MoE(MlxQwen3MoeMoE),
}

impl MlxQwen3MoeDecoderLayer {
    fn new(config: &MlxQwen3MoeConfig, layer_idx: usize) -> Result<Self, Exception> {
        let llama_config = config.llama_config();
        let mlp = if config.is_moe_layer(layer_idx) {
            MlxQwen3MoeMlp::MoE(MlxQwen3MoeMoE::new(config)?)
        } else {
            MlxQwen3MoeMlp::Dense(MlxLlamaMLP::new(
                config.hidden_size as i32,
                config.intermediate_size as i32,
            )?)
        };

        Ok(Self {
            self_attn: MlxLlamaAttention::new(&llama_config)?,
            mlp,
            input_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        self.self_attn
            .load_weights(weights, &format!("{prefix}.self_attn"));
        match &mut self.mlp {
            MlxQwen3MoeMlp::Dense(mlp) => {
                mlp.load_weights(weights, &format!("{prefix}.mlp"));
            }
            MlxQwen3MoeMlp::MoE(moe) => {
                moe.load_weights(weights, &format!("{prefix}.mlp"));
            }
        }
        assign_weight(
            &mut self.input_layernorm.weight,
            weights,
            &format!("{prefix}.input_layernorm.weight"),
        );
        assign_weight(
            &mut self.post_attention_layernorm.weight,
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        );
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        // Pre-attention layernorm + attention + residual.
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed, positions, cache)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        // Post-attention layernorm + MLP/MoE + residual.
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = match &mut self.mlp {
            MlxQwen3MoeMlp::Dense(mlp) => mlp.forward(&normed)?,
            MlxQwen3MoeMlp::MoE(moe) => moe.forward(&normed)?,
        };
        hidden_states.add(&mlp_output)
    }
}

// ---------------------------------------------------------------------------
// MlxQwen3MoeForCausalLM (float)
// ---------------------------------------------------------------------------

/// Qwen3 MoE for causal language modeling using MLX (float).
pub struct MlxQwen3MoeForCausalLM {
    embed_tokens: nn::Embedding,
    layers: Vec<MlxQwen3MoeDecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: Option<nn::Linear>,
    tie_word_embeddings: bool,
    #[allow(dead_code)]
    config: MlxQwen3MoeConfig,
}

impl MlxQwen3MoeForCausalLM {
    /// Load model weights from safetensors files in a directory.
    pub fn load(
        model_dir: &Path,
        config: &MlxQwen3MoeConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MlxQwen3MoeDecoderLayer::new(config, i)?);
        }

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(
                nn::LinearBuilder::new(config.hidden_size as i32, config.vocab_size as i32)
                    .bias(false)
                    .build()?,
            )
        };

        let mut model = Self {
            embed_tokens: nn::Embedding::new(config.vocab_size as i32, config.hidden_size as i32)?,
            layers,
            norm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            lm_head,
            tie_word_embeddings: config.tie_word_embeddings,
            config: config.clone(),
        };

        // Load weights.
        let weights = load_safetensors_weights(model_dir)?;
        assign_weight(
            &mut model.embed_tokens.weight,
            &weights,
            "model.embed_tokens.weight",
        );
        for (i, layer) in model.layers.iter_mut().enumerate() {
            layer.load_weights(&weights, &format!("model.layers.{i}"));
        }
        assign_weight(&mut model.norm.weight, &weights, "model.norm.weight");
        if let Some(ref mut lm_head) = model.lm_head {
            assign_weight(&mut lm_head.weight, &weights, "lm_head.weight");
        }

        mlx_rs::transforms::eval(weights.values())?;
        Ok(model)
    }
}

impl super::MlxModel for MlxQwen3MoeForCausalLM {
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

        logits.as_dtype(Dtype::Float32)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ===========================================================================
// Quantized MoE
// ===========================================================================

// ---------------------------------------------------------------------------
// MlxQuantizedQwen3MoeMoE
// ---------------------------------------------------------------------------

/// Router gate that may or may not be quantized.
///
/// mlx-community DeepSeek V2 models have float gates, but some Qwen3 MoE
/// models quantize the gate at different bit widths (e.g., 8-bit while the
/// rest of the model is 4-bit). Auto-detect from weight presence.
pub(crate) enum MlxGate {
    Float(nn::Linear),
    Quantized(nn::QuantizedLinear),
}

impl MlxGate {
    /// Load from weights: if `{prefix}.scales` exists, load as quantized
    /// (auto-detecting bits from packed weight shape); otherwise float.
    pub(crate) fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        qc: &QuantConfig,
    ) -> Self {
        if weights.contains_key(&format!("{prefix}.scales")) {
            // Auto-detect bits from packed weight shape.
            // packed_cols = in_features * bits / 32, so
            // bits = packed_cols * 32 / in_features
            // in_features = num_groups * group_size
            let bits = if let (Some(w), Some(s)) = (
                weights.get(&format!("{prefix}.weight")),
                weights.get(&format!("{prefix}.scales")),
            ) {
                let packed_cols = w.dim(1) as i64;
                let num_groups = s.dim(1) as i64;
                let in_features = num_groups * qc.group_size as i64;
                if in_features > 0 {
                    (packed_cols * 32 / in_features) as i32
                } else {
                    qc.bits
                }
            } else {
                qc.bits
            };
            MlxGate::Quantized(make_quantized_linear(weights, prefix, qc.group_size, bits))
        } else if let Some(w) = weights.get(&format!("{prefix}.weight")) {
            MlxGate::Float(nn::Linear {
                weight: Param::new(w.clone()),
                bias: Param::new(None),
            })
        } else {
            tracing::warn!("Gate weights not found at {prefix}, creating stub");
            MlxGate::Float(
                nn::LinearBuilder::new(1, 1)
                    .bias(false)
                    .build()
                    .expect("stub linear"),
            )
        }
    }

    pub(crate) fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        match self {
            MlxGate::Float(l) => l.forward(x),
            MlxGate::Quantized(l) => l.forward(x),
        }
    }
}

/// Mixture of Experts layer with sigmoid-gated shared expert (MLX quantized).
struct MlxQuantizedQwen3MoeMoE {
    gate: MlxGate,
    experts: Vec<MlxQuantizedLlamaMLP>,
    shared_expert: Option<MlxQuantizedLlamaMLP>,
    shared_expert_gate: Option<nn::Linear>,
    top_k: usize,
    norm_topk_prob: bool,
}

impl MlxQuantizedQwen3MoeMoE {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &MlxQwen3MoeConfig,
        qc: &QuantConfig,
    ) -> Self {
        let n = config.num_experts;

        // Gate: auto-detect quantized vs float.
        let gate = MlxGate::from_weights(weights, &format!("{prefix}.gate"), qc);

        // Load experts: try fused switch_mlp format, fall back to per-expert.
        let switch_prefix = format!("{prefix}.switch_mlp");
        let has_switch_mlp = weights.contains_key(&format!("{switch_prefix}.gate_proj.weight"));

        let experts = if has_switch_mlp {
            let gate_w = weights.get(&format!("{switch_prefix}.gate_proj.weight"));
            let gate_s = weights.get(&format!("{switch_prefix}.gate_proj.scales"));
            let gate_b = weights.get(&format!("{switch_prefix}.gate_proj.biases"));
            let up_w = weights.get(&format!("{switch_prefix}.up_proj.weight"));
            let up_s = weights.get(&format!("{switch_prefix}.up_proj.scales"));
            let up_b = weights.get(&format!("{switch_prefix}.up_proj.biases"));
            let down_w = weights.get(&format!("{switch_prefix}.down_proj.weight"));
            let down_s = weights.get(&format!("{switch_prefix}.down_proj.scales"));
            let down_b = weights.get(&format!("{switch_prefix}.down_proj.biases"));

            (0..n)
                .map(|i| {
                    let idx = i as i32;
                    MlxQuantizedLlamaMLP {
                        gate_proj: slice_quantized_linear(gate_w, gate_s, gate_b, idx, qc),
                        up_proj: slice_quantized_linear(up_w, up_s, up_b, idx, qc),
                        down_proj: slice_quantized_linear(down_w, down_s, down_b, idx, qc),
                    }
                })
                .collect()
        } else {
            (0..n)
                .map(|i| {
                    MlxQuantizedLlamaMLP::from_weights(
                        weights,
                        &format!("{prefix}.experts.{i}"),
                        qc,
                    )
                })
                .collect()
        };

        // Shared expert (quantized).
        let shared_expert = if config.shared_expert_intermediate_size > 0 {
            Some(MlxQuantizedLlamaMLP::from_weights(
                weights,
                &format!("{prefix}.shared_expert"),
                qc,
            ))
        } else {
            None
        };

        // Shared expert gate (always float, hidden_size -> 1).
        let shared_expert_gate = if config.shared_expert_intermediate_size > 0 {
            let mut seg = nn::LinearBuilder::new(config.hidden_size as i32, 1)
                .bias(false)
                .build()
                .expect("failed to create shared_expert_gate");
            assign_weight(
                &mut seg.weight,
                weights,
                &format!("{prefix}.shared_expert_gate.weight"),
            );
            Some(seg)
        } else {
            None
        };

        Self {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
            top_k: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // Evaluate x for routing decisions.
        mlx_rs::transforms::eval(std::iter::once(x))?;

        let seq_len = x.dim(0);
        let hidden = x.dim(1);

        let router_logits = self.gate.forward(x)?;
        let probs = mlx_rs::ops::softmax_axis(&router_logits, -1, None)?;
        mlx_rs::transforms::eval(std::iter::once(&probs))?;

        let probs_flat: Vec<f32> = probs.as_dtype(Dtype::Float32)?.as_slice().to_vec();
        let n_experts = self.experts.len();

        let mut output_data = vec![0.0f32; (seq_len * hidden) as usize];

        for tok in 0..(seq_len as usize) {
            let tok_probs = &probs_flat[tok * n_experts..(tok + 1) * n_experts];

            let mut indexed: Vec<(usize, f32)> = tok_probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(self.top_k);

            let total: f32 = indexed.iter().map(|(_, p)| p).sum();
            let scale = if self.norm_topk_prob && total > 0.0 {
                1.0 / total
            } else {
                1.0
            };

            let token_x = x.try_index(tok as i32)?;

            for &(expert_idx, prob) in &indexed {
                let token_x_2d = token_x.reshape(&[1, hidden])?;
                let expert_out = self.experts[expert_idx].forward(&token_x_2d)?;
                mlx_rs::transforms::eval(std::iter::once(&expert_out))?;

                let weight = prob * scale;
                let vals: Vec<f32> = expert_out.as_dtype(Dtype::Float32)?.as_slice().to_vec();
                for (j, &v) in vals.iter().enumerate() {
                    output_data[tok * (hidden as usize) + j] += v * weight;
                }
            }
        }

        let mut output = Array::from_slice(&output_data, &[seq_len, hidden]);
        output = output.as_dtype(x.dtype())?;

        // Shared expert with sigmoid gate.
        if let (Some(se), Some(seg)) = (&mut self.shared_expert, &mut self.shared_expert_gate) {
            let shared_out = se.forward(x)?;
            let gate_val = seg.forward(x)?;
            let gate_sigmoid = nn::sigmoid(&gate_val)?;
            let shared_gated = shared_out.multiply(&gate_sigmoid)?;
            output = output.add(&shared_gated)?;
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedQwen3MoeDecoderLayer
// ---------------------------------------------------------------------------

/// A single quantized decoder layer.
struct MlxQuantizedQwen3MoeDecoderLayer {
    self_attn: MlxQuantizedLlamaAttention,
    mlp: MlxQuantizedQwen3MoeMlp,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
}

enum MlxQuantizedQwen3MoeMlp {
    Dense(MlxQuantizedLlamaMLP),
    MoE(MlxQuantizedQwen3MoeMoE),
}

impl MlxQuantizedQwen3MoeDecoderLayer {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &MlxQwen3MoeConfig,
        layer_idx: usize,
        qc: &QuantConfig,
    ) -> Result<Self, Exception> {
        let llama_config = config.llama_config();

        let mlp = if config.is_moe_layer(layer_idx) {
            MlxQuantizedQwen3MoeMlp::MoE(MlxQuantizedQwen3MoeMoE::from_weights(
                weights,
                &format!("{prefix}.mlp"),
                config,
                qc,
            ))
        } else {
            MlxQuantizedQwen3MoeMlp::Dense(MlxQuantizedLlamaMLP::from_weights(
                weights,
                &format!("{prefix}.mlp"),
                qc,
            ))
        };

        let mut input_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        let mut post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        if let Some(w) = weights.get(&format!("{prefix}.input_layernorm.weight")) {
            input_layernorm.weight.value = w.clone();
        }
        if let Some(w) = weights.get(&format!("{prefix}.post_attention_layernorm.weight")) {
            post_attention_layernorm.weight.value = w.clone();
        }

        Ok(Self {
            self_attn: MlxQuantizedLlamaAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                &llama_config,
                qc,
            ),
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed, positions, cache)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = match &mut self.mlp {
            MlxQuantizedQwen3MoeMlp::Dense(mlp) => mlp.forward(&normed)?,
            MlxQuantizedQwen3MoeMlp::MoE(moe) => moe.forward(&normed)?,
        };
        hidden_states.add(&mlp_output)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedQwen3MoeForCausalLM
// ---------------------------------------------------------------------------

/// Quantized Qwen3 MoE for causal language modeling using MLX.
pub struct MlxQuantizedQwen3MoeForCausalLM {
    embed_tokens: MlxEmbedTokens,
    layers: Vec<MlxQuantizedQwen3MoeDecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: Option<MlxLmHead>,
    tie_word_embeddings: bool,
    #[allow(dead_code)]
    config: MlxQwen3MoeConfig,
}

impl MlxQuantizedQwen3MoeForCausalLM {
    pub fn load(
        model_dir: &Path,
        config: &MlxQwen3MoeConfig,
        qc: &QuantConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let weights = load_safetensors_weights(model_dir)?;

        let embed_tokens =
            MlxEmbedTokens::from_weights(&weights, "model.embed_tokens", qc.group_size, qc.bits);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MlxQuantizedQwen3MoeDecoderLayer::from_weights(
                &weights,
                &format!("model.layers.{i}"),
                config,
                i,
                qc,
            )?);
        }

        let mut norm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        if let Some(w) = weights.get("model.norm.weight") {
            norm.weight.value = w.clone();
        }

        let lm_head = if config.tie_word_embeddings {
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
            tie_word_embeddings: config.tie_word_embeddings,
            config: config.clone(),
        })
    }
}

impl super::MlxModel for MlxQuantizedQwen3MoeForCausalLM {
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

        logits.as_dtype(Dtype::Float32)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ===========================================================================
// Factory functions
// ===========================================================================

/// Factory function for creating a float MLX Qwen3 MoE model.
pub fn create_mlx_qwen3_moe(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let moe_config = MlxQwen3MoeConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let model = MlxQwen3MoeForCausalLM::load(model_dir, &moe_config, dtype)?;
    Ok(Box::new(model))
}

/// Factory function for creating a quantized MLX Qwen3 MoE model.
pub fn create_mlx_quantized_qwen3_moe(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let moe_config = MlxQwen3MoeConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();
    tracing::info!(
        "Loading quantized MLX Qwen3 MoE (group_size={}, bits={})",
        qc.group_size,
        qc.bits
    );

    let model = MlxQuantizedQwen3MoeForCausalLM::load(model_dir, &moe_config, &qc, dtype)?;
    Ok(Box::new(model))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    fn test_config() -> MlxQwen3MoeConfig {
        MlxQwen3MoeConfig {
            hidden_size: 32,
            num_attention_heads: 4,
            num_kv_heads: 2,
            num_hidden_layers: 4,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            head_dim: 8,
            tie_word_embeddings: true,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 32,
            shared_expert_intermediate_size: 48,
            norm_topk_prob: true,
            decoder_sparse_step: 2,
            mlp_only_layers: vec![],
        }
    }

    #[test]
    fn test_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["Qwen3MoeForCausalLM"],
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "num_hidden_layers": 24,
                "intermediate_size": 8192,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "num_experts": 64,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 1408,
                "shared_expert_intermediate_size": 5632,
                "norm_topk_prob": true,
                "decoder_sparse_step": 2,
                "mlp_only_layers": [0]
            }"#,
        )
        .unwrap();

        let config = MlxQwen3MoeConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_experts, 64);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.moe_intermediate_size, 1408);
        assert_eq!(config.shared_expert_intermediate_size, 5632);
        assert_eq!(config.decoder_sparse_step, 2);
        assert_eq!(config.mlp_only_layers, vec![0]);
    }

    #[test]
    fn test_is_moe_layer() {
        let config = test_config();
        assert!(!config.is_moe_layer(0));
        assert!(config.is_moe_layer(1));
        assert!(!config.is_moe_layer(2));
        assert!(config.is_moe_layer(3));
    }

    #[test]
    fn test_moe_forward() {
        let config = test_config();
        let mut moe = MlxQwen3MoeMoE::new(&config).unwrap();
        let x = mlx_rs::ops::ones::<f32>(&[2, config.hidden_size as i32]).unwrap();
        let out = moe.forward(&x).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[2, config.hidden_size as i32]);
    }

    #[test]
    fn test_float_model_forward() {
        let config = test_config();
        // Build model with default init.
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MlxQwen3MoeDecoderLayer::new(&config, i).unwrap());
        }

        let model_inner = MlxQwen3MoeForCausalLM {
            embed_tokens: nn::Embedding::new(config.vocab_size as i32, config.hidden_size as i32)
                .unwrap(),
            layers,
            norm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()
                .unwrap(),
            lm_head: None,
            tie_word_embeddings: true,
            config: config.clone(),
        };

        let mut model: Box<dyn super::super::MlxModel> = Box::new(model_inner);

        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        let logits = model
            .forward(&input_ids, &positions, &mut kv_cache)
            .unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.shape(), &[3, config.vocab_size as i32]);
    }
}
