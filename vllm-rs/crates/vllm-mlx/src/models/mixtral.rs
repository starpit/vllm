// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Mixtral MoE model architecture for MLX.
//!
//! Both float and quantized variants. Reuses LLaMA attention for GQA.
//! All layers are MoE — no dense/MoE alternation, no shared experts.
//!
//! Expert weights use w1/w2/w3 naming: w1 = gate_proj, w3 = up_proj, w2 = down_proj.
//! The MoE module is named `block_sparse_moe` (not `mlp`).
//!
//! Port of: `vllm/model_executor/models/mixtral.py`

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use crate::cache::MlxKvCache;
use crate::models::deepseek_v2::slice_quantized_linear;
use crate::models::llama::{
    LlamaConfig, MlxLlamaAttention, assign_weight, load_safetensors_weights,
};
use crate::models::quantized_llama::{
    MlxEmbedTokens, MlxLmHead, MlxQuantizedLlamaAttention, QuantConfig, make_quantized_linear,
};
use crate::models::qwen3_moe::MlxGate;
use vllm_model::weight::HfModelConfig;

// ---------------------------------------------------------------------------
// MlxMixtralConfig
// ---------------------------------------------------------------------------

/// Parsed configuration for a Mixtral model (MLX backend).
#[derive(Debug, Clone)]
pub struct MlxMixtralConfig {
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
    pub sliding_window: Option<usize>,

    // MoE-specific.
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
}

impl MlxMixtralConfig {
    /// Parse from a HuggingFace config.json.
    pub fn from_hf_config(config: &HfModelConfig) -> Result<Self, String> {
        let llama = LlamaConfig::from_hf_config(config)?;
        let extra = &config.extra;

        let get_usize = |key: &str| -> Option<usize> {
            extra.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
        };

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
            sliding_window: llama.sliding_window,
            num_local_experts: get_usize("num_local_experts").unwrap_or(8),
            num_experts_per_tok: get_usize("num_experts_per_tok").unwrap_or(2),
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
            sliding_window: self.sliding_window,
            partial_rotary_factor: 1.0,
            long_rope_scaling: None,
        }
    }
}

// ===========================================================================
// Float Mixtral
// ===========================================================================

// ---------------------------------------------------------------------------
// MlxMixtralExpertMLP
// ---------------------------------------------------------------------------

/// A single Mixtral expert MLP using w1/w2/w3 naming.
struct MlxMixtralExpertMLP {
    w1: nn::Linear, // gate_proj
    w2: nn::Linear, // down_proj
    w3: nn::Linear, // up_proj
}

impl MlxMixtralExpertMLP {
    fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            w1: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(false)
                .build()?,
            w2: nn::LinearBuilder::new(intermediate_size, hidden_size)
                .bias(false)
                .build()?,
            w3: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(false)
                .build()?,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(&mut self.w1.weight, weights, &format!("{prefix}.w1.weight"));
        assign_weight(&mut self.w2.weight, weights, &format!("{prefix}.w2.weight"));
        assign_weight(&mut self.w3.weight, weights, &format!("{prefix}.w3.weight"));
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.w1.forward(x)?;
        let gate = nn::silu(&gate)?;
        let up = self.w3.forward(x)?;
        let hidden = gate.multiply(&up)?;
        self.w2.forward(&hidden)
    }
}

// ---------------------------------------------------------------------------
// MlxMixtralMoE
// ---------------------------------------------------------------------------

/// Mixture of Experts layer — no shared expert (MLX float).
struct MlxMixtralMoE {
    gate: nn::Linear,
    experts: Vec<MlxMixtralExpertMLP>,
    top_k: usize,
}

impl MlxMixtralMoE {
    fn new(config: &MlxMixtralConfig) -> Result<Self, Exception> {
        let hidden = config.hidden_size as i32;
        let n = config.num_local_experts;

        let gate = nn::LinearBuilder::new(hidden, n as i32)
            .bias(false)
            .build()?;

        let mut experts = Vec::with_capacity(n);
        for _ in 0..n {
            experts.push(MlxMixtralExpertMLP::new(
                hidden,
                config.intermediate_size as i32,
            )?);
        }

        Ok(Self {
            gate,
            experts,
            top_k: config.num_experts_per_tok,
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
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
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

            // Mixtral does not renormalize top-k probabilities.
            let token_x = x.try_index(tok as i32)?;

            for &(expert_idx, prob) in &indexed {
                let token_x_2d = token_x.reshape(&[1, hidden])?;
                let expert_out = self.experts[expert_idx].forward(&token_x_2d)?;
                mlx_rs::transforms::eval(std::iter::once(&expert_out))?;

                let vals: Vec<f32> = expert_out.as_dtype(Dtype::Float32)?.as_slice().to_vec();
                for (j, &v) in vals.iter().enumerate() {
                    output_data[tok * (hidden as usize) + j] += v * prob;
                }
            }
        }

        let output = Array::from_slice(&output_data, &[seq_len, hidden]);
        output.as_dtype(x.dtype())
    }
}

// ---------------------------------------------------------------------------
// MlxMixtralDecoderLayer (float)
// ---------------------------------------------------------------------------

/// A single float decoder layer (always MoE).
struct MlxMixtralDecoderLayer {
    self_attn: MlxLlamaAttention,
    block_sparse_moe: MlxMixtralMoE,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
}

impl MlxMixtralDecoderLayer {
    fn new(config: &MlxMixtralConfig) -> Result<Self, Exception> {
        let llama_config = config.llama_config();

        Ok(Self {
            self_attn: MlxLlamaAttention::new(&llama_config)?,
            block_sparse_moe: MlxMixtralMoE::new(config)?,
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
        self.block_sparse_moe
            .load_weights(weights, &format!("{prefix}.block_sparse_moe"));
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
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed, positions, cache)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.block_sparse_moe.forward(&normed)?;
        hidden_states.add(&mlp_output)
    }
}

// ---------------------------------------------------------------------------
// MlxMixtralForCausalLM (float)
// ---------------------------------------------------------------------------

/// Mixtral for causal language modeling using MLX (float).
pub struct MlxMixtralForCausalLM {
    embed_tokens: nn::Embedding,
    layers: Vec<MlxMixtralDecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: Option<nn::Linear>,
    tie_word_embeddings: bool,
    #[allow(dead_code)]
    config: MlxMixtralConfig,
}

impl MlxMixtralForCausalLM {
    /// Load model weights from safetensors files in a directory.
    pub fn load(
        model_dir: &Path,
        config: &MlxMixtralConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(MlxMixtralDecoderLayer::new(config)?);
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

impl super::MlxModel for MlxMixtralForCausalLM {
    fn inject_lora(
        &mut self,
        adapter: &crate::lora::MlxLoraAdapter,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::lora::merge_lora_into_weight;
        let targets = &adapter.config.target_modules;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Attention projections only — Mixtral has no dense MLP layers.
            let attn_prefix = format!("model.layers.{}.self_attn", i);
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                if targets.iter().any(|t| t == name) {
                    let key = format!("{}.{}", attn_prefix, name);
                    if let Some((a, b)) = adapter.weights.get(&key) {
                        let proj = match name {
                            "q_proj" => &mut layer.self_attn.q_proj,
                            "k_proj" => &mut layer.self_attn.k_proj,
                            "v_proj" => &mut layer.self_attn.v_proj,
                            "o_proj" => &mut layer.self_attn.o_proj,
                            _ => unreachable!(),
                        };
                        merge_lora_into_weight(&mut proj.weight, a, b, adapter.scaling)?;
                    }
                }
            }
        }
        Ok(())
    }

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

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut kv_cache: MlxKvCache = (0..self.layers.len()).map(|_| None).collect();
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i])?;
        }
        self.norm.forward(&hidden_states)
    }
}

// ===========================================================================
// Quantized Mixtral
// ===========================================================================

// ---------------------------------------------------------------------------
// MlxQuantizedMixtralExpertMLP
// ---------------------------------------------------------------------------

/// A single quantized Mixtral expert using w1/w2/w3 naming.
struct MlxQuantizedMixtralExpertMLP {
    w1: nn::QuantizedLinear, // gate_proj
    w2: nn::QuantizedLinear, // down_proj
    w3: nn::QuantizedLinear, // up_proj
}

impl MlxQuantizedMixtralExpertMLP {
    fn from_weights(weights: &HashMap<String, Array>, prefix: &str, qc: &QuantConfig) -> Self {
        Self {
            w1: make_quantized_linear(weights, &format!("{prefix}.w1"), qc.group_size, qc.bits),
            w2: make_quantized_linear(weights, &format!("{prefix}.w2"), qc.group_size, qc.bits),
            w3: make_quantized_linear(weights, &format!("{prefix}.w3"), qc.group_size, qc.bits),
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.w1.forward(x)?;
        let gate = nn::silu(&gate)?;
        let up = self.w3.forward(x)?;
        let hidden = gate.multiply(&up)?;
        self.w2.forward(&hidden)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedMixtralMoE
// ---------------------------------------------------------------------------

/// Mixture of Experts layer — no shared expert (MLX quantized).
struct MlxQuantizedMixtralMoE {
    gate: MlxGate,
    experts: Vec<MlxQuantizedMixtralExpertMLP>,
    top_k: usize,
}

impl MlxQuantizedMixtralMoE {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &MlxMixtralConfig,
        qc: &QuantConfig,
    ) -> Self {
        let n = config.num_local_experts;

        let gate = MlxGate::from_weights(weights, &format!("{prefix}.gate"), qc);

        // Try fused switch_mlp format first, fall back to per-expert w1/w2/w3.
        let switch_prefix = format!("{prefix}.switch_mlp");
        let has_switch_mlp = weights.contains_key(&format!("{switch_prefix}.gate_proj.weight"));

        let experts = if has_switch_mlp {
            // switch_mlp stores fused 3D tensors; gate_proj=w1, up_proj=w3, down_proj=w2.
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
                    MlxQuantizedMixtralExpertMLP {
                        w1: slice_quantized_linear(gate_w, gate_s, gate_b, idx, qc),
                        w3: slice_quantized_linear(up_w, up_s, up_b, idx, qc),
                        w2: slice_quantized_linear(down_w, down_s, down_b, idx, qc),
                    }
                })
                .collect()
        } else {
            (0..n)
                .map(|i| {
                    MlxQuantizedMixtralExpertMLP::from_weights(
                        weights,
                        &format!("{prefix}.experts.{i}"),
                        qc,
                    )
                })
                .collect()
        };

        Self {
            gate,
            experts,
            top_k: config.num_experts_per_tok,
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
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

            let token_x = x.try_index(tok as i32)?;

            for &(expert_idx, prob) in &indexed {
                let token_x_2d = token_x.reshape(&[1, hidden])?;
                let expert_out = self.experts[expert_idx].forward(&token_x_2d)?;
                mlx_rs::transforms::eval(std::iter::once(&expert_out))?;

                let vals: Vec<f32> = expert_out.as_dtype(Dtype::Float32)?.as_slice().to_vec();
                for (j, &v) in vals.iter().enumerate() {
                    output_data[tok * (hidden as usize) + j] += v * prob;
                }
            }
        }

        let output = Array::from_slice(&output_data, &[seq_len, hidden]);
        output.as_dtype(x.dtype())
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedMixtralDecoderLayer
// ---------------------------------------------------------------------------

/// A single quantized decoder layer (always MoE).
struct MlxQuantizedMixtralDecoderLayer {
    self_attn: MlxQuantizedLlamaAttention,
    block_sparse_moe: MlxQuantizedMixtralMoE,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
}

impl MlxQuantizedMixtralDecoderLayer {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &MlxMixtralConfig,
        qc: &QuantConfig,
    ) -> Result<Self, Exception> {
        let llama_config = config.llama_config();

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
            block_sparse_moe: MlxQuantizedMixtralMoE::from_weights(
                weights,
                &format!("{prefix}.block_sparse_moe"),
                config,
                qc,
            ),
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
        let mlp_output = self.block_sparse_moe.forward(&normed)?;
        hidden_states.add(&mlp_output)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedMixtralForCausalLM
// ---------------------------------------------------------------------------

/// Quantized Mixtral for causal language modeling using MLX.
pub struct MlxQuantizedMixtralForCausalLM {
    embed_tokens: MlxEmbedTokens,
    layers: Vec<MlxQuantizedMixtralDecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: Option<MlxLmHead>,
    tie_word_embeddings: bool,
    #[allow(dead_code)]
    config: MlxMixtralConfig,
}

impl MlxQuantizedMixtralForCausalLM {
    pub fn load(
        model_dir: &Path,
        config: &MlxMixtralConfig,
        qc: &QuantConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let weights = load_safetensors_weights(model_dir)?;

        let embed_tokens =
            MlxEmbedTokens::from_weights(&weights, "model.embed_tokens", qc.group_size, qc.bits);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MlxQuantizedMixtralDecoderLayer::from_weights(
                &weights,
                &format!("model.layers.{i}"),
                config,
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

impl super::MlxModel for MlxQuantizedMixtralForCausalLM {
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

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut kv_cache: MlxKvCache = (0..self.layers.len()).map(|_| None).collect();
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i])?;
        }
        self.norm.forward(&hidden_states)
    }
}

// ===========================================================================
// Factory functions
// ===========================================================================

/// Factory function for creating a float MLX Mixtral model.
pub fn create_mlx_mixtral(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let mixtral_config = MlxMixtralConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let model = MlxMixtralForCausalLM::load(model_dir, &mixtral_config, dtype)?;
    Ok(Box::new(model))
}

/// Factory function for creating a quantized MLX Mixtral model.
pub fn create_mlx_quantized_mixtral(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let mixtral_config = MlxMixtralConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();
    tracing::info!(
        "Loading quantized MLX Mixtral (group_size={}, bits={})",
        qc.group_size,
        qc.bits
    );

    let model = MlxQuantizedMixtralForCausalLM::load(model_dir, &mixtral_config, &qc, dtype)?;
    Ok(Box::new(model))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> MlxMixtralConfig {
        MlxMixtralConfig {
            hidden_size: 32,
            num_attention_heads: 4,
            num_kv_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 1000000.0,
            head_dim: 8,
            tie_word_embeddings: true,
            sliding_window: Some(4096),
            num_local_experts: 4,
            num_experts_per_tok: 2,
        }
    }

    #[test]
    fn test_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["MixtralForCausalLM"],
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 32,
                "intermediate_size": 14336,
                "vocab_size": 32000,
                "rms_norm_eps": 1e-5,
                "rope_theta": 1000000.0,
                "sliding_window": 4096,
                "num_local_experts": 8,
                "num_experts_per_tok": 2
            }"#,
        )
        .unwrap();

        let config = MlxMixtralConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_local_experts, 8);
        assert_eq!(config.num_experts_per_tok, 2);
        assert_eq!(config.intermediate_size, 14336);
        assert_eq!(config.sliding_window, Some(4096));
    }

    #[test]
    fn test_moe_forward() {
        let config = test_config();
        let mut moe = MlxMixtralMoE::new(&config).unwrap();
        let x = mlx_rs::ops::ones::<f32>(&[2, config.hidden_size as i32]).unwrap();
        let out = moe.forward(&x).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[2, config.hidden_size as i32]);
    }

    #[test]
    fn test_float_model_forward() {
        let config = test_config();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(MlxMixtralDecoderLayer::new(&config).unwrap());
        }

        let model_inner = MlxMixtralForCausalLM {
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
