// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! DeepSeek V2/V3 model architecture for MLX.
//!
//! Implements MLA (Multi-head Latent Attention) and MoE (Mixture of Experts)
//! using MLX nn primitives. All operations are lazy — the entire forward pass
//! builds a compute graph materialized with a single `eval()`.
//!
//! Port of: `vllm/model_executor/models/deepseek_v2.py`

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use crate::cache::MlxKvCache;
use crate::models::llama::{MlxLlamaMLP, assign_weight, load_safetensors_weights};
use crate::models::quantized_llama::{
    MlxEmbedTokens, MlxLmHead, MlxQuantizedLlamaMLP, QuantConfig, make_quantized_linear,
};
use vllm_model::weight::HfModelConfig;

// ---------------------------------------------------------------------------
// MlxDeepSeekV2Config
// ---------------------------------------------------------------------------

/// Parsed configuration for a DeepSeek V2/V3 model (MLX backend).
#[derive(Debug, Clone)]
pub struct MlxDeepSeekV2Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,

    // MLA-specific.
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,

    // MoE-specific.
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    pub moe_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
}

impl MlxDeepSeekV2Config {
    /// Parse from a HuggingFace config.json.
    pub fn from_hf_config(config: &HfModelConfig) -> Result<Self, String> {
        let hidden_size = config
            .hidden_size
            .ok_or_else(|| "missing hidden_size".to_string())?;
        let num_attention_heads = config
            .num_attention_heads
            .ok_or_else(|| "missing num_attention_heads".to_string())?;

        let extra = &config.extra;

        let get_usize = |key: &str| -> Option<usize> {
            extra.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
        };
        let get_f64 = |key: &str| -> Option<f64> { extra.get(key).and_then(|v| v.as_f64()) };
        let get_bool = |key: &str| -> Option<bool> { extra.get(key).and_then(|v| v.as_bool()) };

        Ok(Self {
            hidden_size,
            num_attention_heads,
            num_kv_heads: config.num_kv_heads().unwrap_or(num_attention_heads),
            num_hidden_layers: config
                .num_hidden_layers
                .ok_or_else(|| "missing num_hidden_layers".to_string())?,
            intermediate_size: config
                .intermediate_size
                .ok_or_else(|| "missing intermediate_size".to_string())?,
            vocab_size: config
                .vocab_size
                .ok_or_else(|| "missing vocab_size".to_string())?,
            max_position_embeddings: config.max_position_embeddings.unwrap_or(4096),
            rms_norm_eps: config.norm_eps() as f32,
            rope_theta: config.rope_theta.unwrap_or(10000.0) as f32,
            tie_word_embeddings: config.tie_word_embeddings.unwrap_or(false),
            qk_nope_head_dim: get_usize("qk_nope_head_dim").unwrap_or(128),
            qk_rope_head_dim: get_usize("qk_rope_head_dim").unwrap_or(64),
            v_head_dim: get_usize("v_head_dim").unwrap_or(128),
            q_lora_rank: get_usize("q_lora_rank"),
            kv_lora_rank: get_usize("kv_lora_rank").unwrap_or(512),
            n_routed_experts: get_usize("n_routed_experts").unwrap_or(0),
            n_shared_experts: get_usize("n_shared_experts").unwrap_or(0),
            num_experts_per_tok: get_usize("num_experts_per_tok").unwrap_or(6),
            first_k_dense_replace: get_usize("first_k_dense_replace").unwrap_or(1),
            moe_intermediate_size: get_usize("moe_intermediate_size")
                .unwrap_or(config.intermediate_size.unwrap_or(11008)),
            norm_topk_prob: get_bool("norm_topk_prob").unwrap_or(true),
            routed_scaling_factor: get_f64("routed_scaling_factor").unwrap_or(1.0) as f32,
        })
    }

    /// Total Q/K head dimension.
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Whether this model uses MoE layers.
    pub fn has_moe(&self) -> bool {
        self.n_routed_experts > 0
    }
}

// ---------------------------------------------------------------------------
// MlxDeepSeekV2MoE
// ---------------------------------------------------------------------------

/// Mixture of Experts layer with optional shared experts.
struct MlxDeepSeekV2MoE {
    gate: nn::Linear,
    experts: Vec<MlxLlamaMLP>,
    shared_experts: Option<MlxLlamaMLP>,
    top_k: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
}

impl MlxDeepSeekV2MoE {
    fn new(config: &MlxDeepSeekV2Config) -> Result<Self, Exception> {
        let hidden = config.hidden_size as i32;
        let n = config.n_routed_experts;

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

        let shared_experts = if config.n_shared_experts > 0 {
            Some(MlxLlamaMLP::new(
                hidden,
                (config.moe_intermediate_size * config.n_shared_experts) as i32,
            )?)
        } else {
            None
        };

        Ok(Self {
            gate,
            experts,
            shared_experts,
            top_k: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
            routed_scaling_factor: config.routed_scaling_factor,
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

        if let Some(ref mut shared) = self.shared_experts {
            shared.load_weights(weights, &format!("{prefix}.shared_experts"));
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // Evaluate x so we can read its values for routing.
        mlx_rs::transforms::eval(std::iter::once(x))?;

        let seq_len = x.dim(0);
        let hidden = x.dim(1);

        // Compute router logits and softmax.
        let router_logits = self.gate.forward(x)?; // [seq, n_experts]
        let probs = mlx_rs::ops::softmax_axis(&router_logits, -1, None)?;
        mlx_rs::transforms::eval(std::iter::once(&probs))?;

        let probs_flat: Vec<f32> = probs.as_dtype(Dtype::Float32)?.as_slice().to_vec();
        let n_experts = self.experts.len();

        // Route each token.
        let mut output_data = vec![0.0f32; (seq_len * hidden) as usize];

        for tok in 0..(seq_len as usize) {
            let tok_probs = &probs_flat[tok * n_experts..(tok + 1) * n_experts];

            // Top-k selection.
            let mut indexed: Vec<(usize, f32)> = tok_probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(self.top_k);

            let total: f32 = indexed.iter().map(|(_, p)| p).sum();
            let scale = if self.norm_topk_prob && total > 0.0 {
                1.0 / total
            } else {
                1.0
            };

            // Token input.
            let token_x = x.try_index(tok as i32)?;

            for &(expert_idx, prob) in &indexed {
                let token_x_2d = token_x.reshape(&[1, hidden])?;
                let expert_out = self.experts[expert_idx].forward(&token_x_2d)?;
                mlx_rs::transforms::eval(std::iter::once(&expert_out))?;

                let weight = prob * scale * self.routed_scaling_factor;
                let vals: Vec<f32> = expert_out.as_dtype(Dtype::Float32)?.as_slice().to_vec();
                for (j, &v) in vals.iter().enumerate() {
                    output_data[tok * (hidden as usize) + j] += v * weight;
                }
            }
        }

        let mut output = Array::from_slice(&output_data, &[seq_len, hidden]);
        output = output.as_dtype(x.dtype())?;

        // Add shared expert contribution.
        if let Some(ref mut shared) = self.shared_experts {
            let shared_out = shared.forward(x)?;
            output = output.add(&shared_out)?;
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// MlxDeepSeekV2Attention (MLA)
// ---------------------------------------------------------------------------

/// DeepSeek V2 Multi-head Latent Attention for MLX.
struct MlxDeepSeekV2Attention {
    // Q path.
    q_a_proj: Option<nn::Linear>,
    q_a_layernorm: Option<nn::RmsNorm>,
    q_b_proj: Option<nn::Linear>,
    q_proj: Option<nn::Linear>,

    // KV path.
    kv_a_proj_with_mqa: nn::Linear,
    kv_a_layernorm: nn::RmsNorm,
    kv_b_proj: nn::Linear,

    // Output.
    o_proj: nn::Linear,

    // RoPE for the rope dimensions only.
    rope: nn::Rope,

    // Dimensions.
    num_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    scale: f32,
}

impl MlxDeepSeekV2Attention {
    fn new(config: &MlxDeepSeekV2Config) -> Result<Self, Exception> {
        let hidden = config.hidden_size as i32;
        let num_heads = config.num_attention_heads;
        let qk_nope_head_dim = config.qk_nope_head_dim;
        let qk_rope_head_dim = config.qk_rope_head_dim;
        let v_head_dim = config.v_head_dim;
        let kv_lora_rank = config.kv_lora_rank;
        let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;

        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) = if let Some(q_lora_rank) =
            config.q_lora_rank
        {
            let q_a = nn::LinearBuilder::new(hidden, q_lora_rank as i32)
                .bias(false)
                .build()?;
            let q_a_ln = nn::RmsNormBuilder::new(q_lora_rank as i32)
                .eps(config.rms_norm_eps)
                .build()?;
            let q_b = nn::LinearBuilder::new(q_lora_rank as i32, (num_heads * qk_head_dim) as i32)
                .bias(false)
                .build()?;
            (Some(q_a), Some(q_a_ln), Some(q_b), None)
        } else {
            let q = nn::LinearBuilder::new(hidden, (num_heads * qk_head_dim) as i32)
                .bias(false)
                .build()?;
            (None, None, None, Some(q))
        };

        let kv_a_proj_with_mqa =
            nn::LinearBuilder::new(hidden, (kv_lora_rank + qk_rope_head_dim) as i32)
                .bias(false)
                .build()?;
        let kv_a_layernorm = nn::RmsNormBuilder::new(kv_lora_rank as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        let kv_b_proj = nn::LinearBuilder::new(
            kv_lora_rank as i32,
            (num_heads * (qk_nope_head_dim + v_head_dim)) as i32,
        )
        .bias(false)
        .build()?;

        let o_proj = nn::LinearBuilder::new((num_heads * v_head_dim) as i32, hidden)
            .bias(false)
            .build()?;

        let rope = {
            let mut r = nn::Rope::new(qk_rope_head_dim as i32);
            r.base = config.rope_theta;
            r
        };

        let scale = 1.0 / (qk_head_dim as f32).sqrt();

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rope,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_lora_rank,
            scale,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        if let Some(ref mut q_a) = self.q_a_proj {
            assign_weight(
                &mut q_a.weight,
                weights,
                &format!("{prefix}.q_a_proj.weight"),
            );
        }
        if let Some(ref mut q_a_ln) = self.q_a_layernorm {
            assign_weight(
                &mut q_a_ln.weight,
                weights,
                &format!("{prefix}.q_a_layernorm.weight"),
            );
        }
        if let Some(ref mut q_b) = self.q_b_proj {
            assign_weight(
                &mut q_b.weight,
                weights,
                &format!("{prefix}.q_b_proj.weight"),
            );
        }
        if let Some(ref mut q) = self.q_proj {
            assign_weight(&mut q.weight, weights, &format!("{prefix}.q_proj.weight"));
        }

        assign_weight(
            &mut self.kv_a_proj_with_mqa.weight,
            weights,
            &format!("{prefix}.kv_a_proj_with_mqa.weight"),
        );
        assign_weight(
            &mut self.kv_a_layernorm.weight,
            weights,
            &format!("{prefix}.kv_a_layernorm.weight"),
        );
        assign_weight(
            &mut self.kv_b_proj.weight,
            weights,
            &format!("{prefix}.kv_b_proj.weight"),
        );
        assign_weight(
            &mut self.o_proj.weight,
            weights,
            &format!("{prefix}.o_proj.weight"),
        );
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);

        // --- Q path ---
        let q_full = if let (Some(q_a), Some(q_a_ln), Some(q_b)) = (
            self.q_a_proj.as_mut(),
            self.q_a_layernorm.as_mut(),
            self.q_b_proj.as_mut(),
        ) {
            let q_latent = q_a.forward(hidden_states)?;
            let q_latent = q_a_ln.forward(&q_latent)?;
            q_b.forward(&q_latent)?
        } else {
            self.q_proj.as_mut().unwrap().forward(hidden_states)?
        };

        // Reshape: [seq, num_heads * qk_head_dim] -> [seq, num_heads, qk_head_dim]
        let q = q_full.reshape(&[seq_len, self.num_heads as i32, self.qk_head_dim as i32])?;

        // Split Q into nope and rope parts: [seq, heads, nope] and [seq, heads, rope]
        let q_parts = q.split_axis(&[self.qk_nope_head_dim as i32], -1)?;
        let q_nope = &q_parts[0];
        let q_pe = &q_parts[1];

        // --- KV path ---
        let kv_a = self.kv_a_proj_with_mqa.forward(hidden_states)?;
        // [seq, kv_lora_rank + rope_dim]

        let kv_a_parts = kv_a.split_axis(&[self.kv_lora_rank as i32], -1)?;
        let kv_latent = &kv_a_parts[0];
        let k_pe = &kv_a_parts[1];

        // kv_latent → RMSNorm → kv_b_proj
        let kv_latent = self.kv_a_layernorm.forward(kv_latent)?;
        let kv_b = self.kv_b_proj.forward(&kv_latent)?;
        // [seq, num_heads * (nope + v_dim)]

        let kv_b_total = self.qk_nope_head_dim + self.v_head_dim;
        let kv_b = kv_b.reshape(&[seq_len, self.num_heads as i32, kv_b_total as i32])?;

        let kv_b_parts = kv_b.split_axis(&[self.qk_nope_head_dim as i32], -1)?;
        let k_nope = &kv_b_parts[0];
        let v = &kv_b_parts[1];

        // --- Apply RoPE to q_pe and k_pe ---
        // Transform to SDPA layout for RoPE: [1, heads, seq, rope_dim]
        let q_pe = q_pe.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        // k_pe: [seq, rope_dim] -> [1, 1, seq, rope_dim]
        let k_pe = k_pe
            .reshape(&[seq_len, 1, self.qk_rope_head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        let offset = if positions.size() > 0 {
            positions.reshape(&[-1])?.min(None)?.item::<i32>()
        } else {
            0
        };
        let q_pe = self.rope.forward((&q_pe, offset))?;
        let k_pe = self.rope.forward((&k_pe, offset))?;

        // Back to [seq, heads, dim] layout for assembly.
        let q_pe = q_pe.squeeze_axes(&[0])?.transpose_axes(&[1, 0, 2])?;
        let k_pe = k_pe.squeeze_axes(&[0])?.transpose_axes(&[1, 0, 2])?;

        // --- Assemble full Q and K ---
        let q = concatenate_axis(&[q_nope, &q_pe], 2)?;

        // Expand k_pe from [seq, 1, rope_dim] to [seq, num_heads, rope_dim]
        let k_pe = mlx_rs::ops::broadcast_to(
            &k_pe,
            &[seq_len, self.num_heads as i32, self.qk_rope_head_dim as i32],
        )?;
        let k = concatenate_axis(&[k_nope, &k_pe], 2)?;

        // --- Pad V to qk_head_dim ---
        let v_padded = if self.v_head_dim < self.qk_head_dim {
            let pad_size = self.qk_head_dim - self.v_head_dim;
            let padding =
                mlx_rs::ops::zeros::<f32>(&[seq_len, self.num_heads as i32, pad_size as i32])?;
            let padding = padding.as_dtype(v.dtype())?;
            concatenate_axis(&[v, &padding], 2)?
        } else {
            v.clone()
        };

        // --- Transform to SDPA layout: [1, heads, seq, head_dim] ---
        let q = q.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut v_sdpa = v_padded.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;

        // KV cache update
        if let Some((ck, cv)) = cache.take() {
            k = concatenate_axis(&[ck, k], 2)?;
            v_sdpa = concatenate_axis(&[cv, v_sdpa], 2)?;
        }
        *cache = Some((k.clone(), v_sdpa.clone()));

        // Fused SDPA
        let mask = if seq_len > 1 {
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let out = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v_sdpa, self.scale, mask)?;

        // --- Slice V back from padded dim ---
        // out: [1, heads, seq, qk_head_dim] -> slice -> [1, heads, seq, v_head_dim]
        let out = if self.v_head_dim < self.qk_head_dim {
            let parts = out.split_axis(&[self.v_head_dim as i32], -1)?;
            parts.into_iter().next().unwrap()
        } else {
            out
        };

        // [1, heads, seq, v_head_dim] -> [seq, heads * v_head_dim]
        let hidden = (self.num_heads * self.v_head_dim) as i32;
        let out = out
            .squeeze_axes(&[0])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[seq_len, hidden])?;

        self.o_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// MlxDeepSeekV2DecoderLayer
// ---------------------------------------------------------------------------

/// A single DeepSeek V2 decoder layer using MLX.
struct MlxDeepSeekV2DecoderLayer {
    self_attn: MlxDeepSeekV2Attention,
    mlp: MlxDeepSeekV2Mlp,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
}

enum MlxDeepSeekV2Mlp {
    Dense(MlxLlamaMLP),
    MoE(MlxDeepSeekV2MoE),
}

impl MlxDeepSeekV2DecoderLayer {
    fn new(config: &MlxDeepSeekV2Config, layer_idx: usize) -> Result<Self, Exception> {
        let mlp = if config.has_moe() && layer_idx >= config.first_k_dense_replace {
            MlxDeepSeekV2Mlp::MoE(MlxDeepSeekV2MoE::new(config)?)
        } else {
            MlxDeepSeekV2Mlp::Dense(MlxLlamaMLP::new(
                config.hidden_size as i32,
                config.intermediate_size as i32,
            )?)
        };

        Ok(Self {
            self_attn: MlxDeepSeekV2Attention::new(config)?,
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
            MlxDeepSeekV2Mlp::Dense(mlp) => {
                mlp.load_weights(weights, &format!("{prefix}.mlp"));
            }
            MlxDeepSeekV2Mlp::MoE(moe) => {
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
            MlxDeepSeekV2Mlp::Dense(mlp) => mlp.forward(&normed)?,
            MlxDeepSeekV2Mlp::MoE(moe) => moe.forward(&normed)?,
        };
        hidden_states.add(&mlp_output)
    }
}

// ---------------------------------------------------------------------------
// MlxDeepSeekV2ForCausalLM
// ---------------------------------------------------------------------------

/// DeepSeek V2 for causal language modeling using MLX.
pub struct MlxDeepSeekV2ForCausalLM {
    embed_tokens: nn::Embedding,
    layers: Vec<MlxDeepSeekV2DecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: Option<nn::Linear>,
    tie_word_embeddings: bool,
    #[allow(dead_code)]
    config: MlxDeepSeekV2Config,
}

impl MlxDeepSeekV2ForCausalLM {
    fn new(config: &MlxDeepSeekV2Config) -> Result<Self, Exception> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MlxDeepSeekV2DecoderLayer::new(config, i)?);
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

        Ok(Self {
            embed_tokens: nn::Embedding::new(config.vocab_size as i32, config.hidden_size as i32)?,
            layers,
            norm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            lm_head,
            tie_word_embeddings: config.tie_word_embeddings,
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
        assign_weight(&mut self.norm.weight, weights, "model.norm.weight");
        if let Some(ref mut lm_head) = self.lm_head {
            assign_weight(&mut lm_head.weight, weights, "lm_head.weight");
        }
    }

    fn load(
        model_dir: &Path,
        config: &MlxDeepSeekV2Config,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut model = Self::new(config)?;
        let weights = load_safetensors_weights(model_dir)?;
        model.load_weights(&weights);
        mlx_rs::transforms::eval(weights.values())?;
        Ok(model)
    }
}

impl super::MlxModel for MlxDeepSeekV2ForCausalLM {
    fn inject_lora(
        &mut self,
        adapter: &crate::lora::MlxLoraAdapter,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::lora::merge_lora_into_weight;
        let targets = &adapter.config.target_modules;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // MLA attention projections.
            let attn_prefix = format!("model.layers.{}.self_attn", i);
            // Optional low-rank Q path.
            if targets.iter().any(|t| t == "q_a_proj") {
                let key = format!("{}.q_a_proj", attn_prefix);
                if let (Some((a, b)), Some(proj)) =
                    (adapter.weights.get(&key), layer.self_attn.q_a_proj.as_mut())
                {
                    merge_lora_into_weight(&mut proj.weight, a, b, adapter.scaling)?;
                }
            }
            if targets.iter().any(|t| t == "q_b_proj") {
                let key = format!("{}.q_b_proj", attn_prefix);
                if let (Some((a, b)), Some(proj)) =
                    (adapter.weights.get(&key), layer.self_attn.q_b_proj.as_mut())
                {
                    merge_lora_into_weight(&mut proj.weight, a, b, adapter.scaling)?;
                }
            }
            // Direct Q projection (non-low-rank path).
            if targets.iter().any(|t| t == "q_proj") {
                let key = format!("{}.q_proj", attn_prefix);
                if let (Some((a, b)), Some(proj)) =
                    (adapter.weights.get(&key), layer.self_attn.q_proj.as_mut())
                {
                    merge_lora_into_weight(&mut proj.weight, a, b, adapter.scaling)?;
                }
            }
            // KV path and output.
            for name in ["kv_a_proj_with_mqa", "kv_b_proj", "o_proj"] {
                if targets.iter().any(|t| t == name) {
                    let key = format!("{}.{}", attn_prefix, name);
                    if let Some((a, b)) = adapter.weights.get(&key) {
                        let proj = match name {
                            "kv_a_proj_with_mqa" => &mut layer.self_attn.kv_a_proj_with_mqa,
                            "kv_b_proj" => &mut layer.self_attn.kv_b_proj,
                            "o_proj" => &mut layer.self_attn.o_proj,
                            _ => unreachable!(),
                        };
                        merge_lora_into_weight(&mut proj.weight, a, b, adapter.scaling)?;
                    }
                }
            }
            // Dense MLP only (MoE layers are skipped).
            if let MlxDeepSeekV2Mlp::Dense(ref mut mlp) = layer.mlp {
                let mlp_prefix = format!("model.layers.{}.mlp", i);
                for name in ["gate_proj", "up_proj", "down_proj"] {
                    if targets.iter().any(|t| t == name) {
                        let key = format!("{}.{}", mlp_prefix, name);
                        if let Some((a, b)) = adapter.weights.get(&key) {
                            let proj = match name {
                                "gate_proj" => &mut mlp.gate_proj,
                                "up_proj" => &mut mlp.up_proj,
                                "down_proj" => &mut mlp.down_proj,
                                _ => unreachable!(),
                            };
                            merge_lora_into_weight(&mut proj.weight, a, b, adapter.scaling)?;
                        }
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

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/// Factory function for creating an MLX DeepSeek V2 model (float weights).
pub fn create_mlx_deepseek_v2(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let ds_config = MlxDeepSeekV2Config::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let model = MlxDeepSeekV2ForCausalLM::load(model_dir, &ds_config, dtype)?;
    Ok(Box::new(model))
}

// ===========================================================================
// Quantized variant
// ===========================================================================

// ---------------------------------------------------------------------------
// MlxQuantizedDeepSeekV2Attention (MLA, quantized)
// ---------------------------------------------------------------------------

/// Quantized DeepSeek V2 Multi-head Latent Attention for MLX.
struct MlxQuantizedDeepSeekV2Attention {
    // Q path.
    q_a_proj: Option<nn::QuantizedLinear>,
    q_a_layernorm: Option<nn::RmsNorm>,
    q_b_proj: Option<nn::QuantizedLinear>,
    q_proj: Option<nn::QuantizedLinear>,

    // KV path.
    kv_a_proj_with_mqa: nn::QuantizedLinear,
    kv_a_layernorm: nn::RmsNorm,
    kv_b_proj: nn::QuantizedLinear,

    // Output.
    o_proj: nn::QuantizedLinear,

    // RoPE for the rope dimensions only.
    rope: nn::Rope,

    // Dimensions.
    num_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    scale: f32,
}

impl MlxQuantizedDeepSeekV2Attention {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &MlxDeepSeekV2Config,
        qc: &QuantConfig,
    ) -> Result<Self, Exception> {
        let num_heads = config.num_attention_heads;
        let qk_nope_head_dim = config.qk_nope_head_dim;
        let qk_rope_head_dim = config.qk_rope_head_dim;
        let v_head_dim = config.v_head_dim;
        let kv_lora_rank = config.kv_lora_rank;
        let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;

        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) = if let Some(q_lora_rank) =
            config.q_lora_rank
        {
            let q_a = make_quantized_linear(
                weights,
                &format!("{prefix}.q_a_proj"),
                qc.group_size,
                qc.bits,
            );
            let mut q_a_ln = nn::RmsNormBuilder::new(q_lora_rank as i32)
                .eps(config.rms_norm_eps)
                .build()?;
            if let Some(w) = weights.get(&format!("{prefix}.q_a_layernorm.weight")) {
                q_a_ln.weight.value = w.clone();
            }
            let q_b = make_quantized_linear(
                weights,
                &format!("{prefix}.q_b_proj"),
                qc.group_size,
                qc.bits,
            );
            (Some(q_a), Some(q_a_ln), Some(q_b), None)
        } else {
            let q =
                make_quantized_linear(weights, &format!("{prefix}.q_proj"), qc.group_size, qc.bits);
            (None, None, None, Some(q))
        };

        let kv_a_proj_with_mqa = make_quantized_linear(
            weights,
            &format!("{prefix}.kv_a_proj_with_mqa"),
            qc.group_size,
            qc.bits,
        );
        let mut kv_a_layernorm = nn::RmsNormBuilder::new(kv_lora_rank as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        if let Some(w) = weights.get(&format!("{prefix}.kv_a_layernorm.weight")) {
            kv_a_layernorm.weight.value = w.clone();
        }
        let kv_b_proj = make_quantized_linear(
            weights,
            &format!("{prefix}.kv_b_proj"),
            qc.group_size,
            qc.bits,
        );

        let o_proj =
            make_quantized_linear(weights, &format!("{prefix}.o_proj"), qc.group_size, qc.bits);

        let rope = {
            let mut r = nn::Rope::new(qk_rope_head_dim as i32);
            r.base = config.rope_theta;
            r
        };

        let scale = 1.0 / (qk_head_dim as f32).sqrt();

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rope,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_lora_rank,
            scale,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);

        // --- Q path ---
        let q_full = if let (Some(q_a), Some(q_a_ln), Some(q_b)) = (
            self.q_a_proj.as_mut(),
            self.q_a_layernorm.as_mut(),
            self.q_b_proj.as_mut(),
        ) {
            let q_latent = q_a.forward(hidden_states)?;
            let q_latent = q_a_ln.forward(&q_latent)?;
            q_b.forward(&q_latent)?
        } else {
            self.q_proj.as_mut().unwrap().forward(hidden_states)?
        };

        let q = q_full.reshape(&[seq_len, self.num_heads as i32, self.qk_head_dim as i32])?;

        let q_parts = q.split_axis(&[self.qk_nope_head_dim as i32], -1)?;
        let q_nope = &q_parts[0];
        let q_pe = &q_parts[1];

        // --- KV path ---
        let kv_a = self.kv_a_proj_with_mqa.forward(hidden_states)?;

        let kv_a_parts = kv_a.split_axis(&[self.kv_lora_rank as i32], -1)?;
        let kv_latent = &kv_a_parts[0];
        let k_pe = &kv_a_parts[1];

        let kv_latent = self.kv_a_layernorm.forward(kv_latent)?;
        let kv_b = self.kv_b_proj.forward(&kv_latent)?;

        let kv_b_total = self.qk_nope_head_dim + self.v_head_dim;
        let kv_b = kv_b.reshape(&[seq_len, self.num_heads as i32, kv_b_total as i32])?;

        let kv_b_parts = kv_b.split_axis(&[self.qk_nope_head_dim as i32], -1)?;
        let k_nope = &kv_b_parts[0];
        let v = &kv_b_parts[1];

        // --- Apply RoPE to q_pe and k_pe ---
        let q_pe = q_pe.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let k_pe = k_pe
            .reshape(&[seq_len, 1, self.qk_rope_head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        let offset = if positions.size() > 0 {
            positions.reshape(&[-1])?.min(None)?.item::<i32>()
        } else {
            0
        };
        let q_pe = self.rope.forward((&q_pe, offset))?;
        let k_pe = self.rope.forward((&k_pe, offset))?;

        let q_pe = q_pe.squeeze_axes(&[0])?.transpose_axes(&[1, 0, 2])?;
        let k_pe = k_pe.squeeze_axes(&[0])?.transpose_axes(&[1, 0, 2])?;

        // --- Assemble full Q and K ---
        let q = concatenate_axis(&[q_nope, &q_pe], 2)?;

        let k_pe = mlx_rs::ops::broadcast_to(
            &k_pe,
            &[seq_len, self.num_heads as i32, self.qk_rope_head_dim as i32],
        )?;
        let k = concatenate_axis(&[k_nope, &k_pe], 2)?;

        // --- Pad V to qk_head_dim ---
        let v_padded = if self.v_head_dim < self.qk_head_dim {
            let pad_size = self.qk_head_dim - self.v_head_dim;
            let padding =
                mlx_rs::ops::zeros::<f32>(&[seq_len, self.num_heads as i32, pad_size as i32])?;
            let padding = padding.as_dtype(v.dtype())?;
            concatenate_axis(&[v, &padding], 2)?
        } else {
            v.clone()
        };

        // --- Transform to SDPA layout: [1, heads, seq, head_dim] ---
        let q = q.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut v_sdpa = v_padded.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;

        // KV cache update
        if let Some((ck, cv)) = cache.take() {
            k = concatenate_axis(&[ck, k], 2)?;
            v_sdpa = concatenate_axis(&[cv, v_sdpa], 2)?;
        }
        *cache = Some((k.clone(), v_sdpa.clone()));

        // Fused SDPA
        let mask = if seq_len > 1 {
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let out = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v_sdpa, self.scale, mask)?;

        // --- Slice V back from padded dim ---
        let out = if self.v_head_dim < self.qk_head_dim {
            let parts = out.split_axis(&[self.v_head_dim as i32], -1)?;
            parts.into_iter().next().unwrap()
        } else {
            out
        };

        // [1, heads, seq, v_head_dim] -> [seq, heads * v_head_dim]
        let hidden = (self.num_heads * self.v_head_dim) as i32;
        let out = out
            .squeeze_axes(&[0])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[seq_len, hidden])?;

        self.o_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedDeepSeekV2MoE
// ---------------------------------------------------------------------------

/// Quantized Mixture of Experts layer with optional shared experts.
///
/// MLX quantized models store routed experts as fused 3D `switch_mlp` tensors
/// (shape `[n_experts, out, in_packed]`) rather than per-expert `experts.{i}`
/// weights.  The gate (router) is a regular float linear, not quantized.
struct MlxQuantizedDeepSeekV2MoE {
    /// Gate (router) — float linear, not quantized.
    gate: nn::Linear,
    experts: Vec<MlxQuantizedLlamaMLP>,
    shared_experts: Option<MlxQuantizedLlamaMLP>,
    top_k: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
}

/// Slice expert `i` from a fused 3D quantized weight tensor triplet.
pub(crate) fn slice_quantized_linear(
    w3d: Option<&Array>,
    s3d: Option<&Array>,
    b3d: Option<&Array>,
    idx: i32,
    qc: &QuantConfig,
) -> nn::QuantizedLinear {
    let slice_or_zero = |arr: Option<&Array>| {
        arr.and_then(|t| t.try_index(idx).ok())
            .unwrap_or(Array::from_f32(0.0))
    };

    nn::QuantizedLinear {
        group_size: qc.group_size,
        bits: qc.bits,
        scales: Param::new(slice_or_zero(s3d)),
        biases: Param::new(slice_or_zero(b3d)),
        inner: nn::Linear {
            weight: Param::new(slice_or_zero(w3d)),
            bias: Param::new(None),
        },
    }
}

impl MlxQuantizedDeepSeekV2MoE {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &MlxDeepSeekV2Config,
        qc: &QuantConfig,
    ) -> Self {
        let n = config.n_routed_experts;

        // Gate is a regular float linear in mlx-community quantized format.
        let mut gate = nn::LinearBuilder::new(config.hidden_size as i32, n as i32)
            .bias(false)
            .build()
            .expect("failed to create gate linear");
        assign_weight(&mut gate.weight, weights, &format!("{prefix}.gate.weight"));

        // Load experts: try fused switch_mlp format first, fall back to per-expert.
        let switch_prefix = format!("{prefix}.switch_mlp");
        let has_switch_mlp = weights.contains_key(&format!("{switch_prefix}.gate_proj.weight"));

        let experts = if has_switch_mlp {
            // Fused 3D format: [n_experts, out, in_packed].
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
            // Per-expert format: experts.{i}.gate_proj.{weight,scales,biases}.
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

        let shared_experts = if config.n_shared_experts > 0 {
            Some(MlxQuantizedLlamaMLP::from_weights(
                weights,
                &format!("{prefix}.shared_experts"),
                qc,
            ))
        } else {
            None
        };

        Self {
            gate,
            experts,
            shared_experts,
            top_k: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
            routed_scaling_factor: config.routed_scaling_factor,
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // Evaluate x so we can read its values for routing.
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

                let weight = prob * scale * self.routed_scaling_factor;
                let vals: Vec<f32> = expert_out.as_dtype(Dtype::Float32)?.as_slice().to_vec();
                for (j, &v) in vals.iter().enumerate() {
                    output_data[tok * (hidden as usize) + j] += v * weight;
                }
            }
        }

        let mut output = Array::from_slice(&output_data, &[seq_len, hidden]);
        output = output.as_dtype(x.dtype())?;

        if let Some(ref mut shared) = self.shared_experts {
            let shared_out = shared.forward(x)?;
            output = output.add(&shared_out)?;
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedDeepSeekV2DecoderLayer
// ---------------------------------------------------------------------------

/// A single quantized DeepSeek V2 decoder layer.
struct MlxQuantizedDeepSeekV2DecoderLayer {
    self_attn: MlxQuantizedDeepSeekV2Attention,
    mlp: MlxQuantizedDeepSeekV2Mlp,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
}

enum MlxQuantizedDeepSeekV2Mlp {
    Dense(MlxQuantizedLlamaMLP),
    MoE(MlxQuantizedDeepSeekV2MoE),
}

impl MlxQuantizedDeepSeekV2DecoderLayer {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &MlxDeepSeekV2Config,
        layer_idx: usize,
        qc: &QuantConfig,
    ) -> Result<Self, Exception> {
        let mlp = if config.has_moe() && layer_idx >= config.first_k_dense_replace {
            MlxQuantizedDeepSeekV2Mlp::MoE(MlxQuantizedDeepSeekV2MoE::from_weights(
                weights,
                &format!("{prefix}.mlp"),
                config,
                qc,
            ))
        } else {
            MlxQuantizedDeepSeekV2Mlp::Dense(MlxQuantizedLlamaMLP::from_weights(
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
            self_attn: MlxQuantizedDeepSeekV2Attention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config,
                qc,
            )?,
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
            MlxQuantizedDeepSeekV2Mlp::Dense(mlp) => mlp.forward(&normed)?,
            MlxQuantizedDeepSeekV2Mlp::MoE(moe) => moe.forward(&normed)?,
        };
        hidden_states.add(&mlp_output)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedDeepSeekV2ForCausalLM
// ---------------------------------------------------------------------------

/// Quantized DeepSeek V2 for causal language modeling using MLX.
pub struct MlxQuantizedDeepSeekV2ForCausalLM {
    embed_tokens: MlxEmbedTokens,
    layers: Vec<MlxQuantizedDeepSeekV2DecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: Option<MlxLmHead>,
    tie_word_embeddings: bool,
    #[allow(dead_code)]
    config: MlxDeepSeekV2Config,
}

impl MlxQuantizedDeepSeekV2ForCausalLM {
    /// Construct a quantized DeepSeek V2 model from pre-loaded weights.
    pub fn from_weights(
        weights: &HashMap<String, Array>,
        config: &MlxDeepSeekV2Config,
        qc: &QuantConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let embed_tokens =
            MlxEmbedTokens::from_weights(weights, "model.embed_tokens", qc.group_size, qc.bits);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MlxQuantizedDeepSeekV2DecoderLayer::from_weights(
                weights,
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
                weights,
                "lm_head",
                qc.group_size,
                qc.bits,
            ))
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            tie_word_embeddings: config.tie_word_embeddings,
            config: config.clone(),
        })
    }

    pub fn load(
        model_dir: &Path,
        config: &MlxDeepSeekV2Config,
        qc: &QuantConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let weights = load_safetensors_weights(model_dir)?;
        let model = Self::from_weights(&weights, config, qc)?;
        mlx_rs::transforms::eval(weights.values())?;
        Ok(model)
    }
}

impl super::MlxModel for MlxQuantizedDeepSeekV2ForCausalLM {
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

/// Factory function for creating a quantized MLX DeepSeek V2 model.
pub fn create_mlx_quantized_deepseek_v2(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let ds_config = MlxDeepSeekV2Config::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();
    tracing::info!(
        "Loading quantized MLX DeepSeek V2 (group_size={}, bits={})",
        qc.group_size,
        qc.bits
    );
    let model = MlxQuantizedDeepSeekV2ForCausalLM::load(model_dir, &ds_config, &qc, dtype)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Kimi K2.5 factory functions (weight prefix stripping)
// ---------------------------------------------------------------------------

/// Strip a prefix from all weight names, keeping only matching entries.
fn strip_weight_prefix(weights: HashMap<String, Array>, prefix: &str) -> HashMap<String, Array> {
    weights
        .into_iter()
        .filter_map(|(name, tensor)| {
            name.strip_prefix(prefix)
                .map(|rest| (rest.to_string(), tensor))
        })
        .collect()
}

/// Factory for Kimi K2.5 float model (MLX).
///
/// Loads weights with `language_model.` prefix stripped, then delegates to
/// the standard DeepSeek V2 model construction.
pub fn create_mlx_kimi_k25(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let ds_config = MlxDeepSeekV2Config::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let mut model = MlxDeepSeekV2ForCausalLM::new(&ds_config)?;
    let weights = load_safetensors_weights(model_dir)?;
    let weights = strip_weight_prefix(weights, "language_model.");
    model.load_weights(&weights);
    mlx_rs::transforms::eval(weights.values())?;
    let _ = dtype; // dtype used by lazy eval, not construction
    Ok(Box::new(model))
}

/// Factory for Kimi K2.5 quantized model (MLX).
///
/// Loads weights with `language_model.` prefix stripped, then delegates to
/// the quantized DeepSeek V2 model construction.
pub fn create_mlx_quantized_kimi_k25(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let ds_config = MlxDeepSeekV2Config::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();
    tracing::info!(
        "Loading quantized MLX Kimi K2.5 (group_size={}, bits={})",
        qc.group_size,
        qc.bits
    );
    let weights = load_safetensors_weights(model_dir)?;
    let weights = strip_weight_prefix(weights, "language_model.");
    let model = MlxQuantizedDeepSeekV2ForCausalLM::from_weights(&weights, &ds_config, &qc)?;
    mlx_rs::transforms::eval(weights.values())?;
    let _ = dtype;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> MlxDeepSeekV2Config {
        MlxDeepSeekV2Config {
            hidden_size: 32,
            num_attention_heads: 4,
            num_kv_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 4,
            q_lora_rank: Some(16),
            kv_lora_rank: 8,
            n_routed_experts: 0,
            n_shared_experts: 0,
            num_experts_per_tok: 2,
            first_k_dense_replace: 1,
            moe_intermediate_size: 32,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
        }
    }

    #[test]
    fn test_mlx_deepseek_v2_config() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["DeepseekV2ForCausalLM"],
                "hidden_size": 5120,
                "num_attention_heads": 128,
                "num_hidden_layers": 60,
                "intermediate_size": 12288,
                "vocab_size": 102400,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
                "kv_lora_rank": 512,
                "n_routed_experts": 160,
                "n_shared_experts": 2,
                "num_experts_per_tok": 6,
                "first_k_dense_replace": 1,
                "moe_intermediate_size": 1536
            }"#,
        )
        .unwrap();

        let config = MlxDeepSeekV2Config::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.qk_nope_head_dim, 128);
        assert_eq!(config.qk_rope_head_dim, 64);
        assert_eq!(config.v_head_dim, 128);
        assert_eq!(config.kv_lora_rank, 512);
        assert_eq!(config.n_routed_experts, 160);
        assert!(config.has_moe());
        assert_eq!(config.qk_head_dim(), 192);
    }

    #[test]
    fn test_mlx_deepseek_v2_attention_forward() {
        let config = test_config();
        let mut attn = MlxDeepSeekV2Attention::new(&config).unwrap();

        let x = mlx_rs::ops::ones::<f32>(&[4, config.hidden_size as i32]).unwrap();
        let positions = Array::from_iter(0..4i32, &[4]);
        let mut cache = None;

        let out = attn.forward(&x, &positions, &mut cache).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[4, config.hidden_size as i32]);
        assert!(cache.is_some());
    }

    #[test]
    fn test_mlx_deepseek_v2_model_forward() {
        let config = test_config();
        let mut model = MlxDeepSeekV2ForCausalLM::new(&config).unwrap();

        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        let logits = <MlxDeepSeekV2ForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &input_ids,
            &positions,
            &mut kv_cache,
        )
        .unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.shape(), &[3, config.vocab_size as i32]);
    }

    #[test]
    fn test_mlx_deepseek_v2_kv_cache_prefill_decode() {
        let config = test_config();
        let mut model = MlxDeepSeekV2ForCausalLM::new(&config).unwrap();
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        // Prefill: 3 tokens
        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let logits = <MlxDeepSeekV2ForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &input_ids,
            &positions,
            &mut kv_cache,
        )
        .unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.shape(), &[3, config.vocab_size as i32]);

        // KV cache should be populated.
        for entry in &kv_cache {
            assert!(entry.is_some());
        }

        // Decode: 1 token at position 3
        let decode_ids = Array::from_iter(vec![15i32], &[1]);
        let decode_pos = Array::from_iter(vec![3i32], &[1]);
        let logits2 = <MlxDeepSeekV2ForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &decode_ids,
            &decode_pos,
            &mut kv_cache,
        )
        .unwrap();
        logits2.eval().unwrap();
        assert_eq!(logits2.shape(), &[1, config.vocab_size as i32]);
    }

    #[test]
    fn test_mlx_deepseek_v2_registry() {
        let registry = crate::models::MlxModelRegistry::default_registry();
        assert!(registry.contains("DeepseekV2ForCausalLM"));
    }

    // --- Quantized variant tests ---

    fn test_quant_config() -> QuantConfig {
        QuantConfig {
            group_size: 32,
            bits: 4,
        }
    }

    /// Config with dimensions large enough for quantization (all input dims divisible by group_size=32).
    fn test_quantized_config() -> MlxDeepSeekV2Config {
        MlxDeepSeekV2Config {
            hidden_size: 64,
            num_attention_heads: 4,
            num_kv_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 100,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            qk_nope_head_dim: 8,
            qk_rope_head_dim: 8,
            v_head_dim: 8,
            q_lora_rank: Some(32),
            kv_lora_rank: 32,
            n_routed_experts: 0,
            n_shared_experts: 0,
            num_experts_per_tok: 2,
            first_k_dense_replace: 1,
            moe_intermediate_size: 64,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
        }
    }

    /// Build a quantized DeepSeek V2 model with builder-initialized weights for testing.
    fn build_quantized_test_model(
        config: &MlxDeepSeekV2Config,
        qc: &QuantConfig,
    ) -> Result<MlxQuantizedDeepSeekV2ForCausalLM, Exception> {
        use mlx_rs::builder::Builder;
        use mlx_rs::module::Param;

        let hidden = config.hidden_size as i32;
        let qk_head_dim = config.qk_head_dim() as i32;
        let qk_nope = config.qk_nope_head_dim as i32;
        let qk_rope = config.qk_rope_head_dim as i32;
        let v_dim = config.v_head_dim as i32;
        let kv_lora_rank = config.kv_lora_rank as i32;
        let num_heads = config.num_attention_heads as i32;
        let intermediate = config.intermediate_size as i32;

        let embed_tokens = MlxEmbedTokens::Quantized(
            nn::QuantizedEmbeddingBuilder::new(config.vocab_size as i32, hidden)
                .group_size(qc.group_size)
                .bits(qc.bits)
                .build()?,
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            // Build quantized attention.
            let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) = if let Some(q_lora_rank) =
                config.q_lora_rank
            {
                let q_a = nn::QuantizedLinearBuilder::new(hidden, q_lora_rank as i32)
                    .group_size(qc.group_size)
                    .bits(qc.bits)
                    .bias(false)
                    .build()?;
                let q_a_ln = nn::RmsNormBuilder::new(q_lora_rank as i32)
                    .eps(config.rms_norm_eps)
                    .build()?;
                let q_b =
                    nn::QuantizedLinearBuilder::new(q_lora_rank as i32, num_heads * qk_head_dim)
                        .group_size(qc.group_size)
                        .bits(qc.bits)
                        .bias(false)
                        .build()?;
                (Some(q_a), Some(q_a_ln), Some(q_b), None)
            } else {
                let q = nn::QuantizedLinearBuilder::new(hidden, num_heads * qk_head_dim)
                    .group_size(qc.group_size)
                    .bits(qc.bits)
                    .bias(false)
                    .build()?;
                (None, None, None, Some(q))
            };

            let kv_a_proj_with_mqa =
                nn::QuantizedLinearBuilder::new(hidden, kv_lora_rank + qk_rope)
                    .group_size(qc.group_size)
                    .bits(qc.bits)
                    .bias(false)
                    .build()?;
            let kv_a_layernorm = nn::RmsNormBuilder::new(kv_lora_rank)
                .eps(config.rms_norm_eps)
                .build()?;
            let kv_b_proj =
                nn::QuantizedLinearBuilder::new(kv_lora_rank, num_heads * (qk_nope + v_dim))
                    .group_size(qc.group_size)
                    .bits(qc.bits)
                    .bias(false)
                    .build()?;
            let o_proj = nn::QuantizedLinearBuilder::new(num_heads * v_dim, hidden)
                .group_size(qc.group_size)
                .bits(qc.bits)
                .bias(false)
                .build()?;

            let rope = {
                let mut r = nn::Rope::new(qk_rope);
                r.base = config.rope_theta;
                r
            };

            let self_attn = MlxQuantizedDeepSeekV2Attention {
                q_a_proj,
                q_a_layernorm,
                q_b_proj,
                q_proj,
                kv_a_proj_with_mqa,
                kv_a_layernorm,
                kv_b_proj,
                o_proj,
                rope,
                num_heads: config.num_attention_heads,
                qk_nope_head_dim: config.qk_nope_head_dim,
                qk_rope_head_dim: config.qk_rope_head_dim,
                qk_head_dim: config.qk_head_dim(),
                v_head_dim: config.v_head_dim,
                kv_lora_rank: config.kv_lora_rank,
                scale: 1.0 / (config.qk_head_dim() as f32).sqrt(),
            };

            // Build quantized MLP (dense for this test config — no MoE).
            let mlp = if config.has_moe() && layer_idx >= config.first_k_dense_replace {
                // MoE path — not exercised in default test_config.
                let gate = nn::LinearBuilder::new(hidden, config.n_routed_experts as i32)
                    .bias(false)
                    .build()?;
                let moe_intermediate = config.moe_intermediate_size as i32;
                let mut experts = Vec::with_capacity(config.n_routed_experts);
                for _ in 0..config.n_routed_experts {
                    experts.push(MlxQuantizedLlamaMLP {
                        gate_proj: nn::QuantizedLinearBuilder::new(hidden, moe_intermediate)
                            .group_size(qc.group_size)
                            .bits(qc.bits)
                            .bias(false)
                            .build()?,
                        up_proj: nn::QuantizedLinearBuilder::new(hidden, moe_intermediate)
                            .group_size(qc.group_size)
                            .bits(qc.bits)
                            .bias(false)
                            .build()?,
                        down_proj: nn::QuantizedLinearBuilder::new(moe_intermediate, hidden)
                            .group_size(qc.group_size)
                            .bits(qc.bits)
                            .bias(false)
                            .build()?,
                    });
                }
                let shared_experts = if config.n_shared_experts > 0 {
                    let shared_intermediate =
                        (config.moe_intermediate_size * config.n_shared_experts) as i32;
                    Some(MlxQuantizedLlamaMLP {
                        gate_proj: nn::QuantizedLinearBuilder::new(hidden, shared_intermediate)
                            .group_size(qc.group_size)
                            .bits(qc.bits)
                            .bias(false)
                            .build()?,
                        up_proj: nn::QuantizedLinearBuilder::new(hidden, shared_intermediate)
                            .group_size(qc.group_size)
                            .bits(qc.bits)
                            .bias(false)
                            .build()?,
                        down_proj: nn::QuantizedLinearBuilder::new(shared_intermediate, hidden)
                            .group_size(qc.group_size)
                            .bits(qc.bits)
                            .bias(false)
                            .build()?,
                    })
                } else {
                    None
                };
                MlxQuantizedDeepSeekV2Mlp::MoE(MlxQuantizedDeepSeekV2MoE {
                    gate,
                    experts,
                    shared_experts,
                    top_k: config.num_experts_per_tok,
                    norm_topk_prob: config.norm_topk_prob,
                    routed_scaling_factor: config.routed_scaling_factor,
                })
            } else {
                MlxQuantizedDeepSeekV2Mlp::Dense(MlxQuantizedLlamaMLP {
                    gate_proj: nn::QuantizedLinearBuilder::new(hidden, intermediate)
                        .group_size(qc.group_size)
                        .bits(qc.bits)
                        .bias(false)
                        .build()?,
                    up_proj: nn::QuantizedLinearBuilder::new(hidden, intermediate)
                        .group_size(qc.group_size)
                        .bits(qc.bits)
                        .bias(false)
                        .build()?,
                    down_proj: nn::QuantizedLinearBuilder::new(intermediate, hidden)
                        .group_size(qc.group_size)
                        .bits(qc.bits)
                        .bias(false)
                        .build()?,
                })
            };

            layers.push(MlxQuantizedDeepSeekV2DecoderLayer {
                self_attn,
                mlp,
                input_layernorm: nn::RmsNormBuilder::new(hidden)
                    .eps(config.rms_norm_eps)
                    .build()?,
                post_attention_layernorm: nn::RmsNormBuilder::new(hidden)
                    .eps(config.rms_norm_eps)
                    .build()?,
            });
        }

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(MlxLmHead::Float(nn::Linear {
                weight: Param::new(
                    mlx_rs::ops::ones::<f32>(&[config.vocab_size as i32, hidden]).unwrap(),
                ),
                bias: Param::new(None),
            }))
        };

        Ok(MlxQuantizedDeepSeekV2ForCausalLM {
            embed_tokens,
            layers,
            norm: nn::RmsNormBuilder::new(hidden)
                .eps(config.rms_norm_eps)
                .build()?,
            lm_head,
            tie_word_embeddings: config.tie_word_embeddings,
            config: config.clone(),
        })
    }

    #[test]
    fn test_quantized_deepseek_v2_forward() {
        let config = test_quantized_config();
        let qc = test_quant_config();
        let mut model = build_quantized_test_model(&config, &qc).unwrap();

        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        let logits = <MlxQuantizedDeepSeekV2ForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &input_ids,
            &positions,
            &mut kv_cache,
        )
        .unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.shape(), &[3, config.vocab_size as i32]);
    }

    #[test]
    fn test_quantized_deepseek_v2_registry() {
        let registry = crate::models::MlxModelRegistry::default_registry();
        assert!(registry.contains_quantized("DeepseekV2ForCausalLM"));
    }
}
