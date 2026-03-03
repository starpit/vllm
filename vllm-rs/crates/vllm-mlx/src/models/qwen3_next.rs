// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Qwen3-Next model architecture for MLX.
//!
//! Hybrid GDN linear attention + full attention + MoE.
//! Both float and quantized variants.
//!
//! Port of: `vllm/model_executor/models/qwen3_next.py`

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use crate::cache::{MlxKvCache, MlxLayerKvCache};
use crate::models::gemma2::assign_gemma_norm_weight;
use crate::models::llama::{LlamaConfig, MlxLlamaMLP, assign_weight, load_safetensors_weights};
use crate::models::quantized_llama::{
    MlxEmbedTokens, MlxLmHead, MlxQuantizedLlamaMLP, QuantConfig, make_quantized_linear,
};
use crate::models::qwen3_moe::{MlxQuantizedQwen3MoeMoE, MlxQwen3MoeConfig, MlxQwen3MoeMoE};
use vllm_model::weight::HfModelConfig;

// ---------------------------------------------------------------------------
// MlxQwen3NextConfig
// ---------------------------------------------------------------------------

/// Parsed configuration for a Qwen3-Next model (MLX backend).
#[derive(Debug, Clone)]
pub struct MlxQwen3NextConfig {
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
    pub partial_rotary_factor: f64,

    // GDN fields.
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,

    // MoE fields.
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub decoder_sparse_step: usize,
    pub mlp_only_layers: Vec<usize>,

    pub layer_types: Vec<String>,
}

impl MlxQwen3NextConfig {
    pub fn from_hf_config(config: &HfModelConfig) -> Result<Self, String> {
        let llama = LlamaConfig::from_hf_config(config)?;
        let extra = &config.extra;

        let get_usize = |key: &str| -> Option<usize> {
            extra.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
        };
        let get_f64 = |key: &str| -> Option<f64> { extra.get(key).and_then(|v| v.as_f64()) };
        let get_bool = |key: &str| -> Option<bool> { extra.get(key).and_then(|v| v.as_bool()) };

        let partial_rotary_factor = get_f64("partial_rotary_factor").unwrap_or_else(|| {
            extra
                .get("rope_parameters")
                .and_then(|rp| rp.get("partial_rotary_factor"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.25)
        });

        let rope_theta = extra
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(llama.rope_theta as f64) as f32;

        let num_hidden_layers = llama.num_hidden_layers;

        let layer_types = if let Some(arr) = extra.get("layer_types").and_then(|v| v.as_array()) {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        } else {
            (0..num_hidden_layers)
                .map(|i| {
                    if (i + 1) % 4 == 0 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect()
        };

        let head_dim = config
            .head_dim()
            .unwrap_or(llama.hidden_size / llama.num_attention_heads);

        Ok(Self {
            hidden_size: llama.hidden_size,
            num_attention_heads: llama.num_attention_heads,
            num_kv_heads: llama.num_kv_heads,
            num_hidden_layers,
            intermediate_size: llama.intermediate_size,
            vocab_size: llama.vocab_size,
            max_position_embeddings: llama.max_position_embeddings,
            rms_norm_eps: llama.rms_norm_eps,
            rope_theta,
            head_dim,
            tie_word_embeddings: llama.tie_word_embeddings,
            partial_rotary_factor,
            linear_conv_kernel_dim: get_usize("linear_conv_kernel_dim").unwrap_or(4),
            linear_key_head_dim: get_usize("linear_key_head_dim").unwrap_or(128),
            linear_value_head_dim: get_usize("linear_value_head_dim").unwrap_or(128),
            linear_num_key_heads: get_usize("linear_num_key_heads").unwrap_or(16),
            linear_num_value_heads: get_usize("linear_num_value_heads").unwrap_or(32),
            num_experts: get_usize("num_experts").unwrap_or(0),
            num_experts_per_tok: get_usize("num_experts_per_tok").unwrap_or(10),
            moe_intermediate_size: get_usize("moe_intermediate_size").unwrap_or(512),
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
            layer_types,
        })
    }

    fn is_full_attention(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .is_some_and(|t| t == "full_attention")
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        !self.mlp_only_layers.contains(&layer_idx)
            && self.num_experts > 0
            && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }

    fn key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    fn value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    #[allow(dead_code)]
    fn llama_config(&self) -> LlamaConfig {
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
            partial_rotary_factor: self.partial_rotary_factor,
            long_rope_scaling: None,
        }
    }

    fn moe_config(&self) -> MlxQwen3MoeConfig {
        MlxQwen3MoeConfig {
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
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            moe_intermediate_size: self.moe_intermediate_size,
            shared_expert_intermediate_size: self.shared_expert_intermediate_size,
            norm_topk_prob: self.norm_topk_prob,
            decoder_sparse_step: self.decoder_sparse_step,
            mlp_only_layers: self.mlp_only_layers.clone(),
        }
    }

    fn num_full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| t.as_str() == "full_attention")
            .count()
    }
}

// ===========================================================================
// Float variants
// ===========================================================================

// ---------------------------------------------------------------------------
// MlxQwen3NextAttention (full attention with output gating)
// ---------------------------------------------------------------------------

/// Full attention with output gating for Qwen3-Next (MLX float).
struct MlxQwen3NextAttention {
    /// Q projection (doubled output: q + gate).
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    o_proj: nn::Linear,
    q_norm: nn::RmsNorm,
    k_norm: nn::RmsNorm,
    rope: nn::Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    scale: f32,
}

impl MlxQwen3NextAttention {
    fn new(config: &MlxQwen3NextConfig) -> Result<Self, Exception> {
        let hidden = config.hidden_size as i32;
        let q_size = (config.num_attention_heads * config.head_dim * 2) as i32; // doubled for gate
        let kv_size = (config.num_kv_heads * config.head_dim) as i32;
        let o_size = (config.num_attention_heads * config.head_dim) as i32;

        let rotary_dim = (config.head_dim as f64 * config.partial_rotary_factor).round() as usize;
        let rotary_dim = rotary_dim - (rotary_dim % 2);

        Ok(Self {
            q_proj: nn::LinearBuilder::new(hidden, q_size).bias(false).build()?,
            k_proj: nn::LinearBuilder::new(hidden, kv_size)
                .bias(false)
                .build()?,
            v_proj: nn::LinearBuilder::new(hidden, kv_size)
                .bias(false)
                .build()?,
            o_proj: nn::LinearBuilder::new(o_size, hidden).bias(false).build()?,
            q_norm: nn::RmsNormBuilder::new(config.head_dim as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            k_norm: nn::RmsNormBuilder::new(config.head_dim as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            rope: {
                let mut r = nn::Rope::new(rotary_dim as i32);
                r.base = config.rope_theta;
                r
            },
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            rotary_dim,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
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
        // GemmaRMSNorm: weight+1 for QK norms.
        assign_gemma_norm_weight(
            &mut self.q_norm,
            weights,
            &format!("{prefix}.q_norm.weight"),
        );
        assign_gemma_norm_weight(
            &mut self.k_norm,
            weights,
            &format!("{prefix}.k_norm.weight"),
        );
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        _positions: &Array,
        cache: &mut Option<MlxLayerKvCache>,
        rope_offset: i32,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);

        // Q projection (doubled for gate).
        let q_gate = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Split q_gate into q and gate: [seq, 2*num_heads*head_dim].
        let q_gate =
            q_gate.reshape(&[seq_len, self.num_heads as i32, (2 * self.head_dim) as i32])?;
        let q = q_gate.try_index((.., .., ..self.head_dim as i32))?;
        let gate = q_gate.try_index((.., .., self.head_dim as i32..))?;

        // Reshape K.
        let k = k.reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?;

        // Apply GemmaRMSNorm QK norms.
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Transpose to [1, heads, seq, head_dim] for attention.
        let q = q.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let v = v
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        // Apply partial RoPE (offset threaded from caller — no sync needed).
        let offset = rope_offset;

        // RoPE only applies to the first rotary_dim dimensions (handled by nn::Rope
        // which was initialized with rotary_dim). For partial RoPE, we split, apply, concat.
        let q = if self.rotary_dim < self.head_dim {
            let q_rot = q.try_index((.., .., .., ..self.rotary_dim as i32))?;
            let q_pass = q.try_index((.., .., .., self.rotary_dim as i32..))?;
            let q_rot = self.rope.forward((&q_rot, offset))?;
            mlx_rs::ops::concatenate_axis(&[q_rot, q_pass], -1)?
        } else {
            self.rope.forward((&q, offset))?
        };

        let k = if self.rotary_dim < self.head_dim {
            let k_rot = k.try_index((.., .., .., ..self.rotary_dim as i32))?;
            let k_pass = k.try_index((.., .., .., self.rotary_dim as i32..))?;
            let k_rot = self.rope.forward((&k_rot, offset))?;
            mlx_rs::ops::concatenate_axis(&[k_rot, k_pass], -1)?
        } else {
            self.rope.forward((&k, offset))?
        };

        // KV cache update — pre-allocated buffer with O(1) slice_update.
        let (k, v) = crate::cache::kv_cache_update(cache, &k, &v)?;

        // Scaled dot-product attention.
        let mask = if seq_len > 1 {
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let attn_out = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;

        // Reshape: [1, heads, seq, head_dim] -> [seq, heads, head_dim].
        let attn_out = attn_out.squeeze_axes(&[0])?.transpose_axes(&[1, 0, 2])?;

        // Output gating: sigmoid(gate) * attn_output.
        let gate_sigmoid = nn::sigmoid(&gate)?;
        let gated = attn_out.multiply(&gate_sigmoid)?;

        // Flatten and project.
        let hidden = (self.num_heads * self.head_dim) as i32;
        let gated = gated.reshape(&[seq_len, hidden])?;
        self.o_proj.forward(&gated)
    }
}

// ---------------------------------------------------------------------------
// MlxGatedDeltaNet (GDN linear attention, float)
// ---------------------------------------------------------------------------

/// GDN linear attention for MLX (float variant).
struct MlxGatedDeltaNet {
    in_proj_qkvz: nn::Linear,
    in_proj_ba: nn::Linear,
    conv1d_weight: Param<Array>,
    a_log: Param<Array>,
    dt_bias: Param<Array>,
    norm_weight: Param<Array>,
    out_proj: nn::Linear,
    norm_eps: f32,

    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    conv_kernel_size: usize,

    conv_state: RefCell<Option<Array>>,
    ssm_state: RefCell<Option<Array>>,
}

impl MlxGatedDeltaNet {
    fn new(config: &MlxQwen3NextConfig) -> Result<Self, Exception> {
        let hidden = config.hidden_size as i32;
        let key_dim = config.key_dim();
        let value_dim = config.value_dim();
        let conv_dim = config.conv_dim();

        Ok(Self {
            in_proj_qkvz: nn::LinearBuilder::new(hidden, (2 * key_dim + 2 * value_dim) as i32)
                .bias(false)
                .build()?,
            in_proj_ba: nn::LinearBuilder::new(hidden, (2 * config.linear_num_value_heads) as i32)
                .bias(false)
                .build()?,
            conv1d_weight: Param::new(Array::zeros::<f32>(&[
                conv_dim as i32,
                config.linear_conv_kernel_dim as i32,
            ])?),
            a_log: Param::new(Array::zeros::<f32>(
                &[config.linear_num_value_heads as i32],
            )?),
            dt_bias: Param::new(Array::zeros::<f32>(
                &[config.linear_num_value_heads as i32],
            )?),
            norm_weight: Param::new(Array::ones::<f32>(&[config.linear_value_head_dim as i32])?),
            out_proj: nn::LinearBuilder::new(value_dim as i32, hidden)
                .bias(false)
                .build()?,
            norm_eps: config.rms_norm_eps,
            num_k_heads: config.linear_num_key_heads,
            num_v_heads: config.linear_num_value_heads,
            head_k_dim: config.linear_key_head_dim,
            head_v_dim: config.linear_value_head_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size: config.linear_conv_kernel_dim,
            conv_state: RefCell::new(None),
            ssm_state: RefCell::new(None),
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(
            &mut self.in_proj_qkvz.weight,
            weights,
            &format!("{prefix}.in_proj_qkvz.weight"),
        );
        assign_weight(
            &mut self.in_proj_ba.weight,
            weights,
            &format!("{prefix}.in_proj_ba.weight"),
        );

        // Conv1d weight: may be [conv_dim, 1, kernel_size] → squeeze to [conv_dim, kernel_size].
        if let Some(w) = weights.get(&format!("{prefix}.conv1d.weight")) {
            let w = if w.ndim() == 3 {
                w.squeeze_axes(&[1]).unwrap_or_else(|_| w.clone())
            } else {
                w.clone()
            };
            self.conv1d_weight.value = w;
        }

        if let Some(w) = weights.get(&format!("{prefix}.A_log")) {
            self.a_log.value = w.clone();
        }
        if let Some(w) = weights.get(&format!("{prefix}.dt_bias")) {
            self.dt_bias.value = w.clone();
        }
        if let Some(w) = weights.get(&format!("{prefix}.norm.weight")) {
            self.norm_weight.value = w.clone();
        }
        assign_weight(
            &mut self.out_proj.weight,
            weights,
            &format!("{prefix}.out_proj.weight"),
        );
    }

    fn reset_state(&self) {
        *self.conv_state.borrow_mut() = None;
        *self.ssm_state.borrow_mut() = None;
    }

    fn extract_state(&self) -> Option<(Array, Array)> {
        let conv = self.conv_state.borrow_mut().take();
        let ssm = self.ssm_state.borrow_mut().take();
        match (conv, ssm) {
            (Some(c), Some(s)) => Some((c, s)),
            _ => None,
        }
    }

    fn inject_state(&self, state: &Option<(Array, Array)>) {
        match state {
            Some((c, s)) => {
                *self.conv_state.borrow_mut() = Some(c.clone());
                *self.ssm_state.borrow_mut() = Some(s.clone());
            }
            None => {
                *self.conv_state.borrow_mut() = None;
                *self.ssm_state.borrow_mut() = None;
            }
        }
    }

    fn forward(&mut self, hidden_states: &Array) -> Result<Array, Exception> {
        // Evaluate input for shape queries.
        mlx_rs::transforms::eval(std::iter::once(hidden_states))?;
        let num_tokens = hidden_states.dim(0) as usize;
        let dtype = hidden_states.dtype();

        // --- 1. Input projections ---
        let proj_qkvz = self.in_proj_qkvz.forward(hidden_states)?;
        let proj_ba = self.in_proj_ba.forward(hidden_states)?;

        // Split qkvz grouped by k-heads.
        let v_per_k = self.num_v_heads / self.num_k_heads;
        let per_group = self.head_k_dim
            + self.head_k_dim
            + v_per_k * self.head_v_dim
            + v_per_k * self.head_v_dim;
        let proj_qkvz =
            proj_qkvz.reshape(&[num_tokens as i32, self.num_k_heads as i32, per_group as i32])?;

        let hk = self.head_k_dim as i32;
        let q_grouped = proj_qkvz.try_index((.., .., ..hk))?;
        let k_grouped = proj_qkvz.try_index((.., .., hk..2 * hk))?;
        let v_end = 2 * hk + (v_per_k * self.head_v_dim) as i32;
        let v_grouped = proj_qkvz.try_index((.., .., 2 * hk..v_end))?;
        let z_grouped = proj_qkvz.try_index((.., .., v_end..))?;

        let q_flat = q_grouped.reshape(&[num_tokens as i32, self.key_dim as i32])?;
        let k_flat = k_grouped.reshape(&[num_tokens as i32, self.key_dim as i32])?;
        let v_flat = v_grouped.reshape(&[num_tokens as i32, self.value_dim as i32])?;
        let z = z_grouped.reshape(&[num_tokens as i32, self.value_dim as i32])?;

        // Split ba.
        let proj_ba = proj_ba.reshape(&[
            num_tokens as i32,
            self.num_k_heads as i32,
            (2 * v_per_k) as i32,
        ])?;
        let b = proj_ba
            .try_index((.., .., ..v_per_k as i32))?
            .reshape(&[num_tokens as i32, self.num_v_heads as i32])?;
        let a = proj_ba
            .try_index((.., .., v_per_k as i32..))?
            .reshape(&[num_tokens as i32, self.num_v_heads as i32])?;

        // Concatenate q, k, v for conv1d.
        let mixed_qkv = mlx_rs::ops::concatenate_axis(&[q_flat, k_flat, v_flat], 1)?;

        // --- 2. Causal conv1d + SiLU ---
        // Evaluate for the conv loop.
        mlx_rs::transforms::eval(std::iter::once(&mixed_qkv))?;
        let conv_out = self.causal_conv1d(&mixed_qkv, num_tokens)?;

        // --- 3. Split conv output ---
        let q_conv = conv_out.try_index((.., ..self.key_dim as i32))?;
        let k_conv = conv_out.try_index((.., self.key_dim as i32..(2 * self.key_dim) as i32))?;
        let v_conv = conv_out.try_index((.., (2 * self.key_dim) as i32..))?;

        // --- 4. Gating ---
        let a_plus_bias = a.add(&self.dt_bias)?;
        // softplus(x) = log(1 + exp(x))
        mlx_rs::transforms::eval(std::iter::once(&a_plus_bias))?;
        let sp = mlx_rs::ops::log(&a_plus_bias.exp()?.add(Array::from_f32(1.0))?)?;
        let a_exp = self.a_log.as_dtype(Dtype::Float32)?.exp()?;
        let g = sp.as_dtype(Dtype::Float32)?.multiply(&a_exp)?.negative()?;
        let g = g.as_dtype(dtype)?;
        let beta = nn::sigmoid(&b)?;

        // --- 5. Recurrence ---
        mlx_rs::transforms::eval([&q_conv, &k_conv, &v_conv, &g, &beta].iter().copied())?;
        let output =
            self.gated_delta_recurrence(&q_conv, &k_conv, &v_conv, &g, &beta, num_tokens)?;

        // --- 6. RMSNormGated ---
        let z = z.reshape(&[
            num_tokens as i32,
            self.num_v_heads as i32,
            self.head_v_dim as i32,
        ])?;
        let normed = self.rms_norm_gated(&output, &z)?;

        // --- 7. Output projection ---
        let normed_flat = normed.reshape(&[num_tokens as i32, self.value_dim as i32])?;
        self.out_proj.forward(&normed_flat)
    }

    fn causal_conv1d(&self, mixed_qkv: &Array, num_tokens: usize) -> Result<Array, Exception> {
        let k = self.conv_kernel_size;
        let mut conv_st = self.conv_state.borrow_mut();

        let pad = match conv_st.take() {
            Some(s) => s,
            None => Array::zeros::<f32>(&[(k - 1) as i32, self.conv_dim as i32])?,
        };

        let padded = mlx_rs::ops::concatenate_axis(&[pad, mixed_qkv.clone()], 0)?;
        mlx_rs::transforms::eval(std::iter::once(&padded))?;

        let mut outputs = Vec::with_capacity(num_tokens);
        for t in 0..num_tokens {
            let window = padded.try_index(t as i32..(t + k) as i32)?;
            let window_t = window.transpose_axes(&[1, 0])?; // [D, k]
            let out = window_t.multiply(&self.conv1d_weight)?.sum_axis(-1, None)?;
            outputs.push(out);
        }
        let output = mlx_rs::ops::stack_axis(&outputs, 0)?;
        let output = nn::silu(&output)?;

        // Update conv state.
        let start = num_tokens.saturating_sub(k - 1);
        let len = num_tokens.min(k - 1);
        *conv_st = Some(mixed_qkv.try_index(start as i32..(start + len) as i32)?);

        Ok(output)
    }

    fn gated_delta_recurrence(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        num_tokens: usize,
    ) -> Result<Array, Exception> {
        let mut ssm_st = self.ssm_state.borrow_mut();
        let mut state = match ssm_st.take() {
            Some(s) => s,
            None => Array::zeros::<f32>(&[
                self.num_v_heads as i32,
                self.head_v_dim as i32,
                self.head_k_dim as i32,
            ])?,
        };

        let q = q.reshape(&[
            num_tokens as i32,
            self.num_k_heads as i32,
            self.head_k_dim as i32,
        ])?;
        let k = k.reshape(&[
            num_tokens as i32,
            self.num_k_heads as i32,
            self.head_k_dim as i32,
        ])?;
        let v = v.reshape(&[
            num_tokens as i32,
            self.num_v_heads as i32,
            self.head_v_dim as i32,
        ])?;

        let mut outputs = Vec::with_capacity(num_tokens);

        for t in 0..num_tokens {
            let q_t = q.try_index(t as i32)?; // [n_k, hk]
            let k_t = k.try_index(t as i32)?;
            let v_t = v.try_index(t as i32)?; // [n_v, hv]
            let g_t = g.try_index(t as i32)?; // [n_v]
            let beta_t = beta.try_index(t as i32)?;

            // L2 normalize q, k per-head.
            let q_norm = mlx_rs::ops::sqrt(&q_t.square()?.sum_axis(-1, true)?)?;
            let q_t = q_t.divide(&q_norm.add(Array::from_f32(1e-12))?)?;
            let k_norm = mlx_rs::ops::sqrt(&k_t.square()?.sum_axis(-1, true)?)?;
            let k_t = k_t.divide(&k_norm.add(Array::from_f32(1e-12))?)?;

            let q_f32 = q_t.as_dtype(Dtype::Float32)?;
            let k_f32 = k_t.as_dtype(Dtype::Float32)?;
            let v_f32 = v_t.as_dtype(Dtype::Float32)?;
            let g_f32 = g_t.as_dtype(Dtype::Float32)?;
            let beta_f32 = beta_t.as_dtype(Dtype::Float32)?;

            let mut head_outputs = Vec::with_capacity(self.num_v_heads);

            for h_v in 0..self.num_v_heads {
                let h_k = h_v * self.num_k_heads / self.num_v_heads;

                let g_h = g_f32.try_index(h_v as i32)?.exp()?;
                let beta_h = beta_f32.try_index(h_v as i32)?;
                let k_head = k_f32.try_index(h_k as i32)?; // [hk]
                let v_head = v_f32.try_index(h_v as i32)?; // [hv]

                // outer(v, k) = v[:, None] * k[None, :] → [hv, hk]
                let v_col = v_head.reshape(&[self.head_v_dim as i32, 1])?;
                let k_row = k_head.reshape(&[1, self.head_k_dim as i32])?;
                let outer = v_col.multiply(&k_row)?;

                // S[h] = exp(g_h) * S[h] + beta_h * outer
                let s_h = state.try_index(h_v as i32)?;
                let new_s = s_h.multiply(&g_h)?.add(&outer.multiply(&beta_h)?)?;

                // o[h] = S[h] @ q[h_k] → [hv]
                let q_head = q_f32.try_index(h_k as i32)?;
                let q_col = q_head.reshape(&[self.head_k_dim as i32, 1])?;
                let o_h = mlx_rs::ops::matmul(&new_s, &q_col)?.squeeze_axes(&[-1])?;

                // Write back state using index assignment isn't trivial in MLX,
                // so we'll rebuild the state tensor after processing all heads.
                head_outputs.push((h_v, o_h, new_s));
            }

            // Rebuild state and output.
            let mut state_slices = Vec::with_capacity(self.num_v_heads);
            let mut out_slices = Vec::with_capacity(self.num_v_heads);
            for (_, o_h, new_s) in &head_outputs {
                state_slices.push(new_s.expand_dims(0)?);
                out_slices.push(o_h.as_dtype(v.dtype())?.expand_dims(0)?);
            }
            state = mlx_rs::ops::concatenate_axis(&state_slices, 0)?;
            let token_out = mlx_rs::ops::concatenate_axis(&out_slices, 0)?; // [n_v, hv]
            outputs.push(token_out);

            // Eval periodically to avoid graph explosion.
            if (t + 1) % 32 == 0 || t + 1 == num_tokens {
                mlx_rs::transforms::eval(std::iter::once(&state))?;
            }
        }

        *ssm_st = Some(state);
        let result = mlx_rs::ops::stack_axis(&outputs, 0)?; // [seq, n_v, hv]
        Ok(result)
    }

    fn rms_norm_gated(&self, x: &Array, z: &Array) -> Result<Array, Exception> {
        // RMS norm on last dim.
        let x_f32 = x.as_dtype(Dtype::Float32)?;
        let variance = x_f32.square()?.mean_axis(-1, true)?;
        let rsqrt = mlx_rs::ops::rsqrt(&variance.add(Array::from_f32(self.norm_eps))?)?;
        let normed = x_f32.multiply(&rsqrt)?.as_dtype(x.dtype())?;
        let normed = normed.multiply(&self.norm_weight)?;
        let z_sigmoid = nn::sigmoid(z)?;
        normed.multiply(&z_sigmoid)
    }
}

// ---------------------------------------------------------------------------
// MlxQwen3NextDecoderLayer (float)
// ---------------------------------------------------------------------------

enum MlxQwen3NextAttnVariant {
    FullAttention(MlxQwen3NextAttention),
    LinearAttention(MlxGatedDeltaNet),
}

enum MlxQwen3NextMlpVariant {
    Dense(MlxLlamaMLP),
    MoE(MlxQwen3MoeMoE),
}

struct MlxQwen3NextDecoderLayer {
    attn: MlxQwen3NextAttnVariant,
    mlp: MlxQwen3NextMlpVariant,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
    is_full_attn: bool,
}

impl MlxQwen3NextDecoderLayer {
    fn new(config: &MlxQwen3NextConfig, layer_idx: usize) -> Result<Self, Exception> {
        let is_full_attn = config.is_full_attention(layer_idx);
        let attn = if is_full_attn {
            MlxQwen3NextAttnVariant::FullAttention(MlxQwen3NextAttention::new(config)?)
        } else {
            MlxQwen3NextAttnVariant::LinearAttention(MlxGatedDeltaNet::new(config)?)
        };

        let moe_config = config.moe_config();
        let mlp = if config.is_moe_layer(layer_idx) {
            MlxQwen3NextMlpVariant::MoE(MlxQwen3MoeMoE::new(&moe_config)?)
        } else {
            MlxQwen3NextMlpVariant::Dense(MlxLlamaMLP::new(
                config.hidden_size as i32,
                config.intermediate_size as i32,
            )?)
        };

        Ok(Self {
            attn,
            mlp,
            input_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            is_full_attn,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        match &mut self.attn {
            MlxQwen3NextAttnVariant::FullAttention(attn) => {
                attn.load_weights(weights, &format!("{prefix}.self_attn"));
            }
            MlxQwen3NextAttnVariant::LinearAttention(gdn) => {
                gdn.load_weights(weights, &format!("{prefix}.linear_attn"));
            }
        }
        match &mut self.mlp {
            MlxQwen3NextMlpVariant::Dense(mlp) => {
                mlp.load_weights(weights, &format!("{prefix}.mlp"));
            }
            MlxQwen3NextMlpVariant::MoE(moe) => {
                moe.load_weights(weights, &format!("{prefix}.mlp"));
            }
        }
        // GemmaRMSNorm: weight+1.
        assign_gemma_norm_weight(
            &mut self.input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm.weight"),
        );
        assign_gemma_norm_weight(
            &mut self.post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        );
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<MlxLayerKvCache>,
        rope_offset: i32,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = match &mut self.attn {
            MlxQwen3NextAttnVariant::FullAttention(attn) => {
                attn.forward(&normed, positions, cache, rope_offset)?
            }
            MlxQwen3NextAttnVariant::LinearAttention(gdn) => gdn.forward(&normed)?,
        };
        let hidden_states = hidden_states.add(&attn_output)?;

        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = match &mut self.mlp {
            MlxQwen3NextMlpVariant::Dense(mlp) => mlp.forward(&normed)?,
            MlxQwen3NextMlpVariant::MoE(moe) => moe.forward(&normed)?,
        };
        hidden_states.add(&mlp_output)
    }

    fn reset_recurrent_state(&self) {
        if let MlxQwen3NextAttnVariant::LinearAttention(gdn) = &self.attn {
            gdn.reset_state();
        }
    }

    fn extract_recurrent_state(&self) -> Option<Option<(Array, Array)>> {
        match &self.attn {
            MlxQwen3NextAttnVariant::LinearAttention(gdn) => Some(gdn.extract_state()),
            MlxQwen3NextAttnVariant::FullAttention(_) => None,
        }
    }

    fn inject_recurrent_state(&self, state: &Option<(Array, Array)>) {
        if let MlxQwen3NextAttnVariant::LinearAttention(gdn) = &self.attn {
            gdn.inject_state(state);
        }
    }
}

// ---------------------------------------------------------------------------
// MlxQwen3NextForCausalLM (float)
// ---------------------------------------------------------------------------

/// Qwen3-Next for causal language modeling using MLX (float).
pub struct MlxQwen3NextForCausalLM {
    embed_tokens: nn::Embedding,
    layers: Vec<MlxQwen3NextDecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: Option<nn::Linear>,
    tie_word_embeddings: bool,
    num_attn_layers: usize,
    #[allow(dead_code)]
    config: MlxQwen3NextConfig,
}

impl MlxQwen3NextForCausalLM {
    pub fn load(
        model_dir: &Path,
        config: &MlxQwen3NextConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MlxQwen3NextDecoderLayer::new(config, i)?);
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
            num_attn_layers: config.num_full_attention_layers(),
            config: config.clone(),
        };

        let weights = load_safetensors_weights(model_dir)?;
        assign_weight(
            &mut model.embed_tokens.weight,
            &weights,
            "model.embed_tokens.weight",
        );
        for (i, layer) in model.layers.iter_mut().enumerate() {
            layer.load_weights(&weights, &format!("model.layers.{i}"));
        }
        assign_gemma_norm_weight(&mut model.norm, &weights, "model.norm.weight");
        if let Some(ref mut lm_head) = model.lm_head {
            assign_weight(&mut lm_head.weight, &weights, "lm_head.weight");
        }

        mlx_rs::transforms::eval(weights.values())?;
        Ok(model)
    }

    /// Reset all GDN recurrent state.
    pub fn reset_recurrent_state(&self) {
        for layer in &self.layers {
            layer.reset_recurrent_state();
        }
    }
}

impl super::MlxModel for MlxQwen3NextForCausalLM {
    fn forward(
        &mut self,
        input_ids: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        let offset = rope_offset.unwrap_or(0);
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Only full attention layers use KV cache slots.
        let mut kv_slot = 0;
        for layer in self.layers.iter_mut() {
            if layer.is_full_attn {
                hidden_states =
                    layer.forward(&hidden_states, positions, &mut kv_cache[kv_slot], offset)?;
                kv_slot += 1;
            } else {
                let mut dummy_cache = None;
                hidden_states =
                    layer.forward(&hidden_states, positions, &mut dummy_cache, offset)?;
            }
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        let logits = if self.tie_word_embeddings {
            self.embed_tokens.as_linear(&hidden_states)?
        } else {
            self.lm_head.as_mut().unwrap().forward(&hidden_states)?
        };

        Ok(logits)
    }

    fn num_layers(&self) -> usize {
        self.num_attn_layers
    }

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut kv_cache: MlxKvCache = (0..self.num_attn_layers).map(|_| None).collect();
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        let mut kv_slot = 0;
        for layer in self.layers.iter_mut() {
            if layer.is_full_attn {
                hidden_states =
                    layer.forward(&hidden_states, positions, &mut kv_cache[kv_slot], 0)?;
                kv_slot += 1;
            } else {
                let mut dummy_cache = None;
                hidden_states = layer.forward(&hidden_states, positions, &mut dummy_cache, 0)?;
            }
        }
        self.norm.forward(&hidden_states)
    }

    fn reset_recurrent_state(&self) {
        for layer in &self.layers {
            layer.reset_recurrent_state();
        }
    }

    fn num_recurrent_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l.attn, MlxQwen3NextAttnVariant::LinearAttention(_)))
            .count()
    }

    fn extract_recurrent_state(&self) -> super::MlxRecurrentState {
        self.layers
            .iter()
            .filter_map(|l| l.extract_recurrent_state())
            .collect()
    }

    fn inject_recurrent_state(&self, state: &[Option<(Array, Array)>]) {
        let mut idx = 0;
        for layer in &self.layers {
            if matches!(layer.attn, MlxQwen3NextAttnVariant::LinearAttention(_)) {
                if let Some(s) = state.get(idx) {
                    layer.inject_recurrent_state(s);
                }
                idx += 1;
            }
        }
    }
}

// ===========================================================================
// Quantized variants
// ===========================================================================

// ---------------------------------------------------------------------------
// MlxQuantizedQwen3NextAttention
// ---------------------------------------------------------------------------

/// Quantized full attention with output gating for Qwen3-Next (MLX).
struct MlxQuantizedQwen3NextAttention {
    q_proj: nn::QuantizedLinear,
    k_proj: nn::QuantizedLinear,
    v_proj: nn::QuantizedLinear,
    o_proj: nn::QuantizedLinear,
    q_norm: nn::RmsNorm,
    k_norm: nn::RmsNorm,
    rope: nn::Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    scale: f32,
}

impl MlxQuantizedQwen3NextAttention {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &MlxQwen3NextConfig,
        qc: &QuantConfig,
    ) -> Self {
        let q_proj =
            make_quantized_linear(weights, &format!("{prefix}.q_proj"), qc.group_size, qc.bits);
        let k_proj =
            make_quantized_linear(weights, &format!("{prefix}.k_proj"), qc.group_size, qc.bits);
        let v_proj =
            make_quantized_linear(weights, &format!("{prefix}.v_proj"), qc.group_size, qc.bits);
        let o_proj =
            make_quantized_linear(weights, &format!("{prefix}.o_proj"), qc.group_size, qc.bits);

        // QK norms: GemmaRMSNorm (weight+1).
        let mut q_norm = nn::RmsNormBuilder::new(config.head_dim as i32)
            .eps(config.rms_norm_eps)
            .build()
            .unwrap();
        assign_gemma_norm_weight(&mut q_norm, weights, &format!("{prefix}.q_norm.weight"));
        let mut k_norm = nn::RmsNormBuilder::new(config.head_dim as i32)
            .eps(config.rms_norm_eps)
            .build()
            .unwrap();
        assign_gemma_norm_weight(&mut k_norm, weights, &format!("{prefix}.k_norm.weight"));

        let rotary_dim = {
            let d = (config.head_dim as f64 * config.partial_rotary_factor).round() as usize;
            d - (d % 2)
        };

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope: {
                let mut r = nn::Rope::new(rotary_dim as i32);
                r.base = config.rope_theta;
                r
            },
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            rotary_dim,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        }
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        _positions: &Array,
        cache: &mut Option<MlxLayerKvCache>,
        rope_offset: i32,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);
        let n_h = self.num_heads as i32;
        let n_kv = self.num_kv_heads as i32;
        let hd = self.head_dim as i32;

        // Q projection (doubled for gate).
        let q_gate = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Split q_gate into q and gate: [seq, num_heads, 2*head_dim].
        let q_gate = q_gate.reshape(&[seq_len, n_h, 2 * hd])?;
        let q = q_gate.try_index((.., .., ..hd))?;
        let gate = q_gate.try_index((.., .., hd..))?;

        // Reshape K.
        let k = k.reshape(&[seq_len, n_kv, hd])?;

        // QK norms (GemmaRMSNorm).
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Transpose to [1, heads, seq, head_dim] for attention.
        let q = q.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let v = v
            .reshape(&[seq_len, n_kv, hd])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        // Apply partial RoPE (offset threaded from caller — no sync needed).
        let offset = rope_offset;

        let rd = self.rotary_dim as i32;
        let q = if self.rotary_dim < self.head_dim {
            let q_rot = q.try_index((.., .., .., ..rd))?;
            let q_pass = q.try_index((.., .., .., rd..))?;
            let q_rot = self.rope.forward((&q_rot, offset))?;
            mlx_rs::ops::concatenate_axis(&[q_rot, q_pass], -1)?
        } else {
            self.rope.forward((&q, offset))?
        };

        let k = if self.rotary_dim < self.head_dim {
            let k_rot = k.try_index((.., .., .., ..rd))?;
            let k_pass = k.try_index((.., .., .., rd..))?;
            let k_rot = self.rope.forward((&k_rot, offset))?;
            mlx_rs::ops::concatenate_axis(&[k_rot, k_pass], -1)?
        } else {
            self.rope.forward((&k, offset))?
        };

        // KV cache update — pre-allocated buffer with O(1) slice_update.
        let (k, v) = crate::cache::kv_cache_update(cache, &k, &v)?;

        // Scaled dot-product attention.
        let mask = if seq_len > 1 {
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let attn_out = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;

        // Reshape: [1, heads, seq, head_dim] -> [seq, heads, head_dim].
        let attn_out = attn_out.squeeze_axes(&[0])?.transpose_axes(&[1, 0, 2])?;

        // Output gating: sigmoid(gate) * attn_output.
        let gate_sigmoid = nn::sigmoid(&gate)?;
        let gated = attn_out.multiply(&gate_sigmoid)?;

        // Flatten and project.
        let hidden = n_h * hd;
        let gated = gated.reshape(&[seq_len, hidden])?;
        self.o_proj.forward(&gated)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedGatedDeltaNet
// ---------------------------------------------------------------------------

/// Quantized GDN linear attention layer for Qwen3-Next (MLX).
struct MlxQuantizedGatedDeltaNet {
    in_proj_qkvz: nn::QuantizedLinear,
    in_proj_ba: nn::QuantizedLinear,
    conv1d_weight: Param<Array>,
    a_log: Param<Array>,
    dt_bias: Param<Array>,
    norm_weight: Param<Array>,
    norm_eps: f32,
    out_proj: nn::QuantizedLinear,

    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    conv_kernel_size: usize,

    conv_state: RefCell<Option<Array>>,
    ssm_state: RefCell<Option<Array>>,
}

impl MlxQuantizedGatedDeltaNet {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &MlxQwen3NextConfig,
        qc: &QuantConfig,
    ) -> Self {
        let in_proj_qkvz = make_quantized_linear(
            weights,
            &format!("{prefix}.in_proj_qkvz"),
            qc.group_size,
            qc.bits,
        );
        let in_proj_ba = make_quantized_linear(
            weights,
            &format!("{prefix}.in_proj_ba"),
            qc.group_size,
            qc.bits,
        );
        let out_proj = make_quantized_linear(
            weights,
            &format!("{prefix}.out_proj"),
            qc.group_size,
            qc.bits,
        );

        // Raw parameter tensors (always float).
        let conv1d_weight_raw = weights
            .get(&format!("{prefix}.conv1d.weight"))
            .cloned()
            .unwrap_or_else(|| {
                Array::zeros::<f32>(&[
                    config.conv_dim() as i32,
                    config.linear_conv_kernel_dim as i32,
                ])
                .unwrap()
            });
        // Squeeze if 3D [conv_dim, 1, kernel_size] → [conv_dim, kernel_size].
        let conv1d_weight = if conv1d_weight_raw.ndim() == 3 {
            conv1d_weight_raw.squeeze_axes(&[1]).unwrap()
        } else {
            conv1d_weight_raw
        };

        let a_log = weights
            .get(&format!("{prefix}.A_log"))
            .cloned()
            .unwrap_or_else(|| {
                Array::zeros::<f32>(&[config.linear_num_value_heads as i32]).unwrap()
            });
        let dt_bias = weights
            .get(&format!("{prefix}.dt_bias"))
            .cloned()
            .unwrap_or_else(|| {
                Array::zeros::<f32>(&[config.linear_num_value_heads as i32]).unwrap()
            });
        let norm_weight = weights
            .get(&format!("{prefix}.norm.weight"))
            .cloned()
            .unwrap_or_else(|| Array::ones::<f32>(&[config.linear_value_head_dim as i32]).unwrap());

        Self {
            in_proj_qkvz,
            in_proj_ba,
            conv1d_weight: Param::new(conv1d_weight),
            a_log: Param::new(a_log),
            dt_bias: Param::new(dt_bias),
            norm_weight: Param::new(norm_weight),
            norm_eps: config.rms_norm_eps,
            out_proj,
            num_k_heads: config.linear_num_key_heads,
            num_v_heads: config.linear_num_value_heads,
            head_k_dim: config.linear_key_head_dim,
            head_v_dim: config.linear_value_head_dim,
            key_dim: config.key_dim(),
            value_dim: config.value_dim(),
            conv_dim: config.conv_dim(),
            conv_kernel_size: config.linear_conv_kernel_dim,
            conv_state: RefCell::new(None),
            ssm_state: RefCell::new(None),
        }
    }

    fn reset_state(&self) {
        *self.conv_state.borrow_mut() = None;
        *self.ssm_state.borrow_mut() = None;
    }

    fn extract_state(&self) -> Option<(Array, Array)> {
        let conv = self.conv_state.borrow_mut().take();
        let ssm = self.ssm_state.borrow_mut().take();
        match (conv, ssm) {
            (Some(c), Some(s)) => Some((c, s)),
            _ => None,
        }
    }

    fn inject_state(&self, state: &Option<(Array, Array)>) {
        match state {
            Some((c, s)) => {
                *self.conv_state.borrow_mut() = Some(c.clone());
                *self.ssm_state.borrow_mut() = Some(s.clone());
            }
            None => {
                *self.conv_state.borrow_mut() = None;
                *self.ssm_state.borrow_mut() = None;
            }
        }
    }

    fn forward(&mut self, hidden_states: &Array) -> Result<Array, Exception> {
        // Reuse the same forward logic as the float variant — the only difference
        // is that in_proj_qkvz, in_proj_ba, out_proj are QuantizedLinear.
        mlx_rs::transforms::eval(std::iter::once(hidden_states))?;
        let num_tokens = hidden_states.dim(0) as usize;
        let dtype = hidden_states.dtype();

        // --- 1. Input projections ---
        let proj_qkvz = self.in_proj_qkvz.forward(hidden_states)?;
        let proj_ba = self.in_proj_ba.forward(hidden_states)?;

        let v_per_k = self.num_v_heads / self.num_k_heads;
        let per_group = self.head_k_dim
            + self.head_k_dim
            + v_per_k * self.head_v_dim
            + v_per_k * self.head_v_dim;
        let proj_qkvz =
            proj_qkvz.reshape(&[num_tokens as i32, self.num_k_heads as i32, per_group as i32])?;

        let hk = self.head_k_dim as i32;
        let q_grouped = proj_qkvz.try_index((.., .., ..hk))?;
        let k_grouped = proj_qkvz.try_index((.., .., hk..2 * hk))?;
        let v_end = 2 * hk + (v_per_k * self.head_v_dim) as i32;
        let v_grouped = proj_qkvz.try_index((.., .., 2 * hk..v_end))?;
        let z_grouped = proj_qkvz.try_index((.., .., v_end..))?;

        let q_flat = q_grouped.reshape(&[num_tokens as i32, self.key_dim as i32])?;
        let k_flat = k_grouped.reshape(&[num_tokens as i32, self.key_dim as i32])?;
        let v_flat = v_grouped.reshape(&[num_tokens as i32, self.value_dim as i32])?;
        let z = z_grouped.reshape(&[num_tokens as i32, self.value_dim as i32])?;

        let proj_ba = proj_ba.reshape(&[
            num_tokens as i32,
            self.num_k_heads as i32,
            (2 * v_per_k) as i32,
        ])?;
        let b = proj_ba
            .try_index((.., .., ..v_per_k as i32))?
            .reshape(&[num_tokens as i32, self.num_v_heads as i32])?;
        let a = proj_ba
            .try_index((.., .., v_per_k as i32..))?
            .reshape(&[num_tokens as i32, self.num_v_heads as i32])?;

        let mixed_qkv = mlx_rs::ops::concatenate_axis(&[q_flat, k_flat, v_flat], 1)?;

        // --- 2. Causal conv1d + SiLU ---
        mlx_rs::transforms::eval(std::iter::once(&mixed_qkv))?;
        let conv_out = self.causal_conv1d(&mixed_qkv, num_tokens)?;

        // --- 3. Split conv output ---
        let q_conv = conv_out.try_index((.., ..self.key_dim as i32))?;
        let k_conv = conv_out.try_index((.., self.key_dim as i32..(2 * self.key_dim) as i32))?;
        let v_conv = conv_out.try_index((.., (2 * self.key_dim) as i32..))?;

        // --- 4. Gating ---
        let a_plus_bias = a.add(&self.dt_bias)?;
        mlx_rs::transforms::eval(std::iter::once(&a_plus_bias))?;
        let sp = mlx_rs::ops::log(&a_plus_bias.exp()?.add(Array::from_f32(1.0))?)?;
        let a_exp = self.a_log.as_dtype(Dtype::Float32)?.exp()?;
        let g = sp.as_dtype(Dtype::Float32)?.multiply(&a_exp)?.negative()?;
        let g = g.as_dtype(dtype)?;
        let beta = nn::sigmoid(&b)?;

        // --- 5. Recurrence ---
        mlx_rs::transforms::eval([&q_conv, &k_conv, &v_conv, &g, &beta].iter().copied())?;
        let output =
            self.gated_delta_recurrence(&q_conv, &k_conv, &v_conv, &g, &beta, num_tokens)?;

        // --- 6. RMSNormGated ---
        let z = z.reshape(&[
            num_tokens as i32,
            self.num_v_heads as i32,
            self.head_v_dim as i32,
        ])?;
        let normed = self.rms_norm_gated(&output, &z)?;

        // --- 7. Output projection ---
        let normed_flat = normed.reshape(&[num_tokens as i32, self.value_dim as i32])?;
        self.out_proj.forward(&normed_flat)
    }

    fn causal_conv1d(&self, mixed_qkv: &Array, num_tokens: usize) -> Result<Array, Exception> {
        let k = self.conv_kernel_size;
        let mut conv_st = self.conv_state.borrow_mut();

        let pad = match conv_st.take() {
            Some(s) => s,
            None => Array::zeros::<f32>(&[(k - 1) as i32, self.conv_dim as i32])?,
        };

        let padded = mlx_rs::ops::concatenate_axis(&[pad, mixed_qkv.clone()], 0)?;
        mlx_rs::transforms::eval(std::iter::once(&padded))?;

        let mut outputs = Vec::with_capacity(num_tokens);
        for t in 0..num_tokens {
            let window = padded.try_index(t as i32..(t + k) as i32)?;
            let window_t = window.transpose_axes(&[1, 0])?;
            let out = window_t.multiply(&self.conv1d_weight)?.sum_axis(-1, None)?;
            outputs.push(out);
        }
        let output = mlx_rs::ops::stack_axis(&outputs, 0)?;
        let output = nn::silu(&output)?;

        let start = num_tokens.saturating_sub(k - 1);
        let len = num_tokens.min(k - 1);
        *conv_st = Some(mixed_qkv.try_index(start as i32..(start + len) as i32)?);

        Ok(output)
    }

    fn gated_delta_recurrence(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        num_tokens: usize,
    ) -> Result<Array, Exception> {
        let mut ssm_st = self.ssm_state.borrow_mut();
        let mut state = match ssm_st.take() {
            Some(s) => s,
            None => Array::zeros::<f32>(&[
                self.num_v_heads as i32,
                self.head_v_dim as i32,
                self.head_k_dim as i32,
            ])?,
        };

        let q = q.reshape(&[
            num_tokens as i32,
            self.num_k_heads as i32,
            self.head_k_dim as i32,
        ])?;
        let k = k.reshape(&[
            num_tokens as i32,
            self.num_k_heads as i32,
            self.head_k_dim as i32,
        ])?;
        let v = v.reshape(&[
            num_tokens as i32,
            self.num_v_heads as i32,
            self.head_v_dim as i32,
        ])?;

        let mut outputs = Vec::with_capacity(num_tokens);

        for t in 0..num_tokens {
            let q_t = q.try_index(t as i32)?;
            let k_t = k.try_index(t as i32)?;
            let v_t = v.try_index(t as i32)?;
            let g_t = g.try_index(t as i32)?;
            let beta_t = beta.try_index(t as i32)?;

            let q_norm = mlx_rs::ops::sqrt(&q_t.square()?.sum_axis(-1, true)?)?;
            let q_t = q_t.divide(&q_norm.add(Array::from_f32(1e-12))?)?;
            let k_norm = mlx_rs::ops::sqrt(&k_t.square()?.sum_axis(-1, true)?)?;
            let k_t = k_t.divide(&k_norm.add(Array::from_f32(1e-12))?)?;

            let q_f32 = q_t.as_dtype(Dtype::Float32)?;
            let k_f32 = k_t.as_dtype(Dtype::Float32)?;
            let v_f32 = v_t.as_dtype(Dtype::Float32)?;
            let g_f32 = g_t.as_dtype(Dtype::Float32)?;
            let beta_f32 = beta_t.as_dtype(Dtype::Float32)?;

            let mut head_outputs = Vec::with_capacity(self.num_v_heads);

            for h_v in 0..self.num_v_heads {
                let h_k = h_v * self.num_k_heads / self.num_v_heads;

                let g_h = g_f32.try_index(h_v as i32)?.exp()?;
                let beta_h = beta_f32.try_index(h_v as i32)?;
                let k_head = k_f32.try_index(h_k as i32)?;
                let v_head = v_f32.try_index(h_v as i32)?;

                let v_col = v_head.reshape(&[self.head_v_dim as i32, 1])?;
                let k_row = k_head.reshape(&[1, self.head_k_dim as i32])?;
                let outer = v_col.multiply(&k_row)?;

                let s_h = state.try_index(h_v as i32)?;
                let new_s = s_h.multiply(&g_h)?.add(&outer.multiply(&beta_h)?)?;

                let q_head = q_f32.try_index(h_k as i32)?;
                let q_col = q_head.reshape(&[self.head_k_dim as i32, 1])?;
                let o_h = mlx_rs::ops::matmul(&new_s, &q_col)?.squeeze_axes(&[-1])?;

                head_outputs.push((h_v, o_h, new_s));
            }

            let mut state_slices = Vec::with_capacity(self.num_v_heads);
            let mut out_slices = Vec::with_capacity(self.num_v_heads);
            for (_, o_h, new_s) in &head_outputs {
                state_slices.push(new_s.expand_dims(0)?);
                out_slices.push(o_h.as_dtype(v.dtype())?.expand_dims(0)?);
            }
            state = mlx_rs::ops::concatenate_axis(&state_slices, 0)?;
            let token_out = mlx_rs::ops::concatenate_axis(&out_slices, 0)?;
            outputs.push(token_out);

            if (t + 1) % 32 == 0 || t + 1 == num_tokens {
                mlx_rs::transforms::eval(std::iter::once(&state))?;
            }
        }

        *ssm_st = Some(state);
        let result = mlx_rs::ops::stack_axis(&outputs, 0)?;
        Ok(result)
    }

    fn rms_norm_gated(&self, x: &Array, z: &Array) -> Result<Array, Exception> {
        let x_f32 = x.as_dtype(Dtype::Float32)?;
        let variance = x_f32.square()?.mean_axis(-1, true)?;
        let rsqrt = mlx_rs::ops::rsqrt(&variance.add(Array::from_f32(self.norm_eps))?)?;
        let normed = x_f32.multiply(&rsqrt)?.as_dtype(x.dtype())?;
        let normed = normed.multiply(&self.norm_weight)?;
        let z_sigmoid = nn::sigmoid(z)?;
        normed.multiply(&z_sigmoid)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedQwen3NextDecoderLayer
// ---------------------------------------------------------------------------

enum MlxQuantizedQwen3NextAttnVariant {
    FullAttention(MlxQuantizedQwen3NextAttention),
    LinearAttention(MlxQuantizedGatedDeltaNet),
}

enum MlxQuantizedQwen3NextMlpVariant {
    Dense(MlxQuantizedLlamaMLP),
    MoE(MlxQuantizedQwen3MoeMoE),
}

struct MlxQuantizedQwen3NextDecoderLayer {
    attn: MlxQuantizedQwen3NextAttnVariant,
    mlp: MlxQuantizedQwen3NextMlpVariant,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
    is_full_attn: bool,
}

impl MlxQuantizedQwen3NextDecoderLayer {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &MlxQwen3NextConfig,
        layer_idx: usize,
        qc: &QuantConfig,
    ) -> Self {
        let is_full_attn = config.is_full_attention(layer_idx);
        let attn = if is_full_attn {
            MlxQuantizedQwen3NextAttnVariant::FullAttention(
                MlxQuantizedQwen3NextAttention::from_weights(
                    weights,
                    &format!("{prefix}.self_attn"),
                    config,
                    qc,
                ),
            )
        } else {
            MlxQuantizedQwen3NextAttnVariant::LinearAttention(
                MlxQuantizedGatedDeltaNet::from_weights(
                    weights,
                    &format!("{prefix}.linear_attn"),
                    config,
                    qc,
                ),
            )
        };

        let moe_config = config.moe_config();
        let mlp = if config.is_moe_layer(layer_idx) {
            MlxQuantizedQwen3NextMlpVariant::MoE(MlxQuantizedQwen3MoeMoE::from_weights(
                weights,
                &format!("{prefix}.mlp"),
                &moe_config,
                qc,
            ))
        } else {
            MlxQuantizedQwen3NextMlpVariant::Dense(MlxQuantizedLlamaMLP::from_weights(
                weights,
                &format!("{prefix}.mlp"),
                qc,
            ))
        };

        // GemmaRMSNorm: weight+1.
        let mut input_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()
            .unwrap();
        assign_gemma_norm_weight(
            &mut input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm.weight"),
        );
        let mut post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()
            .unwrap();
        assign_gemma_norm_weight(
            &mut post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        );

        Self {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            is_full_attn,
        }
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<MlxLayerKvCache>,
        rope_offset: i32,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = match &mut self.attn {
            MlxQuantizedQwen3NextAttnVariant::FullAttention(attn) => {
                attn.forward(&normed, positions, cache, rope_offset)?
            }
            MlxQuantizedQwen3NextAttnVariant::LinearAttention(gdn) => gdn.forward(&normed)?,
        };
        let hidden_states = hidden_states.add(&attn_output)?;

        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = match &mut self.mlp {
            MlxQuantizedQwen3NextMlpVariant::Dense(mlp) => mlp.forward(&normed)?,
            MlxQuantizedQwen3NextMlpVariant::MoE(moe) => moe.forward(&normed)?,
        };
        hidden_states.add(&mlp_output)
    }

    fn reset_recurrent_state(&self) {
        if let MlxQuantizedQwen3NextAttnVariant::LinearAttention(gdn) = &self.attn {
            gdn.reset_state();
        }
    }

    fn extract_recurrent_state(&self) -> Option<Option<(Array, Array)>> {
        match &self.attn {
            MlxQuantizedQwen3NextAttnVariant::LinearAttention(gdn) => Some(gdn.extract_state()),
            MlxQuantizedQwen3NextAttnVariant::FullAttention(_) => None,
        }
    }

    fn inject_recurrent_state(&self, state: &Option<(Array, Array)>) {
        if let MlxQuantizedQwen3NextAttnVariant::LinearAttention(gdn) = &self.attn {
            gdn.inject_state(state);
        }
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedQwen3NextForCausalLM
// ---------------------------------------------------------------------------

/// Quantized Qwen3-Next for causal language modeling using MLX.
pub struct MlxQuantizedQwen3NextForCausalLM {
    embed_tokens: MlxEmbedTokens,
    layers: Vec<MlxQuantizedQwen3NextDecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: Option<MlxLmHead>,
    tie_word_embeddings: bool,
    num_attn_layers: usize,
}

impl MlxQuantizedQwen3NextForCausalLM {
    pub fn load(
        model_dir: &Path,
        config: &MlxQwen3NextConfig,
        qc: &QuantConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let weights = load_safetensors_weights(model_dir)?;

        let embed_tokens =
            MlxEmbedTokens::from_weights(&weights, "model.embed_tokens", qc.group_size, qc.bits);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MlxQuantizedQwen3NextDecoderLayer::from_weights(
                &weights,
                &format!("model.layers.{i}"),
                config,
                i,
                qc,
            ));
        }

        let mut norm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        assign_gemma_norm_weight(&mut norm, &weights, "model.norm.weight");

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
            num_attn_layers: config.num_full_attention_layers(),
        })
    }
}

impl super::MlxModel for MlxQuantizedQwen3NextForCausalLM {
    fn forward(
        &mut self,
        input_ids: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        let offset = rope_offset.unwrap_or(0);
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        let mut kv_slot = 0;
        for layer in self.layers.iter_mut() {
            if layer.is_full_attn {
                hidden_states =
                    layer.forward(&hidden_states, positions, &mut kv_cache[kv_slot], offset)?;
                kv_slot += 1;
            } else {
                let mut dummy_cache = None;
                hidden_states =
                    layer.forward(&hidden_states, positions, &mut dummy_cache, offset)?;
            }
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        let logits = if self.tie_word_embeddings {
            self.embed_tokens.as_linear(&hidden_states)?
        } else {
            self.lm_head.as_mut().unwrap().forward(&hidden_states)?
        };

        Ok(logits)
    }

    fn num_layers(&self) -> usize {
        self.num_attn_layers
    }

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut kv_cache: MlxKvCache = (0..self.num_attn_layers).map(|_| None).collect();
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        let mut kv_slot = 0;
        for layer in self.layers.iter_mut() {
            if layer.is_full_attn {
                hidden_states =
                    layer.forward(&hidden_states, positions, &mut kv_cache[kv_slot], 0)?;
                kv_slot += 1;
            } else {
                let mut dummy_cache = None;
                hidden_states = layer.forward(&hidden_states, positions, &mut dummy_cache, 0)?;
            }
        }
        self.norm.forward(&hidden_states)
    }

    fn reset_recurrent_state(&self) {
        for layer in &self.layers {
            layer.reset_recurrent_state();
        }
    }

    fn num_recurrent_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l.attn, MlxQuantizedQwen3NextAttnVariant::LinearAttention(_)))
            .count()
    }

    fn extract_recurrent_state(&self) -> super::MlxRecurrentState {
        self.layers
            .iter()
            .filter_map(|l| l.extract_recurrent_state())
            .collect()
    }

    fn inject_recurrent_state(&self, state: &[Option<(Array, Array)>]) {
        let mut idx = 0;
        for layer in &self.layers {
            if matches!(
                layer.attn,
                MlxQuantizedQwen3NextAttnVariant::LinearAttention(_)
            ) {
                if let Some(s) = state.get(idx) {
                    layer.inject_recurrent_state(s);
                }
                idx += 1;
            }
        }
    }
}

// ===========================================================================
// Factory functions
// ===========================================================================

/// Factory function for creating a float MLX Qwen3-Next model.
pub fn create_mlx_qwen3_next(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let next_config = MlxQwen3NextConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let model = MlxQwen3NextForCausalLM::load(model_dir, &next_config, dtype)?;
    Ok(Box::new(model))
}

/// Factory function for creating a quantized MLX Qwen3-Next model.
pub fn create_mlx_quantized_qwen3_next(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let next_config = MlxQwen3NextConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();
    let model = MlxQuantizedQwen3NextForCausalLM::load(model_dir, &next_config, &qc, dtype)?;
    Ok(Box::new(model))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> MlxQwen3NextConfig {
        MlxQwen3NextConfig {
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
            partial_rotary_factor: 0.5,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_num_key_heads: 4,
            linear_num_value_heads: 4,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 16,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: true,
            decoder_sparse_step: 1,
            mlp_only_layers: vec![],
            layer_types: vec![
                "linear_attention".into(),
                "linear_attention".into(),
                "linear_attention".into(),
                "full_attention".into(),
            ],
        }
    }

    #[test]
    fn test_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["Qwen3NextForCausalLM"],
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "num_hidden_layers": 48,
                "intermediate_size": 5632,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "head_dim": 256,
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000.0,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_value_head_dim": 128,
                "linear_num_value_heads": 32,
                "num_experts": 512,
                "decoder_sparse_step": 1
            }"#,
        )
        .unwrap();

        let config = MlxQwen3NextConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.linear_num_value_heads, 32);
        assert_eq!(config.num_experts, 512);
        assert_eq!(config.layer_types.len(), 48);
        assert_eq!(config.layer_types[3], "full_attention");
    }

    #[test]
    fn test_float_model_forward() {
        let config = test_config();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MlxQwen3NextDecoderLayer::new(&config, i).unwrap());
        }

        let model_inner = MlxQwen3NextForCausalLM {
            embed_tokens: nn::Embedding::new(config.vocab_size as i32, config.hidden_size as i32)
                .unwrap(),
            layers,
            norm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()
                .unwrap(),
            lm_head: None,
            tie_word_embeddings: true,
            num_attn_layers: config.num_full_attention_layers(),
            config: config.clone(),
        };

        let mut model: Box<dyn super::super::MlxModel> = Box::new(model_inner);

        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_full_attention_layers());

        let logits = model
            .forward(&input_ids, &positions, &mut kv_cache, None)
            .unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.shape(), &[3, config.vocab_size as i32]);
    }

    #[test]
    fn test_registry() {
        let registry = crate::models::MlxModelRegistry::default_registry();
        assert!(registry.contains("Qwen3NextForCausalLM"));
    }
}
