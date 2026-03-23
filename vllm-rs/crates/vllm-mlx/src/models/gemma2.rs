// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Gemma 2 model architecture for MLX.
//!
//! Key differences from LLaMA:
//! - GemmaRmsNorm: adds 1.0 to weight before applying (`y = (1 + w) * normed`)
//! - GELU approximate activation instead of SiLU
//! - 4 norms per layer: input, post-attention, pre-feedforward, post-feedforward
//! - `query_pre_attn_scalar` for attention scaling (not `1/sqrt(head_dim)`)
//! - Embedding multiplied by `sqrt(hidden_size)` after lookup
//! - Always tied embeddings
//! - `final_logit_softcapping`: `cap * tanh(logits / cap)`

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use crate::cache::{MlxBatchInfo, MlxKvCache, MlxLayerKvCache};
use crate::models::llama::{assign_weight, load_safetensors_weights};
use crate::models::quantized_llama::{MlxEmbedTokens, QuantConfig, make_quantized_linear};
use vllm_model::weight::HfModelConfig;

// ---------------------------------------------------------------------------
// MlxGemma2Config
// ---------------------------------------------------------------------------

/// Parsed configuration for a Gemma 2 model.
#[derive(Debug, Clone)]
pub struct MlxGemma2Config {
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
    /// Scaling factor for query before attention (replaces 1/sqrt(head_dim)).
    pub query_pre_attn_scalar: f32,
    /// Soft cap for attention logits. None = no capping.
    pub attn_logit_softcapping: Option<f32>,
    /// Soft cap for final logits. None = no capping.
    pub final_logit_softcapping: Option<f32>,
    /// Whether attention projections use bias.
    pub attention_bias: bool,
    /// Sliding window size for "sliding_attention" layers. None if not set.
    pub sliding_window: Option<usize>,
    /// Per-layer attention type: `true` = sliding attention, `false` = full attention.
    pub layer_is_sliding: Vec<bool>,
}

impl MlxGemma2Config {
    /// Parse from a HuggingFace config.json.
    pub fn from_hf_config(config: &HfModelConfig) -> Result<Self, String> {
        let hidden_size = config
            .hidden_size
            .ok_or_else(|| "missing hidden_size".to_string())?;
        let num_attention_heads = config
            .num_attention_heads
            .ok_or_else(|| "missing num_attention_heads".to_string())?;
        let head_dim = config
            .head_dim()
            .unwrap_or(hidden_size / num_attention_heads);

        let query_pre_attn_scalar = config
            .extra
            .get("query_pre_attn_scalar")
            .and_then(|v| v.as_f64())
            .unwrap_or(head_dim as f64) as f32;

        let attn_logit_softcapping = config
            .extra
            .get("attn_logit_softcapping")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);

        let final_logit_softcapping = config
            .extra
            .get("final_logit_softcapping")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);

        let attention_bias = config
            .extra
            .get("attention_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let sliding_window = config
            .extra
            .get("sliding_window")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let layer_is_sliding =
            if let Some(layer_types) = config.extra.get("layer_types").and_then(|v| v.as_array()) {
                layer_types
                    .iter()
                    .map(|v| v.as_str() == Some("sliding_attention"))
                    .collect()
            } else {
                Vec::new()
            };

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
            max_position_embeddings: config.max_position_embeddings.unwrap_or(8192),
            rms_norm_eps: config.norm_eps() as f32,
            rope_theta: config.rope_theta.unwrap_or(10000.0) as f32,
            head_dim,
            query_pre_attn_scalar,
            attn_logit_softcapping,
            final_logit_softcapping,
            attention_bias,
            sliding_window,
            layer_is_sliding,
        })
    }
}

// ---------------------------------------------------------------------------
// GemmaRmsNorm helper: load weight with +1 offset
// ---------------------------------------------------------------------------

/// Assign a Gemma RMS norm weight: `norm.weight = loaded_weight + 1.0`.
///
/// Standard RmsNorm computes `weight * normed`. Gemma wants `(1 + weight) * normed`,
/// so we store `weight + 1` in the norm's weight parameter.
pub(crate) fn assign_gemma_norm_weight(
    norm: &mut nn::RmsNorm,
    weights: &HashMap<String, Array>,
    name: &str,
) {
    if let Some(w) = weights.get(name) {
        norm.weight.value = w.add(Array::from_f32(1.0)).unwrap();
    } else {
        tracing::warn!("Weight not found: {name}");
    }
}

// ---------------------------------------------------------------------------
// MlxGemma2MLP
// ---------------------------------------------------------------------------

/// Gemma MLP (GELU-gated feed-forward network) using MLX.
///
/// - Gemma v1: exact GELU (erf-based, `hidden_act: "gelu"`)
/// - Gemma2: approximate GELU (tanh-based, `hidden_act: "gelu_pytorch_tanh"`)
struct MlxGemma2MLP {
    gate_proj: nn::Linear,
    up_proj: nn::Linear,
    down_proj: nn::Linear,
    /// true → gelu_approximate (Gemma2), false → exact gelu (Gemma v1)
    approximate_gelu: bool,
}

impl MlxGemma2MLP {
    fn new(
        hidden_size: i32,
        intermediate_size: i32,
        approximate_gelu: bool,
    ) -> Result<Self, Exception> {
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
            approximate_gelu,
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
        let gate = if self.approximate_gelu {
            nn::gelu_approximate(&gate)?
        } else {
            nn::gelu(&gate)?
        };
        let up = self.up_proj.forward(x)?;
        let hidden = gate.multiply(&up)?;
        self.down_proj.forward(&hidden)
    }
}

// ---------------------------------------------------------------------------
// MlxGemma2Attention
// ---------------------------------------------------------------------------

/// Gemma2 multi-head attention with custom query scaling and optional RoPE.
struct MlxGemma2Attention {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    o_proj: nn::Linear,
    rope: nn::Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    #[allow(dead_code)]
    attn_logit_softcapping: Option<f32>,
    /// Per-layer sliding window. `Some(w)` for sliding-attention layers, `None` for full.
    sliding_window: Option<usize>,
    /// When true, K is stored without RoPE and RoPE is applied to the full
    /// cached K at attention time (relocatable span blocks).
    block_needs_positioning: bool,
}

impl MlxGemma2Attention {
    fn new(
        config: &MlxGemma2Config,
        layer_sliding_window: Option<usize>,
    ) -> Result<Self, Exception> {
        let hidden = config.hidden_size as i32;
        let q_size = (config.num_attention_heads * config.head_dim) as i32;
        let kv_size = (config.num_kv_heads * config.head_dim) as i32;

        Ok(Self {
            q_proj: nn::LinearBuilder::new(hidden, q_size)
                .bias(config.attention_bias)
                .build()?,
            k_proj: nn::LinearBuilder::new(hidden, kv_size)
                .bias(config.attention_bias)
                .build()?,
            v_proj: nn::LinearBuilder::new(hidden, kv_size)
                .bias(config.attention_bias)
                .build()?,
            o_proj: nn::LinearBuilder::new(q_size, hidden)
                .bias(config.attention_bias)
                .build()?,
            rope: {
                let mut r = nn::Rope::new(config.head_dim as i32);
                r.base = config.rope_theta;
                r
            },
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: config.query_pre_attn_scalar.powf(-0.5),
            attn_logit_softcapping: config.attn_logit_softcapping,
            sliding_window: layer_sliding_window,
            block_needs_positioning: false, // TODO: enable when MLX gets paged KV cache for per-block relocation
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
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        rope_offset: i32,
        cache: &mut Option<MlxLayerKvCache>,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape: [seq, hidden] -> [1, heads, seq, head_dim]
        let q = q
            .reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;
        let k = k
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;
        let v = v
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        // RoPE + KV cache
        let (q, mut k, mut v) = if self.block_needs_positioning {
            let q = crate::models::llama::apply_rope_to_cached_k(&q, &self.rope, rope_offset)?;
            let (mut k, v) = crate::cache::kv_cache_update(cache, &k, &v)?;
            k = crate::models::llama::apply_rope_to_cached_k(&k, &self.rope, 0)?;
            (q, k, v)
        } else {
            let q = self.rope.forward((&q, rope_offset))?;
            let k = self.rope.forward((&k, rope_offset))?;
            let (k, v) = crate::cache::kv_cache_update(cache, &k, &v)?;
            (q, k, v)
        };

        // Sliding window: trim K/V to only the last `w` positions.
        if let Some(w) = self.sliding_window {
            let kv_len = k.dim(2) as usize;
            if kv_len > w {
                let start = (kv_len - w) as i32;
                let end = kv_len as i32;
                k = k.try_index((.., .., start..end, ..))?;
                v = v.try_index((.., .., start..end, ..))?;
            }
        }

        // Fused SDPA
        let mask = if seq_len > 1 {
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let out = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;

        // [1, heads, seq, head_dim] -> [seq, hidden]
        let hidden = (self.num_heads * self.head_dim) as i32;
        let out = out
            .squeeze_axes(&[0])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[seq_len, hidden])?;

        self.o_proj.forward(&out)
    }

    /// Batched forward: projections batched on `[total_tokens, hidden]`,
    /// split per-request for reshape/RoPE/KV-cache/SDPA, then rejoin for O projection.
    fn forward_batch(
        &mut self,
        hidden_states: &Array,
        batch_info: &MlxBatchInfo,
        caches: &mut [Option<MlxLayerKvCache>],
    ) -> Result<Array, Exception> {
        // Batched Q/K/V projections on [total_tokens, hidden].
        let q_all = self.q_proj.forward(hidden_states)?;
        let k_all = self.k_proj.forward(hidden_states)?;
        let v_all = self.v_proj.forward(hidden_states)?;

        // Check if all requests are decode (q_len=1) -> can attempt batched SDPA.
        let all_decode = batch_info.num_reqs > 1
            && batch_info.q_lens.iter().all(|&ql| ql == 1)
            && self.sliding_window.is_none();

        // Per-request: reshape, RoPE, KV cache update.
        let mut per_req_q = Vec::with_capacity(batch_info.num_reqs);
        let mut per_req_k = Vec::with_capacity(batch_info.num_reqs);
        let mut per_req_v = Vec::with_capacity(batch_info.num_reqs);
        let mut kv_lens = Vec::with_capacity(batch_info.num_reqs);

        #[allow(clippy::needless_range_loop)]
        for i in 0..batch_info.num_reqs {
            let start = batch_info.offsets[i] as i32;
            let seq_len = batch_info.q_lens[i] as i32;
            let offset = batch_info.rope_offsets[i];

            let q = q_all.try_index((start..start + seq_len, ..))?;
            let k = k_all.try_index((start..start + seq_len, ..))?;
            let v = v_all.try_index((start..start + seq_len, ..))?;

            let q = q.reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?;
            let k = k.reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?;

            // [seq, heads, head_dim] -> [1, heads, seq, head_dim]
            let q = q.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
            let mut k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
            let v = v
                .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
                .transpose_axes(&[1, 0, 2])?
                .expand_dims(0)?;

            let (q, k, v) = if self.block_needs_positioning {
                let q = crate::models::llama::apply_rope_to_cached_k(&q, &self.rope, offset)?;
                let (mut k, v) = crate::cache::kv_cache_update(&mut caches[i], &k, &v)?;
                k = crate::models::llama::apply_rope_to_cached_k(&k, &self.rope, 0)?;
                (q, k, v)
            } else {
                let q = self.rope.forward((&q, offset))?;
                k = self.rope.forward((&k, offset))?;
                let (k, v) = crate::cache::kv_cache_update(&mut caches[i], &k, &v)?;
                (q, k, v)
            };

            kv_lens.push(k.dim(2) as usize);
            per_req_q.push(q);
            per_req_k.push(k);
            per_req_v.push(v);
        }

        // Decide: batched SDPA (all decode + same KV len) or per-request.
        let can_batch_sdpa =
            all_decode && !kv_lens.is_empty() && kv_lens.iter().all(|&l| l == kv_lens[0]);

        let concat = if can_batch_sdpa {
            // Stack Q/K/V across batch dim: [batch, heads, seq/kv_len, head_dim]
            let q_stacked = mlx_rs::ops::concatenate_axis(&per_req_q, 0)?;
            let k_stacked = mlx_rs::ops::concatenate_axis(&per_req_k, 0)?;
            let v_stacked = mlx_rs::ops::concatenate_axis(&per_req_v, 0)?;

            // Single SDPA: q_len=1 decode -> no mask needed.
            let out = mlx_rs::fast::scaled_dot_product_attention(
                &q_stacked, &k_stacked, &v_stacked, self.scale, None,
            )?;

            // out: [batch, heads, 1, head_dim] -> [batch, heads*head_dim]
            let hidden = (self.num_heads * self.head_dim) as i32;
            out.squeeze_axes(&[2])?
                .reshape(&[batch_info.num_reqs as i32, hidden])?
        } else {
            // Per-request SDPA (prefill or mixed KV lengths).
            let mut attn_outputs = Vec::with_capacity(batch_info.num_reqs);
            for i in 0..batch_info.num_reqs {
                let seq_len = batch_info.q_lens[i] as i32;
                let mut k = per_req_k[i].clone();
                let mut v = per_req_v[i].clone();

                if let Some(w) = self.sliding_window {
                    let kv_len = kv_lens[i];
                    if kv_len > w {
                        let s = (kv_len - w) as i32;
                        let e = kv_len as i32;
                        k = k.try_index((.., .., s..e, ..))?;
                        v = v.try_index((.., .., s..e, ..))?;
                    }
                }

                let mask = if seq_len > 1 {
                    Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
                } else {
                    None
                };
                let out = mlx_rs::fast::scaled_dot_product_attention(
                    &per_req_q[i],
                    &k,
                    &v,
                    self.scale,
                    mask,
                )?;

                let hidden = (self.num_heads * self.head_dim) as i32;
                let out = out
                    .squeeze_axes(&[0])?
                    .transpose_axes(&[1, 0, 2])?
                    .reshape(&[seq_len, hidden])?;
                attn_outputs.push(out);
            }

            if attn_outputs.len() == 1 {
                attn_outputs.into_iter().next().unwrap()
            } else {
                mlx_rs::ops::concatenate_axis(&attn_outputs, 0)?
            }
        };

        self.o_proj.forward(&concat)
    }
}

// ---------------------------------------------------------------------------
// MlxGemma2DecoderLayer
// ---------------------------------------------------------------------------

/// A single Gemma2 decoder layer with 4 norms.
struct MlxGemma2DecoderLayer {
    self_attn: MlxGemma2Attention,
    mlp: MlxGemma2MLP,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
    pre_feedforward_layernorm: nn::RmsNorm,
    post_feedforward_layernorm: nn::RmsNorm,
}

impl MlxGemma2DecoderLayer {
    fn new(
        config: &MlxGemma2Config,
        layer_sliding_window: Option<usize>,
    ) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: MlxGemma2Attention::new(config, layer_sliding_window)?,
            mlp: MlxGemma2MLP::new(
                config.hidden_size as i32,
                config.intermediate_size as i32,
                true,
            )?,
            input_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            pre_feedforward_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            post_feedforward_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        self.self_attn
            .load_weights(weights, &format!("{prefix}.self_attn"));
        self.mlp.load_weights(weights, &format!("{prefix}.mlp"));
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
        assign_gemma_norm_weight(
            &mut self.pre_feedforward_layernorm,
            weights,
            &format!("{prefix}.pre_feedforward_layernorm.weight"),
        );
        assign_gemma_norm_weight(
            &mut self.post_feedforward_layernorm,
            weights,
            &format!("{prefix}.post_feedforward_layernorm.weight"),
        );
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        rope_offset: i32,
        cache: &mut Option<MlxLayerKvCache>,
    ) -> Result<Array, Exception> {
        // Pre-attention norm + attention + post-attention norm + residual.
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed, rope_offset, cache)?;
        let attn_output = self.post_attention_layernorm.forward(&attn_output)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        // Pre-feedforward norm + MLP + post-feedforward norm + residual.
        let normed = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        let mlp_output = self.post_feedforward_layernorm.forward(&mlp_output)?;
        hidden_states.add(&mlp_output)
    }

    fn forward_batch(
        &mut self,
        hidden_states: &Array,
        batch_info: &MlxBatchInfo,
        caches: &mut [Option<MlxLayerKvCache>],
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward_batch(&normed, batch_info, caches)?;
        let attn_output = self.post_attention_layernorm.forward(&attn_output)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        let normed = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        let mlp_output = self.post_feedforward_layernorm.forward(&mlp_output)?;
        hidden_states.add(&mlp_output)
    }
}

// ---------------------------------------------------------------------------
// MlxGemma2ForCausalLM
// ---------------------------------------------------------------------------

/// Gemma2 for causal language modeling using MLX (float weights).
pub struct MlxGemma2ForCausalLM {
    embed_tokens: nn::Embedding,
    layers: Vec<MlxGemma2DecoderLayer>,
    norm: nn::RmsNorm,
    /// sqrt(hidden_size) — multiply embeddings after lookup.
    normalizer: f32,
    /// Soft cap for final logits.
    final_logit_softcapping: Option<f32>,
    #[allow(dead_code)]
    config: MlxGemma2Config,
}

impl MlxGemma2ForCausalLM {
    fn new(config: &MlxGemma2Config) -> Result<Self, Exception> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer_sliding_window =
                if i < config.layer_is_sliding.len() && config.layer_is_sliding[i] {
                    config.sliding_window
                } else {
                    None
                };
            layers.push(MlxGemma2DecoderLayer::new(config, layer_sliding_window)?);
        }

        Ok(Self {
            embed_tokens: nn::Embedding::new(config.vocab_size as i32, config.hidden_size as i32)?,
            layers,
            norm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            normalizer: (config.hidden_size as f32).sqrt(),
            final_logit_softcapping: config.final_logit_softcapping,
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
        assign_gemma_norm_weight(&mut self.norm, weights, "model.norm.weight");
    }

    fn load(
        model_dir: &Path,
        config: &MlxGemma2Config,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut model = Self::new(config)?;
        let weights = load_safetensors_weights(model_dir)?;
        model.load_weights(&weights);
        mlx_rs::transforms::eval(weights.values())?;
        Ok(model)
    }

    /// Apply logit softcapping: `cap * tanh(logits / cap)`.
    fn apply_softcap(logits: &Array, cap: f32) -> Result<Array, Exception> {
        let scaled = logits.divide(Array::from_f32(cap))?;
        let capped = mlx_rs::ops::tanh(&scaled)?;
        capped.multiply(Array::from_f32(cap))
    }
}

impl super::MlxModel for MlxGemma2ForCausalLM {
    fn inject_lora(
        &mut self,
        adapter: &crate::lora::MlxLoraAdapter,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::lora::merge_lora_into_weight;
        let targets = &adapter.config.target_modules;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Attention projections.
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
            // MLP projections.
            let mlp_prefix = format!("model.layers.{}.mlp", i);
            for name in ["gate_proj", "up_proj", "down_proj"] {
                if targets.iter().any(|t| t == name) {
                    let key = format!("{}.{}", mlp_prefix, name);
                    if let Some((a, b)) = adapter.weights.get(&key) {
                        let proj = match name {
                            "gate_proj" => &mut layer.mlp.gate_proj,
                            "up_proj" => &mut layer.mlp.up_proj,
                            "down_proj" => &mut layer.mlp.down_proj,
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
        _positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        let offset = rope_offset.unwrap_or(0);
        // Embed and normalize by sqrt(hidden_size).
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = hidden_states.multiply(Array::from_f32(self.normalizer))?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, offset, &mut kv_cache[i])?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        // Tied embeddings: embed_tokens.as_linear.
        let logits = self.embed_tokens.as_linear(&hidden_states)?;

        // Final logit softcapping.
        let logits = if let Some(cap) = self.final_logit_softcapping {
            Self::apply_softcap(&logits, cap)?
        } else {
            logits
        };

        Ok(logits)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn forward_batch(
        &mut self,
        input_ids: &Array,
        _positions: &Array,
        batch_info: &MlxBatchInfo,
        kv_caches: &mut [MlxKvCache],
    ) -> mlx_rs::error::Result<Array> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = hidden_states.multiply(Array::from_f32(self.normalizer))?;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let mut layer_caches: Vec<&mut Option<MlxLayerKvCache>> =
                kv_caches.iter_mut().map(|kv| &mut kv[layer_idx]).collect();
            let mut temp_caches: Vec<Option<MlxLayerKvCache>> =
                layer_caches.iter_mut().map(|c| c.take()).collect();

            hidden_states = layer.forward_batch(&hidden_states, batch_info, &mut temp_caches)?;

            for (dst, src) in layer_caches.iter_mut().zip(temp_caches.into_iter()) {
                **dst = src;
            }
        }

        hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self.embed_tokens.as_linear(&hidden_states)?;
        if let Some(cap) = self.final_logit_softcapping {
            return Self::apply_softcap(&logits, cap);
        }
        Ok(logits)
    }

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        _positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut kv_cache: MlxKvCache = (0..self.layers.len()).map(|_| None).collect();
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = hidden_states.multiply(Array::from_f32(self.normalizer))?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, 0, &mut kv_cache[i])?;
        }
        self.norm.forward(&hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Quantized Gemma2
// ---------------------------------------------------------------------------

/// Quantized Gemma2 MLP using QuantizedLinear.
struct MlxQuantizedGemma2MLP {
    gate_proj: nn::QuantizedLinear,
    up_proj: nn::QuantizedLinear,
    down_proj: nn::QuantizedLinear,
    approximate_gelu: bool,
}

impl MlxQuantizedGemma2MLP {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        qc: &QuantConfig,
        approximate_gelu: bool,
    ) -> Self {
        Self {
            gate_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.gate_proj"),
                qc.group_size,
                qc.bits,
            ),
            up_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.up_proj"),
                qc.group_size,
                qc.bits,
            ),
            down_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.down_proj"),
                qc.group_size,
                qc.bits,
            ),
            approximate_gelu,
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let gate = if self.approximate_gelu {
            nn::gelu_approximate(&gate)?
        } else {
            nn::gelu(&gate)?
        };
        let up = self.up_proj.forward(x)?;
        let hidden = gate.multiply(&up)?;
        self.down_proj.forward(&hidden)
    }
}

/// Quantized Gemma2 attention.
struct MlxQuantizedGemma2Attention {
    q_proj: nn::QuantizedLinear,
    k_proj: nn::QuantizedLinear,
    v_proj: nn::QuantizedLinear,
    o_proj: nn::QuantizedLinear,
    rope: nn::Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    #[allow(dead_code)]
    attn_logit_softcapping: Option<f32>,
    /// Per-layer sliding window. `Some(w)` for sliding-attention layers, `None` for full.
    sliding_window: Option<usize>,
    /// When true, K is stored without RoPE and RoPE is applied to the full
    /// cached K at attention time (relocatable span blocks).
    block_needs_positioning: bool,
}

impl MlxQuantizedGemma2Attention {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &MlxGemma2Config,
        qc: &QuantConfig,
        layer_sliding_window: Option<usize>,
    ) -> Self {
        Self {
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
            rope: {
                let mut r = nn::Rope::new(config.head_dim as i32);
                r.base = config.rope_theta;
                r
            },
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: config.query_pre_attn_scalar.powf(-0.5),
            attn_logit_softcapping: config.attn_logit_softcapping,
            sliding_window: layer_sliding_window,
            block_needs_positioning: false, // TODO: enable when MLX gets paged KV cache for per-block relocation
        }
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        rope_offset: i32,
        cache: &mut Option<MlxLayerKvCache>,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        let q = q
            .reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;
        let k = k
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;
        let v = v
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        // RoPE + KV cache
        let (q, mut k, mut v) = if self.block_needs_positioning {
            let q = crate::models::llama::apply_rope_to_cached_k(&q, &self.rope, rope_offset)?;
            let (mut k, v) = crate::cache::kv_cache_update(cache, &k, &v)?;
            k = crate::models::llama::apply_rope_to_cached_k(&k, &self.rope, 0)?;
            (q, k, v)
        } else {
            let q = self.rope.forward((&q, rope_offset))?;
            let k = self.rope.forward((&k, rope_offset))?;
            let (k, v) = crate::cache::kv_cache_update(cache, &k, &v)?;
            (q, k, v)
        };

        // Sliding window: trim K/V to only the last `w` positions.
        if let Some(w) = self.sliding_window {
            let kv_len = k.dim(2) as usize;
            if kv_len > w {
                let start = (kv_len - w) as i32;
                let end = kv_len as i32;
                k = k.try_index((.., .., start..end, ..))?;
                v = v.try_index((.., .., start..end, ..))?;
            }
        }

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

    /// Batched forward: projections batched on `[total_tokens, hidden]`,
    /// split per-request for reshape/RoPE/KV-cache/SDPA, then rejoin for O projection.
    fn forward_batch(
        &mut self,
        hidden_states: &Array,
        batch_info: &MlxBatchInfo,
        caches: &mut [Option<MlxLayerKvCache>],
    ) -> Result<Array, Exception> {
        // Batched Q/K/V projections on [total_tokens, hidden].
        let q_all = self.q_proj.forward(hidden_states)?;
        let k_all = self.k_proj.forward(hidden_states)?;
        let v_all = self.v_proj.forward(hidden_states)?;

        // Check if all requests are decode (q_len=1) -> can attempt batched SDPA.
        let all_decode = batch_info.num_reqs > 1
            && batch_info.q_lens.iter().all(|&ql| ql == 1)
            && self.sliding_window.is_none();

        // Per-request: reshape, RoPE, KV cache update.
        let mut per_req_q = Vec::with_capacity(batch_info.num_reqs);
        let mut per_req_k = Vec::with_capacity(batch_info.num_reqs);
        let mut per_req_v = Vec::with_capacity(batch_info.num_reqs);
        let mut kv_lens = Vec::with_capacity(batch_info.num_reqs);

        #[allow(clippy::needless_range_loop)]
        for i in 0..batch_info.num_reqs {
            let start = batch_info.offsets[i] as i32;
            let seq_len = batch_info.q_lens[i] as i32;
            let offset = batch_info.rope_offsets[i];

            let q = q_all.try_index((start..start + seq_len, ..))?;
            let k = k_all.try_index((start..start + seq_len, ..))?;
            let v = v_all.try_index((start..start + seq_len, ..))?;

            let q = q.reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?;
            let k = k.reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?;

            let q = q.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
            let mut k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
            let v = v
                .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
                .transpose_axes(&[1, 0, 2])?
                .expand_dims(0)?;

            let (q, k, v) = if self.block_needs_positioning {
                let q = crate::models::llama::apply_rope_to_cached_k(&q, &self.rope, offset)?;
                let (mut k, v) = crate::cache::kv_cache_update(&mut caches[i], &k, &v)?;
                k = crate::models::llama::apply_rope_to_cached_k(&k, &self.rope, 0)?;
                (q, k, v)
            } else {
                let q = self.rope.forward((&q, offset))?;
                k = self.rope.forward((&k, offset))?;
                let (k, v) = crate::cache::kv_cache_update(&mut caches[i], &k, &v)?;
                (q, k, v)
            };

            kv_lens.push(k.dim(2) as usize);
            per_req_q.push(q);
            per_req_k.push(k);
            per_req_v.push(v);
        }

        // Decide: batched SDPA (all decode + same KV len) or per-request.
        let can_batch_sdpa =
            all_decode && !kv_lens.is_empty() && kv_lens.iter().all(|&l| l == kv_lens[0]);

        let concat = if can_batch_sdpa {
            let q_stacked = mlx_rs::ops::concatenate_axis(&per_req_q, 0)?;
            let k_stacked = mlx_rs::ops::concatenate_axis(&per_req_k, 0)?;
            let v_stacked = mlx_rs::ops::concatenate_axis(&per_req_v, 0)?;

            let out = mlx_rs::fast::scaled_dot_product_attention(
                &q_stacked, &k_stacked, &v_stacked, self.scale, None,
            )?;

            let hidden = (self.num_heads * self.head_dim) as i32;
            out.squeeze_axes(&[2])?
                .reshape(&[batch_info.num_reqs as i32, hidden])?
        } else {
            let mut attn_outputs = Vec::with_capacity(batch_info.num_reqs);
            for i in 0..batch_info.num_reqs {
                let seq_len = batch_info.q_lens[i] as i32;
                let mut k = per_req_k[i].clone();
                let mut v = per_req_v[i].clone();

                if let Some(w) = self.sliding_window {
                    let kv_len = kv_lens[i];
                    if kv_len > w {
                        let s = (kv_len - w) as i32;
                        let e = kv_len as i32;
                        k = k.try_index((.., .., s..e, ..))?;
                        v = v.try_index((.., .., s..e, ..))?;
                    }
                }

                let mask = if seq_len > 1 {
                    Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
                } else {
                    None
                };
                let out = mlx_rs::fast::scaled_dot_product_attention(
                    &per_req_q[i],
                    &k,
                    &v,
                    self.scale,
                    mask,
                )?;

                let hidden = (self.num_heads * self.head_dim) as i32;
                let out = out
                    .squeeze_axes(&[0])?
                    .transpose_axes(&[1, 0, 2])?
                    .reshape(&[seq_len, hidden])?;
                attn_outputs.push(out);
            }

            if attn_outputs.len() == 1 {
                attn_outputs.into_iter().next().unwrap()
            } else {
                mlx_rs::ops::concatenate_axis(&attn_outputs, 0)?
            }
        };

        self.o_proj.forward(&concat)
    }
}

/// A single quantized Gemma2 decoder layer with 4 norms.
struct MlxQuantizedGemma2DecoderLayer {
    self_attn: MlxQuantizedGemma2Attention,
    mlp: MlxQuantizedGemma2MLP,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
    pre_feedforward_layernorm: nn::RmsNorm,
    post_feedforward_layernorm: nn::RmsNorm,
}

impl MlxQuantizedGemma2DecoderLayer {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &MlxGemma2Config,
        qc: &QuantConfig,
        layer_sliding_window: Option<usize>,
    ) -> Result<Self, Exception> {
        let mut input_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        let mut post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        let mut pre_feedforward_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        let mut post_feedforward_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;

        assign_gemma_norm_weight(
            &mut input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm.weight"),
        );
        assign_gemma_norm_weight(
            &mut post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        );
        assign_gemma_norm_weight(
            &mut pre_feedforward_layernorm,
            weights,
            &format!("{prefix}.pre_feedforward_layernorm.weight"),
        );
        assign_gemma_norm_weight(
            &mut post_feedforward_layernorm,
            weights,
            &format!("{prefix}.post_feedforward_layernorm.weight"),
        );

        Ok(Self {
            self_attn: MlxQuantizedGemma2Attention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config,
                qc,
                layer_sliding_window,
            ),
            mlp: MlxQuantizedGemma2MLP::from_weights(weights, &format!("{prefix}.mlp"), qc, true),
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        rope_offset: i32,
        cache: &mut Option<MlxLayerKvCache>,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed, rope_offset, cache)?;
        let attn_output = self.post_attention_layernorm.forward(&attn_output)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        let normed = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        let mlp_output = self.post_feedforward_layernorm.forward(&mlp_output)?;
        hidden_states.add(&mlp_output)
    }

    fn forward_batch(
        &mut self,
        hidden_states: &Array,
        batch_info: &MlxBatchInfo,
        caches: &mut [Option<MlxLayerKvCache>],
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward_batch(&normed, batch_info, caches)?;
        let attn_output = self.post_attention_layernorm.forward(&attn_output)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        let normed = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        let mlp_output = self.post_feedforward_layernorm.forward(&mlp_output)?;
        hidden_states.add(&mlp_output)
    }
}

/// Quantized Gemma2 for causal language modeling using MLX.
pub struct MlxQuantizedGemma2ForCausalLM {
    embed_tokens: MlxEmbedTokens,
    layers: Vec<MlxQuantizedGemma2DecoderLayer>,
    norm: nn::RmsNorm,
    normalizer: f32,
    final_logit_softcapping: Option<f32>,
    #[allow(dead_code)]
    config: MlxGemma2Config,
}

impl MlxQuantizedGemma2ForCausalLM {
    fn load(
        model_dir: &Path,
        config: &MlxGemma2Config,
        qc: &QuantConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let weights = load_safetensors_weights(model_dir)?;

        let embed_tokens =
            MlxEmbedTokens::from_weights(&weights, "model.embed_tokens", qc.group_size, qc.bits);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer_sliding_window =
                if i < config.layer_is_sliding.len() && config.layer_is_sliding[i] {
                    config.sliding_window
                } else {
                    None
                };
            layers.push(MlxQuantizedGemma2DecoderLayer::from_weights(
                &weights,
                &format!("model.layers.{i}"),
                config,
                qc,
                layer_sliding_window,
            )?);
        }

        let mut norm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        assign_gemma_norm_weight(&mut norm, &weights, "model.norm.weight");

        mlx_rs::transforms::eval(weights.values())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            normalizer: (config.hidden_size as f32).sqrt(),
            final_logit_softcapping: config.final_logit_softcapping,
            config: config.clone(),
        })
    }
}

impl super::MlxModel for MlxQuantizedGemma2ForCausalLM {
    fn forward(
        &mut self,
        input_ids: &Array,
        _positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        let offset = rope_offset.unwrap_or(0);
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = hidden_states.multiply(Array::from_f32(self.normalizer))?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, offset, &mut kv_cache[i])?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        let logits = self.embed_tokens.as_linear(&hidden_states)?;

        let logits = if let Some(cap) = self.final_logit_softcapping {
            MlxGemma2ForCausalLM::apply_softcap(&logits, cap)?
        } else {
            logits
        };

        Ok(logits)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn forward_batch(
        &mut self,
        input_ids: &Array,
        _positions: &Array,
        batch_info: &MlxBatchInfo,
        kv_caches: &mut [MlxKvCache],
    ) -> mlx_rs::error::Result<Array> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = hidden_states.multiply(Array::from_f32(self.normalizer))?;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let mut layer_caches: Vec<&mut Option<MlxLayerKvCache>> =
                kv_caches.iter_mut().map(|kv| &mut kv[layer_idx]).collect();
            let mut temp_caches: Vec<Option<MlxLayerKvCache>> =
                layer_caches.iter_mut().map(|c| c.take()).collect();

            hidden_states = layer.forward_batch(&hidden_states, batch_info, &mut temp_caches)?;

            for (dst, src) in layer_caches.iter_mut().zip(temp_caches.into_iter()) {
                **dst = src;
            }
        }

        hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self.embed_tokens.as_linear(&hidden_states)?;
        if let Some(cap) = self.final_logit_softcapping {
            return MlxGemma2ForCausalLM::apply_softcap(&logits, cap);
        }
        Ok(logits)
    }

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        _positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut kv_cache: MlxKvCache = (0..self.layers.len()).map(|_| None).collect();
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = hidden_states.multiply(Array::from_f32(self.normalizer))?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, 0, &mut kv_cache[i])?;
        }
        self.norm.forward(&hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/// Factory function for creating a float MLX Gemma2 model.
pub fn create_mlx_gemma2(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let gemma2_config = MlxGemma2Config::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let model = MlxGemma2ForCausalLM::load(model_dir, &gemma2_config, dtype)?;
    Ok(Box::new(model))
}

/// Factory function for creating a quantized MLX Gemma2 model.
pub fn create_mlx_quantized_gemma2(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let gemma2_config = MlxGemma2Config::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();
    tracing::info!(
        "Loading quantized MLX Gemma2 (group_size={}, bits={})",
        qc.group_size,
        qc.bits
    );
    let model = MlxQuantizedGemma2ForCausalLM::load(model_dir, &gemma2_config, &qc, dtype)?;
    Ok(Box::new(model))
}

// ===========================================================================
// Gemma v1 — same as Gemma2 but with 2 norms per layer (no pre/post-ff norms,
// no softcapping, no query_pre_attn_scalar). Reuses MlxGemma2Config (those
// fields default to None/head_dim when absent in config.json).
// ===========================================================================

// ---------------------------------------------------------------------------
// MlxGemmaDecoderLayer (2 norms)
// ---------------------------------------------------------------------------

struct MlxGemmaDecoderLayer {
    self_attn: MlxGemma2Attention,
    mlp: MlxGemma2MLP,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
}

impl MlxGemmaDecoderLayer {
    fn new(config: &MlxGemma2Config) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: MlxGemma2Attention::new(config, None)?, // Gemma v1: no sliding window
            mlp: MlxGemma2MLP::new(
                config.hidden_size as i32,
                config.intermediate_size as i32,
                false,
            )?,
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
        self.mlp.load_weights(weights, &format!("{prefix}.mlp"));
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
        rope_offset: i32,
        cache: &mut Option<MlxLayerKvCache>,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed, rope_offset, cache)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        hidden_states.add(&mlp_output)
    }

    fn forward_batch(
        &mut self,
        hidden_states: &Array,
        batch_info: &MlxBatchInfo,
        caches: &mut [Option<MlxLayerKvCache>],
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward_batch(&normed, batch_info, caches)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        hidden_states.add(&mlp_output)
    }
}

// ---------------------------------------------------------------------------
// MlxGemmaForCausalLM (float)
// ---------------------------------------------------------------------------

pub struct MlxGemmaForCausalLM {
    embed_tokens: nn::Embedding,
    layers: Vec<MlxGemmaDecoderLayer>,
    norm: nn::RmsNorm,
    normalizer: f32,
    #[allow(dead_code)]
    config: MlxGemma2Config,
}

impl MlxGemmaForCausalLM {
    fn new(config: &MlxGemma2Config) -> Result<Self, Exception> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(MlxGemmaDecoderLayer::new(config)?);
        }

        Ok(Self {
            embed_tokens: nn::Embedding::new(config.vocab_size as i32, config.hidden_size as i32)?,
            layers,
            norm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            normalizer: (config.hidden_size as f32).sqrt(),
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
        assign_gemma_norm_weight(&mut self.norm, weights, "model.norm.weight");
    }

    fn load(
        model_dir: &Path,
        config: &MlxGemma2Config,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut model = Self::new(config)?;
        let weights = load_safetensors_weights(model_dir)?;
        model.load_weights(&weights);
        mlx_rs::transforms::eval(weights.values())?;
        Ok(model)
    }
}

impl super::MlxModel for MlxGemmaForCausalLM {
    fn inject_lora(
        &mut self,
        adapter: &crate::lora::MlxLoraAdapter,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::lora::merge_lora_into_weight;
        let targets = &adapter.config.target_modules;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Attention projections.
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
            // MLP projections.
            let mlp_prefix = format!("model.layers.{}.mlp", i);
            for name in ["gate_proj", "up_proj", "down_proj"] {
                if targets.iter().any(|t| t == name) {
                    let key = format!("{}.{}", mlp_prefix, name);
                    if let Some((a, b)) = adapter.weights.get(&key) {
                        let proj = match name {
                            "gate_proj" => &mut layer.mlp.gate_proj,
                            "up_proj" => &mut layer.mlp.up_proj,
                            "down_proj" => &mut layer.mlp.down_proj,
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
        _positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        let offset = rope_offset.unwrap_or(0);
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = hidden_states.multiply(Array::from_f32(self.normalizer))?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, offset, &mut kv_cache[i])?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        // Gemma v1 always uses tied embeddings.
        let logits = self.embed_tokens.as_linear(&hidden_states)?;
        Ok(logits)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn forward_batch(
        &mut self,
        input_ids: &Array,
        _positions: &Array,
        batch_info: &MlxBatchInfo,
        kv_caches: &mut [MlxKvCache],
    ) -> mlx_rs::error::Result<Array> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = hidden_states.multiply(Array::from_f32(self.normalizer))?;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let mut layer_caches: Vec<&mut Option<MlxLayerKvCache>> =
                kv_caches.iter_mut().map(|kv| &mut kv[layer_idx]).collect();
            let mut temp_caches: Vec<Option<MlxLayerKvCache>> =
                layer_caches.iter_mut().map(|c| c.take()).collect();

            hidden_states = layer.forward_batch(&hidden_states, batch_info, &mut temp_caches)?;

            for (dst, src) in layer_caches.iter_mut().zip(temp_caches.into_iter()) {
                **dst = src;
            }
        }

        hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self.embed_tokens.as_linear(&hidden_states)?;
        Ok(logits)
    }

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        _positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut kv_cache: MlxKvCache = (0..self.layers.len()).map(|_| None).collect();
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = hidden_states.multiply(Array::from_f32(self.normalizer))?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, 0, &mut kv_cache[i])?;
        }
        self.norm.forward(&hidden_states)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedGemmaForCausalLM
// ---------------------------------------------------------------------------

struct MlxQuantizedGemmaDecoderLayer {
    self_attn: MlxQuantizedGemma2Attention,
    mlp: MlxQuantizedGemma2MLP,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
}

impl MlxQuantizedGemmaDecoderLayer {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &MlxGemma2Config,
        qc: &QuantConfig,
    ) -> Result<Self, Exception> {
        let mut input_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        let mut post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;

        assign_gemma_norm_weight(
            &mut input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm.weight"),
        );
        assign_gemma_norm_weight(
            &mut post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        );

        Ok(Self {
            self_attn: MlxQuantizedGemma2Attention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config,
                qc,
                None, // Gemma v1 does not use sliding window
            ),
            mlp: MlxQuantizedGemma2MLP::from_weights(weights, &format!("{prefix}.mlp"), qc, false),
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        rope_offset: i32,
        cache: &mut Option<MlxLayerKvCache>,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed, rope_offset, cache)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        hidden_states.add(&mlp_output)
    }

    fn forward_batch(
        &mut self,
        hidden_states: &Array,
        batch_info: &MlxBatchInfo,
        caches: &mut [Option<MlxLayerKvCache>],
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward_batch(&normed, batch_info, caches)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        hidden_states.add(&mlp_output)
    }
}

pub struct MlxQuantizedGemmaForCausalLM {
    embed_tokens: MlxEmbedTokens,
    layers: Vec<MlxQuantizedGemmaDecoderLayer>,
    norm: nn::RmsNorm,
    normalizer: f32,
    #[allow(dead_code)]
    config: MlxGemma2Config,
}

impl MlxQuantizedGemmaForCausalLM {
    fn load(
        model_dir: &Path,
        config: &MlxGemma2Config,
        qc: &QuantConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let weights = load_safetensors_weights(model_dir)?;

        let embed_tokens =
            MlxEmbedTokens::from_weights(&weights, "model.embed_tokens", qc.group_size, qc.bits);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MlxQuantizedGemmaDecoderLayer::from_weights(
                &weights,
                &format!("model.layers.{i}"),
                config,
                qc,
            )?);
        }

        let mut norm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        assign_gemma_norm_weight(&mut norm, &weights, "model.norm.weight");

        mlx_rs::transforms::eval(weights.values())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            normalizer: (config.hidden_size as f32).sqrt(),
            config: config.clone(),
        })
    }
}

impl super::MlxModel for MlxQuantizedGemmaForCausalLM {
    fn forward(
        &mut self,
        input_ids: &Array,
        _positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        let offset = rope_offset.unwrap_or(0);
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = hidden_states.multiply(Array::from_f32(self.normalizer))?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, offset, &mut kv_cache[i])?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        let logits = self.embed_tokens.as_linear(&hidden_states)?;
        Ok(logits)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn forward_batch(
        &mut self,
        input_ids: &Array,
        _positions: &Array,
        batch_info: &MlxBatchInfo,
        kv_caches: &mut [MlxKvCache],
    ) -> mlx_rs::error::Result<Array> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = hidden_states.multiply(Array::from_f32(self.normalizer))?;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let mut layer_caches: Vec<&mut Option<MlxLayerKvCache>> =
                kv_caches.iter_mut().map(|kv| &mut kv[layer_idx]).collect();
            let mut temp_caches: Vec<Option<MlxLayerKvCache>> =
                layer_caches.iter_mut().map(|c| c.take()).collect();

            hidden_states = layer.forward_batch(&hidden_states, batch_info, &mut temp_caches)?;

            for (dst, src) in layer_caches.iter_mut().zip(temp_caches.into_iter()) {
                **dst = src;
            }
        }

        hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self.embed_tokens.as_linear(&hidden_states)?;
        Ok(logits)
    }

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        _positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut kv_cache: MlxKvCache = (0..self.layers.len()).map(|_| None).collect();
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = hidden_states.multiply(Array::from_f32(self.normalizer))?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, 0, &mut kv_cache[i])?;
        }
        self.norm.forward(&hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Gemma v1 factory functions
// ---------------------------------------------------------------------------

/// Factory function for creating a float MLX Gemma v1 model.
pub fn create_mlx_gemma(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let gemma_config = MlxGemma2Config::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let model = MlxGemmaForCausalLM::load(model_dir, &gemma_config, dtype)?;
    Ok(Box::new(model))
}

/// Factory function for creating a quantized MLX Gemma v1 model.
pub fn create_mlx_quantized_gemma(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let gemma_config = MlxGemma2Config::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();
    tracing::info!(
        "Loading quantized MLX Gemma (group_size={}, bits={})",
        qc.group_size,
        qc.bits
    );
    let model = MlxQuantizedGemmaForCausalLM::load(model_dir, &gemma_config, &qc, dtype)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::indexing::IndexOp;

    fn test_config() -> MlxGemma2Config {
        MlxGemma2Config {
            hidden_size: 32,
            num_attention_heads: 4,
            num_kv_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            head_dim: 8,
            query_pre_attn_scalar: 8.0,
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
            attention_bias: false,
            sliding_window: None,
            layer_is_sliding: Vec::new(),
        }
    }

    #[test]
    fn test_gemma2_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["Gemma2ForCausalLM"],
                "hidden_size": 2304,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "num_hidden_layers": 26,
                "intermediate_size": 9216,
                "vocab_size": 256000,
                "max_position_embeddings": 8192,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "head_dim": 256,
                "query_pre_attn_scalar": 256,
                "attn_logit_softcapping": 50.0,
                "final_logit_softcapping": 30.0,
                "attention_bias": false
            }"#,
        )
        .unwrap();

        let config = MlxGemma2Config::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 2304);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.head_dim, 256);
        assert!((config.query_pre_attn_scalar - 256.0).abs() < 1.0);
        assert!((config.attn_logit_softcapping.unwrap() - 50.0).abs() < 1.0);
        assert!((config.final_logit_softcapping.unwrap() - 30.0).abs() < 1.0);
        assert!(!config.attention_bias);
    }

    #[test]
    fn test_gemma_norm_offset() {
        // Verify assign_gemma_norm_weight produces weight + 1.0.
        let mut norm = nn::RmsNormBuilder::new(4).eps(1e-6).build().unwrap();
        let mut weights = HashMap::new();
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[4]);
        weights.insert("test.weight".to_string(), w);
        assign_gemma_norm_weight(&mut norm, &weights, "test.weight");
        norm.weight.value.eval().unwrap();
        let vals: Vec<f32> = (0..4)
            .map(|i| norm.weight.value.index(i).item::<f32>())
            .collect();
        assert!((vals[0] - 1.1).abs() < 1e-5);
        assert!((vals[1] - 1.2).abs() < 1e-5);
        assert!((vals[2] - 1.3).abs() < 1e-5);
        assert!((vals[3] - 1.4).abs() < 1e-5);
    }

    #[test]
    fn test_gemma2_mlp_forward() {
        let mut mlp = MlxGemma2MLP::new(32, 64, true).unwrap();
        let x = mlx_rs::ops::ones::<f32>(&[3, 32]).unwrap();
        let out = mlp.forward(&x).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[3, 32]);
    }

    #[test]
    fn test_gemma2_attention_forward() {
        let config = test_config();
        let mut attn = MlxGemma2Attention::new(&config, None).unwrap();

        let x = mlx_rs::ops::ones::<f32>(&[4, 32]).unwrap();
        let mut cache = None;

        let out = attn.forward(&x, 0, &mut cache).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[4, 32]);
        assert!(cache.is_some());
    }

    #[test]
    fn test_gemma2_model_forward() {
        let config = test_config();
        let mut model = MlxGemma2ForCausalLM::new(&config).unwrap();

        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        let logits = <MlxGemma2ForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &input_ids,
            &positions,
            &mut kv_cache,
            Some(0),
        )
        .unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.shape(), &[3, config.vocab_size as i32]);
    }

    #[test]
    fn test_gemma2_prefill_and_decode() {
        let config = test_config();
        let mut model = MlxGemma2ForCausalLM::new(&config).unwrap();
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        // Prefill: 3 tokens
        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let logits = <MlxGemma2ForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &input_ids,
            &positions,
            &mut kv_cache,
            Some(0),
        )
        .unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.shape(), &[3, config.vocab_size as i32]);

        for entry in &kv_cache {
            assert!(entry.is_some());
        }

        // Decode: 1 token at position 3
        let decode_ids = Array::from_iter(vec![15i32], &[1]);
        let decode_pos = Array::from_iter(vec![3i32], &[1]);
        let logits2 = <MlxGemma2ForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &decode_ids,
            &decode_pos,
            &mut kv_cache,
            Some(3),
        )
        .unwrap();
        logits2.eval().unwrap();
        assert_eq!(logits2.shape(), &[1, config.vocab_size as i32]);
    }

    #[test]
    fn test_gemma2_logit_softcapping() {
        // Verify cap * tanh(logits / cap) bounds output.
        let cap = 30.0f32;
        let logits = Array::from_slice(&[100.0f32, -100.0, 0.0, 15.0], &[1, 4]);
        let capped = MlxGemma2ForCausalLM::apply_softcap(&logits, cap).unwrap();
        capped.eval().unwrap();

        let vals: Vec<f32> = (0..4).map(|i| capped.index((0, i)).item::<f32>()).collect();
        // 100/30 -> tanh -> ~1.0 -> *30 -> ~30
        assert!((vals[0] - 30.0).abs() < 0.1);
        // -100/30 -> tanh -> ~-1.0 -> *30 -> ~-30
        assert!((vals[1] - (-30.0)).abs() < 0.1);
        // 0 -> tanh -> 0 -> 0
        assert!(vals[2].abs() < 0.01);
        // 15/30 = 0.5 -> tanh(0.5) -> ~0.462 -> *30 -> ~13.86
        assert!((vals[3] - 13.86).abs() < 0.1);
    }

    // -----------------------------------------------------------------------
    // Gemma v1 tests
    // -----------------------------------------------------------------------

    fn gemma_v1_config() -> MlxGemma2Config {
        MlxGemma2Config {
            hidden_size: 32,
            num_attention_heads: 4,
            num_kv_heads: 2, // GQA
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            head_dim: 8,
            query_pre_attn_scalar: 8.0, // defaults to head_dim when absent
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            attention_bias: false,
            sliding_window: None,
            layer_is_sliding: Vec::new(),
        }
    }

    #[test]
    fn test_gemma_v1_model_forward() {
        let config = gemma_v1_config();
        let mut model = MlxGemmaForCausalLM::new(&config).unwrap();

        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        let logits = <MlxGemmaForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &input_ids,
            &positions,
            &mut kv_cache,
            Some(0),
        )
        .unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.shape(), &[3, config.vocab_size as i32]);
    }

    #[test]
    fn test_gemma_v1_prefill_and_decode() {
        let config = gemma_v1_config();
        let mut model = MlxGemmaForCausalLM::new(&config).unwrap();
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        // Prefill
        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let logits = <MlxGemmaForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &input_ids,
            &positions,
            &mut kv_cache,
            Some(0),
        )
        .unwrap();
        logits.eval().unwrap();

        for entry in &kv_cache {
            assert!(entry.is_some());
        }

        // Decode
        let decode_ids = Array::from_iter(vec![15i32], &[1]);
        let decode_pos = Array::from_iter(vec![3i32], &[1]);
        let logits2 = <MlxGemmaForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &decode_ids,
            &decode_pos,
            &mut kv_cache,
            Some(3),
        )
        .unwrap();
        logits2.eval().unwrap();
        assert_eq!(logits2.shape(), &[1, config.vocab_size as i32]);
    }
}
