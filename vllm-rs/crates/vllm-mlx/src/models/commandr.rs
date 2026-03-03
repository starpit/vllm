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
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use crate::cache::{MlxBatchInfo, MlxKvCache, MlxLayerKvCache};
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
        _positions: &Array,
        cache: &mut Option<MlxLayerKvCache>,
        rope_offset: i32,
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
        let k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let v = v
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        // RoPE (interleaved via traditional=true): offset passed from caller (avoids .item() sync).
        let q = self.rope.forward((&q, rope_offset))?;
        let k = self.rope.forward((&k, rope_offset))?;

        // KV cache update — pre-allocated buffer with O(1) slice_update.
        let (k, v) = crate::cache::kv_cache_update(cache, &k, &v)?;

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

        // Check if all requests are decode (q_len=1) → can attempt batched SDPA.
        // CommandR has no sliding window.
        let all_decode = batch_info.num_reqs > 1 && batch_info.q_lens.iter().all(|&ql| ql == 1);

        // Per-request: reshape, optional QK norm, RoPE, KV cache update.
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

            // Optional QK norms (applied per-head before RoPE).
            let q = if let Some(ref mut norm) = self.q_norm {
                norm.forward(&q)?
            } else {
                q
            };
            let k = if let Some(ref mut norm) = self.k_norm {
                norm.forward(&k)?
            } else {
                k
            };

            let q = q.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
            let mut k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
            let v = v
                .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
                .transpose_axes(&[1, 0, 2])?
                .expand_dims(0)?;

            let q = self.rope.forward((&q, offset))?;
            k = self.rope.forward((&k, offset))?;

            let (k, v) = crate::cache::kv_cache_update(&mut caches[i], &k, &v)?;

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

            // Single SDPA: q_len=1 decode → no mask needed.
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
                let k = per_req_k[i].clone();
                let v = per_req_v[i].clone();

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
        cache: &mut Option<MlxLayerKvCache>,
        rope_offset: i32,
    ) -> Result<Array, Exception> {
        let residual = hidden_states;

        // Single norm (shared by attn and mlp).
        let normed = self.input_layernorm.forward(hidden_states)?;

        // Parallel attention + MLP.
        let attn_output = self
            .self_attn
            .forward(&normed, positions, cache, rope_offset)?;
        let mlp_output = self.mlp.forward(&normed)?;

        // residual + attn_output + mlp_output
        residual.add(&attn_output)?.add(&mlp_output)
    }

    fn forward_batch(
        &mut self,
        hidden_states: &Array,
        batch_info: &MlxBatchInfo,
        caches: &mut [Option<MlxLayerKvCache>],
    ) -> Result<Array, Exception> {
        let residual = hidden_states;

        // Single norm (shared by attn and mlp) — batched over [total_tokens, hidden].
        let normed = self.input_layernorm.forward(hidden_states)?;

        // Parallel attention + MLP.
        let attn_output = self.self_attn.forward_batch(&normed, batch_info, caches)?;
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
        positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        let offset = rope_offset.unwrap_or(0);
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i], offset)?;
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

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Collect the per-layer cache entry from each request's KV cache.
            let mut layer_caches: Vec<&mut Option<MlxLayerKvCache>> =
                kv_caches.iter_mut().map(|kv| &mut kv[layer_idx]).collect();

            // Build a contiguous slice for the layer's forward_batch.
            let mut temp_caches: Vec<Option<MlxLayerKvCache>> =
                layer_caches.iter_mut().map(|c| c.take()).collect();

            hidden_states = layer.forward_batch(&hidden_states, batch_info, &mut temp_caches)?;

            // Put caches back.
            for (dst, src) in layer_caches.iter_mut().zip(temp_caches.into_iter()) {
                **dst = src;
            }
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

        Ok(logits)
    }

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut kv_cache: MlxKvCache = (0..self.layers.len()).map(|_| None).collect();
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i], 0)?;
        }
        self.norm.forward(&hidden_states)
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
        _positions: &Array,
        cache: &mut Option<MlxLayerKvCache>,
        rope_offset: i32,
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
        let k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let v = v
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        // RoPE (interleaved via traditional=true): offset passed from caller (avoids .item() sync).
        let q = self.rope.forward((&q, rope_offset))?;
        let k = self.rope.forward((&k, rope_offset))?;

        // KV cache update — pre-allocated buffer with O(1) slice_update.
        let (k, v) = crate::cache::kv_cache_update(cache, &k, &v)?;

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

        // Check if all requests are decode (q_len=1) → can attempt batched SDPA.
        // CommandR has no sliding window.
        let all_decode = batch_info.num_reqs > 1 && batch_info.q_lens.iter().all(|&ql| ql == 1);

        // Per-request: reshape, optional QK norm, RoPE, KV cache update.
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

            // Optional QK norms (applied per-head before RoPE).
            let q = if let Some(ref mut norm) = self.q_norm {
                norm.forward(&q)?
            } else {
                q
            };
            let k = if let Some(ref mut norm) = self.k_norm {
                norm.forward(&k)?
            } else {
                k
            };

            let q = q.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
            let mut k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
            let v = v
                .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
                .transpose_axes(&[1, 0, 2])?
                .expand_dims(0)?;

            let q = self.rope.forward((&q, offset))?;
            k = self.rope.forward((&k, offset))?;

            let (k, v) = crate::cache::kv_cache_update(&mut caches[i], &k, &v)?;

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

            // Single SDPA: q_len=1 decode → no mask needed.
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
                let k = per_req_k[i].clone();
                let v = per_req_v[i].clone();

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
        cache: &mut Option<MlxLayerKvCache>,
        rope_offset: i32,
    ) -> Result<Array, Exception> {
        let residual = hidden_states;
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self
            .self_attn
            .forward(&normed, positions, cache, rope_offset)?;
        let mlp_output = self.mlp.forward(&normed)?;
        residual.add(&attn_output)?.add(&mlp_output)
    }

    fn forward_batch(
        &mut self,
        hidden_states: &Array,
        batch_info: &MlxBatchInfo,
        caches: &mut [Option<MlxLayerKvCache>],
    ) -> Result<Array, Exception> {
        let residual = hidden_states;

        // Single norm (shared by attn and mlp) — batched over [total_tokens, hidden].
        let normed = self.input_layernorm.forward(hidden_states)?;

        // Parallel attention + MLP.
        let attn_output = self.self_attn.forward_batch(&normed, batch_info, caches)?;
        let mlp_output = self.mlp.forward(&normed)?;

        // residual + attn_output + mlp_output
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
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        let offset = rope_offset.unwrap_or(0);
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i], offset)?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        let logits = if self.tie_word_embeddings {
            self.embed_tokens.as_linear(&hidden_states)?
        } else {
            self.lm_head.as_mut().unwrap().forward(&hidden_states)?
        };

        let logits = logits.multiply(Array::from_f32(self.logit_scale))?;
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

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Collect the per-layer cache entry from each request's KV cache.
            let mut layer_caches: Vec<&mut Option<MlxLayerKvCache>> =
                kv_caches.iter_mut().map(|kv| &mut kv[layer_idx]).collect();

            // Build a contiguous slice for the layer's forward_batch.
            let mut temp_caches: Vec<Option<MlxLayerKvCache>> =
                layer_caches.iter_mut().map(|c| c.take()).collect();

            hidden_states = layer.forward_batch(&hidden_states, batch_info, &mut temp_caches)?;

            // Put caches back.
            for (dst, src) in layer_caches.iter_mut().zip(temp_caches.into_iter()) {
                **dst = src;
            }
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        let logits = if self.tie_word_embeddings {
            self.embed_tokens.as_linear(&hidden_states)?
        } else {
            self.lm_head.as_mut().unwrap().forward(&hidden_states)?
        };

        let logits = logits.multiply(Array::from_f32(self.logit_scale))?;
        Ok(logits)
    }

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut kv_cache: MlxKvCache = (0..self.layers.len()).map(|_| None).collect();
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i], 0)?;
        }
        self.norm.forward(&hidden_states)
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
