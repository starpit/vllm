// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Quantized LLaMA model architecture for MLX.
//!
//! Loads mlx-community pre-quantized safetensors (4-bit group quantization):
//! `nn::QuantizedLinear` for Q/K/V/O projections and MLP linear layers,
//! `nn::QuantizedEmbedding` for embed_tokens and optionally lm_head.
//! Norms stay `nn::RmsNorm` (f32). RoPE is unchanged (half-split HF convention).
//!
//! Weight format (per linear layer):
//! - `*.weight` — packed u32 quantized weights
//! - `*.scales` — f16 quantization scales
//! - `*.biases` — f16 quantization biases

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use crate::cache::{BatchMlxLayerKvCache, MlxBatchInfo, MlxKvCache, MlxLayerKvCache};
use crate::models::llama::{LlamaConfig, load_safetensors_weights};
use vllm_model::weight::HfModelConfig;

// ---------------------------------------------------------------------------
// Quantization config
// ---------------------------------------------------------------------------

/// Quantization parameters parsed from config.json.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub group_size: i32,
    pub bits: i32,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            group_size: 64,
            bits: 4,
        }
    }
}

impl QuantConfig {
    /// Parse from the `"quantization"` field in config.json extras.
    pub fn from_hf_config(config: &HfModelConfig) -> Option<Self> {
        let quant = config.extra.get("quantization")?;
        let obj = quant.as_object()?;
        let group_size = obj.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64) as i32;
        let bits = obj.get("bits").and_then(|v| v.as_i64()).unwrap_or(4) as i32;
        Some(Self { group_size, bits })
    }
}

// ---------------------------------------------------------------------------
// Helpers: construct quantized layers from loaded weights
// ---------------------------------------------------------------------------

/// Construct a `QuantizedLinear` directly from loaded weight arrays.
/// Avoids the builder's wasteful random-init + quantize step.
pub(crate) fn make_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> nn::QuantizedLinear {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .cloned()
        .unwrap_or_else(|| {
            tracing::warn!("Weight not found: {prefix}.weight");
            Array::from_f32(0.0)
        });
    let scales = weights
        .get(&format!("{prefix}.scales"))
        .cloned()
        .unwrap_or_else(|| {
            tracing::warn!("Weight not found: {prefix}.scales");
            Array::from_f32(0.0)
        });
    let biases = weights
        .get(&format!("{prefix}.biases"))
        .cloned()
        .unwrap_or_else(|| {
            tracing::warn!("Weight not found: {prefix}.biases");
            Array::from_f32(0.0)
        });

    nn::QuantizedLinear {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner: nn::Linear {
            weight: Param::new(weight),
            bias: Param::new(None),
        },
    }
}

/// Construct a `QuantizedEmbedding` directly from loaded weight arrays.
pub(crate) fn make_quantized_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> nn::QuantizedEmbedding {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .cloned()
        .unwrap_or_else(|| {
            tracing::warn!("Weight not found: {prefix}.weight");
            Array::from_f32(0.0)
        });
    let scales = weights
        .get(&format!("{prefix}.scales"))
        .cloned()
        .unwrap_or_else(|| {
            tracing::warn!("Weight not found: {prefix}.scales");
            Array::from_f32(0.0)
        });
    let biases = weights
        .get(&format!("{prefix}.biases"))
        .cloned()
        .unwrap_or_else(|| {
            tracing::warn!("Weight not found: {prefix}.biases");
            Array::from_f32(0.0)
        });

    nn::QuantizedEmbedding {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner: nn::Embedding {
            weight: Param::new(weight),
        },
    }
}

/// Look up a weight by name and assign it.
pub(crate) fn assign_weight(
    target: &mut Param<Array>,
    weights: &HashMap<String, Array>,
    name: &str,
) {
    if let Some(w) = weights.get(name) {
        target.value = w.clone();
    } else {
        tracing::warn!("Weight not found: {name}");
    }
}

// ---------------------------------------------------------------------------
// MlxEmbedTokens / MlxLmHead — handle mixed quantized/float weights
// ---------------------------------------------------------------------------

/// Embedding that may or may not be quantized.
///
/// mlx-community quantized models often leave the embedding layer as float16
/// (no scales/biases). This enum dispatches to the right implementation.
pub(crate) enum MlxEmbedTokens {
    Quantized(nn::QuantizedEmbedding),
    Float(nn::Embedding),
}

impl MlxEmbedTokens {
    /// Create from loaded weights. Uses `QuantizedEmbedding` if `{prefix}.scales`
    /// exists, otherwise falls back to a regular float `Embedding`.
    pub(crate) fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Self {
        if weights.contains_key(&format!("{prefix}.scales")) {
            Self::Quantized(make_quantized_embedding(weights, prefix, group_size, bits))
        } else if let Some(w) = weights.get(&format!("{prefix}.weight")) {
            tracing::info!("Embedding at {prefix} is not quantized, using float");
            let mut emb =
                nn::Embedding::new(w.dim(0), w.dim(1)).expect("failed to create Embedding");
            emb.weight.value = w.clone();
            Self::Float(emb)
        } else {
            tracing::warn!("Embedding weights not found at {prefix}, creating quantized stub");
            Self::Quantized(make_quantized_embedding(weights, prefix, group_size, bits))
        }
    }

    pub(crate) fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        match self {
            Self::Quantized(emb) => emb.forward(x),
            Self::Float(emb) => emb.forward(x),
        }
    }

    pub(crate) fn as_linear(&self, x: &Array) -> Result<Array, Exception> {
        match self {
            Self::Quantized(emb) => emb.as_linear(x),
            Self::Float(emb) => emb.as_linear(x),
        }
    }
}

/// LM head that may or may not be quantized.
///
/// Same rationale as `MlxEmbedTokens` — some quantized models leave lm_head
/// as float.
pub(crate) enum MlxLmHead {
    Quantized(nn::QuantizedLinear),
    Float(nn::Linear),
}

impl MlxLmHead {
    /// Create from loaded weights. Uses `QuantizedLinear` if `{prefix}.scales`
    /// exists, otherwise falls back to a regular float `Linear`.
    pub(crate) fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Self {
        if weights.contains_key(&format!("{prefix}.scales")) {
            Self::Quantized(make_quantized_linear(weights, prefix, group_size, bits))
        } else if let Some(w) = weights.get(&format!("{prefix}.weight")) {
            tracing::info!("LM head at {prefix} is not quantized, using float");
            Self::Float(nn::Linear {
                weight: Param::new(w.clone()),
                bias: Param::new(None),
            })
        } else {
            tracing::warn!("LM head weights not found at {prefix}, creating quantized stub");
            Self::Quantized(make_quantized_linear(weights, prefix, group_size, bits))
        }
    }

    pub(crate) fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        match self {
            Self::Quantized(lin) => lin.forward(x),
            Self::Float(lin) => lin.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedLlamaMLP
// ---------------------------------------------------------------------------

/// Quantized LLaMA MLP (SiLU-gated feed-forward network).
pub struct MlxQuantizedLlamaMLP {
    pub(crate) gate_proj: nn::QuantizedLinear,
    pub(crate) up_proj: nn::QuantizedLinear,
    pub(crate) down_proj: nn::QuantizedLinear,
}

impl MlxQuantizedLlamaMLP {
    /// Create from loaded weights.
    pub fn from_weights(weights: &HashMap<String, Array>, prefix: &str, qc: &QuantConfig) -> Self {
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
        }
    }

    /// Forward pass: gate_proj(x) → SwiGLU(gate, up) → down_proj
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let hidden = super::llama::swiglu(&gate, &up)?;
        self.down_proj.forward(&hidden)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedLlamaAttention
// ---------------------------------------------------------------------------

/// Quantized LLaMA multi-head attention with RoPE and optional GQA.
pub struct MlxQuantizedLlamaAttention {
    q_proj: nn::QuantizedLinear,
    k_proj: nn::QuantizedLinear,
    v_proj: nn::QuantizedLinear,
    o_proj: nn::QuantizedLinear,
    /// Optional per-head Q norm (Qwen3 uses this).
    q_norm: Option<nn::RmsNorm>,
    /// Optional per-head K norm (Qwen3 uses this).
    k_norm: Option<nn::RmsNorm>,
    rope: nn::Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pub(crate) scale: f32,
    sliding_window: Option<usize>,
    /// When true, K is stored without RoPE and RoPE is applied to the full
    /// cached K at attention time (relocatable span blocks).
    fuse_rope: bool,
}

impl MlxQuantizedLlamaAttention {
    /// Create from loaded weights.
    pub fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &LlamaConfig,
        qc: &QuantConfig,
    ) -> Self {
        // Optional QK norms (Qwen3 has per-head q_norm and k_norm).
        let q_norm = weights
            .get(&format!("{prefix}.q_norm.weight"))
            .and_then(|w| {
                let head_dim = w.dim(0);
                let mut norm = nn::RmsNormBuilder::new(head_dim).eps(1e-6).build().ok()?;
                norm.weight.value = w.clone();
                Some(norm)
            });
        let k_norm = weights
            .get(&format!("{prefix}.k_norm.weight"))
            .and_then(|w| {
                let head_dim = w.dim(0);
                let mut norm = nn::RmsNormBuilder::new(head_dim).eps(1e-6).build().ok()?;
                norm.weight.value = w.clone();
                Some(norm)
            });

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
            q_norm,
            k_norm,
            rope: {
                let mut r = nn::Rope::new(config.head_dim as i32);
                r.base = config.rope_theta;
                r
            },
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
            sliding_window: config.sliding_window,
            fuse_rope: vllm_config::SpansConfig::from_env().fuse_rope(),
        }
    }

    /// Forward pass.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        _positions: &Array,
        cache: &mut Option<MlxLayerKvCache>,
        rope_offset: i32,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);

        // Q/K/V projections.
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape: [seq, hidden] -> [seq, heads, head_dim]
        let q = q.reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?;
        let k = k.reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?;

        // Apply optional per-head QK norms (Qwen3).
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

        // [seq, heads, head_dim] -> [1, heads, seq, head_dim]
        let q = q.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let v = v
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        // RoPE + KV cache update.
        // When fuse_rope: use rope_dynamic (via apply_rope_to_cached_k) for both Q and K
        // to ensure consistency. MLX's fast::rope and fast::rope_dynamic produce
        // different values for the same position, so Q and K must use the same impl.
        let (q, mut k, mut v) = if self.fuse_rope {
            let q = crate::models::llama::apply_rope_to_cached_k(&q, &self.rope, rope_offset)?;
            let (mut k, v) = crate::cache::kv_cache_update(cache, &k, &v)?;
            k = crate::models::llama::apply_rope_to_cached_k(&k, &self.rope, 0)?;
            (q, k, v)
        } else {
            let q = self.rope.forward((&q, rope_offset))?;
            k = self.rope.forward((&k, rope_offset))?;
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

    /// Batched forward: projections batched, split per-request for RoPE/KV/SDPA.
    ///
    /// When all requests are decode (q_len=1) and KV lengths match, SDPA is
    /// batched into a single kernel launch.
    pub fn forward_batch(
        &mut self,
        hidden_states: &Array,
        batch_info: &MlxBatchInfo,
        caches: &mut [Option<MlxLayerKvCache>],
    ) -> Result<Array, Exception> {
        let q_all = self.q_proj.forward(hidden_states)?;
        let k_all = self.k_proj.forward(hidden_states)?;
        let v_all = self.v_proj.forward(hidden_states)?;

        let all_decode = batch_info.num_reqs > 1
            && batch_info.q_lens.iter().all(|&ql| ql == 1)
            && self.sliding_window.is_none();

        let mut per_req_q = Vec::with_capacity(batch_info.num_reqs);
        let mut per_req_k = Vec::with_capacity(batch_info.num_reqs);
        let mut per_req_v = Vec::with_capacity(batch_info.num_reqs);
        let mut kv_lens = Vec::with_capacity(batch_info.num_reqs);

        if all_decode {
            let n = batch_info.num_reqs as i32;
            let nh = self.num_heads as i32;
            let nkv = self.num_kv_heads as i32;
            let hd = self.head_dim as i32;

            let mut q = q_all.reshape(&[n, nh, 1, hd])?;
            let mut k = k_all.reshape(&[n, nkv, 1, hd])?;
            let v = v_all.reshape(&[n, nkv, 1, hd])?;

            if let Some(ref mut norm) = self.q_norm {
                q = norm.forward(&q.squeeze_axes(&[2])?)?.expand_dims(2)?;
            }
            if let Some(ref mut norm) = self.k_norm {
                k = norm.forward(&k.squeeze_axes(&[2])?)?.expand_dims(2)?;
            }

            // Per-request RoPE + KV cache update.
            for (i, cache) in caches.iter_mut().enumerate().take(batch_info.num_reqs) {
                let ii = i as i32;
                let offset = batch_info.rope_offsets[i];
                let qi = q.try_index((ii..ii + 1, .., .., ..))?;
                let ki = k.try_index((ii..ii + 1, .., .., ..))?;
                let vi = v.try_index((ii..ii + 1, .., .., ..))?;

                let qi = self.rope.forward((&qi, offset))?;
                let ki = if self.fuse_rope {
                    ki
                } else {
                    self.rope.forward((&ki, offset))?
                };

                let (ki, vi) = crate::cache::kv_cache_update(cache, &ki, &vi)?;
                let ki = if self.fuse_rope {
                    crate::models::llama::apply_rope_to_cached_k(&ki, &self.rope, 0)?
                } else {
                    ki
                };
                kv_lens.push(ki.dim(2) as usize);
                per_req_q.push(qi);
                per_req_k.push(ki);
                per_req_v.push(vi);
            }
        } else {
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

                let (q, k, v) = if self.fuse_rope {
                    let q = crate::models::llama::apply_rope_to_cached_k(&q, &self.rope, offset)?;
                    let (k, v) = crate::cache::kv_cache_update(&mut caches[i], &k, &v)?;
                    let k = crate::models::llama::apply_rope_to_cached_k(&k, &self.rope, 0)?;
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
        }

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

    /// Batched decode forward: all N sequences processed with a single KV cache
    /// update and a single SDPA kernel launch.
    ///
    /// Requires: all requests are decode (q_len=1), no sliding window.
    /// `batch_cache` holds the persistent batched [B, heads, kv_len, dim] cache for this layer.
    /// `mask` is the pre-built left-padding mask (shared across all layers).
    /// `offsets_arr` is the pre-built rope offsets array (shared across all layers).
    /// `left_pads` — per-request left-padding counts (needed for fuse_rope negative offsets).
    pub fn forward_batch_decode(
        &mut self,
        hidden_states: &Array,
        batch_info: &MlxBatchInfo,
        batch_cache: &mut BatchMlxLayerKvCache,
        mask: &Option<Array>,
        _offsets_arr: &Array,
        left_pads: &[usize],
    ) -> Result<Array, Exception> {
        let n = batch_info.num_reqs as i32;
        let nh = self.num_heads as i32;
        let nkv = self.num_kv_heads as i32;
        let hd = self.head_dim as i32;

        // Batched Q/K/V projections: [N, hidden] -> ...
        let q_all = self.q_proj.forward(hidden_states)?;
        let k_all = self.k_proj.forward(hidden_states)?;
        let v_all = self.v_proj.forward(hidden_states)?;

        // Reshape to [N, heads, 1, hd]
        let mut q = q_all.reshape(&[n, nh, 1, hd])?;
        let mut k = k_all.reshape(&[n, nkv, 1, hd])?;
        let v = v_all.reshape(&[n, nkv, 1, hd])?;

        // Optional QK norms (batched).
        if let Some(ref mut norm) = self.q_norm {
            q = norm.forward(&q.squeeze_axes(&[2])?)?.expand_dims(2)?;
        }
        if let Some(ref mut norm) = self.k_norm {
            k = norm.forward(&k.squeeze_axes(&[2])?)?.expand_dims(2)?;
        }

        // Per-request RoPE (rope_dynamic not available in mlx-rs 0.25).
        {
            let mut q_parts = Vec::with_capacity(batch_info.num_reqs);
            let mut k_parts = Vec::with_capacity(batch_info.num_reqs);
            for i in 0..batch_info.num_reqs {
                let ii = i as i32;
                let offset = batch_info.rope_offsets[i];
                let qi = q.try_index((ii..ii + 1, .., .., ..))?;
                let ki = k.try_index((ii..ii + 1, .., .., ..))?;
                q_parts.push(self.rope.forward((&qi, offset))?);
                if self.fuse_rope {
                    k_parts.push(ki);
                } else {
                    k_parts.push(self.rope.forward((&ki, offset))?);
                }
            }
            let q_refs: Vec<&Array> = q_parts.iter().collect();
            let k_refs: Vec<&Array> = k_parts.iter().collect();
            q = mlx_rs::ops::concatenate_axis(&q_refs, 0)?;
            k = mlx_rs::ops::concatenate_axis(&k_refs, 0)?;
        }

        // Single batched KV cache update for all B sequences.
        // When fuse_rope, K is stored unrotated (position-independent).
        let (k_cached, v_cached) = batch_cache.update_and_view(&k, &v)?;

        // When fuse_rope, apply RoPE to full cached K via apply_rope_to_cached_k_batched.
        // The negative left_pads become start positions so real tokens get 0-based positions.
        let k_cached = if self.fuse_rope {
            let start_positions: Vec<i32> = left_pads.iter().map(|&p| -(p as i32)).collect();
            crate::models::llama::apply_rope_to_cached_k_batched(
                &k_cached,
                &self.rope,
                &start_positions,
            )?
        } else {
            k_cached
        };

        // Single SDPA for all B sequences (mask built once, shared across layers).
        let sdpa_mask = mask
            .as_ref()
            .map(mlx_rs::fast::ScaledDotProductAttentionMask::Array);
        let out = mlx_rs::fast::scaled_dot_product_attention(
            &q, &k_cached, &v_cached, self.scale, sdpa_mask,
        )?;

        // out: [B, heads, 1, hd] -> [B, hidden]
        let hidden = (self.num_heads * self.head_dim) as i32;
        let out = out.squeeze_axes(&[2])?.reshape(&[n, hidden])?;

        self.o_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedLlamaDecoderLayer
// ---------------------------------------------------------------------------

/// A single quantized LLaMA decoder layer.
pub struct MlxQuantizedLlamaDecoderLayer {
    self_attn: MlxQuantizedLlamaAttention,
    mlp: MlxQuantizedLlamaMLP,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
}

impl MlxQuantizedLlamaDecoderLayer {
    /// Create from loaded weights.
    pub fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &LlamaConfig,
        qc: &QuantConfig,
    ) -> Result<Self, Exception> {
        let mut input_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        let mut post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
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
            self_attn: MlxQuantizedLlamaAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config,
                qc,
            ),
            mlp: MlxQuantizedLlamaMLP::from_weights(weights, &format!("{prefix}.mlp"), qc),
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<MlxLayerKvCache>,
        rope_offset: i32,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self
            .self_attn
            .forward(&normed, positions, cache, rope_offset)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        hidden_states.add(&mlp_output)
    }

    /// Batched forward: norms and MLP batched, attention splits per-request.
    pub fn forward_batch(
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

    /// Batched decode forward using a persistent batched KV cache for this layer.
    pub fn forward_batch_decode(
        &mut self,
        hidden_states: &Array,
        batch_info: &MlxBatchInfo,
        batch_cache: &mut BatchMlxLayerKvCache,
        mask: &Option<Array>,
        offsets_arr: &Array,
        left_pads: &[usize],
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward_batch_decode(
            &normed,
            batch_info,
            batch_cache,
            mask,
            offsets_arr,
            left_pads,
        )?;
        let hidden_states = hidden_states.add(&attn_output)?;

        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        hidden_states.add(&mlp_output)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedLlamaForCausalLM
// ---------------------------------------------------------------------------

/// Quantized LLaMA for causal language modeling using MLX.
///
/// Uses `nn::QuantizedLinear` for projections and MLP.
/// Embedding and lm_head may be float if the quantized model didn't quantize them.
pub struct MlxQuantizedLlamaForCausalLM {
    embed_tokens: MlxEmbedTokens,
    layers: Vec<MlxQuantizedLlamaDecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: Option<MlxLmHead>,
    tie_word_embeddings: bool,
    #[allow(dead_code)]
    config: LlamaConfig,
}

impl MlxQuantizedLlamaForCausalLM {
    /// Load model directly from safetensors weights.
    pub fn load(
        model_dir: &Path,
        config: &LlamaConfig,
        qc: &QuantConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let weights = load_safetensors_weights(model_dir)?;

        // Build embedding (auto-detects quantized vs float).
        let embed_tokens =
            MlxEmbedTokens::from_weights(&weights, "model.embed_tokens", qc.group_size, qc.bits);

        // Build decoder layers.
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MlxQuantizedLlamaDecoderLayer::from_weights(
                &weights,
                &format!("model.layers.{i}"),
                config,
                qc,
            )?);
        }

        // Build final norm.
        let mut norm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        assign_weight(&mut norm.weight, &weights, "model.norm.weight");

        // Build lm_head (auto-detects quantized vs float).
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

        // Eval to materialize loaded weights.
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

impl MlxQuantizedLlamaForCausalLM {
    /// Load model from pre-loaded weights with a configurable prefix.
    ///
    /// `prefix` is the model backbone prefix (e.g., "model" or "language_model.model").
    /// The lm_head prefix is auto-detected: if the backbone prefix contains "language_model",
    /// the head is at "language_model.lm_head", otherwise "lm_head".
    pub fn from_weights_with_prefix(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &LlamaConfig,
        qc: &QuantConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let embed_tokens = MlxEmbedTokens::from_weights(
            weights,
            &format!("{prefix}.embed_tokens"),
            qc.group_size,
            qc.bits,
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MlxQuantizedLlamaDecoderLayer::from_weights(
                weights,
                &format!("{prefix}.layers.{i}"),
                config,
                qc,
            )?);
        }

        let mut norm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        assign_weight(&mut norm.weight, weights, &format!("{prefix}.norm.weight"));

        let head_prefix = if prefix.contains("language_model") {
            "language_model.lm_head"
        } else {
            "lm_head"
        };
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(MlxLmHead::from_weights(
                weights,
                head_prefix,
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

    /// Embed token IDs → hidden states.
    pub fn embed(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        self.embed_tokens.forward(input_ids)
    }

    /// Run backbone on embeddings → logits.
    pub fn forward_embeds(
        &mut self,
        inputs_embeds: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> Result<Array, Exception> {
        let offset = rope_offset.unwrap_or(0);
        let mut hidden_states = inputs_embeds.clone();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i], offset)?;
        }
        hidden_states = self.norm.forward(&hidden_states)?;

        let logits = if self.tie_word_embeddings {
            self.embed_tokens.as_linear(&hidden_states)?
        } else {
            self.lm_head.as_mut().unwrap().forward(&hidden_states)?
        };
        Ok(logits)
    }
}

impl super::MlxModel for MlxQuantizedLlamaForCausalLM {
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

        Ok(logits)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn supports_batch_decode(&self) -> bool {
        true
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

        let logits = if self.tie_word_embeddings {
            self.embed_tokens.as_linear(&hidden_states)?
        } else {
            self.lm_head.as_mut().unwrap().forward(&hidden_states)?
        };

        Ok(logits)
    }

    fn forward_batch_decode(
        &mut self,
        input_ids: &Array,
        batch_info: &MlxBatchInfo,
        layer_caches: &mut [BatchMlxLayerKvCache],
        left_padding: &[Vec<usize>],
    ) -> mlx_rs::error::Result<Array> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Hoist rope offsets: create once, reuse across all layers.
        let n = batch_info.num_reqs as i32;
        let offsets_arr = Array::from_iter(batch_info.rope_offsets.iter().copied(), &[n]);

        // Hoist mask: compute expected post-update kv_len and build once.
        // Decode appends 1 token, so post-update kv_len = current + 1.
        let kv_len = layer_caches[0].kv_len() + 1;
        let mask = BatchMlxLayerKvCache::build_left_padding_mask(
            &left_padding[0],
            kv_len,
            hidden_states.dtype(),
        )?;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward_batch_decode(
                &hidden_states,
                batch_info,
                &mut layer_caches[layer_idx],
                &mask,
                &offsets_arr,
                &left_padding[0],
            )?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        let logits = if self.tie_word_embeddings {
            self.embed_tokens.as_linear(&hidden_states)?
        } else {
            self.lm_head.as_mut().unwrap().forward(&hidden_states)?
        };

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
// Factory function
// ---------------------------------------------------------------------------

/// Factory function for creating a quantized MLX LLaMA model.
pub fn create_mlx_quantized_llama(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let llama_config = LlamaConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();
    tracing::info!(
        "Loading quantized MLX LLaMA (group_size={}, bits={})",
        qc.group_size,
        qc.bits
    );

    let model = MlxQuantizedLlamaForCausalLM::load(model_dir, &llama_config, &qc, dtype)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::builder::Builder;

    fn test_config() -> LlamaConfig {
        LlamaConfig {
            hidden_size: 32,
            num_attention_heads: 4,
            num_kv_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            head_dim: 8,
            tie_word_embeddings: true,
            sliding_window: None,
            partial_rotary_factor: 1.0,
            long_rope_scaling: None,
        }
    }

    fn test_quant_config() -> QuantConfig {
        QuantConfig {
            group_size: 32,
            bits: 4,
        }
    }

    /// Build a quantized model with random-initialized weights for testing.
    /// Uses the builder approach (quantizes random data) since we have no
    /// pre-quantized safetensors in tests.
    fn build_test_model(
        config: &LlamaConfig,
        qc: &QuantConfig,
    ) -> Result<MlxQuantizedLlamaForCausalLM, Exception> {
        let embed_tokens = MlxEmbedTokens::Quantized(
            nn::QuantizedEmbeddingBuilder::new(config.vocab_size as i32, config.hidden_size as i32)
                .group_size(qc.group_size)
                .bits(qc.bits)
                .build()?,
        );

        let hidden = config.hidden_size as i32;
        let intermediate = config.intermediate_size as i32;
        let q_size = (config.num_attention_heads * config.head_dim) as i32;
        let kv_size = (config.num_kv_heads * config.head_dim) as i32;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            let self_attn = MlxQuantizedLlamaAttention {
                q_proj: nn::QuantizedLinearBuilder::new(hidden, q_size)
                    .group_size(qc.group_size)
                    .bits(qc.bits)
                    .bias(false)
                    .build()?,
                k_proj: nn::QuantizedLinearBuilder::new(hidden, kv_size)
                    .group_size(qc.group_size)
                    .bits(qc.bits)
                    .bias(false)
                    .build()?,
                v_proj: nn::QuantizedLinearBuilder::new(hidden, kv_size)
                    .group_size(qc.group_size)
                    .bits(qc.bits)
                    .bias(false)
                    .build()?,
                o_proj: nn::QuantizedLinearBuilder::new(q_size, hidden)
                    .group_size(qc.group_size)
                    .bits(qc.bits)
                    .bias(false)
                    .build()?,
                q_norm: None,
                k_norm: None,
                rope: {
                    let mut r = nn::Rope::new(config.head_dim as i32);
                    r.base = config.rope_theta;
                    r
                },
                num_heads: config.num_attention_heads,
                num_kv_heads: config.num_kv_heads,
                head_dim: config.head_dim,
                scale: 1.0 / (config.head_dim as f32).sqrt(),
                sliding_window: config.sliding_window,
                fuse_rope: false,
            };

            let mlp = MlxQuantizedLlamaMLP {
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
            };

            layers.push(MlxQuantizedLlamaDecoderLayer {
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

        Ok(MlxQuantizedLlamaForCausalLM {
            embed_tokens,
            layers,
            norm: nn::RmsNormBuilder::new(hidden)
                .eps(config.rms_norm_eps)
                .build()?,
            lm_head: None,
            tie_word_embeddings: config.tie_word_embeddings,
            config: config.clone(),
        })
    }

    #[test]
    fn test_quant_config_parse() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "intermediate_size": 11008,
                "vocab_size": 32000,
                "quantization": { "group_size": 64, "bits": 4 }
            }"#,
        )
        .unwrap();

        let qc = QuantConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(qc.group_size, 64);
        assert_eq!(qc.bits, 4);
    }

    #[test]
    fn test_quant_config_missing() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "intermediate_size": 11008,
                "vocab_size": 32000
            }"#,
        )
        .unwrap();

        assert!(QuantConfig::from_hf_config(&hf_config).is_none());
    }

    #[test]
    fn test_quantized_llama_forward() {
        let config = test_config();
        let qc = test_quant_config();
        let mut model = build_test_model(&config, &qc).unwrap();

        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        let logits = <MlxQuantizedLlamaForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &input_ids,
            &positions,
            &mut kv_cache,
            None,
        )
        .unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.shape(), &[3, config.vocab_size as i32]);
    }

    #[test]
    fn test_quantized_llama_prefill_and_decode() {
        let config = test_config();
        let qc = test_quant_config();
        let mut model = build_test_model(&config, &qc).unwrap();
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        // Prefill: 3 tokens
        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let logits = <MlxQuantizedLlamaForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &input_ids,
            &positions,
            &mut kv_cache,
            None,
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
        let logits2 = <MlxQuantizedLlamaForCausalLM as crate::models::MlxModel>::forward(
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

    /// Verify that quantized_matmul produces results consistent with dequantize + matmul.
    ///
    /// This catches ABI mismatches in the mlx-rs bindings where quantized_matmul
    /// compiles but produces wrong results (e.g. wrong group_size/bits passed through
    /// the C API, or mode parameter corruption).
    #[test]
    fn test_quantized_matmul_consistency() {
        use mlx_rs::ops;

        // Create a known weight matrix and quantize it.
        mlx_rs::random::seed(42).unwrap();
        let w_float = mlx_rs::random::normal::<f32>(&[64, 32], None, None, None).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[1, 32], None, None, None).unwrap();

        let (w_q, scales, biases) = ops::quantize(&w_float, 32, 4).unwrap();

        // Path A: quantized_matmul (the fast path used during inference)
        let result_qmm = ops::quantized_matmul(&x, &w_q, &scales, &biases, true, 32, 4).unwrap();
        result_qmm.eval().unwrap();

        // Path B: dequantize then regular matmul (reference)
        let w_deq = ops::dequantize(&w_q, &scales, &biases, 32, 4).unwrap();
        let result_ref = ops::matmul(&x, &w_deq.transpose_axes(&[1, 0]).unwrap()).unwrap();
        result_ref.eval().unwrap();

        // They should match closely (both use the same dequantized values).
        let diff = result_qmm.subtract(&result_ref).unwrap();
        let max_err = diff.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(
            max_err < 0.1,
            "quantized_matmul vs dequantize+matmul max error: {max_err} (expected < 0.1)"
        );
    }

    /// Verify that greedy decode is deterministic and logits are not degenerate.
    ///
    /// Catches regressions where the forward pass produces garbage logits due to
    /// broken mlx-rs bindings, incorrect SDPA calls, or corrupted KV cache.
    /// The vendored mlx-rs incident (b15ab6845) produced degenerate output that
    /// this test would have caught.
    #[test]
    fn test_greedy_decode_deterministic() {
        use mlx_rs::ops::indexing::IndexOp;
        let config = test_config();
        let qc = test_quant_config();
        mlx_rs::random::seed(123).unwrap();
        let mut model = build_test_model(&config, &qc).unwrap();
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        // Prefill
        let prompt = Array::from_iter(vec![1i32, 5, 10, 20, 30], &[5]);
        let positions = Array::from_iter(0..5i32, &[5]);
        let logits = <MlxQuantizedLlamaForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &prompt,
            &positions,
            &mut kv_cache,
            None,
        )
        .unwrap();
        logits.eval().unwrap();

        // Greedy decode 10 tokens
        let mut generated = Vec::new();
        let mut next_pos = 5i32;
        let vocab = config.vocab_size as i32;

        for _ in 0..10 {
            let last_logits = logits.index((-1, ..));
            let last_logits = if generated.is_empty() {
                last_logits
            } else {
                let tok = Array::from_iter(vec![*generated.last().unwrap()], &[1]);
                let pos = Array::from_iter(vec![next_pos - 1], &[1]);
                let l = <MlxQuantizedLlamaForCausalLM as crate::models::MlxModel>::forward(
                    &mut model,
                    &tok,
                    &pos,
                    &mut kv_cache,
                    Some(next_pos - 1),
                )
                .unwrap();
                l.eval().unwrap();
                l.index((0, ..))
            };

            let flat = last_logits.as_slice::<f32>();

            // Sanity: logits should not be NaN or all identical.
            assert!(
                !flat.iter().any(|v| v.is_nan()),
                "NaN in logits at decode step {}",
                generated.len()
            );
            let min = flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                (max - min) > 1e-3,
                "Logits are flat (max-min={}) at step {} — model likely broken",
                max - min,
                generated.len()
            );

            // Greedy pick
            let token_id = flat
                .iter()
                .enumerate()
                .max_by(|(_, a): &(usize, &f32), (_, b): &(usize, &f32)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i32)
                .unwrap();
            assert!(token_id >= 0 && token_id < vocab);
            generated.push(token_id);
            next_pos += 1;
        }

        // Run the same thing again with a fresh model + same seed — must match.
        mlx_rs::random::seed(123).unwrap();
        let mut model2 = build_test_model(&config, &qc).unwrap();
        let mut kv_cache2 = crate::cache::empty_kv_cache(config.num_hidden_layers);

        let logits2 = <MlxQuantizedLlamaForCausalLM as crate::models::MlxModel>::forward(
            &mut model2,
            &prompt,
            &positions,
            &mut kv_cache2,
            None,
        )
        .unwrap();
        logits2.eval().unwrap();

        let mut generated2 = Vec::new();
        let mut next_pos2 = 5i32;
        for _ in 0..10 {
            let last_logits = if generated2.is_empty() {
                logits2.index((-1, ..))
            } else {
                let tok = Array::from_iter(vec![*generated2.last().unwrap()], &[1]);
                let pos = Array::from_iter(vec![next_pos2 - 1], &[1]);
                let l = <MlxQuantizedLlamaForCausalLM as crate::models::MlxModel>::forward(
                    &mut model2,
                    &tok,
                    &pos,
                    &mut kv_cache2,
                    Some(next_pos2 - 1),
                )
                .unwrap();
                l.eval().unwrap();
                l.index((0, ..))
            };

            let flat = last_logits.as_slice::<f32>();
            let token_id = flat
                .iter()
                .enumerate()
                .max_by(|(_, a): &(usize, &f32), (_, b): &(usize, &f32)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i32)
                .unwrap();
            generated2.push(token_id);
            next_pos2 += 1;
        }

        assert_eq!(
            generated, generated2,
            "Greedy decode not deterministic with same seed"
        );

        // Note: with random weights, the model may legitimately repeat tokens.
        // The key assertions above (no NaN, non-flat logits, determinism) are
        // what catch broken inference. A real model test would also check
        // semantic quality, but that requires loading actual weights.
    }
}
