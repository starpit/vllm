// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! LLaMA model architecture for MLX.
//!
//! Uses mlx-rs nn primitives: `Linear`, `RmsNorm`, `Rope`, `Embedding`, and
//! `fast::scaled_dot_product_attention`. All operations are lazy — the entire
//! forward pass builds a compute graph that materializes with a single `eval()`.
//!
//! Also covers Mistral and Qwen2 (same architecture with minor config diffs).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use crate::cache::{BatchMlxLayerKvCache, MlxBatchInfo, MlxKvCache, MlxLayerKvCache};
use vllm_model::weight::HfModelConfig;

/// SwiGLU activation: `silu(gate) * up = sigmoid(gate) * gate * up`.
///
/// Uses raw MLX ops without `compile()` wrapper. The compile dispatch
/// overhead (closure creation + mlx_closure_apply + compile_trace/fuse)
/// is ~8% of single-request decode time — more expensive than the
/// fusion benefit for small tensors.
pub fn swiglu(gate: &Array, up: &Array) -> Result<Array, Exception> {
    let sig = mlx_rs::ops::sigmoid(gate)?;
    sig.multiply(gate)?.multiply(up)
}

/// Apply RoPE to a cached K tensor using explicit per-position offsets.
///
/// For fused RoPE, we need to apply position i to token i in the cached K.
/// MLX's `rope.forward((&k, offset))` doesn't match implicit positions when
/// offset > 0 and seq_len > 1. Instead, we reshape K so each sequence position
/// becomes a separate batch element with seq_len=1, use `rope_dynamic` with
/// per-element offsets, then reshape back.
///
/// `k` shape: `[batch, heads, kv_len, head_dim]`
/// `start_pos`: position offset for the first token (usually 0)
///
/// Returns: `k` with RoPE applied at positions `[start_pos, start_pos+1, ..., start_pos+kv_len-1]`
pub(crate) fn apply_rope_to_cached_k(
    k: &Array,
    rope: &nn::Rope,
    start_pos: i32,
) -> Result<Array, mlx_rs::error::Exception> {
    let kv_len = k.dim(2);

    // Per-position RoPE on each kv_len slice (rope_dynamic not in mlx-rs 0.25).
    // k: [batch, heads, kv_len, dim]
    let mut k_parts = Vec::with_capacity(kv_len as usize);
    for pos in 0..kv_len {
        let offset = start_pos + pos;
        // [batch, heads, 1, dim]
        let k_slice = k.try_index((.., .., pos..pos + 1, ..))?;
        k_parts.push(mlx_rs::fast::rope(
            &k_slice,
            rope.dimensions,
            rope.traditional,
            rope.base,
            rope.scale,
            offset,
            None,
        )?);
    }
    let k_refs: Vec<&Array> = k_parts.iter().collect();
    mlx_rs::ops::concatenate_axis(&k_refs, 2)
}

/// Like `apply_rope_to_cached_k` but with per-batch start positions.
///
/// Used for batched decode with left-padding: each batch element starts at
/// a different position (negative of left_pad) so real tokens get 0-based positions.
///
/// `k` shape: `[B, heads, kv_len, head_dim]`
/// `start_positions`: `[start_0, start_1, ..., start_B-1]`
pub(crate) fn apply_rope_to_cached_k_batched(
    k: &Array,
    rope: &nn::Rope,
    start_positions: &[i32],
) -> Result<Array, mlx_rs::error::Exception> {
    let batch = k.dim(0);
    let kv_len = k.dim(2);

    // Per-batch, per-position RoPE (rope_dynamic not in mlx-rs 0.25).
    // k: [B, heads, kv_len, dim]
    // Process each batch element separately since start_positions differ.
    let mut batch_parts = Vec::with_capacity(batch as usize);
    for b in 0..batch as usize {
        let bb = b as i32;
        let k_b = k.try_index((bb..bb + 1, .., .., ..))?; // [1, heads, kv_len, dim]
        let mut pos_parts = Vec::with_capacity(kv_len as usize);
        for pos in 0..kv_len {
            let offset = start_positions[b] + pos;
            let k_slice = k_b.try_index((.., .., pos..pos + 1, ..))?;
            pos_parts.push(mlx_rs::fast::rope(
                &k_slice,
                rope.dimensions,
                rope.traditional,
                rope.base,
                rope.scale,
                offset,
                None,
            )?);
        }
        let pos_refs: Vec<&Array> = pos_parts.iter().collect();
        batch_parts.push(mlx_rs::ops::concatenate_axis(&pos_refs, 2)?);
    }
    let batch_refs: Vec<&Array> = batch_parts.iter().collect();
    mlx_rs::ops::concatenate_axis(&batch_refs, 0)
}

// ---------------------------------------------------------------------------
// LlamaConfig
// ---------------------------------------------------------------------------

/// LongRoPE scaling configuration (used by Phi-3/Phi-4 family).
///
/// Per-dimension frequency rescale factors for short and long contexts.
#[derive(Debug, Clone)]
pub struct LongRopeScaling {
    /// Rescale factors for short contexts (len = rotary_dim / 2).
    pub short_factor: Vec<f64>,
    /// Rescale factors for long contexts (len = rotary_dim / 2).
    pub long_factor: Vec<f64>,
    /// Original max position embeddings before LongRoPE extension.
    pub original_max_position_embeddings: usize,
}

/// Parsed configuration for a LLaMA model.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
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
    /// Sliding window size for attention. When `Some(w)`, each token only
    /// attends to the most recent `w` positions. Used by Mistral, Qwen2, etc.
    pub sliding_window: Option<usize>,
    /// Fraction of head dimensions that get RoPE (default 1.0). Used by Phi-3/4.
    pub partial_rotary_factor: f64,
    /// LongRoPE scaling parameters. `None` means standard RoPE.
    pub long_rope_scaling: Option<LongRopeScaling>,
}

impl LlamaConfig {
    /// Parse from a HuggingFace config.json.
    pub fn from_hf_config(config: &HfModelConfig) -> Result<Self, String> {
        let hidden_size = config
            .hidden_size
            .ok_or_else(|| "missing hidden_size".to_string())?;
        let num_attention_heads = config
            .num_attention_heads
            .ok_or_else(|| "missing num_attention_heads".to_string())?;

        // Parse sliding_window from config.json extras (used by Mistral, Qwen2, Phi-3, etc.).
        // Handles both scalar (4096) and array ([null, 4096, null, 4096, ...]) formats.
        // The array format is used by newer Mistral models (3.x) — we extract the first
        // non-null value as the scalar sliding window size.
        let sliding_window = config.extra.get("sliding_window").and_then(|v| {
            if let Some(n) = v.as_u64() {
                Some(n as usize)
            } else if let Some(arr) = v.as_array() {
                arr.iter()
                    .find_map(|item| item.as_u64())
                    .map(|n| n as usize)
            } else {
                None
            }
        });

        let num_hidden_layers = config
            .num_hidden_layers
            .ok_or_else(|| "missing num_hidden_layers".to_string())?;

        // Validate Qwen2 max_window_layers: if set and < num_hidden_layers, the model
        // wants partial-layer sliding window (only top layers). We don't support
        // that — disable sliding window and warn.
        let sliding_window = if let Some(max_window_layers) = config
            .extra
            .get("max_window_layers")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
        {
            if max_window_layers < num_hidden_layers {
                tracing::warn!(
                    "config has max_window_layers={} < num_hidden_layers={}: \
                     partial-layer sliding window not supported, disabling sliding window",
                    max_window_layers,
                    num_hidden_layers
                );
                None
            } else {
                sliding_window
            }
        } else {
            sliding_window
        };

        // Parse partial_rotary_factor (Phi-3/4 family).
        let partial_rotary_factor = config
            .extra
            .get("partial_rotary_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        // Parse LongRoPE scaling (rope_scaling.type == "longrope").
        let long_rope_scaling = config.extra.get("rope_scaling").and_then(|rs| {
            let scaling_type = rs.get("type")?.as_str()?;
            if scaling_type != "longrope" {
                return None;
            }
            let short_factor: Vec<f64> = rs
                .get("short_factor")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_f64())
                .collect();
            let long_factor: Vec<f64> = rs
                .get("long_factor")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_f64())
                .collect();
            let original_max = config
                .extra
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(4096) as usize;

            if short_factor.is_empty() || long_factor.is_empty() {
                return None;
            }

            Some(LongRopeScaling {
                short_factor,
                long_factor,
                original_max_position_embeddings: original_max,
            })
        });

        Ok(Self {
            hidden_size,
            num_attention_heads,
            num_kv_heads: config.num_kv_heads().unwrap_or(num_attention_heads),
            num_hidden_layers,
            intermediate_size: config
                .intermediate_size
                .ok_or_else(|| "missing intermediate_size".to_string())?,
            vocab_size: config
                .vocab_size
                .ok_or_else(|| "missing vocab_size".to_string())?,
            max_position_embeddings: config.max_position_embeddings.unwrap_or(4096),
            rms_norm_eps: config.norm_eps() as f32,
            rope_theta: config.rope_theta.unwrap_or(10000.0) as f32,
            head_dim: config
                .head_dim()
                .unwrap_or(hidden_size / num_attention_heads),
            tie_word_embeddings: config.tie_word_embeddings.unwrap_or(false),
            sliding_window,
            partial_rotary_factor,
            long_rope_scaling,
        })
    }
}

// ---------------------------------------------------------------------------
// Helper: assign weight from loaded HashMap
// ---------------------------------------------------------------------------

/// Look up a weight by name and assign it. Logs a warning if not found.
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
// MlxLlamaMLP
// ---------------------------------------------------------------------------

/// LLaMA MLP (SiLU-gated feed-forward network) using MLX.
pub struct MlxLlamaMLP {
    pub(crate) gate_proj: nn::Linear,
    pub(crate) up_proj: nn::Linear,
    pub(crate) down_proj: nn::Linear,
}

impl MlxLlamaMLP {
    /// Create with random initialization.
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
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

    /// Load weights from a flat HashMap.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
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

    /// Forward pass: gate_proj(x) → SwiGLU(gate, up) → down_proj
    ///
    /// Uses a single compiled swiglu (silu(gate) * up) matching mlx-lm's
    /// `activations.py`, instead of separate compiled_silu + multiply which
    /// doubles the compile dispatch overhead per layer.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let hidden = swiglu(&gate, &up)?;
        self.down_proj.forward(&hidden)
    }
}

// ---------------------------------------------------------------------------
// MlxLlamaAttention
// ---------------------------------------------------------------------------

/// LLaMA multi-head attention with RoPE and optional GQA, using MLX.
pub struct MlxLlamaAttention {
    pub(crate) q_proj: nn::Linear,
    pub(crate) k_proj: nn::Linear,
    pub(crate) v_proj: nn::Linear,
    pub(crate) o_proj: nn::Linear,
    /// Optional per-head Q norm (Qwen3 uses this).
    q_norm: Option<nn::RmsNorm>,
    /// Optional per-head K norm (Qwen3 uses this).
    k_norm: Option<nn::RmsNorm>,
    rope: nn::Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
    /// When true, K is stored without RoPE and RoPE is applied to the full
    /// cached K at attention time. This makes cached K position-independent,
    /// enabling relocatable span blocks.
    block_needs_positioning: bool,
}

impl MlxLlamaAttention {
    /// Create a new attention layer.
    pub fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        let hidden = config.hidden_size as i32;
        let q_size = (config.num_attention_heads * config.head_dim) as i32;
        let kv_size = (config.num_kv_heads * config.head_dim) as i32;

        Ok(Self {
            q_proj: nn::LinearBuilder::new(hidden, q_size).bias(false).build()?,
            k_proj: nn::LinearBuilder::new(hidden, kv_size)
                .bias(false)
                .build()?,
            v_proj: nn::LinearBuilder::new(hidden, kv_size)
                .bias(false)
                .build()?,
            o_proj: nn::LinearBuilder::new(q_size, hidden).bias(false).build()?,
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
            block_needs_positioning: false, // TODO: enable when MLX gets paged KV cache for per-block relocation
        })
    }

    /// Load weights from a flat HashMap.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
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
        // Optional bias (Qwen2 has attention bias, LLaMA/Mistral do not).
        for (proj, name) in [
            (&mut self.q_proj, "q_proj"),
            (&mut self.k_proj, "k_proj"),
            (&mut self.v_proj, "v_proj"),
            (&mut self.o_proj, "o_proj"),
        ] {
            if let Some(b) = weights.get(&format!("{prefix}.{name}.bias")) {
                proj.bias.value = Some(b.clone());
            }
        }

        // Optional QK norms (Qwen3 has per-head q_norm and k_norm).
        if let Some(w) = weights.get(&format!("{prefix}.q_norm.weight")) {
            let head_dim = w.dim(0);
            if let Ok(mut norm) = nn::RmsNormBuilder::new(head_dim).eps(1e-6).build() {
                norm.weight.value = w.clone();
                self.q_norm = Some(norm);
            }
        }
        if let Some(w) = weights.get(&format!("{prefix}.k_norm.weight")) {
            let head_dim = w.dim(0);
            if let Ok(mut norm) = nn::RmsNormBuilder::new(head_dim).eps(1e-6).build() {
                norm.weight.value = w.clone();
                self.k_norm = Some(norm);
            }
        }
    }

    /// Forward pass.
    ///
    /// * `hidden_states` — shape `[seq_len, hidden_size]`
    /// * `positions` — shape `[seq_len]` (min value used as RoPE offset)
    /// * `cache` — optional per-layer KV cache entry
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
        // RmsNorm normalizes the last dimension, so [seq, heads, head_dim] → per-head norm.
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
        // When block_needs_positioning: use rope_dynamic for both Q and K (via apply_rope_to_cached_k)
        // to ensure consistency. MLX's fast::rope and fast::rope_dynamic produce
        // different values for the same position, so Q and K must use the same impl.
        let (q, mut k, mut v) = if self.block_needs_positioning {
            let q = apply_rope_to_cached_k(&q, &self.rope, rope_offset)?;
            let (mut k, v) = crate::cache::kv_cache_update(cache, &k, &v)?;
            k = apply_rope_to_cached_k(&k, &self.rope, 0)?;
            (q, k, v)
        } else {
            let q = self.rope.forward((&q, rope_offset))?;
            k = self.rope.forward((&k, rope_offset))?;
            let (k, v) = crate::cache::kv_cache_update(cache, &k, &v)?;
            (q, k, v)
        };

        // Sliding window: trim K/V to only the last `w` positions.
        // The stored cache remains full (future tokens may still be in window),
        // but we only attend to the windowed subset.
        if let Some(w) = self.sliding_window {
            let kv_len = k.dim(2) as usize;
            if kv_len > w {
                let start = (kv_len - w) as i32;
                let end = kv_len as i32;
                // k, v shape: [1, heads, kv_len, head_dim]
                k = k.try_index((.., .., start..end, ..))?;
                v = v.try_index((.., .., start..end, ..))?;
            }
        }

        // Fused SDPA (single Metal kernel for decode when q_len=1).
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

    /// Batched forward: projections batched on `[total_tokens, hidden]`,
    /// split per-request for reshape/RoPE/KV-cache/SDPA, then rejoin for O projection.
    ///
    /// When all requests are decode (q_len=1) and KV lengths match, SDPA is
    /// batched into a single kernel launch.
    pub fn forward_batch(
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
        let all_decode = batch_info.num_reqs > 1
            && batch_info.q_lens.iter().all(|&ql| ql == 1)
            && self.sliding_window.is_none();

        // Reshape, RoPE, KV cache update.
        let mut per_req_q = Vec::with_capacity(batch_info.num_reqs);
        let mut per_req_k = Vec::with_capacity(batch_info.num_reqs);
        let mut per_req_v = Vec::with_capacity(batch_info.num_reqs);
        let mut kv_lens = Vec::with_capacity(batch_info.num_reqs);

        if all_decode {
            // --- Batched decode: single reshape + single RoPE for all N requests ---
            let n = batch_info.num_reqs as i32;
            let nh = self.num_heads as i32;
            let nkv = self.num_kv_heads as i32;
            let hd = self.head_dim as i32;

            // [N, hidden] → [N, heads, 1, hd] (one reshape, no per-request slicing)
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

            // Per-request RoPE + KV cache update.
            for (i, cache) in caches.iter_mut().enumerate().take(batch_info.num_reqs) {
                let ii = i as i32;
                let offset = batch_info.rope_offsets[i];
                let qi = q.try_index((ii..ii + 1, .., .., ..))?;
                let ki = k.try_index((ii..ii + 1, .., .., ..))?;
                let vi = v.try_index((ii..ii + 1, .., .., ..))?;

                let qi = self.rope.forward((&qi, offset))?;
                let ki = if self.block_needs_positioning {
                    ki
                } else {
                    self.rope.forward((&ki, offset))?
                };

                let (ki, vi) = crate::cache::kv_cache_update(cache, &ki, &vi)?;
                let ki = if self.block_needs_positioning {
                    apply_rope_to_cached_k(&ki, &self.rope, 0)?
                } else {
                    ki
                };
                kv_lens.push(ki.dim(2) as usize);
                per_req_q.push(qi);
                per_req_k.push(ki);
                per_req_v.push(vi);
            }
        } else {
            // --- Per-request fallback (prefill or mixed) ---
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

                let q = self.rope.forward((&q, offset))?;
                if !self.block_needs_positioning {
                    k = self.rope.forward((&k, offset))?;
                }

                let (k, v) = crate::cache::kv_cache_update(&mut caches[i], &k, &v)?;

                let k = if self.block_needs_positioning {
                    apply_rope_to_cached_k(&k, &self.rope, 0)?
                } else {
                    k
                };

                kv_lens.push(k.dim(2) as usize);
                per_req_q.push(q);
                per_req_k.push(k);
                per_req_v.push(v);
            }
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
    /// `left_pads` — per-request left-padding counts (needed for block_needs_positioning negative offsets).
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
                if self.block_needs_positioning {
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
        // When block_needs_positioning, K is stored unrotated (position-independent).
        let (k_cached, v_cached) = batch_cache.update_and_view(&k, &v)?;

        // When block_needs_positioning, apply RoPE to full cached K with per-batch start positions.
        // Left-padded batch: start at -left_pad so real tokens get 0-based positions.
        // Padding positions get negative RoPE (masked out by -inf in the SDPA mask).
        let k_cached = if self.block_needs_positioning {
            let start_positions: Vec<i32> = left_pads.iter().map(|&p| -(p as i32)).collect();
            apply_rope_to_cached_k_batched(&k_cached, &self.rope, &start_positions)?
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
// MlxLlamaDecoderLayer
// ---------------------------------------------------------------------------

/// A single LLaMA decoder layer using MLX.
pub struct MlxLlamaDecoderLayer {
    self_attn: MlxLlamaAttention,
    mlp: MlxLlamaMLP,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
}

impl MlxLlamaDecoderLayer {
    /// Create a new decoder layer.
    pub fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: MlxLlamaAttention::new(config)?,
            mlp: MlxLlamaMLP::new(config.hidden_size as i32, config.intermediate_size as i32)?,
            input_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
        })
    }

    /// Load weights from a flat HashMap.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        self.self_attn
            .load_weights(weights, &format!("{prefix}.self_attn"));
        self.mlp.load_weights(weights, &format!("{prefix}.mlp"));
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

    /// Forward pass.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<MlxLayerKvCache>,
        rope_offset: i32,
    ) -> Result<Array, Exception> {
        // Pre-attention layernorm + attention + residual.
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self
            .self_attn
            .forward(&normed, positions, cache, rope_offset)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        // Post-attention layernorm + MLP + residual.
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        hidden_states.add(&mlp_output)
    }

    /// Batched forward: norms and MLP run on `[total_tokens, hidden]`,
    /// attention splits per-request for RoPE/KV/SDPA.
    pub fn forward_batch(
        &mut self,
        hidden_states: &Array,
        batch_info: &MlxBatchInfo,
        caches: &mut [Option<MlxLayerKvCache>],
    ) -> Result<Array, Exception> {
        // Batched input layernorm + batched attention + residual.
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward_batch(&normed, batch_info, caches)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        // Batched post-attention layernorm + batched MLP + residual.
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
// MlxLlamaForCausalLM
// ---------------------------------------------------------------------------

/// LLaMA for causal language modeling using MLX.
pub struct MlxLlamaForCausalLM {
    embed_tokens: nn::Embedding,
    layers: Vec<MlxLlamaDecoderLayer>,
    norm: nn::RmsNorm,
    pub(crate) lm_head: Option<nn::Linear>,
    tie_word_embeddings: bool,
    #[allow(dead_code)]
    config: LlamaConfig,
}

impl MlxLlamaForCausalLM {
    /// Create a new model with random initialization.
    pub fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(MlxLlamaDecoderLayer::new(config)?);
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

    /// Load weights from a flat HashMap of name → Array.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>) {
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

    /// Load weights with a configurable prefix (e.g., "model" or "language_model.model").
    pub fn load_weights_with_prefix(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(
            &mut self.embed_tokens.weight,
            weights,
            &format!("{prefix}.embed_tokens.weight"),
        );
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, &format!("{prefix}.layers.{i}"));
        }
        assign_weight(
            &mut self.norm.weight,
            weights,
            &format!("{prefix}.norm.weight"),
        );
        if let Some(ref mut lm_head) = self.lm_head {
            assign_weight(&mut lm_head.weight, weights, "lm_head.weight");
        }
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

    /// Load model weights from safetensors files in a directory.
    pub fn load(
        model_dir: &Path,
        config: &LlamaConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut model = Self::new(config)?;

        // Load all safetensors weights into a flat HashMap.
        let weights = load_safetensors_weights(model_dir)?;

        // Assign weights to model parameters.
        model.load_weights(&weights);

        // Eval to materialize the loaded weights.
        mlx_rs::transforms::eval(weights.values())?;

        Ok(model)
    }
}

impl super::MlxModel for MlxLlamaForCausalLM {
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
            // Collect the per-layer cache entry from each request's KV cache.
            let mut layer_caches: Vec<&mut Option<MlxLayerKvCache>> =
                kv_caches.iter_mut().map(|kv| &mut kv[layer_idx]).collect();

            // Build a contiguous slice for the layer's forward_batch.
            // We need to pass &mut [Option<MlxLayerKvCache>], but we have Vec<&mut Option<...>>.
            // Use a temporary Vec to hold the values, then put them back.
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
// Weight loading helper
// ---------------------------------------------------------------------------

/// Load all safetensors weights from a model directory into a flat HashMap.
pub fn load_safetensors_weights(
    model_dir: &Path,
) -> Result<HashMap<String, Array>, Box<dyn std::error::Error + Send + Sync>> {
    let single_path = model_dir.join("model.safetensors");
    if single_path.exists() {
        let weights = Array::load_safetensors(&single_path)?;
        return Ok(weights);
    }

    // Sharded model.
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index = vllm_model::weight::SafeTensorsIndex::from_file(&index_path)?;
        let shard_files = index.shard_files();

        let mut all_weights = HashMap::new();
        for shard_name in &shard_files {
            let shard_path = model_dir.join(shard_name);
            let shard_data = Array::load_safetensors(&shard_path)?;
            all_weights.extend(shard_data);
        }
        return Ok(all_weights);
    }

    Err(format!(
        "no model.safetensors or model.safetensors.index.json in {}",
        model_dir.display()
    )
    .into())
}

/// Factory function for creating an MLX LLaMA model.
pub fn create_mlx_llama(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let llama_config = LlamaConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let model = MlxLlamaForCausalLM::load(model_dir, &llama_config, dtype)?;
    Ok(Box::new(model))
}

/// Factory function for creating an MLX LLaMA model from GPTQ weights.
///
/// Dequantizes GPTQ INT4 weights at load time, then uses the standard
/// (non-quantized) model architecture.
pub fn create_mlx_gptq_llama(
    model_dir: &Path,
    config: &HfModelConfig,
    _dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let llama_config = LlamaConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    // Parse GPTQ config.
    let gptq_cfg = vllm_model::gptq_config::GptqQuantizeConfig::from_dir(model_dir)
        .or_else(|_| {
            config
                .extra
                .get("quantization_config")
                .ok_or_else(|| vllm_model::error::ModelError::Other("no GPTQ config found".into()))
                .and_then(vllm_model::gptq_config::GptqQuantizeConfig::from_json_value)
        })
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    let gptq = gptq_cfg.to_gptq_config();

    // Load raw safetensors weights (includes qweight/qzeros/scales/g_idx).
    let raw_weights = load_safetensors_weights(model_dir)?;

    // Dequantize GPTQ weights into standard .weight tensors.
    let weights = crate::gptq::dequantize_gptq_weights(raw_weights, &gptq)?;

    // Build standard model and load dequantized weights.
    let mut model = MlxLlamaForCausalLM::new(&llama_config)?;
    model.load_weights(&weights);
    mlx_rs::transforms::eval(weights.values())?;

    Ok(Box::new(model))
}

/// Factory function for creating an MLX Qwen2 model from GPTQ weights.
pub fn create_mlx_gptq_qwen2(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    // Qwen2 is architecturally identical to LLaMA, reuse the same factory.
    create_mlx_gptq_llama(model_dir, config, dtype)
}

/// Factory function for creating an MLX LLaMA model from AWQ weights.
///
/// Dequantizes AWQ INT4 weights at load time, then uses the standard
/// (non-quantized) model architecture.
pub fn create_mlx_awq_llama(
    model_dir: &Path,
    config: &HfModelConfig,
    _dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let llama_config = LlamaConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    // Parse AWQ config.
    let awq_cfg = vllm_model::awq_config::AwqQuantizeConfig::from_dir(model_dir)
        .or_else(|_| {
            config
                .extra
                .get("quantization_config")
                .ok_or_else(|| vllm_model::error::ModelError::Other("no AWQ config found".into()))
                .and_then(vllm_model::awq_config::AwqQuantizeConfig::from_json_value)
        })
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    let awq = awq_cfg.to_awq_config();

    // Load raw safetensors weights (includes qweight/qzeros/scales).
    let raw_weights = load_safetensors_weights(model_dir)?;

    // Dequantize AWQ weights into standard .weight tensors.
    let weights = crate::awq::dequantize_awq_weights(raw_weights, &awq)?;

    // Build standard model and load dequantized weights.
    let mut model = MlxLlamaForCausalLM::new(&llama_config)?;
    model.load_weights(&weights);
    mlx_rs::transforms::eval(weights.values())?;

    Ok(Box::new(model))
}

/// Factory function for creating an MLX Qwen2 model from AWQ weights.
pub fn create_mlx_awq_qwen2(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    // Qwen2 is architecturally identical to LLaMA, reuse the same factory.
    create_mlx_awq_llama(model_dir, config, dtype)
}

/// Factory function for creating an MLX LLaMA model from BitsAndBytes NF4 weights.
///
/// Dequantizes BnB NF4 weights at load time, then uses the standard
/// (non-quantized) model architecture.
pub fn create_mlx_bnb_llama(
    model_dir: &Path,
    config: &HfModelConfig,
    _dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let llama_config = LlamaConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    // Parse BnB config.
    let bnb_cfg = config
        .extra
        .get("quantization_config")
        .ok_or_else(|| vllm_model::error::ModelError::Other("no BnB config found".into()))
        .and_then(vllm_model::bnb_config::BnbQuantizeConfig::from_json_value)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    let bnb = bnb_cfg.to_bnb_config();

    // Load raw safetensors weights (includes packed uint8 + absmax).
    let raw_weights = load_safetensors_weights(model_dir)?;

    // Dequantize BnB weights into standard .weight tensors.
    let weights = crate::bnb::dequantize_bnb_weights(raw_weights, &bnb, config)?;

    // Build standard model and load dequantized weights.
    let mut model = MlxLlamaForCausalLM::new(&llama_config)?;
    model.load_weights(&weights);
    mlx_rs::transforms::eval(weights.values())?;

    Ok(Box::new(model))
}

/// Factory function for creating an MLX Qwen2 model from BitsAndBytes NF4 weights.
pub fn create_mlx_bnb_qwen2(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    // Qwen2 is architecturally identical to LLaMA, reuse the same factory.
    create_mlx_bnb_llama(model_dir, config, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            tie_word_embeddings: false,
            sliding_window: None,
            partial_rotary_factor: 1.0,
            long_rope_scaling: None,
        }
    }

    #[test]
    fn test_llama_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 32,
                "intermediate_size": 11008,
                "vocab_size": 32000,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false
            }"#,
        )
        .unwrap();

        let config = LlamaConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
    }

    #[test]
    fn test_mlx_llama_mlp_forward() {
        let mut mlp = MlxLlamaMLP::new(32, 64).unwrap();
        let x = mlx_rs::ops::ones::<f32>(&[3, 32]).unwrap();
        let out = mlp.forward(&x).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[3, 32]);
    }

    #[test]
    fn test_mlx_llama_attention_forward() {
        let config = test_config();
        let mut attn = MlxLlamaAttention::new(&config).unwrap();

        let x = mlx_rs::ops::ones::<f32>(&[4, 32]).unwrap();
        let positions = Array::from_iter(0..4i32, &[4]);
        let mut cache = None;

        let out = attn.forward(&x, &positions, &mut cache, 0).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[4, 32]);
        assert!(cache.is_some());
    }

    #[test]
    fn test_mlx_llama_model_forward() {
        let config = test_config();
        let mut model = MlxLlamaForCausalLM::new(&config).unwrap();

        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        let logits = <MlxLlamaForCausalLM as crate::models::MlxModel>::forward(
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
    fn test_mlx_llama_kv_cache_prefill_and_decode() {
        let config = test_config();
        let mut model = MlxLlamaForCausalLM::new(&config).unwrap();
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        // Prefill: 3 tokens
        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let logits = <MlxLlamaForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &input_ids,
            &positions,
            &mut kv_cache,
            None,
        )
        .unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.shape(), &[3, config.vocab_size as i32]);

        // Verify KV cache is populated.
        for entry in &kv_cache {
            assert!(entry.is_some());
        }

        // Decode: 1 token at position 3
        let decode_ids = Array::from_iter(vec![15i32], &[1]);
        let decode_pos = Array::from_iter(vec![3i32], &[1]);
        let logits2 = <MlxLlamaForCausalLM as crate::models::MlxModel>::forward(
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

    /// Test that the two RoPE application strategies produce identical K.
    /// Strategy A: RoPE → cache → read (normal)
    /// Strategy B: cache → read → RoPE (fused)
    #[test]
    fn test_block_needs_positioning_cache_roundtrip() {
        let mut rope = nn::Rope::new(8);
        rope.base = 10000.0;

        // K projection output (4 tokens during prefill, 1 during decode)
        let k_prefill = mlx_rs::random::normal::<f32>(&[1, 4, 4, 8], None, None, None).unwrap();
        k_prefill.eval().unwrap();
        let k_decode_tok = mlx_rs::random::normal::<f32>(&[1, 4, 1, 8], None, None, None).unwrap();
        k_decode_tok.eval().unwrap();
        let v_prefill = mlx_rs::ops::zeros::<f32>(&[1, 4, 4, 8]).unwrap();
        let v_decode = mlx_rs::ops::zeros::<f32>(&[1, 4, 1, 8]).unwrap();

        // --- Strategy A (reference): per-token RoPE using apply_rope_to_cached_k ---
        let k_prefill_rotated = apply_rope_to_cached_k(&k_prefill, &rope, 0).unwrap();
        k_prefill_rotated.eval().unwrap();
        let mut cache_a: Option<MlxLayerKvCache> = None;
        let _ =
            crate::cache::kv_cache_update(&mut cache_a, &k_prefill_rotated, &v_prefill).unwrap();

        let k_decode_rotated = apply_rope_to_cached_k(&k_decode_tok, &rope, 4).unwrap();
        k_decode_rotated.eval().unwrap();
        let (k_a, _) = cache_a
            .as_mut()
            .unwrap()
            .update_and_view(&k_decode_rotated, &v_decode)
            .unwrap();
        k_a.eval().unwrap();

        // --- Strategy B (fused): cache then RoPE ---
        let mut cache_b: Option<MlxLayerKvCache> = None;
        let _ = crate::cache::kv_cache_update(&mut cache_b, &k_prefill, &v_prefill).unwrap();

        let (k_b_raw, _) = cache_b
            .as_mut()
            .unwrap()
            .update_and_view(&k_decode_tok, &v_decode)
            .unwrap();
        k_b_raw.eval().unwrap();
        let k_b = apply_rope_to_cached_k(&k_b_raw, &rope, 0).unwrap();
        k_b.eval().unwrap();

        // Compare all 5 tokens
        let diff = k_a.subtract(&k_b).unwrap();
        let max_err: f32 = mlx_rs::ops::max(&mlx_rs::ops::abs(&diff).unwrap(), false)
            .unwrap()
            .item();
        eprintln!("decode K max_err = {max_err}");

        // Also compare just the LAST token (position 4)
        let k_a_last = k_a.try_index((.., .., 4i32..5, ..)).unwrap();
        let k_b_last = k_b.try_index((.., .., 4i32..5, ..)).unwrap();
        let diff_last = k_a_last.subtract(&k_b_last).unwrap();
        let max_err_last: f32 = mlx_rs::ops::max(&mlx_rs::ops::abs(&diff_last).unwrap(), false)
            .unwrap()
            .item();
        eprintln!("decode K last token max_err = {max_err_last}");

        // Compare first 4 tokens
        let k_a_first = k_a.try_index((.., .., ..4i32, ..)).unwrap();
        let k_b_first = k_b.try_index((.., .., ..4i32, ..)).unwrap();
        let diff_first = k_a_first.subtract(&k_b_first).unwrap();
        let max_err_first: f32 = mlx_rs::ops::max(&mlx_rs::ops::abs(&diff_first).unwrap(), false)
            .unwrap()
            .item();
        eprintln!("decode K first 4 tokens max_err = {max_err_first}");

        assert!(
            max_err < 1e-6,
            "block_needs_positioning decode K values differ: max_err={max_err}"
        );
    }

    /// Verify block_needs_positioning self-consistency: running prefill+decode twice with
    /// block_needs_positioning=true produces identical results (deterministic).
    ///
    /// Note: block_needs_positioning uses rope_dynamic internally while the non-fused path uses
    /// fast::rope. MLX's two rope implementations produce different absolute values,
    /// so cross-implementation comparison is not meaningful. Instead we verify:
    /// 1. K values are position-correct (test_block_needs_positioning_cache_roundtrip)
    /// 2. Output is deterministic (this test)
    /// 3. Shapes are correct
    #[test]
    fn test_block_needs_positioning_self_consistency() {
        let config = test_config();
        let mut attn = MlxLlamaAttention::new(&config).unwrap();
        attn.block_needs_positioning = true;

        let x = mlx_rs::random::normal::<f32>(&[4, 32], None, None, None).unwrap();
        x.eval().unwrap();
        let positions = Array::from_iter(0..4i32, &[4]);
        let x_decode = mlx_rs::random::normal::<f32>(&[1, 32], None, None, None).unwrap();
        x_decode.eval().unwrap();
        let pos_decode = Array::from_iter(vec![4i32], &[1]);

        // Run 1
        let mut cache1 = None;
        let out1 = attn.forward(&x, &positions, &mut cache1, 0).unwrap();
        out1.eval().unwrap();
        let out1_d = attn
            .forward(&x_decode, &pos_decode, &mut cache1, 4)
            .unwrap();
        out1_d.eval().unwrap();

        // Run 2 (fresh cache)
        let mut cache2 = None;
        let out2 = attn.forward(&x, &positions, &mut cache2, 0).unwrap();
        out2.eval().unwrap();
        let out2_d = attn
            .forward(&x_decode, &pos_decode, &mut cache2, 4)
            .unwrap();
        out2_d.eval().unwrap();

        // Both runs should produce identical results.
        let diff = out1.subtract(&out2).unwrap();
        let max_err: f32 = mlx_rs::ops::max(&mlx_rs::ops::abs(&diff).unwrap(), false)
            .unwrap()
            .item();
        assert_eq!(
            max_err, 0.0,
            "block_needs_positioning prefill non-deterministic"
        );

        let diff_d = out1_d.subtract(&out2_d).unwrap();
        let max_err_d: f32 = mlx_rs::ops::max(&mlx_rs::ops::abs(&diff_d).unwrap(), false)
            .unwrap()
            .item();
        assert_eq!(
            max_err_d, 0.0,
            "block_needs_positioning decode non-deterministic"
        );

        // Shapes should be correct.
        assert_eq!(out1.shape(), &[4, 32]); // prefill: 4 tokens
        assert_eq!(out1_d.shape(), &[1, 32]); // decode: 1 token
    }

    /// Verify block_needs_positioning model-level self-consistency: prefill + multi-step decode.
    #[test]
    fn test_block_needs_positioning_model_consistency() {
        let config = test_config();
        let mut model = MlxLlamaForCausalLM::new(&config).unwrap();
        for layer in &mut model.layers {
            layer.self_attn.block_needs_positioning = true;
        }

        let input_ids = Array::from_iter(vec![1i32, 5, 10, 20], &[4]);
        let positions = Array::from_iter(0..4i32, &[4]);

        // Run 1: prefill + 3 decode steps
        let mut kv1 = crate::cache::empty_kv_cache(config.num_hidden_layers);
        let logits1 = <MlxLlamaForCausalLM as crate::models::MlxModel>::forward(
            &mut model, &input_ids, &positions, &mut kv1, None,
        )
        .unwrap();
        logits1.eval().unwrap();

        let mut decode_logits_1 = Vec::new();
        for step in 0..3 {
            let tok = Array::from_iter(vec![7i32 + step], &[1]);
            let pos = Array::from_iter(vec![4i32 + step], &[1]);
            let logits = <MlxLlamaForCausalLM as crate::models::MlxModel>::forward(
                &mut model,
                &tok,
                &pos,
                &mut kv1,
                Some(4 + step),
            )
            .unwrap();
            logits.eval().unwrap();
            decode_logits_1.push(logits);
        }

        // Run 2: same sequence, fresh cache
        let mut kv2 = crate::cache::empty_kv_cache(config.num_hidden_layers);
        let logits2 = <MlxLlamaForCausalLM as crate::models::MlxModel>::forward(
            &mut model, &input_ids, &positions, &mut kv2, None,
        )
        .unwrap();
        logits2.eval().unwrap();

        let mut decode_logits_2 = Vec::new();
        for step in 0..3 {
            let tok = Array::from_iter(vec![7i32 + step], &[1]);
            let pos = Array::from_iter(vec![4i32 + step], &[1]);
            let logits = <MlxLlamaForCausalLM as crate::models::MlxModel>::forward(
                &mut model,
                &tok,
                &pos,
                &mut kv2,
                Some(4 + step),
            )
            .unwrap();
            logits.eval().unwrap();
            decode_logits_2.push(logits);
        }

        // Both runs should be identical.
        let diff = logits1.subtract(&logits2).unwrap();
        let max_err: f32 = mlx_rs::ops::max(&mlx_rs::ops::abs(&diff).unwrap(), false)
            .unwrap()
            .item();
        assert_eq!(
            max_err, 0.0,
            "block_needs_positioning model prefill non-deterministic"
        );

        for (i, (l1, l2)) in decode_logits_1
            .iter()
            .zip(decode_logits_2.iter())
            .enumerate()
        {
            let diff = l1.subtract(l2).unwrap();
            let max_err: f32 = mlx_rs::ops::max(&mlx_rs::ops::abs(&diff).unwrap(), false)
                .unwrap()
                .item();
            assert_eq!(
                max_err, 0.0,
                "block_needs_positioning decode step {i} non-deterministic"
            );
        }
    }
}
