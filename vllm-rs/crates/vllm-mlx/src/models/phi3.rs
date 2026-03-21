// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Phi-3 model architecture for MLX.
//!
//! Phi-3 is similar to LLaMA but uses fused projections:
//! - `qkv_proj` — fused Q/K/V projection (split after projection)
//! - `gate_up_proj` — fused gate+up projection (split after projection)
//! - `o_proj`, `down_proj` — separate (same as LLaMA)
//!
//! Activation: SiLU (same as LLaMA).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use crate::cache::{MlxBatchInfo, MlxKvCache, MlxLayerKvCache};
use crate::models::llama::{LlamaConfig, LongRopeScaling, assign_weight, load_safetensors_weights};
use crate::models::quantized_llama::{
    MlxEmbedTokens, MlxLmHead, QuantConfig, make_quantized_linear,
};
use vllm_model::weight::HfModelConfig;

// ---------------------------------------------------------------------------
// Phi3Rope — RoPE with optional LongRoPE scaling + partial rotation
// ---------------------------------------------------------------------------

/// Custom RoPE supporting partial rotation (`partial_rotary_factor`) and
/// LongRoPE per-dimension frequency scaling (Phi-3/Phi-4 family).
enum Phi3Rope {
    /// Standard: full head_dim rotation via nn::Rope (most Phi-3 models).
    Standard(nn::Rope),
    /// Custom: LongRoPE and/or partial rotation (Phi-4 mini, etc.).
    Custom {
        /// Inverse frequencies [rotary_dim / 2], already scaled by LongRoPE factors.
        inv_freq: Array,
        /// Magnitude scaling factor applied to cos/sin.
        mscale: f32,
        /// Number of head dimensions that get rotation.
        rotary_dim: usize,
        /// Full head dimension.
        head_dim: usize,
    },
}

impl Phi3Rope {
    fn new(config: &LlamaConfig) -> Self {
        let rotary_dim = (config.head_dim as f64 * config.partial_rotary_factor) as usize;

        // Standard RoPE: no LongRoPE, full head_dim rotation.
        if config.long_rope_scaling.is_none() && rotary_dim == config.head_dim {
            let mut r = nn::Rope::new(config.head_dim as i32);
            r.base = config.rope_theta;
            return Self::Standard(r);
        }

        // Custom RoPE: compute inv_freq with optional LongRoPE scaling.
        let half_dim = rotary_dim / 2;
        let (inv_freq_vec, mscale) = compute_longrope_inv_freq(
            config.rope_theta as f64,
            half_dim,
            rotary_dim,
            config.max_position_embeddings,
            config.long_rope_scaling.as_ref(),
        );

        let inv_freq = Array::from_slice(&inv_freq_vec, &[half_dim as i32]);

        Self::Custom {
            inv_freq,
            mscale,
            rotary_dim,
            head_dim: config.head_dim,
        }
    }

    /// Apply RoPE to Q and K tensors.
    ///
    /// Input shapes: `[1, heads, seq_len, head_dim]`
    /// Returns rotated (Q, K) with same shapes.
    fn apply(&mut self, q: &Array, k: &Array, offset: i32) -> Result<(Array, Array), Exception> {
        match self {
            Self::Standard(rope) => {
                let q_rot = rope.forward((q, offset))?;
                let k_rot = rope.forward((k, offset))?;
                Ok((q_rot, k_rot))
            }
            Self::Custom {
                inv_freq,
                mscale,
                rotary_dim,
                head_dim,
            } => {
                let seq_len = q.dim(2);
                let half_dim = (*rotary_dim / 2) as i32;

                // Position indices: [offset, offset+1, ..., offset+seq_len-1]
                let positions: Vec<f32> = (0..seq_len).map(|i| (offset + i) as f32).collect();
                let pos = Array::from_slice(&positions, &[seq_len]);

                // Outer product: [seq, 1] * [1, half_dim] = [seq, half_dim]
                let freqs = pos
                    .reshape(&[seq_len, 1])?
                    .matmul(&inv_freq.reshape(&[1, half_dim])?)?;

                // cos/sin with mscale: [seq, half_dim]
                let cos_half = freqs.cos()?.multiply(Array::from_f32(*mscale))?;
                let sin_half = freqs.sin()?.multiply(Array::from_f32(*mscale))?;

                // Duplicate for full rotary_dim: [seq, rotary_dim]
                let cos = concatenate_axis(&[&cos_half, &cos_half], -1)?;
                let sin = concatenate_axis(&[&sin_half, &sin_half], -1)?;

                // Broadcast: [seq, rotary_dim] -> [1, 1, seq, rotary_dim]
                let cos_b = cos.reshape(&[1, 1, seq_len, *rotary_dim as i32])?;
                let sin_b = sin.reshape(&[1, 1, seq_len, *rotary_dim as i32])?;

                let q_rot = apply_partial_rope(q, &cos_b, &sin_b, *rotary_dim, *head_dim)?;
                let k_rot = apply_partial_rope(k, &cos_b, &sin_b, *rotary_dim, *head_dim)?;
                Ok((q_rot, k_rot))
            }
        }
    }

    /// Apply RoPE to Q only (used with fuse_rope).
    fn apply_q_only(&mut self, q: &Array, offset: i32) -> Result<Array, Exception> {
        match self {
            Self::Standard(rope) => rope.forward((q, offset)),
            Self::Custom {
                inv_freq,
                mscale,
                rotary_dim,
                head_dim,
            } => {
                let seq_len = q.dim(2);
                let half_dim = (*rotary_dim / 2) as i32;

                let positions: Vec<f32> = (0..seq_len).map(|i| (offset + i) as f32).collect();
                let pos = Array::from_slice(&positions, &[seq_len]);

                let freqs = pos
                    .reshape(&[seq_len, 1])?
                    .matmul(&inv_freq.reshape(&[1, half_dim])?)?;

                let cos_half = freqs.cos()?.multiply(Array::from_f32(*mscale))?;
                let sin_half = freqs.sin()?.multiply(Array::from_f32(*mscale))?;

                let cos = concatenate_axis(&[&cos_half, &cos_half], -1)?;
                let sin = concatenate_axis(&[&sin_half, &sin_half], -1)?;

                let cos_b = cos.reshape(&[1, 1, seq_len, *rotary_dim as i32])?;
                let sin_b = sin.reshape(&[1, 1, seq_len, *rotary_dim as i32])?;

                apply_partial_rope(q, &cos_b, &sin_b, *rotary_dim, *head_dim)
            }
        }
    }

    /// Apply RoPE to K only (used with fuse_rope after cache update).
    fn apply_k_only(&mut self, k: &Array, offset: i32) -> Result<Array, Exception> {
        match self {
            Self::Standard(rope) => rope.forward((k, offset)),
            Self::Custom {
                inv_freq,
                mscale,
                rotary_dim,
                head_dim,
            } => {
                let seq_len = k.dim(2);
                let half_dim = (*rotary_dim / 2) as i32;

                let positions: Vec<f32> = (0..seq_len).map(|i| (offset + i) as f32).collect();
                let pos = Array::from_slice(&positions, &[seq_len]);

                let freqs = pos
                    .reshape(&[seq_len, 1])?
                    .matmul(&inv_freq.reshape(&[1, half_dim])?)?;

                let cos_half = freqs.cos()?.multiply(Array::from_f32(*mscale))?;
                let sin_half = freqs.sin()?.multiply(Array::from_f32(*mscale))?;

                let cos = concatenate_axis(&[&cos_half, &cos_half], -1)?;
                let sin = concatenate_axis(&[&sin_half, &sin_half], -1)?;

                let cos_b = cos.reshape(&[1, 1, seq_len, *rotary_dim as i32])?;
                let sin_b = sin.reshape(&[1, 1, seq_len, *rotary_dim as i32])?;

                apply_partial_rope(k, &cos_b, &sin_b, *rotary_dim, *head_dim)
            }
        }
    }
}

/// Compute inverse frequencies and mscale for LongRoPE.
fn compute_longrope_inv_freq(
    base: f64,
    half_dim: usize,
    rotary_dim: usize,
    max_position_embeddings: usize,
    long_rope: Option<&LongRopeScaling>,
) -> (Vec<f32>, f32) {
    if let Some(lr) = long_rope {
        // Choose long vs short factors based on max_position vs original_max_position.
        let use_long = max_position_embeddings > lr.original_max_position_embeddings;
        let factors = if use_long {
            &lr.long_factor
        } else {
            &lr.short_factor
        };

        if use_long {
            tracing::info!(
                "Using LongRoPE long factors (max_pos={} > original_max_pos={})",
                max_position_embeddings,
                lr.original_max_position_embeddings
            );
        }

        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| {
                let factor = factors.get(i).copied().unwrap_or(1.0);
                (1.0 / (factor * base.powf(2.0 * i as f64 / rotary_dim as f64))) as f32
            })
            .collect();

        // mscale: magnitude correction for extended context.
        let scale = max_position_embeddings as f64 / lr.original_max_position_embeddings as f64;
        let mscale = if scale <= 1.0 {
            1.0f32
        } else {
            (1.0 + scale.ln() / (lr.original_max_position_embeddings as f64).ln()).sqrt() as f32
        };

        (inv_freq, mscale)
    } else {
        // Standard RoPE with partial rotation (no scaling factors).
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| (1.0 / base.powf(2.0 * i as f64 / rotary_dim as f64)) as f32)
            .collect();
        (inv_freq, 1.0f32)
    }
}

/// Apply rotary embedding with partial rotation support.
///
/// `x` shape: `[1, heads, seq, head_dim]`
/// `cos`/`sin` shape: `[1, 1, seq, rotary_dim]`
fn apply_partial_rope(
    x: &Array,
    cos: &Array,
    sin: &Array,
    rotary_dim: usize,
    head_dim: usize,
) -> Result<Array, Exception> {
    let rd = rotary_dim as i32;
    let half = (rotary_dim / 2) as i32;

    // Extract rotary portion: [1, heads, seq, rotary_dim]
    let x_rot = x.try_index((.., .., .., ..rd))?;

    // Split into first/second halves along last dim.
    let x1 = x_rot.try_index((.., .., .., ..half))?;
    let x2 = x_rot.try_index((.., .., .., half..rd))?;

    // Rotate: [-x2, x1]
    let neg_x2 = x2.negative()?;
    let x_rotated = concatenate_axis(&[&neg_x2, &x1], -1)?;

    // Apply: x_rot * cos + rotated * sin
    let rotated = x_rot.multiply(cos)?.add(&x_rotated.multiply(sin)?)?;

    if rotary_dim == head_dim {
        Ok(rotated)
    } else {
        // Concatenate passthrough dims.
        let x_pass = x.try_index((.., .., .., rd..head_dim as i32))?;
        concatenate_axis(&[&rotated, &x_pass], -1)
    }
}

// ---------------------------------------------------------------------------
// MlxPhi3MLP — fused gate_up_proj
// ---------------------------------------------------------------------------

struct MlxPhi3MLP {
    gate_up_proj: nn::Linear,
    down_proj: nn::Linear,
    intermediate_size: usize,
}

impl MlxPhi3MLP {
    fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_up_proj: nn::LinearBuilder::new(hidden_size, 2 * intermediate_size)
                .bias(false)
                .build()?,
            down_proj: nn::LinearBuilder::new(intermediate_size, hidden_size)
                .bias(false)
                .build()?,
            intermediate_size: intermediate_size as usize,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(
            &mut self.gate_up_proj.weight,
            weights,
            &format!("{prefix}.gate_up_proj.weight"),
        );
        assign_weight(
            &mut self.down_proj.weight,
            weights,
            &format!("{prefix}.down_proj.weight"),
        );
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate_up = self.gate_up_proj.forward(x)?;
        // Split along last dim: [seq, 2*intermediate] → gate [seq, intermediate], up [seq, intermediate]
        let parts = gate_up.split_axis(&[self.intermediate_size as i32], -1)?;
        let gate = nn::silu(&parts[0])?;
        let hidden = gate.multiply(&parts[1])?;
        self.down_proj.forward(&hidden)
    }
}

// ---------------------------------------------------------------------------
// MlxPhi3Attention — fused qkv_proj
// ---------------------------------------------------------------------------

struct MlxPhi3Attention {
    qkv_proj: nn::Linear,
    o_proj: nn::Linear,
    rope: Phi3Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    q_size: usize,
    kv_size: usize,
    sliding_window: Option<usize>,
    fuse_rope: bool,
}

impl MlxPhi3Attention {
    fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        let hidden = config.hidden_size as i32;
        let q_size = config.num_attention_heads * config.head_dim;
        let kv_size = config.num_kv_heads * config.head_dim;
        let qkv_size = (q_size + 2 * kv_size) as i32;

        Ok(Self {
            qkv_proj: nn::LinearBuilder::new(hidden, qkv_size)
                .bias(false)
                .build()?,
            o_proj: nn::LinearBuilder::new(q_size as i32, hidden)
                .bias(false)
                .build()?,
            rope: Phi3Rope::new(config),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
            q_size,
            kv_size,
            sliding_window: config.sliding_window,
            fuse_rope: vllm_config::SpansConfig::from_env().fuse_rope(),
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(
            &mut self.qkv_proj.weight,
            weights,
            &format!("{prefix}.qkv_proj.weight"),
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

        // Fused QKV projection.
        let qkv = self.qkv_proj.forward(hidden_states)?;
        // Split: [seq, q_size + kv_size + kv_size]
        let parts = qkv.split_axis(
            &[self.q_size as i32, (self.q_size + self.kv_size) as i32],
            -1,
        )?;

        // Reshape: [seq, size] -> [1, heads, seq, head_dim]
        let q = parts[0]
            .reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;
        let k = parts[1]
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;
        let v = parts[2]
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        // RoPE (supports partial rotation + LongRoPE).
        let (q, k) = if self.fuse_rope {
            let q = match &self.rope {
                Phi3Rope::Standard(rope) => {
                    crate::models::llama::apply_rope_to_cached_k(&q, rope, rope_offset)?
                }
                Phi3Rope::Custom { .. } => self.rope.apply_q_only(&q, rope_offset)?,
            };
            (q, k)
        } else {
            self.rope.apply(&q, &k, rope_offset)?
        };

        // KV cache update — pre-allocated buffer with O(1) slice_update.
        // When fuse_rope, K is stored without RoPE (position-independent).
        let (mut k, mut v) = crate::cache::kv_cache_update(cache, &k, &v)?;

        // When fuse_rope, apply RoPE to full cached K with offset 0.
        if self.fuse_rope {
            k = match &self.rope {
                Phi3Rope::Standard(rope) => {
                    crate::models::llama::apply_rope_to_cached_k(&k, rope, 0)?
                }
                Phi3Rope::Custom { .. } => self.rope.apply_k_only(&k, 0)?,
            };
        }

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

        // SDPA
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

    /// Batched forward: fused QKV projection on `[total_tokens, hidden]`,
    /// split per-request for reshape/RoPE/KV-cache/SDPA, then rejoin for O projection.
    ///
    /// When all requests are decode (q_len=1) and KV lengths match, SDPA is
    /// batched into a single kernel launch.
    fn forward_batch(
        &mut self,
        hidden_states: &Array,
        batch_info: &MlxBatchInfo,
        caches: &mut [Option<MlxLayerKvCache>],
    ) -> Result<Array, Exception> {
        // Batched fused QKV projection on [total_tokens, hidden].
        let qkv_all = self.qkv_proj.forward(hidden_states)?;

        // Check if all requests are decode (q_len=1) -> can attempt batched SDPA.
        let all_decode = batch_info.num_reqs > 1
            && batch_info.q_lens.iter().all(|&ql| ql == 1)
            && self.sliding_window.is_none();

        // Per-request: split QKV, reshape, RoPE, KV cache update.
        let mut per_req_q = Vec::with_capacity(batch_info.num_reqs);
        let mut per_req_k = Vec::with_capacity(batch_info.num_reqs);
        let mut per_req_v = Vec::with_capacity(batch_info.num_reqs);
        let mut kv_lens = Vec::with_capacity(batch_info.num_reqs);

        #[allow(clippy::needless_range_loop)]
        for i in 0..batch_info.num_reqs {
            let start = batch_info.offsets[i] as i32;
            let seq_len = batch_info.q_lens[i] as i32;
            let offset = batch_info.rope_offsets[i];

            // Slice this request's QKV: [seq_len, q_size + kv_size + kv_size]
            let qkv = qkv_all.try_index((start..start + seq_len, ..))?;

            // Split into Q, K, V along last dim.
            let parts = qkv.split_axis(
                &[self.q_size as i32, (self.q_size + self.kv_size) as i32],
                -1,
            )?;

            // Reshape: [seq, size] -> [1, heads, seq, head_dim]
            let q = parts[0]
                .reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?
                .transpose_axes(&[1, 0, 2])?
                .expand_dims(0)?;
            let k = parts[1]
                .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
                .transpose_axes(&[1, 0, 2])?
                .expand_dims(0)?;
            let v = parts[2]
                .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
                .transpose_axes(&[1, 0, 2])?
                .expand_dims(0)?;

            // RoPE (supports partial rotation + LongRoPE).
            let (q, k) = if self.fuse_rope {
                let q = match &self.rope {
                    Phi3Rope::Standard(rope) => {
                        crate::models::llama::apply_rope_to_cached_k(&q, rope, offset)?
                    }
                    Phi3Rope::Custom { .. } => self.rope.apply_q_only(&q, offset)?,
                };
                (q, k)
            } else {
                self.rope.apply(&q, &k, offset)?
            };

            // KV cache update.
            // When fuse_rope, K is stored without RoPE (position-independent).
            let (k, v) = crate::cache::kv_cache_update(&mut caches[i], &k, &v)?;

            // When fuse_rope, apply RoPE to full cached K with offset 0.
            let k = if self.fuse_rope {
                match &self.rope {
                    Phi3Rope::Standard(rope) => {
                        crate::models::llama::apply_rope_to_cached_k(&k, rope, 0)?
                    }
                    Phi3Rope::Custom { .. } => self.rope.apply_k_only(&k, 0)?,
                }
            } else {
                k
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
// MlxPhi3DecoderLayer
// ---------------------------------------------------------------------------

struct MlxPhi3DecoderLayer {
    self_attn: MlxPhi3Attention,
    mlp: MlxPhi3MLP,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
}

impl MlxPhi3DecoderLayer {
    fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: MlxPhi3Attention::new(config)?,
            mlp: MlxPhi3MLP::new(config.hidden_size as i32, config.intermediate_size as i32)?,
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

    /// Batched forward: norms and MLP run on `[total_tokens, hidden]`,
    /// attention splits per-request for RoPE/KV/SDPA.
    fn forward_batch(
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
}

// ---------------------------------------------------------------------------
// MlxPhi3ForCausalLM (float)
// ---------------------------------------------------------------------------

pub struct MlxPhi3ForCausalLM {
    embed_tokens: nn::Embedding,
    layers: Vec<MlxPhi3DecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: Option<nn::Linear>,
    tie_word_embeddings: bool,
    #[allow(dead_code)]
    config: LlamaConfig,
}

impl MlxPhi3ForCausalLM {
    fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(MlxPhi3DecoderLayer::new(config)?);
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
        config: &LlamaConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut model = Self::new(config)?;
        let weights = load_safetensors_weights(model_dir)?;
        model.load_weights(&weights);
        mlx_rs::transforms::eval(weights.values())?;
        Ok(model)
    }
}

impl super::MlxModel for MlxPhi3ForCausalLM {
    fn inject_lora(
        &mut self,
        adapter: &crate::lora::MlxLoraAdapter,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::lora::merge_lora_into_weight;
        let targets = &adapter.config.target_modules;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Phi-3 uses fused qkv_proj; only the unfused o_proj is injectable.
            if targets.iter().any(|t| t == "o_proj") {
                let key = format!("model.layers.{}.self_attn.o_proj", i);
                if let Some((a, b)) = adapter.weights.get(&key) {
                    merge_lora_into_weight(
                        &mut layer.self_attn.o_proj.weight,
                        a,
                        b,
                        adapter.scaling,
                    )?;
                }
            }
            // Phi-3 uses fused gate_up_proj; only the unfused down_proj is injectable.
            if targets.iter().any(|t| t == "down_proj") {
                let key = format!("model.layers.{}.mlp.down_proj", i);
                if let Some((a, b)) = adapter.weights.get(&key) {
                    merge_lora_into_weight(&mut layer.mlp.down_proj.weight, a, b, adapter.scaling)?;
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

        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, offset, &mut kv_cache[i])?;
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

        Ok(logits)
    }

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        _positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut kv_cache: MlxKvCache = (0..self.layers.len()).map(|_| None).collect();
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, 0, &mut kv_cache[i])?;
        }
        self.norm.forward(&hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Quantized Phi-3
// ---------------------------------------------------------------------------

struct MlxQuantizedPhi3MLP {
    gate_up_proj: nn::QuantizedLinear,
    down_proj: nn::QuantizedLinear,
    intermediate_size: usize,
}

impl MlxQuantizedPhi3MLP {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        qc: &QuantConfig,
        intermediate_size: usize,
    ) -> Self {
        Self {
            gate_up_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.gate_up_proj"),
                qc.group_size,
                qc.bits,
            ),
            down_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.down_proj"),
                qc.group_size,
                qc.bits,
            ),
            intermediate_size,
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let parts = gate_up.split_axis(&[self.intermediate_size as i32], -1)?;
        let gate = nn::silu(&parts[0])?;
        let hidden = gate.multiply(&parts[1])?;
        self.down_proj.forward(&hidden)
    }
}

struct MlxQuantizedPhi3Attention {
    qkv_proj: nn::QuantizedLinear,
    o_proj: nn::QuantizedLinear,
    rope: Phi3Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    q_size: usize,
    kv_size: usize,
    sliding_window: Option<usize>,
    fuse_rope: bool,
}

impl MlxQuantizedPhi3Attention {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &LlamaConfig,
        qc: &QuantConfig,
    ) -> Self {
        Self {
            qkv_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.qkv_proj"),
                qc.group_size,
                qc.bits,
            ),
            o_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.o_proj"),
                qc.group_size,
                qc.bits,
            ),
            rope: Phi3Rope::new(config),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
            q_size: config.num_attention_heads * config.head_dim,
            kv_size: config.num_kv_heads * config.head_dim,
            sliding_window: config.sliding_window,
            fuse_rope: vllm_config::SpansConfig::from_env().fuse_rope(),
        }
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        rope_offset: i32,
        cache: &mut Option<MlxLayerKvCache>,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);

        let qkv = self.qkv_proj.forward(hidden_states)?;
        let parts = qkv.split_axis(
            &[self.q_size as i32, (self.q_size + self.kv_size) as i32],
            -1,
        )?;

        let q = parts[0]
            .reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;
        let k = parts[1]
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;
        let v = parts[2]
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        let (q, k) = if self.fuse_rope {
            let q = match &self.rope {
                Phi3Rope::Standard(rope) => {
                    crate::models::llama::apply_rope_to_cached_k(&q, rope, rope_offset)?
                }
                Phi3Rope::Custom { .. } => self.rope.apply_q_only(&q, rope_offset)?,
            };
            (q, k)
        } else {
            self.rope.apply(&q, &k, rope_offset)?
        };

        // KV cache update — pre-allocated buffer with O(1) slice_update.
        // When fuse_rope, K is stored without RoPE (position-independent).
        let (mut k, mut v) = crate::cache::kv_cache_update(cache, &k, &v)?;

        // When fuse_rope, apply RoPE to full cached K with offset 0.
        if self.fuse_rope {
            k = match &self.rope {
                Phi3Rope::Standard(rope) => {
                    crate::models::llama::apply_rope_to_cached_k(&k, rope, 0)?
                }
                Phi3Rope::Custom { .. } => self.rope.apply_k_only(&k, 0)?,
            };
        }

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

    /// Batched forward for quantized attention: fused QKV projection on
    /// `[total_tokens, hidden]`, split per-request for reshape/RoPE/KV-cache/SDPA.
    fn forward_batch(
        &mut self,
        hidden_states: &Array,
        batch_info: &MlxBatchInfo,
        caches: &mut [Option<MlxLayerKvCache>],
    ) -> Result<Array, Exception> {
        // Batched fused QKV projection on [total_tokens, hidden].
        let qkv_all = self.qkv_proj.forward(hidden_states)?;

        let all_decode = batch_info.num_reqs > 1
            && batch_info.q_lens.iter().all(|&ql| ql == 1)
            && self.sliding_window.is_none();

        let mut per_req_q = Vec::with_capacity(batch_info.num_reqs);
        let mut per_req_k = Vec::with_capacity(batch_info.num_reqs);
        let mut per_req_v = Vec::with_capacity(batch_info.num_reqs);
        let mut kv_lens = Vec::with_capacity(batch_info.num_reqs);

        #[allow(clippy::needless_range_loop)]
        for i in 0..batch_info.num_reqs {
            let start = batch_info.offsets[i] as i32;
            let seq_len = batch_info.q_lens[i] as i32;
            let offset = batch_info.rope_offsets[i];

            let qkv = qkv_all.try_index((start..start + seq_len, ..))?;
            let parts = qkv.split_axis(
                &[self.q_size as i32, (self.q_size + self.kv_size) as i32],
                -1,
            )?;

            let q = parts[0]
                .reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?
                .transpose_axes(&[1, 0, 2])?
                .expand_dims(0)?;
            let k = parts[1]
                .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
                .transpose_axes(&[1, 0, 2])?
                .expand_dims(0)?;
            let v = parts[2]
                .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
                .transpose_axes(&[1, 0, 2])?
                .expand_dims(0)?;

            let (q, k) = if self.fuse_rope {
                let q = match &self.rope {
                    Phi3Rope::Standard(rope) => {
                        crate::models::llama::apply_rope_to_cached_k(&q, rope, offset)?
                    }
                    Phi3Rope::Custom { .. } => self.rope.apply_q_only(&q, offset)?,
                };
                (q, k)
            } else {
                self.rope.apply(&q, &k, offset)?
            };

            // When fuse_rope, K is stored without RoPE (position-independent).
            let (k, v) = crate::cache::kv_cache_update(&mut caches[i], &k, &v)?;

            // When fuse_rope, apply RoPE to full cached K with offset 0.
            let k = if self.fuse_rope {
                match &self.rope {
                    Phi3Rope::Standard(rope) => {
                        crate::models::llama::apply_rope_to_cached_k(&k, rope, 0)?
                    }
                    Phi3Rope::Custom { .. } => self.rope.apply_k_only(&k, 0)?,
                }
            } else {
                k
            };

            kv_lens.push(k.dim(2) as usize);
            per_req_q.push(q);
            per_req_k.push(k);
            per_req_v.push(v);
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
}

struct MlxQuantizedPhi3DecoderLayer {
    self_attn: MlxQuantizedPhi3Attention,
    mlp: MlxQuantizedPhi3MLP,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
}

impl MlxQuantizedPhi3DecoderLayer {
    fn from_weights(
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

        crate::models::quantized_llama::assign_weight(
            &mut input_layernorm.weight,
            weights,
            &format!("{prefix}.input_layernorm.weight"),
        );
        crate::models::quantized_llama::assign_weight(
            &mut post_attention_layernorm.weight,
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        );

        Ok(Self {
            self_attn: MlxQuantizedPhi3Attention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config,
                qc,
            ),
            mlp: MlxQuantizedPhi3MLP::from_weights(
                weights,
                &format!("{prefix}.mlp"),
                qc,
                config.intermediate_size,
            ),
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

    /// Batched forward: norms and MLP batched, attention splits per-request.
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

pub struct MlxQuantizedPhi3ForCausalLM {
    embed_tokens: MlxEmbedTokens,
    layers: Vec<MlxQuantizedPhi3DecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: Option<MlxLmHead>,
    tie_word_embeddings: bool,
    #[allow(dead_code)]
    config: LlamaConfig,
}

impl MlxQuantizedPhi3ForCausalLM {
    fn load(
        model_dir: &Path,
        config: &LlamaConfig,
        qc: &QuantConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let weights = load_safetensors_weights(model_dir)?;

        let embed_tokens =
            MlxEmbedTokens::from_weights(&weights, "model.embed_tokens", qc.group_size, qc.bits);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MlxQuantizedPhi3DecoderLayer::from_weights(
                &weights,
                &format!("model.layers.{i}"),
                config,
                qc,
            )?);
        }

        let mut norm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        crate::models::quantized_llama::assign_weight(
            &mut norm.weight,
            &weights,
            "model.norm.weight",
        );

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

impl super::MlxModel for MlxQuantizedPhi3ForCausalLM {
    fn forward(
        &mut self,
        input_ids: &Array,
        _positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        let offset = rope_offset.unwrap_or(0);
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, offset, &mut kv_cache[i])?;
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

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        _positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut kv_cache: MlxKvCache = (0..self.layers.len()).map(|_| None).collect();
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, 0, &mut kv_cache[i])?;
        }
        self.norm.forward(&hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

pub fn create_mlx_phi3(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let phi3_config = LlamaConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let model = MlxPhi3ForCausalLM::load(model_dir, &phi3_config, dtype)?;
    Ok(Box::new(model))
}

pub fn create_mlx_quantized_phi3(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let phi3_config = LlamaConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();
    tracing::info!(
        "Loading quantized MLX Phi-3 (group_size={}, bits={})",
        qc.group_size,
        qc.bits
    );
    let model = MlxQuantizedPhi3ForCausalLM::load(model_dir, &phi3_config, &qc, dtype)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
    fn test_phi3_mlp_forward() {
        let mut mlp = MlxPhi3MLP::new(32, 64).unwrap();
        let x = mlx_rs::ops::ones::<f32>(&[3, 32]).unwrap();
        let out = mlp.forward(&x).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[3, 32]);
    }

    #[test]
    fn test_phi3_attention_forward() {
        let config = test_config();
        let mut attn = MlxPhi3Attention::new(&config).unwrap();

        let x = mlx_rs::ops::ones::<f32>(&[4, 32]).unwrap();
        let mut cache = None;

        let out = attn.forward(&x, 0, &mut cache).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[4, 32]);
        assert!(cache.is_some());
    }

    #[test]
    fn test_phi3_model_forward() {
        let config = test_config();
        let mut model = MlxPhi3ForCausalLM::new(&config).unwrap();

        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        let logits = <MlxPhi3ForCausalLM as crate::models::MlxModel>::forward(
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
    fn test_phi3_prefill_and_decode() {
        let config = test_config();
        let mut model = MlxPhi3ForCausalLM::new(&config).unwrap();
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        // Prefill
        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let logits = <MlxPhi3ForCausalLM as crate::models::MlxModel>::forward(
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
        let logits2 = <MlxPhi3ForCausalLM as crate::models::MlxModel>::forward(
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

    /// Config mimicking Phi-4 mini: partial_rotary_factor=0.75, LongRoPE.
    fn phi4_mini_config() -> LlamaConfig {
        LlamaConfig {
            hidden_size: 32,
            num_attention_heads: 4,
            num_kv_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 256, // > original_max to trigger long factors
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            head_dim: 8,
            tie_word_embeddings: true,
            sliding_window: None,
            partial_rotary_factor: 0.75, // rotary_dim = 6
            long_rope_scaling: Some(LongRopeScaling {
                // 3 factors for half_dim = 3 (rotary_dim=6, half=3)
                short_factor: vec![1.0, 1.0, 1.0],
                long_factor: vec![2.0, 4.0, 8.0],
                original_max_position_embeddings: 64,
            }),
        }
    }

    #[test]
    fn test_phi3_partial_rotation_rope() {
        // partial_rotary_factor = 0.75, head_dim = 8 → rotary_dim = 6
        let config = LlamaConfig {
            partial_rotary_factor: 0.75,
            ..test_config()
        };
        let mut attn = MlxPhi3Attention::new(&config).unwrap();

        let x = mlx_rs::ops::ones::<f32>(&[3, 32]).unwrap();
        let mut cache = None;

        let out = attn.forward(&x, 0, &mut cache).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[3, 32]);
        assert!(cache.is_some());
    }

    #[test]
    fn test_phi3_longrope_attention_forward() {
        let config = phi4_mini_config();
        let mut attn = MlxPhi3Attention::new(&config).unwrap();

        let x = mlx_rs::ops::ones::<f32>(&[4, 32]).unwrap();
        let mut cache = None;

        let out = attn.forward(&x, 0, &mut cache).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[4, 32]);
        assert!(cache.is_some());
    }

    #[test]
    fn test_phi3_longrope_model_forward() {
        let config = phi4_mini_config();
        let mut model = MlxPhi3ForCausalLM::new(&config).unwrap();

        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        let logits = <MlxPhi3ForCausalLM as crate::models::MlxModel>::forward(
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
    fn test_phi3_longrope_prefill_and_decode() {
        let config = phi4_mini_config();
        let mut model = MlxPhi3ForCausalLM::new(&config).unwrap();
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        // Prefill
        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let logits = <MlxPhi3ForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &input_ids,
            &positions,
            &mut kv_cache,
            Some(0),
        )
        .unwrap();
        logits.eval().unwrap();

        // Decode
        let decode_ids = Array::from_iter(vec![15i32], &[1]);
        let decode_pos = Array::from_iter(vec![3i32], &[1]);
        let logits2 = <MlxPhi3ForCausalLM as crate::models::MlxModel>::forward(
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
    fn test_phi3_longrope_inv_freq_computation() {
        let half_dim = 3;
        let rotary_dim = 6;
        let base = 10000.0;
        let max_pos = 256;
        let lr = LongRopeScaling {
            short_factor: vec![1.0, 1.0, 1.0],
            long_factor: vec![2.0, 4.0, 8.0],
            original_max_position_embeddings: 64,
        };

        // max_pos > original → uses long factors.
        let (inv_freq, mscale) =
            compute_longrope_inv_freq(base, half_dim, rotary_dim, max_pos, Some(&lr));

        assert_eq!(inv_freq.len(), 3);
        // inv_freq[0] = 1.0 / (2.0 * base^(0/6)) = 1.0 / (2.0 * 1.0) = 0.5
        assert!((inv_freq[0] - 0.5).abs() < 1e-5);
        // mscale should be > 1.0 since max_pos > original_max_pos.
        assert!(mscale > 1.0, "mscale={mscale} should be > 1.0");

        // max_pos <= original → uses short factors (all 1.0).
        let (inv_freq_short, mscale_short) =
            compute_longrope_inv_freq(base, half_dim, rotary_dim, 32, Some(&lr));
        // Short factors are all 1.0 → same as standard RoPE.
        let (inv_freq_std, _) = compute_longrope_inv_freq(base, half_dim, rotary_dim, 32, None);
        for (a, b) in inv_freq_short.iter().zip(inv_freq_std.iter()) {
            assert!((a - b).abs() < 1e-6, "short factors should match standard");
        }
        assert!(
            (mscale_short - 1.0).abs() < 1e-6,
            "short mscale should be 1.0"
        );
    }

    #[test]
    fn test_phi3_sliding_window_trims_kv() {
        let config = LlamaConfig {
            sliding_window: Some(3),
            ..test_config()
        };
        let mut attn = MlxPhi3Attention::new(&config).unwrap();
        let mut cache = None;

        // Prefill 5 tokens → cache has 5 KV entries.
        let x = mlx_rs::ops::ones::<f32>(&[5, 32]).unwrap();
        let _ = attn.forward(&x, 0, &mut cache).unwrap();

        // Full cache should still store all 5 tokens.
        assert_eq!(
            cache.as_ref().unwrap().seq_len(),
            5,
            "cache should store all 5 tokens"
        );

        // Decode one more token → cache has 6 KV entries.
        let x2 = mlx_rs::ops::ones::<f32>(&[1, 32]).unwrap();
        let out = attn.forward(&x2, 5, &mut cache).unwrap();
        out.eval().unwrap();

        // Full cache should still store all 6 tokens.
        assert_eq!(
            cache.as_ref().unwrap().seq_len(),
            6,
            "cache should store all 6 tokens"
        );
        // Output shape should be correct.
        assert_eq!(out.shape(), &[1, 32]);
    }
}
