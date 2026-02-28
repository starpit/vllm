// SPDX-License-Identifier: Apache-2.0
//! Attention computation for model forward passes.
//!
//! Provides basic scaled dot-product attention for CPU testing.
//! For GPU inference, models use the paged attention kernels via
//! the `AttentionKernels` trait in `vllm-kernels`.
//!
//! Port of: `vllm/model_executor/layers/attention/`

use candle_core::{DType, Tensor};

use vllm_model::ModelResult;
use vllm_model::error::ModelError;

use crate::{LayerKvHandle, PagedKvBlockRefs};

/// Compute scaled dot-product attention with optional sliding window.
///
/// * `q` — queries, shape `[q_len, num_q_heads, head_dim]`
/// * `k` — keys, shape `[kv_len, num_kv_heads, head_dim]`
/// * `v` — values, shape `[kv_len, num_kv_heads, head_dim]`
/// * `scale` — attention scaling factor (typically `1 / sqrt(head_dim)`)
/// * `sliding_window` — if `Some(w)`, each query attends only to the most
///   recent `w` KV positions (plus itself). `None` means full attention.
///
/// Returns attention output of shape `[q_len, num_q_heads, head_dim]`.
///
/// Supports `q_len != kv_len` for KV-cached decode steps where `q_len = 1`
/// and `kv_len = cached_len + 1`. Uses a general causal mask where query
/// position i (absolute position `kv_len - q_len + i`) can attend to KV
/// positions `0..=(kv_len - q_len + i)`.
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    sliding_window: Option<usize>,
) -> ModelResult<Tensor> {
    let (q_len, num_q_heads, _head_dim) = q.dims3().map_err(ModelError::Candle)?;
    let (kv_len, num_kv_heads, _head_dim_k) = k.dims3().map_err(ModelError::Candle)?;

    // Handle GQA: repeat KV heads to match Q heads.
    let k_expanded;
    let v_expanded;
    let k_for_attn: &Tensor;
    let v_for_attn: &Tensor;
    if num_kv_heads < num_q_heads {
        let repeats = num_q_heads / num_kv_heads;
        k_expanded = repeat_kv(k, repeats)?;
        v_expanded = repeat_kv(v, repeats)?;
        k_for_attn = &k_expanded;
        v_for_attn = &v_expanded;
    } else {
        k_for_attn = k;
        v_for_attn = v;
    };

    // Transpose to [num_heads, seq_len, head_dim] for batched matmul.
    let q_t = q.transpose(0, 1).map_err(ModelError::Candle)?; // [num_q_heads, q_len, head_dim]
    let k_t = k_for_attn.transpose(0, 1).map_err(ModelError::Candle)?; // [num_q_heads, kv_len, head_dim]
    let v_t = v_for_attn
        .transpose(0, 1)
        .map_err(ModelError::Candle)?
        .contiguous()
        .map_err(ModelError::Candle)?;

    // Attention scores: Q * K^T * scale → [num_q_heads, q_len, kv_len]
    // Make tensors contiguous after transpose — required by Accelerate BLAS
    // and Metal matmul kernels which do not support strided inputs.
    let q_t = q_t.contiguous().map_err(ModelError::Candle)?;
    let k_tr = k_t
        .transpose(1, 2)
        .map_err(ModelError::Candle)?
        .contiguous()
        .map_err(ModelError::Candle)?;
    let scores = q_t.matmul(&k_tr).map_err(ModelError::Candle)?;
    let scores_scaled = (scores * scale).map_err(ModelError::Candle)?;

    // Apply causal mask if needed.
    // Single-token decode (q_len == 1) with no sliding window: the token can
    // attend to all kv_len positions, so no masking is needed.
    // With sliding window, even single-token decode needs a mask.
    let input_dtype = scores_scaled.dtype();
    let need_mask = q_len > 1 || sliding_window.is_some();

    if need_mask {
        let mask = create_causal_mask(
            q_len,
            kv_len,
            sliding_window,
            scores_scaled.dtype(),
            scores_scaled.device(),
        )?;
        let scores_masked = scores_scaled
            .broadcast_add(&mask)
            .map_err(ModelError::Candle)?;

        // Upcast to f32 for softmax numerical stability, then cast back.
        let scores_for_softmax = if needs_upcast(input_dtype) {
            scores_masked
                .to_dtype(DType::F32)
                .map_err(ModelError::Candle)?
        } else {
            scores_masked
        };
        let attn_weights = softmax_last_dim(&scores_for_softmax)?;
        let attn_weights = if needs_upcast(input_dtype) {
            attn_weights
                .to_dtype(input_dtype)
                .map_err(ModelError::Candle)?
        } else {
            attn_weights
        };
        let output = attn_weights.matmul(&v_t).map_err(ModelError::Candle)?;
        output.transpose(0, 1).map_err(ModelError::Candle)
    } else {
        // Single token, no sliding window: no masking needed.
        let scores_for_softmax = if needs_upcast(input_dtype) {
            scores_scaled
                .to_dtype(DType::F32)
                .map_err(ModelError::Candle)?
        } else {
            scores_scaled
        };
        let attn_weights = softmax_last_dim(&scores_for_softmax)?;
        let attn_weights = if needs_upcast(input_dtype) {
            attn_weights
                .to_dtype(input_dtype)
                .map_err(ModelError::Candle)?
        } else {
            attn_weights
        };
        let output = attn_weights.matmul(&v_t).map_err(ModelError::Candle)?;
        output.transpose(0, 1).map_err(ModelError::Candle)
    }
}

/// Returns `true` if the dtype should be upcast to f32 for reduction ops.
fn needs_upcast(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16)
}

/// Repeat KV heads to match the number of query heads (for GQA).
///
/// Input shape: `[num_tokens, num_kv_heads, head_dim]`
/// Output shape: `[num_tokens, num_kv_heads * repeats, head_dim]`
fn repeat_kv(x: &Tensor, repeats: usize) -> ModelResult<Tensor> {
    if repeats == 1 {
        return Ok(x.clone());
    }
    let (num_tokens, num_kv_heads, head_dim) = x.dims3().map_err(ModelError::Candle)?;

    // [num_tokens, num_kv_heads, 1, head_dim]
    let x_4d = x
        .reshape((num_tokens, num_kv_heads, 1, head_dim))
        .map_err(ModelError::Candle)?;

    // Expand: [num_tokens, num_kv_heads, repeats, head_dim]
    let expanded = x_4d
        .expand((num_tokens, num_kv_heads, repeats, head_dim))
        .map_err(ModelError::Candle)?;

    // Reshape: [num_tokens, num_kv_heads * repeats, head_dim]
    expanded
        .reshape((num_tokens, num_kv_heads * repeats, head_dim))
        .map_err(ModelError::Candle)
}

/// Create a causal attention mask, optionally with a sliding window.
///
/// Returns a tensor of shape `[1, q_len, kv_len]`.
///
/// Query position `i` has absolute position `kv_len - q_len + i` and can
/// attend to KV positions `0..=(kv_len - q_len + i)`. Positions beyond
/// that are masked with `-inf`.
///
/// When `sliding_window` is `Some(w)`, positions where
/// `kv_pos < abs_q_pos - w + 1` are also masked out.
///
/// When `q_len == kv_len` this produces the standard lower-triangular mask.
fn create_causal_mask(
    q_len: usize,
    kv_len: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &candle_core::Device,
) -> ModelResult<Tensor> {
    let offset = kv_len - q_len;
    let mut mask_data = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        let abs_pos = offset + i; // absolute position of query token i
        // Causal: mask future positions.
        for j in (abs_pos + 1)..kv_len {
            mask_data[i * kv_len + j] = f32::NEG_INFINITY;
        }
        // Sliding window: mask positions that are too far back.
        if let Some(w) = sliding_window
            && abs_pos >= w
        {
            let cutoff = abs_pos - w + 1;
            for j in 0..cutoff.min(kv_len) {
                mask_data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::from_slice(&mask_data, (q_len, kv_len), device)
        .map_err(ModelError::Candle)?
        .unsqueeze(0)
        .map_err(ModelError::Candle)?;

    if dtype != DType::F32 {
        mask.to_dtype(dtype).map_err(ModelError::Candle)
    } else {
        Ok(mask)
    }
}

/// Softmax along the last dimension.
fn softmax_last_dim(x: &Tensor) -> ModelResult<Tensor> {
    let max_vals = x
        .max_keepdim(candle_core::D::Minus1)
        .map_err(ModelError::Candle)?;
    let shifted = x.broadcast_sub(&max_vals).map_err(ModelError::Candle)?;
    let exp = shifted.exp().map_err(ModelError::Candle)?;
    let sum = exp
        .sum_keepdim(candle_core::D::Minus1)
        .map_err(ModelError::Candle)?;
    exp.broadcast_div(&sum).map_err(ModelError::Candle)
}

// ---------------------------------------------------------------------------
// Paged decode attention — reads K/V directly from block tensors
// ---------------------------------------------------------------------------

/// Compute attention for a single decode token reading K/V from paged blocks.
///
/// Instead of gathering all cached K/V into a contiguous tensor, this iterates
/// over blocks to compute scores, applies a global softmax, then computes the
/// weighted V sum per block. This eliminates the O(seq_len) gather copy.
///
/// * `q` — query, shape `[1, num_q_heads, head_dim]`
/// * `block_refs` — references to cached K/V block tensors
/// * `k_new` — new token K, shape `[1, num_kv_heads, head_dim]`
/// * `v_new` — new token V, shape `[1, num_kv_heads, head_dim]`
/// * `scale` — attention scaling factor (typically `1 / sqrt(head_dim)`)
/// * `sliding_window` — if `Some(w)`, only attend to the most recent `w`
///   tokens in the cache (plus the new token). Older blocks are skipped.
///
/// Returns attention output of shape `[1, num_q_heads, head_dim]`.
pub fn paged_decode_attention(
    q: &Tensor,
    block_refs: &PagedKvBlockRefs<'_>,
    k_new: &Tensor,
    v_new: &Tensor,
    scale: f64,
    sliding_window: Option<usize>,
) -> ModelResult<Tensor> {
    let (_one, num_q_heads, _head_dim) = q.dims3().map_err(ModelError::Candle)?;
    let (_one2, num_kv_heads, _hd) = k_new.dims3().map_err(ModelError::Candle)?;
    let gqa_repeats = num_q_heads / num_kv_heads;
    let input_dtype = q.dtype();

    // Q: [1, num_q_heads, head_dim] → [num_q_heads, 1, head_dim]
    let q_t = q
        .transpose(0, 1)
        .map_err(ModelError::Candle)?
        .contiguous()
        .map_err(ModelError::Candle)?;

    // Sliding window: determine which cached tokens are visible.
    // The new token is at absolute position `num_tokens`. With window `w`,
    // only positions `max(0, num_tokens + 1 - w)..=num_tokens` are visible.
    let total_tokens = block_refs.num_tokens + 1; // cached + new
    let window_start = sliding_window
        .map(|w| total_tokens.saturating_sub(w))
        .unwrap_or(0);

    // Collect partial scores from each cached block.
    let mut all_scores: Vec<Tensor> = Vec::new();
    let mut remaining = block_refs.num_tokens;
    // Track which blocks contribute (for the weighted V sum later).
    let mut block_contributions: Vec<(usize, usize)> = Vec::new(); // (block_idx, n_valid)
    let mut global_pos = 0usize; // absolute position of start of current block

    for (i, (k_blk, _v_blk)) in block_refs
        .k_blocks
        .iter()
        .zip(block_refs.v_blocks.iter())
        .enumerate()
    {
        let n = remaining.min(block_refs.block_size);
        if n == 0 {
            break;
        }

        let block_end = global_pos + n; // exclusive end of this block
        if block_end <= window_start {
            // Entire block is outside the sliding window — skip.
            remaining -= n;
            global_pos += n;
            continue;
        }

        // Determine the visible slice within this block.
        let local_start = window_start.saturating_sub(global_pos);
        let n_visible = n - local_start;

        // Narrow to visible tokens: [n_visible, num_kv_heads, head_dim]
        let k_valid = k_blk
            .narrow(0, local_start, n_visible)
            .map_err(ModelError::Candle)?;

        // GQA expand: [n_visible, num_q_heads, head_dim]
        let k_exp = if gqa_repeats > 1 {
            repeat_kv(&k_valid, gqa_repeats)?
        } else {
            k_valid
        };

        // [num_q_heads, n_visible, head_dim] → scores = Q @ K^T → [num_q_heads, 1, n_visible]
        let k_t = k_exp
            .transpose(0, 1)
            .map_err(ModelError::Candle)?
            .contiguous()
            .map_err(ModelError::Candle)?;
        let k_tr = k_t
            .transpose(1, 2)
            .map_err(ModelError::Candle)?
            .contiguous()
            .map_err(ModelError::Candle)?;
        let block_scores = q_t.matmul(&k_tr).map_err(ModelError::Candle)?;
        all_scores.push(block_scores);
        block_contributions.push((i, n_visible));

        remaining -= n;
        global_pos += n;
    }

    // New token scores: [1, num_kv_heads, head_dim] → expand → [num_q_heads, 1, 1]
    let k_new_squeezed = k_new.clone(); // [1, num_kv_heads, head_dim]
    let k_new_exp = if gqa_repeats > 1 {
        repeat_kv(&k_new_squeezed, gqa_repeats)?
    } else {
        k_new_squeezed
    };
    let k_new_t = k_new_exp
        .transpose(0, 1)
        .map_err(ModelError::Candle)?
        .contiguous()
        .map_err(ModelError::Candle)?;
    let k_new_tr = k_new_t
        .transpose(1, 2)
        .map_err(ModelError::Candle)?
        .contiguous()
        .map_err(ModelError::Candle)?;
    let new_scores = q_t.matmul(&k_new_tr).map_err(ModelError::Candle)?;
    all_scores.push(new_scores);

    // Cat all scores → [num_q_heads, 1, total_kv_len], scale, softmax.
    let scores_cat = Tensor::cat(&all_scores, 2).map_err(ModelError::Candle)?;
    let scores_scaled = (scores_cat * scale).map_err(ModelError::Candle)?;

    // Upcast to f32 for softmax numerical stability.
    let scores_f32 = if needs_upcast(input_dtype) {
        scores_scaled
            .to_dtype(DType::F32)
            .map_err(ModelError::Candle)?
    } else {
        scores_scaled
    };
    let attn_weights = softmax_last_dim(&scores_f32)?;
    let attn_weights = if needs_upcast(input_dtype) {
        attn_weights
            .to_dtype(input_dtype)
            .map_err(ModelError::Candle)?
    } else {
        attn_weights
    };

    // Weighted V sum per contributing block (only blocks within the window).
    let mut output_parts: Vec<Tensor> = Vec::new();
    let mut score_offset = 0usize;

    for &(blk_idx, n_visible) in &block_contributions {
        let v_blk = &block_refs.v_blocks[blk_idx];

        // For the first contributing block, we may need a partial slice.
        let blk_global_start = blk_idx * block_refs.block_size;
        let local_start = window_start.saturating_sub(blk_global_start);

        let v_valid = v_blk
            .narrow(0, local_start, n_visible)
            .map_err(ModelError::Candle)?;

        let v_exp = if gqa_repeats > 1 {
            repeat_kv(&v_valid, gqa_repeats)?
        } else {
            v_valid
        };

        // Extract weight slice: [num_q_heads, 1, n_visible]
        let w_slice = attn_weights
            .narrow(2, score_offset, n_visible)
            .map_err(ModelError::Candle)?;

        // V: [num_q_heads, n_visible, head_dim]
        let v_t = v_exp
            .transpose(0, 1)
            .map_err(ModelError::Candle)?
            .contiguous()
            .map_err(ModelError::Candle)?;

        // [num_q_heads, 1, n_visible] @ [num_q_heads, n_visible, head_dim] → [num_q_heads, 1, head_dim]
        let partial = w_slice.matmul(&v_t).map_err(ModelError::Candle)?;
        output_parts.push(partial);

        score_offset += n_visible;
    }

    // New token's weighted V.
    let v_new_exp = if gqa_repeats > 1 {
        repeat_kv(v_new, gqa_repeats)?
    } else {
        v_new.clone()
    };
    let v_new_t = v_new_exp
        .transpose(0, 1)
        .map_err(ModelError::Candle)?
        .contiguous()
        .map_err(ModelError::Candle)?;
    let w_new = attn_weights
        .narrow(2, score_offset, 1)
        .map_err(ModelError::Candle)?;
    let partial_new = w_new.matmul(&v_new_t).map_err(ModelError::Candle)?;
    output_parts.push(partial_new);

    // Sum all partial outputs.
    let mut output = output_parts[0].clone();
    for part in &output_parts[1..] {
        output = (output + part).map_err(ModelError::Candle)?;
    }

    // [num_q_heads, 1, head_dim] → [1, num_q_heads, head_dim]
    output.transpose(0, 1).map_err(ModelError::Candle)
}

// ---------------------------------------------------------------------------
// attention_with_cache — unified cache-merge + attention helper
// ---------------------------------------------------------------------------

/// Unified attention helper that merges KV cache and computes attention.
///
/// This replaces the duplicated cache-merge+attention pattern in all model
/// attention layers. For paged decode (q_len==1 with cached blocks), it uses
/// `paged_decode_attention` to read K/V directly from blocks. Otherwise it
/// falls back to the standard gather+concat+`scaled_dot_product_attention` path.
///
/// * `q` — queries after RoPE, shape `[q_len, num_q_heads, head_dim]`
/// * `k_new` — new keys after RoPE, shape `[q_len, num_kv_heads, head_dim]`
/// * `v_new` — new values, shape `[q_len, num_kv_heads, head_dim]`
/// * `scale` — attention scaling factor
/// * `kv_cache` — optional per-layer KV handle
/// * `sliding_window` — if `Some(w)`, each query attends only to the most
///   recent `w` KV positions. `None` means full attention.
///
/// Returns attention output of shape `[q_len, num_q_heads, head_dim]`.
pub fn attention_with_cache(
    q: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
    scale: f64,
    kv_cache: Option<LayerKvHandle<'_>>,
    sliding_window: Option<usize>,
) -> ModelResult<Tensor> {
    let q_len = q.dim(0).map_err(ModelError::Candle)?;

    if let Some(mut handle) = kv_cache {
        // Paged decode fast path: q_len == 1 with cached blocks.
        // Reads K/V directly from block tensors — no O(seq_len) gather.
        if q_len == 1
            && let Some(block_refs) = handle.paged_block_refs()
        {
            // TODO: replace paged_decode_attention body with fused Metal/CUDA
            // kernel. The current Rust per-block loop is a correct reference
            // implementation but slower than gather+single-matmul due to
            // per-block kernel dispatch overhead.
            let output =
                paged_decode_attention(q, &block_refs, k_new, v_new, scale, sliding_window)?;
            let k_token = k_new.squeeze(0).map_err(ModelError::Candle)?;
            let v_token = v_new.squeeze(0).map_err(ModelError::Candle)?;
            handle.store_new_token(k_token, v_token)?;
            return Ok(output);
        }

        // Standard path: gather cached K/V, concatenate, attend, store full.
        if let Some((cached_k, cached_v)) = handle.take_cached()? {
            let k_cat = Tensor::cat(&[&cached_k, k_new], 0).map_err(ModelError::Candle)?;
            let v_cat = Tensor::cat(&[&cached_v, v_new], 0).map_err(ModelError::Candle)?;

            // Sliding window: trim the cache to keep only the last `w` entries.
            // This saves memory and ensures the SDPA mask matches the data.
            let (k_for_attn, v_for_attn) = if let Some(w) = sliding_window {
                let kv_len = k_cat.dim(0).map_err(ModelError::Candle)?;
                if kv_len > w {
                    let start = kv_len - w;
                    let k_trimmed = k_cat.narrow(0, start, w).map_err(ModelError::Candle)?;
                    let v_trimmed = v_cat.narrow(0, start, w).map_err(ModelError::Candle)?;
                    // Store the full cache (tokens still in window for future steps).
                    handle.store(k_cat, v_cat)?;
                    (k_trimmed, v_trimmed)
                } else {
                    handle.store(k_cat.clone(), v_cat.clone())?;
                    (k_cat, v_cat)
                }
            } else {
                handle.store(k_cat.clone(), v_cat.clone())?;
                (k_cat, v_cat)
            };

            scaled_dot_product_attention(q, &k_for_attn, &v_for_attn, scale, sliding_window)
        } else {
            // First call (prefill): populate the cache.
            handle.store(k_new.clone(), v_new.clone())?;
            scaled_dot_product_attention(q, k_new, v_new, scale, sliding_window)
        }
    } else {
        // No caching requested.
        scaled_dot_product_attention(q, k_new, v_new, scale, sliding_window)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_attention_single_token() {
        // Single token, single head: attention output = V (trivially)
        let q = Tensor::ones(&[1, 1, 4], DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones(&[1, 1, 4], DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::new(&[[[1.0f32, 2.0, 3.0, 4.0]]], &Device::Cpu).unwrap();

        let out = scaled_dot_product_attention(&q, &k, &v, 0.5, None).unwrap();
        assert_eq!(out.dims(), &[1, 1, 4]);

        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-4);
        assert!((vals[1] - 2.0).abs() < 1e-4);
        assert!((vals[2] - 3.0).abs() < 1e-4);
        assert!((vals[3] - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_attention_shape_preservation() {
        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 8;

        let q = Tensor::ones(&[seq_len, num_heads, head_dim], DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones(&[seq_len, num_heads, head_dim], DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::ones(&[seq_len, num_heads, head_dim], DType::F32, &Device::Cpu).unwrap();

        let scale = 1.0 / (head_dim as f64).sqrt();
        let out = scaled_dot_product_attention(&q, &k, &v, scale, None).unwrap();
        assert_eq!(out.dims(), &[seq_len, num_heads, head_dim]);
    }

    #[test]
    fn test_attention_gqa() {
        // GQA: 4 query heads, 2 KV heads (repeat factor = 2)
        let seq_len = 3;
        let num_q_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;

        let q = Tensor::ones(&[seq_len, num_q_heads, head_dim], DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones(&[seq_len, num_kv_heads, head_dim], DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::ones(&[seq_len, num_kv_heads, head_dim], DType::F32, &Device::Cpu).unwrap();

        let scale = 1.0 / (head_dim as f64).sqrt();
        let out = scaled_dot_product_attention(&q, &k, &v, scale, None).unwrap();
        assert_eq!(out.dims(), &[seq_len, num_q_heads, head_dim]);
    }

    #[test]
    fn test_causal_mask_square() {
        // q_len == kv_len: standard lower-triangular mask.
        let mask = create_causal_mask(3, 3, None, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(mask.dims(), &[1, 3, 3]);

        let vals = mask.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        // Row 0: [0, -inf, -inf]
        assert_eq!(vals[0][0], 0.0);
        assert!(vals[0][1].is_infinite() && vals[0][1] < 0.0);
        assert!(vals[0][2].is_infinite() && vals[0][2] < 0.0);
        // Row 1: [0, 0, -inf]
        assert_eq!(vals[1][0], 0.0);
        assert_eq!(vals[1][1], 0.0);
        assert!(vals[1][2].is_infinite() && vals[1][2] < 0.0);
        // Row 2: [0, 0, 0]
        assert_eq!(vals[2][0], 0.0);
        assert_eq!(vals[2][1], 0.0);
        assert_eq!(vals[2][2], 0.0);
    }

    #[test]
    fn test_causal_mask_decode() {
        // q_len=2, kv_len=5: query tokens at absolute positions 3,4.
        // Token at pos 3 attends to KV 0..=3, token at pos 4 attends to 0..=4.
        let mask = create_causal_mask(2, 5, None, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(mask.dims(), &[1, 2, 5]);

        let vals = mask.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        // Row 0 (abs pos 3): [0, 0, 0, 0, -inf]
        for j in 0..4 {
            assert_eq!(vals[0][j], 0.0, "row 0, col {j}");
        }
        assert!(vals[0][4].is_infinite() && vals[0][4] < 0.0);
        // Row 1 (abs pos 4): [0, 0, 0, 0, 0]
        for j in 0..5 {
            assert_eq!(vals[1][j], 0.0, "row 1, col {j}");
        }
    }

    #[test]
    fn test_attention_decode_q_shorter_than_kv() {
        // Simulates a decode step: q_len=1, kv_len=4 (3 cached + 1 new).
        let num_heads = 2;
        let head_dim = 4;

        let q = Tensor::ones(&[1, num_heads, head_dim], DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones(&[4, num_heads, head_dim], DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::ones(&[4, num_heads, head_dim], DType::F32, &Device::Cpu).unwrap();

        let scale = 1.0 / (head_dim as f64).sqrt();
        let out = scaled_dot_product_attention(&q, &k, &v, scale, None).unwrap();
        // Output should be [1, num_heads, head_dim].
        assert_eq!(out.dims(), &[1, num_heads, head_dim]);
    }

    #[test]
    fn test_attention_partial_prefill() {
        // Simulates partial prefill: q_len=2, kv_len=5 (3 cached + 2 new).
        let num_heads = 2;
        let head_dim = 4;

        let q = Tensor::ones(&[2, num_heads, head_dim], DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones(&[5, num_heads, head_dim], DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::ones(&[5, num_heads, head_dim], DType::F32, &Device::Cpu).unwrap();

        let scale = 1.0 / (head_dim as f64).sqrt();
        let out = scaled_dot_product_attention(&q, &k, &v, scale, None).unwrap();
        assert_eq!(out.dims(), &[2, num_heads, head_dim]);
    }

    #[test]
    fn test_repeat_kv() {
        let x = Tensor::new(
            &[[[1.0f32, 2.0], [3.0, 4.0]]], // [1, 2, 2]
            &Device::Cpu,
        )
        .unwrap();

        let repeated = repeat_kv(&x, 3).unwrap();
        assert_eq!(repeated.dims(), &[1, 6, 2]);

        let vals = repeated.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        // Each KV head is repeated 3 times.
        assert_eq!(vals[0], vec![1.0, 2.0]);
        assert_eq!(vals[1], vec![1.0, 2.0]);
        assert_eq!(vals[2], vec![1.0, 2.0]);
        assert_eq!(vals[3], vec![3.0, 4.0]);
        assert_eq!(vals[4], vec![3.0, 4.0]);
        assert_eq!(vals[5], vec![3.0, 4.0]);
    }

    #[test]
    fn test_softmax() {
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0]]], &Device::Cpu).unwrap();
        let probs = softmax_last_dim(&x).unwrap();
        let vals = probs.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Probabilities should sum to 1.
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Higher logit = higher probability.
        assert!(vals[2] > vals[1]);
        assert!(vals[1] > vals[0]);
    }

    #[test]
    fn test_attention_f16() {
        let seq_len = 3;
        let num_heads = 2;
        let head_dim = 4;

        let q = Tensor::ones(&[seq_len, num_heads, head_dim], DType::F16, &Device::Cpu).unwrap();
        let k = Tensor::ones(&[seq_len, num_heads, head_dim], DType::F16, &Device::Cpu).unwrap();
        let v = Tensor::ones(&[seq_len, num_heads, head_dim], DType::F16, &Device::Cpu).unwrap();

        let scale = 1.0 / (head_dim as f64).sqrt();
        let out = scaled_dot_product_attention(&q, &k, &v, scale, None).unwrap();
        assert_eq!(out.dims(), &[seq_len, num_heads, head_dim]);
        assert_eq!(out.dtype(), DType::F16);
    }

    #[test]
    fn test_attention_f16_decode() {
        let num_heads = 2;
        let head_dim = 4;

        let q = Tensor::ones(&[1, num_heads, head_dim], DType::F16, &Device::Cpu).unwrap();
        let k = Tensor::ones(&[4, num_heads, head_dim], DType::F16, &Device::Cpu).unwrap();
        let v = Tensor::ones(&[4, num_heads, head_dim], DType::F16, &Device::Cpu).unwrap();

        let scale = 1.0 / (head_dim as f64).sqrt();
        let out = scaled_dot_product_attention(&q, &k, &v, scale, None).unwrap();
        assert_eq!(out.dims(), &[1, num_heads, head_dim]);
        assert_eq!(out.dtype(), DType::F16);
    }

    // -----------------------------------------------------------------------
    // Paged decode attention tests
    // -----------------------------------------------------------------------

    use crate::KvBlockPool;

    /// Helper: build a KvBlockPool and populate blocks from a contiguous KV tensor.
    fn make_pool_with_kv(
        num_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        dtype: DType,
        k_data: &Tensor,
        v_data: &Tensor,
    ) -> KvBlockPool {
        let num_blocks = (num_tokens + block_size - 1) / block_size;
        let num_layers = 1;
        let mut pool = KvBlockPool::new(
            num_blocks,
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            dtype,
            &Device::Cpu,
        )
        .unwrap();
        // Scatter the full K/V into blocks.
        pool.scatter_new_kv(0, &(0..num_blocks).collect::<Vec<_>>(), 0, k_data, v_data)
            .unwrap();
        pool
    }

    #[test]
    fn test_paged_decode_attention_basic() {
        // Single head, single block with 3 cached tokens + 1 new.
        let num_q_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 4;
        let block_size = 4;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Create some non-trivial K/V data for 3 cached tokens.
        let k_cached =
            Tensor::randn(0.0f32, 1.0, &[3, num_kv_heads, head_dim], &Device::Cpu).unwrap();
        let v_cached =
            Tensor::randn(0.0f32, 1.0, &[3, num_kv_heads, head_dim], &Device::Cpu).unwrap();

        let q = Tensor::randn(0.0f32, 1.0, &[1, num_q_heads, head_dim], &Device::Cpu).unwrap();
        let k_new = Tensor::randn(0.0f32, 1.0, &[1, num_kv_heads, head_dim], &Device::Cpu).unwrap();
        let v_new = Tensor::randn(0.0f32, 1.0, &[1, num_kv_heads, head_dim], &Device::Cpu).unwrap();

        // Reference: gather + standard attention.
        let k_full = Tensor::cat(&[&k_cached, &k_new], 0).unwrap();
        let v_full = Tensor::cat(&[&v_cached, &v_new], 0).unwrap();
        let ref_out = scaled_dot_product_attention(&q, &k_full, &v_full, scale, None).unwrap();

        // Paged: put cached tokens into pool.
        let pool = make_pool_with_kv(
            3,
            num_kv_heads,
            head_dim,
            block_size,
            DType::F32,
            &k_cached,
            &v_cached,
        );
        let block_refs = PagedKvBlockRefs {
            k_blocks: vec![pool.k_block(0, 0)],
            v_blocks: vec![pool.v_block(0, 0)],
            num_tokens: 3,
            block_size,
        };
        let paged_out =
            paged_decode_attention(&q, &block_refs, &k_new, &v_new, scale, None).unwrap();

        assert_eq!(paged_out.dims(), &[1, num_q_heads, head_dim]);
        let ref_vals = ref_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let paged_vals = paged_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (r, p) in ref_vals.iter().zip(paged_vals.iter()) {
            assert!((r - p).abs() < 1e-4, "ref={r}, paged={p}");
        }
    }

    #[test]
    fn test_paged_decode_attention_multi_block() {
        // 2 heads, 6 cached tokens across 2 blocks (block_size=4) + 1 new.
        let num_q_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 8;
        let block_size = 4;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let k_cached =
            Tensor::randn(0.0f32, 1.0, &[6, num_kv_heads, head_dim], &Device::Cpu).unwrap();
        let v_cached =
            Tensor::randn(0.0f32, 1.0, &[6, num_kv_heads, head_dim], &Device::Cpu).unwrap();

        let q = Tensor::randn(0.0f32, 1.0, &[1, num_q_heads, head_dim], &Device::Cpu).unwrap();
        let k_new = Tensor::randn(0.0f32, 1.0, &[1, num_kv_heads, head_dim], &Device::Cpu).unwrap();
        let v_new = Tensor::randn(0.0f32, 1.0, &[1, num_kv_heads, head_dim], &Device::Cpu).unwrap();

        // Reference.
        let k_full = Tensor::cat(&[&k_cached, &k_new], 0).unwrap();
        let v_full = Tensor::cat(&[&v_cached, &v_new], 0).unwrap();
        let ref_out = scaled_dot_product_attention(&q, &k_full, &v_full, scale, None).unwrap();

        // Paged.
        let pool = make_pool_with_kv(
            6,
            num_kv_heads,
            head_dim,
            block_size,
            DType::F32,
            &k_cached,
            &v_cached,
        );
        let block_refs = PagedKvBlockRefs {
            k_blocks: vec![pool.k_block(0, 0), pool.k_block(0, 1)],
            v_blocks: vec![pool.v_block(0, 0), pool.v_block(0, 1)],
            num_tokens: 6,
            block_size,
        };
        let paged_out =
            paged_decode_attention(&q, &block_refs, &k_new, &v_new, scale, None).unwrap();

        assert_eq!(paged_out.dims(), &[1, num_q_heads, head_dim]);
        let ref_vals = ref_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let paged_vals = paged_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (r, p) in ref_vals.iter().zip(paged_vals.iter()) {
            assert!((r - p).abs() < 1e-4, "ref={r}, paged={p}");
        }
    }

    #[test]
    fn test_paged_decode_attention_gqa() {
        // GQA: 4 query heads, 2 KV heads.
        let num_q_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let block_size = 4;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let k_cached =
            Tensor::randn(0.0f32, 1.0, &[5, num_kv_heads, head_dim], &Device::Cpu).unwrap();
        let v_cached =
            Tensor::randn(0.0f32, 1.0, &[5, num_kv_heads, head_dim], &Device::Cpu).unwrap();

        let q = Tensor::randn(0.0f32, 1.0, &[1, num_q_heads, head_dim], &Device::Cpu).unwrap();
        let k_new = Tensor::randn(0.0f32, 1.0, &[1, num_kv_heads, head_dim], &Device::Cpu).unwrap();
        let v_new = Tensor::randn(0.0f32, 1.0, &[1, num_kv_heads, head_dim], &Device::Cpu).unwrap();

        // Reference.
        let k_full = Tensor::cat(&[&k_cached, &k_new], 0).unwrap();
        let v_full = Tensor::cat(&[&v_cached, &v_new], 0).unwrap();
        let ref_out = scaled_dot_product_attention(&q, &k_full, &v_full, scale, None).unwrap();

        // Paged.
        let pool = make_pool_with_kv(
            5,
            num_kv_heads,
            head_dim,
            block_size,
            DType::F32,
            &k_cached,
            &v_cached,
        );
        let block_refs = PagedKvBlockRefs {
            k_blocks: vec![pool.k_block(0, 0), pool.k_block(0, 1)],
            v_blocks: vec![pool.v_block(0, 0), pool.v_block(0, 1)],
            num_tokens: 5,
            block_size,
        };
        let paged_out =
            paged_decode_attention(&q, &block_refs, &k_new, &v_new, scale, None).unwrap();

        assert_eq!(paged_out.dims(), &[1, num_q_heads, head_dim]);
        let ref_vals = ref_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let paged_vals = paged_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (r, p) in ref_vals.iter().zip(paged_vals.iter()) {
            assert!((r - p).abs() < 1e-4, "ref={r}, paged={p}");
        }
    }

    #[test]
    fn test_paged_decode_attention_f16() {
        // Half-precision correctness.
        let num_q_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 8;
        let block_size = 4;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Create f32 data then convert to f16.
        let k_cached_f32 =
            Tensor::randn(0.0f32, 1.0, &[3, num_kv_heads, head_dim], &Device::Cpu).unwrap();
        let v_cached_f32 =
            Tensor::randn(0.0f32, 1.0, &[3, num_kv_heads, head_dim], &Device::Cpu).unwrap();
        let k_cached = k_cached_f32.to_dtype(DType::F16).unwrap();
        let v_cached = v_cached_f32.to_dtype(DType::F16).unwrap();

        let q = Tensor::randn(0.0f32, 1.0, &[1, num_q_heads, head_dim], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let k_new = Tensor::randn(0.0f32, 1.0, &[1, num_kv_heads, head_dim], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let v_new = Tensor::randn(0.0f32, 1.0, &[1, num_kv_heads, head_dim], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        // Reference.
        let k_full = Tensor::cat(&[&k_cached, &k_new], 0).unwrap();
        let v_full = Tensor::cat(&[&v_cached, &v_new], 0).unwrap();
        let ref_out = scaled_dot_product_attention(&q, &k_full, &v_full, scale, None).unwrap();

        // Paged.
        let pool = make_pool_with_kv(
            3,
            num_kv_heads,
            head_dim,
            block_size,
            DType::F16,
            &k_cached,
            &v_cached,
        );
        let block_refs = PagedKvBlockRefs {
            k_blocks: vec![pool.k_block(0, 0)],
            v_blocks: vec![pool.v_block(0, 0)],
            num_tokens: 3,
            block_size,
        };
        let paged_out =
            paged_decode_attention(&q, &block_refs, &k_new, &v_new, scale, None).unwrap();

        assert_eq!(paged_out.dims(), &[1, num_q_heads, head_dim]);
        assert_eq!(paged_out.dtype(), DType::F16);
        // f16 has less precision, so use looser tolerance.
        let ref_vals = ref_out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let paged_vals = paged_out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (r, p) in ref_vals.iter().zip(paged_vals.iter()) {
            assert!((r - p).abs() < 0.05, "ref={r}, paged={p}");
        }
    }

    #[test]
    fn test_attention_with_cache_paged_decode() {
        // Full integration: attention_with_cache uses paged decode path.
        let num_q_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 8;
        let block_size = 4;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let k_cached =
            Tensor::randn(0.0f32, 1.0, &[3, num_kv_heads, head_dim], &Device::Cpu).unwrap();
        let v_cached =
            Tensor::randn(0.0f32, 1.0, &[3, num_kv_heads, head_dim], &Device::Cpu).unwrap();

        let q = Tensor::randn(0.0f32, 1.0, &[1, num_q_heads, head_dim], &Device::Cpu).unwrap();
        let k_new = Tensor::randn(0.0f32, 1.0, &[1, num_kv_heads, head_dim], &Device::Cpu).unwrap();
        let v_new = Tensor::randn(0.0f32, 1.0, &[1, num_kv_heads, head_dim], &Device::Cpu).unwrap();

        // Reference: standard attention.
        let k_full = Tensor::cat(&[&k_cached, &k_new], 0).unwrap();
        let v_full = Tensor::cat(&[&v_cached, &v_new], 0).unwrap();
        let ref_out = scaled_dot_product_attention(&q, &k_full, &v_full, scale, None).unwrap();

        // Set up paged storage.
        let mut pool = make_pool_with_kv(
            3,
            num_kv_heads,
            head_dim,
            block_size,
            DType::F32,
            &k_cached,
            &v_cached,
        );
        let block_ids = vec![0usize];
        let mut storage = crate::KvCacheStorage::paged(&mut pool, &block_ids, 3);
        let handle = storage.layer_handle(0);

        let paged_out =
            attention_with_cache(&q, &k_new, &v_new, scale, Some(handle), None).unwrap();

        // Flush pending writes.
        storage.flush().unwrap();

        assert_eq!(paged_out.dims(), &[1, num_q_heads, head_dim]);
        let ref_vals = ref_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let paged_vals = paged_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (r, p) in ref_vals.iter().zip(paged_vals.iter()) {
            assert!((r - p).abs() < 1e-4, "ref={r}, paged={p}");
        }

        // Verify the new token was written to the pool.
        assert_eq!(pool.tokens_stored(0), 4);
    }

    #[test]
    fn test_attention_with_cache_prefill() {
        // Prefill (q_len > 1) should use the standard path, not paged decode.
        let num_q_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 8;
        let block_size = 8;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let q = Tensor::randn(0.0f32, 1.0, &[4, num_q_heads, head_dim], &Device::Cpu).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, &[4, num_kv_heads, head_dim], &Device::Cpu).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, &[4, num_kv_heads, head_dim], &Device::Cpu).unwrap();

        // Reference: standard attention (no cache).
        let ref_out = scaled_dot_product_attention(&q, &k, &v, scale, None).unwrap();

        // Paged storage with tokens_before=0 (prefill).
        let mut pool = KvBlockPool::new(
            2,
            1,
            num_kv_heads,
            head_dim,
            block_size,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let block_ids = vec![0usize];
        let mut storage = crate::KvCacheStorage::paged(&mut pool, &block_ids, 0);
        let handle = storage.layer_handle(0);

        let paged_out = attention_with_cache(&q, &k, &v, scale, Some(handle), None).unwrap();
        storage.flush().unwrap();

        assert_eq!(paged_out.dims(), &[4, num_q_heads, head_dim]);
        let ref_vals = ref_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let paged_vals = paged_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (r, p) in ref_vals.iter().zip(paged_vals.iter()) {
            assert!((r - p).abs() < 1e-4, "ref={r}, paged={p}");
        }

        // Verify the prefill tokens were written to the pool.
        assert_eq!(pool.tokens_stored(0), 4);
    }

    // -----------------------------------------------------------------------
    // Sliding window attention tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sliding_window_mask() {
        // q_len=4, kv_len=4, window=2.
        // Token at abs pos 0 can attend to [0] (window covers pos -1..0, clamped to [0]).
        // Token at abs pos 1 can attend to [0, 1] (window covers pos 0..1).
        // Token at abs pos 2 can attend to [1, 2] (window covers pos 1..2).
        // Token at abs pos 3 can attend to [2, 3] (window covers pos 2..3).
        let mask = create_causal_mask(4, 4, Some(2), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(mask.dims(), &[1, 4, 4]);

        let vals = mask.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        // Row 0 (abs pos 0): [0, -inf, -inf, -inf]
        assert_eq!(vals[0][0], 0.0);
        assert!(vals[0][1].is_infinite() && vals[0][1] < 0.0);
        // Row 1 (abs pos 1): [0, 0, -inf, -inf]
        assert_eq!(vals[1][0], 0.0);
        assert_eq!(vals[1][1], 0.0);
        assert!(vals[1][2].is_infinite() && vals[1][2] < 0.0);
        // Row 2 (abs pos 2): [-inf, 0, 0, -inf] (pos 0 is outside window)
        assert!(vals[2][0].is_infinite() && vals[2][0] < 0.0);
        assert_eq!(vals[2][1], 0.0);
        assert_eq!(vals[2][2], 0.0);
        assert!(vals[2][3].is_infinite() && vals[2][3] < 0.0);
        // Row 3 (abs pos 3): [-inf, -inf, 0, 0] (pos 0,1 are outside window)
        assert!(vals[3][0].is_infinite() && vals[3][0] < 0.0);
        assert!(vals[3][1].is_infinite() && vals[3][1] < 0.0);
        assert_eq!(vals[3][2], 0.0);
        assert_eq!(vals[3][3], 0.0);
    }

    #[test]
    fn test_sliding_window_mask_larger_window() {
        // Window larger than seq_len: mask should be identical to plain causal.
        let with_window = create_causal_mask(3, 3, Some(10), DType::F32, &Device::Cpu).unwrap();
        let without_window = create_causal_mask(3, 3, None, DType::F32, &Device::Cpu).unwrap();

        let w_vals = with_window.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        let wo_vals = without_window.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    w_vals[i][j].is_infinite(),
                    wo_vals[i][j].is_infinite(),
                    "mismatch at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn test_attention_sliding_window_prefill() {
        // Prefill with sliding_window: short sequences should produce same output
        // as no window; longer sequences should differ.
        let num_heads = 2;
        let head_dim = 8;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Short sequence (len=3, window=10): within window — identical output.
        let q = Tensor::randn(0.0f32, 1.0, &[3, num_heads, head_dim], &Device::Cpu).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, &[3, num_heads, head_dim], &Device::Cpu).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, &[3, num_heads, head_dim], &Device::Cpu).unwrap();

        let out_none = scaled_dot_product_attention(&q, &k, &v, scale, None).unwrap();
        let out_large_window = scaled_dot_product_attention(&q, &k, &v, scale, Some(10)).unwrap();

        let vals_none = out_none.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vals_window = out_large_window
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (a, b) in vals_none.iter().zip(vals_window.iter()) {
            assert!((a - b).abs() < 1e-4, "none={a}, window={b}");
        }

        // Longer sequence (len=6, window=2): output should differ because window
        // restricts what each token can attend to.
        let q6 = Tensor::randn(0.0f32, 1.0, &[6, num_heads, head_dim], &Device::Cpu).unwrap();
        let k6 = Tensor::randn(0.0f32, 1.0, &[6, num_heads, head_dim], &Device::Cpu).unwrap();
        let v6 = Tensor::randn(0.0f32, 1.0, &[6, num_heads, head_dim], &Device::Cpu).unwrap();

        let out_full = scaled_dot_product_attention(&q6, &k6, &v6, scale, None).unwrap();
        let out_sw2 = scaled_dot_product_attention(&q6, &k6, &v6, scale, Some(2)).unwrap();

        // The last few tokens should differ because they can't see early tokens.
        let vals_full = out_full.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vals_sw2 = out_sw2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut differs = false;
        for (a, b) in vals_full.iter().zip(vals_sw2.iter()) {
            if (a - b).abs() > 1e-3 {
                differs = true;
                break;
            }
        }
        assert!(
            differs,
            "sliding window should produce different output for long sequences"
        );
    }

    #[test]
    fn test_attention_sliding_window_decode() {
        // Decode with sliding_window: single token decode with window=3 and 5 cached tokens.
        // Only the last 3 cached + 1 new = 4 visible positions (out of 6 total).
        let num_heads = 2;
        let head_dim = 8;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let q = Tensor::randn(0.0f32, 1.0, &[1, num_heads, head_dim], &Device::Cpu).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, &[6, num_heads, head_dim], &Device::Cpu).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, &[6, num_heads, head_dim], &Device::Cpu).unwrap();

        let out_full = scaled_dot_product_attention(&q, &k, &v, scale, None).unwrap();
        let out_sw3 = scaled_dot_product_attention(&q, &k, &v, scale, Some(3)).unwrap();

        assert_eq!(out_full.dims(), &[1, num_heads, head_dim]);
        assert_eq!(out_sw3.dims(), &[1, num_heads, head_dim]);

        // With window=3, the output should differ from full attention because
        // early tokens are masked out.
        let vals_full = out_full.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vals_sw3 = out_sw3.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut differs = false;
        for (a, b) in vals_full.iter().zip(vals_sw3.iter()) {
            if (a - b).abs() > 1e-3 {
                differs = true;
                break;
            }
        }
        assert!(
            differs,
            "sliding window decode should differ from full attention"
        );
    }

    #[test]
    fn test_paged_decode_attention_sliding_window() {
        // Paged decode with sliding_window: 7 cached tokens across 2 blocks
        // (block_size=4), window=3, + 1 new token.
        // Total = 8 tokens, window=3 → only positions 5,6,7 visible.
        let num_q_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 8;
        let block_size = 4;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let k_cached =
            Tensor::randn(0.0f32, 1.0, &[7, num_kv_heads, head_dim], &Device::Cpu).unwrap();
        let v_cached =
            Tensor::randn(0.0f32, 1.0, &[7, num_kv_heads, head_dim], &Device::Cpu).unwrap();

        let q = Tensor::randn(0.0f32, 1.0, &[1, num_q_heads, head_dim], &Device::Cpu).unwrap();
        let k_new = Tensor::randn(0.0f32, 1.0, &[1, num_kv_heads, head_dim], &Device::Cpu).unwrap();
        let v_new = Tensor::randn(0.0f32, 1.0, &[1, num_kv_heads, head_dim], &Device::Cpu).unwrap();

        // Reference: gather full + SDPA with sliding window mask.
        let k_full = Tensor::cat(&[&k_cached, &k_new], 0).unwrap();
        let v_full = Tensor::cat(&[&v_cached, &v_new], 0).unwrap();
        let ref_out = scaled_dot_product_attention(&q, &k_full, &v_full, scale, Some(3)).unwrap();

        // Paged.
        let pool = make_pool_with_kv(
            7,
            num_kv_heads,
            head_dim,
            block_size,
            DType::F32,
            &k_cached,
            &v_cached,
        );
        let block_refs = PagedKvBlockRefs {
            k_blocks: vec![pool.k_block(0, 0), pool.k_block(0, 1)],
            v_blocks: vec![pool.v_block(0, 0), pool.v_block(0, 1)],
            num_tokens: 7,
            block_size,
        };
        let paged_out =
            paged_decode_attention(&q, &block_refs, &k_new, &v_new, scale, Some(3)).unwrap();

        assert_eq!(paged_out.dims(), &[1, num_q_heads, head_dim]);
        let ref_vals = ref_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let paged_vals = paged_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (r, p) in ref_vals.iter().zip(paged_vals.iter()) {
            assert!((r - p).abs() < 1e-4, "ref={r}, paged={p}");
        }

        // Also verify it differs from no-window paged decode.
        let paged_full =
            paged_decode_attention(&q, &block_refs, &k_new, &v_new, scale, None).unwrap();
        let full_vals = paged_full.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut differs = false;
        for (w, f) in paged_vals.iter().zip(full_vals.iter()) {
            if (w - f).abs() > 1e-3 {
                differs = true;
                break;
            }
        }
        assert!(
            differs,
            "paged sliding window should differ from full attention"
        );
    }

    #[test]
    fn test_attention_with_cache_sliding_window_contiguous() {
        // Contiguous cache with sliding window: prefill 6 tokens, decode 1.
        // Window=3 means only last 3 KV entries are attended to.
        let num_q_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 8;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Prefill: 6 tokens with sliding_window=3.
        let q = Tensor::randn(0.0f32, 1.0, &[6, num_q_heads, head_dim], &Device::Cpu).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, &[6, num_kv_heads, head_dim], &Device::Cpu).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, &[6, num_kv_heads, head_dim], &Device::Cpu).unwrap();

        let mut cache: Option<(Tensor, Tensor)> = None;
        let handle = crate::LayerKvHandle::Contiguous(&mut cache);
        let out = attention_with_cache(&q, &k, &v, scale, Some(handle), Some(3)).unwrap();
        assert_eq!(out.dims(), &[6, num_q_heads, head_dim]);

        // Cache should hold full 6 tokens (we store full, trim only for attention).
        let (cached_k, _cached_v) = cache.as_ref().unwrap();
        assert_eq!(cached_k.dim(0).unwrap(), 6);

        // Decode: 1 token at position 6.
        let q_dec = Tensor::randn(0.0f32, 1.0, &[1, num_q_heads, head_dim], &Device::Cpu).unwrap();
        let k_dec = Tensor::randn(0.0f32, 1.0, &[1, num_kv_heads, head_dim], &Device::Cpu).unwrap();
        let v_dec = Tensor::randn(0.0f32, 1.0, &[1, num_kv_heads, head_dim], &Device::Cpu).unwrap();

        let handle = crate::LayerKvHandle::Contiguous(&mut cache);
        let out_dec =
            attention_with_cache(&q_dec, &k_dec, &v_dec, scale, Some(handle), Some(3)).unwrap();
        assert_eq!(out_dec.dims(), &[1, num_q_heads, head_dim]);

        // Cache should hold 7 tokens now.
        let (cached_k, _cached_v) = cache.as_ref().unwrap();
        assert_eq!(cached_k.dim(0).unwrap(), 7);
    }

    #[test]
    fn test_sliding_window_config_parsing() {
        // Verify sliding_window is parsed from config.json extras.
        use crate::llama::LlamaConfig;
        use vllm_model::weight::HfModelConfig;

        // With sliding_window.
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["MistralForCausalLM"],
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 32,
                "intermediate_size": 14336,
                "vocab_size": 32000,
                "sliding_window": 4096
            }"#,
        )
        .unwrap();

        let config = LlamaConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.sliding_window, Some(4096));

        // Without sliding_window.
        let hf_config2: HfModelConfig = serde_json::from_str(
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

        let config2 = LlamaConfig::from_hf_config(&hf_config2).unwrap();
        assert_eq!(config2.sliding_window, None);
    }
}
