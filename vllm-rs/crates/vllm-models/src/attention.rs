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
// FlashAttention v2 — CUDA-only, F16/BF16 only (SM80+)
// ---------------------------------------------------------------------------

/// Compute attention using FlashAttention v2 for a single sequence.
///
/// Wraps `candle_flash_attn::flash_attn` / `flash_attn_windowed` for our
/// 3D tensor layout. Adds a batch dimension, calls FA2, removes it.
///
/// * `q` — queries, shape `[q_len, num_q_heads, head_dim]`
/// * `k` — keys, shape `[kv_len, num_kv_heads, head_dim]`
/// * `v` — values, shape `[kv_len, num_kv_heads, head_dim]`
/// * `scale` — attention scaling factor (typically `1 / sqrt(head_dim)`)
/// * `sliding_window` — if `Some(w)`, apply sliding window attention
///
/// Returns attention output of shape `[q_len, num_q_heads, head_dim]`.
#[cfg(feature = "cuda")]
fn flash_attention_single_seq(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    sliding_window: Option<usize>,
) -> ModelResult<Tensor> {
    // FA2 expects [batch, seq_len, num_heads, head_dim] — add batch dim.
    let q_4d = q.unsqueeze(0).map_err(ModelError::Candle)?;
    let k_4d = k.unsqueeze(0).map_err(ModelError::Candle)?;
    let v_4d = v.unsqueeze(0).map_err(ModelError::Candle)?;

    let out_4d = if let Some(w) = sliding_window {
        // Sliding window: our `sliding_window=w` means each query attends to `w`
        // positions total (itself + w-1 to the left). FA2's `window_size_left`
        // means positions to the left *excluding* self, so pass `w - 1`.
        // `window_size_right=Some(0)` = causal (no future positions).
        candle_flash_attn::flash_attn_windowed(
            &q_4d,
            &k_4d,
            &v_4d,
            scale as f32,
            Some(w.saturating_sub(1)),
            Some(0),
        )
        .map_err(ModelError::Candle)?
    } else {
        // Standard causal attention.
        candle_flash_attn::flash_attn(&q_4d, &k_4d, &v_4d, scale as f32, true)
            .map_err(ModelError::Candle)?
    };

    // Remove batch dim: [1, q_len, num_q_heads, head_dim] → [q_len, num_q_heads, head_dim]
    out_4d.squeeze(0).map_err(ModelError::Candle)
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
    block_refs: &PagedKvBlockRefs,
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

    // FlashAttention v2: CUDA + F16/BF16 only (SM80+).
    #[cfg(feature = "cuda")]
    let use_flash = q.device().is_cuda() && matches!(q.dtype(), DType::F16 | DType::BF16);
    #[cfg(not(feature = "cuda"))]
    let use_flash = false;

    if let Some(mut handle) = kv_cache {
        // CPU paged decode fast path: q_len == 1 with cached blocks.
        // Reads K/V directly from block tensors — no O(seq_len) gather.
        // On CUDA, skip this path — FA2 via the standard gather path is faster
        // than the per-block Rust loop.
        if !use_flash
            && q_len == 1
            && let Some(block_refs) = handle.paged_block_refs()
        {
            let output =
                paged_decode_attention(q, &block_refs, k_new, v_new, scale, sliding_window)?;
            let k_token = k_new.squeeze(0).map_err(ModelError::Candle)?;
            let v_token = v_new.squeeze(0).map_err(ModelError::Candle)?;
            handle.store_new_token(k_token, v_token)?;
            return Ok(output);
        }

        // Standard path: gather cached K/V, concatenate, attend, store.
        if let Some((cached_k, cached_v)) = handle.take_cached()? {
            let k_cat = Tensor::cat(&[&cached_k, k_new], 0).map_err(ModelError::Candle)?;
            let v_cat = Tensor::cat(&[&cached_v, v_new], 0).map_err(ModelError::Candle)?;

            // Paged decode (q_len==1): only store the single new token, avoiding
            // a full O(seq_len) scatter + clone of the concatenated KV.  The old
            // tokens are already in the pool (gather was non-destructive).
            let is_paged_decode = q_len == 1 && handle.paged_block_refs().is_some();
            if is_paged_decode {
                let k_token = k_new.squeeze(0).map_err(ModelError::Candle)?;
                let v_token = v_new.squeeze(0).map_err(ModelError::Candle)?;
                handle.store_new_token(k_token, v_token)?;
            }

            // Sliding window: trim for attention (storage already written above).
            let (k_for_attn, v_for_attn) = if let Some(w) = sliding_window {
                let kv_len = k_cat.dim(0).map_err(ModelError::Candle)?;
                if kv_len > w {
                    let trim_start = kv_len - w;
                    let k_trimmed = k_cat.narrow(0, trim_start, w).map_err(ModelError::Candle)?;
                    let v_trimmed = v_cat.narrow(0, trim_start, w).map_err(ModelError::Candle)?;
                    if !is_paged_decode {
                        // Contiguous / prefill: store full cache for future windows.
                        handle.store(k_cat, v_cat)?;
                    }
                    (k_trimmed, v_trimmed)
                } else {
                    if !is_paged_decode {
                        handle.store(k_cat.clone(), v_cat.clone())?;
                    }
                    (k_cat, v_cat)
                }
            } else {
                if !is_paged_decode {
                    handle.store(k_cat.clone(), v_cat.clone())?;
                }
                (k_cat, v_cat)
            };

            dispatch_attention(
                q,
                &k_for_attn,
                &v_for_attn,
                scale,
                sliding_window,
                use_flash,
            )
        } else {
            // First call (prefill): populate the cache.
            handle.store(k_new.clone(), v_new.clone())?;
            dispatch_attention(q, k_new, v_new, scale, sliding_window, use_flash)
        }
    } else {
        // No caching requested.
        dispatch_attention(q, k_new, v_new, scale, sliding_window, use_flash)
    }
}

/// Dispatch to FlashAttention v2 (CUDA F16/BF16) or naive SDPA.
#[inline]
fn dispatch_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    sliding_window: Option<usize>,
    _use_flash: bool,
) -> ModelResult<Tensor> {
    #[cfg(feature = "cuda")]
    if _use_flash {
        return flash_attention_single_seq(q, k, v, scale, sliding_window);
    }
    scaled_dot_product_attention(q, k, v, scale, sliding_window)
}

// ---------------------------------------------------------------------------
// Batched FlashAttention v2 via flash_attn_varlen — CUDA-only
// ---------------------------------------------------------------------------

/// Per-layer attention config for batched FA2.
#[cfg(feature = "cuda")]
pub struct BatchedAttnConfig {
    pub scale: f64,
    pub layer_idx: usize,
    pub sliding_window: Option<usize>,
}

/// Batched attention across multiple requests using a single `flash_attn_varlen` call.
///
/// Replaces the per-request `attention_with_cache` loop in `forward_batch()` with:
/// 1. Per-request KV cache management (gather cached, concat with new, store back)
/// 2. A single `flash_attn_varlen` / `flash_attn_varlen_windowed` call on concatenated Q/K/V
///
/// This reduces N separate FA2 kernel launches to 1, which is the continuous batching
/// optimization that makes Python vLLM fast.
///
/// Returns attention output of shape `[total_q_tokens, num_q_heads, head_dim]`.
#[cfg(feature = "cuda")]
pub fn batched_flash_attention_with_cache(
    q: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
    config: &BatchedAttnConfig,
    attn_meta: &crate::AttentionMetadata,
    storage: &mut crate::BatchedKvCacheStorage<'_>,
) -> ModelResult<Tensor> {
    let device = q.device();
    let num_reqs = attn_meta.num_reqs;
    let BatchedAttnConfig {
        scale,
        layer_idx,
        sliding_window,
    } = *config;

    // Phase 1: Per-request KV cache management.
    // Gather cached K/V, concatenate with new tokens, store back, collect for batched attention.
    let mut all_k_parts: Vec<Tensor> = Vec::with_capacity(num_reqs);
    let mut all_v_parts: Vec<Tensor> = Vec::with_capacity(num_reqs);
    let mut kv_cumlen: Vec<u32> = Vec::with_capacity(num_reqs + 1);
    kv_cumlen.push(0);
    let mut max_seqlen_q: usize = 0;
    let mut max_seqlen_k: usize = 0;

    for req_idx in 0..num_reqs {
        let (start, q_len) = attn_meta.request_slice(req_idx);
        let k_req = k_new.narrow(0, start, q_len).map_err(ModelError::Candle)?;
        let v_req = v_new.narrow(0, start, q_len).map_err(ModelError::Candle)?;

        // Cache: gather previously stored K/V, concat with new, store back.
        let mut handle = storage.request_layer_handle(req_idx, layer_idx);
        let (k_full, v_full) = if let Some((cached_k, cached_v)) = handle.take_cached()? {
            let k_cat = Tensor::cat(&[&cached_k, &k_req], 0).map_err(ModelError::Candle)?;
            let v_cat = Tensor::cat(&[&cached_v, &v_req], 0).map_err(ModelError::Candle)?;
            // Decode (q_len==1): store only the new token, not the full sequence.
            // The old tokens are already in the pool (gather was non-destructive).
            if q_len == 1 {
                let k_token = k_req.squeeze(0).map_err(ModelError::Candle)?;
                let v_token = v_req.squeeze(0).map_err(ModelError::Candle)?;
                handle.store_new_token(k_token, v_token)?;
            } else {
                handle.store(k_cat.clone(), v_cat.clone())?;
            }
            (k_cat, v_cat)
        } else {
            // First call (prefill): no cached data.
            handle.store(k_req.clone(), v_req.clone())?;
            (k_req, v_req)
        };
        // handle dropped here — borrow on storage released

        // Sliding window: trim K/V for attention (full sequence stored in cache above).
        let (k_for_attn, v_for_attn) = if let Some(w) = sliding_window {
            let kv_len = k_full.dim(0).map_err(ModelError::Candle)?;
            if kv_len > w {
                let trim_start = kv_len - w;
                (
                    k_full
                        .narrow(0, trim_start, w)
                        .map_err(ModelError::Candle)?,
                    v_full
                        .narrow(0, trim_start, w)
                        .map_err(ModelError::Candle)?,
                )
            } else {
                (k_full, v_full)
            }
        } else {
            (k_full, v_full)
        };

        let kv_len = k_for_attn.dim(0).map_err(ModelError::Candle)?;
        all_k_parts.push(k_for_attn);
        all_v_parts.push(v_for_attn);
        kv_cumlen.push(kv_cumlen.last().unwrap() + kv_len as u32);
        max_seqlen_q = max_seqlen_q.max(q_len);
        max_seqlen_k = max_seqlen_k.max(kv_len);
    }

    // Phase 2: Build flat K/V and cu_seqlens tensors on the GPU.
    let flat_k = Tensor::cat(&all_k_parts, 0).map_err(ModelError::Candle)?;
    let flat_v = Tensor::cat(&all_v_parts, 0).map_err(ModelError::Candle)?;

    // cu_seqlens: use cached GPU tensors when possible to avoid per-layer H2D copies.
    let cu_seqlens_q = attn_meta
        .cu_seqlens_q_gpu(device)
        .map_err(ModelError::Candle)?;
    // cu_seqlens_k can be cached when there's no sliding window (kv_len == seq_lens[i]).
    let cu_seqlens_k_owned;
    let cu_seqlens_k = if sliding_window.is_none() {
        attn_meta
            .cu_seqlens_k_gpu(device)
            .map_err(ModelError::Candle)?
    } else {
        cu_seqlens_k_owned =
            Tensor::from_slice(&kv_cumlen, kv_cumlen.len(), device).map_err(ModelError::Candle)?;
        &cu_seqlens_k_owned
    };

    // Phase 3: Single batched FlashAttention v2 call.
    if let Some(w) = sliding_window {
        candle_flash_attn::flash_attn_varlen_windowed(
            q,
            &flat_k,
            &flat_v,
            &cu_seqlens_q,
            &cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            scale as f32,
            Some(w.saturating_sub(1)),
            Some(0),
        )
        .map_err(ModelError::Candle)
    } else {
        candle_flash_attn::flash_attn_varlen(
            q,
            &flat_k,
            &flat_v,
            &cu_seqlens_q,
            &cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            scale as f32,
            true,
        )
        .map_err(ModelError::Candle)
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

    #[test]
    fn test_sliding_window_array_format() {
        // Newer Mistral models (3.x) use array-format sliding_window:
        // [null, 4096, null, 4096, ...]
        use crate::llama::LlamaConfig;
        use vllm_model::weight::HfModelConfig;

        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["MistralForCausalLM"],
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 4,
                "intermediate_size": 14336,
                "vocab_size": 32000,
                "sliding_window": [null, 4096, null, 4096]
            }"#,
        )
        .unwrap();

        let config = LlamaConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.sliding_window, Some(4096));
    }

    // -----------------------------------------------------------------------
    // FlashAttention v2 CUDA tests
    // -----------------------------------------------------------------------

    /// Helper: compare FA2 output against CPU SDPA reference.
    ///
    /// CPU reference is computed in F32 (candle CPU doesn't support BF16 matmul),
    /// with inputs quantized to the target dtype first to match the precision loss
    /// that FA2 sees on GPU.
    #[cfg(feature = "cuda")]
    fn assert_flash_matches_sdpa(
        q_len: usize,
        kv_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        sliding_window: Option<usize>,
        tol: f32,
    ) {
        let cuda = Device::cuda_if_available(0).unwrap();
        assert!(cuda.is_cuda(), "CUDA device required");

        let scale = 1.0 / (head_dim as f64).sqrt();

        // Generate random data on CPU in F32.
        let q_f32 =
            Tensor::randn(0.0f32, 1.0, &[q_len, num_q_heads, head_dim], &Device::Cpu).unwrap();
        let k_f32 =
            Tensor::randn(0.0f32, 1.0, &[kv_len, num_kv_heads, head_dim], &Device::Cpu).unwrap();
        let v_f32 =
            Tensor::randn(0.0f32, 1.0, &[kv_len, num_kv_heads, head_dim], &Device::Cpu).unwrap();

        // Round-trip through target dtype to match FA2's precision loss, then
        // compute CPU reference in F32 (candle CPU doesn't support BF16 matmul).
        let q_ref = q_f32.to_dtype(dtype).unwrap().to_dtype(DType::F32).unwrap();
        let k_ref = k_f32.to_dtype(dtype).unwrap().to_dtype(DType::F32).unwrap();
        let v_ref = v_f32.to_dtype(dtype).unwrap().to_dtype(DType::F32).unwrap();
        let ref_out =
            scaled_dot_product_attention(&q_ref, &k_ref, &v_ref, scale, sliding_window).unwrap();

        // CUDA FlashAttention: convert to target dtype, move to GPU.
        let q_cuda = q_f32.to_dtype(dtype).unwrap().to_device(&cuda).unwrap();
        let k_cuda = k_f32.to_dtype(dtype).unwrap().to_device(&cuda).unwrap();
        let v_cuda = v_f32.to_dtype(dtype).unwrap().to_device(&cuda).unwrap();
        let fa_out = flash_attention_single_seq(&q_cuda, &k_cuda, &v_cuda, scale, sliding_window)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        assert_eq!(fa_out.dims(), ref_out.dims());
        let ref_vals = ref_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let fa_vals = fa_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, (r, f)) in ref_vals.iter().zip(fa_vals.iter()).enumerate() {
            assert!(
                (r - f).abs() < tol,
                "mismatch at idx {i}: ref={r}, fa={f} (tol={tol})"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_flash_attn_decode_bf16() {
        // Single-token decode: q_len=1, kv_len=128, MHA.
        assert_flash_matches_sdpa(1, 128, 8, 8, 64, DType::BF16, None, 0.05);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_flash_attn_prefill_bf16() {
        // Multi-token prefill: q_len=64, kv_len=64, MHA.
        assert_flash_matches_sdpa(64, 64, 8, 8, 64, DType::BF16, None, 0.05);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_flash_attn_gqa_bf16() {
        // GQA: 8 Q heads, 2 KV heads, head_dim=128.
        assert_flash_matches_sdpa(1, 64, 8, 2, 128, DType::BF16, None, 0.05);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_flash_attn_prefill_gqa_bf16() {
        // GQA prefill: q_len=32, 8 Q heads, 2 KV heads, head_dim=128.
        assert_flash_matches_sdpa(32, 32, 8, 2, 128, DType::BF16, None, 0.05);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_flash_attn_decode_f16() {
        // Single-token decode in F16.
        assert_flash_matches_sdpa(1, 128, 8, 8, 64, DType::F16, None, 0.05);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_flash_attn_prefill_f16() {
        // Multi-token prefill in F16.
        assert_flash_matches_sdpa(64, 64, 4, 4, 64, DType::F16, None, 0.05);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_flash_attn_head_dim_128() {
        // Head dim 128 (common in larger models like Llama-70B).
        assert_flash_matches_sdpa(1, 64, 8, 8, 128, DType::BF16, None, 0.05);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_flash_attn_sliding_window() {
        // Sliding window attention (Mistral-style).
        assert_flash_matches_sdpa(32, 32, 8, 8, 64, DType::BF16, Some(16), 0.05);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_flash_attn_dispatch_in_attention_with_cache() {
        // Verify that attention_with_cache dispatches to FA2 on CUDA BF16.
        let cuda = Device::cuda_if_available(0).unwrap();
        assert!(cuda.is_cuda(), "CUDA device required");

        let num_q_heads = 8;
        let num_kv_heads = 2;
        let head_dim = 64;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Prefill: 16 tokens.
        let q = Tensor::randn(0.0f32, 1.0, &[16, num_q_heads, head_dim], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&cuda)
            .unwrap();
        let k = Tensor::randn(0.0f32, 1.0, &[16, num_kv_heads, head_dim], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&cuda)
            .unwrap();
        let v = Tensor::randn(0.0f32, 1.0, &[16, num_kv_heads, head_dim], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&cuda)
            .unwrap();

        // Use contiguous cache.
        let mut cache: Option<(Tensor, Tensor)> = None;
        let handle = crate::LayerKvHandle::Contiguous(&mut cache);
        let out = attention_with_cache(&q, &k, &v, scale, Some(handle), None).unwrap();
        assert_eq!(out.dims(), &[16, num_q_heads, head_dim]);
        assert_eq!(out.dtype(), DType::BF16);

        // Decode step.
        let q_dec = Tensor::randn(0.0f32, 1.0, &[1, num_q_heads, head_dim], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&cuda)
            .unwrap();
        let k_dec = Tensor::randn(0.0f32, 1.0, &[1, num_kv_heads, head_dim], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&cuda)
            .unwrap();
        let v_dec = Tensor::randn(0.0f32, 1.0, &[1, num_kv_heads, head_dim], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&cuda)
            .unwrap();

        let handle = crate::LayerKvHandle::Contiguous(&mut cache);
        let out_dec =
            attention_with_cache(&q_dec, &k_dec, &v_dec, scale, Some(handle), None).unwrap();
        assert_eq!(out_dec.dims(), &[1, num_q_heads, head_dim]);
        assert_eq!(out_dec.dtype(), DType::BF16);

        // Cache should hold 17 tokens now.
        let (cached_k, _) = cache.as_ref().unwrap();
        assert_eq!(cached_k.dim(0).unwrap(), 17);
    }

    // -----------------------------------------------------------------------
    // Batched FlashAttention v2 (flash_attn_varlen) CUDA tests
    // -----------------------------------------------------------------------

    /// Helper: compare batched FA2 (varlen) against per-request single-seq FA2 as reference.
    ///
    /// Builds a batch of `requests` where each entry is `(q_len, cached_kv_len)`.
    /// Runs per-request `flash_attention_single_seq` as the reference, then runs
    /// `batched_flash_attention_with_cache` and asserts outputs match within `tol`.
    #[cfg(feature = "cuda")]
    fn assert_batched_fa2_matches_per_request(
        requests: &[(usize, usize)], // (q_len, tokens_before)
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        sliding_window: Option<usize>,
        tol: f32,
    ) {
        let cuda = Device::cuda_if_available(0).unwrap();
        assert!(cuda.is_cuda(), "CUDA device required");

        let scale = 1.0 / (head_dim as f64).sqrt();
        let num_reqs = requests.len();
        let block_size = 16;

        // Generate per-request Q/K_new/V_new and cached K/V.
        let mut per_req_q: Vec<Tensor> = Vec::new();
        let mut per_req_k_new: Vec<Tensor> = Vec::new();
        let mut per_req_v_new: Vec<Tensor> = Vec::new();
        let mut per_req_k_cached: Vec<Option<Tensor>> = Vec::new();
        let mut per_req_v_cached: Vec<Option<Tensor>> = Vec::new();

        for &(q_len, tokens_before) in requests {
            let q = Tensor::randn(0.0f32, 1.0, &[q_len, num_q_heads, head_dim], &Device::Cpu)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
                .to_device(&cuda)
                .unwrap();
            let k_new = Tensor::randn(0.0f32, 1.0, &[q_len, num_kv_heads, head_dim], &Device::Cpu)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
                .to_device(&cuda)
                .unwrap();
            let v_new = Tensor::randn(0.0f32, 1.0, &[q_len, num_kv_heads, head_dim], &Device::Cpu)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
                .to_device(&cuda)
                .unwrap();

            if tokens_before > 0 {
                let k_cached = Tensor::randn(
                    0.0f32,
                    1.0,
                    &[tokens_before, num_kv_heads, head_dim],
                    &Device::Cpu,
                )
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
                .to_device(&cuda)
                .unwrap();
                let v_cached = Tensor::randn(
                    0.0f32,
                    1.0,
                    &[tokens_before, num_kv_heads, head_dim],
                    &Device::Cpu,
                )
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
                .to_device(&cuda)
                .unwrap();
                per_req_k_cached.push(Some(k_cached));
                per_req_v_cached.push(Some(v_cached));
            } else {
                per_req_k_cached.push(None);
                per_req_v_cached.push(None);
            }

            per_req_q.push(q);
            per_req_k_new.push(k_new);
            per_req_v_new.push(v_new);
        }

        // --- Reference: per-request single-seq FA2 ---
        let mut ref_outputs: Vec<Tensor> = Vec::new();
        for req_idx in 0..num_reqs {
            let k_full = if let Some(ref cached) = per_req_k_cached[req_idx] {
                Tensor::cat(&[cached, &per_req_k_new[req_idx]], 0).unwrap()
            } else {
                per_req_k_new[req_idx].clone()
            };
            let v_full = if let Some(ref cached) = per_req_v_cached[req_idx] {
                Tensor::cat(&[cached, &per_req_v_new[req_idx]], 0).unwrap()
            } else {
                per_req_v_new[req_idx].clone()
            };

            // Apply same sliding window trim as batched path.
            let (k_for_attn, v_for_attn) = if let Some(w) = sliding_window {
                let kv_len = k_full.dim(0).unwrap();
                if kv_len > w {
                    let s = kv_len - w;
                    (
                        k_full.narrow(0, s, w).unwrap(),
                        v_full.narrow(0, s, w).unwrap(),
                    )
                } else {
                    (k_full, v_full)
                }
            } else {
                (k_full, v_full)
            };

            let out = flash_attention_single_seq(
                &per_req_q[req_idx],
                &k_for_attn,
                &v_for_attn,
                scale,
                sliding_window,
            )
            .unwrap();
            ref_outputs.push(out);
        }
        let ref_output = Tensor::cat(&ref_outputs, 0).unwrap();

        // --- Batched: flash_attn_varlen via batched_flash_attention_with_cache ---
        // Build flat Q/K_new/V_new + AttentionMetadata + BatchedKvCacheStorage.
        let flat_q = Tensor::cat(&per_req_q, 0).unwrap();
        let flat_k_new = Tensor::cat(&per_req_k_new, 0).unwrap();
        let flat_v_new = Tensor::cat(&per_req_v_new, 0).unwrap();

        // Build AttentionMetadata.
        let mut query_start_loc = vec![0usize];
        let mut q_lens = Vec::new();
        let mut seq_lens = Vec::new();
        let mut block_ids_all = Vec::new();
        let mut tokens_before_all = Vec::new();
        let mut is_prefill = Vec::new();
        let mut total_tokens = 0usize;

        for (req_idx, &(q_len, tokens_before)) in requests.iter().enumerate() {
            total_tokens += q_len;
            query_start_loc.push(total_tokens);
            q_lens.push(q_len);
            seq_lens.push(tokens_before + q_len);
            tokens_before_all.push(tokens_before);
            is_prefill.push(tokens_before == 0);

            // Allocate enough blocks for the full sequence.
            let total_kv = tokens_before + q_len;
            let num_blocks = (total_kv + block_size - 1) / block_size;
            let base_block = req_idx * 10; // spread blocks apart
            block_ids_all.push((0..num_blocks).map(|b| base_block + b).collect::<Vec<_>>());
        }

        let attn_meta = crate::AttentionMetadata::new(
            num_reqs,
            total_tokens,
            query_start_loc,
            q_lens,
            seq_lens,
            block_ids_all.clone(),
            tokens_before_all.clone(),
            is_prefill,
            (0..num_reqs).map(|i| format!("req-{i}")).collect(),
        );

        // Build pool with enough blocks, pre-populate cached K/V.
        let max_block_id = block_ids_all
            .iter()
            .flat_map(|v| v.iter())
            .copied()
            .max()
            .unwrap_or(0);
        let num_pool_blocks = max_block_id + 1;
        let mut pool = KvBlockPool::new(
            num_pool_blocks,
            1, // single layer
            num_kv_heads,
            head_dim,
            block_size,
            dtype,
            &cuda,
        )
        .unwrap();

        // Pre-populate cached K/V into the pool.
        for (req_idx, &(_, tokens_before)) in requests.iter().enumerate() {
            if tokens_before > 0 {
                let cached_k = per_req_k_cached[req_idx].as_ref().unwrap();
                let cached_v = per_req_v_cached[req_idx].as_ref().unwrap();
                pool.scatter_new_kv(
                    0, // layer 0
                    &block_ids_all[req_idx],
                    0, // offset 0
                    cached_k,
                    cached_v,
                )
                .unwrap();
            }
        }

        let mut batched_storage =
            crate::BatchedKvCacheStorage::new(&mut pool, block_ids_all, tokens_before_all);

        let config = BatchedAttnConfig {
            scale,
            layer_idx: 0,
            sliding_window,
        };
        let batched_output = batched_flash_attention_with_cache(
            &flat_q,
            &flat_k_new,
            &flat_v_new,
            &config,
            &attn_meta,
            &mut batched_storage,
        )
        .unwrap();

        // Flush deferred writes.
        batched_storage.flush_all().unwrap();

        // --- Compare ---
        assert_eq!(batched_output.dims(), ref_output.dims());
        let ref_vals = ref_output
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let batched_vals = batched_output
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, (r, b)) in ref_vals.iter().zip(batched_vals.iter()).enumerate() {
            assert!(
                (r - b).abs() < tol,
                "mismatch at idx {i}: ref={r}, batched={b} (tol={tol})"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_batched_fa2_all_decode() {
        // 4 decode requests (q_len=1), different cache lengths.
        assert_batched_fa2_matches_per_request(
            &[(1, 32), (1, 64), (1, 16), (1, 128)],
            8,  // num_q_heads
            2,  // num_kv_heads (GQA)
            64, // head_dim
            DType::BF16,
            None,
            0.05,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_batched_fa2_all_prefill() {
        // 3 prefill requests (no cached data), different prompt lengths.
        assert_batched_fa2_matches_per_request(
            &[(16, 0), (32, 0), (8, 0)],
            8,  // num_q_heads
            8,  // num_kv_heads (MHA)
            64, // head_dim
            DType::BF16,
            None,
            0.05,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_batched_fa2_mixed_prefill_decode() {
        // Mixed batch: 1 prefill + 2 decode requests.
        assert_batched_fa2_matches_per_request(
            &[(24, 0), (1, 50), (1, 100)],
            8,   // num_q_heads
            2,   // num_kv_heads (GQA)
            128, // head_dim
            DType::BF16,
            None,
            0.05,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_batched_fa2_sliding_window() {
        // 3 decode requests with sliding window.
        assert_batched_fa2_matches_per_request(
            &[(1, 64), (1, 128), (1, 32)],
            8,  // num_q_heads
            2,  // num_kv_heads (GQA)
            64, // head_dim
            DType::BF16,
            Some(32), // sliding window
            0.05,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_batched_fa2_f16() {
        // Same as all_decode but in F16.
        assert_batched_fa2_matches_per_request(
            &[(1, 32), (1, 64)],
            4,  // num_q_heads
            4,  // num_kv_heads (MHA)
            64, // head_dim
            DType::F16,
            None,
            0.05,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_batched_fa2_single_request() {
        // Single request: batched path should produce same output as single-seq path.
        assert_batched_fa2_matches_per_request(
            &[(1, 64)],
            8,  // num_q_heads
            2,  // num_kv_heads (GQA)
            64, // head_dim
            DType::BF16,
            None,
            0.05,
        );
    }
}
