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

/// Compute scaled dot-product attention.
///
/// * `q` — queries, shape `[q_len, num_q_heads, head_dim]`
/// * `k` — keys, shape `[kv_len, num_kv_heads, head_dim]`
/// * `v` — values, shape `[kv_len, num_kv_heads, head_dim]`
/// * `scale` — attention scaling factor (typically `1 / sqrt(head_dim)`)
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
    // Single-token decode (q_len == 1): the token can attend to all kv_len
    // positions, so no masking is needed.
    if q_len > 1 {
        let mask =
            create_causal_mask(q_len, kv_len, scores_scaled.dtype(), scores_scaled.device())?;
        let scores_masked = scores_scaled
            .broadcast_add(&mask)
            .map_err(ModelError::Candle)?;

        let attn_weights = softmax_last_dim(&scores_masked)?;
        let output = attn_weights.matmul(&v_t).map_err(ModelError::Candle)?;
        output.transpose(0, 1).map_err(ModelError::Candle)
    } else {
        // Single token: no masking needed.
        let attn_weights = softmax_last_dim(&scores_scaled)?;
        let output = attn_weights.matmul(&v_t).map_err(ModelError::Candle)?;
        output.transpose(0, 1).map_err(ModelError::Candle)
    }
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

/// Create a causal attention mask.
///
/// Returns a tensor of shape `[1, q_len, kv_len]`.
///
/// Query position `i` has absolute position `kv_len - q_len + i` and can
/// attend to KV positions `0..=(kv_len - q_len + i)`. Positions beyond
/// that are masked with `-inf`.
///
/// When `q_len == kv_len` this produces the standard lower-triangular mask.
fn create_causal_mask(
    q_len: usize,
    kv_len: usize,
    dtype: DType,
    device: &candle_core::Device,
) -> ModelResult<Tensor> {
    let offset = kv_len - q_len;
    let mut mask_data = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        let abs_pos = offset + i; // absolute position of query token i
        for j in (abs_pos + 1)..kv_len {
            mask_data[i * kv_len + j] = f32::NEG_INFINITY;
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

        let out = scaled_dot_product_attention(&q, &k, &v, 0.5).unwrap();
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
        let out = scaled_dot_product_attention(&q, &k, &v, scale).unwrap();
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
        let out = scaled_dot_product_attention(&q, &k, &v, scale).unwrap();
        assert_eq!(out.dims(), &[seq_len, num_q_heads, head_dim]);
    }

    #[test]
    fn test_causal_mask_square() {
        // q_len == kv_len: standard lower-triangular mask.
        let mask = create_causal_mask(3, 3, DType::F32, &Device::Cpu).unwrap();
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
        let mask = create_causal_mask(2, 5, DType::F32, &Device::Cpu).unwrap();
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
        let out = scaled_dot_product_attention(&q, &k, &v, scale).unwrap();
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
        let out = scaled_dot_product_attention(&q, &k, &v, scale).unwrap();
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
}
