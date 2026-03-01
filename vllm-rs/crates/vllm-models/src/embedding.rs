// SPDX-License-Identifier: Apache-2.0
//! Embedding utilities: pooling, normalization, and dimension truncation.
//!
//! Used by the embedding endpoint to convert model hidden states into
//! normalized embedding vectors.

use candle_core::{DType, Tensor};
use vllm_model::error::{ModelError, ModelResult};

/// Pooling strategy for extracting a single vector from hidden states.
#[derive(Debug, Clone, Copy)]
pub enum PoolingStrategy {
    /// Use the last token's hidden state (default for decoder models).
    Last,
}

/// Pool hidden states into a single embedding vector.
///
/// `hidden_states` has shape `[num_tokens, hidden_size]`.
/// Returns a tensor of shape `[hidden_size]`.
pub fn pool(hidden_states: &Tensor, strategy: PoolingStrategy) -> ModelResult<Tensor> {
    match strategy {
        PoolingStrategy::Last => {
            let num_tokens = hidden_states.dim(0).map_err(ModelError::Candle)?;
            if num_tokens == 0 {
                return Err(ModelError::Other("empty hidden states".into()));
            }
            hidden_states
                .narrow(0, num_tokens - 1, 1)
                .and_then(|t| t.squeeze(0))
                .map_err(ModelError::Candle)
        }
    }
}

/// L2-normalize an embedding vector.
///
/// `embedding` has shape `[hidden_size]`.
/// Returns a tensor of the same shape with L2 norm = 1.
pub fn l2_normalize(embedding: &Tensor) -> ModelResult<Tensor> {
    let embedding_f32 = embedding.to_dtype(DType::F32).map_err(ModelError::Candle)?;
    let norm = embedding_f32
        .sqr()
        .and_then(|t| t.sum_all())
        .and_then(|t| t.sqrt())
        .map_err(ModelError::Candle)?;
    let norm_val: f32 = norm.to_scalar().map_err(ModelError::Candle)?;
    if norm_val == 0.0 {
        return Ok(embedding_f32);
    }
    embedding_f32
        .broadcast_div(&norm)
        .map_err(ModelError::Candle)
}

/// Truncate an embedding to the specified number of dimensions (Matryoshka).
///
/// `embedding` has shape `[hidden_size]`.
/// Returns a tensor of shape `[dims]`.
pub fn truncate_dims(embedding: &Tensor, dims: usize) -> ModelResult<Tensor> {
    let current_dims = embedding.dim(0).map_err(ModelError::Candle)?;
    if dims >= current_dims {
        return Ok(embedding.clone());
    }
    embedding.narrow(0, 0, dims).map_err(ModelError::Candle)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_pool_last() {
        let device = Device::Cpu;
        // 3 tokens, hidden_size=4
        let hidden = Tensor::new(
            &[
                [1.0f32, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            &device,
        )
        .unwrap();
        let pooled = pool(&hidden, PoolingStrategy::Last).unwrap();
        let vals: Vec<f32> = pooled.to_vec1().unwrap();
        assert_eq!(vals, vec![9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_pool_last_single_token() {
        let device = Device::Cpu;
        let hidden = Tensor::new(&[[1.0f32, 2.0, 3.0]], &device).unwrap();
        let pooled = pool(&hidden, PoolingStrategy::Last).unwrap();
        let vals: Vec<f32> = pooled.to_vec1().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_l2_normalize() {
        let device = Device::Cpu;
        let emb = Tensor::new(&[3.0f32, 4.0], &device).unwrap();
        let normed = l2_normalize(&emb).unwrap();
        let vals: Vec<f32> = normed.to_vec1().unwrap();
        // norm = 5, so [0.6, 0.8]
        assert!((vals[0] - 0.6).abs() < 1e-5);
        assert!((vals[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_l2_normalize_unit() {
        let device = Device::Cpu;
        let emb = Tensor::new(&[1.0f32, 0.0, 0.0], &device).unwrap();
        let normed = l2_normalize(&emb).unwrap();
        let vals: Vec<f32> = normed.to_vec1().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!(vals[1].abs() < 1e-5);
    }

    #[test]
    fn test_l2_normalize_zero() {
        let device = Device::Cpu;
        let emb = Tensor::new(&[0.0f32, 0.0], &device).unwrap();
        let normed = l2_normalize(&emb).unwrap();
        let vals: Vec<f32> = normed.to_vec1().unwrap();
        assert_eq!(vals, vec![0.0, 0.0]);
    }

    #[test]
    fn test_truncate_dims() {
        let device = Device::Cpu;
        let emb = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &device).unwrap();
        let trunc = truncate_dims(&emb, 3).unwrap();
        let vals: Vec<f32> = trunc.to_vec1().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_truncate_dims_larger() {
        let device = Device::Cpu;
        let emb = Tensor::new(&[1.0f32, 2.0], &device).unwrap();
        let trunc = truncate_dims(&emb, 10).unwrap();
        let vals: Vec<f32> = trunc.to_vec1().unwrap();
        assert_eq!(vals, vec![1.0, 2.0]);
    }
}
