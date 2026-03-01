// SPDX-License-Identifier: Apache-2.0
//! Embedding utilities: pooling, normalization, and dimension truncation.
//!
//! Used by the embedding endpoint to convert model hidden states into
//! normalized embedding vectors.

use candle_core::{DType, Tensor};
use vllm_model::error::{ModelError, ModelResult};

/// Pooling strategy for extracting a single vector from hidden states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Use the last token's hidden state (default for decoder models).
    Last,
    /// Use the first token's hidden state (CLS token for encoder models).
    Cls,
    /// Average all token hidden states.
    Mean,
}

impl std::str::FromStr for PoolingStrategy {
    type Err = String;

    /// Parse a pooling strategy from a string.
    ///
    /// Accepts: "last", "cls", "mean" (case-insensitive).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "last" => Ok(Self::Last),
            "cls" => Ok(Self::Cls),
            "mean" => Ok(Self::Mean),
            other => Err(format!("unknown pooling strategy: {other}")),
        }
    }
}

/// Pool hidden states into a single embedding vector.
///
/// `hidden_states` has shape `[num_tokens, hidden_size]`.
/// Returns a tensor of shape `[hidden_size]`.
pub fn pool(hidden_states: &Tensor, strategy: PoolingStrategy) -> ModelResult<Tensor> {
    let num_tokens = hidden_states.dim(0).map_err(ModelError::Candle)?;
    if num_tokens == 0 {
        return Err(ModelError::Other("empty hidden states".into()));
    }
    match strategy {
        PoolingStrategy::Last => hidden_states
            .narrow(0, num_tokens - 1, 1)
            .and_then(|t| t.squeeze(0))
            .map_err(ModelError::Candle),
        PoolingStrategy::Cls => hidden_states
            .narrow(0, 0, 1)
            .and_then(|t| t.squeeze(0))
            .map_err(ModelError::Candle),
        PoolingStrategy::Mean => {
            // Average across the token dimension (dim 0).
            let sum = hidden_states.sum(0).map_err(ModelError::Candle)?;
            (sum / num_tokens as f64).map_err(ModelError::Candle)
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

/// Detect pooling strategy from a sentence-transformers `1_Pooling/config.json`.
///
/// The config has boolean fields like `pooling_mode_mean_tokens`,
/// `pooling_mode_cls_token`, `pooling_mode_lasttoken`. The first
/// `true` field wins.
///
/// Returns `None` if the file doesn't exist or has no recognized mode.
pub fn detect_pooling_strategy(model_dir: &std::path::Path) -> Option<PoolingStrategy> {
    let config_path = model_dir.join("1_Pooling").join("config.json");
    let data = std::fs::read_to_string(config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&data).ok()?;

    if json
        .get("pooling_mode_mean_tokens")
        .and_then(|v| v.as_bool())
        == Some(true)
    {
        return Some(PoolingStrategy::Mean);
    }
    if json.get("pooling_mode_cls_token").and_then(|v| v.as_bool()) == Some(true) {
        return Some(PoolingStrategy::Cls);
    }
    if json.get("pooling_mode_lasttoken").and_then(|v| v.as_bool()) == Some(true) {
        return Some(PoolingStrategy::Last);
    }

    None
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

    #[test]
    fn test_pool_cls() {
        let device = Device::Cpu;
        let hidden = Tensor::new(
            &[
                [1.0f32, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            &device,
        )
        .unwrap();
        let pooled = pool(&hidden, PoolingStrategy::Cls).unwrap();
        let vals: Vec<f32> = pooled.to_vec1().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_pool_cls_single_token() {
        let device = Device::Cpu;
        let hidden = Tensor::new(&[[7.0f32, 8.0, 9.0]], &device).unwrap();
        let pooled = pool(&hidden, PoolingStrategy::Cls).unwrap();
        let vals: Vec<f32> = pooled.to_vec1().unwrap();
        assert_eq!(vals, vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_pool_mean() {
        let device = Device::Cpu;
        let hidden = Tensor::new(
            &[[1.0f32, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]],
            &device,
        )
        .unwrap();
        let pooled = pool(&hidden, PoolingStrategy::Mean).unwrap();
        let vals: Vec<f32> = pooled.to_vec1().unwrap();
        assert!((vals[0] - 3.0).abs() < 1e-5);
        assert!((vals[1] - 4.0).abs() < 1e-5);
        assert!((vals[2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_pool_mean_single_token() {
        let device = Device::Cpu;
        let hidden = Tensor::new(&[[2.0f32, 4.0]], &device).unwrap();
        let pooled = pool(&hidden, PoolingStrategy::Mean).unwrap();
        let vals: Vec<f32> = pooled.to_vec1().unwrap();
        assert!((vals[0] - 2.0).abs() < 1e-5);
        assert!((vals[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_pool_empty_hidden_states() {
        let device = Device::Cpu;
        let hidden = Tensor::zeros((0, 4), DType::F32, &device).unwrap();
        assert!(pool(&hidden, PoolingStrategy::Last).is_err());
        assert!(pool(&hidden, PoolingStrategy::Cls).is_err());
        assert!(pool(&hidden, PoolingStrategy::Mean).is_err());
    }

    #[test]
    fn test_pooling_strategy_from_str() {
        assert_eq!(
            "last".parse::<PoolingStrategy>().unwrap(),
            PoolingStrategy::Last
        );
        assert_eq!(
            "Last".parse::<PoolingStrategy>().unwrap(),
            PoolingStrategy::Last
        );
        assert_eq!(
            "cls".parse::<PoolingStrategy>().unwrap(),
            PoolingStrategy::Cls
        );
        assert_eq!(
            "CLS".parse::<PoolingStrategy>().unwrap(),
            PoolingStrategy::Cls
        );
        assert_eq!(
            "mean".parse::<PoolingStrategy>().unwrap(),
            PoolingStrategy::Mean
        );
        assert_eq!(
            "MEAN".parse::<PoolingStrategy>().unwrap(),
            PoolingStrategy::Mean
        );
        assert!("unknown".parse::<PoolingStrategy>().is_err());
    }

    #[test]
    fn test_detect_pooling_strategy_mean() {
        let dir = tempfile::tempdir().unwrap();
        let pool_dir = dir.path().join("1_Pooling");
        std::fs::create_dir_all(&pool_dir).unwrap();
        std::fs::write(
            pool_dir.join("config.json"),
            r#"{"pooling_mode_mean_tokens": true, "pooling_mode_cls_token": false}"#,
        )
        .unwrap();
        assert_eq!(
            detect_pooling_strategy(dir.path()),
            Some(PoolingStrategy::Mean)
        );
    }

    #[test]
    fn test_detect_pooling_strategy_cls() {
        let dir = tempfile::tempdir().unwrap();
        let pool_dir = dir.path().join("1_Pooling");
        std::fs::create_dir_all(&pool_dir).unwrap();
        std::fs::write(
            pool_dir.join("config.json"),
            r#"{"pooling_mode_mean_tokens": false, "pooling_mode_cls_token": true}"#,
        )
        .unwrap();
        assert_eq!(
            detect_pooling_strategy(dir.path()),
            Some(PoolingStrategy::Cls)
        );
    }

    #[test]
    fn test_detect_pooling_strategy_last() {
        let dir = tempfile::tempdir().unwrap();
        let pool_dir = dir.path().join("1_Pooling");
        std::fs::create_dir_all(&pool_dir).unwrap();
        std::fs::write(
            pool_dir.join("config.json"),
            r#"{"pooling_mode_lasttoken": true}"#,
        )
        .unwrap();
        assert_eq!(
            detect_pooling_strategy(dir.path()),
            Some(PoolingStrategy::Last)
        );
    }

    #[test]
    fn test_detect_pooling_strategy_missing() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(detect_pooling_strategy(dir.path()), None);
    }

    #[test]
    fn test_detect_pooling_strategy_all_false() {
        let dir = tempfile::tempdir().unwrap();
        let pool_dir = dir.path().join("1_Pooling");
        std::fs::create_dir_all(&pool_dir).unwrap();
        std::fs::write(
            pool_dir.join("config.json"),
            r#"{"pooling_mode_mean_tokens": false, "pooling_mode_cls_token": false}"#,
        )
        .unwrap();
        assert_eq!(detect_pooling_strategy(dir.path()), None);
    }
}
