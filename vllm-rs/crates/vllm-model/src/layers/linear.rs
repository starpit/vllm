// SPDX-License-Identifier: Apache-2.0
//! Linear layer implementations.
//!
//! Port of: `vllm/model_executor/layers/linear.py`

use candle_core::{DType, Device, Module, Tensor};

use crate::error::ModelResult;
use crate::tensor;
use crate::weight::ModelWeights;

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

/// A standard dense linear layer: y = xW^T + b
///
/// Weights are stored **pre-transposed** as `[in_features, out_features]`
/// so that forward is a single `x.matmul(&weight)` with no per-call
/// transpose or contiguous copy. This is critical for Metal/Accelerate
/// performance.
///
/// Bias shape: [out_features] (optional)
pub struct Linear {
    /// Pre-transposed weight: [in_features, out_features]
    weight: Tensor,
    bias: Option<Tensor>,
    out_features: usize,
    in_features: usize,
}

impl Linear {
    /// Create a linear layer from a weight in standard `[out, in]` layout.
    /// The weight is transposed and made contiguous at construction time.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        let out_features = weight.dim(0).unwrap_or(0);
        let in_features = weight.dim(1).unwrap_or(0);
        // Pre-transpose + contiguous so forward is a single matmul.
        let weight_t = weight.t().unwrap().contiguous().unwrap();
        Self { weight: weight_t, bias, out_features, in_features }
    }

    /// Load a linear layer from model weights.
    ///
    /// Looks for `{prefix}.weight` and optionally `{prefix}.bias`.
    pub fn load(weights: &ModelWeights, prefix: &str, dtype: DType) -> ModelResult<Self> {
        let weight_name = format!("{}.weight", prefix);
        let bias_name = format!("{}.bias", prefix);

        let weight = weights.get_cast(&weight_name, dtype)?;
        let bias = if weights.contains(&bias_name) {
            Some(weights.get_cast(&bias_name, dtype)?)
        } else {
            None
        };

        Ok(Self::new(weight, bias))
    }

    /// Create a zero-initialized linear layer (for testing).
    pub fn zeros(in_features: usize, out_features: usize, dtype: DType, device: &Device) -> ModelResult<Self> {
        let weight = tensor::zeros(&[out_features, in_features], dtype, device)?;
        Ok(Self::new(weight, None))
    }

    /// Output dimension.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Input dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Access the weight tensor (pre-transposed: [in, out]).
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Access the bias tensor.
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for Linear {
    /// Forward pass: y = x @ W_t + b (weight already pre-transposed)
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = x.matmul(&self.weight)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

// ---------------------------------------------------------------------------
// ColumnParallelLinear
// ---------------------------------------------------------------------------

/// Column-parallel linear layer for tensor parallelism.
///
/// Splits the output dimension across TP ranks. Each rank holds
/// `out_features / world_size` columns of the weight matrix.
///
/// Port of: `vllm/model_executor/layers/linear.py::ColumnParallelLinear`
pub struct ColumnParallelLinear {
    inner: Linear,
    /// Whether to gather outputs across ranks after forward.
    gather_output: bool,
}

impl ColumnParallelLinear {
    /// Create from an already-sharded linear layer.
    pub fn new(linear: Linear, gather_output: bool) -> Self {
        Self {
            inner: linear,
            gather_output,
        }
    }

    /// Load from model weights, sharding the weight along dim 0.
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        dtype: DType,
        rank: usize,
        world_size: usize,
        gather_output: bool,
    ) -> ModelResult<Self> {
        let weight_name = format!("{}.weight", prefix);
        let bias_name = format!("{}.bias", prefix);

        let full_weight = weights.get_cast(&weight_name, dtype)?;
        let weight = tensor::shard_tensor(&full_weight, 0, rank, world_size)?;

        let bias = if weights.contains(&bias_name) {
            let full_bias = weights.get_cast(&bias_name, dtype)?;
            Some(tensor::shard_tensor(&full_bias, 0, rank, world_size)?)
        } else {
            None
        };

        Ok(Self {
            inner: Linear::new(weight, bias),
            gather_output,
        })
    }

    /// Whether outputs should be gathered across ranks.
    pub fn gather_output(&self) -> bool {
        self.gather_output
    }

    /// Access the inner linear layer.
    pub fn inner(&self) -> &Linear {
        &self.inner
    }
}

impl Module for ColumnParallelLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Note: actual all-gather for gather_output=true requires
        // distributed communication (Phase 5c). For now, just do the
        // local matmul.
        self.inner.forward(x)
    }
}

// ---------------------------------------------------------------------------
// RowParallelLinear
// ---------------------------------------------------------------------------

/// Row-parallel linear layer for tensor parallelism.
///
/// Splits the input dimension across TP ranks. Each rank holds
/// `in_features / world_size` rows of the weight matrix.
///
/// Port of: `vllm/model_executor/layers/linear.py::RowParallelLinear`
pub struct RowParallelLinear {
    inner: Linear,
    /// Whether the input is already sharded across ranks.
    #[allow(dead_code)]
    input_is_parallel: bool,
}

impl RowParallelLinear {
    /// Create from an already-sharded linear layer.
    pub fn new(linear: Linear, input_is_parallel: bool) -> Self {
        Self {
            inner: linear,
            input_is_parallel,
        }
    }

    /// Load from model weights, sharding the weight along dim 1.
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        dtype: DType,
        rank: usize,
        world_size: usize,
        input_is_parallel: bool,
    ) -> ModelResult<Self> {
        let weight_name = format!("{}.weight", prefix);
        let bias_name = format!("{}.bias", prefix);

        let full_weight = weights.get_cast(&weight_name, dtype)?;
        let weight = tensor::shard_tensor(&full_weight, 1, rank, world_size)?;

        // Bias is NOT sharded for row-parallel (added after all-reduce).
        let bias = if weights.contains(&bias_name) {
            Some(weights.get_cast(&bias_name, dtype)?)
        } else {
            None
        };

        Ok(Self {
            inner: Linear::new(weight, bias),
            input_is_parallel,
        })
    }

    /// Access the inner linear layer.
    pub fn inner(&self) -> &Linear {
        &self.inner
    }
}

impl Module for RowParallelLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Note: actual all-reduce requires distributed communication.
        // For now, just do the local matmul + bias.
        self.inner.forward(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        // Create a 2x3 weight matrix [[1,0,0],[0,1,0]]
        let weight = Tensor::new(&[[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]], &Device::Cpu).unwrap();
        let linear = Linear::new(weight, None);

        assert_eq!(linear.out_features(), 2);
        assert_eq!(linear.in_features(), 3);

        // Input: [1, 2, 3] -> output should be [1, 2]
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0]], &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 2]);
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_with_bias() {
        let weight = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu).unwrap();
        let bias = Tensor::new(&[10.0f32, 20.0], &Device::Cpu).unwrap();
        let linear = Linear::new(weight, Some(bias));

        let x = Tensor::new(&[[3.0f32, 4.0]], &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[0] - 13.0).abs() < 1e-6); // 3 + 10
        assert!((vals[1] - 24.0).abs() < 1e-6); // 4 + 20
    }

    #[test]
    fn test_linear_batch() {
        let weight = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu).unwrap();
        let linear = Linear::new(weight, None);

        // Batch of 3
        let x = Tensor::new(
            &[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]],
            &Device::Cpu,
        )
        .unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, 2]);
    }

    #[test]
    fn test_linear_zeros() {
        let linear = Linear::zeros(4, 3, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(linear.out_features(), 3);
        assert_eq!(linear.in_features(), 4);

        let x = Tensor::ones(&[2, 4], DType::F32, &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(vals.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn test_linear_load_from_weights() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        // Create a weight [2, 3] and bias [2]
        let w_data: Vec<u8> = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let b_data: Vec<u8> = [0.5f32, 0.5]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        crate::weight::tests_helper::create_safetensors_file(
            &path,
            &[
                ("layer.weight", vec![2, 3], DType::F32, &w_data),
                ("layer.bias", vec![2], DType::F32, &b_data),
            ],
        );

        let weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();
        let linear = Linear::load(&weights, "layer", DType::F32).unwrap();
        assert_eq!(linear.out_features(), 2);
        assert_eq!(linear.in_features(), 3);
        assert!(linear.bias().is_some());
    }

    #[test]
    fn test_column_parallel_linear() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        // Weight [4, 2] — will be sharded along dim 0 into [2, 2] per rank.
        let w_data: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        crate::weight::tests_helper::create_safetensors_file(
            &path,
            &[("proj.weight", vec![4, 2], DType::F32, &w_data)],
        );

        let weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();

        // Rank 0 gets first half.
        let cp0 = ColumnParallelLinear::load(&weights, "proj", DType::F32, 0, 2, false).unwrap();
        assert_eq!(cp0.inner().out_features(), 2);

        // Rank 1 gets second half.
        let cp1 = ColumnParallelLinear::load(&weights, "proj", DType::F32, 1, 2, false).unwrap();
        assert_eq!(cp1.inner().out_features(), 2);
    }

    #[test]
    fn test_row_parallel_linear() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        // Weight [2, 4] — will be sharded along dim 1 into [2, 2] per rank.
        let w_data: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        crate::weight::tests_helper::create_safetensors_file(
            &path,
            &[("proj.weight", vec![2, 4], DType::F32, &w_data)],
        );

        let weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();

        let rp0 = RowParallelLinear::load(&weights, "proj", DType::F32, 0, 2, true).unwrap();
        assert_eq!(rp0.inner().in_features(), 2);
    }
}
