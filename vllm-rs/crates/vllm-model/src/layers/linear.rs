// SPDX-License-Identifier: Apache-2.0
//! Linear layer implementations.
//!
//! Port of: `vllm/model_executor/layers/linear.py`

use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor};

use crate::error::{ModelError, ModelResult};
use crate::process_group::ProcessGroup;
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
///
/// Supports optional LoRA (Low-Rank Adaptation) weights. When `lora_a`
/// and `lora_b` are set, forward computes `y = xW + x·A·B` where A and B
/// are low-rank matrices with scaling already absorbed into B.
pub struct Linear {
    /// Pre-transposed weight: [in_features, out_features]
    weight: Tensor,
    bias: Option<Tensor>,
    out_features: usize,
    in_features: usize,
    /// LoRA A matrix: [in_features, rank] (pre-transposed at attach time)
    lora_a: Option<Tensor>,
    /// LoRA B matrix: [rank, out_features] (pre-transposed, scaling absorbed)
    lora_b: Option<Tensor>,
}

impl Linear {
    /// Create a linear layer from a weight in standard `[out, in]` layout.
    /// The weight is transposed and made contiguous at construction time.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        let out_features = weight.dim(0).unwrap_or(0);
        let in_features = weight.dim(1).unwrap_or(0);
        // Pre-transpose + contiguous so forward is a single matmul.
        let weight_t = weight.t().unwrap().contiguous().unwrap();
        Self {
            weight: weight_t,
            bias,
            out_features,
            in_features,
            lora_a: None,
            lora_b: None,
        }
    }

    /// Load a linear layer from model weights.
    ///
    /// Looks for `{prefix}.weight` and optionally `{prefix}.bias`.
    /// Takes ownership of the tensors from the weights HashMap to reduce
    /// peak GPU memory (original freed before transposed copy allocated).
    pub fn load(weights: &mut ModelWeights, prefix: &str, dtype: DType) -> ModelResult<Self> {
        let weight_name = format!("{}.weight", prefix);
        let bias_name = format!("{}.bias", prefix);

        let bias = if weights.contains(&bias_name) {
            Some(weights.take_cast(&bias_name, dtype)?)
        } else {
            None
        };
        let weight = weights.take_cast(&weight_name, dtype)?;

        Ok(Self::new(weight, bias))
    }

    /// Create a zero-initialized linear layer (for testing).
    pub fn zeros(
        in_features: usize,
        out_features: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
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

    /// Attach LoRA adapter weights to this linear layer.
    ///
    /// * `lora_a` — shape `[rank, in_features]` (raw from safetensors)
    /// * `lora_b` — shape `[out_features, rank]` (raw from safetensors)
    /// * `scaling` — alpha / rank (or alpha / sqrt(rank) for rsLoRA)
    ///
    /// The weights are pre-transposed and scaling is absorbed into B so that
    /// forward is `y + x @ A_t @ (B_t * scaling)` with no per-call overhead.
    pub fn attach_lora(&mut self, lora_a: Tensor, lora_b: Tensor, scaling: f64) -> ModelResult<()> {
        // lora_a raw: [rank, in_features] → transpose to [in_features, rank]
        let a_t = lora_a.t()?.contiguous()?;
        // lora_b raw: [out_features, rank] → transpose to [rank, out_features]
        let b_t = lora_b.t()?.contiguous()?;
        // Absorb scaling into B.
        let b_scaled = (b_t * scaling)?;
        self.lora_a = Some(a_t);
        self.lora_b = Some(b_scaled);
        Ok(())
    }

    /// Remove LoRA adapter weights from this layer.
    pub fn detach_lora(&mut self) {
        self.lora_a = None;
        self.lora_b = None;
    }

    /// Whether this layer has a LoRA adapter attached.
    pub fn has_lora(&self) -> bool {
        self.lora_a.is_some() && self.lora_b.is_some()
    }
}

impl Module for Linear {
    /// Forward pass: y = x @ W_t + b + x @ A @ B (LoRA delta when attached)
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.contiguous()?;
        let y = x.matmul(&self.weight)?;
        let y = match &self.bias {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        };
        match (&self.lora_a, &self.lora_b) {
            (Some(a), Some(b)) => y.add(&x.matmul(a)?.matmul(b)?),
            _ => Ok(y),
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
    /// NCCL process group for all-gather (only used when gather_output=true and TP>1).
    tp_group: Option<Arc<dyn ProcessGroup>>,
}

impl ColumnParallelLinear {
    /// Create from an already-sharded linear layer.
    pub fn new(linear: Linear, gather_output: bool) -> Self {
        Self {
            inner: linear,
            gather_output,
            tp_group: None,
        }
    }

    /// Load from model weights, sharding the weight along dim 0.
    pub fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        dtype: DType,
        rank: usize,
        world_size: usize,
        gather_output: bool,
    ) -> ModelResult<Self> {
        let weight_name = format!("{}.weight", prefix);
        let bias_name = format!("{}.bias", prefix);

        let bias = if weights.contains(&bias_name) {
            let full_bias = weights.take_cast(&bias_name, dtype)?;
            Some(tensor::shard_tensor(&full_bias, 0, rank, world_size)?)
        } else {
            None
        };

        let full_weight = weights.take_cast(&weight_name, dtype)?;
        let weight = tensor::shard_tensor(&full_weight, 0, rank, world_size)?;

        Ok(Self {
            inner: Linear::new(weight, bias),
            gather_output,
            tp_group: None,
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

    /// Mutable access to the inner linear layer (for LoRA injection).
    pub fn inner_mut(&mut self) -> &mut Linear {
        &mut self.inner
    }

    /// Set the NCCL process group for tensor-parallel communication.
    pub fn set_tp_group(&mut self, group: Arc<dyn ProcessGroup>) {
        self.tp_group = Some(group);
    }
}

impl Module for ColumnParallelLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let output = self.inner.forward(x)?;
        // All-gather when gather_output=true and TP > 1 (used for lm_head).
        if self.gather_output
            && let Some(ref group) = self.tp_group
            && group.world_size() > 1
        {
            return group.all_gather(&output, 0);
        }
        Ok(output)
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
    /// NCCL process group for all-reduce (used when TP > 1).
    tp_group: Option<Arc<dyn ProcessGroup>>,
}

impl RowParallelLinear {
    /// Create from an already-sharded linear layer.
    pub fn new(linear: Linear, input_is_parallel: bool) -> Self {
        Self {
            inner: linear,
            input_is_parallel,
            tp_group: None,
        }
    }

    /// Load from model weights, sharding the weight along dim 1.
    pub fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        dtype: DType,
        rank: usize,
        world_size: usize,
        input_is_parallel: bool,
    ) -> ModelResult<Self> {
        let weight_name = format!("{}.weight", prefix);
        let bias_name = format!("{}.bias", prefix);

        // Bias is NOT sharded for row-parallel (added after all-reduce).
        let bias = if weights.contains(&bias_name) {
            Some(weights.take_cast(&bias_name, dtype)?)
        } else {
            None
        };

        let full_weight = weights.take_cast(&weight_name, dtype)?;
        let weight = tensor::shard_tensor(&full_weight, 1, rank, world_size)?;

        Ok(Self {
            inner: Linear::new(weight, bias),
            input_is_parallel,
            tp_group: None,
        })
    }

    /// Access the inner linear layer.
    pub fn inner(&self) -> &Linear {
        &self.inner
    }

    /// Mutable access to the inner linear layer (for LoRA injection).
    pub fn inner_mut(&mut self) -> &mut Linear {
        &mut self.inner
    }

    /// Set the NCCL process group for tensor-parallel communication.
    pub fn set_tp_group(&mut self, group: Arc<dyn ProcessGroup>) {
        self.tp_group = Some(group);
    }
}

impl Module for RowParallelLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Local matmul (no bias — bias is added after all-reduce).
        let x = x.contiguous()?;
        let output = x.matmul(self.inner.weight())?;

        // Apply LoRA delta if attached.
        let output = match (&self.inner.lora_a, &self.inner.lora_b) {
            (Some(a), Some(b)) => output.add(&x.matmul(a)?.matmul(b)?)?,
            _ => output,
        };

        // All-reduce across TP ranks.
        let output = if let Some(ref group) = self.tp_group {
            if group.world_size() > 1 {
                group.all_reduce(&output)?
            } else {
                output
            }
        } else {
            output
        };

        // Bias is added AFTER all-reduce (it's the full, unsharded bias).
        match self.inner.bias() {
            Some(b) => output.broadcast_add(b),
            None => Ok(output),
        }
    }
}

// ---------------------------------------------------------------------------
// Fused projection helpers
// ---------------------------------------------------------------------------

/// Load a fused QKV linear layer by concatenating separate Q, K, V weights.
///
/// Each component is TP-sharded independently along dim 0 (output dim) before
/// concatenation — critical for GQA where Q has `num_heads` and K/V have
/// `num_kv_heads`.
///
/// Handles optional bias (e.g. Qwen2 has QKV bias).
///
/// Returns `(linear, q_size, kv_size)` where sizes are the post-shard output
/// dimensions of Q and each of K/V respectively.
pub fn load_fused_qkv(
    weights: &mut ModelWeights,
    prefix: &str,
    dtype: DType,
    rank: usize,
    world_size: usize,
) -> ModelResult<(Linear, usize, usize)> {
    let q_name = format!("{prefix}.q_proj.weight");
    let k_name = format!("{prefix}.k_proj.weight");
    let v_name = format!("{prefix}.v_proj.weight");

    // Take biases first (before weights) to free them early.
    let q_bias_name = format!("{prefix}.q_proj.bias");
    let qkv_bias = if weights.contains(&q_bias_name) {
        let k_bias_name = format!("{prefix}.k_proj.bias");
        let v_bias_name = format!("{prefix}.v_proj.bias");
        let q_b = tensor::shard_tensor(
            &weights.take_cast(&q_bias_name, dtype)?,
            0,
            rank,
            world_size,
        )?;
        let k_b = tensor::shard_tensor(
            &weights.take_cast(&k_bias_name, dtype)?,
            0,
            rank,
            world_size,
        )?;
        let v_b = tensor::shard_tensor(
            &weights.take_cast(&v_bias_name, dtype)?,
            0,
            rank,
            world_size,
        )?;
        Some(Tensor::cat(&[&q_b, &k_b, &v_b], 0).map_err(ModelError::Candle)?)
    } else {
        None
    };

    let q_w = tensor::shard_tensor(&weights.take_cast(&q_name, dtype)?, 0, rank, world_size)?;
    let k_w = tensor::shard_tensor(&weights.take_cast(&k_name, dtype)?, 0, rank, world_size)?;
    let v_w = tensor::shard_tensor(&weights.take_cast(&v_name, dtype)?, 0, rank, world_size)?;

    let q_size = q_w.dim(0).map_err(ModelError::Candle)?;
    let kv_size = k_w.dim(0).map_err(ModelError::Candle)?;

    let qkv_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0).map_err(ModelError::Candle)?;

    Ok((Linear::new(qkv_w, qkv_bias), q_size, kv_size))
}

/// Load a fused gate+up linear layer by concatenating separate gate and up weights.
///
/// Each component is TP-sharded independently along dim 0 before concatenation.
///
/// `gate_name` / `up_name` are the weight name prefixes (e.g. `"layer.mlp.gate_proj"`
/// or `"layer.experts.0.w1"` for MoE).
///
/// Returns `(linear, half_size)` where `half_size` is the post-shard output
/// dimension of each component (used to split the fused output).
pub fn load_fused_gate_up(
    weights: &mut ModelWeights,
    gate_prefix: &str,
    up_prefix: &str,
    dtype: DType,
    rank: usize,
    world_size: usize,
) -> ModelResult<(Linear, usize)> {
    let gate_name = format!("{gate_prefix}.weight");
    let up_name = format!("{up_prefix}.weight");

    // Handle optional bias first.
    let gate_bias_name = format!("{gate_prefix}.bias");
    let fused_bias = if weights.contains(&gate_bias_name) {
        let up_bias_name = format!("{up_prefix}.bias");
        let g_b = tensor::shard_tensor(
            &weights.take_cast(&gate_bias_name, dtype)?,
            0,
            rank,
            world_size,
        )?;
        let u_b = tensor::shard_tensor(
            &weights.take_cast(&up_bias_name, dtype)?,
            0,
            rank,
            world_size,
        )?;
        Some(Tensor::cat(&[&g_b, &u_b], 0).map_err(ModelError::Candle)?)
    } else {
        None
    };

    let gate_w = tensor::shard_tensor(&weights.take_cast(&gate_name, dtype)?, 0, rank, world_size)?;
    let up_w = tensor::shard_tensor(&weights.take_cast(&up_name, dtype)?, 0, rank, world_size)?;

    let half_size = gate_w.dim(0).map_err(ModelError::Candle)?;

    let fused_w = Tensor::cat(&[&gate_w, &up_w], 0).map_err(ModelError::Candle)?;

    Ok((Linear::new(fused_w, fused_bias), half_size))
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
        let x = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &Device::Cpu).unwrap();
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
        let b_data: Vec<u8> = [0.5f32, 0.5].iter().flat_map(|f| f.to_le_bytes()).collect();

        crate::weight::tests_helper::create_safetensors_file(
            &path,
            &[
                ("layer.weight", vec![2, 3], DType::F32, &w_data),
                ("layer.bias", vec![2], DType::F32, &b_data),
            ],
        );

        let mut weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();
        let linear = Linear::load(&mut weights, "layer", DType::F32).unwrap();
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

        let mut weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();

        // Rank 0 gets first half.
        let cp0 =
            ColumnParallelLinear::load(&mut weights, "proj", DType::F32, 0, 2, false).unwrap();
        assert_eq!(cp0.inner().out_features(), 2);

        // Load fresh weights for rank 1 (take consumed rank 0's).
        let mut weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();
        let cp1 =
            ColumnParallelLinear::load(&mut weights, "proj", DType::F32, 1, 2, false).unwrap();
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

        let mut weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();

        let rp0 = RowParallelLinear::load(&mut weights, "proj", DType::F32, 0, 2, true).unwrap();
        assert_eq!(rp0.inner().in_features(), 2);
    }

    #[test]
    fn test_lora_forward() {
        // Identity weight: y = x
        let weight = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu).unwrap();
        let mut linear = Linear::new(weight, None);
        assert!(!linear.has_lora());

        // LoRA A: [rank=1, in=2], B: [out=2, rank=1]
        let lora_a = Tensor::new(&[[1.0f32, 0.0]], &Device::Cpu).unwrap(); // [1, 2]
        let lora_b = Tensor::new(&[[1.0f32], [0.0]], &Device::Cpu).unwrap(); // [2, 1]
        linear.attach_lora(lora_a, lora_b, 1.0).unwrap();
        assert!(linear.has_lora());

        // x = [1, 2] → base y = [1, 2], delta = x·A_t·B_t = [1]·[1, 0] = [1, 0]
        // total = [2, 2]
        let x = Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[0] - 2.0).abs() < 1e-5);
        assert!((vals[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_lora_scaling() {
        let weight = Tensor::zeros(&[2, 2], DType::F32, &Device::Cpu).unwrap();
        let mut linear = Linear::new(weight, None);

        let lora_a = Tensor::new(&[[1.0f32, 0.0]], &Device::Cpu).unwrap();
        let lora_b = Tensor::new(&[[1.0f32], [0.0]], &Device::Cpu).unwrap();
        // scaling = 0.5
        linear.attach_lora(lora_a, lora_b, 0.5).unwrap();

        let x = Tensor::new(&[[2.0f32, 0.0]], &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // delta = 2*1*0.5 = 1, base = 0 → [1, 0]
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!(vals[1].abs() < 1e-5);
    }

    #[test]
    fn test_lora_zero_cost() {
        // Without LoRA attached, forward should be identical to base.
        let weight = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu).unwrap();
        let linear = Linear::new(weight, None);
        assert!(!linear.has_lora());

        let x = Tensor::new(&[[3.0f32, 4.0]], &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[0] - 3.0).abs() < 1e-5);
        assert!((vals[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_lora_detach() {
        let weight = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu).unwrap();
        let mut linear = Linear::new(weight, None);

        let lora_a = Tensor::new(&[[1.0f32, 0.0]], &Device::Cpu).unwrap();
        let lora_b = Tensor::new(&[[1.0f32], [0.0]], &Device::Cpu).unwrap();
        linear.attach_lora(lora_a, lora_b, 1.0).unwrap();
        assert!(linear.has_lora());

        linear.detach_lora();
        assert!(!linear.has_lora());

        // After detach, output should be pure base again.
        let x = Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[1] - 2.0).abs() < 1e-5);
    }
}
