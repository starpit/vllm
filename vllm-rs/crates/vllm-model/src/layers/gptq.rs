// SPDX-License-Identifier: Apache-2.0
//! GPTQ quantized linear layer.
//!
//! Implements INT4 GPTQ dequantization and matmul for weights stored in the
//! standard HuggingFace GPTQ format:
//! - `*.qweight` — packed INT4-in-INT32, shape `[in_features/pack_factor, out_features]`
//! - `*.qzeros` — packed INT4-in-INT32, shape `[num_groups, out_features/pack_factor]`
//! - `*.scales` — f16, shape `[num_groups, out_features]`
//! - `*.g_idx` — i32, shape `[in_features]` (optional, for desc_act models)

use candle_core::{DType, Device, Module, Tensor};

use crate::error::{ModelError, ModelResult};
use crate::weight::ModelWeights;

// ---------------------------------------------------------------------------
// GptqConfig
// ---------------------------------------------------------------------------

/// GPTQ quantization parameters parsed from `quantize_config.json`.
#[derive(Debug, Clone)]
pub struct GptqConfig {
    pub bits: usize,
    pub group_size: usize,
    pub desc_act: bool,
    pub sym: bool,
}

impl Default for GptqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            desc_act: false,
            sym: true,
        }
    }
}

// ---------------------------------------------------------------------------
// CPU-side INT4 unpacking helpers
// ---------------------------------------------------------------------------

/// Unpack a 2D tensor packed along rows (dimension 0).
///
/// Input: `[packed_rows, cols]` where each i32 packs `pack_factor` row values.
/// Output: `[unpacked_rows, cols]` of f32.
/// Used for `qweight` which has shape `[in_features/pack_factor, out_features]`.
fn unpack_rows(packed: &Tensor, bits: usize, out_rows: usize) -> ModelResult<Tensor> {
    let device = packed.device();
    let pack_factor = 32 / bits;
    let mask = (1u32 << bits) - 1;
    let dims = packed.dims2().map_err(ModelError::Candle)?;
    let packed_rows = dims.0;
    let cols = dims.1;

    let packed_data = read_i32_data(packed)?;

    let total_rows = packed_rows * pack_factor;
    let actual_rows = total_rows.min(out_rows);
    let mut unpacked = vec![0.0f32; actual_rows * cols];

    for packed_row in 0..packed_rows {
        for col in 0..cols {
            let packed_val = packed_data[packed_row * cols + col] as u32;
            for j in 0..pack_factor {
                let out_row = packed_row * pack_factor + j;
                if out_row >= actual_rows {
                    break;
                }
                let val = (packed_val >> (j * bits)) & mask;
                unpacked[out_row * cols + col] = val as f32;
            }
        }
    }

    Tensor::from_vec(unpacked, (actual_rows, cols), device).map_err(ModelError::Candle)
}

/// Unpack a 2D tensor packed along columns (dimension 1).
///
/// Input: `[rows, packed_cols]` where each i32 packs `pack_factor` column values.
/// Output: `[rows, unpacked_cols]` of f32.
/// Used for `qzeros` which has shape `[num_groups, out_features/pack_factor]`.
fn unpack_cols(packed: &Tensor, bits: usize, out_cols: usize) -> ModelResult<Tensor> {
    let device = packed.device();
    let pack_factor = 32 / bits;
    let mask = (1u32 << bits) - 1;
    let dims = packed.dims2().map_err(ModelError::Candle)?;
    let rows = dims.0;
    let packed_cols = dims.1;

    let packed_data = read_i32_data(packed)?;

    let total_cols = packed_cols * pack_factor;
    let actual_cols = total_cols.min(out_cols);
    let mut unpacked = vec![0.0f32; rows * actual_cols];

    for row in 0..rows {
        for packed_col in 0..packed_cols {
            let packed_val = packed_data[row * packed_cols + packed_col] as u32;
            for j in 0..pack_factor {
                let out_col = packed_col * pack_factor + j;
                if out_col >= actual_cols {
                    break;
                }
                let val = (packed_val >> (j * bits)) & mask;
                unpacked[row * actual_cols + out_col] = val as f32;
            }
        }
    }

    Tensor::from_vec(unpacked, (rows, actual_cols), device).map_err(ModelError::Candle)
}

/// Read a tensor as a Vec<i32> on CPU.
fn read_i32_data(t: &Tensor) -> ModelResult<Vec<i32>> {
    Ok(t.to_dtype(DType::I64)
        .map_err(ModelError::Candle)?
        .flatten_all()
        .map_err(ModelError::Candle)?
        .to_vec1::<i64>()
        .map_err(ModelError::Candle)?
        .into_iter()
        .map(|v| v as i32)
        .collect())
}

// ---------------------------------------------------------------------------
// GptqLinear
// ---------------------------------------------------------------------------

/// A linear layer backed by GPTQ-quantized weights.
///
/// On each forward pass the packed INT4 weights are dequantized to the
/// working dtype, multiplied with the input, and optionally bias is added.
/// This is a CPU-friendly approach (no custom CUDA kernels).
pub struct GptqLinear {
    qweight: Tensor,       // [in_features/pack_factor, out_features] i32
    qzeros: Tensor,        // [num_groups, out_features/pack_factor] i32
    scales: Tensor,        // [num_groups, out_features] f16
    g_idx: Option<Tensor>, // [in_features] i32
    bias: Option<Tensor>,
    bits: usize,
    in_features: usize,
    out_features: usize,
}

impl GptqLinear {
    /// Load a GPTQ linear layer from model weights.
    ///
    /// Looks for `{prefix}.qweight`, `{prefix}.qzeros`, `{prefix}.scales`,
    /// and optionally `{prefix}.g_idx` and `{prefix}.bias`.
    pub fn from_weights(
        weights: &ModelWeights,
        prefix: &str,
        config: &GptqConfig,
        _device: &Device,
    ) -> ModelResult<Self> {
        let qweight = weights.get(&format!("{prefix}.qweight"))?.clone();
        let qzeros = weights.get(&format!("{prefix}.qzeros"))?.clone();
        let scales = weights.get(&format!("{prefix}.scales"))?.clone();

        let g_idx = if config.desc_act {
            Some(weights.get(&format!("{prefix}.g_idx"))?.clone())
        } else {
            // Try to load g_idx even for non-desc_act (some models include it).
            weights.get(&format!("{prefix}.g_idx")).ok().cloned()
        };

        let bias = weights.get(&format!("{prefix}.bias")).ok().cloned();

        // Derive dimensions from tensor shapes.
        let pack_factor = 32 / config.bits;
        let qw_shape = qweight.dims2().map_err(ModelError::Candle)?;
        let in_features = qw_shape.0 * pack_factor;
        let out_features = qw_shape.1;

        Ok(Self {
            qweight,
            qzeros,
            scales,
            g_idx,
            bias,
            bits: config.bits,
            in_features,
            out_features,
        })
    }

    /// Dequantize packed weights to a full float weight matrix.
    ///
    /// Returns shape `[in_features, out_features]` in the scales dtype.
    pub fn dequantize(&self) -> ModelResult<Tensor> {
        let device = self.qweight.device();
        let scales_dtype = self.scales.dtype();

        // Unpack qweight: [in/pack, out] i32 -> [in, out] f32
        // qweight is packed along rows (dim 0).
        let unpacked_weight = unpack_rows(&self.qweight, self.bits, self.in_features)?;
        let unpacked_weight = unpacked_weight
            .to_dtype(scales_dtype)
            .map_err(ModelError::Candle)?;

        // Unpack qzeros: [num_groups, out/pack] -> [num_groups, out_features] f32
        // qzeros is packed along columns (dim 1).
        let qz_2d = self.qzeros.dims2().map_err(ModelError::Candle)?;
        let num_groups = qz_2d.0;
        let zeros = unpack_cols(&self.qzeros, self.bits, self.out_features)?;
        let zeros = zeros.to_dtype(scales_dtype).map_err(ModelError::Candle)?;

        // Build group index and gather scales/zeros per row.
        let group_size = if num_groups > 0 {
            self.in_features.div_ceil(num_groups)
        } else {
            self.in_features
        };

        let (scales_per_row, zeros_per_row) = if let Some(ref g_idx) = self.g_idx {
            let g_idx_u32 = g_idx.to_dtype(DType::U32).map_err(ModelError::Candle)?;
            let sp = self
                .scales
                .index_select(&g_idx_u32, 0)
                .map_err(ModelError::Candle)?;
            let zp = zeros
                .index_select(&g_idx_u32, 0)
                .map_err(ModelError::Candle)?;
            (sp, zp)
        } else {
            let indices: Vec<u32> = (0..self.in_features)
                .map(|i| (i / group_size) as u32)
                .collect();
            let idx_tensor = Tensor::new(indices.as_slice(), device).map_err(ModelError::Candle)?;
            let sp = self
                .scales
                .index_select(&idx_tensor, 0)
                .map_err(ModelError::Candle)?;
            let zp = zeros
                .index_select(&idx_tensor, 0)
                .map_err(ModelError::Candle)?;
            (sp, zp)
        };

        // Dequantize: weight = scales * (unpacked - zeros)
        let dequantized = unpacked_weight
            .sub(&zeros_per_row)
            .map_err(ModelError::Candle)?
            .mul(&scales_per_row)
            .map_err(ModelError::Candle)?;

        Ok(dequantized)
    }

    /// Input features (unquantized).
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output features.
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for GptqLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let w = self
            .dequantize()
            .map_err(|e| candle_core::Error::Msg(format!("GPTQ dequantize: {e}")))?;

        // x: [..., in_features], w: [in_features, out_features]
        // output: [..., out_features] = x @ w
        let x_dtype = x.dtype();
        let x = if x.dtype() != w.dtype() {
            x.to_dtype(w.dtype())?
        } else {
            x.clone()
        };

        let output = x.contiguous()?.matmul(&w.contiguous()?)?;
        let output = if output.dtype() != x_dtype {
            output.to_dtype(x_dtype)?
        } else {
            output
        };

        match &self.bias {
            Some(b) => output.broadcast_add(b),
            None => Ok(output),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Test INT4 unpacking from packed i32 values.
    #[test]
    fn test_gptq_unpack_int4() {
        let device = Device::Cpu;
        // Pack 8 INT4 values (0..7) into one i32.
        let packed: i32 =
            0 | (1 << 4) | (2 << 8) | (3 << 12) | (4 << 16) | (5 << 20) | (6 << 24) | (7 << 28);

        let qweight = Tensor::new(&[[packed]], &device).unwrap();
        let qzeros = Tensor::new(&[[0i32]], &device).unwrap();
        let scales = Tensor::new(&[[1.0f32]], &device).unwrap();

        let linear = GptqLinear {
            qweight,
            qzeros,
            scales,
            g_idx: None,
            bias: None,
            bits: 4,
            in_features: 8,
            out_features: 1,
        };

        let w = linear.dequantize().unwrap();
        assert_eq!(w.dims(), &[8, 1]);
        let vals: Vec<f32> = w
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();

        for i in 0..8 {
            assert!(
                (vals[i] - i as f32).abs() < 0.01,
                "expected {}, got {} at position {}",
                i,
                vals[i],
                i
            );
        }
    }

    /// Test dequantization with known scales and zeros.
    #[test]
    fn test_gptq_dequantize_with_scales() {
        let device = Device::Cpu;
        let val: i32 = 8;
        let packed: i32 = (0..8).fold(0i32, |acc, j: i32| acc | (val << (j * 4)));

        let qweight = Tensor::new(&[[packed]], &device).unwrap();
        let zero_packed: i32 = (0..8).fold(0i32, |acc, j: i32| acc | (8i32 << (j * 4)));
        let qzeros = Tensor::new(&[[zero_packed]], &device).unwrap();
        let scales = Tensor::new(&[[2.0f32]], &device).unwrap();

        let linear = GptqLinear {
            qweight,
            qzeros,
            scales,
            g_idx: None,
            bias: None,
            bits: 4,
            in_features: 8,
            out_features: 1,
        };

        let w = linear.dequantize().unwrap();
        let vals: Vec<f32> = w
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();

        for v in &vals {
            assert!(v.abs() < 0.01, "expected 0.0, got {v}");
        }
    }

    /// Test forward pass shape.
    #[test]
    fn test_gptq_linear_forward_shape() {
        let device = Device::Cpu;
        let qweight = Tensor::zeros((1, 2), DType::I64, &device).unwrap();
        let qzeros = Tensor::zeros((1, 1), DType::I64, &device).unwrap();
        let scales = Tensor::ones((1, 2), DType::F32, &device).unwrap();

        let linear = GptqLinear {
            qweight,
            qzeros,
            scales,
            g_idx: None,
            bias: None,
            bits: 4,
            in_features: 8,
            out_features: 2,
        };

        let x = Tensor::ones((3, 8), DType::F32, &device).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, 2]);
    }

    /// Test that dequantized weight produces correct matmul vs known float.
    #[test]
    fn test_gptq_linear_matches_float() {
        let device = Device::Cpu;
        let packed: i32 = (0..8).fold(0i32, |acc, j| acc | (((j + 1) as i32) << (j * 4)));

        let qweight = Tensor::new(&[[packed]], &device).unwrap();
        let qzeros = Tensor::new(&[[0i32]], &device).unwrap();
        let scales = Tensor::new(&[[1.0f32]], &device).unwrap();

        let linear = GptqLinear {
            qweight,
            qzeros,
            scales,
            g_idx: None,
            bias: None,
            bits: 4,
            in_features: 8,
            out_features: 1,
        };

        let x = Tensor::ones((1, 8), DType::F32, &device).unwrap();
        let y = linear.forward(&x).unwrap();
        let val: f32 = y.flatten_all().unwrap().to_vec1().unwrap()[0];
        assert!((val - 36.0).abs() < 0.1, "expected 36.0, got {val}");
    }
}
