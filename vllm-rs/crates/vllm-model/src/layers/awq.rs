// SPDX-License-Identifier: Apache-2.0
//! AWQ (Activation-aware Weight Quantization) linear layer.
//!
//! Implements INT4 AWQ dequantization and matmul for weights stored in the
//! standard HuggingFace AWQ format:
//! - `*.qweight` — packed INT4-in-INT32, shape `[in_features, out_features/pack_factor]`
//! - `*.qzeros` — packed INT4-in-INT32, shape `[num_groups, out_features/pack_factor]`
//! - `*.scales` — f16, shape `[num_groups, out_features]`
//!
//! AWQ packs along columns (dimension 1) with an interleave order:
//! stored order in i32: `[col0, col2, col4, col6, col1, col3, col5, col7]`
//! To recover logical order, apply reverse-interleave `[0,4,1,5,2,6,3,7]`.

use candle_core::{Device, Module, Tensor};

use crate::error::{ModelError, ModelResult};
use crate::layers::gptq::read_i32_data;
use crate::weight::ModelWeights;

// ---------------------------------------------------------------------------
// AwqConfig
// ---------------------------------------------------------------------------

/// AWQ quantization parameters parsed from `quant_config.json`.
#[derive(Debug, Clone)]
pub struct AwqConfig {
    pub bits: usize,
    pub group_size: usize,
    pub zero_point: bool,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            zero_point: true,
        }
    }
}

// ---------------------------------------------------------------------------
// AWQ column unpacking with reverse-interleave
// ---------------------------------------------------------------------------

/// AWQ reverse-interleave mapping for INT4 (pack_factor=8).
///
/// AWQ stores 8 INT4 nibbles in one i32 with interleave order:
///   bit positions [0:3, 4:7, 8:11, 12:15, 16:19, 20:23, 24:27, 28:31]
///   correspond to logical columns [0, 2, 4, 6, 1, 3, 5, 7].
///
/// The reverse map takes extracted nibble index → logical column offset:
///   nibble 0 → col 0, nibble 1 → col 4, nibble 2 → col 1, nibble 3 → col 5,
///   nibble 4 → col 2, nibble 5 → col 6, nibble 6 → col 3, nibble 7 → col 7
const AWQ_REVERSE_INTERLEAVE: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

/// Unpack a 2D tensor packed along columns with AWQ interleave order.
///
/// Input: `[rows, packed_cols]` where each i32 packs `pack_factor` column values.
/// Output: `[rows, out_cols]` of f32 with correct logical column order.
///
/// Used for both `qweight` `[in_features, out_features/pack_factor]` and
/// `qzeros` `[num_groups, out_features/pack_factor]`.
#[allow(clippy::needless_range_loop)]
fn unpack_cols_awq(packed: &Tensor, bits: usize, out_cols: usize) -> ModelResult<Tensor> {
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
                // AWQ reverse-interleave: nibble j maps to logical offset.
                let logical_offset = if pack_factor == 8 {
                    AWQ_REVERSE_INTERLEAVE[j]
                } else {
                    j // Fallback for non-INT4
                };
                let out_col = packed_col * pack_factor + logical_offset;
                if out_col >= actual_cols {
                    continue;
                }
                let val = (packed_val >> (j * bits)) & mask;
                unpacked[row * actual_cols + out_col] = val as f32;
            }
        }
    }

    Tensor::from_vec(unpacked, (rows, actual_cols), device).map_err(ModelError::Candle)
}

// ---------------------------------------------------------------------------
// AwqLinear
// ---------------------------------------------------------------------------

/// A linear layer backed by AWQ-quantized weights.
///
/// On each forward pass the packed INT4 weights are dequantized to the
/// working dtype, multiplied with the input, and optionally bias is added.
///
/// On CPU, dequantization uses scalar unpacking. On CUDA, callers should
/// use `ops::awq_forward()` which dispatches to a GPU dequantize kernel.
pub struct AwqLinear {
    qweight: Tensor, // [in_features, out_features/pack_factor] i32
    qzeros: Tensor,  // [num_groups, out_features/pack_factor] i32
    scales: Tensor,  // [num_groups, out_features] f16
    bias: Option<Tensor>,
    bits: usize,
    in_features: usize,
    out_features: usize,
}

impl AwqLinear {
    /// Load an AWQ linear layer from model weights.
    ///
    /// Looks for `{prefix}.qweight`, `{prefix}.qzeros`, `{prefix}.scales`,
    /// and optionally `{prefix}.bias`.
    pub fn from_weights(
        weights: &mut ModelWeights,
        prefix: &str,
        config: &AwqConfig,
        _device: &Device,
    ) -> ModelResult<Self> {
        let bias = weights.take(&format!("{prefix}.bias")).ok();
        let qweight = weights.take(&format!("{prefix}.qweight"))?;
        let qzeros = weights.take(&format!("{prefix}.qzeros"))?;
        let scales = weights.take(&format!("{prefix}.scales"))?;

        // AWQ packs along columns: qweight is [in_features, out_features/pack_factor].
        let pack_factor = 32 / config.bits;
        let qw_shape = qweight.dims2().map_err(ModelError::Candle)?;
        let in_features = qw_shape.0;
        let out_features = qw_shape.1 * pack_factor;

        Ok(Self {
            qweight,
            qzeros,
            scales,
            bias,
            bits: config.bits,
            in_features,
            out_features,
        })
    }

    /// Dequantize packed weights to a full float weight matrix (CPU path).
    ///
    /// Returns shape `[in_features, out_features]` in the scales dtype.
    pub fn dequantize(&self) -> ModelResult<Tensor> {
        let device = self.qweight.device();
        let scales_dtype = self.scales.dtype();

        // Unpack qweight: [in, out/pack] i32 -> [in, out] f32
        // AWQ packs along columns (dim 1) with interleave.
        let unpacked_weight = unpack_cols_awq(&self.qweight, self.bits, self.out_features)?;
        let unpacked_weight = unpacked_weight
            .to_dtype(scales_dtype)
            .map_err(ModelError::Candle)?;

        // Unpack qzeros: [num_groups, out/pack] -> [num_groups, out_features] f32
        // qzeros also use AWQ interleave order.
        let qz_2d = self.qzeros.dims2().map_err(ModelError::Candle)?;
        let num_groups = qz_2d.0;
        let zeros = unpack_cols_awq(&self.qzeros, self.bits, self.out_features)?;
        let zeros = zeros.to_dtype(scales_dtype).map_err(ModelError::Candle)?;

        // Build group index and gather scales/zeros per row.
        let group_size = if num_groups > 0 {
            self.in_features.div_ceil(num_groups)
        } else {
            self.in_features
        };

        // AWQ has no g_idx — always use sequential group mapping.
        let indices: Vec<u32> = (0..self.in_features)
            .map(|i| (i / group_size) as u32)
            .collect();
        let idx_tensor = Tensor::new(indices.as_slice(), device).map_err(ModelError::Candle)?;
        let scales_per_row = self
            .scales
            .index_select(&idx_tensor, 0)
            .map_err(ModelError::Candle)?;
        let zeros_per_row = zeros
            .index_select(&idx_tensor, 0)
            .map_err(ModelError::Candle)?;

        // Dequantize: weight = scales * (unpacked - zeros)
        let dequantized = unpacked_weight
            .sub(&zeros_per_row)
            .map_err(ModelError::Candle)?
            .mul(&scales_per_row)
            .map_err(ModelError::Candle)?;

        Ok(dequantized)
    }

    /// Access qweight tensor (for CUDA dequant dispatch).
    pub fn qweight(&self) -> &Tensor {
        &self.qweight
    }

    /// Access qzeros tensor (for CUDA dequant dispatch).
    pub fn qzeros(&self) -> &Tensor {
        &self.qzeros
    }

    /// Access scales tensor (for CUDA dequant dispatch).
    pub fn scales(&self) -> &Tensor {
        &self.scales
    }

    /// Access bias tensor.
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
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

impl Module for AwqLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let w = self
            .dequantize()
            .map_err(|e| candle_core::Error::Msg(format!("AWQ dequantize: {e}")))?;

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
    use candle_core::DType;

    /// Helper: pack 8 INT4 values into one i32 using AWQ interleave order.
    ///
    /// AWQ packing: nibble j stores value from logical column ORDER[j],
    /// where ORDER = [0, 2, 4, 6, 1, 3, 5, 7].
    fn pack_awq_i32(vals: &[u32; 8]) -> i32 {
        let order_map: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
        let mut packed: u32 = 0;
        for (j, &src_col) in order_map.iter().enumerate() {
            packed |= (vals[src_col] & 0xF) << (j * 4);
        }
        packed as i32
    }

    /// Test AWQ INT4 unpacking with interleave order.
    #[test]
    fn test_awq_unpack_int4() {
        let device = Device::Cpu;

        // Pack values 0..7 in AWQ interleave order.
        let packed = pack_awq_i32(&[0, 1, 2, 3, 4, 5, 6, 7]);

        // qweight is [in_features, out_features/8] for AWQ
        // 1 row, 1 packed col → 1 row, 8 cols
        let qweight = Tensor::new(&[[packed]], &device).unwrap();
        let qzeros = Tensor::new(&[[pack_awq_i32(&[0; 8])]], &device).unwrap();
        let scales = Tensor::new(&[[1.0f32; 8]], &device).unwrap();

        let linear = AwqLinear {
            qweight,
            qzeros,
            scales,
            bias: None,
            bits: 4,
            in_features: 1,
            out_features: 8,
        };

        let w = linear.dequantize().unwrap();
        assert_eq!(w.dims(), &[1, 8]);
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
    fn test_awq_dequantize_with_scales() {
        let device = Device::Cpu;

        // All values = 8, zeros = 8, scales = 2.0 → dequantized = 2*(8-8) = 0
        let packed = pack_awq_i32(&[8, 8, 8, 8, 8, 8, 8, 8]);
        let zero_packed = pack_awq_i32(&[8, 8, 8, 8, 8, 8, 8, 8]);

        let qweight = Tensor::new(&[[packed]], &device).unwrap();
        let qzeros = Tensor::new(&[[zero_packed]], &device).unwrap();
        let scales = Tensor::new(&[[2.0f32; 8]], &device).unwrap();

        let linear = AwqLinear {
            qweight,
            qzeros,
            scales,
            bias: None,
            bits: 4,
            in_features: 1,
            out_features: 8,
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
    fn test_awq_linear_forward_shape() {
        let device = Device::Cpu;
        // 8 input features, 16 output features → qweight [8, 2] (2 packed cols of 8 each)
        let qweight = Tensor::zeros((8, 2), DType::I64, &device).unwrap();
        let qzeros = Tensor::zeros((1, 2), DType::I64, &device).unwrap();
        let scales = Tensor::ones((1, 16), DType::F32, &device).unwrap();

        let linear = AwqLinear {
            qweight,
            qzeros,
            scales,
            bias: None,
            bits: 4,
            in_features: 8,
            out_features: 16,
        };

        let x = Tensor::ones((3, 8), DType::F32, &device).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, 16]);
    }

    /// Test that dequantized weight produces correct matmul vs known float.
    #[test]
    fn test_awq_linear_matches_float() {
        let device = Device::Cpu;
        // 1 input row, 8 output cols. Pack values 1..8 in AWQ order.
        let packed = pack_awq_i32(&[1, 2, 3, 4, 5, 6, 7, 8]);

        let qweight = Tensor::new(&[[packed]], &device).unwrap();
        let qzeros = Tensor::new(&[[pack_awq_i32(&[0; 8])]], &device).unwrap();
        let scales = Tensor::new(&[[1.0f32; 8]], &device).unwrap();

        let linear = AwqLinear {
            qweight,
            qzeros,
            scales,
            bias: None,
            bits: 4,
            in_features: 1,
            out_features: 8,
        };

        // x = [1.0] → output = x @ w = w itself = [1,2,3,4,5,6,7,8]
        let x = Tensor::ones((1, 1), DType::F32, &device).unwrap();
        let y = linear.forward(&x).unwrap();
        let vals: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        let expected_sum: f32 = (1..=8).map(|i| i as f32).sum();
        let actual_sum: f32 = vals.iter().sum();
        assert!(
            (actual_sum - expected_sum).abs() < 0.1,
            "expected sum {expected_sum}, got {actual_sum}"
        );
    }
}
