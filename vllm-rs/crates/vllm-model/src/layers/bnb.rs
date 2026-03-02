// SPDX-License-Identifier: Apache-2.0
//! BitsAndBytes quantized linear layers (NF4/FP4 4-bit and INT8 8-bit).
//!
//! NF4/FP4 (4-bit):
//! - `*.weight` — packed uint8, two nibbles per byte, flat `[num_elements/2]`
//! - `*.weight.absmax` — f32, per-block absmax scales `[num_blocks]`
//! - `*.weight.quant_state.bitsandbytes__nf4` — metadata (ignored, we parse shape ourselves)
//! - Dequantization: `weight[i] = absmax[i / blocksize] * TABLE[nibble(packed, i)]`
//!
//! INT8 (8-bit):
//! - `*.weight` — int8 stored as uint8 `[out_features, in_features]`
//! - `*.weight.SCB` — f32, per-row absmax scales `[out_features]`
//! - Dequantization: `weight[row][col] = int8_val * (SCB[row] / 127.0)`

use candle_core::{DType, Device, Module, Tensor};

use crate::error::{ModelError, ModelResult};
use crate::weight::ModelWeights;

// ---------------------------------------------------------------------------
// NF4 / FP4 lookup tables
// ---------------------------------------------------------------------------

/// NF4 (Normal Float 4) lookup table — 16 normal distribution quantiles.
#[allow(clippy::excessive_precision)]
const NF4_TABLE: [f32; 16] = [
    -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0.0, 0.0795803,
    0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1.0,
];

/// FP4 (Float Point 4) lookup table — values from bitsandbytes.
const FP4_TABLE: [f32; 16] = [
    0.0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0.0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0,
];

// ---------------------------------------------------------------------------
// BnbQuantType + BnbNf4Config
// ---------------------------------------------------------------------------

/// Quantization type for BitsAndBytes 4-bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BnbQuantType {
    NF4,
    FP4,
}

/// Layer-level BnB NF4 configuration.
#[derive(Debug, Clone)]
pub struct BnbNf4Config {
    pub quant_type: BnbQuantType,
    pub blocksize: usize,
    pub double_quant: bool,
}

impl Default for BnbNf4Config {
    fn default() -> Self {
        Self {
            quant_type: BnbQuantType::NF4,
            blocksize: 64,
            double_quant: false,
        }
    }
}

// ---------------------------------------------------------------------------
// BnbLayerConfig — unified config for NF4 or INT8 layers
// ---------------------------------------------------------------------------

/// Layer-level BnB configuration — either NF4 or INT8.
#[derive(Debug, Clone)]
pub enum BnbLayerConfig {
    Nf4(BnbNf4Config),
    Int8,
}

// ---------------------------------------------------------------------------
// BnbInt8Linear
// ---------------------------------------------------------------------------

/// A linear layer backed by BitsAndBytes INT8 quantized weights.
///
/// On each forward pass the int8 weights are dequantized to the working dtype
/// using per-row absmax scales, multiplied with the input, and optionally bias
/// is added.
pub struct BnbInt8Linear {
    weight_u8: Tensor, // uint8 (reinterpreted as int8), [out_features, in_features]
    absmax: Tensor,    // f32, per-row scales [out_features]
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
    original_dtype: DType,
}

impl BnbInt8Linear {
    /// Load a BnB INT8 linear layer from model weights.
    ///
    /// Looks for `{prefix}.weight` (int8 stored as uint8) and `{prefix}.SCB` (f32 absmax).
    pub fn from_weights(
        weights: &ModelWeights,
        prefix: &str,
        out_features: usize,
        in_features: usize,
        dtype: DType,
        _device: &Device,
    ) -> ModelResult<Self> {
        let weight_u8 = weights.get(&format!("{prefix}.weight"))?.clone();
        let absmax = weights.get(&format!("{prefix}.SCB"))?.clone();
        let bias = weights.get(&format!("{prefix}.bias")).ok().cloned();

        Ok(Self {
            weight_u8,
            absmax,
            bias,
            in_features,
            out_features,
            original_dtype: dtype,
        })
    }

    /// Dequantize INT8 weights to a full float weight matrix.
    ///
    /// Returns shape `[out_features, in_features]` in the original dtype.
    pub fn dequantize(&self) -> ModelResult<Tensor> {
        let device = self.weight_u8.device();

        // Read weight bytes (stored as uint8, reinterpreted as signed int8).
        let weight_data: Vec<u8> = self
            .weight_u8
            .flatten_all()
            .map_err(ModelError::Candle)?
            .to_vec1()
            .map_err(ModelError::Candle)?;

        // Read per-row absmax scales.
        let absmax_data: Vec<f32> = self
            .absmax
            .to_dtype(DType::F32)
            .map_err(ModelError::Candle)?
            .flatten_all()
            .map_err(ModelError::Candle)?
            .to_vec1()
            .map_err(ModelError::Candle)?;

        let mut out = vec![0.0f32; self.out_features * self.in_features];

        for (row, &scale_raw) in absmax_data.iter().enumerate().take(self.out_features) {
            let scale = scale_raw / 127.0;
            for col in 0..self.in_features {
                let idx = row * self.in_features + col;
                let signed_val = weight_data[idx] as i8;
                out[idx] = signed_val as f32 * scale;
            }
        }

        let t = Tensor::from_vec(out, (self.out_features, self.in_features), device)
            .map_err(ModelError::Candle)?;
        t.to_dtype(self.original_dtype).map_err(ModelError::Candle)
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

impl Module for BnbInt8Linear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let w = self
            .dequantize()
            .map_err(|e| candle_core::Error::Msg(format!("BnB INT8 dequantize: {e}")))?;

        let x_dtype = x.dtype();
        let x = if x.dtype() != w.dtype() {
            x.to_dtype(w.dtype())?
        } else {
            x.clone()
        };

        let wt = w.t()?.contiguous()?;
        let output = x.contiguous()?.matmul(&wt)?;
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
// BnbLinear — unified NF4/INT8 enum
// ---------------------------------------------------------------------------

/// Unified BitsAndBytes linear layer supporting both NF4 (4-bit) and INT8 (8-bit).
pub enum BnbLinear {
    Nf4(BnbNf4Linear),
    Int8(BnbInt8Linear),
}

impl BnbLinear {
    /// Load a BnB linear layer, selecting NF4 or INT8 based on the config.
    pub fn from_weights(
        weights: &ModelWeights,
        prefix: &str,
        config: &BnbLayerConfig,
        out_features: usize,
        in_features: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        match config {
            BnbLayerConfig::Nf4(nf4_cfg) => Ok(Self::Nf4(BnbNf4Linear::from_weights(
                weights,
                prefix,
                nf4_cfg,
                out_features,
                in_features,
                dtype,
                device,
            )?)),
            BnbLayerConfig::Int8 => Ok(Self::Int8(BnbInt8Linear::from_weights(
                weights,
                prefix,
                out_features,
                in_features,
                dtype,
                device,
            )?)),
        }
    }

    /// Dequantize to a full float weight matrix `[out_features, in_features]`.
    pub fn dequantize(&self) -> ModelResult<Tensor> {
        match self {
            Self::Nf4(inner) => inner.dequantize(),
            Self::Int8(inner) => inner.dequantize(),
        }
    }

    /// Input features (unquantized).
    pub fn in_features(&self) -> usize {
        match self {
            Self::Nf4(inner) => inner.in_features(),
            Self::Int8(inner) => inner.in_features(),
        }
    }

    /// Output features.
    pub fn out_features(&self) -> usize {
        match self {
            Self::Nf4(inner) => inner.out_features(),
            Self::Int8(inner) => inner.out_features(),
        }
    }
}

impl Module for BnbLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Nf4(inner) => inner.forward(x),
            Self::Int8(inner) => inner.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// BnbNf4Linear
// ---------------------------------------------------------------------------

/// A linear layer backed by BitsAndBytes NF4/FP4 quantized weights.
///
/// On each forward pass the packed 4-bit weights are dequantized to the
/// working dtype, multiplied with the input, and optionally bias is added.
pub struct BnbNf4Linear {
    packed: Tensor, // uint8, flat [num_elements/2]
    absmax: Tensor, // f32, [num_blocks]
    bias: Option<Tensor>,
    blocksize: usize,
    quant_type: BnbQuantType,
    in_features: usize,
    out_features: usize,
    original_dtype: DType,
}

impl BnbNf4Linear {
    /// Load a BnB NF4 linear layer from model weights.
    ///
    /// Looks for `{prefix}.weight` (packed uint8) and `{prefix}.weight.absmax`.
    /// Handles double quantization: if `{prefix}.weight.nested_absmax` exists,
    /// the absmax is dequantized from uint8 using the nested codebook+scales.
    pub fn from_weights(
        weights: &ModelWeights,
        prefix: &str,
        config: &BnbNf4Config,
        out_features: usize,
        in_features: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let packed = weights.get(&format!("{prefix}.weight"))?.clone();
        let absmax_raw = weights.get(&format!("{prefix}.weight.absmax"))?.clone();
        let bias = weights.get(&format!("{prefix}.bias")).ok().cloned();

        // Parse blocksize from quant_state JSON if available, else use config.
        let blocksize = if let Ok(qs) =
            weights.get(&format!("{prefix}.weight.quant_state.bitsandbytes__nf4"))
        {
            let qs_data: Vec<u8> = qs
                .flatten_all()
                .map_err(ModelError::Candle)?
                .to_vec1()
                .map_err(ModelError::Candle)?;
            if let Ok(meta) = serde_json::from_slice::<serde_json::Value>(&qs_data) {
                meta.get("blocksize")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(config.blocksize)
            } else {
                config.blocksize
            }
        } else {
            config.blocksize
        };

        // Handle double quantization.
        let absmax = if let (Ok(nested_absmax), Ok(nested_quant_map)) = (
            weights.get(&format!("{prefix}.weight.nested_absmax")),
            weights.get(&format!("{prefix}.weight.nested_quant_map")),
        ) {
            // Parse nested_offset from quant_state JSON.
            let nested_offset = if let Ok(qs) =
                weights.get(&format!("{prefix}.weight.quant_state.bitsandbytes__nf4"))
            {
                let qs_data: Vec<u8> = qs
                    .flatten_all()
                    .map_err(ModelError::Candle)?
                    .to_vec1()
                    .map_err(ModelError::Candle)?;
                serde_json::from_slice::<serde_json::Value>(&qs_data)
                    .ok()
                    .and_then(|m| m.get("nested_offset").and_then(|v| v.as_f64()))
                    .unwrap_or(0.0) as f32
            } else {
                0.0
            };

            let nested_blocksize = if let Ok(qs) =
                weights.get(&format!("{prefix}.weight.quant_state.bitsandbytes__nf4"))
            {
                let qs_data: Vec<u8> = qs
                    .flatten_all()
                    .map_err(ModelError::Candle)?
                    .to_vec1()
                    .map_err(ModelError::Candle)?;
                serde_json::from_slice::<serde_json::Value>(&qs_data)
                    .ok()
                    .and_then(|m| m.get("nested_blocksize").and_then(|v| v.as_u64()))
                    .map(|v| v as usize)
                    .unwrap_or(256)
            } else {
                256
            };

            dequantize_absmax(
                &absmax_raw,
                nested_absmax,
                nested_quant_map,
                nested_blocksize,
                nested_offset,
                device,
            )?
        } else {
            absmax_raw
        };

        Ok(Self {
            packed,
            absmax,
            bias,
            blocksize,
            quant_type: config.quant_type,
            in_features,
            out_features,
            original_dtype: dtype,
        })
    }

    /// Dequantize packed weights to a full float weight matrix.
    ///
    /// Returns shape `[out_features, in_features]` in the original dtype.
    pub fn dequantize(&self) -> ModelResult<Tensor> {
        let device = self.packed.device();

        // Read packed uint8 data to CPU.
        let packed_data: Vec<u8> = self
            .packed
            .flatten_all()
            .map_err(ModelError::Candle)?
            .to_vec1()
            .map_err(ModelError::Candle)?;

        // Read absmax f32 data.
        let absmax_data: Vec<f32> = self
            .absmax
            .to_dtype(DType::F32)
            .map_err(ModelError::Candle)?
            .flatten_all()
            .map_err(ModelError::Candle)?
            .to_vec1()
            .map_err(ModelError::Candle)?;

        let table = match self.quant_type {
            BnbQuantType::NF4 => &NF4_TABLE,
            BnbQuantType::FP4 => &FP4_TABLE,
        };

        let total = self.out_features * self.in_features;
        let mut out = vec![0.0f32; total];

        for i in 0..total {
            let byte = packed_data[i / 2];
            let nibble = ((byte >> ((i % 2) * 4)) & 0xF) as usize;
            let block_idx = i / self.blocksize;
            out[i] = absmax_data[block_idx] * table[nibble];
        }

        let t = Tensor::from_vec(out, (self.out_features, self.in_features), device)
            .map_err(ModelError::Candle)?;
        t.to_dtype(self.original_dtype).map_err(ModelError::Candle)
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

/// Dequantize double-quantized absmax from uint8 to f32.
///
/// `absmax_raw`: uint8 indices into `nested_quant_map` codebook
/// `nested_absmax`: f32 per-block scales for nested quantization
/// `nested_quant_map`: f32[256] codebook
fn dequantize_absmax(
    absmax_raw: &Tensor,
    nested_absmax: &Tensor,
    nested_quant_map: &Tensor,
    nested_blocksize: usize,
    nested_offset: f32,
    device: &Device,
) -> ModelResult<Tensor> {
    let raw_data: Vec<u8> = absmax_raw
        .flatten_all()
        .map_err(ModelError::Candle)?
        .to_vec1()
        .map_err(ModelError::Candle)?;

    let nested_scales: Vec<f32> = nested_absmax
        .to_dtype(DType::F32)
        .map_err(ModelError::Candle)?
        .flatten_all()
        .map_err(ModelError::Candle)?
        .to_vec1()
        .map_err(ModelError::Candle)?;

    let codebook: Vec<f32> = nested_quant_map
        .to_dtype(DType::F32)
        .map_err(ModelError::Candle)?
        .flatten_all()
        .map_err(ModelError::Candle)?
        .to_vec1()
        .map_err(ModelError::Candle)?;

    let n = raw_data.len();
    let bs = if nested_blocksize > 0 {
        nested_blocksize
    } else {
        256
    };

    let mut result = vec![0.0f32; n];
    for i in 0..n {
        let code = raw_data[i] as usize;
        let block_idx = i / bs;
        let scale = if block_idx < nested_scales.len() {
            nested_scales[block_idx]
        } else {
            1.0
        };
        result[i] = scale * codebook[code] + nested_offset;
    }

    Tensor::from_vec(result, n, device).map_err(ModelError::Candle)
}

impl Module for BnbNf4Linear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let w = self
            .dequantize()
            .map_err(|e| candle_core::Error::Msg(format!("BnB dequantize: {e}")))?;

        // x: [..., in_features], w: [out_features, in_features]
        // output: [..., out_features] = x @ w^T
        let x_dtype = x.dtype();
        let x = if x.dtype() != w.dtype() {
            x.to_dtype(w.dtype())?
        } else {
            x.clone()
        };

        // w is [out, in], need [in, out] for matmul: x @ w.T
        let wt = w.t()?.contiguous()?;
        let output = x.contiguous()?.matmul(&wt)?;
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

    /// Create a packed uint8 tensor from a sequence of 4-bit nibble values.
    fn pack_nibbles(nibbles: &[u8]) -> Vec<u8> {
        let num_bytes = (nibbles.len() + 1) / 2;
        let mut bytes = vec![0u8; num_bytes];
        for (i, &nib) in nibbles.iter().enumerate() {
            bytes[i / 2] |= (nib & 0xF) << ((i % 2) * 4);
        }
        bytes
    }

    /// Test nibble packing round-trip.
    #[test]
    fn test_nibble_packing() {
        let nibbles = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let packed = pack_nibbles(&nibbles);
        assert_eq!(packed.len(), 8);

        // Verify unpacking.
        for (i, &expected) in nibbles.iter().enumerate() {
            let byte = packed[i / 2];
            let actual = (byte >> ((i % 2) * 4)) & 0xF;
            assert_eq!(actual, expected, "mismatch at index {i}");
        }
    }

    /// Test dequantization with known absmax + table values.
    #[test]
    fn test_bnb_dequantize_basic() {
        let device = Device::Cpu;

        // 4 elements: nibble indices [0, 7, 8, 15] → NF4_TABLE values [-1.0, 0.0, 0.0796, 1.0]
        // absmax = 2.0 for all (blocksize=64, 1 block)
        let nibbles = vec![0u8, 7, 8, 15];
        let packed_bytes = pack_nibbles(&nibbles);
        let packed = Tensor::new(packed_bytes.as_slice(), &device).unwrap();
        let absmax = Tensor::new(&[2.0f32], &device).unwrap();

        let linear = BnbNf4Linear {
            packed,
            absmax,
            bias: None,
            blocksize: 64,
            quant_type: BnbQuantType::NF4,
            in_features: 4,
            out_features: 1,
            original_dtype: DType::F32,
        };

        let w = linear.dequantize().unwrap();
        assert_eq!(w.dims(), &[1, 4]);
        let vals: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();

        assert!((vals[0] - (-2.0)).abs() < 0.001, "got {}", vals[0]); // 2.0 * -1.0
        assert!((vals[1] - 0.0).abs() < 0.001, "got {}", vals[1]); // 2.0 * 0.0
        assert!((vals[2] - 0.15916).abs() < 0.001, "got {}", vals[2]); // 2.0 * 0.0795803
        assert!((vals[3] - 2.0).abs() < 0.001, "got {}", vals[3]); // 2.0 * 1.0
    }

    /// Test forward pass shape.
    #[test]
    fn test_bnb_linear_forward_shape() {
        let device = Device::Cpu;

        // 2 output features, 4 input features → 8 total elements → 4 packed bytes
        let nibbles = vec![8u8; 8]; // all index 8 → NF4_TABLE[8] = 0.0796
        let packed_bytes = pack_nibbles(&nibbles);
        let packed = Tensor::new(packed_bytes.as_slice(), &device).unwrap();
        let absmax = Tensor::new(&[1.0f32], &device).unwrap();

        let linear = BnbNf4Linear {
            packed,
            absmax,
            bias: None,
            blocksize: 64,
            quant_type: BnbQuantType::NF4,
            in_features: 4,
            out_features: 2,
            original_dtype: DType::F32,
        };

        let x = Tensor::ones((3, 4), DType::F32, &device).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, 2]);
    }

    /// Test forward pass produces expected values.
    #[test]
    fn test_bnb_linear_matches_float() {
        let device = Device::Cpu;

        // 1 output, 2 inputs. Nibbles: [15, 15] → NF4[15] = 1.0
        // absmax = 3.0 → dequant = [3.0, 3.0]
        // x = [1.0, 1.0] → output = 1.0*3.0 + 1.0*3.0 = 6.0
        let nibbles = vec![15u8, 15];
        let packed_bytes = pack_nibbles(&nibbles);
        let packed = Tensor::new(packed_bytes.as_slice(), &device).unwrap();
        let absmax = Tensor::new(&[3.0f32], &device).unwrap();

        let linear = BnbNf4Linear {
            packed,
            absmax,
            bias: None,
            blocksize: 64,
            quant_type: BnbQuantType::NF4,
            in_features: 2,
            out_features: 1,
            original_dtype: DType::F32,
        };

        let x = Tensor::ones((1, 2), DType::F32, &device).unwrap();
        let y = linear.forward(&x).unwrap();
        let val: f32 = y.flatten_all().unwrap().to_vec1().unwrap()[0];
        assert!((val - 6.0).abs() < 0.01, "expected 6.0, got {val}");
    }

    /// Test FP4 variant.
    #[test]
    fn test_bnb_fp4_dequantize() {
        let device = Device::Cpu;

        // FP4_TABLE[1] = 0.0625, absmax = 4.0 → 0.25
        let nibbles = vec![1u8, 1];
        let packed_bytes = pack_nibbles(&nibbles);
        let packed = Tensor::new(packed_bytes.as_slice(), &device).unwrap();
        let absmax = Tensor::new(&[4.0f32], &device).unwrap();

        let linear = BnbNf4Linear {
            packed,
            absmax,
            bias: None,
            blocksize: 64,
            quant_type: BnbQuantType::FP4,
            in_features: 2,
            out_features: 1,
            original_dtype: DType::F32,
        };

        let w = linear.dequantize().unwrap();
        let vals: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        for v in &vals {
            assert!((v - 0.25).abs() < 0.001, "expected 0.25, got {v}");
        }
    }

    /// Test bias handling.
    #[test]
    fn test_bnb_linear_with_bias() {
        let device = Device::Cpu;

        // Weight: all zeros (nibble 7 = NF4[7] = 0.0), bias = [10.0]
        let nibbles = vec![7u8; 4];
        let packed_bytes = pack_nibbles(&nibbles);
        let packed = Tensor::new(packed_bytes.as_slice(), &device).unwrap();
        let absmax = Tensor::new(&[1.0f32], &device).unwrap();
        let bias = Tensor::new(&[10.0f32], &device).unwrap();

        let linear = BnbNf4Linear {
            packed,
            absmax,
            bias: Some(bias),
            blocksize: 64,
            quant_type: BnbQuantType::NF4,
            in_features: 4,
            out_features: 1,
            original_dtype: DType::F32,
        };

        let x = Tensor::ones((1, 4), DType::F32, &device).unwrap();
        let y = linear.forward(&x).unwrap();
        let val: f32 = y.flatten_all().unwrap().to_vec1().unwrap()[0];
        assert!((val - 10.0).abs() < 0.01, "expected 10.0, got {val}");
    }

    /// Test multiple blocks (blocksize doesn't evenly divide elements).
    #[test]
    fn test_bnb_multiple_blocks() {
        let device = Device::Cpu;

        // 1 output, 4 inputs, blocksize=2 → 2 blocks.
        // Block 0: absmax=2.0, nibbles [15, 15] → [2.0, 2.0]
        // Block 1: absmax=3.0, nibbles [15, 15] → [3.0, 3.0]
        let nibbles = vec![15u8; 4];
        let packed_bytes = pack_nibbles(&nibbles);
        let packed = Tensor::new(packed_bytes.as_slice(), &device).unwrap();
        let absmax = Tensor::new(&[2.0f32, 3.0], &device).unwrap();

        let linear = BnbNf4Linear {
            packed,
            absmax,
            bias: None,
            blocksize: 2,
            quant_type: BnbQuantType::NF4,
            in_features: 4,
            out_features: 1,
            original_dtype: DType::F32,
        };

        let w = linear.dequantize().unwrap();
        let vals: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        assert!((vals[0] - 2.0).abs() < 0.001);
        assert!((vals[1] - 2.0).abs() < 0.001);
        assert!((vals[2] - 3.0).abs() < 0.001);
        assert!((vals[3] - 3.0).abs() < 0.001);
    }

    /// Test dtype conversion (BF16 output).
    #[test]
    fn test_bnb_dequantize_bf16() {
        let device = Device::Cpu;

        let nibbles = vec![15u8, 0]; // NF4[15]=1.0, NF4[0]=-1.0
        let packed_bytes = pack_nibbles(&nibbles);
        let packed = Tensor::new(packed_bytes.as_slice(), &device).unwrap();
        let absmax = Tensor::new(&[1.0f32], &device).unwrap();

        let linear = BnbNf4Linear {
            packed,
            absmax,
            bias: None,
            blocksize: 64,
            quant_type: BnbQuantType::NF4,
            in_features: 2,
            out_features: 1,
            original_dtype: DType::BF16,
        };

        let w = linear.dequantize().unwrap();
        assert_eq!(w.dtype(), DType::BF16);
        let vals: Vec<f32> = w
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!((vals[0] - 1.0).abs() < 0.02);
        assert!((vals[1] - (-1.0)).abs() < 0.02);
    }

    // -----------------------------------------------------------------------
    // INT8 tests
    // -----------------------------------------------------------------------

    /// Create a uint8 tensor from signed i8 values (two's complement).
    fn i8_to_u8_bytes(vals: &[i8]) -> Vec<u8> {
        vals.iter().map(|&v| v as u8).collect()
    }

    /// Test INT8 dequantization with known values.
    #[test]
    fn test_bnb_int8_dequantize_basic() {
        let device = Device::Cpu;

        // 1 row, 4 cols. INT8 values: [127, -127, 0, 64]
        // absmax (SCB) = 2.54 → scale = 2.54/127 = 0.02
        // Expected: [127*0.02, -127*0.02, 0, 64*0.02] = [2.54, -2.54, 0.0, 1.28]
        let weight_bytes = i8_to_u8_bytes(&[127, -127, 0, 64]);
        let weight = Tensor::new(weight_bytes.as_slice(), &device).unwrap();
        let absmax = Tensor::new(&[2.54f32], &device).unwrap();

        let linear = BnbInt8Linear {
            weight_u8: weight,
            absmax,
            bias: None,
            in_features: 4,
            out_features: 1,
            original_dtype: DType::F32,
        };

        let w = linear.dequantize().unwrap();
        assert_eq!(w.dims(), &[1, 4]);
        let vals: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();

        assert!(
            (vals[0] - 2.54).abs() < 0.01,
            "expected 2.54, got {}",
            vals[0]
        );
        assert!(
            (vals[1] - (-2.54)).abs() < 0.01,
            "expected -2.54, got {}",
            vals[1]
        );
        assert!(
            (vals[2] - 0.0).abs() < 0.01,
            "expected 0.0, got {}",
            vals[2]
        );
        assert!(
            (vals[3] - 1.28).abs() < 0.01,
            "expected 1.28, got {}",
            vals[3]
        );
    }

    /// Test INT8 forward pass shape.
    #[test]
    fn test_bnb_int8_forward_shape() {
        let device = Device::Cpu;

        // 2 outputs, 4 inputs — 8 bytes.
        let weight_bytes = i8_to_u8_bytes(&[127i8; 8]);
        let weight = Tensor::new(weight_bytes.as_slice(), &device).unwrap();
        let absmax = Tensor::new(&[1.0f32, 1.0], &device).unwrap();

        let linear = BnbInt8Linear {
            weight_u8: weight,
            absmax,
            bias: None,
            in_features: 4,
            out_features: 2,
            original_dtype: DType::F32,
        };

        let x = Tensor::ones((3, 4), DType::F32, &device).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, 2]);
    }

    /// Test INT8 forward pass values.
    #[test]
    fn test_bnb_int8_forward_values() {
        let device = Device::Cpu;

        // 1 output, 2 inputs. INT8 = [127, 127], absmax = 1.27
        // scale = 1.27/127 = 0.01, dequant = [1.27, 1.27]
        // x = [1.0, 1.0] → output = 1.27 + 1.27 = 2.54
        let weight_bytes = i8_to_u8_bytes(&[127, 127]);
        let weight = Tensor::new(weight_bytes.as_slice(), &device).unwrap();
        let absmax = Tensor::new(&[1.27f32], &device).unwrap();

        let linear = BnbInt8Linear {
            weight_u8: weight,
            absmax,
            bias: None,
            in_features: 2,
            out_features: 1,
            original_dtype: DType::F32,
        };

        let x = Tensor::ones((1, 2), DType::F32, &device).unwrap();
        let y = linear.forward(&x).unwrap();
        let val: f32 = y.flatten_all().unwrap().to_vec1().unwrap()[0];
        assert!((val - 2.54).abs() < 0.01, "expected 2.54, got {val}");
    }

    /// Test INT8 with bias.
    #[test]
    fn test_bnb_int8_with_bias() {
        let device = Device::Cpu;

        // All zeros (int8=0), bias=10.0 → output = 10.0
        let weight_bytes = vec![0u8; 4];
        let weight = Tensor::new(weight_bytes.as_slice(), &device).unwrap();
        let absmax = Tensor::new(&[1.0f32], &device).unwrap();
        let bias = Tensor::new(&[10.0f32], &device).unwrap();

        let linear = BnbInt8Linear {
            weight_u8: weight,
            absmax,
            bias: Some(bias),
            in_features: 4,
            out_features: 1,
            original_dtype: DType::F32,
        };

        let x = Tensor::ones((1, 4), DType::F32, &device).unwrap();
        let y = linear.forward(&x).unwrap();
        let val: f32 = y.flatten_all().unwrap().to_vec1().unwrap()[0];
        assert!((val - 10.0).abs() < 0.01, "expected 10.0, got {val}");
    }

    /// Test INT8 BF16 dtype conversion.
    #[test]
    fn test_bnb_int8_dequantize_bf16() {
        let device = Device::Cpu;

        let weight_bytes = i8_to_u8_bytes(&[127, -127]);
        let weight = Tensor::new(weight_bytes.as_slice(), &device).unwrap();
        let absmax = Tensor::new(&[1.27f32], &device).unwrap();

        let linear = BnbInt8Linear {
            weight_u8: weight,
            absmax,
            bias: None,
            in_features: 2,
            out_features: 1,
            original_dtype: DType::BF16,
        };

        let w = linear.dequantize().unwrap();
        assert_eq!(w.dtype(), DType::BF16);
        let vals: Vec<f32> = w
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!((vals[0] - 1.27).abs() < 0.02);
        assert!((vals[1] - (-1.27)).abs() < 0.02);
    }

    /// Test INT8 multiple rows (different absmax per row).
    #[test]
    fn test_bnb_int8_multiple_rows() {
        let device = Device::Cpu;

        // 2 rows, 2 cols. Row 0: absmax=2.54, Row 1: absmax=1.27
        // INT8 vals: [[127, 127], [127, 127]]
        // Row 0 dequant: [2.54, 2.54], Row 1 dequant: [1.27, 1.27]
        let weight_bytes = i8_to_u8_bytes(&[127, 127, 127, 127]);
        let weight = Tensor::new(weight_bytes.as_slice(), &device).unwrap();
        let absmax = Tensor::new(&[2.54f32, 1.27], &device).unwrap();

        let linear = BnbInt8Linear {
            weight_u8: weight,
            absmax,
            bias: None,
            in_features: 2,
            out_features: 2,
            original_dtype: DType::F32,
        };

        let w = linear.dequantize().unwrap();
        let vals: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        assert!((vals[0] - 2.54).abs() < 0.01); // row 0, col 0
        assert!((vals[1] - 2.54).abs() < 0.01); // row 0, col 1
        assert!((vals[2] - 1.27).abs() < 0.01); // row 1, col 0
        assert!((vals[3] - 1.27).abs() < 0.01); // row 1, col 1
    }

    // -----------------------------------------------------------------------
    // BnbLinear enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bnb_linear_nf4_variant() {
        let device = Device::Cpu;

        let nibbles = vec![15u8, 15];
        let packed_bytes = pack_nibbles(&nibbles);
        let packed = Tensor::new(packed_bytes.as_slice(), &device).unwrap();
        let absmax = Tensor::new(&[3.0f32], &device).unwrap();

        let nf4 = BnbNf4Linear {
            packed,
            absmax,
            bias: None,
            blocksize: 64,
            quant_type: BnbQuantType::NF4,
            in_features: 2,
            out_features: 1,
            original_dtype: DType::F32,
        };
        let unified = BnbLinear::Nf4(nf4);

        let x = Tensor::ones((1, 2), DType::F32, &device).unwrap();
        let y = unified.forward(&x).unwrap();
        let val: f32 = y.flatten_all().unwrap().to_vec1().unwrap()[0];
        assert!((val - 6.0).abs() < 0.01, "expected 6.0, got {val}");
    }

    #[test]
    fn test_bnb_linear_int8_variant() {
        let device = Device::Cpu;

        let weight_bytes = i8_to_u8_bytes(&[127, 127]);
        let weight = Tensor::new(weight_bytes.as_slice(), &device).unwrap();
        let absmax = Tensor::new(&[1.27f32], &device).unwrap();

        let int8 = BnbInt8Linear {
            weight_u8: weight,
            absmax,
            bias: None,
            in_features: 2,
            out_features: 1,
            original_dtype: DType::F32,
        };
        let unified = BnbLinear::Int8(int8);

        let x = Tensor::ones((1, 2), DType::F32, &device).unwrap();
        let y = unified.forward(&x).unwrap();
        let val: f32 = y.flatten_all().unwrap().to_vec1().unwrap()[0];
        assert!((val - 2.54).abs() < 0.01, "expected 2.54, got {val}");
    }
}
