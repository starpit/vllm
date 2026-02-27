// SPDX-License-Identifier: Apache-2.0
//! Quantized linear layer wrapping candle's `QMatMul`.
//!
//! Used by GGUF quantized models where weights are stored in quantized
//! formats (Q4_0, Q4_K, Q8_0, etc.) rather than f16/f32. The `QMatMul`
//! type handles dequantization-on-the-fly or native quantized matmul
//! depending on the backend (CPU, Metal, CUDA).

use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{Module, Tensor};

use crate::error::{ModelError, ModelResult};
use crate::gguf::GgufFile;

// ---------------------------------------------------------------------------
// QuantizedLinear
// ---------------------------------------------------------------------------

/// A linear layer backed by quantized weights.
///
/// Wraps `candle_core::quantized::QMatMul` which supports:
/// - Native quantized matmul on CPU (via GGML kernels)
/// - Metal quantized matmul (via MSL kernels)
/// - CUDA quantized matmul
/// - Automatic dequantization fallback for F32/F16/BF16 weights
pub struct QuantizedLinear {
    inner: QMatMul,
}

impl QuantizedLinear {
    /// Create from a pre-loaded `QTensor`.
    pub fn from_qtensor(qtensor: QTensor) -> ModelResult<Self> {
        let inner = QMatMul::from_qtensor(qtensor)
            .map_err(|e| ModelError::Other(format!("QMatMul creation failed: {e}")))?;
        Ok(Self { inner })
    }

    /// Load a single quantized tensor from a GGUF file by name.
    pub fn from_gguf(
        gguf: &mut GgufFile,
        name: &str,
        device: &candle_core::Device,
    ) -> ModelResult<Self> {
        let qtensor = gguf.tensor(name, device)?;
        Self::from_qtensor(qtensor)
    }

    /// Access the inner `QMatMul`.
    pub fn inner(&self) -> &QMatMul {
        &self.inner
    }
}

impl Module for QuantizedLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.inner.forward(x)
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
    fn test_quantized_linear_from_f32_qtensor() {
        // QTensor with F32 dtype gets dequantized by QMatMul internally,
        // so this tests the full pipeline with a simple case.
        let data: &[f32] = &[1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let tensor = Tensor::new(data, &Device::Cpu)
            .unwrap()
            .reshape((2, 2))
            .unwrap();
        let qtensor = QTensor::quantize(&tensor, candle_core::quantized::GgmlDType::F32).unwrap();

        let linear = QuantizedLinear::from_qtensor(qtensor).unwrap();

        let x = Tensor::new(&[[3.0f32, 4.0]], &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 2]);

        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[0] - 3.0).abs() < 1e-4);
        assert!((vals[1] - 4.0).abs() < 1e-4);
    }
}
