// SPDX-License-Identifier: Apache-2.0
//! Normalization layers.
//!
//! Port of: `vllm/model_executor/layers/layernorm.py`

use candle_core::{DType, Device, Module, Tensor};

use crate::error::ModelResult;
use crate::tensor;
use crate::weight::ModelWeights;

// ---------------------------------------------------------------------------
// RmsNorm
// ---------------------------------------------------------------------------

/// Root Mean Square Layer Normalization.
///
/// Normalizes the input by the RMS of its elements, then scales by a
/// learned weight vector. Used by LLaMA, Mistral, Qwen, etc.
///
/// Formula: `y = x / sqrt(mean(x^2) + eps) * weight`
///
/// Port of: `vllm/model_executor/layers/layernorm.py::RMSNorm`
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    /// Create from an explicit weight tensor.
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Load from model weights.
    ///
    /// Looks for `{prefix}.weight`.
    pub fn load(weights: &ModelWeights, prefix: &str, eps: f64, dtype: DType) -> ModelResult<Self> {
        let weight_name = format!("{}.weight", prefix);
        let weight = weights.get_cast(&weight_name, dtype)?;
        Ok(Self { weight, eps })
    }

    /// Create with ones (for testing).
    pub fn ones(hidden_size: usize, eps: f64, dtype: DType, device: &Device) -> ModelResult<Self> {
        let weight = tensor::ones(&[hidden_size], dtype, device)?;
        Ok(Self { weight, eps })
    }

    /// The epsilon value.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// The weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Hidden size.
    pub fn hidden_size(&self) -> usize {
        self.weight.dim(0).unwrap_or(0)
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // x: [..., hidden_size]
        // 1. Compute variance = mean(x^2) over last dim
        let x_sq = x.sqr()?;
        let variance = x_sq.mean_keepdim(candle_core::D::Minus1)?;
        // 2. Normalize: x * rsqrt(variance + eps)
        let rsqrt = (variance + self.eps)?.sqrt()?.recip()?;
        let normed = x.broadcast_mul(&rsqrt)?;
        // 3. Scale by weight
        normed.broadcast_mul(&self.weight)
    }
}

// ---------------------------------------------------------------------------
// GemmaRmsNorm
// ---------------------------------------------------------------------------

/// Gemma variant of RMS normalization.
///
/// Identical to `RmsNorm` except that it adds 1.0 to the learned weight
/// before scaling. This means the model stores `weight` such that the
/// effective scale is `(1 + weight)`, which biases the initial effective
/// weight towards identity.
///
/// Formula: `y = x / sqrt(mean(x^2) + eps) * (1 + weight)`
///
/// Port of: `vllm/model_executor/layers/layernorm.py::GemmaRMSNorm`
pub struct GemmaRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl GemmaRmsNorm {
    /// Create from an explicit weight tensor.
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Load from model weights.
    ///
    /// Looks for `{prefix}.weight`.
    pub fn load(weights: &ModelWeights, prefix: &str, eps: f64, dtype: DType) -> ModelResult<Self> {
        let weight_name = format!("{}.weight", prefix);
        let weight = weights.get_cast(&weight_name, dtype)?;
        Ok(Self { weight, eps })
    }

    /// Create with zeros (so effective weight = 1, for testing).
    pub fn zeros(hidden_size: usize, eps: f64, dtype: DType, device: &Device) -> ModelResult<Self> {
        let weight = tensor::zeros(&[hidden_size], dtype, device)?;
        Ok(Self { weight, eps })
    }

    /// The epsilon value.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// The weight tensor (before +1).
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Hidden size.
    pub fn hidden_size(&self) -> usize {
        self.weight.dim(0).unwrap_or(0)
    }
}

impl Module for GemmaRmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // x: [..., hidden_size]
        // 1. Compute variance = mean(x^2) over last dim
        let x_sq = x.sqr()?;
        let variance = x_sq.mean_keepdim(candle_core::D::Minus1)?;
        // 2. Normalize: x * rsqrt(variance + eps)
        let rsqrt = (variance + self.eps)?.sqrt()?.recip()?;
        let normed = x.broadcast_mul(&rsqrt)?;
        // 3. Scale by (1 + weight)
        let effective_weight = (&self.weight + 1.0)?;
        normed.broadcast_mul(&effective_weight)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_identity() {
        // With weight = 1 and input already normalized, output ~= input.
        let norm = RmsNorm::ones(4, 1e-5, DType::F32, &Device::Cpu).unwrap();

        // Input with RMS = 1: [1, 1, 1, 1] / sqrt(1) = [1,1,1,1]
        // Actually [1,1,1,1] has RMS = 1, so output = [1,1,1,1] * 1 = [1,1,1,1]
        let x = Tensor::ones(&[1, 4], DType::F32, &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap();
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in vals {
            assert!((v - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_rms_norm_scaling() {
        // weight = [2, 2, 2, 2], input = [1, 1, 1, 1]
        // RMS(x) = 1, so output = [2, 2, 2, 2]
        let weight = Tensor::new(&[2.0f32, 2.0, 2.0, 2.0], &Device::Cpu).unwrap();
        let norm = RmsNorm::new(weight, 1e-5);

        let x = Tensor::ones(&[1, 4], DType::F32, &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap();
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in vals {
            assert!((v - 2.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_rms_norm_normalization() {
        // Input = [2, 0, 0, 0], weight = [1, 1, 1, 1]
        // x^2 = [4, 0, 0, 0], mean(x^2) = 1, RMS = 1
        // normalized = [2, 0, 0, 0] / 1 = [2, 0, 0, 0]
        let norm = RmsNorm::ones(4, 1e-6, DType::F32, &Device::Cpu).unwrap();

        let x = Tensor::new(&[[2.0f32, 0.0, 0.0, 0.0]], &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap();
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[0] - 2.0).abs() < 1e-4);
        assert!(vals[1].abs() < 1e-6);
    }

    #[test]
    fn test_rms_norm_batch() {
        let norm = RmsNorm::ones(3, 1e-5, DType::F32, &Device::Cpu).unwrap();
        let x = Tensor::ones(&[4, 3], DType::F32, &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.dims(), &[4, 3]);
    }

    #[test]
    fn test_rms_norm_properties() {
        let norm = RmsNorm::ones(128, 1e-6, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(norm.hidden_size(), 128);
        assert!((norm.eps() - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_rms_norm_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        let w_data: Vec<u8> = [1.0f32, 1.0, 1.0, 1.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        crate::weight::tests_helper::create_safetensors_file(
            &path,
            &[("norm.weight", vec![4], DType::F32, &w_data)],
        );

        let weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();
        let norm = RmsNorm::load(&weights, "norm", 1e-5, DType::F32).unwrap();
        assert_eq!(norm.hidden_size(), 4);
    }

    // -----------------------------------------------------------------------
    // GemmaRmsNorm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gemma_rms_norm_identity() {
        // weight = 0 → effective weight = 1 → identity scaling.
        let norm = GemmaRmsNorm::zeros(4, 1e-5, DType::F32, &Device::Cpu).unwrap();

        let x = Tensor::ones(&[1, 4], DType::F32, &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap();
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in vals {
            assert!((v - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_gemma_rms_norm_weight_offset() {
        // weight = [1, 1, 1, 1] → effective weight = [2, 2, 2, 2]
        // input = [1, 1, 1, 1], RMS = 1 → output = [2, 2, 2, 2]
        let weight = Tensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
        let norm = GemmaRmsNorm::new(weight, 1e-5);

        let x = Tensor::ones(&[1, 4], DType::F32, &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap();
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in vals {
            assert!((v - 2.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_gemma_rms_norm_vs_standard() {
        // When Gemma weight = w, effective is (1+w).
        // When standard weight = (1+w), they should produce the same output.
        let gemma_w = Tensor::new(&[0.5f32, 0.5, 0.5, 0.5], &Device::Cpu).unwrap();
        let gemma_norm = GemmaRmsNorm::new(gemma_w, 1e-5);

        let std_w = Tensor::new(&[1.5f32, 1.5, 1.5, 1.5], &Device::Cpu).unwrap();
        let std_norm = RmsNorm::new(std_w, 1e-5);

        let x = Tensor::new(&[[2.0f32, 0.5, 1.0, 3.0]], &Device::Cpu).unwrap();
        let gemma_out = gemma_norm.forward(&x).unwrap();
        let std_out = std_norm.forward(&x).unwrap();

        let g_vals = gemma_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let s_vals = std_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (g, s) in g_vals.iter().zip(s_vals.iter()) {
            assert!((g - s).abs() < 1e-5);
        }
    }

    #[test]
    fn test_gemma_rms_norm_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        let w_data: Vec<u8> = [0.0f32, 0.0, 0.0, 0.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        crate::weight::tests_helper::create_safetensors_file(
            &path,
            &[("norm.weight", vec![4], DType::F32, &w_data)],
        );

        let weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();
        let norm = GemmaRmsNorm::load(&weights, "norm", 1e-5, DType::F32).unwrap();
        assert_eq!(norm.hidden_size(), 4);
    }
}
