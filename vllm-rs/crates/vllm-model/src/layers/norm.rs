// SPDX-License-Identifier: Apache-2.0
//! Normalization layers.
//!
//! Port of: `vllm/model_executor/layers/layernorm.py`

use candle_core::{DType, Device, Module, Tensor};

use crate::error::{ModelError, ModelResult};
use crate::tensor;
use crate::weight::ModelWeights;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns `true` if the dtype should be upcast to f32 for reduction ops.
fn needs_upcast(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16)
}

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
    pub fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        eps: f64,
        dtype: DType,
    ) -> ModelResult<Self> {
        let weight_name = format!("{}.weight", prefix);
        let weight = weights.take_cast(&weight_name, dtype)?;
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
        let input_dtype = x.dtype();
        if needs_upcast(input_dtype) {
            // Upcast only for the variance reduction (f16/bf16 lack precision
            // for sqr→mean→sqrt). The rsqrt is shape [.., 1] — tiny to cast.
            // Keep x in native dtype to avoid 2 full-tensor copies.
            let x_f32 = x.to_dtype(DType::F32)?;
            let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
            let rsqrt = (variance + self.eps)?.sqrt()?.recip()?;
            let rsqrt = rsqrt.to_dtype(input_dtype)?; // tiny: [batch, 1]
            x.broadcast_mul(&rsqrt)?.broadcast_mul(&self.weight)
        } else {
            let x_sq = x.sqr()?;
            let variance = x_sq.mean_keepdim(candle_core::D::Minus1)?;
            let rsqrt = (variance + self.eps)?.sqrt()?.recip()?;
            x.broadcast_mul(&rsqrt)?.broadcast_mul(&self.weight)
        }
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
/// The effective weight `(1 + weight)` is precomputed at construction time
/// to avoid a per-forward dispatch.
///
/// Formula: `y = x / sqrt(mean(x^2) + eps) * (1 + weight)`
///
/// Port of: `vllm/model_executor/layers/layernorm.py::GemmaRMSNorm`
pub struct GemmaRmsNorm {
    /// Precomputed (weight + 1.0) to avoid per-forward dispatch.
    effective_weight: Tensor,
    eps: f64,
}

impl GemmaRmsNorm {
    /// Create from an explicit weight tensor.
    pub fn new(weight: Tensor, eps: f64) -> candle_core::Result<Self> {
        let effective_weight = (&weight + 1.0)?;
        Ok(Self {
            effective_weight,
            eps,
        })
    }

    /// Load from model weights.
    ///
    /// Looks for `{prefix}.weight`.
    pub fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        eps: f64,
        dtype: DType,
    ) -> ModelResult<Self> {
        let weight_name = format!("{}.weight", prefix);
        let weight = weights.take_cast(&weight_name, dtype)?;
        Ok(Self::new(weight, eps)?)
    }

    /// Create with zeros (so effective weight = 1, for testing).
    pub fn zeros(hidden_size: usize, eps: f64, dtype: DType, device: &Device) -> ModelResult<Self> {
        let weight = tensor::zeros(&[hidden_size], dtype, device)?;
        Ok(Self::new(weight, eps)?)
    }

    /// The epsilon value.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// The effective weight tensor (weight + 1).
    pub fn weight(&self) -> &Tensor {
        &self.effective_weight
    }

    /// Hidden size.
    pub fn hidden_size(&self) -> usize {
        self.effective_weight.dim(0).unwrap_or(0)
    }
}

impl Module for GemmaRmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let input_dtype = x.dtype();
        if needs_upcast(input_dtype) {
            let x_f32 = x.to_dtype(DType::F32)?;
            let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
            let rsqrt = (variance + self.eps)?.sqrt()?.recip()?;
            let rsqrt = rsqrt.to_dtype(input_dtype)?;
            x.broadcast_mul(&rsqrt)?
                .broadcast_mul(&self.effective_weight)
        } else {
            let x_sq = x.sqr()?;
            let variance = x_sq.mean_keepdim(candle_core::D::Minus1)?;
            let rsqrt = (variance + self.eps)?.sqrt()?.recip()?;
            x.broadcast_mul(&rsqrt)?
                .broadcast_mul(&self.effective_weight)
        }
    }
}

// ---------------------------------------------------------------------------
// LayerNorm (standard, with weight + bias)
// ---------------------------------------------------------------------------

/// Standard Layer Normalization with weight and bias parameters.
///
/// Formula: `y = weight * (x - mean(x)) / sqrt(var(x) + eps) + bias`
///
/// Used by vision encoders (SigLIP, CLIP) which require both weight and bias,
/// unlike RmsNorm (decoder-only LLMs) or CohereLayerNorm (weight only).
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    /// Create from explicit weight and bias tensors.
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self { weight, bias, eps }
    }

    /// Load from model weights.
    ///
    /// Looks for `{prefix}.weight` and `{prefix}.bias`.
    pub fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        eps: f64,
        dtype: DType,
    ) -> ModelResult<Self> {
        let weight = weights.take_cast(&format!("{prefix}.weight"), dtype)?;
        let bias = weights.take_cast(&format!("{prefix}.bias"), dtype)?;
        Ok(Self { weight, bias, eps })
    }

    /// Create with ones weight and zeros bias (for testing).
    pub fn ones(hidden_size: usize, eps: f64, dtype: DType, device: &Device) -> ModelResult<Self> {
        let weight = tensor::ones(&[hidden_size], dtype, device)?;
        let bias = Tensor::zeros(hidden_size, dtype, device).map_err(ModelError::Candle)?;
        Ok(Self { weight, bias, eps })
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let input_dtype = x.dtype();
        if needs_upcast(input_dtype) {
            let x_f32 = x.to_dtype(DType::F32)?;
            let mean = x_f32.mean_keepdim(candle_core::D::Minus1)?;
            let centered = x_f32.broadcast_sub(&mean)?;
            let variance = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
            let rsqrt = (variance + self.eps)?.sqrt()?.recip()?;
            let normed = centered.broadcast_mul(&rsqrt)?.to_dtype(input_dtype)?;
            normed
                .broadcast_mul(&self.weight)?
                .broadcast_add(&self.bias)
        } else {
            let mean = x.mean_keepdim(candle_core::D::Minus1)?;
            let centered = x.broadcast_sub(&mean)?;
            let variance = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
            let rsqrt = (variance + self.eps)?.sqrt()?.recip()?;
            centered
                .broadcast_mul(&rsqrt)?
                .broadcast_mul(&self.weight)?
                .broadcast_add(&self.bias)
        }
    }
}

// ---------------------------------------------------------------------------
// CohereLayerNorm
// ---------------------------------------------------------------------------

/// Cohere Layer Normalization (used by Command R).
///
/// Full LayerNorm with mean subtraction, but only a weight parameter (no bias).
/// Unlike standard LayerNorm which has both weight and bias, and unlike RMSNorm
/// which skips mean subtraction.
///
/// Formula: `y = weight * (x - mean(x)) / sqrt(var(x) + eps)`
///
/// Port of: `transformers/models/cohere/modeling_cohere.py::CohereLayerNorm`
pub struct CohereLayerNorm {
    weight: Tensor,
    eps: f64,
}

impl CohereLayerNorm {
    /// Create from an explicit weight tensor.
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Load from model weights.
    ///
    /// Looks for `{prefix}.weight`.
    pub fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        eps: f64,
        dtype: DType,
    ) -> ModelResult<Self> {
        let weight_name = format!("{}.weight", prefix);
        let weight = weights.take_cast(&weight_name, dtype)?;
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

impl Module for CohereLayerNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let input_dtype = x.dtype();
        if needs_upcast(input_dtype) {
            let x_f32 = x.to_dtype(DType::F32)?;
            let mean = x_f32.mean_keepdim(candle_core::D::Minus1)?;
            let centered = x_f32.broadcast_sub(&mean)?;
            let variance = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
            let rsqrt = (variance + self.eps)?.sqrt()?.recip()?;
            let rsqrt = rsqrt.to_dtype(input_dtype)?;
            let mean = mean.to_dtype(input_dtype)?;
            x.broadcast_sub(&mean)?
                .broadcast_mul(&rsqrt)?
                .broadcast_mul(&self.weight)
        } else {
            let mean = x.mean_keepdim(candle_core::D::Minus1)?;
            let centered = x.broadcast_sub(&mean)?;
            let variance = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
            let rsqrt = (variance + self.eps)?.sqrt()?.recip()?;
            centered.broadcast_mul(&rsqrt)?.broadcast_mul(&self.weight)
        }
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

        let mut weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();
        let norm = RmsNorm::load(&mut weights, "norm", 1e-5, DType::F32).unwrap();
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
        let norm = GemmaRmsNorm::new(weight, 1e-5).unwrap();

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
        let gemma_norm = GemmaRmsNorm::new(gemma_w, 1e-5).unwrap();

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

        let mut weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();
        let norm = GemmaRmsNorm::load(&mut weights, "norm", 1e-5, DType::F32).unwrap();
        assert_eq!(norm.hidden_size(), 4);
    }

    // -----------------------------------------------------------------------
    // F16 dtype tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rms_norm_f16_output_dtype() {
        // F16 input should produce F16 output.
        let weight = Tensor::ones(&[4], DType::F16, &Device::Cpu).unwrap();
        let norm = RmsNorm::new(weight, 1e-5);

        let x = Tensor::ones(&[2, 4], DType::F16, &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.dtype(), DType::F16);
        assert_eq!(y.dims(), &[2, 4]);
    }

    #[test]
    fn test_rms_norm_f16_values() {
        // Verify numerical correctness: weight=1, input=[1,1,1,1] → output=[1,1,1,1].
        let weight = Tensor::ones(&[4], DType::F16, &Device::Cpu).unwrap();
        let norm = RmsNorm::new(weight, 1e-5);

        let x = Tensor::ones(&[1, 4], DType::F16, &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap();
        let y_f32 = y.to_dtype(DType::F32).unwrap();
        let vals = y_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in vals {
            assert!((v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_gemma_rms_norm_f16_output_dtype() {
        let weight = Tensor::zeros(&[4], DType::F16, &Device::Cpu).unwrap();
        let norm = GemmaRmsNorm::new(weight, 1e-5).unwrap();

        let x = Tensor::ones(&[2, 4], DType::F16, &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.dtype(), DType::F16);
    }

    // -----------------------------------------------------------------------
    // CohereLayerNorm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cohere_layernorm_identity() {
        // weight = 1, input = [1,1,1,1]: mean=1, var=0, output = 1*(0)/sqrt(0+eps) → 0
        // Actually: centered = [0,0,0,0], so output = [0,0,0,0].
        let norm = CohereLayerNorm::ones(4, 1e-5, DType::F32, &Device::Cpu).unwrap();
        let x = Tensor::ones(&[1, 4], DType::F32, &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap();
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in vals {
            assert!(
                v.abs() < 1e-4,
                "constant input should normalize to 0, got {v}"
            );
        }
    }

    #[test]
    fn test_cohere_layernorm_mean_subtraction() {
        // Input = [2, 0, 0, 0], mean = 0.5
        // centered = [1.5, -0.5, -0.5, -0.5]
        // var = (1.5^2 + 3*0.5^2) / 4 = (2.25 + 0.75) / 4 = 0.75
        // output = weight * centered / sqrt(0.75 + eps)
        // With weight=1: output[0] = 1.5 / sqrt(0.75) ≈ 1.7321
        let norm = CohereLayerNorm::ones(4, 1e-6, DType::F32, &Device::Cpu).unwrap();
        let x = Tensor::new(&[[2.0f32, 0.0, 0.0, 0.0]], &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap();
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected_0 = 1.5 / (0.75f32).sqrt();
        assert!(
            (vals[0] - expected_0).abs() < 1e-4,
            "expected {expected_0}, got {}",
            vals[0]
        );
        let expected_1 = -0.5 / (0.75f32).sqrt();
        assert!(
            (vals[1] - expected_1).abs() < 1e-4,
            "expected {expected_1}, got {}",
            vals[1]
        );
    }

    #[test]
    fn test_cohere_layernorm_vs_rms_norm() {
        // CohereLayerNorm subtracts mean; RmsNorm does not. They should differ
        // on non-zero-mean inputs.
        let weight = Tensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
        let cohere = CohereLayerNorm::new(weight.clone(), 1e-5);
        let rms = RmsNorm::new(weight, 1e-5);

        let x = Tensor::new(&[[3.0f32, 1.0, 1.0, 1.0]], &Device::Cpu).unwrap();
        let c_out = cohere
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let r_out = rms
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // Sum of CohereLayerNorm output should be ~0 (mean-centered).
        let c_sum: f32 = c_out.iter().sum();
        assert!(
            c_sum.abs() < 1e-4,
            "CohereLayerNorm output should be zero-mean, sum={c_sum}"
        );

        // Sum of RmsNorm output should NOT be ~0.
        let r_sum: f32 = r_out.iter().sum();
        assert!(
            r_sum.abs() > 0.1,
            "RmsNorm output should not be zero-mean, sum={r_sum}"
        );
    }

    #[test]
    fn test_cohere_layernorm_weight_scaling() {
        // weight = [2, 2, 2, 2], input = [1, -1, 1, -1]
        // mean = 0, var = 1
        // output = 2 * [1, -1, 1, -1] / sqrt(1+eps) ≈ [2, -2, 2, -2]
        let weight = Tensor::new(&[2.0f32, 2.0, 2.0, 2.0], &Device::Cpu).unwrap();
        let norm = CohereLayerNorm::new(weight, 1e-6);

        let x = Tensor::new(&[[1.0f32, -1.0, 1.0, -1.0]], &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap();
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[0] - 2.0).abs() < 1e-4);
        assert!((vals[1] - (-2.0)).abs() < 1e-4);
    }

    #[test]
    fn test_cohere_layernorm_f16() {
        let weight = Tensor::ones(&[4], DType::F16, &Device::Cpu).unwrap();
        let norm = CohereLayerNorm::new(weight, 1e-5);

        let x = Tensor::new(&[[1.0f32, -1.0, 1.0, -1.0]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.dtype(), DType::F16);
        assert_eq!(y.dims(), &[1, 4]);
    }

    #[test]
    fn test_cohere_layernorm_batch() {
        let norm = CohereLayerNorm::ones(3, 1e-5, DType::F32, &Device::Cpu).unwrap();
        let x = Tensor::ones(&[4, 3], DType::F32, &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.dims(), &[4, 3]);
    }
}
