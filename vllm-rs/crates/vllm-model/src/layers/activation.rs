// SPDX-License-Identifier: Apache-2.0
//! Activation functions.
//!
//! Port of: `vllm/model_executor/layers/activation.py`

use candle_core::{Module, Tensor};

// ---------------------------------------------------------------------------
// Activation enum
// ---------------------------------------------------------------------------

/// Common activation functions used in transformer models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Sigmoid Linear Unit: x * sigmoid(x)
    Silu,
    /// Gaussian Error Linear Unit (approximate, tanh-based)
    Gelu,
    /// GELU with exact erf computation
    GeluErf,
    /// Rectified Linear Unit: max(0, x)
    Relu,
    /// Quick GELU: x * sigmoid(1.702 * x) — used by some models
    QuickGelu,
}

impl Module for Activation {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Activation::Silu => silu(x),
            Activation::Gelu => gelu(x),
            Activation::GeluErf => gelu_erf(x),
            Activation::Relu => relu(x),
            Activation::QuickGelu => quick_gelu(x),
        }
    }
}

// ---------------------------------------------------------------------------
// Individual activation functions
// ---------------------------------------------------------------------------

/// SiLU (Swish): x * sigmoid(x)
///
/// Used by LLaMA, Mistral, Qwen, and most modern LLMs.
pub fn silu(x: &Tensor) -> candle_core::Result<Tensor> {
    x.silu()
}

/// GELU (Gaussian Error Linear Unit) with tanh approximation.
///
/// Used by GPT-2, BERT, etc.
pub fn gelu(x: &Tensor) -> candle_core::Result<Tensor> {
    x.gelu()
}

/// GELU with exact erf computation.
pub fn gelu_erf(x: &Tensor) -> candle_core::Result<Tensor> {
    x.gelu_erf()
}

/// ReLU: max(0, x)
pub fn relu(x: &Tensor) -> candle_core::Result<Tensor> {
    x.relu()
}

/// Quick GELU: x * sigmoid(1.702 * x)
///
/// Used by some Phi and CLIP models.
/// sigmoid(z) = 1 / (1 + exp(-z))
pub fn quick_gelu(x: &Tensor) -> candle_core::Result<Tensor> {
    // sigmoid(1.702 * x) = 1 / (1 + exp(-1.702 * x))
    let neg_scaled = (x * (-1.702))?;
    let exp_neg = neg_scaled.exp()?;
    let one_plus_exp = (exp_neg + 1.0)?;
    let sigmoid = one_plus_exp.recip()?;
    x.mul(&sigmoid)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_silu() {
        // silu(0) = 0, silu(large) ~ large
        let x = Tensor::new(&[0.0f32, 1.0, -1.0, 5.0], &Device::Cpu).unwrap();
        let y = silu(&x).unwrap();
        let vals = y.to_vec1::<f32>().unwrap();
        assert!(vals[0].abs() < 1e-6); // silu(0) = 0
        assert!((vals[1] - 0.7311).abs() < 0.01); // silu(1) ~ 0.7311
        assert!((vals[2] - (-0.2689)).abs() < 0.01); // silu(-1) ~ -0.2689
        assert!((vals[3] - 4.966).abs() < 0.01); // silu(5) ~ 4.966
    }

    #[test]
    fn test_relu() {
        let x = Tensor::new(&[-1.0f32, 0.0, 1.0, 5.0], &Device::Cpu).unwrap();
        let y = relu(&x).unwrap();
        let vals = y.to_vec1::<f32>().unwrap();
        assert_eq!(vals[0], 0.0);
        assert_eq!(vals[1], 0.0);
        assert_eq!(vals[2], 1.0);
        assert_eq!(vals[3], 5.0);
    }

    #[test]
    fn test_gelu() {
        let x = Tensor::new(&[0.0f32, 1.0, -1.0], &Device::Cpu).unwrap();
        let y = gelu_erf(&x).unwrap();
        let vals = y.to_vec1::<f32>().unwrap();
        assert!(vals[0].abs() < 1e-6); // gelu(0) = 0
        assert!((vals[1] - 0.8413).abs() < 0.01); // gelu(1) ~ 0.8413
        assert!((vals[2] - (-0.1587)).abs() < 0.01); // gelu(-1) ~ -0.1587
    }

    #[test]
    fn test_quick_gelu() {
        let x = Tensor::new(&[0.0f32, 1.0], &Device::Cpu).unwrap();
        let y = quick_gelu(&x).unwrap();
        let vals = y.to_vec1::<f32>().unwrap();
        assert!(vals[0].abs() < 1e-6); // quick_gelu(0) = 0
        assert!(vals[1] > 0.5 && vals[1] < 1.0); // positive value
    }

    #[test]
    fn test_activation_enum() {
        let x = Tensor::new(&[1.0f32, 2.0], &Device::Cpu).unwrap();

        let y_silu = Activation::Silu.forward(&x).unwrap();
        let y_relu = Activation::Relu.forward(&x).unwrap();
        let y_gelu = Activation::Gelu.forward(&x).unwrap();

        assert_eq!(y_silu.dims(), &[2]);
        assert_eq!(y_relu.dims(), &[2]);
        assert_eq!(y_gelu.dims(), &[2]);
    }

    #[test]
    fn test_activation_batch() {
        let x = Tensor::ones(&[4, 8], DType::F32, &Device::Cpu).unwrap();
        let y = Activation::Silu.forward(&x).unwrap();
        assert_eq!(y.dims(), &[4, 8]);
    }
}
