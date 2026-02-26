// SPDX-License-Identifier: Apache-2.0
//! Normalization kernels.
//!
//! Trait abstraction for RMS norm and fused add-RMS norm kernels.
//! Port of: `csrc/layernorm_kernels.cu`

use candle_core::Tensor;

use crate::error::KernelResult;

/// Normalization kernel interface.
///
/// Abstracts the CUDA RMS norm and fused add-RMS norm kernels.
pub trait NormKernels: Send + Sync {
    /// RMS normalization: `out = input / rms(input) * weight`
    ///
    /// Port of: `void rms_norm(out, input, weight, epsilon)`
    fn rms_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        epsilon: f64,
    ) -> KernelResult<Tensor>;

    /// Fused add + RMS normalization.
    ///
    /// Computes `input = input + residual` in-place, then RMS-normalizes.
    /// Returns `(normalized, updated_residual)`.
    ///
    /// Port of: `void fused_add_rms_norm(input, residual, weight, epsilon)`
    fn fused_add_rms_norm(
        &self,
        input: &Tensor,
        residual: &Tensor,
        weight: &Tensor,
        epsilon: f64,
    ) -> KernelResult<(Tensor, Tensor)>;
}

/// CPU implementation of normalization kernels (for testing).
pub struct CpuNormKernels;

impl NormKernels for CpuNormKernels {
    fn rms_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        epsilon: f64,
    ) -> KernelResult<Tensor> {
        // x^2 -> mean over last dim -> sqrt -> recip -> multiply
        let x_sq = input.sqr()?;
        let variance = x_sq.mean_keepdim(candle_core::D::Minus1)?;
        let rsqrt = (variance + epsilon)?.sqrt()?.recip()?;
        let normed = input.broadcast_mul(&rsqrt)?;
        let out = normed.broadcast_mul(weight)?;
        Ok(out)
    }

    fn fused_add_rms_norm(
        &self,
        input: &Tensor,
        residual: &Tensor,
        weight: &Tensor,
        epsilon: f64,
    ) -> KernelResult<(Tensor, Tensor)> {
        let updated = (input + residual)?;
        let normed = self.rms_norm(&updated, weight, epsilon)?;
        Ok((normed, updated))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_cpu_rms_norm() {
        let kernels = CpuNormKernels;

        let input = Tensor::ones(&[2, 4], DType::F32, &Device::Cpu).unwrap();
        let weight = Tensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();

        let out = kernels.rms_norm(&input, &weight, 1e-5).unwrap();
        assert_eq!(out.dims(), &[2, 4]);

        // All ones: RMS = 1, so output = 1 * 1 = 1
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in vals {
            assert!((v - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_cpu_fused_add_rms_norm() {
        let kernels = CpuNormKernels;

        let input = Tensor::ones(&[1, 4], DType::F32, &Device::Cpu).unwrap();
        let residual = Tensor::ones(&[1, 4], DType::F32, &Device::Cpu).unwrap();
        let weight = Tensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();

        let (normed, updated) = kernels
            .fused_add_rms_norm(&input, &residual, &weight, 1e-5)
            .unwrap();
        assert_eq!(normed.dims(), &[1, 4]);
        assert_eq!(updated.dims(), &[1, 4]);

        // updated = 1 + 1 = 2, RMS(2,2,2,2) = 2, normed = 2/2 * 1 = 1
        let normed_vals = normed.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in normed_vals {
            assert!((v - 1.0).abs() < 1e-4);
        }

        let updated_vals = updated.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in updated_vals {
            assert!((v - 2.0).abs() < 1e-4);
        }
    }
}
