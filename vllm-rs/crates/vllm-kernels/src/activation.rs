// SPDX-License-Identifier: Apache-2.0
//! Activation kernels.
//!
//! Trait abstraction for fused activation kernels (SiLU+mul, GELU+mul).
//! Port of: `csrc/activation_kernels.cu`

use candle_core::Tensor;

use crate::error::KernelResult;

/// Activation kernel interface.
///
/// Provides fused activation+multiply operations that are common in
/// transformer FFN blocks (gate projection * up projection).
pub trait ActivationKernels: Send + Sync {
    /// Fused SiLU and element-wise multiply.
    ///
    /// Computes `silu(gate) * up` where gate and up are the two halves
    /// of the input tensor split along the last dimension.
    ///
    /// Port of: `void silu_and_mul(out, input)` where input is [batch, 2*dim]
    fn silu_and_mul(&self, gate: &Tensor, up: &Tensor) -> KernelResult<Tensor>;

    /// Fused GELU (tanh approx) and element-wise multiply.
    fn gelu_and_mul(&self, gate: &Tensor, up: &Tensor) -> KernelResult<Tensor>;

    /// Fused GELU (new/exact) and element-wise multiply.
    fn gelu_new_and_mul(&self, gate: &Tensor, up: &Tensor) -> KernelResult<Tensor>;
}

/// CPU implementation of activation kernels (for testing).
pub struct CpuActivationKernels;

impl ActivationKernels for CpuActivationKernels {
    fn silu_and_mul(&self, gate: &Tensor, up: &Tensor) -> KernelResult<Tensor> {
        let activated = gate.silu()?;
        let out = activated.mul(up)?;
        Ok(out)
    }

    fn gelu_and_mul(&self, gate: &Tensor, up: &Tensor) -> KernelResult<Tensor> {
        let activated = gate.gelu()?;
        let out = activated.mul(up)?;
        Ok(out)
    }

    fn gelu_new_and_mul(&self, gate: &Tensor, up: &Tensor) -> KernelResult<Tensor> {
        let activated = gate.gelu_erf()?;
        let out = activated.mul(up)?;
        Ok(out)
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
    fn test_cpu_silu_and_mul() {
        let kernels = CpuActivationKernels;

        let gate = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap();
        let up = Tensor::ones(&[2, 2], DType::F32, &Device::Cpu).unwrap();

        let out = kernels.silu_and_mul(&gate, &up).unwrap();
        assert_eq!(out.dims(), &[2, 2]);

        // silu(1) * 1 ~ 0.7311
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[0] - 0.7311).abs() < 0.01);
    }

    #[test]
    fn test_cpu_gelu_and_mul() {
        let kernels = CpuActivationKernels;

        let gate = Tensor::new(&[[1.0f32, -1.0]], &Device::Cpu).unwrap();
        let up = Tensor::new(&[[2.0f32, 2.0]], &Device::Cpu).unwrap();

        let out = kernels.gelu_and_mul(&gate, &up).unwrap();
        assert_eq!(out.dims(), &[1, 2]);

        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // gelu(1) * 2 ~ 0.841 * 2 = 1.682
        assert!(vals[0] > 1.0);
    }

    #[test]
    fn test_cpu_gelu_new_and_mul() {
        let kernels = CpuActivationKernels;

        let gate = Tensor::new(&[[0.0f32, 1.0]], &Device::Cpu).unwrap();
        let up = Tensor::ones(&[1, 2], DType::F32, &Device::Cpu).unwrap();

        let out = kernels.gelu_new_and_mul(&gate, &up).unwrap();
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(vals[0].abs() < 1e-6); // gelu(0) = 0
        assert!((vals[1] - 0.8413).abs() < 0.01); // gelu_erf(1) ~ 0.8413
    }
}
