// SPDX-License-Identifier: Apache-2.0
//! Normalization kernels.
//!
//! Trait abstraction for RMS norm and fused add-RMS norm kernels.
//! Port of: `csrc/layernorm_kernels.cu`
//!
//! ## Simplifications vs Python vLLM
//!
//! The CUDA kernels here are simplified compared to Python vLLM's:
//! - Scalar loads instead of vectorized `vec_n_t<T, VEC_SIZE>` loads (~2-3x slower on large hidden)
//! - Simple warp shuffle reduction instead of CUB `BlockReduce`
//! - 2D only (no 3D/4D for per-head QK-norm)
//! - `fused_add_rms_norm` not yet wired to CUDA (uses add + separate norm)
//!
//! These will be upgraded to match Python vLLM's performance in a follow-up.

use candle_core::Tensor;

use crate::error::KernelResult;

/// Normalization kernel interface.
///
/// Abstracts the CUDA RMS norm and fused add-RMS norm kernels.
pub trait NormKernels: Send + Sync {
    /// RMS normalization: `out = input / rms(input) * weight`
    ///
    /// Port of: `void rms_norm(out, input, weight, epsilon)`
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, epsilon: f64) -> KernelResult<Tensor>;

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
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, epsilon: f64) -> KernelResult<Tensor> {
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
// CUDA implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda_ffi {
    unsafe extern "C" {
        pub fn rms_norm_f32(
            out: *mut f32,
            input: *const f32,
            weight: *const f32,
            epsilon: f32,
            num_tokens: i32,
            hidden_size: i32,
        );
        pub fn rms_norm_f16(
            out: *mut u16,
            input: *const u16,
            weight: *const u16,
            epsilon: f32,
            num_tokens: i32,
            hidden_size: i32,
        );
        pub fn rms_norm_bf16(
            out: *mut u16,
            input: *const u16,
            weight: *const u16,
            epsilon: f32,
            num_tokens: i32,
            hidden_size: i32,
        );
    }
}

/// CUDA implementation of normalization kernels.
#[cfg(feature = "cuda")]
pub struct CudaNormKernels;

#[cfg(feature = "cuda")]
impl CudaNormKernels {
    /// Extract a raw device pointer (as usize) from a contiguous CUDA tensor.
    ///
    /// Uses `DevicePtr::device_ptr()` which requires the CUDA stream to
    /// synchronize outstanding writes. This is correct for kernel launch —
    /// kernels on the same stream see prior writes.
    fn device_ptr_of<T: cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType>(
        tensor: &Tensor,
    ) -> KernelResult<usize> {
        use cudarc::driver::DevicePtr;
        let cuda_dev = tensor
            .device()
            .as_cuda_device()
            .map_err(|e| crate::error::KernelError::Other(format!("{e}")))?;
        let stream = cuda_dev.cuda_stream();
        let (storage, layout) = tensor.storage_and_layout();
        match &*storage {
            candle_core::Storage::Cuda(cuda_storage) => {
                let slice = cuda_storage.as_cuda_slice::<T>()?;
                let view = slice.slice(layout.start_offset()..);
                let (ptr, _sync_guard) = view.device_ptr(&stream);
                Ok(ptr as usize)
            }
            _ => Err(crate::error::KernelError::Other(
                "expected CUDA tensor".to_string(),
            )),
        }
    }
}

#[cfg(feature = "cuda")]
impl NormKernels for CudaNormKernels {
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, epsilon: f64) -> KernelResult<Tensor> {
        use candle_core::DType;

        let dims = input.shape().dims();
        let hidden_size = *dims
            .last()
            .ok_or_else(|| crate::error::KernelError::Other("empty input tensor".to_string()))?;
        let num_tokens: usize = dims[..dims.len() - 1].iter().product();

        let input = input.contiguous()?;
        let weight = weight.contiguous()?;
        let out = Tensor::zeros(input.shape(), input.dtype(), input.device())?;

        match input.dtype() {
            DType::F32 => {
                let o = Self::device_ptr_of::<f32>(&out)?;
                let i = Self::device_ptr_of::<f32>(&input)?;
                let w = Self::device_ptr_of::<f32>(&weight)?;
                unsafe {
                    cuda_ffi::rms_norm_f32(
                        o as *mut f32,
                        i as *const f32,
                        w as *const f32,
                        epsilon as f32,
                        num_tokens as i32,
                        hidden_size as i32,
                    );
                }
            }
            DType::F16 => {
                let o = Self::device_ptr_of::<half::f16>(&out)?;
                let i = Self::device_ptr_of::<half::f16>(&input)?;
                let w = Self::device_ptr_of::<half::f16>(&weight)?;
                unsafe {
                    cuda_ffi::rms_norm_f16(
                        o as *mut u16,
                        i as *const u16,
                        w as *const u16,
                        epsilon as f32,
                        num_tokens as i32,
                        hidden_size as i32,
                    );
                }
            }
            DType::BF16 => {
                let o = Self::device_ptr_of::<half::bf16>(&out)?;
                let i = Self::device_ptr_of::<half::bf16>(&input)?;
                let w = Self::device_ptr_of::<half::bf16>(&weight)?;
                unsafe {
                    cuda_ffi::rms_norm_bf16(
                        o as *mut u16,
                        i as *const u16,
                        w as *const u16,
                        epsilon as f32,
                        num_tokens as i32,
                        hidden_size as i32,
                    );
                }
            }
            _ => return CpuNormKernels.rms_norm(&input, &weight, epsilon),
        }
        Ok(out)
    }

    fn fused_add_rms_norm(
        &self,
        input: &Tensor,
        residual: &Tensor,
        weight: &Tensor,
        epsilon: f64,
    ) -> KernelResult<(Tensor, Tensor)> {
        // TODO: wire fused_add_rms_norm CUDA kernel for in-place residual update.
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

    // -----------------------------------------------------------------------
    // CUDA tests — compare CUDA kernel output against CPU reference
    // -----------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    fn cuda_device() -> Device {
        Device::new_cuda(0).expect("CUDA device required for this test")
    }

    /// Helper: run rms_norm on CPU and CUDA, compare results.
    #[cfg(feature = "cuda")]
    fn assert_rms_norm_cuda_matches_cpu(shape: &[usize], dtype: DType, tol: f64) {
        use super::CudaNormKernels;

        let input_cpu = Tensor::randn(0f32, 1.0, shape, &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let hidden = *shape.last().unwrap();
        let weight_cpu = Tensor::randn(0f32, 1.0, &[hidden], &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let eps = 1e-5;

        // CPU reference (computed in f32 for accuracy).
        let ref_out = CpuNormKernels
            .rms_norm(
                &input_cpu.to_dtype(DType::F32).unwrap(),
                &weight_cpu.to_dtype(DType::F32).unwrap(),
                eps,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        // CUDA kernel.
        let dev = cuda_device();
        let input_gpu = input_cpu.to_device(&dev).unwrap();
        let weight_gpu = weight_cpu.to_device(&dev).unwrap();
        let cuda_out = CudaNormKernels
            .rms_norm(&input_gpu, &weight_gpu, eps)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();

        // Compare.
        let ref_vals = ref_out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cuda_vals = cuda_out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(ref_vals.len(), cuda_vals.len());
        for (i, (r, c)) in ref_vals.iter().zip(cuda_vals.iter()).enumerate() {
            assert!(
                (r - c).abs() as f64 <= tol,
                "rms_norm {:?} mismatch at {}: cpu={} cuda={}",
                dtype,
                i,
                r,
                c
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rms_norm_f32() {
        assert_rms_norm_cuda_matches_cpu(&[4, 128], DType::F32, 1e-4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rms_norm_f16() {
        assert_rms_norm_cuda_matches_cpu(&[4, 128], DType::F16, 5e-2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rms_norm_bf16() {
        assert_rms_norm_cuda_matches_cpu(&[4, 128], DType::BF16, 5e-2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rms_norm_large_hidden() {
        // Typical model hidden size (e.g., Qwen2.5-0.5B uses 896).
        assert_rms_norm_cuda_matches_cpu(&[8, 896], DType::F32, 1e-4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rms_norm_3d() {
        // 3D input: [batch, seq, hidden].
        assert_rms_norm_cuda_matches_cpu(&[2, 4, 128], DType::F32, 1e-4);
    }
}
