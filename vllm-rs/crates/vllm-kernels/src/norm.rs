// SPDX-License-Identifier: Apache-2.0
//! Normalization kernels.
//!
//! Trait abstraction for RMS norm and fused add-RMS norm kernels.
//! Port of: `csrc/layernorm_kernels.cu`
//!
//! ## Simplifications vs Python vLLM
//!
//! The CUDA kernels here are simplified compared to Python vLLM's:
//! - Vectorized loads for improved throughput on large hidden sizes
//! - Simple warp shuffle reduction instead of CUB `BlockReduce`
//! - 2D only (no 3D/4D for per-head QK-norm)

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

        // Fused add + RMS norm kernels
        pub fn fused_add_rms_norm_f32(
            input: *mut f32,
            residual: *mut f32,
            weight: *const f32,
            epsilon: f32,
            num_tokens: i32,
            hidden_size: i32,
        );
        pub fn fused_add_rms_norm_f16(
            input: *mut u16,
            residual: *mut u16,
            weight: *const u16,
            epsilon: f32,
            num_tokens: i32,
            hidden_size: i32,
        );
        pub fn fused_add_rms_norm_bf16(
            input: *mut u16,
            residual: *mut u16,
            weight: *const u16,
            epsilon: f32,
            num_tokens: i32,
            hidden_size: i32,
        );

        // Fused QK-norm + RoPE kernels
        pub fn qk_norm_rope_f32(
            query: *mut f32,
            key: *mut f32,
            q_weight: *const f32,
            k_weight: *const f32,
            cos_cache: *const f32,
            sin_cache: *const f32,
            positions: *const u32,
            epsilon: f32,
            num_q_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            num_tokens: i32,
        );
        pub fn qk_norm_rope_f16(
            query: *mut u16,
            key: *mut u16,
            q_weight: *const u16,
            k_weight: *const u16,
            cos_cache: *const u16,
            sin_cache: *const u16,
            positions: *const u32,
            epsilon: f32,
            num_q_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            num_tokens: i32,
        );
        pub fn qk_norm_rope_bf16(
            query: *mut u16,
            key: *mut u16,
            q_weight: *const u16,
            k_weight: *const u16,
            cos_cache: *const u16,
            sin_cache: *const u16,
            positions: *const u32,
            epsilon: f32,
            num_q_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            num_tokens: i32,
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
impl CudaNormKernels {
    /// Fused per-head QK RMS normalization + NeoX RoPE rotation.
    ///
    /// Applies RMS norm (with given weight and epsilon) to each head of Q and K,
    /// then applies RoPE rotation using cos/sin caches, all in a single CUDA kernel.
    ///
    /// The kernel modifies cloned copies (out-of-place semantics).
    ///
    /// * `query` — `[num_tokens, num_q_heads, head_dim]`
    /// * `key` — `[num_tokens, num_kv_heads, head_dim]`
    /// * `q_weight`, `k_weight` — `[head_dim]` (effective weight, already +1 for Gemma)
    /// * `cos_cache`, `sin_cache` — `[max_pos, head_dim]` (full cache, gathered by position internally)
    /// * `positions` — `[num_tokens]` (u32)
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_and_rope(
        &self,
        query: &Tensor,
        key: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        epsilon: f64,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        positions: &Tensor,
    ) -> KernelResult<(Tensor, Tensor)> {
        use candle_core::DType;

        let q_dims = query.shape().dims();
        let k_dims = key.shape().dims();
        assert_eq!(q_dims.len(), 3, "query must be [tokens, q_heads, head_dim]");
        assert_eq!(k_dims.len(), 3, "key must be [tokens, kv_heads, head_dim]");

        let num_tokens = q_dims[0];
        let num_q_heads = q_dims[1];
        let head_dim = q_dims[2];
        let num_kv_heads = k_dims[1];

        // Clone for out-of-place semantics, then make contiguous.
        let q_out = query.clone().contiguous()?;
        let k_out = key.clone().contiguous()?;
        let q_weight = q_weight.contiguous()?;
        let k_weight = k_weight.contiguous()?;
        let cos_cache = cos_cache.contiguous()?;
        let sin_cache = sin_cache.contiguous()?;
        let positions = positions.contiguous()?;

        match query.dtype() {
            DType::F32 => {
                let q = Self::device_ptr_of::<f32>(&q_out)?;
                let k = Self::device_ptr_of::<f32>(&k_out)?;
                let qw = Self::device_ptr_of::<f32>(&q_weight)?;
                let kw = Self::device_ptr_of::<f32>(&k_weight)?;
                let cos = Self::device_ptr_of::<f32>(&cos_cache)?;
                let sin = Self::device_ptr_of::<f32>(&sin_cache)?;
                let pos = Self::device_ptr_of::<u32>(&positions)?;
                unsafe {
                    cuda_ffi::qk_norm_rope_f32(
                        q as *mut f32,
                        k as *mut f32,
                        qw as *const f32,
                        kw as *const f32,
                        cos as *const f32,
                        sin as *const f32,
                        pos as *const u32,
                        epsilon as f32,
                        num_q_heads as i32,
                        num_kv_heads as i32,
                        head_dim as i32,
                        num_tokens as i32,
                    );
                }
            }
            DType::F16 => {
                let q = Self::device_ptr_of::<half::f16>(&q_out)?;
                let k = Self::device_ptr_of::<half::f16>(&k_out)?;
                let qw = Self::device_ptr_of::<half::f16>(&q_weight)?;
                let kw = Self::device_ptr_of::<half::f16>(&k_weight)?;
                let cos = Self::device_ptr_of::<half::f16>(&cos_cache)?;
                let sin = Self::device_ptr_of::<half::f16>(&sin_cache)?;
                let pos = Self::device_ptr_of::<u32>(&positions)?;
                unsafe {
                    cuda_ffi::qk_norm_rope_f16(
                        q as *mut u16,
                        k as *mut u16,
                        qw as *const u16,
                        kw as *const u16,
                        cos as *const u16,
                        sin as *const u16,
                        pos as *const u32,
                        epsilon as f32,
                        num_q_heads as i32,
                        num_kv_heads as i32,
                        head_dim as i32,
                        num_tokens as i32,
                    );
                }
            }
            DType::BF16 => {
                let q = Self::device_ptr_of::<half::bf16>(&q_out)?;
                let k = Self::device_ptr_of::<half::bf16>(&k_out)?;
                let qw = Self::device_ptr_of::<half::bf16>(&q_weight)?;
                let kw = Self::device_ptr_of::<half::bf16>(&k_weight)?;
                let cos = Self::device_ptr_of::<half::bf16>(&cos_cache)?;
                let sin = Self::device_ptr_of::<half::bf16>(&sin_cache)?;
                let pos = Self::device_ptr_of::<u32>(&positions)?;
                unsafe {
                    cuda_ffi::qk_norm_rope_bf16(
                        q as *mut u16,
                        k as *mut u16,
                        qw as *const u16,
                        kw as *const u16,
                        cos as *const u16,
                        sin as *const u16,
                        pos as *const u32,
                        epsilon as f32,
                        num_q_heads as i32,
                        num_kv_heads as i32,
                        head_dim as i32,
                        num_tokens as i32,
                    );
                }
            }
            _ => {
                return Err(crate::error::KernelError::Other(format!(
                    "qk_norm_rope: unsupported dtype {:?}",
                    query.dtype()
                )));
            }
        }
        Ok((q_out, k_out))
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
        use candle_core::DType;

        let dims = input.shape().dims();
        let hidden_size = *dims
            .last()
            .ok_or_else(|| crate::error::KernelError::Other("empty input tensor".to_string()))?;
        let num_tokens: usize = dims[..dims.len() - 1].iter().product();

        // Clone for out-of-place semantics, then make contiguous.
        let inp = input.clone().contiguous()?;
        let res = residual.clone().contiguous()?;
        let weight = weight.contiguous()?;

        match input.dtype() {
            DType::F32 => {
                let i = Self::device_ptr_of::<f32>(&inp)?;
                let r = Self::device_ptr_of::<f32>(&res)?;
                let w = Self::device_ptr_of::<f32>(&weight)?;
                unsafe {
                    cuda_ffi::fused_add_rms_norm_f32(
                        i as *mut f32,
                        r as *mut f32,
                        w as *const f32,
                        epsilon as f32,
                        num_tokens as i32,
                        hidden_size as i32,
                    );
                }
            }
            DType::F16 => {
                let i = Self::device_ptr_of::<half::f16>(&inp)?;
                let r = Self::device_ptr_of::<half::f16>(&res)?;
                let w = Self::device_ptr_of::<half::f16>(&weight)?;
                unsafe {
                    cuda_ffi::fused_add_rms_norm_f16(
                        i as *mut u16,
                        r as *mut u16,
                        w as *const u16,
                        epsilon as f32,
                        num_tokens as i32,
                        hidden_size as i32,
                    );
                }
            }
            DType::BF16 => {
                let i = Self::device_ptr_of::<half::bf16>(&inp)?;
                let r = Self::device_ptr_of::<half::bf16>(&res)?;
                let w = Self::device_ptr_of::<half::bf16>(&weight)?;
                unsafe {
                    cuda_ffi::fused_add_rms_norm_bf16(
                        i as *mut u16,
                        r as *mut u16,
                        w as *const u16,
                        epsilon as f32,
                        num_tokens as i32,
                        hidden_size as i32,
                    );
                }
            }
            _ => return CpuNormKernels.fused_add_rms_norm(input, residual, &weight, epsilon),
        }
        // inp now contains the normalized result, res contains the updated residual.
        Ok((inp, res))
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

    // -----------------------------------------------------------------------
    // CUDA tests — fused add + RMS norm
    // -----------------------------------------------------------------------

    /// Helper: run fused_add_rms_norm on CPU and CUDA, compare results.
    #[cfg(feature = "cuda")]
    fn assert_fused_add_rms_norm_cuda_matches_cpu(shape: &[usize], dtype: DType, tol: f64) {
        use super::CudaNormKernels;

        let input_cpu = Tensor::randn(0f32, 1.0, shape, &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let residual_cpu = Tensor::randn(0f32, 1.0, shape, &Device::Cpu)
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
        let (ref_normed, ref_updated) = CpuNormKernels
            .fused_add_rms_norm(
                &input_cpu.to_dtype(DType::F32).unwrap(),
                &residual_cpu.to_dtype(DType::F32).unwrap(),
                &weight_cpu.to_dtype(DType::F32).unwrap(),
                eps,
            )
            .unwrap();
        let ref_normed = ref_normed.to_dtype(dtype).unwrap();
        let ref_updated = ref_updated.to_dtype(dtype).unwrap();

        // CUDA kernel.
        let dev = cuda_device();
        let input_gpu = input_cpu.to_device(&dev).unwrap();
        let residual_gpu = residual_cpu.to_device(&dev).unwrap();
        let weight_gpu = weight_cpu.to_device(&dev).unwrap();
        let (cuda_normed, cuda_updated) = CudaNormKernels
            .fused_add_rms_norm(&input_gpu, &residual_gpu, &weight_gpu, eps)
            .unwrap();
        let cuda_normed = cuda_normed.to_device(&Device::Cpu).unwrap();
        let cuda_updated = cuda_updated.to_device(&Device::Cpu).unwrap();

        // Compare normed output.
        let ref_vals = ref_normed
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cuda_vals = cuda_normed
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
                "fused_add_rms_norm normed {:?} mismatch at {}: cpu={} cuda={}",
                dtype,
                i,
                r,
                c
            );
        }

        // Compare updated residual.
        let ref_upd = ref_updated
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cuda_upd = cuda_updated
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(ref_upd.len(), cuda_upd.len());
        for (i, (r, c)) in ref_upd.iter().zip(cuda_upd.iter()).enumerate() {
            assert!(
                (r - c).abs() as f64 <= tol,
                "fused_add_rms_norm residual {:?} mismatch at {}: cpu={} cuda={}",
                dtype,
                i,
                r,
                c
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_fused_add_rms_norm_f32() {
        assert_fused_add_rms_norm_cuda_matches_cpu(&[4, 128], DType::F32, 1e-4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_fused_add_rms_norm_f16() {
        assert_fused_add_rms_norm_cuda_matches_cpu(&[4, 128], DType::F16, 5e-2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_fused_add_rms_norm_bf16() {
        assert_fused_add_rms_norm_cuda_matches_cpu(&[4, 128], DType::BF16, 5e-2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_fused_add_rms_norm_large_hidden() {
        assert_fused_add_rms_norm_cuda_matches_cpu(&[8, 896], DType::F32, 1e-4);
    }

    // -----------------------------------------------------------------------
    // CUDA tests — fused QK-norm + RoPE
    // -----------------------------------------------------------------------

    /// CPU reference for NeoX RoPE on a 3D tensor [tokens, heads, head_dim].
    /// cos/sin: [tokens, head_dim] (pre-gathered by position).
    #[cfg(feature = "cuda")]
    fn cpu_rope_3d(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let half = x.dim(candle_core::D::Minus1).unwrap() / 2;
        let x1 = x.narrow(candle_core::D::Minus1, 0, half).unwrap();
        let x2 = x.narrow(candle_core::D::Minus1, half, half).unwrap();
        let neg_x2 = x2.neg().unwrap();
        let x_rot = Tensor::cat(&[&neg_x2, &x1], candle_core::D::Minus1).unwrap();
        let cos_b = cos.unsqueeze(1).unwrap(); // [tokens, 1, head_dim]
        let sin_b = sin.unsqueeze(1).unwrap();
        x.broadcast_mul(&cos_b)
            .unwrap()
            .add(&x_rot.broadcast_mul(&sin_b).unwrap())
            .unwrap()
    }

    /// Helper: build cos/sin caches in the same format as RotaryEmbedding.
    /// Returns (cos_cache, sin_cache) each [max_pos, head_dim] in the given dtype.
    #[cfg(feature = "cuda")]
    fn build_cos_sin_cache(
        head_dim: usize,
        max_pos: usize,
        dtype: DType,
        device: &Device,
    ) -> (Tensor, Tensor) {
        let half = head_dim / 2;
        let base = 10000.0f64;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (1.0 / base.powf(2.0 * i as f64 / head_dim as f64)) as f32)
            .collect();
        let inv_freq_t = Tensor::from_slice(&inv_freq, half, device).unwrap();
        let positions: Vec<f32> = (0..max_pos).map(|p| p as f32).collect();
        let pos_t = Tensor::from_slice(&positions, max_pos, device).unwrap();
        let pos_2d = pos_t.reshape((max_pos, 1)).unwrap();
        let inv_2d = inv_freq_t.reshape((1, half)).unwrap();
        let freqs = pos_2d.matmul(&inv_2d).unwrap();
        let freqs_full = Tensor::cat(&[&freqs, &freqs], 1).unwrap();
        let cos = freqs_full.cos().unwrap().to_dtype(dtype).unwrap();
        let sin = freqs_full.sin().unwrap().to_dtype(dtype).unwrap();
        (cos, sin)
    }

    /// Helper: compare fused QK-norm+RoPE on CUDA against CPU reference.
    #[cfg(feature = "cuda")]
    fn assert_qk_norm_rope_cuda_matches_cpu(
        num_tokens: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        tol: f64,
    ) {
        use super::CudaNormKernels;

        let max_pos = 128;
        let eps = 1e-6;

        // Random inputs on CPU.
        let q_cpu = Tensor::randn(
            0f32,
            1.0,
            &[num_tokens, num_q_heads, head_dim],
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
        let k_cpu = Tensor::randn(
            0f32,
            1.0,
            &[num_tokens, num_kv_heads, head_dim],
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
        let qw_cpu = Tensor::randn(0f32, 1.0, &[head_dim], &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let kw_cpu = Tensor::randn(0f32, 1.0, &[head_dim], &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        // Positions (within max_pos range).
        let pos_vals: Vec<u32> = (0..num_tokens as u32).collect();
        let pos_cpu = Tensor::from_slice(&pos_vals, num_tokens, &Device::Cpu).unwrap();

        // Build cos/sin cache in f32 on CPU for reference.
        let (cos_cache_f32, sin_cache_f32) =
            build_cos_sin_cache(head_dim, max_pos, DType::F32, &Device::Cpu);
        // Also in the target dtype for CUDA.
        let (cos_cache_dt, sin_cache_dt) =
            build_cos_sin_cache(head_dim, max_pos, dtype, &Device::Cpu);

        // --- CPU reference (in f32 for accuracy) ---
        let q_f32 = q_cpu.to_dtype(DType::F32).unwrap();
        let k_f32 = k_cpu.to_dtype(DType::F32).unwrap();
        let qw_f32 = qw_cpu.to_dtype(DType::F32).unwrap();
        let kw_f32 = kw_cpu.to_dtype(DType::F32).unwrap();

        let q_normed = CpuNormKernels.rms_norm(&q_f32, &qw_f32, eps).unwrap();
        let k_normed = CpuNormKernels.rms_norm(&k_f32, &kw_f32, eps).unwrap();

        let cos_gathered = cos_cache_f32.index_select(&pos_cpu, 0).unwrap();
        let sin_gathered = sin_cache_f32.index_select(&pos_cpu, 0).unwrap();

        let q_ref = cpu_rope_3d(&q_normed, &cos_gathered, &sin_gathered)
            .to_dtype(dtype)
            .unwrap();
        let k_ref = cpu_rope_3d(&k_normed, &cos_gathered, &sin_gathered)
            .to_dtype(dtype)
            .unwrap();

        // --- CUDA fused kernel ---
        let dev = cuda_device();
        let q_gpu = q_cpu.to_device(&dev).unwrap();
        let k_gpu = k_cpu.to_device(&dev).unwrap();
        let qw_gpu = qw_cpu.to_device(&dev).unwrap();
        let kw_gpu = kw_cpu.to_device(&dev).unwrap();
        let cos_gpu = cos_cache_dt.to_device(&dev).unwrap();
        let sin_gpu = sin_cache_dt.to_device(&dev).unwrap();
        let pos_gpu = pos_cpu.to_device(&dev).unwrap();

        let (q_cuda, k_cuda) = CudaNormKernels
            .qk_norm_and_rope(
                &q_gpu, &k_gpu, &qw_gpu, &kw_gpu, eps, &cos_gpu, &sin_gpu, &pos_gpu,
            )
            .unwrap();

        let q_cuda_cpu = q_cuda.to_device(&Device::Cpu).unwrap();
        let k_cuda_cpu = k_cuda.to_device(&Device::Cpu).unwrap();

        // Compare Q.
        let q_ref_vals = q_ref
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let q_cuda_vals = q_cuda_cpu
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(q_ref_vals.len(), q_cuda_vals.len());
        for (i, (r, c)) in q_ref_vals.iter().zip(q_cuda_vals.iter()).enumerate() {
            assert!(
                (r - c).abs() as f64 <= tol,
                "qk_norm_rope Q {:?} mismatch at {}: cpu={} cuda={}",
                dtype,
                i,
                r,
                c
            );
        }

        // Compare K.
        let k_ref_vals = k_ref
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let k_cuda_vals = k_cuda_cpu
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(k_ref_vals.len(), k_cuda_vals.len());
        for (i, (r, c)) in k_ref_vals.iter().zip(k_cuda_vals.iter()).enumerate() {
            assert!(
                (r - c).abs() as f64 <= tol,
                "qk_norm_rope K {:?} mismatch at {}: cpu={} cuda={}",
                dtype,
                i,
                r,
                c
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_qk_norm_rope_f32() {
        // GQA: 8 Q heads, 2 KV heads, head_dim=128
        assert_qk_norm_rope_cuda_matches_cpu(4, 8, 2, 128, DType::F32, 1e-4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_qk_norm_rope_bf16() {
        assert_qk_norm_rope_cuda_matches_cpu(4, 8, 2, 128, DType::BF16, 5e-2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_qk_norm_rope_head_dim_64() {
        // Smaller head_dim to exercise different block sizes.
        assert_qk_norm_rope_cuda_matches_cpu(4, 4, 4, 64, DType::F32, 1e-4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_qk_norm_rope_head_dim_256() {
        // Gemma3 1B uses head_dim=256.
        assert_qk_norm_rope_cuda_matches_cpu(2, 8, 4, 256, DType::F32, 1e-4);
    }
}
