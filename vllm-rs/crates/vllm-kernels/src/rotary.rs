// SPDX-License-Identifier: Apache-2.0
//! Rotary embedding kernels.
//!
//! Trait abstraction for rotary position embedding (RoPE) kernels.
//! Port of: `csrc/pos_encoding_kernels.cu`
//!
//! ## Simplifications vs Python vLLM
//!
//! The CUDA kernels here are simplified compared to Python vLLM's:
//! - Scalar loads instead of vectorized loads
//! - NeoX-style only (no GPT-J interleaved mode)
//! - No packed half2 arithmetic
//!
//! These will be upgraded to match Python vLLM's performance in a follow-up.

use candle_core::Tensor;

use crate::error::KernelResult;

/// Rotary embedding kernel interface.
pub trait RotaryKernels: Send + Sync {
    /// Apply rotary embedding to query and key tensors.
    ///
    /// * `positions` — position indices [batch] or [seq_len]
    /// * `query` — query tensor [num_tokens, num_heads * head_dim]
    /// * `key` — key tensor [num_tokens, num_kv_heads * head_dim]
    /// * `cos_sin_cache` — precomputed [max_pos, rotary_dim]
    /// * `is_neox` — whether to use NeoX-style rotation (split in half)
    ///
    /// Returns (rotated_query, rotated_key).
    ///
    /// Port of: `void rotary_embedding(positions, query, key, head_size,
    ///           cos_sin_cache, is_neox)`
    fn rotary_embedding(
        &self,
        positions: &Tensor,
        query: &Tensor,
        key: &Tensor,
        cos_sin_cache: &Tensor,
        is_neox: bool,
    ) -> KernelResult<(Tensor, Tensor)>;
}

/// CPU implementation of rotary kernels (for testing).
pub struct CpuRotaryKernels;

impl RotaryKernels for CpuRotaryKernels {
    fn rotary_embedding(
        &self,
        positions: &Tensor,
        query: &Tensor,
        key: &Tensor,
        cos_sin_cache: &Tensor,
        _is_neox: bool,
    ) -> KernelResult<(Tensor, Tensor)> {
        // Gather cos/sin for the given positions.
        // cos_sin_cache shape: [max_pos, rotary_dim] where first half is cos, second half is sin.
        let rotary_dim = cos_sin_cache.dim(1)?;
        let half = rotary_dim / 2;

        let gathered = cos_sin_cache.index_select(positions, 0)?; // [num_tokens, rotary_dim]
        let cos = gathered.narrow(1, 0, half)?; // [num_tokens, half]
        let sin = gathered.narrow(1, half, half)?; // [num_tokens, half]

        let q_rot = apply_rotary_1d(query, &cos, &sin, half)?;
        let k_rot = apply_rotary_1d(key, &cos, &sin, half)?;

        Ok((q_rot, k_rot))
    }
}

/// Apply rotary to a flat [num_tokens, dim] tensor.
/// Only rotates the first `2 * half` dimensions, leaving the rest unchanged.
fn apply_rotary_1d(x: &Tensor, cos: &Tensor, sin: &Tensor, half: usize) -> KernelResult<Tensor> {
    let dim = x.dim(1)?;
    let rot_dim = 2 * half;

    if rot_dim > dim {
        return Err(crate::error::KernelError::Shape(format!(
            "rotary dim {} > tensor dim {}",
            rot_dim, dim
        )));
    }

    let x_rot = x.narrow(1, 0, rot_dim)?;
    let x1 = x_rot.narrow(1, 0, half)?;
    let x2 = x_rot.narrow(1, half, half)?;

    // Rotate: [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
    let r1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
    let r2 = (x1.broadcast_mul(sin)? + x2.broadcast_mul(cos)?)?;
    let rotated = Tensor::cat(&[&r1, &r2], 1)?;

    if rot_dim < dim {
        let pass_through = x.narrow(1, rot_dim, dim - rot_dim)?;
        let result = Tensor::cat(&[&rotated, &pass_through], 1)?;
        Ok(result)
    } else {
        Ok(rotated)
    }
}

// ---------------------------------------------------------------------------
// CUDA implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda_ffi {
    unsafe extern "C" {
        pub fn rotary_embedding_f32(
            positions: *const u32,
            query: *mut f32,
            key: *mut f32,
            cos_sin_cache: *const f32,
            rotary_dim: i32,
            total_q_dim: i32,
            total_k_dim: i32,
            head_size: i32,
            num_tokens: i32,
        );
        pub fn rotary_embedding_f16(
            positions: *const u32,
            query: *mut u16,
            key: *mut u16,
            cos_sin_cache: *const u16,
            rotary_dim: i32,
            total_q_dim: i32,
            total_k_dim: i32,
            head_size: i32,
            num_tokens: i32,
        );
        pub fn rotary_embedding_bf16(
            positions: *const u32,
            query: *mut u16,
            key: *mut u16,
            cos_sin_cache: *const u16,
            rotary_dim: i32,
            total_q_dim: i32,
            total_k_dim: i32,
            head_size: i32,
            num_tokens: i32,
        );
    }
}

/// CUDA implementation of rotary kernels.
#[cfg(feature = "cuda")]
pub struct CudaRotaryKernels;

#[cfg(feature = "cuda")]
impl CudaRotaryKernels {
    /// Extract a raw device pointer (as usize) from a contiguous CUDA tensor.
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
impl RotaryKernels for CudaRotaryKernels {
    fn rotary_embedding(
        &self,
        positions: &Tensor,
        query: &Tensor,
        key: &Tensor,
        cos_sin_cache: &Tensor,
        _is_neox: bool,
    ) -> KernelResult<(Tensor, Tensor)> {
        use candle_core::DType;

        let rotary_dim = cos_sin_cache.dim(1)?;
        let q_dims = query.shape().dims();
        let total_q_dim = *q_dims
            .last()
            .ok_or_else(|| crate::error::KernelError::Other("empty query tensor".to_string()))?;
        let num_tokens: usize = q_dims[..q_dims.len() - 1].iter().product();
        let k_dims = key.shape().dims();
        let total_k_dim = *k_dims
            .last()
            .ok_or_else(|| crate::error::KernelError::Other("empty key tensor".to_string()))?;

        // For the flat trait API, treat the entire dim as one "head".
        // Phase 3.8 will wire per-head rotation via RotaryEmbedding.
        let head_size = total_q_dim;

        // Clone query/key for out-of-place semantics, make contiguous.
        let q_out = query.contiguous()?.clone();
        let k_out = key.contiguous()?.clone();
        let positions = positions.contiguous()?;
        let cos_sin_cache = cos_sin_cache.contiguous()?;

        match query.dtype() {
            DType::F32 => {
                let p = Self::device_ptr_of::<u32>(&positions)?;
                let q = Self::device_ptr_of::<f32>(&q_out)?;
                let k = Self::device_ptr_of::<f32>(&k_out)?;
                let c = Self::device_ptr_of::<f32>(&cos_sin_cache)?;
                unsafe {
                    cuda_ffi::rotary_embedding_f32(
                        p as *const u32,
                        q as *mut f32,
                        k as *mut f32,
                        c as *const f32,
                        rotary_dim as i32,
                        total_q_dim as i32,
                        total_k_dim as i32,
                        head_size as i32,
                        num_tokens as i32,
                    );
                }
            }
            DType::F16 => {
                let p = Self::device_ptr_of::<u32>(&positions)?;
                let q = Self::device_ptr_of::<half::f16>(&q_out)?;
                let k = Self::device_ptr_of::<half::f16>(&k_out)?;
                let c = Self::device_ptr_of::<half::f16>(&cos_sin_cache)?;
                unsafe {
                    cuda_ffi::rotary_embedding_f16(
                        p as *const u32,
                        q as *mut u16,
                        k as *mut u16,
                        c as *const u16,
                        rotary_dim as i32,
                        total_q_dim as i32,
                        total_k_dim as i32,
                        head_size as i32,
                        num_tokens as i32,
                    );
                }
            }
            DType::BF16 => {
                let p = Self::device_ptr_of::<u32>(&positions)?;
                let q = Self::device_ptr_of::<half::bf16>(&q_out)?;
                let k = Self::device_ptr_of::<half::bf16>(&k_out)?;
                let c = Self::device_ptr_of::<half::bf16>(&cos_sin_cache)?;
                unsafe {
                    cuda_ffi::rotary_embedding_bf16(
                        p as *const u32,
                        q as *mut u16,
                        k as *mut u16,
                        c as *const u16,
                        rotary_dim as i32,
                        total_q_dim as i32,
                        total_k_dim as i32,
                        head_size as i32,
                        num_tokens as i32,
                    );
                }
            }
            _ => {
                return CpuRotaryKernels.rotary_embedding(
                    &positions,
                    query,
                    key,
                    &cos_sin_cache,
                    _is_neox,
                );
            }
        }
        Ok((q_out, k_out))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn make_cos_sin_cache(max_pos: usize, half_dim: usize) -> Tensor {
        // Simple cache: cos and sin for frequencies
        let rotary_dim = 2 * half_dim;
        let mut data = vec![0.0f32; max_pos * rotary_dim];
        for pos in 0..max_pos {
            for i in 0..half_dim {
                let freq = 1.0 / 10000f64.powf(2.0 * i as f64 / (2 * half_dim) as f64);
                let angle = pos as f64 * freq;
                data[pos * rotary_dim + i] = angle.cos() as f32;
                data[pos * rotary_dim + half_dim + i] = angle.sin() as f32;
            }
        }
        Tensor::from_slice(&data, (max_pos, rotary_dim), &Device::Cpu).unwrap()
    }

    #[test]
    fn test_cpu_rotary_position_zero() {
        let kernels = CpuRotaryKernels;
        let cache = make_cos_sin_cache(10, 4); // rotary_dim = 8
        let positions = Tensor::new(&[0u32], &Device::Cpu).unwrap();
        let q = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], &Device::Cpu).unwrap();
        let k = q.clone();

        let (q_rot, _k_rot) = kernels
            .rotary_embedding(&positions, &q, &k, &cache, true)
            .unwrap();
        assert_eq!(q_rot.dims(), &[1, 8]);

        // At position 0, cos=1, sin=0 -> output = input
        let vals = q_rot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, v) in vals.iter().enumerate() {
            assert!(
                (v - (i as f32 + 1.0)).abs() < 1e-4,
                "pos 0 should be identity, got {} at idx {}",
                v,
                i
            );
        }
    }

    #[test]
    fn test_cpu_rotary_shape_preservation() {
        let kernels = CpuRotaryKernels;
        let cache = make_cos_sin_cache(100, 4);
        let positions = Tensor::new(&[0u32, 1, 2, 3], &Device::Cpu).unwrap();
        let q = Tensor::ones(&[4, 8], DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones(&[4, 8], DType::F32, &Device::Cpu).unwrap();

        let (q_rot, k_rot) = kernels
            .rotary_embedding(&positions, &q, &k, &cache, true)
            .unwrap();
        assert_eq!(q_rot.dims(), &[4, 8]);
        assert_eq!(k_rot.dims(), &[4, 8]);
    }

    #[test]
    fn test_cpu_rotary_partial_dim() {
        // Test when rotary_dim < total dim (pass-through for remaining dims)
        let kernels = CpuRotaryKernels;
        let cache = make_cos_sin_cache(10, 2); // rotary_dim = 4, but tensor dim = 8
        let positions = Tensor::new(&[0u32], &Device::Cpu).unwrap();
        let q = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], &Device::Cpu).unwrap();
        let k = q.clone();

        let (q_rot, _) = kernels
            .rotary_embedding(&positions, &q, &k, &cache, true)
            .unwrap();
        assert_eq!(q_rot.dims(), &[1, 8]);

        // Last 4 dims should be unchanged (pass-through).
        let vals = q_rot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[4] - 5.0).abs() < 1e-4);
        assert!((vals[5] - 6.0).abs() < 1e-4);
        assert!((vals[6] - 7.0).abs() < 1e-4);
        assert!((vals[7] - 8.0).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // CUDA tests — compare CUDA kernel output against CPU reference
    // -----------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    fn cuda_device() -> Device {
        Device::new_cuda(0).expect("CUDA device required for this test")
    }

    #[cfg(feature = "cuda")]
    fn assert_close(label: &str, cpu: &[f32], cuda: &[f32], tol: f64) {
        assert_eq!(cpu.len(), cuda.len(), "{label}: length mismatch");
        for (i, (c, g)) in cpu.iter().zip(cuda.iter()).enumerate() {
            assert!(
                (c - g).abs() as f64 <= tol,
                "{label} mismatch at {i}: cpu={c} cuda={g}"
            );
        }
    }

    /// Helper: run rotary_embedding on CPU and CUDA, compare results.
    #[cfg(feature = "cuda")]
    fn assert_rotary_cuda_matches_cpu(
        num_tokens: usize,
        dim: usize,
        half_dim: usize,
        dtype: DType,
        tol: f64,
    ) {
        use super::CudaRotaryKernels;

        let max_pos = 128;
        let cache_cpu = make_cos_sin_cache(max_pos, half_dim)
            .to_dtype(dtype)
            .unwrap();
        let positions_cpu = Tensor::new(
            (0..num_tokens as u32).collect::<Vec<_>>().as_slice(),
            &Device::Cpu,
        )
        .unwrap();
        let q_cpu = Tensor::randn(0f32, 1.0, &[num_tokens, dim], &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let k_cpu = Tensor::randn(0f32, 1.0, &[num_tokens, dim], &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        // CPU reference.
        let (q_ref, k_ref) = CpuRotaryKernels
            .rotary_embedding(&positions_cpu, &q_cpu, &k_cpu, &cache_cpu, true)
            .unwrap();

        // CUDA kernel.
        let dev = cuda_device();
        let cache_gpu = cache_cpu.to_device(&dev).unwrap();
        let positions_gpu = positions_cpu.to_device(&dev).unwrap();
        let q_gpu = q_cpu.to_device(&dev).unwrap();
        let k_gpu = k_cpu.to_device(&dev).unwrap();
        let (q_cuda, k_cuda) = CudaRotaryKernels
            .rotary_embedding(&positions_gpu, &q_gpu, &k_gpu, &cache_gpu, true)
            .unwrap();
        let q_cuda = q_cuda.to_device(&Device::Cpu).unwrap();
        let k_cuda = k_cuda.to_device(&Device::Cpu).unwrap();

        // Compare queries.
        let q_ref_vals = q_ref
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let q_cuda_vals = q_cuda
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_close(
            &format!("rotary_q_{dtype:?}"),
            &q_ref_vals,
            &q_cuda_vals,
            tol,
        );

        // Compare keys.
        let k_ref_vals = k_ref
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let k_cuda_vals = k_cuda
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_close(
            &format!("rotary_k_{dtype:?}"),
            &k_ref_vals,
            &k_cuda_vals,
            tol,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rotary_f32() {
        assert_rotary_cuda_matches_cpu(4, 64, 32, DType::F32, 1e-4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rotary_f16() {
        assert_rotary_cuda_matches_cpu(4, 64, 32, DType::F16, 5e-2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rotary_bf16() {
        assert_rotary_cuda_matches_cpu(4, 64, 32, DType::BF16, 5e-2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rotary_partial_dim() {
        // rotary_dim (2*16=32) < total dim (64) → pass-through for remaining dims.
        assert_rotary_cuda_matches_cpu(4, 64, 16, DType::F32, 1e-4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rotary_many_tokens() {
        assert_rotary_cuda_matches_cpu(32, 128, 64, DType::F32, 1e-4);
    }
}
