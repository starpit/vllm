// SPDX-License-Identifier: Apache-2.0
//! Cache operation kernels.
//!
//! Trait abstraction for KV cache kernels (reshape_and_cache, swap_blocks).
//! Port of: `csrc/cache_kernels.cu` and `csrc/cache.h`
//!
//! ## Cache Layout
//!
//! Our KV cache uses NHD layout: `[num_blocks, block_size, num_kv_heads, head_dim]`.
//! This matches the "flash" variant in Python vLLM and is simpler than the
//! reshuffled layout used by the original paged attention kernels.
//!
//! ## Simplifications vs Python vLLM
//!
//! - Scalar loads instead of vectorized loads
//! - No FP8 KV cache dtype support (no k_scale/v_scale)
//! - NHD layout only (no head_size/x reshuffling)

use candle_core::Tensor;

use crate::error::KernelResult;

/// KV cache kernel interface.
///
/// Abstracts the CUDA kernels that manage the paged KV cache:
/// storing new key/value entries and swapping blocks.
pub trait CacheKernels: Send + Sync {
    /// Reshape and cache key/value tensors into the paged KV cache.
    ///
    /// Takes newly computed key/value tensors and stores them into
    /// the appropriate locations in the block-based KV cache.
    ///
    /// * `key` — [num_tokens, num_kv_heads, head_dim]
    /// * `value` — [num_tokens, num_kv_heads, head_dim]
    /// * `key_cache` — [num_blocks, block_size, num_kv_heads, head_dim]
    /// * `value_cache` — [num_blocks, block_size, num_kv_heads, head_dim]
    /// * `slot_mapping` — [num_tokens] i64 tensor mapping each token to a cache slot
    ///   where slot = block_idx * block_size + position_in_block
    fn reshape_and_cache(
        &self,
        key: &Tensor,
        value: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> KernelResult<()>;

    /// Swap cache blocks between GPU and CPU (or between GPUs).
    ///
    /// * `src` — source cache tensor
    /// * `dst` — destination cache tensor
    /// * `block_mapping` — [N, 2] tensor of (src_block, dst_block) pairs
    ///
    /// Port of: `void swap_blocks(src, dst, block_size_in_bytes, block_mapping)`
    fn swap_blocks(
        &self,
        src: &Tensor,
        dst: &mut Tensor,
        block_mapping: &Tensor,
    ) -> KernelResult<()>;
}

/// CPU implementation of cache kernels (for testing).
pub struct CpuCacheKernels;

impl CacheKernels for CpuCacheKernels {
    fn reshape_and_cache(
        &self,
        _key: &Tensor,
        _value: &Tensor,
        _key_cache: &Tensor,
        _value_cache: &Tensor,
        _slot_mapping: &Tensor,
    ) -> KernelResult<()> {
        // CPU path uses KvBlockPool::scatter_new_kv's per-token loop instead.
        Ok(())
    }

    fn swap_blocks(
        &self,
        _src: &Tensor,
        _dst: &mut Tensor,
        _block_mapping: &Tensor,
    ) -> KernelResult<()> {
        // Stub: real implementation would copy blocks between src and dst.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CUDA implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda_ffi {
    unsafe extern "C" {
        pub fn reshape_and_cache_f32(
            key: *const f32,
            value: *const f32,
            key_cache: *mut f32,
            value_cache: *mut f32,
            slot_mapping: *const i64,
            num_tokens: i32,
            num_heads: i32,
            head_dim: i32,
            block_size: i32,
            stream: *mut std::ffi::c_void,
        );
        pub fn reshape_and_cache_f16(
            key: *const u16,
            value: *const u16,
            key_cache: *mut u16,
            value_cache: *mut u16,
            slot_mapping: *const i64,
            num_tokens: i32,
            num_heads: i32,
            head_dim: i32,
            block_size: i32,
            stream: *mut std::ffi::c_void,
        );
        pub fn reshape_and_cache_bf16(
            key: *const u16,
            value: *const u16,
            key_cache: *mut u16,
            value_cache: *mut u16,
            slot_mapping: *const i64,
            num_tokens: i32,
            num_heads: i32,
            head_dim: i32,
            block_size: i32,
            stream: *mut std::ffi::c_void,
        );
    }
}

/// CUDA implementation of cache kernels.
#[cfg(feature = "cuda")]
pub struct CudaCacheKernels;

#[cfg(feature = "cuda")]
impl CudaCacheKernels {
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
impl CacheKernels for CudaCacheKernels {
    fn reshape_and_cache(
        &self,
        key: &Tensor,
        value: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> KernelResult<()> {
        use candle_core::DType;

        let num_tokens = key.dims()[0];
        if num_tokens == 0 {
            return Ok(());
        }
        let num_heads = key.dims()[1];
        let head_dim = key.dims()[2];

        // Cache layout: [num_blocks, block_size, num_heads, head_dim]
        let block_size = key_cache.dims()[1];

        let key = key.contiguous()?;
        let value = value.contiguous()?;
        let slot_mapping = slot_mapping.contiguous()?;

        match key.dtype() {
            DType::F32 => {
                let k = Self::device_ptr_of::<f32>(&key)?;
                let v = Self::device_ptr_of::<f32>(&value)?;
                let kc = Self::device_ptr_of::<f32>(key_cache)?;
                let vc = Self::device_ptr_of::<f32>(value_cache)?;
                let sm = Self::device_ptr_of::<i64>(&slot_mapping)?;
                unsafe {
                    cuda_ffi::reshape_and_cache_f32(
                        k as *const f32,
                        v as *const f32,
                        kc as *mut f32,
                        vc as *mut f32,
                        sm as *const i64,
                        num_tokens as i32,
                        num_heads as i32,
                        head_dim as i32,
                        block_size as i32,
                        std::ptr::null_mut(),
                    );
                }
            }
            DType::F16 => {
                let k = Self::device_ptr_of::<half::f16>(&key)?;
                let v = Self::device_ptr_of::<half::f16>(&value)?;
                let kc = Self::device_ptr_of::<half::f16>(key_cache)?;
                let vc = Self::device_ptr_of::<half::f16>(value_cache)?;
                let sm = Self::device_ptr_of::<i64>(&slot_mapping)?;
                unsafe {
                    cuda_ffi::reshape_and_cache_f16(
                        k as *const u16,
                        v as *const u16,
                        kc as *mut u16,
                        vc as *mut u16,
                        sm as *const i64,
                        num_tokens as i32,
                        num_heads as i32,
                        head_dim as i32,
                        block_size as i32,
                        std::ptr::null_mut(),
                    );
                }
            }
            DType::BF16 => {
                let k = Self::device_ptr_of::<half::bf16>(&key)?;
                let v = Self::device_ptr_of::<half::bf16>(&value)?;
                let kc = Self::device_ptr_of::<half::bf16>(key_cache)?;
                let vc = Self::device_ptr_of::<half::bf16>(value_cache)?;
                let sm = Self::device_ptr_of::<i64>(&slot_mapping)?;
                unsafe {
                    cuda_ffi::reshape_and_cache_bf16(
                        k as *const u16,
                        v as *const u16,
                        kc as *mut u16,
                        vc as *mut u16,
                        sm as *const i64,
                        num_tokens as i32,
                        num_heads as i32,
                        head_dim as i32,
                        block_size as i32,
                        std::ptr::null_mut(),
                    );
                }
            }
            _ => {
                return Err(crate::error::KernelError::DType(format!(
                    "reshape_and_cache: unsupported dtype {:?}",
                    key.dtype()
                )));
            }
        }
        Ok(())
    }

    fn swap_blocks(
        &self,
        _src: &Tensor,
        _dst: &mut Tensor,
        _block_mapping: &Tensor,
    ) -> KernelResult<()> {
        // TODO: implement CUDA swap_blocks via cudaMemcpyAsync.
        // Currently swap_out/swap_in uses candle to_device() which works.
        Ok(())
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
    fn test_cpu_cache_reshape_and_cache_noop() {
        let kernels = CpuCacheKernels;

        let key = Tensor::zeros(&[4, 8, 64], DType::F32, &Device::Cpu).unwrap();
        let value = Tensor::zeros(&[4, 8, 64], DType::F32, &Device::Cpu).unwrap();
        let key_cache = Tensor::zeros(&[100, 16, 8, 64], DType::F32, &Device::Cpu).unwrap();
        let value_cache = Tensor::zeros(&[100, 16, 8, 64], DType::F32, &Device::Cpu).unwrap();
        let slot_mapping = Tensor::new(&[0i64, 1, 2, 3], &Device::Cpu).unwrap();

        kernels
            .reshape_and_cache(&key, &value, &key_cache, &value_cache, &slot_mapping)
            .unwrap();
    }

    #[test]
    fn test_cpu_cache_swap_blocks_noop() {
        let kernels = CpuCacheKernels;

        let src = Tensor::zeros(&[10, 16, 128], DType::F32, &Device::Cpu).unwrap();
        let mut dst = Tensor::zeros(&[10, 16, 128], DType::F32, &Device::Cpu).unwrap();
        let mapping = Tensor::new(&[[0u32, 5], [1, 6]], &Device::Cpu).unwrap();

        kernels.swap_blocks(&src, &mut dst, &mapping).unwrap();
    }

    // -----------------------------------------------------------------------
    // CUDA tests — verify reshape_and_cache scatters correctly
    // -----------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    fn cuda_device() -> Device {
        Device::new_cuda(0).expect("CUDA device required for this test")
    }

    /// Test: scatter 2 tokens into non-adjacent cache slots, verify data lands correctly.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_reshape_and_cache_basic() {
        use super::CudaCacheKernels;

        let dev = cuda_device();
        let num_blocks = 8;
        let block_size = 4;
        let num_heads = 2;
        let head_dim = 4;
        let n_elems = num_heads * head_dim; // 8

        // Create 2 tokens with known values.
        let key_data: Vec<f32> = (0..2 * n_elems).map(|i| (i + 1) as f32).collect();
        let val_data: Vec<f32> = (0..2 * n_elems).map(|i| (i + 100) as f32).collect();
        let key = Tensor::from_slice(&key_data, &[2, num_heads, head_dim], &dev).unwrap();
        let value = Tensor::from_slice(&val_data, &[2, num_heads, head_dim], &dev).unwrap();

        // Allocate cache (all zeros).
        let key_cache = Tensor::zeros(
            &[num_blocks, block_size, num_heads, head_dim],
            DType::F32,
            &dev,
        )
        .unwrap();
        let value_cache = Tensor::zeros(
            &[num_blocks, block_size, num_heads, head_dim],
            DType::F32,
            &dev,
        )
        .unwrap();

        // Scatter: token 0 → slot 5 (block 1, offset 1), token 1 → slot 14 (block 3, offset 2).
        let slot_mapping = Tensor::new(&[5i64, 14], &dev).unwrap();

        CudaCacheKernels
            .reshape_and_cache(&key, &value, &key_cache, &value_cache, &slot_mapping)
            .unwrap();

        // Read back and verify.
        let kc = key_cache.to_device(&Device::Cpu).unwrap();
        let vc = value_cache.to_device(&Device::Cpu).unwrap();

        // Flatten to [num_blocks * block_size, num_heads * head_dim].
        let kc_flat = kc
            .reshape(&[num_blocks * block_size, n_elems])
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let vc_flat = vc
            .reshape(&[num_blocks * block_size, n_elems])
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        // Slot 5 should have token 0's key data [1..8].
        for i in 0..n_elems {
            assert!(
                (kc_flat[5][i] - (i + 1) as f32).abs() < 1e-5,
                "key slot 5 elem {i}: expected {}, got {}",
                i + 1,
                kc_flat[5][i]
            );
            assert!(
                (vc_flat[5][i] - (i + 100) as f32).abs() < 1e-5,
                "value slot 5 elem {i}: expected {}, got {}",
                i + 100,
                vc_flat[5][i]
            );
        }

        // Slot 14 should have token 1's key data [9..16].
        for i in 0..n_elems {
            assert!(
                (kc_flat[14][i] - (n_elems + i + 1) as f32).abs() < 1e-5,
                "key slot 14 elem {i}: expected {}, got {}",
                n_elems + i + 1,
                kc_flat[14][i]
            );
        }

        // Other slots should still be zero.
        for i in 0..n_elems {
            assert!(
                kc_flat[0][i].abs() < 1e-5,
                "key slot 0 should be zero, got {}",
                kc_flat[0][i]
            );
        }
    }

    /// Test: scatter many tokens across multiple blocks, verify with f16.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_reshape_and_cache_f16() {
        use super::CudaCacheKernels;

        let dev = cuda_device();
        let num_blocks = 4;
        let block_size = 4;
        let num_heads = 4;
        let head_dim = 8;

        // 6 tokens filling block 0 (4 tokens) + block 1 (2 tokens).
        let key = Tensor::randn(0f32, 1.0, &[6, num_heads, head_dim], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let value = Tensor::randn(0f32, 1.0, &[6, num_heads, head_dim], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&dev)
            .unwrap();

        let key_cache = Tensor::zeros(
            &[num_blocks, block_size, num_heads, head_dim],
            DType::F16,
            &dev,
        )
        .unwrap();
        let value_cache = Tensor::zeros(
            &[num_blocks, block_size, num_heads, head_dim],
            DType::F16,
            &dev,
        )
        .unwrap();

        // Sequential slots: block 0 slots 0-3, block 1 slots 4-5.
        let slot_mapping = Tensor::new(&[0i64, 1, 2, 3, 4, 5], &dev).unwrap();

        CudaCacheKernels
            .reshape_and_cache(&key, &value, &key_cache, &value_cache, &slot_mapping)
            .unwrap();

        // Read back token 0 from cache slot 0 and compare to input.
        let key_cpu = key.to_device(&Device::Cpu).unwrap();
        let kc_cpu = key_cache.to_device(&Device::Cpu).unwrap();

        let token0_input = key_cpu
            .narrow(0, 0, 1)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let slot0_cache = kc_cpu
            .reshape(&[num_blocks * block_size, num_heads * head_dim])
            .unwrap()
            .narrow(0, 0, 1)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        for (i, (inp, cached)) in token0_input.iter().zip(slot0_cache.iter()).enumerate() {
            assert!(
                (inp - cached).abs() < 1e-2,
                "f16 slot 0 elem {i}: input={inp} cached={cached}"
            );
        }
    }

    /// Test: slot_mapping with -1 (padding tokens should be skipped).
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_reshape_and_cache_padding() {
        use super::CudaCacheKernels;

        let dev = cuda_device();
        let num_blocks = 4;
        let block_size = 4;
        let num_heads = 2;
        let head_dim = 4;

        let key = Tensor::ones(&[3, num_heads, head_dim], DType::F32, &dev).unwrap();
        let value = Tensor::ones(&[3, num_heads, head_dim], DType::F32, &dev).unwrap();

        let key_cache = Tensor::zeros(
            &[num_blocks, block_size, num_heads, head_dim],
            DType::F32,
            &dev,
        )
        .unwrap();
        let value_cache = Tensor::zeros(
            &[num_blocks, block_size, num_heads, head_dim],
            DType::F32,
            &dev,
        )
        .unwrap();

        // Token 1 has slot -1 (padding), should be skipped.
        let slot_mapping = Tensor::new(&[0i64, -1, 2], &dev).unwrap();

        CudaCacheKernels
            .reshape_and_cache(&key, &value, &key_cache, &value_cache, &slot_mapping)
            .unwrap();

        let kc = key_cache.to_device(&Device::Cpu).unwrap();
        let kc_flat = kc
            .reshape(&[num_blocks * block_size, num_heads * head_dim])
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        // Slot 0 and 2 should be 1.0 (written), slot 1 should be 0.0 (skipped).
        assert!(
            (kc_flat[0][0] - 1.0).abs() < 1e-5,
            "slot 0 should be written"
        );
        assert!(
            kc_flat[1][0].abs() < 1e-5,
            "slot 1 should be zero (padding)"
        );
        assert!(
            (kc_flat[2][0] - 1.0).abs() < 1e-5,
            "slot 2 should be written"
        );
    }

    /// Test: single token write (the write_kv path).
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_reshape_and_cache_single_token() {
        use super::CudaCacheKernels;

        let dev = cuda_device();
        let num_blocks = 4;
        let block_size = 4;
        let num_heads = 8;
        let head_dim = 64;
        let n_elems = num_heads * head_dim;

        let key_data: Vec<f32> = (0..n_elems).map(|i| i as f32 * 0.01).collect();
        let key = Tensor::from_slice(&key_data, &[1, num_heads, head_dim], &dev).unwrap();
        let value = Tensor::ones(&[1, num_heads, head_dim], DType::F32, &dev).unwrap();

        let key_cache = Tensor::zeros(
            &[num_blocks, block_size, num_heads, head_dim],
            DType::F32,
            &dev,
        )
        .unwrap();
        let value_cache = Tensor::zeros(
            &[num_blocks, block_size, num_heads, head_dim],
            DType::F32,
            &dev,
        )
        .unwrap();

        // Write to block 2, offset 3 → slot = 2*4 + 3 = 11.
        let slot_mapping = Tensor::new(&[11i64], &dev).unwrap();

        CudaCacheKernels
            .reshape_and_cache(&key, &value, &key_cache, &value_cache, &slot_mapping)
            .unwrap();

        let kc = key_cache.to_device(&Device::Cpu).unwrap();
        let kc_flat = kc
            .reshape(&[num_blocks * block_size, n_elems])
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        // Slot 11 should have the key data.
        for i in 0..n_elems {
            let expected = i as f32 * 0.01;
            assert!(
                (kc_flat[11][i] - expected).abs() < 1e-5,
                "slot 11 elem {i}: expected {expected}, got {}",
                kc_flat[11][i]
            );
        }
    }
}
