// SPDX-License-Identifier: Apache-2.0
//! Cache operation kernels.
//!
//! Trait abstraction for KV cache kernels (reshape_and_cache, swap_blocks).
//! Port of: `csrc/cache_kernels.cu` and `csrc/cache.h`

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
    /// * `key` — [num_tokens, num_kv_heads, head_size]
    /// * `value` — [num_tokens, num_kv_heads, head_size]
    /// * `key_cache` — [num_blocks, num_kv_heads, head_size/x, block_size, x]
    /// * `value_cache` — [num_blocks, num_kv_heads, head_size, block_size]
    /// * `slot_mapping` — [num_tokens] maps each token to a cache slot
    ///
    /// Port of: `void reshape_and_cache(key, value, key_cache, value_cache,
    ///           slot_mapping, kv_cache_dtype, k_scale, v_scale)`
    fn reshape_and_cache(
        &self,
        key: &Tensor,
        value: &Tensor,
        key_cache: &mut Tensor,
        value_cache: &mut Tensor,
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
        _key_cache: &mut Tensor,
        _value_cache: &mut Tensor,
        _slot_mapping: &Tensor,
    ) -> KernelResult<()> {
        // Stub: real implementation would scatter key/value into cache slots.
        // This requires in-place tensor mutation which is complex on CPU
        // with candle. Full implementation deferred to CUDA path.
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
        let mut key_cache = Tensor::zeros(&[100, 8, 8, 16, 8], DType::F32, &Device::Cpu).unwrap();
        let mut value_cache = Tensor::zeros(&[100, 8, 64, 16], DType::F32, &Device::Cpu).unwrap();
        let slot_mapping = Tensor::new(&[0u32, 1, 2, 3], &Device::Cpu).unwrap();

        // Should not error.
        kernels
            .reshape_and_cache(
                &key,
                &value,
                &mut key_cache,
                &mut value_cache,
                &slot_mapping,
            )
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
}
