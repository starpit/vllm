// SPDX-License-Identifier: Apache-2.0
//! Paged KV cache pool using `GpuTensor` — persistent GPU memory.
//!
//! Layout per layer: `[num_blocks, block_size, num_kv_heads, head_dim]`
//! Same layout as the candle-based `KvBlockPool` in vllm-models.

use crate::driver;
use crate::dtype::DType;
use crate::tensor::GpuTensor;
use anyhow::Result;

/// Paged KV cache pool for all transformer layers.
///
/// Allocates persistent GPU memory for K and V caches at init time.
/// Block allocation/freeing is managed by the scheduler — this struct
/// just provides the raw tensor views.
pub struct KvCachePool {
    /// K cache per layer: `[num_blocks, block_size, num_kv_heads, head_dim]`
    k_caches: Vec<GpuTensor>,
    /// V cache per layer: same layout
    v_caches: Vec<GpuTensor>,
    /// Raw pointers for deallocation.
    _k_ptrs: Vec<*mut u8>,
    _v_ptrs: Vec<*mut u8>,
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
}

unsafe impl Send for KvCachePool {}
unsafe impl Sync for KvCachePool {}

impl KvCachePool {
    /// Allocate KV cache for all layers.
    ///
    /// # Safety
    /// Requires valid CUDA context.
    pub unsafe fn new(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> Result<Self> {
        let elems_per_layer = num_blocks * block_size * num_kv_heads * head_dim;
        let bytes_per_layer = elems_per_layer * dtype.size_bytes();

        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);
        let mut k_ptrs = Vec::with_capacity(num_layers);
        let mut v_ptrs = Vec::with_capacity(num_layers);

        let shape = [num_blocks, block_size, num_kv_heads, head_dim];

        for _ in 0..num_layers {
            let k_ptr = driver::mem_alloc(bytes_per_layer)?;
            let v_ptr = driver::mem_alloc(bytes_per_layer)?;

            k_caches.push(GpuTensor::new(k_ptr, &shape, dtype));
            v_caches.push(GpuTensor::new(v_ptr, &shape, dtype));
            k_ptrs.push(k_ptr);
            v_ptrs.push(v_ptr);
        }

        let total_mb = (2 * num_layers * bytes_per_layer) as f64 / (1024.0 * 1024.0);
        tracing::info!(
            "KvCachePool: {num_layers} layers × {num_blocks} blocks × {block_size} slots = {total_mb:.0} MB"
        );

        Ok(Self {
            k_caches,
            v_caches,
            _k_ptrs: k_ptrs,
            _v_ptrs: v_ptrs,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            num_layers,
        })
    }

    /// Get K cache tensor for a layer.
    pub fn k_cache(&self, layer: usize) -> GpuTensor {
        self.k_caches[layer]
    }

    /// Get V cache tensor for a layer.
    pub fn v_cache(&self, layer: usize) -> GpuTensor {
        self.v_caches[layer]
    }

    /// Gather K or V from blocks into a contiguous `[total_tokens, kv_heads, head_dim]` tensor.
    ///
    /// `block_table` is a GPU tensor `[batch_size, max_blocks_per_seq]` of i32 block IDs.
    /// For single-request decode, batch_size=1. `total_tokens` is the total KV length.
    /// `is_key` selects K (true) or V (false) cache.
    ///
    /// This is a CPU-driven D2D copy per block — not fast but correct.
    /// Used as a fallback when paged FA2 has issues.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gather_kv_contiguous(
        &self,
        layer: usize,
        is_key: bool,
        block_table: GpuTensor,
        total_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
        arena: &mut crate::arena::ScratchArena,
        stream: cudarc::driver::sys::CUstream,
    ) -> GpuTensor {
        let cache = if is_key {
            self.k_caches[layer]
        } else {
            self.v_caches[layer]
        };
        let out = arena.alloc(&[total_tokens, num_kv_heads, head_dim], cache.dtype());

        // D2H the block table to get block IDs on CPU.
        let num_blocks_in_table = block_table.dim(1);
        let mut block_ids = vec![0i32; num_blocks_in_table];
        driver::memcpy_dtoh_async(
            block_ids.as_mut_ptr() as *mut u8,
            block_table.raw_ptr() as *const u8,
            num_blocks_in_table * 4,
            stream,
        )
        .expect("D2H block_table");
        driver::stream_synchronize(stream).expect("sync block_table D2H");

        // Copy block by block.
        let elem_size = cache.dtype().size_bytes();
        let tokens_per_block = self.block_size;
        let row_bytes = num_kv_heads * head_dim * elem_size;
        // Cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
        let block_stride_bytes = tokens_per_block * row_bytes;

        let mut tokens_remaining = total_tokens;
        let mut dst_offset: usize = 0;
        for &bid in &block_ids {
            if tokens_remaining == 0 {
                break;
            }
            let n = tokens_remaining.min(tokens_per_block);
            let src = (cache.raw_ptr() as *const u8).add(bid as usize * block_stride_bytes);
            let dst = (out.raw_ptr() as *mut u8).add(dst_offset);
            driver::memcpy_dtod_async(dst, src, n * row_bytes, stream).expect("D2D gather block");
            dst_offset += n * row_bytes;
            tokens_remaining -= n;
        }

        out
    }
}

impl Drop for KvCachePool {
    fn drop(&mut self) {
        for &ptr in self._k_ptrs.iter().chain(self._v_ptrs.iter()) {
            unsafe {
                let _ = driver::mem_free(ptr);
            }
        }
    }
}
