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
