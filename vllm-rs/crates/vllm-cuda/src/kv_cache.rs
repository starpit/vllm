// SPDX-License-Identifier: Apache-2.0
//! Paged KV cache pool using `GpuTensor` — persistent GPU memory.
//!
//! Layout per layer: `[num_blocks, block_size, num_kv_heads, head_dim]`
//! Paged KV cache pool for GPU inference.

use crate::alloc::RawGpuMem;
use crate::driver;
use crate::dtype::DType;
use crate::tensor::{GpuTensor, TensorView};
use anyhow::Result;

/// Paged KV cache pool for all transformer layers.
///
/// Allocates persistent GPU memory for K and V caches at init time.
/// Block allocation/freeing is managed by the scheduler — this struct
/// just provides the raw tensor views.
///
/// When `cache_dtype` is `Fp8E4m3`, the cache stores 1 byte/element and
/// per-layer scale factors are maintained for quantization/dequantization.
pub struct KvCachePool {
    /// K cache per layer: `[num_blocks, block_size, num_kv_heads, head_dim]`
    k_caches: Vec<GpuTensor>,
    /// V cache per layer: same layout
    v_caches: Vec<GpuTensor>,
    /// RAII wrappers for KV cache GPU allocations — auto-freed on drop.
    _k_ptrs: Vec<RawGpuMem>,
    _v_ptrs: Vec<RawGpuMem>,
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    /// The dtype stored in cache (may differ from model dtype when FP8).
    cache_dtype: DType,
    /// Per-layer K scale: GPU f32 scalar RAII wrappers. Only used when FP8.
    k_scale_ptrs: Vec<RawGpuMem>,
    /// Per-layer V scale: GPU f32 scalar RAII wrappers. Only used when FP8.
    v_scale_ptrs: Vec<RawGpuMem>,

    /// Per-physical-block flag: `true` = K is **currently** stored unrotated.
    /// Used by the pre-attention rotation kernel (rotate these blocks).
    /// Indexed by physical block ID.
    pub block_is_unrotated: Vec<bool>,

    /// Per-physical-block flag: `true` = this is a span block (should be
    /// stored unrotated in the resting state between steps).
    /// Used by the post-attention un-rotation kernel (un-rotate these blocks).
    pub block_is_span: Vec<bool>,

    /// GPU mirror of `block_is_unrotated` — for pre-attention forward rotation.
    block_unrotated_gpu_ptr: Option<RawGpuMem>,
    /// GPU mirror of `block_is_span` — for post-attention inverse rotation.
    block_span_gpu_ptr: Option<RawGpuMem>,
}

// Safety: KvCachePool holds GPU device pointers (GpuTensor arrays and raw
// *mut f32 scale pointers). These are allocated via the CUDA driver and are
// accessible from any host thread after context setup. The pool is created
// once and moved to the worker thread; no concurrent mutation occurs.
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
        let mut k_scale_ptrs = Vec::new();
        let mut v_scale_ptrs = Vec::new();

        let shape = [num_blocks, block_size, num_kv_heads, head_dim];

        for _ in 0..num_layers {
            let k_ptr = driver::mem_alloc(bytes_per_layer)?;
            let v_ptr = driver::mem_alloc(bytes_per_layer)?;

            k_caches.push(GpuTensor::new(k_ptr, &shape, dtype));
            v_caches.push(GpuTensor::new(v_ptr, &shape, dtype));
            k_ptrs.push(RawGpuMem::new(k_ptr, bytes_per_layer));
            v_ptrs.push(RawGpuMem::new(v_ptr, bytes_per_layer));
        }

        // Allocate per-layer scale factors for FP8 cache.
        if dtype.is_fp8() {
            for _ in 0..num_layers {
                // Allocate GPU f32 scalars, initialized to 1.0.
                let k_scale_ptr = driver::mem_alloc(4)?;
                let v_scale_ptr = driver::mem_alloc(4)?;
                let one: f32 = 1.0;
                // Use null stream (synchronous) for init-time copy.
                let null_stream = std::ptr::null_mut();
                driver::memcpy_htod_async(
                    k_scale_ptr,
                    &one as *const f32 as *const u8,
                    4,
                    null_stream,
                )?;
                driver::memcpy_htod_async(
                    v_scale_ptr,
                    &one as *const f32 as *const u8,
                    4,
                    null_stream,
                )?;
                k_scale_ptrs.push(RawGpuMem::new(k_scale_ptr, 4));
                v_scale_ptrs.push(RawGpuMem::new(v_scale_ptr, 4));
            }
        }

        let total_mb = (2 * num_layers * bytes_per_layer) as f64 / (1024.0 * 1024.0);
        let dtype_label = if dtype.is_fp8() {
            "FP8 E4M3"
        } else {
            &format!("{}", dtype)
        };
        tracing::info!(
            "KvCachePool: {num_layers} layers × {num_blocks} blocks × {block_size} slots = {total_mb:.0} MB ({dtype_label})"
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
            cache_dtype: dtype,
            k_scale_ptrs,
            v_scale_ptrs,
            block_is_unrotated: vec![false; num_blocks],
            block_is_span: vec![false; num_blocks],
            block_unrotated_gpu_ptr: if vllm_config::SpansConfig::from_env().fuse_rope() {
                driver::mem_alloc(num_blocks)
                    .ok()
                    .map(|p| RawGpuMem::new(p, num_blocks))
            } else {
                None
            },
            block_span_gpu_ptr: if vllm_config::SpansConfig::from_env().fuse_rope() {
                driver::mem_alloc(num_blocks)
                    .ok()
                    .map(|p| RawGpuMem::new(p, num_blocks))
            } else {
                None
            },
        })
    }

    /// Get K cache tensor for a layer as a lifetime-checked view.
    pub fn k_cache(&self, layer: usize) -> TensorView<'_> {
        // Safety: KvCachePool owns the memory via _k_ptrs; view borrows &self.
        unsafe { TensorView::from_raw(self.k_caches[layer]) }
    }

    /// Get V cache tensor for a layer as a lifetime-checked view.
    pub fn v_cache(&self, layer: usize) -> TensorView<'_> {
        // Safety: KvCachePool owns the memory via _v_ptrs; view borrows &self.
        unsafe { TensorView::from_raw(self.v_caches[layer]) }
    }

    /// The dtype used for cache storage.
    pub fn cache_dtype(&self) -> DType {
        self.cache_dtype
    }

    /// Whether this pool stores FP8 data.
    pub fn is_fp8(&self) -> bool {
        self.cache_dtype.is_fp8()
    }

    /// GPU pointer to K scale for a layer (only valid when FP8).
    pub fn k_scale_ptr(&self, layer: usize) -> *const f32 {
        self.k_scale_ptrs[layer].ptr() as *const f32
    }

    /// GPU pointer to V scale for a layer (only valid when FP8).
    pub fn v_scale_ptr(&self, layer: usize) -> *const f32 {
        self.v_scale_ptrs[layer].ptr() as *const f32
    }

    /// Mutable GPU pointer to K scale for a layer (for writing computed scales).
    pub fn k_scale_ptr_mut(&self, layer: usize) -> *mut f32 {
        self.k_scale_ptrs[layer].ptr() as *mut f32
    }

    /// Mutable GPU pointer to V scale for a layer (for writing computed scales).
    pub fn v_scale_ptr_mut(&self, layer: usize) -> *mut f32 {
        self.v_scale_ptrs[layer].ptr() as *mut f32
    }

    /// Set K scale for a layer from a host value.
    ///
    /// # Safety
    /// Requires valid CUDA context.
    pub unsafe fn set_k_scale(
        &self,
        layer: usize,
        val: f32,
        stream: cudarc::driver::sys::CUstream,
    ) {
        driver::memcpy_htod_async(
            self.k_scale_ptrs[layer].ptr(),
            &val as *const f32 as *const u8,
            4,
            stream,
        )
        .expect("set_k_scale H2D");
    }

    /// Set V scale for a layer from a host value.
    ///
    /// # Safety
    /// Requires valid CUDA context.
    pub unsafe fn set_v_scale(
        &self,
        layer: usize,
        val: f32,
        stream: cudarc::driver::sys::CUstream,
    ) {
        driver::memcpy_htod_async(
            self.v_scale_ptrs[layer].ptr(),
            &val as *const f32 as *const u8,
            4,
            stream,
        )
        .expect("set_v_scale H2D");
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

    /// Mark a physical block's current rotation state and span identity.
    pub fn mark_block(&mut self, physical_block_id: usize, is_span: bool, is_unrotated: bool) {
        if physical_block_id < self.block_is_unrotated.len() {
            self.block_is_span[physical_block_id] = is_span;
            self.block_is_unrotated[physical_block_id] = is_unrotated;
        }
    }

    /// Upload both flag arrays to GPU.
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    pub unsafe fn sync_block_flags_to_gpu(&self, stream: cudarc::driver::sys::CUstream) {
        if let Some(ref mem) = self.block_unrotated_gpu_ptr {
            let flags: Vec<u8> = self
                .block_is_unrotated
                .iter()
                .map(|&b| u8::from(b))
                .collect();
            driver::memcpy_htod_async(mem.ptr(), flags.as_ptr(), flags.len(), stream)
                .expect("sync block_unrotated H2D");
        }
        if let Some(ref mem) = self.block_span_gpu_ptr {
            let flags: Vec<u8> = self.block_is_span.iter().map(|&b| u8::from(b)).collect();
            driver::memcpy_htod_async(mem.ptr(), flags.as_ptr(), flags.len(), stream)
                .expect("sync block_span H2D");
        }
    }

    /// GPU pointer to `block_is_unrotated` flags (for pre-attention rotation).
    pub fn block_unrotated_gpu(&self) -> *const u8 {
        self.block_unrotated_gpu_ptr
            .as_ref()
            .map(|m| m.ptr() as *const u8)
            .unwrap_or(std::ptr::null())
    }

    /// GPU pointer to `block_is_span` flags (for post-attention un-rotation).
    pub fn block_span_gpu(&self) -> *const u8 {
        self.block_span_gpu_ptr
            .as_ref()
            .map(|m| m.ptr() as *const u8)
            .unwrap_or(std::ptr::null())
    }
}

// All GPU allocations (_k_ptrs, _v_ptrs, k_scale_ptrs, v_scale_ptrs,
// block_unrotated_gpu_ptr, block_span_gpu_ptr) are RawGpuMem and freed
// automatically via Drop — no manual impl needed.
