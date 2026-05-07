// SPDX-License-Identifier: Apache-2.0
//! Paged KV cache pool using `GpuTensor` — persistent GPU memory.
//!
//! Layout per layer: `[num_blocks, block_size, num_kv_heads, head_dim]`
//! Backend-neutral: storage, sizing, slot decomposition, and span
//! bookkeeping live here for both cuda and metal. The actual buffer
//! allocation is a caller-supplied `alloc_buffer` closure — cuda
//! passes one that wraps `driver::mem_alloc`, metal passes one that
//! wraps `device.new_buffer`. FP8 scale machinery, the
//! `gather_kv_contiguous` D2D copy path, and the GPU mirrors of the
//! span flags use cudarc and stay `cfg(feature = "cuda")` *inside*
//! this unified type.

use anyhow::Result;
use ferrite_cuda_core::RawGpuMem;
#[cfg(feature = "cuda")]
use ferrite_cuda_core::driver;
use ferrite_cuda_core::dtype::DType;
use ferrite_cuda_core::tensor::{GpuTensor, TensorView};

/// Paged KV cache pool for all transformer layers.
///
/// Allocates persistent GPU memory for K and V caches at init time.
/// Block allocation/freeing is managed by the scheduler — this struct
/// just provides the raw tensor views.
///
/// When `cache_dtype` is `Fp8E4m3`, the cache stores 1 byte/element and
/// per-layer scale factors are maintained for quantization/dequantization.
/// FP8 is cuda-only: the scale machinery uses cudarc primitives.
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
    /// Cuda-only: FP8 scale machinery is not implemented under metal.
    #[cfg(feature = "cuda")]
    k_scale_ptrs: Vec<RawGpuMem>,
    /// Per-layer V scale: GPU f32 scalar RAII wrappers. Only used when FP8.
    #[cfg(feature = "cuda")]
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
    /// Cuda-only: span machinery hasn't been ported to metal.
    #[cfg(feature = "cuda")]
    block_unrotated_gpu_ptr: Option<RawGpuMem>,
    /// GPU mirror of `block_is_span` — for post-attention inverse rotation.
    #[cfg(feature = "cuda")]
    block_span_gpu_ptr: Option<RawGpuMem>,
}

// Safety: KvCachePool holds GPU device pointers (GpuTensor arrays and raw
// *mut f32 scale pointers). These are allocated via the backend's device
// memory and are accessible from any host thread after backend setup. The
// pool is created once and moved to the worker thread; no concurrent
// mutation occurs.
unsafe impl Send for KvCachePool {}
unsafe impl Sync for KvCachePool {}

impl KvCachePool {
    /// Allocate KV cache for all layers.
    ///
    /// `alloc_buffer(bytes)` is the backend-neutral way to grab a
    /// raw `bytes`-sized GPU allocation wrapped in a [`RawGpuMem`].
    /// Cuda passes a closure that calls `driver::mem_alloc` and
    /// wraps the result; metal passes one that calls
    /// `device.new_buffer(StorageModeShared)`. Splitting this out
    /// keeps every backend-specific call out of the constructor body
    /// and lets the storage layout / sizing / span bookkeeping live
    /// in one place.
    ///
    /// FP8 scale buffers are also allocated via the same closure, but
    /// the scale-init H2D copy is cuda-only (FP8 cache is not
    /// supported under metal); under metal an FP8 dtype here returns
    /// an error.
    ///
    /// # Safety
    /// Caller must ensure the backend context is current on the
    /// invoking thread (cuda: `ctx_set_current`; metal: any thread
    /// after device init).
    pub unsafe fn new(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        mut alloc_buffer: impl FnMut(usize) -> Result<RawGpuMem>,
    ) -> Result<Self> {
        let elems_per_layer = num_blocks * block_size * num_kv_heads * head_dim;
        let bytes_per_layer = elems_per_layer * dtype.size_bytes();

        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);
        let mut k_ptrs = Vec::with_capacity(num_layers);
        let mut v_ptrs = Vec::with_capacity(num_layers);

        let shape = [num_blocks, block_size, num_kv_heads, head_dim];

        for _ in 0..num_layers {
            let k_mem = alloc_buffer(bytes_per_layer)?;
            let v_mem = alloc_buffer(bytes_per_layer)?;

            k_caches.push(GpuTensor::new(k_mem.ptr(), &shape, dtype));
            v_caches.push(GpuTensor::new(v_mem.ptr(), &shape, dtype));
            k_ptrs.push(k_mem);
            v_ptrs.push(v_mem);
        }

        // FP8 scales: cuda-only. Under metal, fail loudly rather
        // than silently skip — model loaders should not reach here
        // with an FP8 dtype on metal.
        #[cfg(feature = "cuda")]
        let (k_scale_ptrs, v_scale_ptrs) = {
            let mut k_scale_ptrs = Vec::new();
            let mut v_scale_ptrs = Vec::new();
            if dtype.is_fp8() {
                for _ in 0..num_layers {
                    let k_scale = alloc_buffer(4)?;
                    let v_scale = alloc_buffer(4)?;
                    let one: f32 = 1.0;
                    let null_stream = std::ptr::null_mut();
                    driver::memcpy_htod_async(
                        k_scale.ptr(),
                        &one as *const f32 as *const u8,
                        4,
                        null_stream,
                    )?;
                    driver::memcpy_htod_async(
                        v_scale.ptr(),
                        &one as *const f32 as *const u8,
                        4,
                        null_stream,
                    )?;
                    k_scale_ptrs.push(k_scale);
                    v_scale_ptrs.push(v_scale);
                }
            }
            (k_scale_ptrs, v_scale_ptrs)
        };
        #[cfg(not(feature = "cuda"))]
        if dtype.is_fp8() {
            anyhow::bail!("KvCachePool: FP8 cache dtype is cuda-only");
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
            #[cfg(feature = "cuda")]
            k_scale_ptrs,
            #[cfg(feature = "cuda")]
            v_scale_ptrs,
            block_is_unrotated: vec![false; num_blocks],
            block_is_span: vec![false; num_blocks],
            #[cfg(feature = "cuda")]
            block_unrotated_gpu_ptr: None,
            #[cfg(feature = "cuda")]
            block_span_gpu_ptr: None,
        })
    }

    /// Placeholder pool with zero layers and no GPU allocations. Used
    /// to satisfy the `&KvCachePool` field on `ForwardCtx` for vision
    /// encoder forwards, which never reach a kv_cache-touching
    /// instruction (vision body uses `VarlenAttention`, not the paged
    /// `Attention` op). Reading any layer index from this pool would
    /// panic — by contract, vision-mode codegen never emits such reads.
    pub fn empty_for_vision() -> Self {
        Self {
            k_caches: Vec::new(),
            v_caches: Vec::new(),
            _k_ptrs: Vec::new(),
            _v_ptrs: Vec::new(),
            num_blocks: 0,
            block_size: 0,
            num_kv_heads: 0,
            head_dim: 0,
            num_layers: 0,
            cache_dtype: DType::BF16,
            #[cfg(feature = "cuda")]
            k_scale_ptrs: Vec::new(),
            #[cfg(feature = "cuda")]
            v_scale_ptrs: Vec::new(),
            block_is_unrotated: Vec::new(),
            block_is_span: Vec::new(),
            #[cfg(feature = "cuda")]
            block_unrotated_gpu_ptr: None,
            #[cfg(feature = "cuda")]
            block_span_gpu_ptr: None,
        }
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

    /// Per-layer K-cache backing memory. Metal callers use this to
    /// reach into the underlying `metal::Buffer` (via
    /// `RawGpuMem::buffer()`) for ICB binding without re-allocating.
    /// One `RawGpuMem` per layer, shape `[num_layers]`.
    #[cfg(feature = "metal")]
    pub fn k_layer_mem(&self, layer: usize) -> &RawGpuMem {
        &self._k_ptrs[layer]
    }

    /// Per-layer V-cache backing memory. See [`Self::k_layer_mem`].
    #[cfg(feature = "metal")]
    pub fn v_layer_mem(&self, layer: usize) -> &RawGpuMem {
        &self._v_ptrs[layer]
    }

    /// GPU pointer to K scale for a layer (only valid when FP8).
    #[cfg(feature = "cuda")]
    pub fn k_scale_ptr(&self, layer: usize) -> *const f32 {
        self.k_scale_ptrs[layer].ptr() as *const f32
    }

    /// GPU pointer to V scale for a layer (only valid when FP8).
    #[cfg(feature = "cuda")]
    pub fn v_scale_ptr(&self, layer: usize) -> *const f32 {
        self.v_scale_ptrs[layer].ptr() as *const f32
    }

    /// Mutable GPU pointer to K scale for a layer (for writing computed scales).
    #[cfg(feature = "cuda")]
    pub fn k_scale_ptr_mut(&self, layer: usize) -> *mut f32 {
        self.k_scale_ptrs[layer].ptr() as *mut f32
    }

    /// Mutable GPU pointer to V scale for a layer (for writing computed scales).
    #[cfg(feature = "cuda")]
    pub fn v_scale_ptr_mut(&self, layer: usize) -> *mut f32 {
        self.v_scale_ptrs[layer].ptr() as *mut f32
    }

    /// Set K scale for a layer from a host value.
    ///
    /// # Safety
    /// Requires valid CUDA context.
    #[cfg(feature = "cuda")]
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
    #[cfg(feature = "cuda")]
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
    /// Used as a fallback when paged FA2 has issues. Cuda-only: relies
    /// on a `ScratchArena` and the driver D2D path that have no metal
    /// counterpart yet.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gather_kv_contiguous(
        &self,
        layer: usize,
        is_key: bool,
        block_table: GpuTensor,
        total_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
        arena: &mut ferrite_cuda_core::arena::ScratchArena,
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
    /// Requires valid CUDA context and stream. Cuda-only: span
    /// machinery has no metal counterpart yet.
    #[cfg(feature = "cuda")]
    pub unsafe fn sync_block_flags_to_gpu(&mut self, stream: cudarc::driver::sys::CUstream) {
        // Lazily allocate GPU flag buffers on first use.
        let num_blocks = self.block_is_unrotated.len();
        if self.block_unrotated_gpu_ptr.is_none() {
            self.block_unrotated_gpu_ptr = driver::mem_alloc(num_blocks)
                .ok()
                .map(|p| RawGpuMem::new(p, num_blocks));
        }
        if self.block_span_gpu_ptr.is_none() {
            self.block_span_gpu_ptr = driver::mem_alloc(num_blocks)
                .ok()
                .map(|p| RawGpuMem::new(p, num_blocks));
        }

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
    #[cfg(feature = "cuda")]
    pub fn block_unrotated_gpu(&self) -> *const u8 {
        self.block_unrotated_gpu_ptr
            .as_ref()
            .map(|m| m.ptr() as *const u8)
            .unwrap_or(std::ptr::null())
    }

    /// GPU pointer to `block_is_span` flags (for post-attention un-rotation).
    #[cfg(feature = "cuda")]
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
