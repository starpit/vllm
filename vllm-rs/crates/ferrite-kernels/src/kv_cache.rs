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

    // ── Reactive (chunked) KV storage — metal only ──────────────────
    //
    // On metal the per-layer KV cache is NOT one contiguous buffer; it
    // is a series of fixed-size *chunk* buffers (`blocks_per_chunk`
    // blocks each) so the `MTLResidencySet` can hold only the chunks
    // that back live KV. The kernels bind a per-layer chunk-address
    // *table* (device uint64 gpuAddresses) at the `KvCacheK/V` slot and
    // deref `table[block_id / blocks_per_chunk]` then address
    // `block_id % blocks_per_chunk` within that chunk. CUDA keeps the
    // single-buffer `k_caches` / `_k_ptrs` layout above untouched.
    //
    // `k_chunks[layer][chunk]` / `v_chunks[layer][chunk]` own the chunk
    // data buffers (StorageModePrivate); `k_chunk_tables[layer]` /
    // `v_chunk_tables[layer]` own the per-layer address tables
    // (StorageModeShared, host-written via `fill_chunk_tables`).
    #[cfg(feature = "metal")]
    k_chunks: Vec<Vec<RawGpuMem>>,
    #[cfg(feature = "metal")]
    v_chunks: Vec<Vec<RawGpuMem>>,
    #[cfg(feature = "metal")]
    k_chunk_tables: Vec<RawGpuMem>,
    #[cfg(feature = "metal")]
    v_chunk_tables: Vec<RawGpuMem>,
    #[cfg(feature = "metal")]
    blocks_per_chunk: usize,
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
            // CUDA / single-buffer path: no chunked storage.
            #[cfg(feature = "metal")]
            k_chunks: Vec::new(),
            #[cfg(feature = "metal")]
            v_chunks: Vec::new(),
            #[cfg(feature = "metal")]
            k_chunk_tables: Vec::new(),
            #[cfg(feature = "metal")]
            v_chunk_tables: Vec::new(),
            #[cfg(feature = "metal")]
            blocks_per_chunk: 0,
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
            #[cfg(feature = "metal")]
            k_chunks: Vec::new(),
            #[cfg(feature = "metal")]
            v_chunks: Vec::new(),
            #[cfg(feature = "metal")]
            k_chunk_tables: Vec::new(),
            #[cfg(feature = "metal")]
            v_chunk_tables: Vec::new(),
            #[cfg(feature = "metal")]
            blocks_per_chunk: 0,
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

    /// Allocate a **reactive (chunked)** metal KV pool: per layer per
    /// K/V, a series of `blocks_per_chunk`-block chunk buffers plus one
    /// chunk-address table buffer. RAM-neutral vs the single-buffer
    /// `new` (the chunk bytes sum to the same total, last chunk
    /// partial), but it lets the `MTLResidencySet` track only the
    /// chunks that back live KV. The single-buffer fields
    /// (`k_caches` / `_k_ptrs`) stay empty — metal binds chunk *tables*
    /// at the `KvCacheK/V` slots (see [`Self::k_chunk_table_mem`]), not
    /// the cache buffers, and never reads the `k_cache()` TensorView.
    ///
    /// `alloc_chunk(bytes)` allocates a StorageModePrivate chunk data
    /// buffer (and should insert it into the device residency set);
    /// `alloc_table(bytes)` allocates a StorageModeShared table buffer
    /// (host-writable via [`Self::fill_chunk_tables`], also residency-
    /// inserted). The chunk gpuAddresses are written into the tables by
    /// a later `fill_chunk_tables` call (the caller supplies the
    /// metal-specific `gpuAddress` accessor so this crate stays
    /// objc2-free).
    ///
    /// # Safety
    /// Caller must ensure the metal device is initialized on this
    /// thread.
    #[cfg(feature = "metal")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_metal_chunked(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        blocks_per_chunk: usize,
        // Chunks to allocate up front. `1` = reactive/lazy (grow on
        // demand via `grow_to_cover`); `usize::MAX` = eager (allocate
        // the whole pool now — used for the draft pool, whose spec-decode
        // forward path has no growth hook).
        initial_chunks: usize,
        mut alloc_chunk: impl FnMut(usize) -> Result<RawGpuMem>,
        mut alloc_table: impl FnMut(usize) -> Result<RawGpuMem>,
    ) -> Result<Self> {
        assert!(blocks_per_chunk > 0, "blocks_per_chunk must be non-zero");
        if dtype.is_fp8() {
            anyhow::bail!("new_metal_chunked: FP8 cache dtype is cuda-only");
        }
        let num_chunks = num_blocks.div_ceil(blocks_per_chunk);
        let elem = dtype.size_bytes();
        // Elements per paged block = kv_blk_stride (num_kv_heads *
        // block_size * head_dim). A chunk holds `blocks_in_chunk` of these.
        let per_block_elems = num_kv_heads * block_size * head_dim;

        let mut k_chunks: Vec<Vec<RawGpuMem>> = Vec::with_capacity(num_layers);
        let mut v_chunks: Vec<Vec<RawGpuMem>> = Vec::with_capacity(num_layers);
        let mut k_chunk_tables: Vec<RawGpuMem> = Vec::with_capacity(num_layers);
        let mut v_chunk_tables: Vec<RawGpuMem> = Vec::with_capacity(num_layers);

        // REACTIVE (2b): allocate only the FIRST chunk up front; the
        // rest grow on demand via `grow_one_chunk` as sequences fill
        // blocks (driven by the worker's per-step high-water-mark). The
        // chunk-address tables are allocated at FULL size (one u64 slot
        // per potential chunk — tiny, ~8 B/chunk) and bound once; only
        // slot 0 is filled here, the rest filled lazily on growth. This
        // is what drops startup RAM from ~the whole pool to weights +
        // one chunk.
        // At least one chunk (if the pool is non-empty), at most all.
        let initial_chunks = initial_chunks.clamp(num_chunks.min(1), num_chunks);
        for _layer in 0..num_layers {
            let mut kc = Vec::with_capacity(num_chunks);
            let mut vc = Vec::with_capacity(num_chunks);
            for chunk in 0..initial_chunks {
                // Chunk may be partial so chunk bytes sum to exactly
                // num_blocks (no over-allocation).
                let blocks_here = blocks_per_chunk.min(num_blocks - chunk * blocks_per_chunk);
                let bytes = blocks_here * per_block_elems * elem;
                kc.push(alloc_chunk(bytes)?);
                vc.push(alloc_chunk(bytes)?);
            }
            k_chunks.push(kc);
            v_chunks.push(vc);
            // Full-size address tables (one u64 slot per potential chunk).
            k_chunk_tables.push(alloc_table(num_chunks * 8)?);
            v_chunk_tables.push(alloc_table(num_chunks * 8)?);
        }

        let initial_mb =
            (2 * num_layers * initial_chunks * blocks_per_chunk * per_block_elems * elem) as f64
                / (1024.0 * 1024.0);
        let ceiling_mb =
            (2 * num_layers * num_blocks * per_block_elems * elem) as f64 / (1024.0 * 1024.0);
        tracing::info!(
            "KvCachePool(metal chunked, reactive): {num_layers} layers, {initial_chunks}/{num_chunks} \
             chunks resident at init ({initial_mb:.0} MB) — grows on demand up to {num_blocks} \
             blocks ({ceiling_mb:.0} MB ceiling) at {blocks_per_chunk} blocks/chunk ({dtype})"
        );

        Ok(Self {
            k_caches: Vec::new(),
            v_caches: Vec::new(),
            _k_ptrs: Vec::new(),
            _v_ptrs: Vec::new(),
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            num_layers,
            cache_dtype: dtype,
            #[cfg(feature = "cuda")]
            k_scale_ptrs: Vec::new(),
            #[cfg(feature = "cuda")]
            v_scale_ptrs: Vec::new(),
            block_is_unrotated: vec![false; num_blocks],
            block_is_span: vec![false; num_blocks],
            #[cfg(feature = "cuda")]
            block_unrotated_gpu_ptr: None,
            #[cfg(feature = "cuda")]
            block_span_gpu_ptr: None,
            k_chunks,
            v_chunks,
            k_chunk_tables,
            v_chunk_tables,
            blocks_per_chunk,
        })
    }

    /// Per-layer K chunk-address *table* backing memory (the buffer the
    /// kernels bind at the `KvCacheK` slot — a device array of chunk
    /// gpuAddresses, not the cache data). Empty unless built via
    /// [`Self::new_metal_chunked`].
    #[cfg(feature = "metal")]
    pub fn k_chunk_table_mem(&self, layer: usize) -> &RawGpuMem {
        &self.k_chunk_tables[layer]
    }

    /// Per-layer V chunk-address table. See [`Self::k_chunk_table_mem`].
    #[cfg(feature = "metal")]
    pub fn v_chunk_table_mem(&self, layer: usize) -> &RawGpuMem {
        &self.v_chunk_tables[layer]
    }

    /// Blocks backed by one physical chunk buffer (0 unless chunked).
    #[cfg(feature = "metal")]
    pub fn blocks_per_chunk(&self) -> usize {
        self.blocks_per_chunk
    }

    /// Write each chunk's `gpuAddress` into its per-layer chunk-address
    /// table (StorageModeShared, host-writable). `gpu_addr` extracts the
    /// 64-bit GPU virtual address of a chunk buffer — the caller
    /// supplies it (`|m| m.buffer().gpuAddress()`) so this crate needs
    /// no objc2-metal dependency. Call once after
    /// [`Self::new_metal_chunked`] and (in later phases) again whenever
    /// the chunk set changes.
    #[cfg(feature = "metal")]
    pub fn fill_chunk_tables(&self, gpu_addr: impl Fn(&RawGpuMem) -> u64) {
        for layer in 0..self.num_layers {
            let kt = self.k_chunk_tables[layer].ptr() as *mut u64;
            let vt = self.v_chunk_tables[layer].ptr() as *mut u64;
            for (c, buf) in self.k_chunks[layer].iter().enumerate() {
                // Safety: table is a StorageModeShared buffer of
                // `num_chunks` u64 slots; `c` is in range by construction.
                unsafe {
                    kt.add(c).write(gpu_addr(buf));
                }
            }
            for (c, buf) in self.v_chunks[layer].iter().enumerate() {
                unsafe {
                    vt.add(c).write(gpu_addr(buf));
                }
            }
        }
    }

    /// Total chunks this pool can grow to (`ceil(num_blocks / BPC)`).
    #[cfg(feature = "metal")]
    pub fn num_chunks_total(&self) -> usize {
        if self.blocks_per_chunk == 0 {
            0
        } else {
            self.num_blocks.div_ceil(self.blocks_per_chunk)
        }
    }

    /// Number of paged blocks currently backed by allocated chunks
    /// (`allocated_chunks * BPC`, capped at `num_blocks`). The scheduler
    /// may hand out any block id `< num_blocks`; the worker must
    /// `grow_to_cover` before a forward references a block ≥ this.
    #[cfg(feature = "metal")]
    pub fn allocated_blocks(&self) -> usize {
        let chunks = self.k_chunks.first().map_or(0, |v| v.len());
        (chunks * self.blocks_per_chunk).min(self.num_blocks)
    }

    /// Reactive shrink (2c): drop all chunks past the first `keep`,
    /// returning the freed K+V chunk buffers (all layers) so the caller
    /// can `residency.remove` + `commit` them BEFORE they're dropped
    /// (freed). The chunk-address table entries for the dropped chunks
    /// become stale but are never read — a later `grow_to_cover`
    /// re-allocates from the shrunk length and overwrites them. Only
    /// safe to call when no live block falls in a dropped chunk (e.g.
    /// the batch is fully idle); the caller owns that invariant.
    #[cfg(feature = "metal")]
    pub fn shrink_to_chunks(&mut self, keep: usize) -> Vec<RawGpuMem> {
        let keep = keep.max(1); // always retain chunk 0
        let cur = self.k_chunks.first().map_or(0, |v| v.len());
        if keep >= cur {
            return Vec::new();
        }
        let mut freed = Vec::with_capacity((cur - keep) * 2 * self.num_layers);
        for layer in 0..self.num_layers {
            while self.k_chunks[layer].len() > keep {
                freed.push(self.k_chunks[layer].pop().expect("k chunk"));
                freed.push(self.v_chunks[layer].pop().expect("v chunk"));
            }
        }
        freed
    }

    /// Grow the pool until `block_id` is backed by an allocated chunk.
    /// Allocates each missing chunk's K+V buffers for every layer (via
    /// `alloc_chunk`, which must residency-insert) and writes their
    /// `gpuAddress` (via `gpu_addr`) into the per-layer chunk tables.
    /// Returns the number of chunks newly allocated (0 = already
    /// covered); the caller must `residency.commit()` once if > 0 before
    /// dispatching, so the new chunk pages are wired for the bindless
    /// deref. No-op for non-chunked (cuda / single-buffer) pools.
    #[cfg(feature = "metal")]
    pub fn grow_to_cover(
        &mut self,
        block_id: usize,
        mut alloc_chunk: impl FnMut(usize) -> Result<RawGpuMem>,
        gpu_addr: impl Fn(&RawGpuMem) -> u64,
    ) -> Result<usize> {
        if self.blocks_per_chunk == 0 || self.k_chunks.is_empty() {
            return Ok(0); // not a chunked pool
        }
        let target_block = block_id.min(self.num_blocks.saturating_sub(1));
        let num_chunks = self.num_chunks_total();
        let elem = self.cache_dtype.size_bytes();
        let per_block_elems = self.num_kv_heads * self.block_size * self.head_dim;
        let mut grew = 0usize;
        // All layers grow in lockstep, so chunk count == k_chunks[0].len().
        while self.allocated_blocks() <= target_block {
            let next = self.k_chunks[0].len();
            if next >= num_chunks {
                break;
            }
            let blocks_here = self
                .blocks_per_chunk
                .min(self.num_blocks - next * self.blocks_per_chunk);
            let bytes = blocks_here * per_block_elems * elem;
            for layer in 0..self.num_layers {
                let kc = alloc_chunk(bytes)?;
                let vc = alloc_chunk(bytes)?;
                let ka = gpu_addr(&kc);
                let va = gpu_addr(&vc);
                // Write the chunk gpuAddresses into the (already-bound)
                // per-layer address tables at slot `next`.
                unsafe {
                    (self.k_chunk_tables[layer].ptr() as *mut u64)
                        .add(next)
                        .write(ka);
                    (self.v_chunk_tables[layer].ptr() as *mut u64)
                        .add(next)
                        .write(va);
                }
                self.k_chunks[layer].push(kc);
                self.v_chunks[layer].push(vc);
            }
            grew += 1;
        }
        Ok(grew)
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
