// SPDX-License-Identifier: Apache-2.0
//! Safetensors weight loading — pipelined from CPU to GPU.
//!
//! Weights are memory-mapped on CPU. On first access, the OS pages data in from
//! disk. We beat Python vLLM's default (serial mmap + synchronous H2D) with:
//!
//! 1. **madvise(WILLNEED)** on every shard at mmap time — OS starts prefetching
//!    all pages from disk immediately, overlapping I/O across shards.
//! 2. **Parallel shard loading** — multi-shard models parse headers concurrently.
//! 3. **Background pre-cast pipeline** — a thread pool pre-faults mmap pages and
//!    casts float tensors (F32→BF16/F16) into pinned host buffers ahead of
//!    `take()` calls. The main thread just enqueues H2D DMAs from ready buffers,
//!    overlapping CPU work with PCIe transfers.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use cudarc::driver::sys::CUstream;

use crate::driver;
use crate::dtype::DType;
use crate::tensor::GpuTensor;

// ---------------------------------------------------------------------------
// DType conversion
// ---------------------------------------------------------------------------

/// Map safetensors dtype string to our DType.
fn safetensors_dtype(dtype: safetensors::Dtype) -> Result<DType> {
    match dtype {
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::I64 => Ok(DType::I64),
        safetensors::Dtype::U32 => Ok(DType::U32),
        safetensors::Dtype::I32 => Ok(DType::I32),
        safetensors::Dtype::U8 => Ok(DType::U8),
        safetensors::Dtype::F8_E4M3 => Ok(DType::Fp8E4m3),
        other => bail!("unsupported safetensors dtype: {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// CPU dtype conversion helpers (for LoRA merging)
// ---------------------------------------------------------------------------

/// Read raw bytes in `dtype` into a pre-allocated f32 slice.
fn read_to_f32(data: &[u8], dtype: DType, out: &mut [f32]) {
    match dtype {
        DType::F32 => {
            let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, out.len()) };
            out.copy_from_slice(src);
        }
        DType::F16 => {
            let src =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::f16, out.len()) };
            for (s, d) in src.iter().zip(out.iter_mut()) {
                *d = s.to_f32();
            }
        }
        DType::BF16 => {
            let src = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, out.len())
            };
            for (s, d) in src.iter().zip(out.iter_mut()) {
                *d = s.to_f32();
            }
        }
        _ => panic!("read_to_f32: unsupported dtype {dtype}"),
    }
}

/// Write f32 values back to bytes in the given dtype.
fn write_from_f32(data: &[f32], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => {
            let mut out = vec![0u8; data.len() * 4];
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, data.len()) };
            dst.copy_from_slice(data);
            out
        }
        DType::F16 => {
            let mut out = vec![0u8; data.len() * 2];
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u16, data.len()) };
            for (s, d) in data.iter().zip(dst.iter_mut()) {
                *d = half::f16::from_f32(*s).to_bits();
            }
            out
        }
        DType::BF16 => {
            let mut out = vec![0u8; data.len() * 2];
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u16, data.len()) };
            for (s, d) in data.iter().zip(dst.iter_mut()) {
                *d = half::bf16::from_f32(*s).to_bits();
            }
            out
        }
        _ => panic!("write_from_f32: unsupported dtype {dtype}"),
    }
}

// ---------------------------------------------------------------------------
// CpuTensorRef — a reference to tensor data in a mmap'd safetensors file
// ---------------------------------------------------------------------------

/// A CPU-side reference to tensor data — either mmap'd (read-only) or owned
/// (e.g. after LoRA merging).
struct CpuTensorRef {
    /// The mmap that backs this tensor (None for owned data).
    mmap: Option<Arc<memmap2::Mmap>>,
    /// Byte offset within the mmap where tensor data starts.
    data_offset: usize,
    /// Size of tensor data in bytes.
    size_bytes: usize,
    shape: Vec<usize>,
    dtype: DType,
    /// Owned data buffer (used for merged weights). When set, `data()` returns
    /// this instead of the mmap slice.
    owned: Option<Arc<Vec<u8>>>,
}

impl CpuTensorRef {
    fn data(&self) -> &[u8] {
        if let Some(ref buf) = self.owned {
            buf.as_slice()
        } else {
            let mmap = self
                .mmap
                .as_ref()
                .expect("CpuTensorRef: no mmap or owned data");
            &mmap[self.data_offset..self.data_offset + self.size_bytes]
        }
    }
}

// ---------------------------------------------------------------------------
// GpuWeights
// ---------------------------------------------------------------------------

/// Parse a single shard file into a map of tensor references.
///
/// Mmaps the file, issues madvise(WILLNEED) + madvise(SEQUENTIAL) to trigger
/// OS readahead, and parses the safetensors header. Returns tensor references
/// pointing into the mmap — no data is copied.
///
/// This is a free function (not `&mut self`) so it can be called from parallel
/// threads during multi-shard loading.
fn load_shard_into_map(path: &Path) -> Result<(HashMap<String, CpuTensorRef>, Arc<memmap2::Mmap>)> {
    let file = std::fs::File::open(path)?;
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file) }?);

    // Tell the kernel to start paging in the entire shard from disk.
    // This overlaps disk I/O with header parsing and subsequent shard loads.
    #[cfg(unix)]
    unsafe {
        libc::madvise(
            mmap.as_ptr() as *mut libc::c_void,
            mmap.len(),
            libc::MADV_WILLNEED,
        );
        // Sequential access hint for better readahead chunk sizes.
        libc::madvise(
            mmap.as_ptr() as *mut libc::c_void,
            mmap.len(),
            libc::MADV_SEQUENTIAL,
        );
    }

    // Parse safetensors header to find tensor offsets.
    let st = safetensors::SafeTensors::deserialize(&mmap)
        .map_err(|e| anyhow::anyhow!("{}: {}", path.display(), e))?;

    let mut tensors = HashMap::new();
    for name in st.names() {
        let view = st
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("{}: {}", name, e))?;
        let dtype = safetensors_dtype(view.dtype())?;
        let data = view.data();
        let size_bytes = data.len();
        let shape: Vec<usize> = view.shape().to_vec();

        let data_offset = data.as_ptr() as usize - mmap.as_ptr() as usize;

        tensors.insert(
            name.to_string(),
            CpuTensorRef {
                mmap: Some(Arc::clone(&mmap)),
                data_offset,
                size_bytes,
                shape,
                dtype,
                owned: None,
            },
        );
    }

    tracing::info!(
        "Parsed shard {}: {} tensors (mmap + madvise WILLNEED)",
        path.display(),
        tensors.len(),
    );

    Ok((tensors, mmap))
}

// ---------------------------------------------------------------------------
// Pre-cast pipeline — background thread pre-faults + casts tensors into pinned
// buffers so take() just enqueues a DMA from already-ready pinned memory.
// ---------------------------------------------------------------------------

/// Background worker: iterates through tensors, pre-faults mmap pages, casts
/// float data into per-tensor pinned buffers, and stores results in `state.ready`.
fn precast_worker(
    state: Arc<PrecastState>,
    target_dtype: Option<DType>,
    work: Vec<(String, Arc<memmap2::Mmap>, usize, usize, DType)>,
) {
    let mut precast_count = 0usize;
    let mut prefault_count = 0usize;

    for (name, mmap, data_offset, size_bytes, dtype) in &work {
        if state.shutdown.load(Ordering::Relaxed) {
            break;
        }

        let data = &mmap[*data_offset..*data_offset + *size_bytes];

        // Determine if this tensor needs casting.
        let needs_cast = match target_dtype {
            Some(target) => {
                matches!(dtype, DType::F32 | DType::F16 | DType::BF16) && *dtype != target
            }
            None => false,
        };

        if needs_cast {
            let target = target_dtype.unwrap();
            // Cast into a freshly allocated pinned buffer.
            match cast_into_pinned(data, *dtype, target) {
                Ok(entry) => {
                    state.ready.lock().unwrap().insert(name.clone(), entry);
                    precast_count += 1;
                }
                Err(e) => {
                    // Non-fatal — take() will fall back to synchronous path.
                    tracing::debug!("Precast failed for {name}: {e}");
                }
            }
        } else {
            // No casting needed, but pre-fault the mmap pages by reading
            // through the data. This ensures pages are in the page cache
            // by the time take() does the H2D DMA.
            prefault_pages(data);
            prefault_count += 1;
        }
    }

    tracing::info!(
        "Precast pipeline done: {precast_count} tensors cast into pinned buffers, \
         {prefault_count} tensors pre-faulted"
    );
}

/// Pre-fault mmap pages by reading through the data at page-stride intervals.
/// This triggers page faults now so take() doesn't block on disk I/O later.
fn prefault_pages(data: &[u8]) {
    // Read one byte per page (4KB) to fault each page into the page cache.
    // The volatile read prevents the compiler from optimizing this away.
    let page_size = 4096;
    let mut offset = 0;
    while offset < data.len() {
        unsafe {
            std::ptr::read_volatile(&data[offset]);
        }
        offset += page_size;
    }
}

/// Cast tensor data into a new pinned host buffer.
fn cast_into_pinned(data: &[u8], src_dtype: DType, target: DType) -> Result<PrecastEntry> {
    let numel = data.len() / src_dtype.size_bytes();
    let cast_size = numel * target.size_bytes();

    // Allocate pinned host memory for this tensor.
    let alloc_size = cast_size.next_power_of_two().max(4096);
    let pinned_ptr = unsafe { driver::mem_alloc_host(alloc_size) }
        .map_err(|e| anyhow::anyhow!("pinned alloc for precast: {e}"))?;

    // Dispatch the cast.
    match (src_dtype, target) {
        (DType::F32, DType::BF16) => {
            let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, numel) };
            let dst = unsafe { std::slice::from_raw_parts_mut(pinned_ptr as *mut u16, numel) };
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = half::bf16::from_f32(*s).to_bits();
            }
        }
        (DType::F32, DType::F16) => {
            let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, numel) };
            let dst = unsafe { std::slice::from_raw_parts_mut(pinned_ptr as *mut u16, numel) };
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = half::f16::from_f32(*s).to_bits();
            }
        }
        (DType::F16, DType::BF16) => {
            let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, numel) };
            let dst = unsafe { std::slice::from_raw_parts_mut(pinned_ptr as *mut u16, numel) };
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = half::bf16::from_f32(half::f16::from_bits(*s).to_f32()).to_bits();
            }
        }
        (DType::BF16, DType::F16) => {
            let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, numel) };
            let dst = unsafe { std::slice::from_raw_parts_mut(pinned_ptr as *mut u16, numel) };
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = half::f16::from_f32(half::bf16::from_bits(*s).to_f32()).to_bits();
            }
        }
        (DType::BF16 | DType::F16, DType::F32) => {
            let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, numel) };
            let dst = unsafe { std::slice::from_raw_parts_mut(pinned_ptr as *mut f32, numel) };
            if src_dtype == DType::BF16 {
                for (s, d) in src.iter().zip(dst.iter_mut()) {
                    *d = half::bf16::from_bits(*s).to_f32();
                }
            } else {
                for (s, d) in src.iter().zip(dst.iter_mut()) {
                    *d = half::f16::from_bits(*s).to_f32();
                }
            }
        }
        _ => {
            // Free and bail — shouldn't happen for float types.
            unsafe { driver::mem_free_host(pinned_ptr).ok() };
            bail!("unhandled cast: {src_dtype:?} → {target:?}");
        }
    }

    Ok(PrecastEntry {
        pinned_ptr,
        size_bytes: cast_size,
        dtype: target,
    })
}

/// A pre-cast tensor ready for H2D DMA. Data lives in a pinned host buffer.
struct PrecastEntry {
    /// Pinned host buffer containing the (possibly cast) tensor data.
    pinned_ptr: *mut u8,
    /// Size of valid data in bytes.
    size_bytes: usize,
    /// The effective dtype after casting.
    dtype: DType,
}

// Safety: pinned host memory is accessible from any thread.
unsafe impl Send for PrecastEntry {}

/// Shared state for the pre-cast pipeline.
struct PrecastState {
    /// Pre-cast tensors ready for take(). Protected by mutex — contention is
    /// low because the producer adds entries one at a time and the consumer
    /// (take()) removes them.
    ready: Mutex<HashMap<String, PrecastEntry>>,
    /// Signal for the background thread to stop (e.g. on drop).
    shutdown: AtomicBool,
}

/// Model weights loaded from CPU (mmap) to GPU with pipelined pre-casting.
///
/// Weights are memory-mapped on CPU with madvise(WILLNEED) to trigger OS
/// readahead. When `start_precast()` is called, a background thread pool
/// pre-faults mmap pages and casts float tensors into pinned host buffers.
/// `take()` checks for pre-cast data first — if ready, it just enqueues an
/// async H2D DMA without blocking on page faults or CPU casting.
///
/// GPU memory allocated by `take()` is NOT freed on drop — ownership transfers
/// to the caller (model layers).
pub struct GpuWeights {
    /// Per-tensor CPU references, keyed by tensor name.
    tensors: HashMap<String, CpuTensorRef>,
    /// Stream used for H2D copies.
    stream: CUstream,
    /// Target dtype for floating-point weights. When set, F32 weights are cast
    /// to this dtype on CPU before H2D copy.
    target_dtype: Option<DType>,
    /// Reusable pinned host buffer for synchronous dtype casting (fallback
    /// when precast pipeline hasn't processed a tensor yet).
    /// (ptr, capacity_bytes). Grown as needed, never shrunk.
    cast_pinned: (*mut u8, usize),
    /// Pre-cast pipeline state, shared with background thread.
    precast: Option<Arc<PrecastState>>,
    /// Join handle for the background precast thread.
    precast_handle: Option<std::thread::JoinHandle<()>>,
    /// All GPU allocations made by `take()` / `take_into()` / `take_shard()`.
    /// Tracked so the caller can free weight memory on sleep without walking
    /// model structs. RAII: `RawGpuMem` calls `driver::mem_free` on drop.
    gpu_allocs: Vec<crate::alloc::RawGpuMem>,
    /// Keep mmaps alive for the lifetime of GpuWeights.
    ///
    /// `take()` and `take_into()` use `memcpy_htod_async` which reads from
    /// mmap'd memory asynchronously. The `CpuTensorRef` holding the `Arc<Mmap>`
    /// is dropped at the end of those functions. If that was the last reference,
    /// the mmap would be unmapped while the async DMA is still in flight,
    /// causing silent data corruption on GPU. This field retains all mmaps
    /// until the GpuWeights struct is dropped (after model loading completes).
    _mmaps: Vec<Arc<memmap2::Mmap>>,

    /// Quantized linear weights loaded from a GGUF file. Empty when
    /// the backing store is safetensors. Keyed by HF-style tensor
    /// name (e.g. `model.layers.0.self_attn.q_proj.weight`). The
    /// underlying GPU bytes are deliberately leaked for the model's
    /// lifetime; `take_quantized_linear` is non-destructive (returns
    /// a `GgmlStorage` view, leaves the entry in place) so multiple
    /// accessors can share the same source weight — see the docstring
    /// on that method for the workload-fanout case.
    quantized: HashMap<String, crate::ggml_quant::GgmlStorage>,

    /// Already-on-GPU dense weights from a GGUF file (norms,
    /// embeddings, lm_head — the GGUF loader dequantizes these at
    /// load time). Mirrors `quantized` — both are populated only on
    /// the GGUF path. Safetensors-backed `GpuWeights` populates
    /// `tensors` instead and uploads on `take()`.
    gguf_dense: HashMap<String, GpuTensor>,
}

// Safety: GPU device pointers accessible from any host thread.
unsafe impl Send for GpuWeights {}
unsafe impl Sync for GpuWeights {}

impl GpuWeights {
    /// Construct an empty `GpuWeights` — used by the GGUF loader in
    /// `ferrite-kernels`, which then populates `quantized` and
    /// `gguf_dense` directly via `quantized_map_mut` /
    /// `gguf_dense_map_mut`. Safetensors callers should use
    /// `from_dir` / `from_index` / `from_single_file` instead.
    pub fn empty(stream: CUstream) -> Self {
        Self {
            tensors: HashMap::new(),
            stream,
            target_dtype: None,
            cast_pinned: (std::ptr::null_mut(), 0),
            precast: None,
            precast_handle: None,
            gpu_allocs: Vec::new(),
            _mmaps: Vec::new(),
            quantized: HashMap::new(),
            gguf_dense: HashMap::new(),
        }
    }

    /// Push a `RawGpuMem` allocation onto the lifetime tracker. Used
    /// by the GGUF loader so quantized-weight GPU memory is freed
    /// alongside the rest of the `GpuWeights` allocations.
    pub fn push_gpu_alloc(&mut self, alloc: crate::alloc::RawGpuMem) {
        self.gpu_allocs.push(alloc);
    }

    /// Load all weights from a model directory (CPU-only — no GPU allocation).
    ///
    /// Handles both single-file (`model.safetensors`) and sharded
    /// (`model.safetensors.index.json`) models.
    pub fn from_dir(dir: impl AsRef<Path>, stream: CUstream) -> Result<Self> {
        let dir = dir.as_ref();
        let index_path = dir.join("model.safetensors.index.json");
        let single_path = dir.join("model.safetensors");

        if index_path.exists() {
            Self::from_index(&index_path, stream)
        } else if single_path.exists() {
            Self::from_single_file(&single_path, stream)
        } else {
            bail!("No safetensors files found in {}", dir.display());
        }
    }

    /// Unified entry point: loads either a safetensors model
    /// directory or a GGUF file. Detects format from the path
    /// extension (file ending in `.gguf` → GGUF; everything else
    /// is treated as a directory with safetensors).
    ///
    /// For safetensors, `dtype` / `alloc` / `tp_*` are stored as
    /// `target_dtype` (so `take()` casts F32 → model dtype) but
    /// otherwise unused at load time. For GGUF they're consumed
    /// immediately by the registered loader.
    ///
    /// # Safety
    /// Caller must hold a valid CUDA context and stream.
    pub unsafe fn from_path(
        path: impl AsRef<Path>,
        stream: CUstream,
        alloc: &mut crate::CachingAllocator,
        target_dtype: DType,
        tp_rank: usize,
        tp_world_size: usize,
    ) -> Result<Self> {
        let path = path.as_ref();
        let is_gguf_file = path.is_file() && path.extension().is_some_and(|e| e == "gguf");
        let gguf_in_dir = path.is_dir() && {
            std::fs::read_dir(path)
                .ok()
                .and_then(|mut it| {
                    it.find_map(|entry| {
                        let p = entry.ok()?.path();
                        (p.extension().is_some_and(|e| e == "gguf")).then_some(p)
                    })
                })
                .is_some()
        };
        if is_gguf_file {
            return unsafe {
                Self::from_gguf_file(path, target_dtype, alloc, stream, tp_rank, tp_world_size)
            };
        }
        if gguf_in_dir {
            // Find the .gguf file in the dir and load it.
            let gguf_path = std::fs::read_dir(path)?
                .find_map(|entry| {
                    let p = entry.ok()?.path();
                    (p.extension().is_some_and(|e| e == "gguf")).then_some(p)
                })
                .ok_or_else(|| anyhow::anyhow!("no .gguf file in directory"))?;
            return unsafe {
                Self::from_gguf_file(
                    &gguf_path,
                    target_dtype,
                    alloc,
                    stream,
                    tp_rank,
                    tp_world_size,
                )
            };
        }
        let mut gw = Self::from_dir(path, stream)?;
        gw.set_target_dtype(target_dtype);
        Ok(gw)
    }

    /// Load all weights from a single `.gguf` file via the
    /// inventory-registered GGUF loader (provided by
    /// `ferrite-kernels`). Eagerly uploads quantized linears and
    /// dequantizes norms / embeddings / lm_head on GPU.
    ///
    /// `tp_world_size = 1` is implemented today; >1 returns an
    /// error from the registered loader (the per-tensor block-
    /// aligned slicing pass is a follow-up).
    ///
    /// # Safety
    /// Caller must hold a valid CUDA context and stream. The
    /// returned `GpuWeights` retains GGUF tensor pointers for the
    /// lifetime of the model.
    pub unsafe fn from_gguf_file(
        path: impl AsRef<Path>,
        model_dtype: DType,
        alloc: &mut crate::CachingAllocator,
        stream: CUstream,
        tp_rank: usize,
        tp_world_size: usize,
    ) -> Result<Self> {
        let path = path.as_ref();
        let reg = crate::gguf_loader::registered_gguf_loader().ok_or_else(|| {
            anyhow::anyhow!(
                "GGUF loader not registered — link `ferrite-kernels` (which submits a \
                 `GgufLoaderRegistration` via inventory) into the consuming binary"
            )
        })?;
        unsafe { (reg.load)(path, model_dtype, alloc, stream, tp_rank, tp_world_size) }
    }

    /// Load from a single safetensors file (CPU-only).
    pub fn from_single_file(path: impl AsRef<Path>, stream: CUstream) -> Result<Self> {
        let path = path.as_ref();
        let mut gw = Self {
            tensors: HashMap::new(),
            stream,
            target_dtype: None,
            cast_pinned: (std::ptr::null_mut(), 0),
            precast: None,
            precast_handle: None,
            gpu_allocs: Vec::new(),
            _mmaps: Vec::new(),
            quantized: HashMap::new(),
            gguf_dense: HashMap::new(),
        };
        gw.load_shard(path)?;
        Ok(gw)
    }

    /// Load from a sharded model (index.json) (CPU-only).
    ///
    /// Multiple shards are loaded in parallel — each thread mmaps a shard,
    /// issues madvise(WILLNEED) to start readahead, and parses the header.
    /// This overlaps disk I/O across shards.
    pub fn from_index(index_path: impl AsRef<Path>, stream: CUstream) -> Result<Self> {
        let index_path = index_path.as_ref();
        let dir = index_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("index file has no parent dir"))?;

        let index_data = std::fs::read_to_string(index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_data)?;

        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow::anyhow!("missing weight_map in index"))?;

        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shard_files.sort();
        shard_files.dedup();

        let total = shard_files.len();

        if total <= 1 {
            // Single shard — no need for threading.
            let mut gw = Self {
                tensors: HashMap::new(),
                stream,
                target_dtype: None,
                cast_pinned: (std::ptr::null_mut(), 0),
                precast: None,
                precast_handle: None,
                gpu_allocs: Vec::new(),
                _mmaps: Vec::new(),
                quantized: HashMap::new(),
                gguf_dense: HashMap::new(),
            };
            if let Some(name) = shard_files.first() {
                gw.load_shard(&dir.join(name))?;
            }
            return Ok(gw);
        }

        // Multiple shards — load in parallel. Each thread mmaps a shard,
        // triggers madvise(WILLNEED), and parses the header. This overlaps
        // disk I/O and CPU-side parsing across shards.
        tracing::info!("Loading {total} shards in parallel");

        #[allow(clippy::type_complexity)]
        let shard_results: Vec<
            Result<(HashMap<String, CpuTensorRef>, Arc<memmap2::Mmap>)>,
        > = std::thread::scope(|scope| {
            let handles: Vec<_> = shard_files
                .iter()
                .map(|shard_name| {
                    let shard_path = dir.join(shard_name);
                    scope.spawn(move || load_shard_into_map(&shard_path))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut tensors = HashMap::new();
        let mut mmaps = Vec::with_capacity(shard_results.len());
        for result in shard_results {
            let (shard_tensors, mmap) = result?;
            tensors.extend(shard_tensors);
            mmaps.push(mmap);
        }

        Ok(Self {
            tensors,
            stream,
            target_dtype: None,
            cast_pinned: (std::ptr::null_mut(), 0),
            precast: None,
            precast_handle: None,
            gpu_allocs: Vec::new(),
            _mmaps: mmaps,
            quantized: HashMap::new(),
            gguf_dense: HashMap::new(),
        })
    }

    /// Parse a shard file, madvise(WILLNEED), and store tensor references.
    fn load_shard(&mut self, path: &Path) -> Result<()> {
        let (shard_tensors, mmap) = load_shard_into_map(path)?;
        self.tensors.extend(shard_tensors);
        self._mmaps.push(mmap);
        Ok(())
    }

    /// Ensure the pinned cast buffer has at least `needed` bytes.
    /// Grows by freeing + reallocating (pinned memory can't realloc).
    fn ensure_pinned_buf(&mut self, needed: usize) {
        if needed <= self.cast_pinned.1 {
            return;
        }
        // Free old buffer if any.
        if !self.cast_pinned.0.is_null() {
            unsafe { driver::mem_free_host(self.cast_pinned.0).ok() };
        }
        // Allocate new pinned buffer. Round up to 1MB alignment for reuse.
        let alloc_size = needed.next_power_of_two().max(1 << 20);
        let ptr = unsafe { driver::mem_alloc_host(alloc_size) }
            .expect("failed to allocate pinned host memory for dtype cast");
        self.cast_pinned = (ptr, alloc_size);
    }

    /// If target_dtype is set and the weight needs casting, cast on CPU into
    /// pinned host memory. Returns (data_ptr, size_bytes, effective_dtype).
    ///
    /// Only floating-point weights (F32, BF16, F16) are cast. Integer dtypes
    /// (I32, U32, I64) are left untouched — they're used for indices/metadata.
    fn maybe_cast_cpu(&mut self, cpu_ref: &CpuTensorRef) -> (*const u8, usize, DType) {
        let target = match self.target_dtype {
            Some(t) => t,
            None => return (cpu_ref.data().as_ptr(), cpu_ref.size_bytes, cpu_ref.dtype),
        };

        // Only cast floating-point types.
        let is_float = matches!(cpu_ref.dtype, DType::F32 | DType::F16 | DType::BF16);
        if !is_float || cpu_ref.dtype == target {
            return (cpu_ref.data().as_ptr(), cpu_ref.size_bytes, cpu_ref.dtype);
        }

        let numel = cpu_ref.size_bytes / cpu_ref.dtype.size_bytes();
        let cast_size = numel * target.size_bytes();
        self.ensure_pinned_buf(cast_size);

        let src = cpu_ref.data();
        let dst = self.cast_pinned.0;

        // Dispatch cast. The common case is F32 → BF16/F16.
        match (cpu_ref.dtype, target) {
            (DType::F32, DType::BF16) => {
                let src_f32 =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, numel) };
                let dst_u16 = unsafe { std::slice::from_raw_parts_mut(dst as *mut u16, numel) };
                for (s, d) in src_f32.iter().zip(dst_u16.iter_mut()) {
                    *d = half::bf16::from_f32(*s).to_bits();
                }
            }
            (DType::F32, DType::F16) => {
                let src_f32 =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, numel) };
                let dst_u16 = unsafe { std::slice::from_raw_parts_mut(dst as *mut u16, numel) };
                for (s, d) in src_f32.iter().zip(dst_u16.iter_mut()) {
                    *d = half::f16::from_f32(*s).to_bits();
                }
            }
            (DType::F16, DType::BF16) => {
                let src_u16 =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, numel) };
                let dst_u16 = unsafe { std::slice::from_raw_parts_mut(dst as *mut u16, numel) };
                for (s, d) in src_u16.iter().zip(dst_u16.iter_mut()) {
                    *d = half::bf16::from_f32(half::f16::from_bits(*s).to_f32()).to_bits();
                }
            }
            (DType::BF16, DType::F16) => {
                let src_u16 =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, numel) };
                let dst_u16 = unsafe { std::slice::from_raw_parts_mut(dst as *mut u16, numel) };
                for (s, d) in src_u16.iter().zip(dst_u16.iter_mut()) {
                    *d = half::f16::from_f32(half::bf16::from_bits(*s).to_f32()).to_bits();
                }
            }
            (DType::BF16 | DType::F16, DType::F32) => {
                let src_u16 =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, numel) };
                let dst_f32 = unsafe { std::slice::from_raw_parts_mut(dst as *mut f32, numel) };
                if cpu_ref.dtype == DType::BF16 {
                    for (s, d) in src_u16.iter().zip(dst_f32.iter_mut()) {
                        *d = half::bf16::from_bits(*s).to_f32();
                    }
                } else {
                    for (s, d) in src_u16.iter().zip(dst_f32.iter_mut()) {
                        *d = half::f16::from_bits(*s).to_f32();
                    }
                }
            }
            _ => unreachable!("unhandled cast: {:?} → {:?}", cpu_ref.dtype, target),
        }

        tracing::debug!(
            "Cast weight: {:?} → {:?} ({} elements)",
            cpu_ref.dtype,
            target,
            numel,
        );

        (dst as *const u8, cast_size, target)
    }

    /// Set the target dtype for floating-point weight casting.
    ///
    /// When set, floating-point weights (F32, F16, BF16) are cast to the target
    /// dtype on CPU before H2D copy. Integer weights are never cast.
    /// This matches Python vLLM where model parameters are initialized with
    /// `torch_dtype` and PyTorch auto-casts during weight loading.
    pub fn set_target_dtype(&mut self, dtype: DType) {
        self.target_dtype = Some(dtype);
    }

    /// Read back the configured target dtype (the same value
    /// [`Self::take_into`] casts floating-point weights to). `None` when no
    /// target is configured — in that case the caller should treat the
    /// on-disk dtype as authoritative. Used by stacked-tensor loaders
    /// (fused MoE expert stacks) that pre-allocate a single buffer and
    /// need to size it against the post-cast element width.
    pub fn target_dtype(&self) -> Option<DType> {
        self.target_dtype
    }

    /// Start the background pre-cast pipeline.
    ///
    /// Spawns a thread that iterates through all tensors (largest first),
    /// pre-faults their mmap pages (triggering disk I/O), and casts float
    /// tensors into individual pinned host buffers. This runs concurrently
    /// with model construction code calling `take()`.
    ///
    /// For tensors that don't need casting (already in target dtype), the
    /// thread still pre-faults the mmap pages so they're resident in the page
    /// cache by the time `take()` does the H2D DMA.
    ///
    /// Must be called after `set_target_dtype()`. Safe to call multiple times
    /// (subsequent calls are no-ops if already running).
    pub fn start_precast(&mut self) {
        if self.precast.is_some() {
            return; // Already running.
        }

        let target_dtype = self.target_dtype;

        // Collect tensor metadata for the background thread. We give it
        // clones of the CpuTensorRef data it needs (Arc<Mmap> is cheap to clone).
        // Sort largest first so the biggest tensors start pre-faulting early.
        let mut work: Vec<(String, Arc<memmap2::Mmap>, usize, usize, DType)> = self
            .tensors
            .iter()
            .filter_map(|(name, r)| {
                let mmap = r.mmap.as_ref()?.clone();
                Some((name.clone(), mmap, r.data_offset, r.size_bytes, r.dtype))
            })
            .collect();
        work.sort_by_key(|b| std::cmp::Reverse(b.3)); // Largest first.

        let state = Arc::new(PrecastState {
            ready: Mutex::new(HashMap::new()),
            shutdown: AtomicBool::new(false),
        });
        self.precast = Some(Arc::clone(&state));

        let handle = std::thread::Builder::new()
            .name("weight-precast".into())
            .spawn(move || {
                precast_worker(state, target_dtype, work);
            })
            .expect("failed to spawn precast thread");
        self.precast_handle = Some(handle);
    }

    /// Remove a tensor by name and copy it to GPU. Returns a GPU tensor.
    ///
    /// If the pre-cast pipeline has already processed this tensor, the H2D
    /// DMA uses the pre-cast pinned buffer (fast path — no page faults or
    /// CPU casting on the hot path). Otherwise falls back to synchronous
    /// cast from the mmap.
    ///
    /// GGUF backing: if the tensor is in `gguf_dense` (already on GPU,
    /// dequantized at load), this short-circuits and returns it
    /// directly — no upload, no precast.
    pub fn take(&mut self, name: &str) -> Result<GpuTensor> {
        // GGUF fast-path: norms / embeddings / lm_head are
        // pre-uploaded and dequantized by `load_gguf_into_weights`.
        if let Some(t) = self.gguf_dense.remove(name) {
            return Ok(t);
        }

        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;

        // Fast path: check if precast pipeline has this tensor ready.
        if let Some(entry) = self.take_precast(name) {
            let gpu_ptr = unsafe { driver::mem_alloc(entry.size_bytes)? };
            self.gpu_allocs
                .push(unsafe { crate::alloc::RawGpuMem::new(gpu_ptr, entry.size_bytes) });
            unsafe {
                driver::memcpy_htod_async(
                    gpu_ptr,
                    entry.pinned_ptr as *const u8,
                    entry.size_bytes,
                    self.stream,
                )?;
            }
            let tensor = unsafe { GpuTensor::new(gpu_ptr, &cpu_ref.shape, entry.dtype) };
            // Free pinned buffer after DMA completes. We synchronize the stream
            // to ensure the DMA has finished reading from the pinned buffer.
            unsafe {
                driver::stream_synchronize(self.stream)?;
                driver::mem_free_host(entry.pinned_ptr).ok();
            }
            return Ok(tensor);
        }

        // Slow path: synchronous pre-fault + cast + DMA.
        let (data, size_bytes, dtype) = self.maybe_cast_cpu(&cpu_ref);

        let gpu_ptr = unsafe { driver::mem_alloc(size_bytes)? };
        self.gpu_allocs
            .push(unsafe { crate::alloc::RawGpuMem::new(gpu_ptr, size_bytes) });

        // The cast destination is `self.cast_pinned`, a SHARED pinned
        // buffer reused across calls (see `ensure_pinned_buf`). Without
        // a sync here, the next `take` would overwrite this buffer
        // BEFORE the GPU has actually drained the async memcpy — every
        // queued copy then reads whichever cast we wrote last, and
        // many distinct GPU pointers end up with the same payload.
        // (The fast/precast path already syncs before freeing its
        // per-tensor pinned buffer; the slow path needs the same
        // serialization because it reuses one shared buffer.)
        let used_shared_pinned = data == self.cast_pinned.0 as *const u8;
        unsafe {
            driver::memcpy_htod_async(gpu_ptr, data, size_bytes, self.stream)?;
            if used_shared_pinned {
                driver::stream_synchronize(self.stream)?;
            }
        }

        Ok(unsafe { GpuTensor::new(gpu_ptr, &cpu_ref.shape, dtype) })
    }

    /// Same as [`take`] but creates the returned `GpuTensor` with a
    /// caller-provided shape instead of the on-disk shape. The two
    /// shapes must agree on total element count. Used to flatten
    /// `>MAX_DIMS`-dim tensors at load time — e.g. Qwen2-VL's
    /// `visual.patch_embed.proj.weight` which lands as a 5D Conv3d
    /// weight on disk but the runtime treats it as a 2D GEMM kernel
    /// (stride==kernel collapses the conv).
    ///
    /// [`take`]: Self::take
    pub fn take_with_shape(&mut self, name: &str, shape: &[usize]) -> Result<GpuTensor> {
        if let Some(t) = self.gguf_dense.remove(name) {
            // GGUF tensors don't usually overflow MAX_DIMS, but if a
            // future arch's GGUF spec lands a 5D tensor, the same
            // reshape applies. Same-size invariant.
            let new_numel: usize = shape.iter().product();
            anyhow::ensure!(
                t.numel() == new_numel,
                "take_with_shape: {name} numel mismatch (gguf {} vs new {new_numel})",
                t.numel(),
            );
            return Ok(unsafe { GpuTensor::new(t.raw_ptr(), shape, t.dtype()) });
        }

        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;
        let on_disk_numel: usize = cpu_ref.shape.iter().product();
        let new_numel: usize = shape.iter().product();
        anyhow::ensure!(
            on_disk_numel == new_numel,
            "take_with_shape: {name} on-disk shape {:?} (numel {on_disk_numel}) != new shape {:?} (numel {new_numel})",
            cpu_ref.shape,
            shape,
        );

        if let Some(entry) = self.take_precast(name) {
            let gpu_ptr = unsafe { driver::mem_alloc(entry.size_bytes)? };
            self.gpu_allocs
                .push(unsafe { crate::alloc::RawGpuMem::new(gpu_ptr, entry.size_bytes) });
            unsafe {
                driver::memcpy_htod_async(
                    gpu_ptr,
                    entry.pinned_ptr as *const u8,
                    entry.size_bytes,
                    self.stream,
                )?;
            }
            let tensor = unsafe { GpuTensor::new(gpu_ptr, shape, entry.dtype) };
            unsafe {
                driver::stream_synchronize(self.stream)?;
                driver::mem_free_host(entry.pinned_ptr).ok();
            }
            return Ok(tensor);
        }

        let (data, size_bytes, dtype) = self.maybe_cast_cpu(&cpu_ref);

        let gpu_ptr = unsafe { driver::mem_alloc(size_bytes)? };
        self.gpu_allocs
            .push(unsafe { crate::alloc::RawGpuMem::new(gpu_ptr, size_bytes) });

        unsafe {
            driver::memcpy_htod_async(gpu_ptr, data, size_bytes, self.stream)?;
            // The cast destination (`maybe_cast_cpu`) is a SHARED pinned
            // buffer (`self.cast_pinned`) that the next slow-path take()
            // will overwrite. Sync the stream before returning so the
            // next call's cast doesn't race the in-flight H2D from this
            // call. (The fast path already syncs after H2D so it can
            // free the per-tensor pinned buffer.)
            driver::stream_synchronize(self.stream)?;
        }

        Ok(unsafe { GpuTensor::new(gpu_ptr, shape, dtype) })
    }

    /// Copy a tensor's data directly to an offset within an existing GPU buffer.
    ///
    /// Used for fused weight loading (QKV, gate_up) — pre-allocate the fused
    /// tensor, then copy each component directly from CPU to the right offset.
    pub unsafe fn take_into(
        &mut self,
        name: &str,
        dst: *mut u8,
        stream: CUstream,
    ) -> Result<usize> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;

        // Fast path: use pre-cast data if available.
        if let Some(entry) = self.take_precast(name) {
            driver::memcpy_htod_async(
                dst,
                entry.pinned_ptr as *const u8,
                entry.size_bytes,
                stream,
            )?;
            let size = entry.size_bytes;
            // Sync before freeing the pinned source buffer.
            driver::stream_synchronize(stream)?;
            driver::mem_free_host(entry.pinned_ptr).ok();
            return Ok(size);
        }

        // Slow path. Same shared-cast-buffer race as `take()` —
        // sync before returning so the next slow-path take() doesn't
        // overwrite `self.cast_pinned` while this H2D is still reading
        // from it.
        let (data, size_bytes, _dtype) = self.maybe_cast_cpu(&cpu_ref);

        let used_shared_pinned = data == self.cast_pinned.0 as *const u8;
        driver::memcpy_htod_async(dst, data, size_bytes, stream)?;
        if used_shared_pinned {
            // Same race as `take`'s slow path — the next `take_into`
            // would overwrite `cast_pinned` before this async memcpy
            // drains. Sync to make this call effectively synchronous
            // when the cast destination is the shared buffer.
            driver::stream_synchronize(stream)?;
        }

        Ok(size_bytes)
    }

    /// Try to take a pre-cast entry for the given tensor name.
    fn take_precast(&self, name: &str) -> Option<PrecastEntry> {
        let state = self.precast.as_ref()?;
        let mut ready = state.ready.lock().ok()?;
        ready.remove(name)
    }

    /// Take a tensor and return its data as a CPU `Vec<f32>`.
    ///
    /// Useful for small per-head parameters (A_log, dt_bias, norm weights)
    /// that need to be kept on CPU or uploaded to GPU as f32.
    pub fn take_to_cpu_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;

        let data = cpu_ref.data();
        let num_elems: usize = cpu_ref.shape.iter().product();
        let mut result = Vec::with_capacity(num_elems);

        match cpu_ref.dtype {
            DType::F32 => {
                let src =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, num_elems) };
                result.extend_from_slice(src);
            }
            DType::F16 => {
                let src = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const half::f16, num_elems)
                };
                result.extend(src.iter().map(|v| v.to_f32()));
            }
            DType::BF16 => {
                let src = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, num_elems)
                };
                result.extend(src.iter().map(|v| v.to_f32()));
            }
            other => anyhow::bail!("take_to_cpu_f32: unsupported dtype {other}"),
        }

        Ok(result)
    }

    /// Get the shape and effective dtype of a tensor without loading it to GPU.
    ///
    /// If `target_dtype` is set and the tensor is a floating-point type, the
    /// returned dtype reflects the cast target (matching what `take`/`take_into`
    /// will produce). This ensures callers compute correct byte sizes for
    /// pre-allocated buffers.
    pub fn tensor_info(&self, name: &str) -> Option<(&[usize], DType)> {
        self.tensors.get(name).map(|r| {
            let effective_dtype = match self.target_dtype {
                Some(target)
                    if matches!(r.dtype, DType::F32 | DType::F16 | DType::BF16)
                        && r.dtype != target =>
                {
                    target
                }
                _ => r.dtype,
            };
            (r.shape.as_slice(), effective_dtype)
        })
    }

    /// Tensor shape lookup that checks all three backing maps —
    /// safetensors `tensors`, `gguf_dense`, and `quantized`. Used by
    /// the per-variant fingerprint sniff which needs to verify
    /// embedding / first-layer shapes regardless of backing store.
    /// Returns `Vec<usize>` to avoid borrow lifetime tangles across
    /// heterogeneous backings (CpuTensorRef carries usize, GpuTensor
    /// carries u32, GgmlStorage stores nrows/ncols).
    pub fn tensor_shape_any(&self, name: &str) -> Option<Vec<usize>> {
        if let Some(r) = self.tensors.get(name) {
            return Some(r.shape.clone());
        }
        if let Some(t) = self.gguf_dense.get(name) {
            return Some(t.shape().iter().map(|&d| d as usize).collect());
        }
        if let Some(s) = self.quantized.get(name) {
            // 2D row-major weight; nrows = out, ncols = in. Match
            // the safetensors layout convention.
            return Some(vec![s.nrows, s.ncols]);
        }
        None
    }

    /// Get a tensor by name (copies to GPU). For read-only access.
    ///
    /// WARNING: The returned GPU tensor is leaked — caller must arrange cleanup.
    /// Prefer `take()` which is more explicit about ownership transfer.
    pub fn get(&mut self, name: &str) -> Option<GpuTensor> {
        // Remove temporarily to satisfy borrow checker, then re-insert.
        let cpu_ref = self.tensors.remove(name)?;
        let (data, size_bytes, dtype) = self.maybe_cast_cpu(&cpu_ref);

        let gpu_ptr = unsafe { driver::mem_alloc(size_bytes).ok()? };
        self.gpu_allocs
            .push(unsafe { crate::alloc::RawGpuMem::new(gpu_ptr, size_bytes) });

        let used_shared_pinned = data == self.cast_pinned.0 as *const u8;
        unsafe {
            driver::memcpy_htod_async(gpu_ptr, data, size_bytes, self.stream).ok()?;
            if used_shared_pinned {
                driver::stream_synchronize(self.stream).ok()?;
            }
        }

        let shape = cpu_ref.shape.clone();
        self.tensors.insert(name.to_string(), cpu_ref);

        Some(unsafe { GpuTensor::new(gpu_ptr, &shape, dtype) })
    }

    /// Check if a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
            || self.quantized.contains_key(name)
            || self.gguf_dense.contains_key(name)
    }

    // ----------------------------------------------------------------------
    // GGUF backing-store accessors
    // ----------------------------------------------------------------------
    //
    // Populated by `from_gguf` (in `ferrite-kernels`, since it depends on
    // GGUF-specific kernels for dequantizing norms / embeddings). The
    // safetensors construction paths leave these maps empty.

    /// Get a quantized linear weight by HF tensor name. Returns
    /// `None` when the backing store is safetensors or the tensor
    /// is absent.
    ///
    /// **Non-destructive** — the entry stays in the map so multiple
    /// consumers can each obtain a `GgmlStorage` view of the same
    /// GPU buffer. Required because the same source weight (e.g.
    /// `model.layers.5.self_attn.k_proj.weight`) may be referenced
    /// by both a singleton accessor (prefill workload's
    /// `GgmlGemmImpl`) and a fused-QKV accessor (decode workload's
    /// `GgmlFusedQkvRopeCacheImpl`); the codegen emits both load
    /// paths and both must succeed. `GgmlStorage` is `Copy` and
    /// holds no ownership — the underlying GPU bytes are leaked
    /// for the model's lifetime, matching the existing pattern.
    ///
    /// The returned storage is valid as long as the `GpuWeights`
    /// (or its successor after `take_gpu_allocs`) is alive.
    pub fn take_quantized_linear(&mut self, name: &str) -> Option<crate::ggml_quant::GgmlStorage> {
        self.quantized.get(name).copied()
    }

    /// Take a dense (already dequantized) GGUF weight — norms,
    /// embeddings, lm_head. Returns `None` for the safetensors path.
    pub fn take_gguf_dense(&mut self, name: &str) -> Option<GpuTensor> {
        self.gguf_dense.remove(name)
    }

    /// True when this `GpuWeights` was populated from a GGUF file
    /// (i.e. has at least one entry in `quantized` or `gguf_dense`).
    /// Used by codegen to decide between the safetensors and GGUF
    /// load helpers.
    pub fn is_gguf(&self) -> bool {
        !self.quantized.is_empty() || !self.gguf_dense.is_empty()
    }

    /// True iff the named tensor is present as a GGUF-quantized
    /// linear (in `quantized`). Lets concat-loaders probe before
    /// committing to the GGUF byte-pack path vs the dense fallback.
    pub fn contains_quantized_linear(&self, name: &str) -> bool {
        self.quantized.contains_key(name)
    }

    /// True iff the named tensor is present in the GGUF-dense map
    /// (norms / embeddings / lm_head — pre-uploaded and dequantized).
    pub fn gguf_dense_contains(&self, name: &str) -> bool {
        self.gguf_dense.contains_key(name)
    }

    /// True iff the named tensor is present in the safetensors-backed
    /// `tensors` map (CPU-mmap, uploaded on `take`).
    pub fn safetensor_contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Iterator over GGUF-quantized linear tensor names. Diagnostic
    /// helper for `FERRITE_GGUF_TRACE`-style debug output.
    pub fn quantized_linear_names(&self) -> impl Iterator<Item = &String> {
        self.quantized.keys()
    }

    /// Direct access to the quantized-linear map for the GGUF
    /// loader's population step. Not part of the public API for
    /// model code — use `take_quantized_linear` from the model side.
    #[doc(hidden)]
    pub fn quantized_map_mut(&mut self) -> &mut HashMap<String, crate::ggml_quant::GgmlStorage> {
        &mut self.quantized
    }

    /// Direct access to the GGUF-dense map for the loader's
    /// population step. Same caveats as `quantized_map_mut`.
    #[doc(hidden)]
    pub fn gguf_dense_map_mut(&mut self) -> &mut HashMap<String, GpuTensor> {
        &mut self.gguf_dense
    }

    /// Synthesize N virtual per-slice entries from a packed safetensors
    /// source by taking an even row-wise split.
    ///
    /// `packed_prefix` is the full dotted path to the packed tensor (its
    /// `.weight` is at `{packed_prefix}.weight`; `.bias` is optional and
    /// split the same way). `split_targets` lists the sibling suffixes
    /// under the shared parent; each becomes a new entry at
    /// `{grandparent}.{target}.weight` whose `CpuTensorRef` points at the
    /// matching row range within the original mmap (no data is copied).
    /// The packed entry itself is removed on success.
    ///
    /// Returns `Ok(true)` when the split happened, `Ok(false)` when the
    /// packed tensor does not exist (caller should propagate the
    /// downstream `weight not found` error the normal way).
    ///
    /// Even-split convenience wrapper around
    /// `synthesize_packed_row_split_sizes`. Only valid when `total_rows`
    /// is divisible by `split_targets.len()` — e.g. MHA `qkv_proj`
    /// (`num_attention_heads == num_key_value_heads`) and every
    /// `gate_up_proj`. GQA checkpoints must go through the sized
    /// variant with explicit per-slice row counts.
    pub fn synthesize_packed_row_split(
        &mut self,
        packed_prefix: &str,
        split_targets: &[&str],
    ) -> Result<bool> {
        let packed_weight_name = format!("{packed_prefix}.weight");
        if !self.tensors.contains_key(&packed_weight_name) {
            return Ok(false);
        }
        let n = split_targets.len();
        if n == 0 {
            anyhow::bail!("synthesize_packed_row_split: empty split_targets");
        }
        let total_rows = self.tensors[&packed_weight_name].shape[0];
        if !total_rows.is_multiple_of(n) {
            anyhow::bail!(
                "packed source `{packed_weight_name}` rows ({total_rows}) not divisible by {n} slices; \
                 GQA-packed qkv needs `synthesize_packed_row_split_sizes` with explicit per-slice row counts",
            );
        }
        let rows_per_slice = total_rows / n;
        let sized: Vec<(&str, usize)> =
            split_targets.iter().map(|t| (*t, rows_per_slice)).collect();
        self.synthesize_packed_row_split_sizes(packed_prefix, &sized)
    }

    /// Tensor-parallel-aware split. Carves per-rank views out of a
    /// GGUF packed parent (e.g. Phi-3's `self_attn.qkv_proj` fused
    /// across q/k/v heads) so downstream `_sharded` loaders see slices
    /// that are already the right per-rank shape.
    ///
    /// At tp_world_size == 1, identical to
    /// [`Self::synthesize_packed_row_split_sizes`].
    ///
    /// At tp_world_size > 1 with a **quantized** parent (`ShardDim0`
    /// replicated across ranks by the GGUF loader because the fused
    /// parent's name isn't in the shard-kind rule table), carves each
    /// child slice at offset `slice_start + tp_rank * (slice_rows / tp)`
    /// with `slice_rows / tp` rows. Each child ends up per-rank sized
    /// and its `.weight` entry goes into `self.quantized` for the
    /// subsequent `_sharded` load helpers to consume as-is (they see
    /// the correct per-rank shape and skip further sharding).
    ///
    /// Safetensors parents at tp > 1 fall through to the unsharded
    /// carve: the CPU slices keep their full sizes and the downstream
    /// `Linear::load_sharded(dim=0, …)` does the per-rank `take_shard`.
    ///
    /// `split_targets` takes **full** per-slice row counts (what the
    /// unsharded manifest declares). Per-rank division is done inside
    /// this helper against `tp_world_size`; each `full_rows` must be
    /// divisible by `tp_world_size`, else bail.
    pub fn synthesize_packed_row_split_sizes_tp(
        &mut self,
        packed_prefix: &str,
        split_targets: &[(&str, usize)],
        tp_rank: usize,
        tp_world_size: usize,
    ) -> Result<bool> {
        if tp_world_size <= 1 {
            return self.synthesize_packed_row_split_sizes(packed_prefix, split_targets);
        }
        if tp_rank >= tp_world_size {
            anyhow::bail!(
                "synthesize_packed_row_split_sizes_tp: tp_rank ({tp_rank}) >= \
                 tp_world_size ({tp_world_size})"
            );
        }
        if split_targets.is_empty() {
            anyhow::bail!("synthesize_packed_row_split_sizes_tp: empty split_targets");
        }
        let grandparent = packed_prefix
            .rsplit_once('.')
            .map(|(p, _)| p)
            .ok_or_else(|| anyhow::anyhow!("packed prefix has no parent: {packed_prefix}"))?;
        let packed_weight_name = format!("{packed_prefix}.weight");

        // Quantized-parent path: Phi-3 family at tp > 1 ships the
        // fused `attn_qkv` / `ffn_up` parent as a block-quantized
        // GgmlStorage, replicated across ranks by the GGUF loader
        // (the fused-parent name isn't in `gguf_shard_kind_for_hf_name`).
        // Carve per-rank views directly here so `_sharded` helpers
        // below see children of the right shape.
        if self.quantized.contains_key(&packed_weight_name) {
            let packed = self
                .quantized
                .remove(&packed_weight_name)
                .expect("contains");
            let total_rows = packed.nrows;
            let hidden = packed.ncols;
            let full_sum: usize = split_targets.iter().map(|(_, r)| *r).sum();
            if full_sum != total_rows {
                anyhow::bail!(
                    "synthesize_packed_row_split_sizes_tp: `{packed_weight_name}` rows \
                     ({total_rows}) != sum of full split sizes ({full_sum}) across {:?}",
                    split_targets
                        .iter()
                        .map(|(s, r)| format!("{s}={r}"))
                        .collect::<Vec<_>>(),
                );
            }
            for (t, r) in split_targets {
                if !r.is_multiple_of(tp_world_size) {
                    anyhow::bail!(
                        "synthesize_packed_row_split_sizes_tp: slice `{t}` rows ({r}) \
                         not divisible by tp_world_size ({tp_world_size}) in \
                         `{packed_weight_name}`"
                    );
                }
            }
            let block_elems = packed.dtype.block_size();
            let type_size = packed.dtype.type_size();
            if !hidden.is_multiple_of(block_elems) {
                anyhow::bail!(
                    "synthesize_packed_row_split_sizes_tp: `{packed_weight_name}` ncols \
                     ({hidden}) not divisible by block_size ({block_elems}, dtype {:?})",
                    packed.dtype,
                );
            }
            let row_bytes = (hidden / block_elems) * type_size;
            let mut full_row_offset = 0usize;
            for (target, full_rows) in split_targets {
                let per_rank_rows = full_rows / tp_world_size;
                let rank_row_offset = full_row_offset + tp_rank * per_rank_rows;
                let slice_bytes = per_rank_rows * row_bytes;
                // Row-aligned byte offset inside the replicated parent
                // buffer. Each row is a whole number of GGML blocks
                // (validated above via `hidden % block_elems == 0`) so
                // the pointer arithmetic never straddles a block boundary.
                let child_ptr = unsafe { packed.ptr.add(rank_row_offset * row_bytes) };
                let child = crate::ggml_quant::GgmlStorage {
                    ptr: child_ptr,
                    len: slice_bytes,
                    dtype: packed.dtype,
                    nrows: per_rank_rows,
                    ncols: hidden,
                };
                let vname = format!("{grandparent}.{target}.weight");
                self.quantized.insert(vname, child);
                full_row_offset += full_rows;
            }
            return Ok(true);
        }

        // Safetensors parent: fall through to the unsharded carve.
        // Downstream `Linear::load_sharded(dim=0, …)` applies per-rank
        // `take_shard` to each full-size slice, yielding the same
        // per-rank shape the quantized path produces above.
        self.synthesize_packed_row_split_sizes(packed_prefix, split_targets)
    }

    /// Sized-split sibling of `synthesize_packed_row_split`. Takes
    /// per-slice row counts so GQA-packed qkv (q and kv slices differ
    /// in output dim) and any other non-even split can be materialized.
    ///
    /// `split_targets` is `&[(suffix, rows)]`; the sum of `rows` must
    /// equal the packed tensor's first dim. Bias, if present on the
    /// packed parent, is split the same way.
    ///
    /// Used by ferrite-forward's manifest-driven `__packed_splits__`
    /// prelude (codegen emits one call per layer × packed-prefix before
    /// any `Weights::load` field read). No-op when the packed tensor
    /// isn't present — returns `Ok(false)` so non-packed checkpoints
    /// (Llama, Mistral, …) fall through unharmed.
    ///
    /// Two backing maps are checked: dense `tensors` (safetensors mmap
    /// refs) and quantized `quantized` (GGUF `GgmlStorage` on GPU).
    /// GGUF checkpoints (Phi-3, etc.) ship the fused parent in the
    /// quantized map; the row-split there carves the existing GPU
    /// buffer into N children that share the same allocation — no
    /// memory doubling, no extra H2D, no dequant.
    pub fn synthesize_packed_row_split_sizes(
        &mut self,
        packed_prefix: &str,
        split_targets: &[(&str, usize)],
    ) -> Result<bool> {
        let packed_weight_name = format!("{packed_prefix}.weight");
        if split_targets.is_empty() {
            anyhow::bail!("synthesize_packed_row_split_sizes: empty split_targets");
        }
        let grandparent = packed_prefix
            .rsplit_once('.')
            .map(|(p, _)| p)
            .ok_or_else(|| anyhow::anyhow!("packed prefix has no parent: {packed_prefix}"))?;

        // GGUF quantized path — Phi-3 family ships fused qkv_proj /
        // gate_up_proj as block-quantized GgmlStorage. Each child is a
        // view into the parent's GPU buffer at a row-aligned byte
        // offset; the parent allocation is leaked-by-design (model
        // weights live for the model's lifetime, same as before this
        // split) so single-buffer ownership is preserved without any
        // refcount machinery.
        if self.quantized.contains_key(&packed_weight_name) {
            let packed = self
                .quantized
                .remove(&packed_weight_name)
                .expect("contains");
            let total_rows = packed.nrows;
            let hidden = packed.ncols;
            let sum_rows: usize = split_targets.iter().map(|(_, r)| *r).sum();
            if sum_rows != total_rows {
                anyhow::bail!(
                    "packed quantized source `{packed_weight_name}` rows ({total_rows}) != \
                     sum of split sizes ({sum_rows}) across {:?}",
                    split_targets
                        .iter()
                        .map(|(s, r)| format!("{s}={r}"))
                        .collect::<Vec<_>>(),
                );
            }
            let block_elems = packed.dtype.block_size();
            let type_size = packed.dtype.type_size();
            // Each row contributes hidden / block_elems blocks; rows are
            // stored contiguously, so a child at row offset `r` starts
            // at byte offset `r * (hidden / block_elems) * type_size`.
            // Row-alignment check: every quantized format we ship has
            // hidden % block_elems == 0 in practice (k-quants block
            // size 256, hidden ≥ 256 always; legacy quants block size
            // 32). If a future format violates this, error rather than
            // silently produce torn blocks.
            if !hidden.is_multiple_of(block_elems) {
                anyhow::bail!(
                    "quantized row-split unsafe: `{packed_weight_name}` has ncols ({hidden}) \
                     not divisible by `{}` block size ({block_elems})",
                    packed.dtype,
                );
            }
            let row_bytes = (hidden / block_elems) * type_size;
            let mut row_offset = 0usize;
            for (target, rows) in split_targets {
                let slice_bytes = *rows * row_bytes;
                let child_ptr = unsafe { packed.ptr.add(row_offset * row_bytes) };
                let child = crate::ggml_quant::GgmlStorage {
                    ptr: child_ptr,
                    len: slice_bytes,
                    dtype: packed.dtype,
                    nrows: *rows,
                    ncols: hidden,
                };
                let vname = format!("{grandparent}.{target}.weight");
                self.quantized.insert(vname, child);
                row_offset += rows;
            }
            return Ok(true);
        }

        if !self.tensors.contains_key(&packed_weight_name) {
            return Ok(false);
        }

        let packed_weight = self.tensors.remove(&packed_weight_name).expect("contains");
        if packed_weight.shape.len() != 2 {
            anyhow::bail!(
                "packed source `{packed_weight_name}` has rank {}, expected 2",
                packed_weight.shape.len(),
            );
        }
        let total_rows = packed_weight.shape[0];
        let hidden = packed_weight.shape[1];
        let sum_rows: usize = split_targets.iter().map(|(_, r)| *r).sum();
        if sum_rows != total_rows {
            anyhow::bail!(
                "packed source `{packed_weight_name}` rows ({total_rows}) != sum of split sizes ({sum_rows}) \
                 across {:?}",
                split_targets
                    .iter()
                    .map(|(s, r)| format!("{s}={r}"))
                    .collect::<Vec<_>>(),
            );
        }
        let elem = packed_weight.dtype.size_bytes();

        let mut row_offset = 0usize;
        for (target, rows) in split_targets {
            let slice_bytes = rows * hidden * elem;
            let vname = format!("{grandparent}.{target}.weight");
            let entry = CpuTensorRef {
                mmap: packed_weight.mmap.clone(),
                data_offset: packed_weight.data_offset + row_offset * hidden * elem,
                size_bytes: slice_bytes,
                shape: vec![*rows, hidden],
                dtype: packed_weight.dtype,
                owned: packed_weight.owned.clone(),
            };
            self.tensors.insert(vname, entry);
            row_offset += rows;
        }

        let packed_bias_name = format!("{packed_prefix}.bias");
        if let Some(packed_bias) = self.tensors.remove(&packed_bias_name) {
            if packed_bias.shape.len() != 1 || packed_bias.shape[0] != total_rows {
                anyhow::bail!(
                    "packed bias `{packed_bias_name}` shape {:?} inconsistent with weight rows {total_rows}",
                    packed_bias.shape,
                );
            }
            let belem = packed_bias.dtype.size_bytes();
            let mut row_offset = 0usize;
            for (target, rows) in split_targets {
                let bslice_bytes = rows * belem;
                let vname = format!("{grandparent}.{target}.bias");
                let entry = CpuTensorRef {
                    mmap: packed_bias.mmap.clone(),
                    data_offset: packed_bias.data_offset + row_offset * belem,
                    size_bytes: bslice_bytes,
                    shape: vec![*rows],
                    dtype: packed_bias.dtype,
                    owned: packed_bias.owned.clone(),
                };
                self.tensors.insert(vname, entry);
                row_offset += rows;
            }
        }

        Ok(true)
    }

    /// Number of loaded tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether no tensors are loaded.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Record an externally-made GPU allocation so it can be freed on sleep.
    ///
    /// Called by quantized weight loaders (AWQ, GPTQ, Marlin, etc.) that
    /// allocate GPU memory via `driver::mem_alloc` outside of `take()`.
    pub fn record_alloc(&mut self, ptr: *mut u8, size_bytes: usize) {
        self.gpu_allocs
            .push(unsafe { crate::alloc::RawGpuMem::new(ptr, size_bytes) });
    }

    /// Remove a previously-recorded allocation from tracking (e.g. after repack
    /// frees it separately). The removed `RawGpuMem` is leaked — caller is
    /// responsible for freeing the GPU memory.
    pub fn unrecord_alloc(&mut self, ptr: *mut u8) {
        if let Some(pos) = self.gpu_allocs.iter().position(|m| m.ptr() == ptr) {
            let removed = self.gpu_allocs.swap_remove(pos);
            removed.leak(); // prevent Drop from freeing — caller will free
        }
    }

    /// Drain all tracked GPU allocations. The caller takes ownership of the
    /// `RawGpuMem` wrappers — dropping them frees the GPU memory.
    pub fn take_gpu_allocs(&mut self) -> Vec<crate::alloc::RawGpuMem> {
        std::mem::take(&mut self.gpu_allocs)
    }

    /// Iterator over all tensor names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Strip a prefix from all tensor names (e.g. "model.").
    pub fn strip_prefix(&mut self, prefix: &str) {
        let stripped: HashMap<String, CpuTensorRef> = self
            .tensors
            .drain()
            .filter_map(|(name, tensor)| {
                name.strip_prefix(prefix)
                    .map(|rest| (rest.to_string(), tensor))
            })
            .collect();
        self.tensors = stripped;
    }

    /// Get the stream used for H2D copies.
    pub fn stream(&self) -> CUstream {
        self.stream
    }

    // -----------------------------------------------------------------------
    // LoRA weight merging (CPU-side, before H2D copy)
    // -----------------------------------------------------------------------

    /// Merge a LoRA adapter's A/B weight pairs into base weights on CPU.
    ///
    /// For each LoRA target module, computes `W_merged = W + scaling * B @ A`
    /// in f32 intermediate precision and replaces the mmap'd `CpuTensorRef`
    /// with an owned buffer containing the merged result.
    ///
    /// Must be called BEFORE `take()`/`take_into()` so that model construction
    /// picks up already-merged weights (including fused QKV / gate_up).
    ///
    /// Returns the number of weight tensors merged.
    pub fn merge_lora(&mut self, adapter_dir: &Path) -> Result<usize> {
        use vllm_model::lora::LoraAdapterConfig;

        // 1. Parse adapter config.
        let config_path = adapter_dir.join("adapter_config.json");
        let config = LoraAdapterConfig::from_file(&config_path)
            .map_err(|e| anyhow::anyhow!("LoRA config: {e}"))?;
        let scaling = config.scaling();

        // 2. Load adapter weights (CPU mmap).
        let st_path = adapter_dir.join("adapter_model.safetensors");
        if !st_path.exists() {
            bail!(
                "adapter_model.safetensors not found in {}",
                adapter_dir.display()
            );
        }
        let file = std::fs::File::open(&st_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }?;
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| anyhow::anyhow!("LoRA safetensors: {e}"))?;

        // 3. Group A/B pairs by layer prefix.
        //    PEFT names: base_model.model.{prefix}.lora_A.weight
        #[allow(clippy::type_complexity)]
        let mut pairs: HashMap<
            String,
            (Option<&[u8]>, Vec<usize>, Option<&[u8]>, Vec<usize>),
        > = HashMap::new();

        for name in st.names() {
            let view = st
                .tensor(name)
                .map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
            let (prefix, is_a) = if let Some(p) = name.strip_suffix(".lora_A.weight") {
                (p, true)
            } else if let Some(p) = name.strip_suffix(".lora_B.weight") {
                (p, false)
            } else {
                continue;
            };
            let clean = prefix.strip_prefix("base_model.model.").unwrap_or(prefix);
            let entry = pairs
                .entry(clean.to_string())
                .or_insert((None, vec![], None, vec![]));
            if is_a {
                entry.0 = Some(view.data());
                entry.1 = view.shape().to_vec();
            } else {
                entry.2 = Some(view.data());
                entry.3 = view.shape().to_vec();
            }
        }

        // 4. For each pair, merge into the base weight.
        let mut merged_count = 0usize;
        for (prefix, (a_data, a_shape, b_data, b_shape)) in &pairs {
            let a_data = match a_data {
                Some(d) => d,
                None => {
                    tracing::warn!("LoRA: missing lora_A for {prefix}, skipping");
                    continue;
                }
            };
            let b_data = match b_data {
                Some(d) => d,
                None => {
                    tracing::warn!("LoRA: missing lora_B for {prefix}, skipping");
                    continue;
                }
            };

            // Find matching base weight. The prefix should match a key in self.tensors
            // (after strip_prefix("model.") has been applied, or not).
            let base_name = if self.tensors.contains_key(&format!("{prefix}.weight")) {
                format!("{prefix}.weight")
            } else {
                tracing::debug!("LoRA: no base weight for {prefix}, skipping");
                continue;
            };

            let base = &self.tensors[&base_name];
            if base.shape.len() != 2 {
                tracing::warn!("LoRA: base weight {base_name} is not 2D, skipping");
                continue;
            }

            // A: [rank, in], B: [out, rank], W: [out, in]
            let rank = a_shape[0];
            let in_feat = a_shape[1];
            let out_feat = b_shape[0];

            if base.shape != [out_feat, in_feat] {
                tracing::warn!(
                    "LoRA: shape mismatch for {base_name}: base {:?} vs LoRA out={out_feat} in={in_feat}",
                    base.shape
                );
                continue;
            }

            // Read base weight to f32.
            let numel = out_feat * in_feat;
            let mut w_f32 = vec![0.0f32; numel];
            read_to_f32(base.data(), base.dtype, &mut w_f32);

            // Read A to f32 [rank, in_feat].
            let a_numel = rank * in_feat;
            let mut a_f32 = vec![0.0f32; a_numel];
            // LoRA weights are typically F32 in PEFT safetensors.
            read_to_f32(a_data, DType::F32, &mut a_f32);

            // Read B to f32 [out_feat, rank].
            let b_numel = out_feat * rank;
            let mut b_f32 = vec![0.0f32; b_numel];
            read_to_f32(b_data, DType::F32, &mut b_f32);

            // Compute delta = B @ A → [out_feat, in_feat], then W += scaling * delta.
            let scaling_f32 = scaling as f32;
            for i in 0..out_feat {
                for j in 0..in_feat {
                    let mut dot = 0.0f32;
                    for k in 0..rank {
                        dot += b_f32[i * rank + k] * a_f32[k * in_feat + j];
                    }
                    w_f32[i * in_feat + j] += scaling_f32 * dot;
                }
            }

            // Write merged weight back in base dtype.
            let merged_bytes = write_from_f32(&w_f32, base.dtype);
            let size_bytes = merged_bytes.len();
            let owned = Arc::new(merged_bytes);

            // Replace CpuTensorRef with one backed by owned data.
            self.tensors.insert(
                base_name,
                CpuTensorRef {
                    mmap: None,
                    data_offset: 0,
                    size_bytes,
                    shape: vec![out_feat, in_feat],
                    dtype: base.dtype,
                    owned: Some(owned),
                },
            );

            merged_count += 1;
            tracing::debug!("LoRA: merged {prefix} → [{out_feat}, {in_feat}]");
        }

        tracing::info!(
            "LoRA: merged {} weight tensors (rank={}, alpha={}, scaling={:.4})",
            merged_count,
            config.r,
            config.lora_alpha,
            scaling,
        );

        Ok(merged_count)
    }

    // -----------------------------------------------------------------------
    // Tensor-parallel sharding (CPU-side slice → GPU)
    // -----------------------------------------------------------------------

    /// Remove a tensor by name, slice it along `dim` for tensor parallelism,
    /// and copy only the shard to GPU. Returns a GPU tensor of the shard.
    ///
    /// For dim=0 sharding (column parallel): contiguous slice of rows.
    /// For dim=1 sharding (row parallel): strided extraction of columns,
    /// copied row-by-row into a contiguous pinned buffer before H2D.
    pub fn take_shard(
        &mut self,
        name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<GpuTensor> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;

        let (data, shard_shape, dtype) = self.shard_cpu_data(&cpu_ref, dim, rank, world_size);

        let size_bytes = shard_shape.iter().product::<usize>() * dtype.size_bytes();
        dump_shard_head(name, dim, rank, world_size, &shard_shape, dtype, data);
        let gpu_ptr = unsafe { driver::mem_alloc(size_bytes)? };
        self.gpu_allocs
            .push(unsafe { crate::alloc::RawGpuMem::new(gpu_ptr, size_bytes) });
        unsafe {
            driver::memcpy_htod_async(gpu_ptr, data, size_bytes, self.stream)?;
        }

        Ok(unsafe { GpuTensor::new(gpu_ptr, &shard_shape, dtype) })
    }

    /// Copy a shard of a tensor directly to an offset within an existing GPU buffer.
    ///
    /// Used for fused TP weight loading (e.g. QKV shards concatenated into one buffer).
    /// Returns the number of bytes written.
    pub unsafe fn take_shard_into(
        &mut self,
        name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
        dst: *mut u8,
        stream: CUstream,
    ) -> Result<usize> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;

        let (data, shard_shape, dtype) = self.shard_cpu_data(&cpu_ref, dim, rank, world_size);

        let size_bytes = shard_shape.iter().product::<usize>() * dtype.size_bytes();
        dump_shard_head(name, dim, rank, world_size, &shard_shape, dtype, data);
        driver::memcpy_htod_async(dst, data, size_bytes, stream)?;

        Ok(size_bytes)
    }

    /// Internal: extract a shard from CPU tensor data. Returns (ptr, shard_shape, dtype).
    ///
    /// For dim=0: returns a pointer into the original data (contiguous slice).
    /// For dim=1: copies strided columns into the pinned cast buffer, returns pointer to that.
    fn shard_cpu_data(
        &mut self,
        cpu_ref: &CpuTensorRef,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> (*const u8, Vec<usize>, DType) {
        assert!(!cpu_ref.shape.is_empty(), "cannot shard scalar");
        assert!(dim < cpu_ref.shape.len(), "dim out of range");
        let full_size = cpu_ref.shape[dim];
        assert!(
            full_size.is_multiple_of(world_size),
            "dim {dim} size {full_size} not divisible by world_size {world_size}"
        );
        let shard_size = full_size / world_size;

        // Apply dtype casting first if needed.
        let (src_data, _src_bytes, dtype) = self.maybe_cast_cpu(cpu_ref);

        let elem_size = dtype.size_bytes();
        let mut shard_shape = cpu_ref.shape.clone();
        shard_shape[dim] = shard_size;

        if dim == 0 {
            // Contiguous slice: rows [rank*shard_size .. (rank+1)*shard_size].
            // Each row has product(shape[1:]) elements.
            let row_elems: usize = cpu_ref.shape[1..].iter().product();
            let row_bytes = row_elems * elem_size;
            let offset = rank * shard_size * row_bytes;
            let data = unsafe { src_data.add(offset) };
            (data, shard_shape, dtype)
        } else if dim == 1 && cpu_ref.shape.len() == 2 {
            // Strided column extraction for 2D tensor [rows, cols].
            // Extract columns [rank*shard_size .. (rank+1)*shard_size] from each row.
            let rows = cpu_ref.shape[0];
            let cols = cpu_ref.shape[1];
            let col_start = rank * shard_size;
            let shard_row_bytes = shard_size * elem_size;
            let needed = rows * shard_row_bytes;
            self.ensure_pinned_buf(needed);

            let dst = self.cast_pinned.0;
            for r in 0..rows {
                let src_offset = (r * cols + col_start) * elem_size;
                let dst_offset = r * shard_row_bytes;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_data.add(src_offset),
                        dst.add(dst_offset),
                        shard_row_bytes,
                    );
                }
            }
            (dst as *const u8, shard_shape, dtype)
        } else {
            panic!(
                "take_shard: unsupported dim={dim} for {}D tensor",
                cpu_ref.shape.len()
            );
        }
    }

    /// Take a tensor's raw CPU bytes without uploading to GPU.
    /// Returns (data_bytes, shape, dtype).
    pub fn take_cpu(&mut self, name: &str) -> Result<(Vec<u8>, Vec<usize>, DType)> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;
        let data = cpu_ref.data().to_vec();
        Ok((data, cpu_ref.shape, cpu_ref.dtype))
    }
}

/// FERRITE_WEIGHT_DUMP=1: print first 8 elements of the shard plus first 8
/// of the LAST row (catches dim=1 striding bugs that look right at row 0).
/// Output is one line: tag, name, dim/rank/world, shape, head bf16-bits hex,
/// head f32 values, tail-row bf16-bits hex, tail-row f32 values. Designed to
/// diff against an equivalent Python print of `param.data.flatten()[:8]` and
/// `param.data[-1, :8]` after the loader has applied its narrow().
fn dump_shard_head(
    name: &str,
    dim: usize,
    rank: usize,
    world: usize,
    shape: &[usize],
    dtype: DType,
    data: *const u8,
) {
    if std::env::var("FERRITE_WEIGHT_DUMP").ok().as_deref() != Some("1") {
        return;
    }
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32) {
        return;
    }
    let total: usize = shape.iter().product();
    let head_n = 8.min(total);
    if head_n == 0 {
        return;
    }
    let mut head_f32 = vec![0f32; head_n];
    unsafe {
        let head_bytes = std::slice::from_raw_parts(data, head_n * dtype.size_bytes());
        read_to_f32(head_bytes, dtype, &mut head_f32);
    }
    let head_bits: Vec<String> = (0..head_n)
        .map(|i| match dtype {
            DType::BF16 | DType::F16 => {
                format!("{:04x}", unsafe { *(data.add(i * 2) as *const u16) })
            }
            DType::F32 => format!("{:08x}", unsafe { *(data.add(i * 4) as *const u32) }),
            _ => unreachable!(),
        })
        .collect();
    let head_vals: Vec<String> = head_f32.iter().map(|v| format!("{v:+.6e}")).collect();
    let mut tail_part = String::new();
    if shape.len() == 2 && shape[0] > 1 {
        let cols = shape[1];
        let last_row = shape[0] - 1;
        let row_off = last_row * cols * dtype.size_bytes();
        let tail_n = 8.min(cols);
        let mut tail_f32 = vec![0f32; tail_n];
        unsafe {
            let tail_bytes =
                std::slice::from_raw_parts(data.add(row_off), tail_n * dtype.size_bytes());
            read_to_f32(tail_bytes, dtype, &mut tail_f32);
        }
        let tail_bits: Vec<String> = (0..tail_n)
            .map(|i| match dtype {
                DType::BF16 | DType::F16 => format!("{:04x}", unsafe {
                    *(data.add(row_off + i * 2) as *const u16)
                }),
                DType::F32 => format!("{:08x}", unsafe {
                    *(data.add(row_off + i * 4) as *const u32)
                }),
                _ => unreachable!(),
            })
            .collect();
        let tail_vals: Vec<String> = tail_f32.iter().map(|v| format!("{v:+.6e}")).collect();
        tail_part = format!(
            " tail_row={last_row} tail_bits=[{}] tail_vals=[{}]",
            tail_bits.join(","),
            tail_vals.join(","),
        );
    }
    eprintln!(
        "[ferrite-weight-dump] name={name} dim={dim} rank={rank}/{world} shape={shape:?} \
         head_bits=[{}] head_vals=[{}]{}",
        head_bits.join(","),
        head_vals.join(","),
        tail_part,
    );
}

impl Drop for GpuWeights {
    fn drop(&mut self) {
        // Signal precast thread to stop and wait for it.
        if let Some(state) = &self.precast {
            state.shutdown.store(true, Ordering::Relaxed);
        }
        if let Some(handle) = self.precast_handle.take() {
            handle.join().ok();
        }
        // Free any unconsumed precast pinned buffers.
        if let Some(state) = &self.precast
            && let Ok(mut ready) = state.ready.lock()
        {
            for (_name, entry) in ready.drain() {
                unsafe { driver::mem_free_host(entry.pinned_ptr).ok() };
            }
        }
        // Free the synchronous pinned cast buffer if allocated.
        if !self.cast_pinned.0.is_null() {
            unsafe { driver::mem_free_host(self.cast_pinned.0).ok() };
        }
        // GPU memory allocated by take()/take_into() is owned by model layers.
        // CPU mmaps are dropped automatically when Arc<Mmap> refcounts reach zero.
    }
}
