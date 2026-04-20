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
}

// Safety: GPU device pointers accessible from any host thread.
unsafe impl Send for GpuWeights {}
unsafe impl Sync for GpuWeights {}

impl GpuWeights {
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
        work.sort_by(|a, b| b.3.cmp(&a.3)); // Largest first.

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
    pub fn take(&mut self, name: &str) -> Result<GpuTensor> {
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

        unsafe {
            driver::memcpy_htod_async(gpu_ptr, data, size_bytes, self.stream)?;
        }

        Ok(unsafe { GpuTensor::new(gpu_ptr, &cpu_ref.shape, dtype) })
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

        // Slow path.
        let (data, size_bytes, _dtype) = self.maybe_cast_cpu(&cpu_ref);

        driver::memcpy_htod_async(dst, data, size_bytes, stream)?;

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

        unsafe {
            driver::memcpy_htod_async(gpu_ptr, data, size_bytes, self.stream).ok()?;
        }

        let shape = cpu_ref.shape.clone();
        self.tensors.insert(name.to_string(), cpu_ref);

        Some(unsafe { GpuTensor::new(gpu_ptr, &shape, dtype) })
    }

    /// Check if a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
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
    pub fn synthesize_packed_row_split_sizes(
        &mut self,
        packed_prefix: &str,
        split_targets: &[(&str, usize)],
    ) -> Result<bool> {
        let packed_weight_name = format!("{packed_prefix}.weight");
        if !self.tensors.contains_key(&packed_weight_name) {
            return Ok(false);
        }
        let grandparent = packed_prefix
            .rsplit_once('.')
            .map(|(p, _)| p)
            .ok_or_else(|| anyhow::anyhow!("packed prefix has no parent: {packed_prefix}"))?;

        if split_targets.is_empty() {
            anyhow::bail!("synthesize_packed_row_split_sizes: empty split_targets");
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
