// SPDX-License-Identifier: Apache-2.0
//! Metal implementation of the [`DeviceAllocator`] trait.

#![cfg(feature = "metal")]

use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::device_allocator::DeviceAllocator;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;

const DEFAULT_CHUNK_BYTES: usize = 256 * 1024 * 1024;

struct MetalArena {
    buffer: Buffer,
    base: *mut u8,
    capacity: usize,
    used: usize,
}

unsafe impl Send for MetalArena {}
unsafe impl Sync for MetalArena {}

/// Per-tensor record stored in the parent `MmapRegion`. Each tensor's
/// bytes live at `aligned_buffer.contents() + dst_offset` (the
/// `dst_offset` is 16-aligned by construction so kernel bindings
/// pass the strict alignment gate). `ready` is signalled by the
/// background loader thread that ran the `pread` for this tensor;
/// `alloc_and_copy_host` joins on it before handing the pointer out.
struct MmapTensor {
    /// Byte offset within the mmap (i.e. file) where the tensor
    /// starts. Used as the binary-search key when classifying an
    /// incoming `src` pointer.
    src_offset: usize,
    len: usize,
    /// 16-aligned offset within `aligned_buffer.contents()`.
    dst_offset: usize,
    ready: Arc<TensorReady>,
}

/// One-shot ready signal. Set by the background pread worker(s) for
/// the tensor; `take()` waits on it before returning the
/// destination pointer. Multiple chunks may share a single tensor's
/// `TensorReady` via `chunks_remaining`.
struct TensorReady {
    done: AtomicBool,
    chunks_remaining: AtomicUsize,
    waiter: Mutex<()>,
    cv: Condvar,
}

impl TensorReady {
    fn new(n_chunks: usize) -> Self {
        Self {
            done: AtomicBool::new(n_chunks == 0),
            chunks_remaining: AtomicUsize::new(n_chunks),
            waiter: Mutex::new(()),
            cv: Condvar::new(),
        }
    }

    /// Mark one chunk as done. The last chunk to complete flips
    /// `done` and notifies any waiters.
    fn signal_chunk(&self) {
        if self.chunks_remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _g = self.waiter.lock().expect("TensorReady waiter mutex");
            self.done.store(true, Ordering::Release);
            self.cv.notify_all();
        }
    }

    fn wait(&self) {
        if self.done.load(Ordering::Acquire) {
            return;
        }
        let mut g = self.waiter.lock().expect("TensorReady waiter mutex");
        while !self.done.load(Ordering::Acquire) {
            g = self.cv.wait(g).expect("TensorReady cv");
        }
    }
}

struct MmapRegion {
    /// Original mmap base pointer + length. The mmap is kept alive
    /// for **pointer identity only** — callers compute
    /// `src = mmap.as_ptr() + data_offset` from the safetensors
    /// header parse and we look up which `MmapTensor` corresponds
    /// to that address. Tensor *data* pages of the mmap are never
    /// touched after construction; the bytes live in
    /// `aligned_buffer` after a background `pread` from the file.
    base: *const u8,
    len: usize,
    /// **Pre-aligned destination buffer.** One `MTLBuffer`
    /// (storageModeShared) sized to fit every tensor in the shard
    /// laid out at 16-aligned offsets. The CPU-mappable
    /// `.contents()` pointer is what the loader threads `pread`
    /// into. All `alloc_and_copy_host{,_aligned}` zero-copy returns
    /// and `buffer_for` lookups resolve into this buffer.
    aligned_buffer: Buffer,
    aligned_base: *mut u8,
    aligned_capacity: usize,
    /// Per-tensor records, sorted by `src_offset` for binary-search
    /// lookup from the `src` pointer the caller passes to
    /// `alloc_and_copy_host`.
    tensors: Vec<MmapTensor>,
    _mmap: Arc<memmap2::Mmap>,
}

unsafe impl Send for MmapRegion {}
unsafe impl Sync for MmapRegion {}

#[derive(Clone, Copy)]
enum MmapClassify {
    /// Source `src` lies within a registered mmap and its shifted
    /// offset (`mmap_offset + region.shift`) is `min_align`-aligned.
    /// `aligned_ptr` is the address inside the per-region pre-aligned
    /// destination buffer where the bulk-copied bytes live — i.e.
    /// what `alloc_and_copy_host{,_aligned}` returns to the caller.
    Aligned { aligned_ptr: *mut u8 },
    /// Source `src` is in a registered mmap but the shifted offset
    /// isn't `min_align`-aligned — falls through to the arena memcpy
    /// path.
    Unaligned,
    /// Source `src` is outside every registered mmap (e.g. the bytes
    /// were heap-allocated by `maybe_cast_cpu` after a CPU cast).
    Outside,
}

pub type ArenaHook = Arc<dyn Fn(&Buffer) + Send + Sync>;

pub struct MetalAllocator {
    device: Device,
    arenas: Arc<Mutex<Vec<MetalArena>>>,
    chunk_bytes: usize,
    on_new_arena: Arc<Mutex<Option<ArenaHook>>>,
    residency: ferrite_metal_kernels::residency::MetalResidencySet,
    mmaps: Arc<Mutex<Vec<MmapRegion>>>,
    /// Diagnostic counters for `alloc_and_copy_host` routing. Bumped
    /// once per call so the worker can print a one-shot "zero-copy
    /// vs memcpy" breakdown after `try_load` completes. Atomics are
    /// `Relaxed` — these aren't synchronization, just stats.
    load_stats: Arc<LoadStats>,
}

/// Histogram of tensor-offset trailing-zero counts. Index `i` counts
/// tensors whose offset within their mmap has exactly `i` trailing
/// zero bits (i.e. is aligned to `2^i` but not `2^(i+1)`). Capped at
/// 16; anything ≥ 16 lands in bucket 16. Wired through `LoadStats`.
pub const ALIGNMENT_HISTOGRAM_BUCKETS: usize = 17;

#[derive(Default)]
pub struct LoadStats {
    /// `alloc_and_copy_host` call returned an mmap-aliased pointer
    /// (no copy, ≤ a few hundred ns).
    pub zero_copy_calls: AtomicU64,
    pub zero_copy_bytes: AtomicU64,
    /// Subset of `zero_copy_calls` that only succeeded because the
    /// caller passed `min_align < MIN_BIND_ALIGN` via
    /// `alloc_and_copy_host_aligned` (e.g. F16 scales/biases at
    /// `mod 4 == 2` taking the 2-byte-aligned fast path). Bumps
    /// only when the offset would have FAILED the strict 16-byte
    /// gate but PASSED the relaxed gate — pure visibility into how
    /// much the dtype-aware relaxation actually saves.
    pub zero_copy_relaxed_calls: AtomicU64,
    pub zero_copy_relaxed_bytes: AtomicU64,
    /// `alloc_and_copy_host` call fell through to arena memcpy
    /// (~5 GB/s on Apple Silicon). High-volume miss here is the
    /// startup-time bottleneck.
    pub memcpy_calls: AtomicU64,
    pub memcpy_bytes: AtomicU64,
    /// Per-prefix breakdown: keyed by a coarse category derived
    /// from the byte count (so we can tell scales/biases apart
    /// from packed weight blobs without threading prefix strings
    /// down to the allocator).
    pub memcpy_small_calls: AtomicU64, // < 1 MiB
    pub memcpy_med_calls: AtomicU64,   // 1 MiB ≤ ... < 16 MiB
    pub memcpy_large_calls: AtomicU64, // ≥ 16 MiB
    /// Why zero-copy failed: source pointer wasn't in any
    /// registered mmap region (e.g. tensor was already heap-copied
    /// upstream, or mmap was never registered).
    pub memcpy_outside_mmap: AtomicU64,
    /// Why zero-copy failed: pointer was inside a registered mmap,
    /// but the offset wasn't 16-byte aligned. This is the
    /// safetensors-data-section-base alignment problem.
    pub memcpy_unaligned: AtomicU64,
    /// Histogram of observed offset trailing-zero counts (0..=16).
    pub alignment_hist: [AtomicU64; ALIGNMENT_HISTOGRAM_BUCKETS],
}

impl LoadStats {
    fn observe_offset_alignment(&self, tz: u32) {
        let idx = (tz as usize).min(ALIGNMENT_HISTOGRAM_BUCKETS - 1);
        self.alignment_hist[idx].fetch_add(1, Ordering::Relaxed);
    }
}

unsafe impl Send for MetalAllocator {}
unsafe impl Sync for MetalAllocator {}

impl Clone for MetalAllocator {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            arenas: Arc::clone(&self.arenas),
            chunk_bytes: self.chunk_bytes,
            on_new_arena: Arc::clone(&self.on_new_arena),
            residency: self.residency.clone(),
            mmaps: Arc::clone(&self.mmaps),
            load_stats: Arc::clone(&self.load_stats),
        }
    }
}

impl MetalAllocator {
    pub fn new(device: Device) -> Self {
        let residency = ferrite_metal_kernels::residency::MetalResidencySet::new(&device);
        Self {
            device,
            arenas: Arc::new(Mutex::new(Vec::new())),
            chunk_bytes: DEFAULT_CHUNK_BYTES,
            on_new_arena: Arc::new(Mutex::new(None)),
            residency,
            mmaps: Arc::new(Mutex::new(Vec::new())),
            load_stats: Arc::new(LoadStats::default()),
        }
    }

    pub fn with_chunk_bytes(device: Device, chunk_bytes: usize) -> Self {
        let residency = ferrite_metal_kernels::residency::MetalResidencySet::new(&device);
        Self {
            device,
            arenas: Arc::new(Mutex::new(Vec::new())),
            chunk_bytes,
            on_new_arena: Arc::new(Mutex::new(None)),
            residency,
            mmaps: Arc::new(Mutex::new(Vec::new())),
            load_stats: Arc::new(LoadStats::default()),
        }
    }

    /// Snapshot the load-time routing counters. Caller is expected
    /// to log them once (after `try_load` completes) — the atomics
    /// are not reset.
    pub fn load_stats(&self) -> &LoadStats {
        &self.load_stats
    }

    pub fn residency(&self) -> &ferrite_metal_kernels::residency::MetalResidencySet {
        &self.residency
    }

    pub fn set_arena_hook(&self, hook: ArenaHook) {
        let arenas = self.arenas.lock().expect("MetalAllocator arenas Mutex");
        for arena in arenas.iter() {
            (hook)(&arena.buffer);
        }
        *self.on_new_arena.lock().expect("arena hook mutex") = Some(hook);
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn buffer_for(&self, ptr: *const u8) -> Option<(Buffer, u64)> {
        let p = ptr as usize;
        {
            let mmaps = self.mmaps.lock().expect("MetalAllocator mmaps Mutex");
            for region in mmaps.iter() {
                // Zero-copy returns from `alloc_and_copy_host_aligned`
                // point into the per-region pre-aligned MTLBuffer that
                // covers the packed-aligned tensor layout, valid range
                // `[aligned_base, aligned_base + aligned_capacity)`.
                let a_start = region.aligned_base as usize;
                let a_end = a_start + region.aligned_capacity;
                if p >= a_start && p < a_end {
                    return Some((region.aligned_buffer.clone(), (p - a_start) as u64));
                }
            }
        }
        let arenas = self.arenas.lock().expect("MetalAllocator arenas Mutex");
        for arena in arenas.iter() {
            let start = arena.base as usize;
            let end = start + arena.used;
            if p >= start && p < end {
                return Some((arena.buffer.clone(), (p - start) as u64));
            }
        }
        None
    }

    /// Parse a safetensors header (`[u64 header_size_le][JSON header]
    /// [data section]`) and return the list of tensors with their
    /// **file-relative** byte offsets and sizes. Returns `None` if
    /// the header doesn't parse — the caller treats that as an error
    /// for safetensors shards (the load path only feeds safetensors
    /// in).
    fn parse_safetensors_tensors(base: *const u8, len: usize) -> Option<Vec<(usize, usize)>> {
        if len < 8 {
            return None;
        }
        // SAFETY: `base` points to at least 8 mapped bytes.
        let header_size =
            unsafe { std::ptr::read_unaligned(base as *const u64).to_le() } as usize;
        if header_size == 0 || header_size > len.saturating_sub(8) {
            return None;
        }
        let header_bytes =
            unsafe { std::slice::from_raw_parts(base.wrapping_add(8), header_size) };
        let json: serde_json::Value = serde_json::from_slice(header_bytes).ok()?;
        let obj = json.as_object()?;
        let data_section_start = 8 + header_size;
        let mut tensors = Vec::with_capacity(obj.len());
        for (key, val) in obj {
            if key == "__metadata__" {
                continue;
            }
            let offs = val.get("data_offsets")?.as_array()?;
            let lo = offs.first()?.as_u64()? as usize;
            let hi = offs.get(1)?.as_u64()? as usize;
            if hi < lo {
                return None;
            }
            tensors.push((data_section_start + lo, hi - lo));
        }
        tensors.sort_by_key(|&(off, _)| off);
        Some(tensors)
    }

    /// Register a safetensors shard for zero-copy weight binding.
    ///
    /// Lays out every tensor at a 16-aligned `dst_offset` in a single
    /// shared-storage `MTLBuffer`, then dispatches `pread` tasks to
    /// the rayon global pool that read each tensor's bytes straight
    /// from disk into `buffer.contents() + dst_offset`. This function
    /// **returns before any tensor data has been read**: each
    /// per-tensor `TensorReady` is signalled by the background
    /// worker(s) once that tensor's bytes are in the destination
    /// buffer. `take()`-side callers join on the corresponding
    /// `TensorReady` before consuming the pointer.
    ///
    /// The mmap is retained only as a pointer-identity device: the
    /// caller's `CpuTensorRef` holds `mmap.as_ptr() + data_offset`
    /// values that we look up via the per-tensor `src_offset` table.
    /// **Tensor-data pages of the mmap are never faulted in** — the
    /// bytes come from `pread(fd, …)` straight into the destination.
    pub fn register_mmap(&self, path: &Path, mmap: Arc<memmap2::Mmap>) -> Result<()> {
        let base = mmap.as_ptr();
        let len = mmap.len();
        if len == 0 {
            return Ok(());
        }

        let tensors_src = Self::parse_safetensors_tensors(base, len).ok_or_else(|| {
            anyhow::anyhow!(
                "MetalAllocator::register_mmap: failed to parse safetensors header for {}",
                path.display()
            )
        })?;

        // Pack each tensor at the next 16-aligned offset.
        let mut packed: Vec<(usize, usize, usize)> = Vec::with_capacity(tensors_src.len());
        let mut running = 0usize;
        for &(src_off, sz) in &tensors_src {
            let dst_off = running.div_ceil(Self::MIN_BIND_ALIGN) * Self::MIN_BIND_ALIGN;
            packed.push((src_off, sz, dst_off));
            running = dst_off + sz;
        }
        let aligned_capacity = running
            .div_ceil(Self::MIN_BIND_ALIGN)
            .saturating_mul(Self::MIN_BIND_ALIGN)
            .max(Self::MIN_BIND_ALIGN);

        let dst_buffer = self
            .device
            .newBufferWithLength_options(aligned_capacity, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "MetalAllocator::register_mmap: newBufferWithLength({} bytes) returned nil",
                    aligned_capacity
                )
            })?;
        let aligned_base = dst_buffer.contents().as_ptr() as *mut u8;
        anyhow::ensure!(
            !aligned_base.is_null(),
            "MetalAllocator::register_mmap: destination buffer.contents() is null"
        );

        // Validate that the file exists / is openable; per-task we
        // re-open by path so each worker has its own fd (concurrent
        // `pread` on a shared fd on macOS appears to interleave reads
        // in practice — verified by bisect against sync-chunked).
        std::fs::File::open(path).with_context(|| {
            format!("MetalAllocator::register_mmap: open {}", path.display())
        })?;

        // Wrap the aligned destination base as a Send/Sync usize for
        // closure capture. Each worker writes into a disjoint chunk
        // of the buffer; no two workers race on the same byte.
        let dst_base_usize = aligned_base as usize;

        // Chunk size targets ~16 MiB per `pread` so large tensors
        // parallelize across rayon workers and small tensors remain
        // a single dispatch.
        const READ_CHUNK: usize = 16 * 1024 * 1024;

        let mut tensors: Vec<MmapTensor> = Vec::with_capacity(packed.len());
        for (src_off, sz, dst_off) in packed {
            let n_chunks = if sz == 0 { 0 } else { sz.div_ceil(READ_CHUNK) };
            let ready = Arc::new(TensorReady::new(n_chunks));
            for chunk_idx in 0..n_chunks {
                let off_in_tensor = chunk_idx * READ_CHUNK;
                let chunk_sz = (sz - off_in_tensor).min(READ_CHUNK);
                let chunk_src = src_off + off_in_tensor;
                let chunk_dst = dst_off + off_in_tensor;
                let ready_w = Arc::clone(&ready);
                let path_w = path.to_path_buf();
                rayon::spawn(move || {
                    let file_w = std::fs::File::open(&path_w).unwrap_or_else(|e| {
                        panic!("re-open {} failed: {e}", path_w.display())
                    });
                    let fd = file_w.as_raw_fd();
                    let mut written = 0usize;
                    while written < chunk_sz {
                        // SAFETY: dst points to `chunk_sz - written`
                        // bytes inside the destination MTLBuffer's
                        // shared-storage contents (allocated above,
                        // outlives the closure via the region's
                        // ownership of `aligned_buffer`).
                        let dst =
                            (dst_base_usize + chunk_dst + written) as *mut libc::c_void;
                        let n = unsafe {
                            libc::pread(
                                fd,
                                dst,
                                chunk_sz - written,
                                (chunk_src + written) as libc::off_t,
                            )
                        };
                        if n < 0 {
                            let err = std::io::Error::last_os_error();
                            panic!(
                                "pread({}, off={}, len={}) failed: {}",
                                path_w.display(),
                                chunk_src + written,
                                chunk_sz - written,
                                err
                            );
                        }
                        if n == 0 {
                            panic!(
                                "pread({}, off={}) returned 0 (unexpected EOF, want {} more bytes)",
                                path_w.display(),
                                chunk_src + written,
                                chunk_sz - written
                            );
                        }
                        written += n as usize;
                    }
                    ready_w.signal_chunk();
                });
            }
            tensors.push(MmapTensor {
                src_offset: src_off,
                len: sz,
                dst_offset: dst_off,
                ready,
            });
        }

        self.residency.insert(&dst_buffer);

        self.mmaps
            .lock()
            .expect("MetalAllocator mmaps Mutex")
            .push(MmapRegion {
                base,
                len,
                aligned_buffer: dst_buffer,
                aligned_base,
                aligned_capacity,
                tensors,
                _mmap: mmap,
            });
        Ok(())
    }

    /// Largest alignment the qmv / qmm_t / NAX kernels demand on a
    /// buffer offset bound via `setBuffer:offset:atIndex:`. The packed
    /// int4 weight binding is declared `device const uint32_t*`
    /// (`shaders/quantized_qmv.metal:829`, `quantized_qmm.metal:388`),
    /// which requires 4-byte aligned offsets — and Apple's M-series
    /// driver does NOT silently tolerate misalignment (see
    /// `tests/quantized_qmv_test.rs::affine_qmv_fast_b4_bf16_unaligned_packed_offset_1_byte_prefix`,
    /// which reproduces the live divergence by binding at offset
    /// %4 = 1 and gets `worst abs_err = 19.07` vs an allowed 0.39).
    /// 16 covers u32 + simdgroup_float4 + any future SIMD-wide types,
    /// and is cheap enough to gate the mmap-alias short-circuit on.
    const MIN_BIND_ALIGN: usize = 16;

    /// `Some(aligned_ptr)` if `src` matches the start of a tensor in
    /// a registered mmap. The returned pointer points into the
    /// per-region pre-aligned destination buffer at the tensor's
    /// 16-aligned `dst_offset`; blocks (per-tensor `Condvar`) until
    /// the background `pread` for that tensor has completed.
    ///
    /// `min_align` is honored for visibility only — `dst_offset` is
    /// always 16-aligned by construction, so any sane request passes
    /// (the diagnostic counters still observe whether the relaxation
    /// vs strict gate would have mattered).
    ///
    /// `None` if `src` is outside every registered mmap, or doesn't
    /// match a tensor head (e.g. a CPU-cast scratch buffer).
    fn aligned_mmap_offset(
        &self,
        src: *const u8,
        bytes: usize,
        min_align: usize,
    ) -> Option<*mut u8> {
        let p = src as usize;
        let end = p.saturating_add(bytes);
        let ready;
        let aligned_ptr;
        {
            let mmaps = self.mmaps.lock().expect("MetalAllocator mmaps Mutex");
            let region = mmaps
                .iter()
                .find(|r| p >= r.base as usize && end <= r.base as usize + r.len)?;
            let mmap_offset = p - region.base as usize;
            let tensor = match region
                .tensors
                .binary_search_by_key(&mmap_offset, |t| t.src_offset)
            {
                Ok(idx) => &region.tensors[idx],
                Err(_) => return None,
            };
            if tensor.len != bytes {
                return None;
            }
            if min_align != 0 && tensor.dst_offset % min_align != 0 {
                return None;
            }
            aligned_ptr = unsafe { region.aligned_base.add(tensor.dst_offset) };
            ready = Arc::clone(&tensor.ready);
        }
        ready.wait();
        Some(aligned_ptr)
    }

    /// As [`aligned_mmap_offset`] but also bumps the histogram /
    /// classification counters and returns `Unaligned` vs `Outside`
    /// distinctly for the diagnostic log.
    fn classify_mmap_offset(
        &self,
        src: *const u8,
        bytes: usize,
        min_align: usize,
    ) -> MmapClassify {
        let p = src as usize;
        let end = p.saturating_add(bytes);
        let ready;
        let aligned_ptr;
        {
            let mmaps = self.mmaps.lock().expect("MetalAllocator mmaps Mutex");
            let region = match mmaps
                .iter()
                .find(|r| p >= r.base as usize && end <= r.base as usize + r.len)
            {
                Some(r) => r,
                None => return MmapClassify::Outside,
            };
            let mmap_offset = p - region.base as usize;
            // Histogram still buckets by the raw intra-mmap offset:
            // it characterizes the safetensors file layout, not our
            // routing. With packed-aligned dst, every match becomes
            // `Aligned` regardless of the source's trailing zeros —
            // the histogram just documents that file-level fact.
            let tz = mmap_offset.trailing_zeros();
            self.load_stats.observe_offset_alignment(tz);
            let tensor = match region
                .tensors
                .binary_search_by_key(&mmap_offset, |t| t.src_offset)
            {
                Ok(idx) => &region.tensors[idx],
                Err(_) => return MmapClassify::Outside,
            };
            if tensor.len != bytes {
                return MmapClassify::Outside;
            }
            if min_align != 0 && tensor.dst_offset % min_align != 0 {
                return MmapClassify::Unaligned;
            }
            aligned_ptr = unsafe { region.aligned_base.add(tensor.dst_offset) };
            ready = Arc::clone(&tensor.ready);
        }
        ready.wait();
        MmapClassify::Aligned { aligned_ptr }
    }

    pub fn arena_count(&self) -> usize {
        self.arenas
            .lock()
            .expect("MetalAllocator arenas Mutex")
            .len()
    }

    pub fn used_bytes(&self) -> usize {
        self.arenas
            .lock()
            .expect("MetalAllocator arenas Mutex")
            .iter()
            .map(|a| a.used)
            .sum()
    }

    fn push_arena_locked(
        device: &Device,
        arenas: &mut Vec<MetalArena>,
        chunk_bytes: usize,
        min_bytes: usize,
        hook: &Arc<Mutex<Option<ArenaHook>>>,
        residency: &ferrite_metal_kernels::residency::MetalResidencySet,
    ) -> Result<usize> {
        let capacity = min_bytes.max(chunk_bytes);
        let buffer = device
            .newBufferWithLength_options(capacity, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "MetalAllocator: newBufferWithLength_options({} bytes) returned nil",
                    capacity
                )
            })?;
        let base = buffer.contents().as_ptr() as *mut u8;
        if base.is_null() {
            anyhow::bail!(
                "MetalAllocator: buffer contents() returned null pointer (capacity={})",
                capacity
            );
        }
        residency.insert(&buffer);
        if let Some(cb) = hook.lock().expect("arena hook mutex").as_ref() {
            (cb)(&buffer);
        }
        arenas.push(MetalArena {
            buffer,
            base,
            capacity,
            used: 0,
        });
        Ok(arenas.len() - 1)
    }

    pub fn alloc_uninit(&self, bytes: usize) -> Result<*mut u8> {
        let mut arenas = self.arenas.lock().expect("MetalAllocator arenas Mutex");
        if bytes == 0 {
            if arenas.is_empty() {
                Self::push_arena_locked(
                    &self.device,
                    &mut arenas,
                    self.chunk_bytes,
                    0,
                    &self.on_new_arena,
                    &self.residency,
                )?;
            }
            return Ok(arenas[0].base);
        }
        // Pad `used` up to MIN_BIND_ALIGN so every subsequent
        // allocation lands at a properly-aligned offset. Without this,
        // a small tensor whose size isn't a multiple of 16 (e.g. an
        // F16 vector with an odd element count) would shift every
        // following allocation off-alignment, and an int4 U32 weight
        // bound there would hit the same unaligned-binding UB we
        // gate against on the mmap-alias path. Wastes at most 15
        // bytes per allocation; negligible against tensor sizes.
        let aligned_bytes =
            bytes.div_ceil(Self::MIN_BIND_ALIGN) * Self::MIN_BIND_ALIGN;
        let idx = if let Some(idx) = arenas
            .iter()
            .rposition(|a| a.capacity - a.used >= aligned_bytes)
        {
            idx
        } else {
            Self::push_arena_locked(
                &self.device,
                &mut arenas,
                self.chunk_bytes,
                aligned_bytes,
                &self.on_new_arena,
                &self.residency,
            )?
        };
        let arena = &mut arenas[idx];
        let offset = arena.used;
        debug_assert_eq!(
            offset % Self::MIN_BIND_ALIGN,
            0,
            "arena.used is not {}-aligned on entry; previous alloc didn't pad",
            Self::MIN_BIND_ALIGN
        );
        let dst = unsafe { arena.base.add(offset) };
        arena.used = offset + aligned_bytes;
        Ok(dst)
    }
}

impl DeviceAllocator for MetalAllocator {
    unsafe fn alloc_and_copy_host(&mut self, src_host: *const u8, bytes: usize) -> Result<*mut u8> {
        unsafe {
            self.alloc_and_copy_host_aligned(src_host, bytes, Self::MIN_BIND_ALIGN)
        }
    }

    unsafe fn alloc_and_copy_host_aligned(
        &mut self,
        src_host: *const u8,
        bytes: usize,
        min_align: usize,
    ) -> Result<*mut u8> {
        // Zero-copy fast path: the source already lives in a
        // registered safetensors mmap AND lands at a `min_align`-aligned
        // offset. `min_align` is clamped above by `MIN_BIND_ALIGN` since
        // any pointer we return is also reachable via the trait
        // `alloc_and_copy_host` (no dtype guarantee), and may be bound
        // to arbitrary kernels later. The clamp below keeps the
        // SIMD-wide-safe floor; per-dtype relaxation below 16 only
        // helps for offsets in `(MIN_BIND_ALIGN, dtype_size]`.
        //
        // For `mlx-community` 4bit safetensors the data section lands
        // at file-offset `mod 16 = 2`, so the 16-byte gate rejects
        // every tensor (cf. `project_metal_safetensors_alignment`).
        // Relaxing to `min_align = 2` lets F16/BF16 scales/biases/
        // RMSNorm-gain tensors take this path — those kernel bindings
        // read scalar (`sl[0]`, `weight[i]`) and 2-byte alignment is
        // safe.
        let effective_min_align = min_align.min(Self::MIN_BIND_ALIGN).max(1);
        if bytes > 0 {
            match self.classify_mmap_offset(src_host, bytes, effective_min_align) {
                MmapClassify::Aligned { aligned_ptr } => {
                    self.load_stats
                        .zero_copy_calls
                        .fetch_add(1, Ordering::Relaxed);
                    self.load_stats
                        .zero_copy_bytes
                        .fetch_add(bytes as u64, Ordering::Relaxed);
                    // Visibility bookkeeping: did the relaxation
                    // actually do anything? Re-check at the strict
                    // 16-byte gate; if THAT would have failed, the
                    // relaxation is responsible for this zero-copy.
                    // After the register-time bulk-copy lands the
                    // shift, this counter should approach 0 (every
                    // canonical-layout tensor passes the strict gate
                    // already).
                    if effective_min_align < Self::MIN_BIND_ALIGN {
                        let strict = self
                            .aligned_mmap_offset(src_host, bytes, Self::MIN_BIND_ALIGN);
                        if strict.is_none() {
                            self.load_stats
                                .zero_copy_relaxed_calls
                                .fetch_add(1, Ordering::Relaxed);
                            self.load_stats
                                .zero_copy_relaxed_bytes
                                .fetch_add(bytes as u64, Ordering::Relaxed);
                        }
                    }
                    return Ok(aligned_ptr);
                }
                MmapClassify::Unaligned => {
                    self.load_stats
                        .memcpy_unaligned
                        .fetch_add(1, Ordering::Relaxed);
                }
                MmapClassify::Outside => {
                    self.load_stats
                        .memcpy_outside_mmap
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        let aligned_bytes = if bytes == 0 {
            0
        } else {
            bytes.div_ceil(Self::MIN_BIND_ALIGN) * Self::MIN_BIND_ALIGN
        };
        let mut arenas = self.arenas.lock().expect("MetalAllocator arenas Mutex");
        if bytes == 0 {
            if arenas.is_empty() {
                Self::push_arena_locked(
                    &self.device,
                    &mut arenas,
                    self.chunk_bytes,
                    0,
                    &self.on_new_arena,
                    &self.residency,
                )?;
            }
            return Ok(arenas[0].base);
        }

        let idx = if let Some(idx) = arenas
            .iter()
            .rposition(|a| a.capacity - a.used >= aligned_bytes)
        {
            idx
        } else {
            Self::push_arena_locked(
                &self.device,
                &mut arenas,
                self.chunk_bytes,
                aligned_bytes,
                &self.on_new_arena,
                &self.residency,
            )?
        };
        let arena = &mut arenas[idx];
        let offset = arena.used;
        debug_assert_eq!(
            offset % Self::MIN_BIND_ALIGN,
            0,
            "arena.used is not {}-aligned on entry",
            Self::MIN_BIND_ALIGN
        );
        let dst = unsafe { arena.base.add(offset) };
        unsafe { std::ptr::copy_nonoverlapping(src_host, dst, bytes) };
        arena.used = offset + aligned_bytes;
        self.load_stats
            .memcpy_calls
            .fetch_add(1, Ordering::Relaxed);
        self.load_stats
            .memcpy_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        let bucket = if bytes < 1 << 20 {
            &self.load_stats.memcpy_small_calls
        } else if bytes < 16 << 20 {
            &self.load_stats.memcpy_med_calls
        } else {
            &self.load_stats.memcpy_large_calls
        };
        bucket.fetch_add(1, Ordering::Relaxed);
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCreateSystemDefaultDevice;

    fn try_device() -> Option<Device> {
        MTLCreateSystemDefaultDevice()
    }

    #[test]
    fn alloc_returns_pointer_into_arena() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let mut alloc = MetalAllocator::new(device);
        let src = b"hello, metal";
        let ptr = unsafe {
            alloc
                .alloc_and_copy_host(src.as_ptr(), src.len())
                .expect("alloc")
        };
        assert!(!ptr.is_null());
        let read = unsafe { std::slice::from_raw_parts(ptr, src.len()) };
        assert_eq!(read, src);
        // `used_bytes` reports the padded request (rounded up to
        // `MIN_BIND_ALIGN` so the *next* allocation lands at an
        // aligned offset). 12 bytes round up to 16.
        let expected_used = src.len().div_ceil(MetalAllocator::MIN_BIND_ALIGN)
            * MetalAllocator::MIN_BIND_ALIGN;
        assert_eq!(alloc.used_bytes(), expected_used);
        assert_eq!(alloc.arena_count(), 1);

        let (buf, off) = alloc.buffer_for(ptr).expect("buffer_for");
        assert_eq!(off, 0);
        assert_eq!(buf.length(), DEFAULT_CHUNK_BYTES);
    }

    #[test]
    fn multiple_allocs_share_arena_until_full() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let mut alloc = MetalAllocator::with_chunk_bytes(device, 4096);
        let buf_a = vec![0xAAu8; 1024];
        let buf_b = vec![0xBBu8; 1024];
        let buf_c = vec![0xCCu8; 3072];

        let pa = unsafe {
            alloc
                .alloc_and_copy_host(buf_a.as_ptr(), buf_a.len())
                .unwrap()
        };
        let pb = unsafe {
            alloc
                .alloc_and_copy_host(buf_b.as_ptr(), buf_b.len())
                .unwrap()
        };
        let pc = unsafe {
            alloc
                .alloc_and_copy_host(buf_c.as_ptr(), buf_c.len())
                .unwrap()
        };

        let (ba, oa) = alloc.buffer_for(pa).unwrap();
        let (bb, ob) = alloc.buffer_for(pb).unwrap();
        let (bc, oc) = alloc.buffer_for(pc).unwrap();
        assert_eq!(Retained::as_ptr(&ba), Retained::as_ptr(&bb));
        assert_ne!(Retained::as_ptr(&ba), Retained::as_ptr(&bc));
        assert_eq!(oa, 0);
        assert_eq!(ob, 1024);
        assert_eq!(oc, 0);

        let read_a = unsafe { std::slice::from_raw_parts(pa, buf_a.len()) };
        let read_c = unsafe { std::slice::from_raw_parts(pc, buf_c.len()) };
        assert!(read_a.iter().all(|&b| b == 0xAA));
        assert!(read_c.iter().all(|&b| b == 0xCC));

        assert_eq!(alloc.arena_count(), 2);
        // Each allocation rounds up to a multiple of MIN_BIND_ALIGN
        // (16). 1024, 1024, 3072 are already multiples of 16 so the
        // total is unchanged here — this test serves as a guardrail
        // that aligned-size inputs don't grow.
        assert_eq!(alloc.used_bytes(), 1024 + 1024 + 3072);
    }

    #[test]
    fn oversized_request_gets_dedicated_arena() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let mut alloc = MetalAllocator::with_chunk_bytes(device, 4096);
        let big = vec![0x42u8; 16 * 1024];
        let p = unsafe { alloc.alloc_and_copy_host(big.as_ptr(), big.len()).unwrap() };
        let (buf, off) = alloc.buffer_for(p).unwrap();
        assert_eq!(off, 0);
        assert!(buf.length() >= big.len());
        assert_eq!(alloc.arena_count(), 1);
    }

    #[test]
    fn zero_byte_alloc_returns_nonnull_sentinel() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let mut alloc = MetalAllocator::new(device);
        let p = unsafe { alloc.alloc_and_copy_host(std::ptr::null(), 0).unwrap() };
        assert!(!p.is_null());
        assert_eq!(alloc.used_bytes(), 0);
    }

    #[test]
    fn buffer_for_returns_none_for_foreign_pointer() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let alloc = MetalAllocator::new(device);
        let stack = 0u8;
        assert!(alloc.buffer_for(&stack).is_none());
    }
}
