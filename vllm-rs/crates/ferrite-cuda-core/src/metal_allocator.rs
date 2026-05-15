// SPDX-License-Identifier: Apache-2.0
//! Metal implementation of the [`DeviceAllocator`] trait.

#![cfg(feature = "metal")]

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLDevice, MTLResourceOptions,
};

use crate::device_allocator::DeviceAllocator;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

const DEFAULT_CHUNK_BYTES: usize = 256 * 1024 * 1024;

struct MetalArena {
    buffer: Buffer,
    base: *mut u8,
    capacity: usize,
    used: usize,
}

unsafe impl Send for MetalArena {}
unsafe impl Sync for MetalArena {}

struct MmapRegion {
    /// Original mmap base pointer + length. The mmap remains the
    /// canonical *source* identity — callers hand us tensor pointers
    /// computed from the safetensors data section, which are offsets
    /// from `base`. Classification (is `src` in any region?) walks
    /// `[base, base + len)`.
    base: *const u8,
    len: usize,
    /// **Pre-aligned destination buffer.** At `register_mmap` we
    /// allocate a fresh 16-aligned `MTLBuffer` of size `len + shift`
    /// and bulk-copy the mmap contents into it starting at offset
    /// `shift`. After that, every tensor whose intra-mmap offset
    /// matches the canonical safetensors layout (tensors at
    /// `data_section_start + k * 16`) lands at
    /// `aligned_base + offset + shift` where the trailing bits are
    /// 0 mod 16 — letting the strict 16-byte zero-copy gate pass
    /// for U32 packed weights, F32, F16, BF16, and any future SIMD-
    /// wide binding type.
    ///
    /// All `alloc_and_copy_host{,_aligned}` zero-copy returns point
    /// into this buffer (not into the mmap). `buffer_for` maps these
    /// pointers back to `(aligned_buffer, aligned_offset)` for
    /// `setBuffer:offset:atIndex:`.
    aligned_buffer: Buffer,
    aligned_base: *mut u8,
    /// Byte shift such that `(offset + shift) mod 16 == 0` for the
    /// data-section-aligned safetensors layout. Computed from the
    /// 8-byte little-endian header_size prefix of the mmap. Falls
    /// back to `0` if the prefix doesn't look like safetensors —
    /// in that case the zero-copy gate behaves identically to the
    /// pre-bulk-copy world (just operating against the new buffer).
    shift: usize,
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
    /// Command queue lazily allocated on first `register_mmap`, used
    /// for the bulk MTLBlit DMA copy mmap → aligned_buffer. Held by
    /// `Mutex<Option<...>>` so `Clone` shares the queue across
    /// `MetalAllocator` handles and so the queue is dropped together
    /// with the last handle.
    bulk_copy_queue: Arc<Mutex<Option<CommandQueue>>>,
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
            bulk_copy_queue: Arc::clone(&self.bulk_copy_queue),
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
            bulk_copy_queue: Arc::new(Mutex::new(None)),
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
            bulk_copy_queue: Arc::new(Mutex::new(None)),
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
                // point into the per-region pre-aligned MTLBuffer (not
                // into the original mmap). The valid byte range is
                // `[aligned_base + shift, aligned_base + shift + len)`
                // — that's where the bulk-copy landed the mmap bytes.
                let a_start = region.aligned_base as usize + region.shift;
                let a_end = a_start + region.len;
                if p >= a_start && p < a_end {
                    return Some((region.aligned_buffer.clone(), (p - a_start + region.shift) as u64));
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

    /// Read the safetensors header prefix and compute the byte shift
    /// such that the data section starts at a 16-aligned offset within
    /// the pre-aligned destination buffer.
    ///
    /// Safetensors format: `[u64 header_size_le][header JSON
    /// (header_size bytes)][data section]`. The data section starts at
    /// `8 + header_size`. If we copy the mmap into a 16-aligned
    /// destination buffer at offset `shift`, the data section lands at
    /// `dest_base + shift + 8 + header_size`. We want that quantity to
    /// be `mod 16 == 0`, so `shift = (-(8 + header_size)) mod 16`.
    ///
    /// Returns 0 on any of:
    /// - mmap shorter than 8 bytes (no header to read)
    /// - `header_size` would put the data section past EOF
    /// - the bytes don't look like a safetensors prefix
    ///
    /// In those cases the bulk copy still happens (just without the
    /// shift trick), so zero-copy on a future tensor offset is still
    /// possible if that offset is naturally 16-aligned.
    fn compute_safetensors_shift(base: *const u8, len: usize) -> usize {
        if len < 8 {
            return 0;
        }
        // SAFETY: `base` points to at least 8 mapped bytes.
        let header_size =
            unsafe { std::ptr::read_unaligned(base as *const u64).to_le() } as usize;
        if header_size == 0 || header_size > len.saturating_sub(8) {
            return 0;
        }
        let data_section_start = 8 + header_size;
        let r = data_section_start % Self::MIN_BIND_ALIGN;
        if r == 0 {
            0
        } else {
            Self::MIN_BIND_ALIGN - r
        }
    }

    /// Get-or-create the per-allocator command queue used for the bulk
    /// MTLBlit copy. One queue is enough since `register_mmap` calls
    /// `waitUntilCompleted` on every blit (the bulk copy is loader-
    /// side, not on the hot path) — concurrent in-flight blits aren't
    /// needed.
    fn bulk_copy_queue(&self) -> CommandQueue {
        let mut slot = self
            .bulk_copy_queue
            .lock()
            .expect("MetalAllocator bulk_copy_queue Mutex");
        if let Some(q) = slot.as_ref() {
            return q.clone();
        }
        let q = self
            .device
            .newCommandQueue()
            .expect("MTLDevice.newCommandQueue returned nil");
        *slot = Some(q.clone());
        q
    }

    pub fn register_mmap(&self, mmap: Arc<memmap2::Mmap>) {
        let base = mmap.as_ptr();
        let len = mmap.len();
        if len == 0 {
            return;
        }

        // Compute the shift so the safetensors data section lands at
        // a 16-aligned offset in the destination buffer. Falls back
        // to 0 on non-safetensors prefixes (still does the bulk copy
        // but without the alignment trick).
        let shift = Self::compute_safetensors_shift(base, len);
        let dst_capacity = len + shift;

        // Destination: fresh, 16-aligned (MTLDevice returns page-
        // aligned buffers; pages are ≥ 16 bytes) `MTLBuffer`. Sized
        // exactly to hold the mmap contents plus the shift prefix.
        let dst_buffer = self
            .device
            .newBufferWithLength_options(dst_capacity, MTLResourceOptions::StorageModeShared)
            .expect("MetalAllocator::register_mmap: newBufferWithLength returned nil");
        let aligned_base = dst_buffer.contents().as_ptr() as *mut u8;
        assert!(
            !aligned_base.is_null(),
            "MetalAllocator::register_mmap: aligned destination buffer.contents() is null"
        );

        // Source: transient `newBufferWithBytesNoCopy` view of the
        // mmap, needed to give the blit encoder a `MTLBuffer` handle.
        // Released as soon as the blit completes — the underlying
        // bytes stay alive via `mmap`'s Arc on the caller's side, but
        // the noCopy wrapper itself goes away.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let src_buffer_len = (len + page_size - 1) & !(page_size - 1);
        // SAFETY: `base` covers `src_buffer_len` valid mapped pages
        // (mmap over-maps to a page boundary). `deallocator: None`
        // keeps Metal from trying to free the bytes; the Arc held by
        // the caller keeps the pages alive until the blit completes.
        let src_buffer = unsafe {
            let bytes = NonNull::new(base as *mut c_void).expect("non-null mmap base");
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    bytes,
                    src_buffer_len,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
                .expect("newBufferWithBytesNoCopy returned nil")
        };

        // Bulk-copy mmap → aligned_buffer at offset `shift` via the
        // blit engine (~30 GB/s on Apple Silicon vs ~5 GB/s CPU
        // memcpy_nonoverlapping on M2/M3 unified-memory hardware).
        let queue = self.bulk_copy_queue();
        let cmd_buf = queue
            .commandBuffer()
            .expect("MTLCommandQueue.commandBuffer returned nil");
        let blit = cmd_buf
            .blitCommandEncoder()
            .expect("MTLCommandBuffer.blitCommandEncoder returned nil");
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src_buffer,
                0,
                &dst_buffer,
                shift,
                len,
            );
        }
        blit.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        // `src_buffer` drops here — the noCopy wrapper is gone, the
        // mmap pages stay live via the caller's Arc.

        self.residency.insert(&dst_buffer);

        self.mmaps
            .lock()
            .expect("MetalAllocator mmaps Mutex")
            .push(MmapRegion {
                base,
                len,
                aligned_buffer: dst_buffer,
                aligned_base,
                shift,
                _mmap: mmap,
            });
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

    /// `Some(aligned_ptr)` if `src` lies within a registered mmap AND
    /// the shifted offset `offset + region.shift` is a multiple of
    /// `min_align`. The returned pointer points into the per-region
    /// pre-aligned destination buffer (not into the mmap) — the bulk
    /// MTLBlit at `register_mmap` time copied the mmap bytes there at
    /// offset `region.shift`, so reading from `aligned_ptr` produces
    /// the same bytes the caller would have read from `src`.
    ///
    /// `None` otherwise — either `src` is outside all registered
    /// mmaps, or `(offset + shift)` doesn't meet the binding's
    /// `min_align` requirement.
    ///
    /// `min_align` is clamped to `MIN_BIND_ALIGN` from above on the
    /// trait `alloc_and_copy_host` path; the
    /// `alloc_and_copy_host_aligned` path passes the per-binding-dtype
    /// scalar alignment (2 for F16/BF16, 4 for U32/F32, etc.).
    fn aligned_mmap_offset(
        &self,
        src: *const u8,
        bytes: usize,
        min_align: usize,
    ) -> Option<*mut u8> {
        let p = src as usize;
        let end = p.saturating_add(bytes);
        let mmaps = self.mmaps.lock().expect("MetalAllocator mmaps Mutex");
        for region in mmaps.iter() {
            let r_start = region.base as usize;
            let r_end = r_start + region.len;
            if p >= r_start && end <= r_end {
                let mmap_offset = p - r_start;
                let shifted = mmap_offset + region.shift;
                if min_align == 0 || shifted % min_align == 0 {
                    let aligned_ptr =
                        unsafe { region.aligned_base.add(shifted) };
                    return Some(aligned_ptr);
                }
                return None;
            }
        }
        None
    }

    fn src_in_registered_mmap(&self, src: *const u8, bytes: usize) -> bool {
        self.aligned_mmap_offset(src, bytes, Self::MIN_BIND_ALIGN)
            .is_some()
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
        let mmaps = self.mmaps.lock().expect("MetalAllocator mmaps Mutex");
        for region in mmaps.iter() {
            let r_start = region.base as usize;
            let r_end = r_start + region.len;
            if p >= r_start && end <= r_end {
                let mmap_offset = p - r_start;
                // Histogram still buckets by the RAW intra-mmap offset
                // (pre-shift): it characterizes the safetensors file
                // layout, not our routing. After this change the count
                // at `tz == 4` (mod 16 = 0 post-shift) effectively
                // tells you how successful the shift was.
                let tz = mmap_offset.trailing_zeros();
                self.load_stats.observe_offset_alignment(tz);
                let shifted = mmap_offset + region.shift;
                if min_align == 0 || shifted % min_align == 0 {
                    let aligned_ptr =
                        unsafe { region.aligned_base.add(shifted) };
                    return MmapClassify::Aligned { aligned_ptr };
                }
                return MmapClassify::Unaligned;
            }
        }
        MmapClassify::Outside
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
