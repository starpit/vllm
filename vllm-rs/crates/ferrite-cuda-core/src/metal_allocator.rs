// SPDX-License-Identifier: Apache-2.0
//! Metal implementation of the [`DeviceAllocator`] trait.
//!
//! On Apple silicon (unified memory), an `MTLBuffer` allocated with
//! `StorageModeShared` is backed by the same DRAM the CPU uses. The
//! buffer's `contents()` pointer is host-writable and device-visible
//! at the same virtual address — there is no DMA to schedule, no
//! pinned-memory dance. "H2D copy" collapses to a `memcpy` into the
//! buffer's mapped contents.
//!
//! The allocator backs weight memory with one or more arena
//! `MTLBuffer`s. Each `alloc_and_copy_host` bumps within the current
//! arena; if a request doesn't fit, a new arena buffer is allocated
//! sized to at least the request (rounded up to a default chunk
//! size). Returned pointers stay valid for the allocator's lifetime
//! (each arena buffer is retained in `arenas` and freed only on
//! `take_allocations` / drop).
//!
//! # Pointer → buffer lookup
//!
//! Encoder bindings need `(&MTLBuffer, offset)`, not raw pointers.
//! [`MetalAllocator::buffer_for`] does a linear scan over the arenas
//! to find the one containing a given pointer and returns
//! `(&Buffer, offset)`. Linear is fine: weight loads produce a
//! handful of arenas (one per ~256MB), and the lookup runs once per
//! weight binding at ICB-record time, not per forward pass.

#![cfg(feature = "metal")]

use std::sync::{Arc, Mutex};

use anyhow::Result;
use metal::{Buffer, Device, MTLResourceOptions};

use crate::device_allocator::DeviceAllocator;

/// Default chunk size for new arena buffers (256 MB). Picked to
/// balance "few enough arenas to keep `buffer_for` linear scan
/// cheap" against "small enough that wastage on the last arena is
/// bounded". TinyLlama-1.1B (~2.2 GB f16) lands in ~9 chunks.
const DEFAULT_CHUNK_BYTES: usize = 256 * 1024 * 1024;

/// One arena buffer + its bump-pointer state.
struct MetalArena {
    buffer: Buffer,
    /// `buffer.contents()` cached as `*mut u8` so the bump pointer
    /// is a plain pointer add. Equal to the buffer's mapped host
    /// pointer for `StorageModeShared` allocations.
    base: *mut u8,
    capacity: usize,
    used: usize,
}

// Safety: MTLBuffer is documented as thread-safe for concurrent
// reads of `contents()`. The bump pointer is mutated through `&mut
// self` only, so no inter-thread race on `used`.
unsafe impl Send for MetalArena {}
unsafe impl Sync for MetalArena {}

/// Zero-copy mmap region wrapped in an `MTLBuffer` via
/// `newBufferWithBytesNoCopy`. Registered up-front by `register_mmap`
/// (called once per safetensors shard), then consulted by
/// [`MetalAllocator::alloc_and_copy_host`] to short-circuit the arena
/// memcpy when the source pointer falls inside the mmap.
///
/// The `_mmap` `Arc` is what keeps the mapped pages alive — Metal's
/// `newBufferWithBytesNoCopy` takes a deallocator block of `None`,
/// meaning the buffer does NOT free the bytes; we own the lifetime.
struct MmapRegion {
    /// `mmap.as_ptr()` — page-aligned start of the mapped file.
    base: *const u8,
    /// File length (NOT the page-aligned virtual region length).
    /// Bounds checks against this so a stray pointer past EOF doesn't
    /// resolve to a buffer offset.
    len: usize,
    /// `MTLBuffer` aliasing `[base, base + page_align(len))`. Cheap
    /// ObjC-retained handle — clones share the underlying buffer.
    buffer: Buffer,
    _mmap: Arc<memmap2::Mmap>,
}

unsafe impl Send for MmapRegion {}
unsafe impl Sync for MmapRegion {}

/// Metal-backed [`DeviceAllocator`]. Holds a `Device` handle and a
/// shared `Vec<MetalArena>` — one per ~256 MB arena. Allocations are
/// served from the current arena until full, then a new arena is
/// pushed.
///
/// `Clone` shares state via `Arc<Mutex<…>>` on `arenas`. Both the
/// loader-side `GpuWeights<MetalAllocator>` and the runtime-side
/// `GpuDevice.allocator` need to see the SAME arena set so the
/// worker's `buffer_for` reverse-lookup can resolve a freshly-uploaded
/// weight pointer back to its `MTLBuffer`. Cloning the allocator is
/// the only way to share state across owners that take it by value
/// (`GpuWeights::from_dir(_, BackendAllocator)`).
/// Hook fired immediately after a new arena `MTLBuffer` is created.
/// Originally introduced so a higher-layer crate could drop the buffer
/// into a `MTLResidencySet`; that responsibility now lives directly on
/// the allocator (see [`MetalAllocator::residency`]) but the hook is
/// retained as a generic post-allocation extension point in case
/// something else wants to observe arena creation.
pub type ArenaHook = Arc<dyn Fn(&Buffer) + Send + Sync>;

pub struct MetalAllocator {
    device: Device,
    arenas: Arc<Mutex<Vec<MetalArena>>>,
    chunk_bytes: usize,
    on_new_arena: Arc<Mutex<Option<ArenaHook>>>,
    /// Single `MTLResidencySet` covering every arena allocated through
    /// this allocator. Both the runtime worker pool (per-worker arena
    /// buffers) and the KV-cache pool (private-mode cache buffers)
    /// insert into the SAME set via [`Self::residency`], matching MLX's
    /// "one residency set per device queue" layout. Inert (no-op) on
    /// macOS < 15. Created in [`Self::new`] so existing arenas are
    /// pinned automatically as they are pushed.
    residency: ferrite_metal_kernels::residency::MetalResidencySet,
    /// Zero-copy mmap regions registered via [`Self::register_mmap`].
    /// Consulted before the bump arena in
    /// [`Self::alloc_and_copy_host`] — if the source pointer falls
    /// inside one of these regions, the upload is a no-op and we
    /// return the source pointer unchanged. MLX uses the same trick
    /// for safetensors-backed arrays; matches "memcpy-free weight
    /// load" in the vllm-mlx startup path.
    mmaps: Arc<Mutex<Vec<MmapRegion>>>,
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
            // Cheap: `MetalResidencySet` is `Arc`-backed; cloning shares
            // the underlying `MTLResidencySet*` so every clone of the
            // allocator pins into the same set.
            residency: self.residency.clone(),
            mmaps: Arc::clone(&self.mmaps),
        }
    }
}

impl MetalAllocator {
    /// Create an allocator on `device`. No buffer is allocated until
    /// the first `alloc_and_copy_host` call. The allocator's
    /// [`Self::residency`] set is created eagerly so callers can pin
    /// non-arena buffers (KV cache, etc.) into the same set.
    pub fn new(device: Device) -> Self {
        let residency = ferrite_metal_kernels::residency::MetalResidencySet::new(&device);
        Self {
            device,
            arenas: Arc::new(Mutex::new(Vec::new())),
            chunk_bytes: DEFAULT_CHUNK_BYTES,
            on_new_arena: Arc::new(Mutex::new(None)),
            residency,
            mmaps: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Override the default arena chunk size. Useful for tests that
    /// want to exercise the multi-arena code path without uploading
    /// hundreds of megabytes.
    pub fn with_chunk_bytes(device: Device, chunk_bytes: usize) -> Self {
        let residency = ferrite_metal_kernels::residency::MetalResidencySet::new(&device);
        Self {
            device,
            arenas: Arc::new(Mutex::new(Vec::new())),
            chunk_bytes,
            on_new_arena: Arc::new(Mutex::new(None)),
            residency,
            mmaps: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Shared residency set covering every arena allocated through
    /// this allocator. Higher layers (the worker pool, the KV-cache
    /// initializer) `insert` their non-arena buffers here and call
    /// `attach_to_queue` once so every cmdbuf sees a single wired
    /// set — this is what MLX does, and avoids the
    /// two-residency-sets-per-queue layout that produced
    /// non-deterministic decode output on Llama-3.2.
    pub fn residency(&self) -> &ferrite_metal_kernels::residency::MetalResidencySet {
        &self.residency
    }

    /// Install (or replace) the hook fired after every new arena
    /// `MTLBuffer` allocation. Existing arenas are replayed through
    /// the hook immediately so the caller doesn't have to track
    /// initial state.
    ///
    /// Used by `ferrite-metal-kernels::residency` to wire arena
    /// buffers into a `MTLResidencySet` (Metal 3 / macOS 15+) so they
    /// stay resident across cmdbufs. Without that pinning, large
    /// working sets (Llama-3.2-3B+) hit Apple's lazy paging path and
    /// produce non-deterministic decode output.
    pub fn set_arena_hook(&self, hook: ArenaHook) {
        let arenas = self.arenas.lock().expect("MetalAllocator arenas Mutex");
        for arena in arenas.iter() {
            (hook)(&arena.buffer);
        }
        *self.on_new_arena.lock().expect("arena hook mutex") = Some(hook);
    }

    /// The Metal device this allocator targets. Exposed so callers
    /// (e.g. the worker's specialized-pipeline cache) can build
    /// device-bound state alongside.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Look up the buffer containing `ptr` and return
    /// `(buffer, offset)` suitable for `set_buffer(...)`. Returns
    /// `None` if `ptr` is not within any registered mmap region or
    /// any arena (caller bug — raw-pointer parameters never crossed
    /// this allocator).
    ///
    /// mmap regions are checked first because the zero-copy weight
    /// path (registered via [`Self::register_mmap`]) puts every
    /// safetensors-backed weight pointer into one of them. The arena
    /// fallback covers cast-scratch slow-path uploads (F32→BF16) and
    /// transient loader-allocated tensors (RotaryCache, etc.).
    ///
    /// Returns an owned `Buffer` (refcounted ObjC handle, cheap to
    /// clone) so the caller doesn't have to thread the allocator's
    /// `Mutex` lock guard through the dispatch pipeline.
    pub fn buffer_for(&self, ptr: *const u8) -> Option<(Buffer, u64)> {
        let p = ptr as usize;
        {
            let mmaps = self.mmaps.lock().expect("MetalAllocator mmaps Mutex");
            for region in mmaps.iter() {
                let start = region.base as usize;
                let end = start + region.len;
                if p >= start && p < end {
                    return Some((region.buffer.clone(), (p - start) as u64));
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

    /// Register a memory-mapped region (typically a safetensors
    /// shard) as a zero-copy source for subsequent
    /// `alloc_and_copy_host` calls. Wraps the mmap in an `MTLBuffer`
    /// via `newBufferWithBytesNoCopy` (no deallocator — the caller's
    /// `Arc<Mmap>` keeps the pages alive, stashed inside the region
    /// here so a dropped `GpuWeights` doesn't pull the rug).
    ///
    /// Page-aligns the buffer length up; mmap over-maps to a page
    /// boundary so the trailing bytes-past-EOF fall inside an
    /// already-mapped page (zero-filled). `len` is the file length —
    /// `buffer_for` bounds-checks against this so a stray pointer
    /// past the end of the file doesn't resolve to a buffer offset.
    ///
    /// MLX uses the same trick for its safetensors loader; this is
    /// the difference between vllm-mlx's ~300ms startup and the
    /// 2.2 GB memcpy we used to do for Llama-3.2-1B.
    pub fn register_mmap(&self, mmap: Arc<memmap2::Mmap>) {
        let base = mmap.as_ptr();
        let len = mmap.len();
        if len == 0 {
            return;
        }
        // SAFETY: `sysconf(_SC_PAGESIZE)` is documented to return a
        // positive page size on every supported platform.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let buffer_len = (len + page_size - 1) & !(page_size - 1);
        let buffer = self.device.new_buffer_with_bytes_no_copy(
            base as *const std::ffi::c_void,
            buffer_len as metal::NSUInteger,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        // Pin into the shared residency set, but **do not commit** —
        // commit is the expensive step (Apple's residency tracker
        // marks pages wired; on a 5 GB shard this was ~35ms per
        // call). Per the residency.rs doc comment "batching multiple
        // inserts before a single commit cuts down on driver chatter":
        // we let the next commit() in the load chain (initialize_cache,
        // which fires after every load_model) sweep the queued inserts
        // along with the KV-cache buffers it adds. The mmap MTLBuffers
        // aren't used until the first forward — well after init_cache.
        self.residency.insert(&buffer);
        self.mmaps
            .lock()
            .expect("MetalAllocator mmaps Mutex")
            .push(MmapRegion {
                base,
                len,
                buffer,
                _mmap: mmap,
            });
    }

    /// Returns true iff `[src, src + bytes)` is fully contained in
    /// any registered mmap region. Used by `alloc_and_copy_host` to
    /// take the zero-copy path.
    fn src_in_registered_mmap(&self, src: *const u8, bytes: usize) -> bool {
        let p = src as usize;
        let end = p.saturating_add(bytes);
        let mmaps = self.mmaps.lock().expect("MetalAllocator mmaps Mutex");
        mmaps.iter().any(|region| {
            let r_start = region.base as usize;
            let r_end = r_start + region.len;
            p >= r_start && end <= r_end
        })
    }

    /// Number of live arenas. Diagnostic.
    pub fn arena_count(&self) -> usize {
        self.arenas
            .lock()
            .expect("MetalAllocator arenas Mutex")
            .len()
    }

    /// Total bytes allocated across all arenas (sum of `used`).
    /// Diagnostic.
    pub fn used_bytes(&self) -> usize {
        self.arenas
            .lock()
            .expect("MetalAllocator arenas Mutex")
            .iter()
            .map(|a| a.used)
            .sum()
    }

    /// Push a new arena buffer of at least `min_bytes`, rounded up
    /// to `chunk_bytes`. Returns the index of the new arena. The
    /// caller already holds the `arenas` lock.
    fn push_arena_locked(
        device: &Device,
        arenas: &mut Vec<MetalArena>,
        chunk_bytes: usize,
        min_bytes: usize,
        hook: &Arc<Mutex<Option<ArenaHook>>>,
        residency: &ferrite_metal_kernels::residency::MetalResidencySet,
    ) -> Result<usize> {
        let capacity = min_bytes.max(chunk_bytes);
        let buffer = device.new_buffer(capacity as u64, MTLResourceOptions::StorageModeShared);
        let base = buffer.contents() as *mut u8;
        if base.is_null() {
            anyhow::bail!(
                "MetalAllocator: new_buffer({} bytes) returned null contents pointer",
                capacity
            );
        }
        // Pin the new arena into the shared residency set so cmdbufs
        // don't race against Apple's lazy paging once the working set
        // crosses the implicit-residency tracker's threshold. Inert
        // on macOS < 15 (set is null). `commit()` is deferred — see
        // `register_mmap`'s comment on why per-call commits are
        // expensive; the next `commit()` in the load chain (typically
        // `initialize_cache`) sweeps every queued insert.
        residency.insert(&buffer);
        // Then notify any external arena hook (kept as a generic
        // post-allocation extension point — the residency insert
        // itself no longer goes through this hook).
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

    /// Allocate `bytes` of arena space and return the destination
    /// pointer **without copying anything**. Caller is responsible
    /// for writing exactly `bytes` valid bytes before the buffer is
    /// read.
    ///
    /// Used by the load path's pack-into-place optimization
    /// (`load_dense_concat_packed`): instead of building a heap
    /// `Vec<u8>` of concatenated tensor bytes and then memcpying
    /// that Vec into an arena slot (two memcpies, ~3 GB at
    /// 100 MB/layer × 28 layers for Llama-3.2-3B), the loader
    /// pre-allocates the arena slot here and writes each component
    /// directly into it (one memcpy total).
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
        let idx = if let Some(idx) = arenas.iter().rposition(|a| a.capacity - a.used >= bytes) {
            idx
        } else {
            Self::push_arena_locked(
                &self.device,
                &mut arenas,
                self.chunk_bytes,
                bytes,
                &self.on_new_arena,
                &self.residency,
            )?
        };
        let arena = &mut arenas[idx];
        let offset = arena.used;
        let dst = unsafe { arena.base.add(offset) };
        arena.used = offset + bytes;
        Ok(dst)
    }
}

impl DeviceAllocator for MetalAllocator {
    unsafe fn alloc_and_copy_host(&mut self, src_host: *const u8, bytes: usize) -> Result<*mut u8> {
        // Zero-copy fast path: if `src_host` falls inside a registered
        // mmap region (a safetensors shard mapped through
        // `register_mmap`), we already have an `MTLBuffer` aliasing
        // those pages. Skip the arena memcpy and hand the source
        // pointer back unchanged — `buffer_for(ptr)` resolves it to
        // the mmap region's buffer at the right offset. This is the
        // path that closes the 2.2 GB-of-memcpy gap vs vllm-mlx for
        // bf16/f16-on-disk weights with no cast.
        if bytes > 0 && self.src_in_registered_mmap(src_host, bytes) {
            return Ok(src_host as *mut u8);
        }
        let mut arenas = self.arenas.lock().expect("MetalAllocator arenas Mutex");
        // Zero-byte tensors are legal (e.g. an unused bias slot);
        // pick any non-null sentinel so the GpuTensor isn't `is_null()`.
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

        // Find an arena with `bytes` free, or push a new one.
        let idx = if let Some(idx) = arenas.iter().rposition(|a| a.capacity - a.used >= bytes) {
            idx
        } else {
            Self::push_arena_locked(
                &self.device,
                &mut arenas,
                self.chunk_bytes,
                bytes,
                &self.on_new_arena,
                &self.residency,
            )?
        };
        let arena = &mut arenas[idx];
        let offset = arena.used;
        let dst = unsafe { arena.base.add(offset) };
        // Unified-memory memcpy. `dst` is host-writable AND device-
        // visible — no DMA to schedule, no synchronization needed.
        unsafe { std::ptr::copy_nonoverlapping(src_host, dst, bytes) };
        arena.used = offset + bytes;
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_device() -> Option<Device> {
        Device::system_default()
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
        assert_eq!(alloc.used_bytes(), src.len());
        assert_eq!(alloc.arena_count(), 1);

        let (buf, off) = alloc.buffer_for(ptr).expect("buffer_for");
        assert_eq!(off, 0);
        assert_eq!(buf.length() as usize, DEFAULT_CHUNK_BYTES);
    }

    #[test]
    fn multiple_allocs_share_arena_until_full() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        // Tiny chunks (4 KB) so a few allocs cross arena boundaries.
        let mut alloc = MetalAllocator::with_chunk_bytes(device, 4096);
        let buf_a = vec![0xAAu8; 1024];
        let buf_b = vec![0xBBu8; 1024];
        let buf_c = vec![0xCCu8; 3072]; // forces new arena (4 KB - 2 KB used = 2 KB free; 3 KB doesn't fit)

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

        // pa, pb in arena 0; pc in arena 1.
        let (ba, oa) = alloc.buffer_for(pa).unwrap();
        let (bb, ob) = alloc.buffer_for(pb).unwrap();
        let (bc, oc) = alloc.buffer_for(pc).unwrap();
        assert!(std::ptr::eq(ba, bb), "pa, pb should be in same arena");
        assert!(!std::ptr::eq(ba, bc), "pc should be in a new arena");
        assert_eq!(oa, 0);
        assert_eq!(ob, 1024);
        assert_eq!(oc, 0);

        let read_a = unsafe { std::slice::from_raw_parts(pa, buf_a.len()) };
        let read_c = unsafe { std::slice::from_raw_parts(pc, buf_c.len()) };
        assert!(read_a.iter().all(|&b| b == 0xAA));
        assert!(read_c.iter().all(|&b| b == 0xCC));

        assert_eq!(alloc.arena_count(), 2);
        assert_eq!(alloc.used_bytes(), 1024 + 1024 + 3072);
    }

    #[test]
    fn oversized_request_gets_dedicated_arena() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let mut alloc = MetalAllocator::with_chunk_bytes(device, 4096);
        // Request bigger than chunk_bytes — should push an arena
        // sized to the request.
        let big = vec![0x42u8; 16 * 1024];
        let p = unsafe { alloc.alloc_and_copy_host(big.as_ptr(), big.len()).unwrap() };
        let (buf, off) = alloc.buffer_for(p).unwrap();
        assert_eq!(off, 0);
        assert!(buf.length() as usize >= big.len());
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
        // No allocations made. A random pointer isn't in any arena.
        let stack = 0u8;
        assert!(alloc.buffer_for(&stack).is_none());
    }
}
