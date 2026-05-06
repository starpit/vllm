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

/// Metal-backed [`DeviceAllocator`]. Holds a `Device` handle and a
/// `Vec<MetalArena>` — one per ~256 MB arena. Allocations are
/// served from the current arena until full, then a new arena is
/// pushed.
pub struct MetalAllocator {
    device: Device,
    arenas: Vec<MetalArena>,
    chunk_bytes: usize,
}

unsafe impl Send for MetalAllocator {}
unsafe impl Sync for MetalAllocator {}

impl MetalAllocator {
    /// Create an allocator on `device`. No buffer is allocated until
    /// the first `alloc_and_copy_host` call.
    pub fn new(device: Device) -> Self {
        Self {
            device,
            arenas: Vec::new(),
            chunk_bytes: DEFAULT_CHUNK_BYTES,
        }
    }

    /// Override the default arena chunk size. Useful for tests that
    /// want to exercise the multi-arena code path without uploading
    /// hundreds of megabytes.
    pub fn with_chunk_bytes(device: Device, chunk_bytes: usize) -> Self {
        Self {
            device,
            arenas: Vec::new(),
            chunk_bytes,
        }
    }

    /// The Metal device this allocator targets. Exposed so callers
    /// (e.g. the worker's specialized-pipeline cache) can build
    /// device-bound state alongside.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Look up the arena buffer containing `ptr` and return
    /// `(buffer, offset)` suitable for `set_buffer(...)`. Returns
    /// `None` if `ptr` is not within any arena (caller bug —
    /// raw-pointer parameters never crossed this allocator).
    pub fn buffer_for(&self, ptr: *const u8) -> Option<(&Buffer, u64)> {
        for arena in &self.arenas {
            let start = arena.base as usize;
            let end = start + arena.used;
            let p = ptr as usize;
            if p >= start && p < end {
                return Some((&arena.buffer, (p - start) as u64));
            }
        }
        None
    }

    /// Number of live arenas. Diagnostic.
    pub fn arena_count(&self) -> usize {
        self.arenas.len()
    }

    /// Total bytes allocated across all arenas (sum of `used`).
    /// Diagnostic.
    pub fn used_bytes(&self) -> usize {
        self.arenas.iter().map(|a| a.used).sum()
    }

    /// Push a new arena buffer of at least `min_bytes`, rounded up
    /// to `chunk_bytes`. Returns the index of the new arena.
    fn push_arena(&mut self, min_bytes: usize) -> Result<usize> {
        let capacity = min_bytes.max(self.chunk_bytes);
        let buffer = self
            .device
            .new_buffer(capacity as u64, MTLResourceOptions::StorageModeShared);
        let base = buffer.contents() as *mut u8;
        if base.is_null() {
            anyhow::bail!(
                "MetalAllocator: new_buffer({} bytes) returned null contents pointer",
                capacity
            );
        }
        self.arenas.push(MetalArena {
            buffer,
            base,
            capacity,
            used: 0,
        });
        Ok(self.arenas.len() - 1)
    }

    /// Find an arena with `bytes` free, or push a new one. Returns
    /// the arena index and the offset within that arena where
    /// `bytes` will be placed (post-bump, the destination).
    fn reserve(&mut self, bytes: usize) -> Result<(usize, usize)> {
        if let Some(idx) = self
            .arenas
            .iter()
            .rposition(|a| a.capacity - a.used >= bytes)
        {
            let offset = self.arenas[idx].used;
            return Ok((idx, offset));
        }
        let idx = self.push_arena(bytes)?;
        Ok((idx, 0))
    }
}

impl DeviceAllocator for MetalAllocator {
    unsafe fn alloc_and_copy_host(&mut self, src_host: *const u8, bytes: usize) -> Result<*mut u8> {
        // Zero-byte tensors are legal (e.g. an unused bias slot);
        // pick any non-null sentinel so the GpuTensor isn't `is_null()`.
        if bytes == 0 {
            // Push a placeholder arena if none exists so we have a
            // base to point into.
            if self.arenas.is_empty() {
                self.push_arena(0)?;
            }
            return Ok(self.arenas[0].base);
        }

        let (idx, offset) = self.reserve(bytes)?;
        let arena = &mut self.arenas[idx];
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
