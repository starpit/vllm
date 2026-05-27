// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral raw GPU memory ownership.

#![cfg(any(feature = "cuda", feature = "metal"))]

#[cfg(feature = "cuda")]
use crate::driver;

#[cfg(feature = "metal")]
use objc2::rc::Retained;
#[cfg(feature = "metal")]
use objc2::runtime::ProtocolObject;
#[cfg(feature = "metal")]
use objc2_metal::MTLBuffer;

#[cfg(feature = "metal")]
pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

pub struct RawGpuMem {
    #[cfg(feature = "cuda")]
    ptr: *mut u8,
    #[cfg(feature = "cuda")]
    size: usize,
    #[cfg(feature = "metal")]
    buffer: Buffer,
    #[cfg(feature = "metal")]
    base: *mut u8,
    #[cfg(feature = "metal")]
    size: usize,
    // Byte offset into `buffer`'s gpuAddress range. 0 for ordinary
    // `from_buffer`-allocated chunks (one MTLBuffer per chunk); non-zero
    // when this `RawGpuMem` represents a tile sub-range of a shared
    // placement-sparse buffer (one MTLBuffer per LAYER, many chunks
    // co-located at distinct offsets). Dropping such a clone-shared
    // `RawGpuMem` is harmless: the underlying sparse `Buffer` is
    // ref-counted (`Retained`) and stays alive as long as the
    // `SparseKvLayer` owning it does.
    #[cfg(feature = "metal")]
    metal_offset: usize,
}

unsafe impl Send for RawGpuMem {}
unsafe impl Sync for RawGpuMem {}

#[cfg(feature = "cuda")]
impl RawGpuMem {
    pub unsafe fn new(ptr: *mut u8, size: usize) -> Self {
        Self { ptr, size }
    }

    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn leak(mut self) -> (*mut u8, usize) {
        let result = (self.ptr, self.size);
        self.ptr = std::ptr::null_mut();
        std::mem::forget(self);
        result
    }
}

#[cfg(feature = "metal")]
impl RawGpuMem {
    pub fn from_buffer(buffer: Buffer) -> Self {
        let base = buffer.contents().as_ptr() as *mut u8;
        let size = buffer.length();
        Self {
            buffer,
            base,
            size,
            metal_offset: 0,
        }
    }

    /// Wrap a sub-range `[offset, offset+size)` of a SHARED buffer.
    /// Used by the placement-sparse KV path where one MTLBuffer per layer
    /// backs many chunk indices at distinct offsets. `buffer` is a clone
    /// of a `Retained` so dropping this `RawGpuMem` does NOT free the
    /// underlying buffer.
    pub fn from_buffer_with_offset(buffer: Buffer, offset: usize, size: usize) -> Self {
        // For StorageModePrivate sparse buffers, `contents()` returns null;
        // `base` is non-load-bearing on the metal path, kept for parity.
        let base = std::ptr::null_mut();
        Self {
            buffer,
            base,
            size,
            metal_offset: offset,
        }
    }

    pub fn ptr(&self) -> *mut u8 {
        self.base
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// GPU virtual address of this allocation's first byte. For ordinary
    /// per-chunk buffers this equals `buffer.gpuAddress()`; for sparse
    /// sub-ranges it adds the byte offset within the shared buffer.
    /// Use this in `fill_chunk_tables` / `grow_to_cover` gpu_addr closures
    /// so the sparse and dense paths share one accessor.
    pub fn gpu_address(&self) -> u64 {
        use objc2_metal::MTLBuffer;
        self.buffer.gpuAddress() + self.metal_offset as u64
    }

    /// Byte offset within `buffer()` where this allocation begins. Zero
    /// for ordinary per-chunk buffers, non-zero for sparse sub-ranges.
    pub fn metal_offset(&self) -> usize {
        self.metal_offset
    }
}

impl Drop for RawGpuMem {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if !self.ptr.is_null() {
            unsafe {
                let _ = driver::mem_free(self.ptr);
            }
        }
    }
}
