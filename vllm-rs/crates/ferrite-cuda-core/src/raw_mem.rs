// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral raw GPU memory ownership.
//!
//! [`RawGpuMem`] is a RAII handle for a single GPU allocation made
//! outside the caching allocator. It exposes a `*mut u8` pointer + a
//! byte size and frees the underlying memory on drop.
//!
//! - **CUDA:** wraps a `driver::mem_alloc` pointer and frees via
//!   `driver::mem_free` on drop.
//! - **Metal:** wraps a `metal::Buffer` (StorageModeShared) and
//!   relies on the buffer's own refcounted drop to free the
//!   `MTLBuffer`. The cached `base` pointer is `buffer.contents()`.
//!
//! Both backends present the same `ptr() -> *mut u8` / `size() ->
//! usize` API so callers (`KvCachePool`, `GpuWeights` quant load
//! path) read pointer + size without cfg-branches at the call site.
//! Backend-specific entry points (`new` for cuda, `from_buffer` for
//! metal) live behind `#[cfg]` because their inputs differ.

#![cfg(any(feature = "cuda", feature = "metal"))]

#[cfg(feature = "cuda")]
use crate::driver;

/// RAII wrapper for a raw GPU allocation (pointer + size) without
/// tensor metadata. Backend-neutral: same API on cuda and metal,
/// different storage and free path inside.
pub struct RawGpuMem {
    #[cfg(feature = "cuda")]
    ptr: *mut u8,
    #[cfg(feature = "cuda")]
    size: usize,
    #[cfg(feature = "metal")]
    buffer: metal::Buffer,
    #[cfg(feature = "metal")]
    base: *mut u8,
    #[cfg(feature = "metal")]
    size: usize,
}

// Safety: GPU device pointers are accessible from any host thread
// after the CUDA / Metal device handle is established. No
// thread-local state. On metal `metal::Buffer` is documented as
// thread-safe for refcount mutation.
unsafe impl Send for RawGpuMem {}
unsafe impl Sync for RawGpuMem {}

#[cfg(feature = "cuda")]
impl RawGpuMem {
    /// Create from a raw GPU pointer and size. The caller transfers
    /// ownership.
    ///
    /// # Safety
    /// `ptr` must be a valid GPU allocation from
    /// `driver::mem_alloc()` with the given `size`, and the caller
    /// must not free it separately.
    pub unsafe fn new(ptr: *mut u8, size: usize) -> Self {
        Self { ptr, size }
    }

    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Take ownership of the raw pointer, preventing `Drop` from
    /// freeing it. Returns `(ptr, size)`. CUDA-only — the metal
    /// backing is a refcounted buffer, not a free-able pointer, so
    /// `leak` does not have a meaningful counterpart there.
    pub fn leak(mut self) -> (*mut u8, usize) {
        let result = (self.ptr, self.size);
        self.ptr = std::ptr::null_mut();
        std::mem::forget(self);
        result
    }
}

#[cfg(feature = "metal")]
impl RawGpuMem {
    /// Wrap an `MTLBuffer`. The `RawGpuMem` keeps the buffer alive
    /// (its `Drop` releases the refcount and frees the underlying
    /// `MTLBuffer` if no other clones remain). The `base` pointer
    /// is `buffer.contents()`, host-writable and device-visible at
    /// the same VA on Apple silicon's unified memory.
    pub fn from_buffer(buffer: metal::Buffer) -> Self {
        let base = buffer.contents() as *mut u8;
        let size = buffer.length() as usize;
        Self { buffer, base, size }
    }

    pub fn ptr(&self) -> *mut u8 {
        self.base
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Access the underlying `MTLBuffer` for binding to compute
    /// encoders. The buffer is kept alive by `self`.
    pub fn buffer(&self) -> &metal::Buffer {
        &self.buffer
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
        // Metal: `metal::Buffer`'s own Drop releases the MTLBuffer.
    }
}
