// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral RAII tensor wrapper.
//!
//! [`OwnedTensor`] is a `GpuTensor` (descriptor) plus the storage
//! that backs it. Drop returns the memory to the appropriate place
//! per backend:
//!
//! - **CUDA:** holds a `*mut CachingAllocator` and calls
//!   `caching.free(ptr, size)` on drop, returning the block to the
//!   caching allocator's free pool.
//! - **Metal:** holds a `metal::Buffer` and relies on the buffer's
//!   own refcounted drop to release the `MTLBuffer`. No allocator
//!   handshake; each `OwnedTensor` is one `MTLBuffer`.
//!
//! The descriptor (`inner: GpuTensor`) and tensor-level operations
//! (`view`, `view_offset`, `reshape`, `Deref<Target = GpuTensor>`)
//! are identical across backends — the cfg-mutex is purely on the
//! storage field and the drop path.

#![cfg(any(feature = "cuda", feature = "metal"))]

use crate::dtype::DType;
use crate::tensor::{GpuTensor, TensorView};

/// A GPU tensor that owns its memory. Drop returns the memory to
/// the backend allocator (cuda) or releases the `MTLBuffer`
/// refcount (metal).
pub struct OwnedTensor {
    inner: GpuTensor,
    #[cfg(feature = "cuda")]
    pub(crate) alloc: *mut crate::alloc::CachingAllocator,
    #[cfg(feature = "metal")]
    _buffer: metal::Buffer,
    pub(crate) size_bytes: usize,
}

// Safety: same justification as the previous CUDA-only OwnedTensor:
// GPU device pointers are thread-safe after backend setup, and the
// cuda allocator pointer is only used on Drop on the same worker
// thread. On metal, `metal::Buffer` is a refcounted handle that is
// itself Send + Sync per the metal-rs binding.
unsafe impl Send for OwnedTensor {}
unsafe impl Sync for OwnedTensor {}

impl OwnedTensor {
    /// Construct a CUDA-backed `OwnedTensor`. Used by
    /// `CachingAllocator::alloc_tensor` after it has produced the
    /// pointer + size for `inner`.
    #[cfg(feature = "cuda")]
    pub(crate) unsafe fn from_caching_alloc(
        inner: GpuTensor,
        alloc: *mut crate::alloc::CachingAllocator,
        size_bytes: usize,
    ) -> Self {
        Self {
            inner,
            alloc,
            size_bytes,
        }
    }

    /// Construct a metal-backed `OwnedTensor`. The `inner`
    /// `GpuTensor`'s raw pointer must be `buffer.contents()`
    /// (Apple silicon unified memory keeps them VA-equal).
    #[cfg(feature = "metal")]
    pub fn from_metal_buffer(inner: GpuTensor, buffer: metal::Buffer, size_bytes: usize) -> Self {
        Self {
            inner,
            _buffer: buffer,
            size_bytes,
        }
    }

    pub fn as_gpu_tensor(&self) -> GpuTensor {
        self.inner
    }

    /// Borrow the backing `MTLBuffer`. Apple silicon's unified memory
    /// means `buffer.contents()` is the same VA as `inner.raw_ptr()` —
    /// callers that bind the buffer to a compute encoder (e.g.
    /// `dispatch_argmax_f16`) read identical bytes either way. The
    /// buffer reference is what the metal-rs binder API expects;
    /// kernels can't be set with raw pointers.
    #[cfg(feature = "metal")]
    pub fn metal_buffer(&self) -> &metal::Buffer {
        &self._buffer
    }

    /// Consume self, return GpuTensor WITHOUT freeing. The block
    /// stays allocated but is removed from the caching allocator's
    /// active-blocks tracking (so `free_leaked_blocks` can find it).
    ///
    /// CUDA-only: metal's storage is a refcounted `MTLBuffer`; there
    /// is no equivalent "leak the descriptor without dropping the
    /// buffer" operation that doesn't double-free.
    #[cfg(feature = "cuda")]
    pub fn into_gpu_tensor(self) -> GpuTensor {
        let t = self.inner;
        // Remove from active_blocks so free_leaked_blocks can detect
        // this as leaked.
        unsafe {
            (*self.alloc).unregister_active_block(t.raw_ptr() as usize);
        }
        std::mem::forget(self);
        t
    }

    /// Reshape this tensor in-place, keeping the same underlying
    /// memory. The caller must ensure the new shape is compatible
    /// with the allocated size.
    pub unsafe fn reshape(&mut self, shape: &[usize], dtype: DType) {
        self.inner = GpuTensor::new(self.inner.raw_ptr(), shape, dtype);
    }

    /// Borrow this tensor as a lifetime-checked `TensorView`. The
    /// returned view borrows `&self`, so the compiler guarantees the
    /// `OwnedTensor` (and its GPU memory) outlives the view.
    pub fn view(&self) -> TensorView<'_> {
        // Safety: the OwnedTensor owns the memory; the view borrows &self.
        unsafe { TensorView::from_raw(self.inner) }
    }

    /// Create a sub-view at a byte offset into this tensor's memory.
    ///
    /// Useful for the packed sampling parameter pattern where one
    /// `OwnedTensor` backs multiple logical tensors at different
    /// offsets.
    ///
    /// # Safety
    /// `byte_offset + numel(new_shape) * dtype.size_bytes()` must
    /// not exceed the allocated size of this tensor.
    pub unsafe fn view_offset(
        &self,
        byte_offset: usize,
        new_shape: &[usize],
        dtype: DType,
    ) -> TensorView<'_> {
        let inner = GpuTensor::new(self.inner.raw_ptr().add(byte_offset), new_shape, dtype);
        TensorView::from_raw(inner)
    }
}

impl Drop for OwnedTensor {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            (*self.alloc).free(self.inner.raw_ptr(), self.size_bytes);
        }
        // Metal: `metal::Buffer`'s own Drop releases the MTLBuffer.
        #[cfg(feature = "metal")]
        let _ = self.size_bytes; // suppress unused-field lint under metal
    }
}

impl std::ops::Deref for OwnedTensor {
    type Target = GpuTensor;
    fn deref(&self) -> &GpuTensor {
        &self.inner
    }
}

impl std::fmt::Debug for OwnedTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OwnedTensor({:?})", self.inner)
    }
}
