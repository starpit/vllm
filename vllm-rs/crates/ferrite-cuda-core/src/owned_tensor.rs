// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral RAII tensor wrapper.

#![cfg(any(feature = "cuda", feature = "metal"))]

use crate::dtype::DType;
use crate::tensor::{GpuTensor, TensorView};

#[cfg(feature = "metal")]
use objc2::rc::Retained;
#[cfg(feature = "metal")]
use objc2::runtime::ProtocolObject;
#[cfg(feature = "metal")]
use objc2_metal::MTLBuffer;

#[cfg(feature = "metal")]
pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

pub struct OwnedTensor {
    inner: GpuTensor,
    #[cfg(feature = "cuda")]
    pub(crate) alloc: *mut crate::alloc::CachingAllocator,
    #[cfg(feature = "metal")]
    _buffer: Buffer,
    pub(crate) size_bytes: usize,
}

unsafe impl Send for OwnedTensor {}
unsafe impl Sync for OwnedTensor {}

impl OwnedTensor {
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

    #[cfg(feature = "metal")]
    pub fn from_metal_buffer(inner: GpuTensor, buffer: Buffer, size_bytes: usize) -> Self {
        Self {
            inner,
            _buffer: buffer,
            size_bytes,
        }
    }

    pub fn as_gpu_tensor(&self) -> GpuTensor {
        self.inner
    }

    #[cfg(feature = "metal")]
    pub fn metal_buffer(&self) -> &Buffer {
        &self._buffer
    }

    #[cfg(feature = "cuda")]
    pub fn into_gpu_tensor(self) -> GpuTensor {
        let t = self.inner;
        unsafe {
            (*self.alloc).unregister_active_block(t.raw_ptr() as usize);
        }
        std::mem::forget(self);
        t
    }

    pub unsafe fn reshape(&mut self, shape: &[usize], dtype: DType) {
        self.inner = GpuTensor::new(self.inner.raw_ptr(), shape, dtype);
    }

    pub fn view(&self) -> TensorView<'_> {
        unsafe { TensorView::from_raw(self.inner) }
    }

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
        #[cfg(feature = "metal")]
        let _ = self.size_bytes;
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
