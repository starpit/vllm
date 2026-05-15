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
        Self { buffer, base, size }
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
