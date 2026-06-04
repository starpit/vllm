// SPDX-License-Identifier: Apache-2.0
//! `GpuDevice` (metal arm).

#![cfg(feature = "metal")]

use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice};

use crate::MetalAllocator;

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

/// Metal-side `GpuDevice`. Mirrors the cuda struct's role —
/// "everything kernel launches need" — at the Apple-silicon
/// equivalents (Device + CommandQueue + MetalAllocator).
pub struct GpuDevice {
    pub device: Arc<Device>,
    pub queue: CommandQueue,
    pub allocator: Arc<MetalAllocator>,
}

impl GpuDevice {
    pub fn new(device: Arc<Device>, allocator: Arc<MetalAllocator>) -> Self {
        let queue = device
            .newCommandQueue()
            .expect("newCommandQueue returned nil");
        Self {
            device,
            queue,
            allocator,
        }
    }

    /// Allocate device memory and upload host data into it — the
    /// backend-neutral analogue of the cuda `GpuDevice::alloc_gpu_tensor_from_host`.
    /// The vision tower uses this to upload its host-built inputs (pixels,
    /// `cu_seqlens`, the rope cos/sin tables, position ids). On metal the
    /// `MetalAllocator` arena is `StorageModeShared`, so the returned
    /// pointer is host-writable and the copy is a plain memcpy (no separate
    /// staging / stream). The allocator owns the arena, so the tensor's
    /// pointer stays valid for the device's lifetime.
    pub fn alloc_gpu_tensor_from_host(
        &mut self,
        shape: &[usize],
        dtype: crate::dtype::DType,
        data: &[u8],
    ) -> crate::tensor::GpuTensor {
        let numel: usize = shape.iter().product();
        let bytes = numel * dtype.size_bytes();
        debug_assert_eq!(
            data.len(),
            bytes,
            "host data length ({}) != tensor size ({})",
            data.len(),
            bytes
        );
        let ptr = self
            .allocator
            .alloc_uninit(bytes)
            .expect("MetalAllocator::alloc_uninit (vision upload)");
        // Safety: `ptr` is a fresh `bytes`-sized arena allocation (Shared
        // storage, host-writable); `data.len() == bytes` (asserted); the
        // arena outlives the returned tensor (owned by the allocator).
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, bytes);
            crate::tensor::GpuTensor::new(ptr, shape, dtype)
        }
    }
}
