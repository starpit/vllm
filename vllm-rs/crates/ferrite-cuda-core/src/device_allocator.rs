// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral device allocation + host→device transfer.
//!
//! `GpuWeights` does the disk → host pipeline (mmap, dtype cast,
//! name lookup) once for every backend. The platform-specific bit —
//! "give me `N` bytes of device memory and put these host bytes on
//! it" — lives behind this trait.
//!
//! - **CUDA impl** (`cuda_allocator::CudaAllocator`) allocates GPU
//!   memory via `mem_alloc`, queues `memcpy_htod_async` on the
//!   loader's stream, and synchronizes before returning. Allocations
//!   are tracked in a `Vec<RawGpuMem>` for ownership transfer via
//!   `take_allocations`.
//!
//! - **Metal impl** (Apple silicon, unified memory) bump-allocates
//!   inside an arena `MTLBuffer` in `StorageModeShared`. The
//!   "transfer" is a `memcpy(buffer.contents() + offset, src, len)`
//!   — same-RAM copy, no DMA.

use anyhow::Result;

/// The platform-specific bit of weight loading: device-side
/// allocation plus host→device byte transfer.
///
/// `GpuWeights` is generic over `A: DeviceAllocator` so backend-
/// specific accessors (e.g. CUDA's `take_allocations` for the
/// load-then-detach pattern) live on `impl GpuWeights<CudaAllocator>`
/// blocks that already know the concrete allocator type — no `Any`
/// downcasts required at any call site.
///
/// Implementations own the device memory they hand out and free it
/// on drop (or transfer ownership out via a backend-specific method).
pub trait DeviceAllocator: Send + Sync {
    /// Allocate `bytes` of device memory and copy `src_host` (host
    /// memory) into it. Returns the device-visible pointer suitable
    /// for [`GpuTensor::new`](crate::tensor::GpuTensor::new).
    ///
    /// The implementation synchronizes before returning, so the
    /// device sees the bytes by the time the caller receives the
    /// pointer.
    ///
    /// # Safety
    ///
    /// `src_host` must point to `bytes` valid host bytes for the
    /// duration of this call. CUDA implementations issue
    /// `memcpy_htod_async` against `src_host`; passing pageable
    /// memory works but blocks the CPU. Metal implementations
    /// `memcpy` from `src_host`, so any host memory is fine.
    unsafe fn alloc_and_copy_host(
        &mut self,
        src_host: *const u8,
        bytes: usize,
    ) -> Result<*mut u8>;
}
