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
    unsafe fn alloc_and_copy_host(&mut self, src_host: *const u8, bytes: usize) -> Result<*mut u8>;

    /// Variant of [`alloc_and_copy_host`] that promises the resulting
    /// device pointer will only be bound to kernel arguments whose
    /// scalar binding type requires at most `min_align` bytes of
    /// offset alignment (e.g. 2 for `device const half*` /
    /// `device const bfloat*`, 4 for `device const uint32_t*`).
    ///
    /// Metal's zero-copy mmap-alias path is gated on the offset
    /// being a multiple of `MIN_BIND_ALIGN = 16` (covers u32 packed
    /// int4 weights + simdgroup_float4 / any SIMD-wide reads).
    /// `mlx-community` 4bit safetensors land tensor offsets at
    /// `mod 16 = 2`, so the 16-byte gate rejects every F16/BF16
    /// scales/biases/RMSNorm-gain tensor even though those bindings
    /// only do scalar reads. Threading dtype-aware `min_align` lets
    /// those tensors take zero-copy.
    ///
    /// CUDA has no analogous gate (cudaMalloc returns 256-byte
    /// aligned pointers; the host→device copy is the cost-dominant
    /// step regardless of alignment), so the default forwards to
    /// [`alloc_and_copy_host`] and ignores `min_align`.
    ///
    /// # Safety
    ///
    /// Same as [`alloc_and_copy_host`]. Caller is responsible for
    /// ensuring the returned pointer is only bound to kernels that
    /// read at `min_align` granularity — wider SIMD-vector reads
    /// (e.g. `vec<half, 4>`) require strictly higher alignment.
    ///
    /// [`alloc_and_copy_host`]: Self::alloc_and_copy_host
    unsafe fn alloc_and_copy_host_aligned(
        &mut self,
        src_host: *const u8,
        bytes: usize,
        _min_align: usize,
    ) -> Result<*mut u8> {
        unsafe { self.alloc_and_copy_host(src_host, bytes) }
    }
}
