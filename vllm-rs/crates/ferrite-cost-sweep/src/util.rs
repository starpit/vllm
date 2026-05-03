// SPDX-License-Identifier: Apache-2.0
//! Shared CUDA setup + kernel-timing helpers used by both sweep
//! modules. Raw FFI rather than the ferrite-cuda-core wrappers so
//! measured timings reflect the kernel itself, not cublasLt plan
//! lookups or `OwnedTensor` RAII overhead.

#![cfg(feature = "cuda")]

use cudarc::driver::result;
use cudarc::driver::sys;

/// Initialise the CUDA driver + make the primary context current on
/// device 0. Must be called once at program start before any other
/// CUDA call. Panics on failure — cost-sweep is a calibration tool,
/// a driver failure is not a recoverable error.
pub fn init_cuda() {
    result::init().expect("cuInit failed");
    let device = result::device::get(0).expect("cuDeviceGet failed");
    let ctx = unsafe { result::primary_ctx::retain(device) }.expect("cuCtxRetain failed");
    unsafe { result::ctx::set_current(ctx) }.expect("cuCtxSetCurrent failed");
}

/// Allocate `bytes` on device 0 and zero the range. Returned as the
/// raw CUdeviceptr (a `u64`) to match the extern "C" kernel ABIs;
/// free with `cuMemFree_v2`.
pub fn gpu_alloc_zeros(bytes: usize) -> u64 {
    unsafe {
        let dptr = result::malloc_sync(bytes).expect("cuMemAlloc failed");
        result::memset_d8_sync(dptr, 0, bytes).expect("cuMemsetD8 failed");
        dptr
    }
}

/// `(free_bytes, total_bytes)` on device 0. Used by the sweep to gate
/// the SplitK workspace alloc — at vocab-N M=4096 shapes the sk=8
/// workspace is ~20 GB, which fits L40s/H100 but blows past L4's
/// 24 GB once A / B / C / c_up are live. Querying actual free VRAM
/// makes the gate hardware-aware instead of hard-coded for one GPU.
pub fn gpu_mem_info() -> (usize, usize) {
    let mut free: usize = 0;
    let mut total: usize = 0;
    let rc = unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) };
    assert_eq!(
        rc,
        sys::CUresult::CUDA_SUCCESS,
        "cuMemGetInfo_v2 failed: {rc:?}"
    );
    (free, total)
}

/// Allocate `bytes` on device 0 and fill every byte with `pattern`. Used by
/// the attention sweep to prime Q/K/V buffers with a small-positive bf16
/// pattern (`0x3c`) so FA2 and FlashInfer see identical (non-pathological)
/// numeric inputs — matches the prior-session `attn_bench.cu` setup.
pub fn gpu_alloc_fill(bytes: usize, pattern: u8) -> u64 {
    unsafe {
        let dptr = result::malloc_sync(bytes).expect("cuMemAlloc failed");
        result::memset_d8_sync(dptr, pattern, bytes).expect("cuMemsetD8 failed");
        dptr
    }
}

/// Synchronous host-to-device copy. Used to upload the small index buffers
/// (`cu_seqlens_q`, `block_table`, `kv_indices`) that the attention kernels
/// read — too small to be worth an async path.
///
/// # Safety
/// `dptr` must be a valid device allocation of at least `bytes` bytes, and
/// `src` must point to at least `bytes` bytes of initialized host memory.
pub unsafe fn h2d_copy(dptr: u64, src: *const u8, bytes: usize) {
    let rc = unsafe { sys::cuMemcpyHtoD_v2(dptr, src as *const _, bytes) };
    assert_eq!(
        rc,
        sys::CUresult::CUDA_SUCCESS,
        "cuMemcpyHtoD failed: {rc:?}"
    );
}

/// Query the streaming-multiprocessor count of device 0. FlashInfer's
/// persistent scheduler needs this both for workspace sizing
/// (`workspace_bytes`) and as the `target_num_clusters` parameter to
/// `fi_plan_new` — see FLASHINFER_HANDOFF.md on the forked
/// `TwoStageHolisticPlanWithNumSm`.
pub fn query_num_sm() -> i32 {
    let device = result::device::get(0).expect("cuDeviceGet failed");
    unsafe { ferrite_cuda_core::driver::device_get_num_sm(device) }
        .expect("cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) failed")
}

/// Sentinel cost emitted when a kernel is detected as broken — matches
/// `UNCALIBRATED_COST_US` in the proc-macro, so the DP reads these
/// rows as "uncalibrated → skip" instead of "instant → always pick".
///
/// Triggered on three signals:
/// 1. The probe call before timing returns rc ≠ 0 (e.g., CUTLASS's
///    `can_implement` rejects the shape — most often SMEM > 99KB on
///    sm89).
/// 2. The measured per-launch cost is below `BROKEN_KERNEL_THRESHOLD_US`
///    after subtracting launch overhead — kernel "ran" too fast to
///    have done real work.
/// 3. The probe call already raised a CUDA driver error (caught by
///    the call site's existing `cudaGetLastError` discipline).
pub const BROKEN_KERNEL_SENTINEL_US: f64 = 1.0e9;

/// Anything below this is treated as "kernel did nothing". Real GEMM
/// kernels at any shape we care about are >> 1µs; sub-µs measurements
/// are silent failures (SMEM exhaustion, can_implement reject, etc.).
pub const BROKEN_KERNEL_THRESHOLD_US: f64 = 0.5;

/// Benchmark a closure that launches a single kernel:
/// warmup + timed iterations via `cuEventElapsedTime`. Returns mean
/// µs per launch. Caller subtracts `launch_overhead_us` to report
/// compute-only cost.
pub fn bench_kernel(stream: sys::CUstream, warmup: u32, iters: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    unsafe { result::stream::synchronize(stream).unwrap() };

    let elapsed_ms = unsafe {
        let mut start: sys::CUevent = std::ptr::null_mut();
        let mut stop: sys::CUevent = std::ptr::null_mut();
        sys::cuEventCreate(&mut start, 0);
        sys::cuEventCreate(&mut stop, 0);
        sys::cuEventRecord(start, stream);
        for _ in 0..iters {
            f();
        }
        sys::cuEventRecord(stop, stream);
        sys::cuEventSynchronize(stop);
        let mut ms: f32 = 0.0;
        sys::cuEventElapsedTime(&mut ms, start, stop);
        sys::cuEventDestroy_v2(start);
        sys::cuEventDestroy_v2(stop);
        ms / iters as f32
    };
    (elapsed_ms * 1000.0) as f64
}

/// Time a do-nothing kernel to establish the launch-overhead
/// baseline that all subsequent measurements subtract.
pub fn measure_launch_overhead() -> f64 {
    let stream: sys::CUstream = std::ptr::null_mut();
    // Extra warmup — the first null launches also warm JIT caches
    // and page the kernel image in.
    for _ in 0..100 {
        unsafe { null_kernel_launch(stream as u64) };
    }
    unsafe { result::stream::synchronize(stream).unwrap() };
    bench_kernel(stream, 100, 500, || unsafe {
        null_kernel_launch(stream as u64)
    })
}

// The no-op kernel lives in `vllm-cuda/csrc/cutlass_standalone_gemm.cu`.
// It launches a grid that does nothing — a cheap way to measure
// per-launch overhead in isolation.
#[cfg(feature = "cuda")]
unsafe extern "C" {
    pub fn null_kernel_launch(stream: u64);
}
