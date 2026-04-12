// SPDX-License-Identifier: Apache-2.0
//! Barrier / handoff microbenchmarks for the solver cost model.
//!
//! Measures the per-invocation cost of each synchronization mechanism
//! the solver can choose between:
//!   - Grid sync (cooperative_groups::this_grid().sync())
//!   - mbarrier handoff (sm90+ hardware barrier)
//!   - Global memory flag spin (atomicExch + polling)
//!   - __syncthreads() (block-level baseline)
//!
//! Output: CSV to stdout with `mechanism,cost_us` rows.
//!
//! Running:
//!   cargo test -p vllm-tk-test-harness --features cuda \
//!     --test barrier_sweep barrier_sweep -- --ignored --nocapture \
//!     2>/dev/null

#![cfg(feature = "cuda")]

use cudarc::driver::result;
use cudarc::driver::sys;
use vllm_tk_test_harness::ffi;

fn init_cuda() {
    result::init().expect("cuInit failed");
    let device = result::device::get(0).expect("cuDeviceGet failed");
    let ctx = unsafe { result::primary_ctx::retain(device) }.expect("cuCtxRetain failed");
    unsafe { result::ctx::set_current(ctx) }.expect("cuCtxSetCurrent failed");
}

fn gpu_alloc_zeros(bytes: usize) -> u64 {
    unsafe {
        let dptr = result::malloc_sync(bytes).expect("cuMemAlloc failed");
        result::memset_d8_sync(dptr, 0, bytes).expect("cuMemsetD8 failed");
        dptr
    }
}

/// Benchmark a GPU operation: warmup + timed iterations → median µs per iteration.
fn bench_per_iter(
    stream: sys::CUstream,
    warmup: u32,
    iters: u32,
    n_per_launch: u32,
    mut f: impl FnMut(),
) -> f64 {
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
    // Convert to per-iteration µs.
    (elapsed_ms * 1000.0) as f64 / n_per_launch as f64
}

#[test]
#[ignore = "needs GPU"]
fn barrier_sweep() {
    init_cuda();

    let stream: sys::CUstream = std::ptr::null_mut();
    let n_iters: u32 = 1000;

    println!("# Barrier / handoff microbenchmarks");
    println!("# N iterations per measurement: {n_iters}");
    println!("mechanism,cost_us");

    // ── 1. Null kernel (launch overhead baseline) ──
    {
        let us = bench_per_iter(stream, 100, 500, 1, || unsafe {
            ffi::null_kernel_launch(stream as u64);
        });
        println!("null_launch,{us:.3}");
        eprintln!("null_launch: {us:.3} µs");
    }

    // ── 2. __syncthreads() (block barrier baseline) ──
    {
        let block_size = 256u32;
        let us = bench_per_iter(stream, 10, 50, n_iters, || unsafe {
            ffi::syncthreads_bench_launch(n_iters, block_size, stream as u64);
        });
        // Subtract launch overhead.
        let launch_us = bench_per_iter(stream, 100, 500, 1, || unsafe {
            ffi::null_kernel_launch(stream as u64);
        });
        let net = (us - launch_us / n_iters as f64).max(0.0);
        println!("syncthreads_256,{net:.4}");
        eprintln!("syncthreads (bs=256): {net:.4} µs/iter (raw {us:.4})");
    }

    // ── 3. Grid sync (cooperative launch) ──
    // Use a modest grid to keep it representative.
    for &(grid_size, block_size) in &[(132u32, 256u32), (132, 128), (16, 256)] {
        let rc =
            unsafe { ffi::grid_sync_bench_launch(n_iters, grid_size, block_size, stream as u64) };
        if rc != 0 {
            eprintln!("grid_sync ({grid_size}×{block_size}): launch failed (rc={rc})");
            println!("grid_sync_{grid_size}x{block_size},NaN");
            continue;
        }
        unsafe { result::stream::synchronize(stream).unwrap() };

        let us = bench_per_iter(stream, 5, 20, n_iters, || unsafe {
            ffi::grid_sync_bench_launch(n_iters, grid_size, block_size, stream as u64);
        });
        println!("grid_sync_{grid_size}x{block_size},{us:.3}");
        eprintln!("grid_sync ({grid_size}×{block_size}): {us:.3} µs/iter");
    }

    // ── 4. mbarrier handoff (sm90+) ──
    {
        let rc = unsafe { ffi::mbarrier_handoff_bench_launch(1, stream as u64) };
        unsafe { result::stream::synchronize(stream).ok() };
        if rc == -1 {
            eprintln!("mbarrier: not supported on this arch");
            println!("mbarrier,NaN");
        } else if rc != 0 {
            eprintln!("mbarrier: launch failed (rc={rc})");
            println!("mbarrier,NaN");
        } else {
            let us = bench_per_iter(stream, 10, 50, n_iters, || unsafe {
                ffi::mbarrier_handoff_bench_launch(n_iters, stream as u64);
            });
            println!("mbarrier,{us:.4}");
            eprintln!("mbarrier handoff: {us:.4} µs/iter");
        }
    }

    // ── 5. gmem flag spin ──
    {
        let flag_ptr = gpu_alloc_zeros(4);
        let us = bench_per_iter(stream, 10, 50, n_iters, || unsafe {
            result::memset_d8_sync(flag_ptr, 0, 4).unwrap();
            ffi::gmem_flag_bench_launch(n_iters, flag_ptr as u64, stream as u64);
        });
        println!("gmem_flag,{us:.4}");
        eprintln!("gmem flag spin: {us:.4} µs/iter");
        unsafe { sys::cuMemFree_v2(flag_ptr) };
    }

    eprintln!("barrier_sweep: done");
}
