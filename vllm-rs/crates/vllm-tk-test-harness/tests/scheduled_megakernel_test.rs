// SPDX-License-Identifier: Apache-2.0
//! Phase 3b — end-to-end validation of the scheduled megakernel dispatch loop.
//!
//! Builds the same tiny reified DAG that build.rs baked into the .cu file,
//! launches the placeholder kernel on GPU, and asserts:
//!   1. Every node executed exactly once (its tick slot is nonzero).
//!   2. Topological order: every node's tick > all its deps' ticks.
//!
//! The placeholder tile kernels do no real compute — this test only proves
//! the dispatch loop, the gmem flag spin-wait, and the per-CTA work stream
//! are wired correctly. Real phase implementations come in Phase 3c+.

#![cfg(feature = "cuda")]

use cudarc::driver::result;
use vllm_tk_macros_core::{
    SCHEDULED_PREFILL_TINY_CTAS, reified_dag::ReifiedDag, reified_dag::TileSizes,
    scheduled_prefill_tiny_dims,
};
use vllm_tk_test_harness::ffi;

fn init_cuda() {
    result::init().expect("cuInit failed");
    let device = result::device::get(0).expect("cuDeviceGet failed");
    let ctx = unsafe { result::primary_ctx::retain(device) }.expect("cuCtxRetain failed");
    unsafe { result::ctx::set_current(ctx) }.expect("cuCtxSetCurrent failed");
}

fn gpu_alloc_zeros_u32(count: usize) -> *mut u32 {
    unsafe {
        let bytes = count * 4;
        let dptr = result::malloc_sync(bytes).expect("cuMemAlloc failed");
        result::memset_d8_sync(dptr, 0, bytes).expect("cuMemsetD8 failed");
        dptr as *mut u32
    }
}

fn gpu_read_u32(ptr: *const u32, count: usize) -> Vec<u32> {
    let mut host = vec![0u32; count];
    unsafe {
        result::memcpy_dtoh_sync(&mut host, ptr as cudarc::driver::sys::CUdeviceptr)
            .expect("cuMemcpyDtoH failed");
    }
    host
}

#[test]
#[ignore = "needs GPU"]
fn scheduled_megakernel_executes_topologically() {
    init_cuda();

    // Rebuild the same DAG the kernel was generated against.
    let dims = scheduled_prefill_tiny_dims();
    let dag = ReifiedDag::reify_llama(dims, TileSizes::default_v1());
    let n = dag.nodes.len();

    // Sanity: kernel and host agree on node count.
    let kernel_n = unsafe { ffi::scheduled_megakernel_num_nodes() } as usize;
    assert_eq!(
        n, kernel_n,
        "host and kernel disagree on NUM_NODES ({n} vs {kernel_n}) — build.rs and test diverged"
    );
    let kernel_ctas = unsafe { ffi::scheduled_megakernel_num_ctas() };
    assert_eq!(kernel_ctas, SCHEDULED_PREFILL_TINY_CTAS);

    // Allocate flag array and tick counter on GPU.
    let flags = gpu_alloc_zeros_u32(n);
    let tick = gpu_alloc_zeros_u32(1);

    // Launch.
    unsafe {
        ffi::launch_scheduled_megakernel(flags, tick, std::ptr::null_mut());
        result::stream::synchronize(std::ptr::null_mut()).expect("stream sync failed");
    }

    // Read back the tick array.
    let ticks = gpu_read_u32(flags, n);
    let final_tick = gpu_read_u32(tick, 1)[0];
    eprintln!("scheduled_megakernel: {n} nodes, final tick = {final_tick}");

    // ── Invariant 1: every node executed exactly once. ──
    let unexecuted: Vec<usize> = ticks
        .iter()
        .enumerate()
        .filter_map(|(i, &t)| if t == 0 { Some(i) } else { None })
        .collect();
    assert!(
        unexecuted.is_empty(),
        "{} nodes did not execute (first 10: {:?})",
        unexecuted.len(),
        &unexecuted[..unexecuted.len().min(10)]
    );

    // Tick counter should equal node count (every node atomicAdd'd once).
    assert_eq!(
        final_tick as usize, n,
        "tick counter {final_tick} != node count {n}"
    );

    // ── Invariant 2: topological order. ──
    // Every node's tick must be strictly greater than every dep's tick.
    for nd in &dag.nodes {
        let my_tick = ticks[nd.id.0 as usize];
        for d in &nd.deps {
            let dep_tick = ticks[d.0 as usize];
            assert!(
                my_tick > dep_tick,
                "topological violation: node {} (tick={}) ran before dep {} (tick={})",
                nd.id.0,
                my_tick,
                d.0,
                dep_tick
            );
        }
    }
}
