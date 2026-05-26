// SPDX-License-Identifier: Apache-2.0
//! Cross-threadgroup synchronization cost microbenchmark — the PD-clean
//! Stage-2 go/no-go.
//!
//! A persistent multi-TG decode megakernel beats the existing split-K
//! per-op path only if cross-TG synchronization is cheaper than the
//! dispatch boundaries it replaces. The failed `_mt` used a GLOBAL
//! arrival barrier (all-N-TG rendezvous) and measured ~parity-or-worse.
//! The wavefront/systolic design instead needs POINT-TO-POINT
//! producer→consumer flags. This measures both on this chip so we know
//! the real per-edge cost BEFORE building any scheduler / subtiling.
//!
//! Prints a human-readable report to stderr (not a CSV row). Each kernel
//! runs `ITERS` hand-offs internally; we time the command buffer and
//! divide, so the per-hop cost is isolated from launch overhead. Every
//! spin loop is capped so a scheduling stall can't hang the GPU, and we
//! verify the final flag value to confirm the hand-offs actually
//! completed (a capped/deadlocked run is reported as INVALID).

use crate::util;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

type Pipeline = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLComputePipelineState>>;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint SPIN_CAP = 200000000u;  // hard cap: never hang the GPU

// 2-TG ping-pong: TG0 and TG1 hand a token back and forth `iters` times.
// One iter = 2 cross-TG hops (0->1, 1->0); the waiter ALWAYS spins, so
// this is the round-trip LATENCY — a conservative upper bound on the
// cost of a point-to-point flag (the balanced/no-spin case is cheaper).
kernel void pingpong(
    device atomic_uint* flag [[buffer(0)]],
    constant uint& iters     [[buffer(1)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (tid != 0u) return;
    if (tg == 0u) {
        for (uint i = 0u; i < iters; i++) {
            uint want = 2u*i, s = 0u;
            while (atomic_load_explicit(flag, memory_order_relaxed) != want) { if (++s > SPIN_CAP) return; }
            atomic_store_explicit(flag, 2u*i + 1u, memory_order_relaxed);
        }
    } else if (tg == 1u) {
        for (uint i = 0u; i < iters; i++) {
            uint want = 2u*i + 1u, s = 0u;
            while (atomic_load_explicit(flag, memory_order_relaxed) != want) { if (++s > SPIN_CAP) return; }
            atomic_store_explicit(flag, 2u*i + 2u, memory_order_relaxed);
        }
    }
}

// N-TG wavefront ring: a token sweeps 0->1->...->(N-1)->0, `iters` laps.
// Each hop is a point-to-point wait on the predecessor only — models the
// wavefront hand-off. Total hops = iters * num_tg.
kernel void wavefront_ring(
    device atomic_uint* flag [[buffer(0)]],
    constant uint& iters     [[buffer(1)]],
    constant uint& num_tg    [[buffer(2)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (tid != 0u) return;
    for (uint i = 0u; i < iters; i++) {
        uint want = i * num_tg + tg, s = 0u;
        while (atomic_load_explicit(flag, memory_order_relaxed) != want) { if (++s > SPIN_CAP) return; }
        atomic_store_explicit(flag, want + 1u, memory_order_relaxed);
    }
}

// N-TG GLOBAL arrival barrier, `iters` times — the `_mt` mechanism we
// want to beat. bar[0]=arrival count, bar[1]=generation.
kernel void global_barrier_bench(
    device atomic_uint* bar [[buffer(0)]],
    constant uint& iters    [[buffer(1)]],
    constant uint& num_tg   [[buffer(2)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (tid != 0u) return;
    for (uint i = 0u; i < iters; i++) {
        uint g = atomic_load_explicit(&bar[1], memory_order_relaxed);
        if (atomic_fetch_add_explicit(&bar[0], 1u, memory_order_relaxed) == num_tg - 1u) {
            atomic_store_explicit(&bar[0], 0u, memory_order_relaxed);
            atomic_store_explicit(&bar[1], g + 1u, memory_order_relaxed);
        } else {
            uint s = 0u;
            while (atomic_load_explicit(&bar[1], memory_order_relaxed) == g) { if (++s > SPIN_CAP) return; }
        }
    }
}
"#;

fn u32_bytes(enc: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>, v: u32, idx: usize) {
    let p = NonNull::new(&v as *const u32 as *mut c_void).unwrap();
    unsafe { enc.setBytes_length_atIndex(p, 4, idx) };
}

pub fn run(launch_overhead_us: f64) {
    eprintln!("\n=== flag_sync microbenchmark (PD-clean Stage-2 go/no-go) ===");
    let device = util::device();
    let opts = objc2_metal::MTLCompileOptions::new();
    let lib: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLLibrary>> = device
        .newLibraryWithSource_options_error(&NSString::from_str(SRC), Some(&opts))
        .expect("compile flag_sync MSL");
    let pipe = |name: &str| -> Pipeline {
        let f = lib
            .newFunctionWithName(&NSString::from_str(name))
            .expect("function");
        device
            .newComputePipelineStateWithFunction_error(&f)
            .expect("pipeline")
    };
    let pingpong = pipe("pingpong");
    let ring = pipe("wavefront_ring");
    let gbar = pipe("global_barrier_bench");

    let queue = util::new_command_queue();
    let state = util::create_buffer(8); // 2× atomic_uint (flag/bar + gen/pad)
    let iters: u32 = 20_000;

    // Dispatch `pipe` with `num_tg` TGs of 1 thread, `iters` internal
    // hand-offs, after zeroing `state`. Returns per-CB µs (launch
    // subtracted) and the final state[0] (for the completion check).
    let bench = |p: &Pipeline, num_tg: u32, pass_num_tg: bool| -> (f64, u32, u32) {
        let run_one = || {
            util::zero_buffer(&state);
            let cb = queue.commandBuffer().expect("cmd buf");
            let enc = cb.computeCommandEncoder().expect("encoder");
            enc.setComputePipelineState(p);
            unsafe { enc.setBuffer_offset_atIndex(Some(&state), 0, 0) };
            u32_bytes(&enc, iters, 1);
            if pass_num_tg {
                u32_bytes(&enc, num_tg, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: num_tg as usize, height: 1, depth: 1 },
                MTLSize { width: 1, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
        };
        let us = util::time_kernel(launch_overhead_us, 2, 8, run_one);
        // One more (untimed) run to read the completion state. Word 0 is
        // the ping-pong/ring flag; word 1 is the barrier generation.
        util::zero_buffer(&state);
        run_one_capture(&queue, p, &state, iters, num_tg, pass_num_tg);
        let w0 = unsafe { *(state.contents().as_ptr() as *const u32) };
        let w1 = unsafe { *(state.contents().as_ptr() as *const u32).add(1) };
        (us, w0, w1)
    };

    // ── ping-pong: point-to-point round-trip latency (2 TGs) ──────────
    let (pp_us, pp_final, _) = bench(&pingpong, 2, false);
    let pp_ok = pp_final == 2 * iters;
    let pp_per_hop = pp_us / (2.0 * iters as f64); // 2 hops/iter
    eprintln!(
        "ping-pong (2 TG): {:.3} ms/CB → {:.4} µs/hop  [round-trip latency]  {}",
        pp_us / 1000.0,
        pp_per_hop,
        if pp_ok { "OK" } else { "*** INVALID (spin-capped / not co-resident) ***" },
    );

    // ── wavefront ring: point-to-point in an N-TG chain ───────────────
    for &n in &[4u32, 10u32] {
        let (r_us, r_final, _) = bench(&ring, n, true);
        let r_ok = r_final == iters * n;
        let r_per_hop = r_us / (iters as f64 * n as f64);
        eprintln!(
            "wavefront ring ({} TG): {:.3} ms/CB → {:.4} µs/hop  {}",
            n,
            r_us / 1000.0,
            r_per_hop,
            if r_ok { "OK" } else { "*** INVALID ***" },
        );
    }

    // ── global barrier: the `_mt` mechanism (per-barrier cost) ────────
    for &n in &[4u32, 10u32] {
        let (b_us, _, b_gen) = bench(&gbar, n, true);
        let b_ok = b_gen == iters; // generation == iters when complete
        let b_per = b_us / iters as f64;
        eprintln!(
            "global barrier ({} TG): {:.3} ms/CB → {:.4} µs/barrier  {}",
            n,
            b_us / 1000.0,
            b_per,
            if b_ok { "OK" } else { "*** INVALID ***" },
        );
    }

    eprintln!(
        "\nVERDICT: point-to-point is viable iff µs/hop ≪ the ~17 µs/launch dispatch\n\
         boundary it would replace AND ≪ the global-barrier µs/barrier above.\n\
         If µs/hop is sub-µs, the wavefront megakernel can recover the ~1.5 ms\n\
         of per-op dispatch overhead. If it's µs-scale, it cannot beat split-K.\n",
    );
}

#[allow(clippy::too_many_arguments)]
fn run_one_capture(
    queue: &util::CommandQueue,
    p: &Pipeline,
    state: &util::Buffer,
    iters: u32,
    num_tg: u32,
    pass_num_tg: bool,
) {
    let cb = queue.commandBuffer().expect("cmd buf");
    let enc = cb.computeCommandEncoder().expect("encoder");
    enc.setComputePipelineState(p);
    unsafe { enc.setBuffer_offset_atIndex(Some(state), 0, 0) };
    u32_bytes(&enc, iters, 1);
    if pass_num_tg {
        u32_bytes(&enc, num_tg, 2);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: num_tg as usize, height: 1, depth: 1 },
        MTLSize { width: 1, height: 1, depth: 1 },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
}
