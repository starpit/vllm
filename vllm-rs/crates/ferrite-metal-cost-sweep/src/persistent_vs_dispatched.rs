// SPDX-License-Identifier: Apache-2.0
//! Persistent-kernel vs dispatched-kernel A/B microbench.
//!
//! Settles "Risk B" for the persistent-megakernel-for-decode plan:
//! does avoiding kernel boundaries inside a forward pass save wall-
//! clock vs. running the same number of phases as separate dispatches
//! within one command buffer?
//!
//! Three patterns measure the same N "phases" of trivial work
//! (each phase: every TG writes a small per-TG datum to a device
//! buffer; the next phase reads the previous output cross-TG, sums
//! it, writes again — so cross-phase device-memory traffic is
//! realistic-shape):
//!
//!   A: N separate command buffers, one dispatch each. commit+wait
//!      per buffer. Worst case — represents what we DON'T do today.
//!
//!   B: One command buffer with N dispatches encoded back-to-back.
//!      One commit+wait. Represents today's worker.rs path — one
//!      cmdbuf per step, many dispatches inside.
//!
//!   C: One command buffer with ONE dispatch of a persistent kernel
//!      that runs N phases internally, separated by cross-TG ticket-
//!      lock barriers. The proposed persistent-megakernel path.
//!
//! Per-phase work in all three is identical: each TG reads from a
//! ring buffer indexed by phase, sums HIDDEN elements, writes its
//! per-TG slice of the next ring slot. HIDDEN=2048, matches Llama
//! 3.2-1B decode residual stream.

use crate::util::{self, Buffer, CommandQueue};
use objc2::AnyThread;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};
use std::ptr::NonNull;
use std::time::Instant;

const KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant constexpr uint HIDDEN     = 2048;
// 1024 threads/TG, 16 KB TG memory — matches the resource profile
// of `synthesize_pre_attn_chunk`'s emitted kernel (threadgroup
// bfloat x_norm[HIDDEN] + scratch). Tests whether the persistent-
// kernel win holds at production resource pressure.
constant constexpr uint TG_THREADS = 1024;

// Cross-TG visibility on Apple Metal requires atomic memory ops on
// the data itself (regular stores + threadgroup_barrier(mem_device)
// leaves some TGs reading stale values — verified by
// `synth_persistent_test`). Buffers are uint (atomic-typed), the
// bench bit-casts float ↔ uint to keep the work arithmetic float.
//
// MSL restriction: `threadgroup` storage cannot live in a non-kernel
// helper, so the phase body is inlined into each kernel via a
// preprocessor macro instead of a function.
#define PHASE_BODY(in_buf, out_buf, tg_id, num_tgs, tid, partial)              \
    {                                                                          \
        float local = 0.0f;                                                    \
        for (uint i = (tid); i < HIDDEN; i += TG_THREADS) {                    \
            uint v = atomic_load_explicit(                                     \
                (device atomic_uint*)&(in_buf)[i], memory_order_relaxed);      \
            local += as_type<float>(v);                                        \
        }                                                                      \
        (partial)[(tid)] = local;                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        for (uint stride = TG_THREADS / 2; stride > 0; stride >>= 1) {         \
            if ((tid) < stride) (partial)[(tid)] += (partial)[(tid) + stride]; \
            threadgroup_barrier(mem_flags::mem_threadgroup);                   \
        }                                                                      \
        float sum = (partial)[0];                                              \
        uint per_tg = HIDDEN / (num_tgs);                                      \
        for (uint i = 0; i < per_tg; ++i) {                                    \
            uint idx = (tg_id) * per_tg + i;                                   \
            if (idx < HIDDEN && (tid) == i % TG_THREADS) {                     \
                uint in_v = atomic_load_explicit(                              \
                    (device atomic_uint*)&(in_buf)[idx], memory_order_relaxed);\
                float out_f = sum * 0.001f + as_type<float>(in_v);             \
                atomic_store_explicit(                                         \
                    (device atomic_uint*)&(out_buf)[idx],                      \
                    as_type<uint>(out_f), memory_order_relaxed);               \
            }                                                                  \
        }                                                                      \
    }

// Pattern A & B: one dispatch = one phase. Encoder rotates the ring.
[[kernel, max_total_threads_per_threadgroup(TG_THREADS)]]
void one_phase(
    device const uint* in_buf  [[buffer(0)]],
    device       uint* out_buf [[buffer(1)]],
    constant uint& num_tgs       [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]])
{
    threadgroup float partial[TG_THREADS];
    PHASE_BODY(in_buf, out_buf, tg_id, num_tgs, tid, partial)
}

// Heterogeneous phase B: simulates a "different kernel body". Same
// shape, different arithmetic — point-wise transform instead of
// reduce-and-scale. Forces the GPU to use different instructions
// and different data-flow per phase; checks if the persistent
// kernel's win depends on phase-body homogeneity.
[[kernel, max_total_threads_per_threadgroup(TG_THREADS)]]
void one_phase_alt(
    device const uint* in_buf  [[buffer(0)]],
    device       uint* out_buf [[buffer(1)]],
    constant uint& num_tgs       [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]])
{
    // Point-wise: out[i] = in[i] * cos(in[i]) + 0.5 (rope-shaped work)
    uint per_tg = HIDDEN / num_tgs;
    for (uint i = 0; i < per_tg; ++i) {
        uint idx = tg_id * per_tg + i;
        if (idx < HIDDEN && tid == i % TG_THREADS) {
            uint v = atomic_load_explicit(
                (device atomic_uint*)&in_buf[idx], memory_order_relaxed);
            float x = as_type<float>(v);
            float out_f = x * cos(x) + 0.5f;
            atomic_store_explicit(
                (device atomic_uint*)&out_buf[idx],
                as_type<uint>(out_f), memory_order_relaxed);
        }
    }
}

// Pattern C: persistent kernel — runs `num_phases` phases internally,
// cross-TG barrier between each. Two-slot ring buffer rotated by
// phase parity. Each TG-leader bumps the cross-TG counter and spins
// until all TGs reach the barrier; threadgroup_barrier propagates.
[[kernel, max_total_threads_per_threadgroup(TG_THREADS)]]
void persistent_n_phases(
    device       uint* ring0       [[buffer(0)]],
    device       uint* ring1       [[buffer(1)]],
    device atomic_uint*  counter     [[buffer(2)]],
    constant uint&       num_tgs     [[buffer(3)]],
    constant uint&       num_phases  [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]])
{
    threadgroup float partial[TG_THREADS];
    for (uint p = 0; p < num_phases; ++p) {
        device const uint* in_buf  = (p & 1) ? ring1 : ring0;
        device       uint* out_buf = (p & 1) ? ring0 : ring1;
        PHASE_BODY(in_buf, out_buf, tg_id, num_tgs, tid, partial)

        if (tid == 0) {
            atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
            uint target = num_tgs * (p + 1u);
            while (atomic_load_explicit(counter, memory_order_relaxed) < target) {
                // spin
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// Persistent kernel with HETEROGENEOUS phases: alternates between
// reduce-and-scale (even phases) and point-wise rope-shaped work
// (odd phases). Same cross-TG barrier between every phase. Compares
// against pattern B alternating between `one_phase` and
// `one_phase_alt` dispatches.
[[kernel, max_total_threads_per_threadgroup(TG_THREADS)]]
void persistent_n_phases_het(
    device       uint* ring0       [[buffer(0)]],
    device       uint* ring1       [[buffer(1)]],
    device atomic_uint*  counter     [[buffer(2)]],
    constant uint&       num_tgs     [[buffer(3)]],
    constant uint&       num_phases  [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]])
{
    threadgroup float partial[TG_THREADS];
    for (uint p = 0; p < num_phases; ++p) {
        device const uint* in_buf  = (p & 1) ? ring1 : ring0;
        device       uint* out_buf = (p & 1) ? ring0 : ring1;
        if ((p & 1) == 0) {
            // Even phase: reduce-and-scale
            PHASE_BODY(in_buf, out_buf, tg_id, num_tgs, tid, partial)
        } else {
            // Odd phase: point-wise rope-shaped (matches one_phase_alt)
            uint per_tg = HIDDEN / num_tgs;
            for (uint i = 0; i < per_tg; ++i) {
                uint idx = tg_id * per_tg + i;
                if (idx < HIDDEN && tid == i % TG_THREADS) {
                    uint v = atomic_load_explicit(
                        (device atomic_uint*)&in_buf[idx], memory_order_relaxed);
                    float x = as_type<float>(v);
                    float out_f = x * cos(x) + 0.5f;
                    atomic_store_explicit(
                        (device atomic_uint*)&out_buf[idx],
                        as_type<uint>(out_f), memory_order_relaxed);
                }
            }
        }

        if (tid == 0) {
            atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
            uint target = num_tgs * (p + 1u);
            while (atomic_load_explicit(counter, memory_order_relaxed) < target) {
                // spin
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}
"#;

pub fn run(launch_overhead_us: f64) {
    eprintln!("\n=== Persistent vs dispatched A/B microbench ===");

    let device = util::device();
    let queue = util::new_command_queue();

    let source = NSString::from_str(KERNEL_SOURCE);
    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .expect("library compilation");

    let one_phase_fn = library
        .newFunctionWithName(&NSString::from_str("one_phase"))
        .expect("one_phase lookup");
    let one_phase_pso = device
        .newComputePipelineStateWithFunction_error(&one_phase_fn)
        .expect("one_phase pso");

    let persistent_fn = library
        .newFunctionWithName(&NSString::from_str("persistent_n_phases"))
        .expect("persistent lookup");
    let persistent_pso = device
        .newComputePipelineStateWithFunction_error(&persistent_fn)
        .expect("persistent pso");

    let one_phase_alt_fn = library
        .newFunctionWithName(&NSString::from_str("one_phase_alt"))
        .expect("one_phase_alt lookup");
    let one_phase_alt_pso = device
        .newComputePipelineStateWithFunction_error(&one_phase_alt_fn)
        .expect("one_phase_alt pso");

    let persistent_het_fn = library
        .newFunctionWithName(&NSString::from_str("persistent_n_phases_het"))
        .expect("persistent_het lookup");
    let persistent_het_pso = device
        .newComputePipelineStateWithFunction_error(&persistent_het_fn)
        .expect("persistent_het pso");

    eprintln!(
        "  one_phase max threads/TG = {}, persistent max threads/TG = {}",
        one_phase_pso.maxTotalThreadsPerThreadgroup(),
        persistent_pso.maxTotalThreadsPerThreadgroup()
    );

    // 4 bytes/elem (uint storage to allow atomic_load/atomic_store on
    // cross-TG access; float bits are reinterpreted in the kernel).
    let ring0 = util::create_buffer(2048 * 4);
    let ring1 = util::create_buffer(2048 * 4);
    let counter = util::create_buffer(4);

    // Initialize ring0 with f32(1.0) bit-pattern; everything else
    // is overwritten by the kernels.
    {
        let ptr = ring0.contents().as_ptr() as *mut u32;
        for i in 0..2048 {
            unsafe { *ptr.add(i) = 0x3f800000; } // f32(1.0)
        }
    }

    // 128 TGs — heavier kernel with 4 KB TG-mem per TG won't fit 320
    // concurrently on M4. Empirical: 320 deadlocks the persistent
    // kernel (TG waits on barrier for TGs the scheduler hasn't started).
    // At 1024 threads/TG, M4 fits ~10-20 TGs concurrently. 16 is
    // chosen empirically to avoid persistent-kernel deadlock.
    let num_tgs: u32 = 16;

    println!("\npattern,num_phases,total_us,per_phase_us");
    for &num_phases in &[16u32, 64, 130] {
        eprintln!("  starting num_phases={} ...", num_phases);
        let a_us = bench_pattern_a(&queue, &one_phase_pso, &ring0, &ring1, num_tgs, num_phases, launch_overhead_us);
        eprintln!("    A done = {:.2} µs", a_us);
        let b_us = bench_pattern_b(&queue, &one_phase_pso, &ring0, &ring1, num_tgs, num_phases, launch_overhead_us);
        eprintln!("    B done = {:.2} µs", b_us);
        let c_us = bench_pattern_c(&queue, &persistent_pso, &ring0, &ring1, &counter, num_tgs, num_phases, launch_overhead_us);
        eprintln!("    C done = {:.2} µs", c_us);
        let b_het_us = bench_pattern_b_het(&queue, &one_phase_pso, &one_phase_alt_pso, &ring0, &ring1, num_tgs, num_phases, launch_overhead_us);
        eprintln!("    B_het done = {:.2} µs", b_het_us);
        let c_het_us = bench_pattern_c(&queue, &persistent_het_pso, &ring0, &ring1, &counter, num_tgs, num_phases, launch_overhead_us);
        eprintln!("    C_het done = {:.2} µs", c_het_us);
        println!("B_het,{},{:.2},{:.2}", num_phases, b_het_us, b_het_us / num_phases as f64);
        println!("C_het,{},{:.2},{:.2}", num_phases, c_het_us, c_het_us / num_phases as f64);
        eprintln!(
            "  num_phases={:>3}  A(separate cmdbufs)={:>9.2} µs  B(one cmdbuf, N dispatches)={:>9.2} µs  C(persistent)={:>9.2} µs",
            num_phases, a_us, b_us, c_us
        );
        println!("A,{},{:.2},{:.2}", num_phases, a_us, a_us / num_phases as f64);
        println!("B,{},{:.2},{:.2}", num_phases, b_us, b_us / num_phases as f64);
        println!("C,{},{:.2},{:.2}", num_phases, c_us, c_us / num_phases as f64);
    }
}

fn bench_pattern_a(
    queue: &CommandQueue,
    pso: &objc2::runtime::ProtocolObject<dyn MTLComputePipelineState>,
    ring0: &Buffer,
    ring1: &Buffer,
    num_tgs: u32,
    num_phases: u32,
    launch_overhead_us: f64,
) -> f64 {
    let warmup = 2u32;
    let iters = 5u32;
    let mut total_us = 0.0f64;
    for run_idx in 0..(warmup + iters) {
        let start = Instant::now();
        for p in 0..num_phases {
            let (in_buf, out_buf) = if p & 1 == 0 { (ring0, ring1) } else { (ring1, ring0) };
            let cb = queue.commandBuffer().expect("cb");
            let enc = cb.computeCommandEncoder().expect("enc");
            enc.setComputePipelineState(pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(in_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(out_buf), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new_unchecked(&num_tgs as *const u32 as *mut std::ffi::c_void),
                    4, 2,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: num_tgs as usize, height: 1, depth: 1 },
                MTLSize { width: 1024, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
        }
        let wall_us = start.elapsed().as_secs_f64() * 1e6;
        if run_idx >= warmup {
            total_us += (wall_us - launch_overhead_us * num_phases as f64).max(0.0);
        }
    }
    total_us / iters as f64
}

fn bench_pattern_b(
    queue: &CommandQueue,
    pso: &objc2::runtime::ProtocolObject<dyn MTLComputePipelineState>,
    ring0: &Buffer,
    ring1: &Buffer,
    num_tgs: u32,
    num_phases: u32,
    launch_overhead_us: f64,
) -> f64 {
    let warmup = 2u32;
    let iters = 5u32;
    let mut total_us = 0.0f64;
    for run_idx in 0..(warmup + iters) {
        let cb = queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(pso);
        for p in 0..num_phases {
            let (in_buf, out_buf) = if p & 1 == 0 { (ring0, ring1) } else { (ring1, ring0) };
            unsafe {
                enc.setBuffer_offset_atIndex(Some(in_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(out_buf), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new_unchecked(&num_tgs as *const u32 as *mut std::ffi::c_void),
                    4, 2,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: num_tgs as usize, height: 1, depth: 1 },
                MTLSize { width: 1024, height: 1, depth: 1 },
            );
        }
        enc.endEncoding();

        let start = Instant::now();
        cb.commit();
        cb.waitUntilCompleted();
        let wall_us = start.elapsed().as_secs_f64() * 1e6;
        if run_idx >= warmup {
            total_us += (wall_us - launch_overhead_us).max(0.0);
        }
    }
    total_us / iters as f64
}

/// Pattern B but alternates between `one_phase` and `one_phase_alt`
/// pipelines — matches the heterogeneous-phase persistent kernel.
fn bench_pattern_b_het(
    queue: &CommandQueue,
    pso_even: &objc2::runtime::ProtocolObject<dyn MTLComputePipelineState>,
    pso_odd:  &objc2::runtime::ProtocolObject<dyn MTLComputePipelineState>,
    ring0: &Buffer,
    ring1: &Buffer,
    num_tgs: u32,
    num_phases: u32,
    launch_overhead_us: f64,
) -> f64 {
    let warmup = 2u32;
    let iters = 5u32;
    let mut total_us = 0.0f64;
    for run_idx in 0..(warmup + iters) {
        let cb = queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        for p in 0..num_phases {
            let (in_buf, out_buf) = if p & 1 == 0 { (ring0, ring1) } else { (ring1, ring0) };
            let pso = if p & 1 == 0 { pso_even } else { pso_odd };
            enc.setComputePipelineState(pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(in_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(out_buf), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new_unchecked(&num_tgs as *const u32 as *mut std::ffi::c_void),
                    4, 2,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: num_tgs as usize, height: 1, depth: 1 },
                MTLSize { width: 1024, height: 1, depth: 1 },
            );
        }
        enc.endEncoding();
        let start = Instant::now();
        cb.commit();
        cb.waitUntilCompleted();
        let wall_us = start.elapsed().as_secs_f64() * 1e6;
        if run_idx >= warmup {
            total_us += (wall_us - launch_overhead_us).max(0.0);
        }
    }
    total_us / iters as f64
}

fn bench_pattern_c(
    queue: &CommandQueue,
    pso: &objc2::runtime::ProtocolObject<dyn MTLComputePipelineState>,
    ring0: &Buffer,
    ring1: &Buffer,
    counter: &Buffer,
    num_tgs: u32,
    num_phases: u32,
    launch_overhead_us: f64,
) -> f64 {
    let warmup = 2u32;
    let iters = 5u32;
    let mut total_us = 0.0f64;
    for run_idx in 0..(warmup + iters) {
        // Zero counter
        unsafe { (counter.contents().as_ptr() as *mut u32).write(0); }

        let cb = queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(ring0), 0, 0);
            enc.setBuffer_offset_atIndex(Some(ring1), 0, 1);
            enc.setBuffer_offset_atIndex(Some(counter), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new_unchecked(&num_tgs as *const u32 as *mut std::ffi::c_void),
                4, 3,
            );
            enc.setBytes_length_atIndex(
                NonNull::new_unchecked(&num_phases as *const u32 as *mut std::ffi::c_void),
                4, 4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: num_tgs as usize, height: 1, depth: 1 },
            MTLSize { width: 1024, height: 1, depth: 1 },
        );
        enc.endEncoding();

        let start = Instant::now();
        cb.commit();
        cb.waitUntilCompleted();
        let wall_us = start.elapsed().as_secs_f64() * 1e6;
        if run_idx >= warmup {
            total_us += (wall_us - launch_overhead_us).max(0.0);
        }
    }
    total_us / iters as f64
}
