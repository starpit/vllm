// SPDX-License-Identifier: Apache-2.0
//! Cross-TG-barrier microbenchmark.
//!
//! Measures the wall-clock cost of an inter-threadgroup barrier on
//! Apple Silicon. A persistent-megakernel design for ferrite-metal
//! decode would replace per-phase kernel boundaries with cross-TG
//! barriers (device-atomic ticket lock); the feasibility of the whole
//! approach hinges on whether one such barrier costs ~µs or ~tens of
//! µs.
//!
//! Methodology:
//!   1. Dispatch a kernel that, in each TG, performs N back-to-back
//!      barriers separated by trivial work (one device atomic add per
//!      barrier loop iteration to prevent the compiler eliding work).
//!   2. Vary `(num_tgs, num_barriers)`; the per-barrier cost is
//!      `(wall_us - launch_overhead) / num_barriers`.
//!   3. Report a table sweeping `num_tgs ∈ {32, 64, 128, 256, 320}`
//!      and `num_barriers ∈ {64, 256, 1024}`.
//!
//! Barrier primitive: ticket-lock on a device atomic counter. TG-leader
//! threads (`tid == 0`) increment the counter and spin until it reaches
//! `num_tgs * (phase + 1)`. After the spin, a `threadgroup_barrier`
//! propagates the device-memory release-acquire to non-leader threads.
//!
//! Caveats:
//! - If `num_tgs` exceeds the chip's max concurrent-resident TGs, the
//!   barrier deadlocks (a TG waiting for one the scheduler hasn't
//!   started yet). The sweep caps at 320 for M4 — M4 has ~10 cores ×
//!   ~32 resident TGs/core.
//! - We assume Apple's GPU schedules all dispatched TGs concurrently
//!   when the grid fits. If not, the spin loop will hang and the
//!   command buffer will time out.

use crate::util::{self, Buffer, CommandQueue, Device};
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

// Cross-TG ticket-lock barrier. `counter` is a device atomic_uint
// that each TG-leader bumps once per barrier; leaders spin until the
// counter reaches `num_tgs * (phase + 1)`. `threadgroup_barrier`
// after the leader's spin propagates the device-memory acquire to
// all threads in the TG.
[[kernel]] void cross_tg_barrier_bench(
    device atomic_uint*    counter      [[buffer(0)]],
    device atomic_uint*    work_sink    [[buffer(1)]],
    constant uint&         num_tgs      [[buffer(2)]],
    constant uint&         num_barriers [[buffer(3)]],
    uint  tg_id [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]])
{
    for (uint phase = 0; phase < num_barriers; ++phase) {
        // Trivial work to prevent the compiler from fusing all barriers
        // into one: each thread bumps a dummy device atomic. The op
        // is cheap relative to barrier cost — atomic_add to a single
        // device slot from 32+ threads serializes through L2.
        if (tid == 0) {
            atomic_fetch_add_explicit(work_sink, 1u, memory_order_relaxed);
        }

        // Barrier: TG-leader bumps the cross-TG counter, then spins
        // until the counter reaches `num_tgs * (phase + 1)`.
        if (tid == 0) {
            atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
            uint target = num_tgs * (phase + 1u);
            while (atomic_load_explicit(counter, memory_order_relaxed) < target) {
                // spin
            }
        }
        // Propagate the device-memory acquire to non-leader threads
        // in this TG. Without this, non-leader threads might proceed
        // before the leader observes the barrier release.
        threadgroup_barrier(mem_flags::mem_device);
    }
}
"#;

pub fn run(launch_overhead_us: f64) {
    eprintln!("\n=== Cross-TG barrier microbench ===");
    eprintln!(
        "kernel: device-atomic ticket-lock + threadgroup_barrier(mem_device)"
    );

    let device = util::device();
    let queue = util::new_command_queue();

    let library = build_library(device);
    let function = library
        .newFunctionWithName(&NSString::from_str("cross_tg_barrier_bench"))
        .expect("function lookup");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("pipeline creation");

    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    let threads_per_tg: usize = max_threads.min(32);
    eprintln!(
        "  pipeline.maxTotalThreadsPerThreadgroup = {}, using {} threads/TG",
        max_threads, threads_per_tg
    );

    let counter = util::create_buffer(4);
    let work_sink = util::create_buffer(4);

    println!("\nnum_tgs,num_barriers,total_us,per_barrier_us");
    for &num_tgs in &[32u32, 64, 128, 256, 320] {
        for &num_barriers in &[64u32, 256, 1024] {
            let per_barrier_us =
                bench_one(&queue, &pipeline, &counter, &work_sink, num_tgs, num_barriers, threads_per_tg, launch_overhead_us);
            eprintln!(
                "  num_tgs={:>4} num_barriers={:>5} per_barrier={:>7.3} µs",
                num_tgs, num_barriers, per_barrier_us
            );
            println!(
                "{},{},{:.3},{:.3}",
                num_tgs,
                num_barriers,
                per_barrier_us * num_barriers as f64,
                per_barrier_us
            );
        }
    }
}

fn build_library(device: &Device) -> objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLLibrary>> {
    let source = NSString::from_str(KERNEL_SOURCE);
    let options = objc2_metal::MTLCompileOptions::new();
    unsafe {
        device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .expect("library compilation")
    }
}

fn bench_one(
    queue: &CommandQueue,
    pipeline: &objc2::runtime::ProtocolObject<dyn MTLComputePipelineState>,
    counter: &Buffer,
    work_sink: &Buffer,
    num_tgs: u32,
    num_barriers: u32,
    threads_per_tg: usize,
    launch_overhead_us: f64,
) -> f64 {
    let warmup = 2u32;
    let iters = 10u32;
    let mut total_us = 0.0f64;

    for run_idx in 0..(warmup + iters) {
        // Zero the cross-TG counter before each run (work_sink is fine to
        // keep accumulating).
        zero_u32(counter);

        let cb = queue.commandBuffer().expect("commandBuffer");
        let enc = cb.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(counter), 0, 0);
            enc.setBuffer_offset_atIndex(Some(work_sink), 0, 1);
            let nt = num_tgs;
            let nb = num_barriers;
            enc.setBytes_length_atIndex(
                NonNull::new_unchecked(&nt as *const u32 as *mut std::ffi::c_void),
                4,
                2,
            );
            enc.setBytes_length_atIndex(
                NonNull::new_unchecked(&nb as *const u32 as *mut std::ffi::c_void),
                4,
                3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: num_tgs as usize, height: 1, depth: 1 },
            MTLSize { width: threads_per_tg, height: 1, depth: 1 },
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
    let avg_total_us = total_us / iters as f64;
    avg_total_us / num_barriers as f64
}

fn zero_u32(buf: &Buffer) {
    let ptr = buf.contents().as_ptr() as *mut u32;
    unsafe { ptr.write(0) };
}
