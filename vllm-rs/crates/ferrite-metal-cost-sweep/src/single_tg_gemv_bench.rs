// SPDX-License-Identifier: Apache-2.0
//! Single-TG GEMV microbench.
//!
//! The persistent-megakernel design for decode hinges on running
//! o_proj and down_proj inside ONE TG so their HIDDEN-wide output
//! stays TG-resident for the downstream AddRmsNorm. Production
//! qmv_fast launches with ceil(N/8) TGs (256 TGs at N=2048) so the
//! GPU has lots of concurrent work; a single TG only has 1024 threads
//! and can do only ~1 simdgroup-instruction at a time per cycle.
//!
//! This bench answers: how slow is a single-TG GEMV at the decode
//! shape compared to qmv_fast (whose CSV cost at M=1, N=2048, K=2048
//! is ~384 µs)?
//!
//! Method: bf16 dense GEMV (skip int4 dequant for the initial cut —
//! dense isolates the "single TG, K-long dot products" cost; int4
//! adds ~10-30% on top for dequant arithmetic, won't change the
//! qualitative answer).
//!
//! Kernel: 1024 threads per TG. Each thread computes (N / 1024)
//! output elements. X is loaded into TG memory once at the start.
//!
//! Correctness: compares a few output elements against a CPU dot-
//! product. If those match, full output is presumed correct.

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
using namespace metal;

// Single-TG bf16 dense GEMV:
//   y[n] = sum_k x[k] * W[n,k]
// W is row-major (N rows × K cols). Each thread computes
// (N / TG_THREADS) output elements, all using the same x loaded
// once into threadgroup memory.
constant constexpr uint K_DIM = 2048;
constant constexpr uint N_DIM = 2048;
constant constexpr uint TG_THREADS = 1024;
constant constexpr uint OUTPUTS_PER_THREAD = N_DIM / TG_THREADS;

[[kernel, max_total_threads_per_threadgroup(TG_THREADS)]]
void single_tg_gemv_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* w [[buffer(1)]],
    device       bfloat* y [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup bfloat x_smem[K_DIM];

    // Cooperative load of x into TG memory. K_DIM = 2 * TG_THREADS,
    // so each thread loads 2 contiguous bf16 elements.
    for (uint i = tid; i < K_DIM; i += TG_THREADS) {
        x_smem[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread computes OUTPUTS_PER_THREAD output elements.
    for (uint o = 0; o < OUTPUTS_PER_THREAD; ++o) {
        uint n = tid * OUTPUTS_PER_THREAD + o;
        float acc = 0.0f;
        // Stride W by N_DIM-element rows. Direct read from device —
        // each thread reads its own row, no overlap with other
        // threads (the GPU's L2 will catch some adjacency in practice
        // since adjacent threads read adjacent rows).
        device const bfloat* w_row = w + n * K_DIM;
        for (uint k = 0; k < K_DIM; ++k) {
            acc += float(x_smem[k]) * float(w_row[k]);
        }
        y[n] = bfloat(acc);
    }
}
"#;

pub fn run(launch_overhead_us: f64) {
    eprintln!("\n=== Single-TG bf16 dense GEMV microbench ===");
    eprintln!("shape: M=1, N=2048, K=2048, dtype=bf16");

    let device = util::device();
    let queue = util::new_command_queue();

    let source = NSString::from_str(KERNEL_SOURCE);
    let options = objc2_metal::MTLCompileOptions::new();
    let library = unsafe {
        device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .expect("library compilation")
    };
    let function = library
        .newFunctionWithName(&NSString::from_str("single_tg_gemv_bf16"))
        .expect("function lookup");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("pipeline creation");

    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    eprintln!(
        "  pipeline.maxTotalThreadsPerThreadgroup = {} (target: 1024)",
        max_threads
    );
    if max_threads < 1024 {
        eprintln!(
            "  WARNING: pipeline supports only {} threads/TG; bench expects 1024. \
             Likely register-pressure-limited; results will be misleading.",
            max_threads
        );
    }

    // K=2048, N=2048, bf16=2 bytes
    let x_buf = util::create_buffer(2048 * 2);
    let w_buf = util::create_buffer(2048 * 2048 * 2);
    let y_buf = util::create_buffer(2048 * 2);

    // Init x and W with deterministic pattern so we can spot-check
    // correctness against a CPU reference. Use small magnitudes
    // (≤ 1.0) to avoid bf16 overflow in the accumulation.
    fill_pattern_bf16(&x_buf, 2048, 0);
    fill_pattern_bf16(&w_buf, 2048 * 2048, 1);

    // Correctness spot-check: compute y[0] on CPU, compare after one
    // GPU run.
    let cpu_y0 = cpu_dot_bf16(&x_buf, &w_buf, 0, 2048);

    let warmup = 5u32;
    let iters = 50u32;
    let mut total_us = 0.0f64;

    for run_idx in 0..(warmup + iters) {
        let cb = queue.commandBuffer().expect("commandBuffer");
        let enc = cb.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&x_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&w_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&y_buf), 0, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 1, height: 1, depth: 1 },
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
    let avg_us = total_us / iters as f64;
    eprintln!("  single-TG GEMV: {:.2} µs/call (avg of {} iters)", avg_us, iters);

    // Spot-check correctness.
    let gpu_y0 = read_bf16(&y_buf, 0);
    let rel_err = ((cpu_y0 - gpu_y0).abs() / cpu_y0.abs().max(1e-6)) as f64;
    eprintln!(
        "  correctness y[0]: cpu={:.4} gpu={:.4} rel_err={:.4}{}",
        cpu_y0,
        gpu_y0,
        rel_err,
        if rel_err < 0.05 { " ✓" } else { " ✗" }
    );

    // Comparison against production qmv_fast CSV cost at the same
    // shape (isolated, with launch overhead subtracted).
    let qmv_fast_csv_us = 384.02_f64;
    eprintln!(
        "  vs qmv_fast (isolated CSV): {:.2} µs → ratio {:.2}x",
        qmv_fast_csv_us,
        avg_us / qmv_fast_csv_us,
    );
    eprintln!(
        "  (qmv_fast is INT4 with gs64; this is bf16 dense — apples-to-oranges \
         in dtype, but tells us if single-TG GEMV at this dim is feasible at all)"
    );
}

fn fill_pattern_bf16(buf: &Buffer, count: usize, seed: u32) {
    let ptr = buf.contents().as_ptr() as *mut u16; // bf16 stored as u16
    for i in 0..count {
        // Deterministic small float in [-0.5, 0.5).
        let h = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) >> 16;
        let f = ((h as i32 - 32768) as f32) / 65536.0;
        unsafe { *ptr.add(i) = f32_to_bf16_bits(f) };
    }
}

fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    // Round-to-nearest-even.
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn read_bf16(buf: &Buffer, idx: usize) -> f32 {
    let ptr = buf.contents().as_ptr() as *const u16;
    let b = unsafe { *ptr.add(idx) };
    bf16_bits_to_f32(b)
}

fn cpu_dot_bf16(x_buf: &Buffer, w_buf: &Buffer, n: usize, k: usize) -> f32 {
    let x_ptr = x_buf.contents().as_ptr() as *const u16;
    let w_ptr = w_buf.contents().as_ptr() as *const u16;
    let row = unsafe { w_ptr.add(n * k) };
    let mut acc = 0.0f32;
    for i in 0..k {
        let xv = bf16_bits_to_f32(unsafe { *x_ptr.add(i) });
        let wv = bf16_bits_to_f32(unsafe { *row.add(i) });
        acc += xv * wv;
    }
    acc
}
