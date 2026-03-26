//! Benchmark: fused kernel vs two separate launches.
//!
//! Run with: cargo test -p ptx-fusion --features cuda --test cuda_bench -- --nocapture

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ptx_fusion_macros::fuse_kernels;
use std::sync::Arc;
use std::time::Instant;

const RMS_NORM_PTX: &str = include_str!("../kernels/rms_norm.ptx");
const MATVEC_PTX: &str = include_str!("../kernels/matvec.ptx");

fuse_kernels!(
    "kernels/rms_norm.ptx",
    "kernels/matvec.ptx",
    "fused_rms_norm_matvec",
    "output",
    "vec_in"
);

fn get_ctx() -> Arc<CudaContext> {
    CudaContext::new(0).expect("no CUDA device available")
}

#[test]
fn bench_fused_vs_separate() {
    let ctx = get_ctx();
    let stream = ctx.default_stream();

    let k = 128u32;
    let m = 64u32;
    let n = k;
    let epsilon = 1e-5f32;

    let input: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let matrix: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();

    // Load modules
    let mod_rms = ctx.load_module(Ptx::from_src(RMS_NORM_PTX)).unwrap();
    let mod_mv = ctx.load_module(Ptx::from_src(MATVEC_PTX)).unwrap();
    let mod_fused = ctx
        .load_module(Ptx::from_src(FUSED_RMS_NORM_MATVEC))
        .unwrap();
    let func_rms = mod_rms.load_function("rms_norm").unwrap();
    let func_mv = mod_mv.load_function("matvec").unwrap();
    let func_fused = mod_fused.load_function("fused_rms_norm_matvec").unwrap();

    // Allocate GPU memory
    let input_gpu = stream.clone_htod(&input).unwrap();
    let weight_gpu = stream.clone_htod(&weight).unwrap();
    let matrix_gpu = stream.clone_htod(&matrix).unwrap();
    let mut intermediate_gpu: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();
    let mut output_separate: CudaSlice<f32> = stream.alloc_zeros(m as usize).unwrap();
    let mut output_fused: CudaSlice<f32> = stream.alloc_zeros(m as usize).unwrap();

    let block_size = k.max(64);
    let grid_size = 1u32;
    let cfg_rms = LaunchConfig::for_num_elems(n);
    let cfg_mv = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    let cfg_fused = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let warmup = 100;
    let iterations = 10_000;

    // ── Warmup ──
    for _ in 0..warmup {
        unsafe {
            stream
                .launch_builder(&func_rms)
                .arg(&input_gpu)
                .arg(&mut intermediate_gpu)
                .arg(&weight_gpu)
                .arg(&n)
                .arg(&epsilon)
                .launch(cfg_rms)
        }
        .unwrap();
        unsafe {
            stream
                .launch_builder(&func_mv)
                .arg(&matrix_gpu)
                .arg(&intermediate_gpu)
                .arg(&mut output_separate)
                .arg(&m)
                .arg(&k)
                .launch(cfg_mv)
        }
        .unwrap();
    }
    for _ in 0..warmup {
        unsafe {
            stream
                .launch_builder(&func_fused)
                .arg(&input_gpu)
                .arg(&weight_gpu)
                .arg(&n)
                .arg(&epsilon)
                .arg(&matrix_gpu)
                .arg(&mut output_fused)
                .arg(&m)
                .arg(&k)
                .launch(cfg_fused)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();

    // ── Bench separate ──
    let t0 = Instant::now();
    for _ in 0..iterations {
        unsafe {
            stream
                .launch_builder(&func_rms)
                .arg(&input_gpu)
                .arg(&mut intermediate_gpu)
                .arg(&weight_gpu)
                .arg(&n)
                .arg(&epsilon)
                .launch(cfg_rms)
        }
        .unwrap();
        unsafe {
            stream
                .launch_builder(&func_mv)
                .arg(&matrix_gpu)
                .arg(&intermediate_gpu)
                .arg(&mut output_separate)
                .arg(&m)
                .arg(&k)
                .launch(cfg_mv)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();
    let separate_us = t0.elapsed().as_micros() as f64 / iterations as f64;

    // ── Bench fused ──
    let t0 = Instant::now();
    for _ in 0..iterations {
        unsafe {
            stream
                .launch_builder(&func_fused)
                .arg(&input_gpu)
                .arg(&weight_gpu)
                .arg(&n)
                .arg(&epsilon)
                .arg(&matrix_gpu)
                .arg(&mut output_fused)
                .arg(&m)
                .arg(&k)
                .launch(cfg_fused)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();
    let fused_us = t0.elapsed().as_micros() as f64 / iterations as f64;

    let speedup = separate_us / fused_us;
    let saved_us = separate_us - fused_us;

    println!("\n╔══════════════════════════════════════════════╗");
    println!("║  Ferrite Fusion Benchmark (M={m}, K={k})      ║");
    println!("╠══════════════════════════════════════════════╣");
    println!("║  Separate (2 launches): {separate_us:8.2} us/iter      ║");
    println!("║  Fused    (1 launch):   {fused_us:8.2} us/iter      ║");
    println!("║  Speedup:               {speedup:8.2}x            ║");
    println!("║  Saved per iter:        {saved_us:8.2} us           ║");
    println!("╚══════════════════════════════════════════════╝\n");
}
