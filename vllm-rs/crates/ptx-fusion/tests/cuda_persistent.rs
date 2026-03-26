//! Test: persistent kernel wrapper — GPU-filling grid with work-queue loop.
//!
//! A persistent kernel that fuses rms_norm -> GEMM in a loop:
//! - 108 blocks (fills L4), each grabs rows from an atomic counter
//! - Phase 1: rms_norm on grabbed row -> SMEM
//! - Phase 2: GEMM using A from SMEM -> output
//! - Loops until all M rows processed
//!
//! Run with: cargo test -p ptx-fusion --features cuda --test cuda_persistent -- --nocapture

#![cfg(feature = "cuda")]
#![allow(clippy::too_many_arguments)]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ptx_fusion_macros::{extract_entry, fuse_real_kernels, persistent_fuse_real_kernels};
use std::sync::Arc;

// Extract standalone kernels for the "separate" baseline
extract_entry!(
    "kernels/vllm_rms_norm.ptx",
    "rms_norm_kernelIfE",
    RMS_NORM_F32_PTX
);
extract_entry!("kernels/gemm_row_f32.ptx", "gemm_row_f32", GEMM_ROW_PTX);

// Persistent fused kernel
persistent_fuse_real_kernels!(
    "kernels/vllm_rms_norm.ptx",
    "rms_norm_kernelIfE",
    "kernels/gemm_row_f32.ptx",
    "gemm_row_f32",
    "fused_rms_norm_gemm",
    "param_0",
    "param_1",
    "param_3", // gemm_row_f32_param_3 = M (total rows)
    4096,
    PERSISTENT_RMS_GEMM_PTX
);

// Non-persistent fused kernel (for benchmark comparison)
fuse_real_kernels!(
    "kernels/vllm_rms_norm.ptx",
    "rms_norm_kernelIfE",
    "kernels/gemm_row_f32.ptx",
    "gemm_row_f32",
    "fused_rms_norm_gemm",
    "param_0",
    "param_1",
    4096,
    FUSED_RMS_GEMM_PTX
);

fn ctx() -> Arc<CudaContext> {
    CudaContext::new(0).unwrap()
}

fn find_entry_name(ptx: &str) -> String {
    for line in ptx.lines() {
        let t = line.trim();
        if t.starts_with(".visible")
            && t.contains(".entry")
            && let Some(start) = t
                .find("_Z")
                .or_else(|| t.find("persistent_"))
                .or_else(|| t.find("gemm_"))
        {
            let end = t.find('(').unwrap_or(t.len());
            return t[start..end].trim().to_string();
        }
    }
    panic!("no entry found in PTX");
}

/// Run rms_norm then gemm_row as two separate launches (baseline).
fn run_separate(
    ctx: &Arc<CudaContext>,
    input: &[f32],
    weight: &[f32],
    b_matrix: &[f32],
    rows: usize,
    hidden: usize,
    n_out: usize,
    epsilon: f32,
) -> Vec<f32> {
    let stream = ctx.default_stream();

    let mod_rms = ctx.load_module(Ptx::from_src(RMS_NORM_F32_PTX)).unwrap();
    let func_rms = mod_rms
        .load_function(&find_entry_name(RMS_NORM_F32_PTX))
        .unwrap();
    let mod_gemm = ctx.load_module(Ptx::from_src(GEMM_ROW_PTX)).unwrap();
    let func_gemm = mod_gemm
        .load_function(&find_entry_name(GEMM_ROW_PTX))
        .unwrap();

    let inp = stream.clone_htod(input).unwrap();
    let wgt = stream.clone_htod(weight).unwrap();
    let b_gpu = stream.clone_htod(b_matrix).unwrap();
    let mut rms_out: CudaSlice<f32> = stream.alloc_zeros(rows * hidden).unwrap();
    let mut gemm_out: CudaSlice<f32> = stream.alloc_zeros(rows * n_out).unwrap();

    // rms_norm
    unsafe {
        stream
            .launch_builder(&func_rms)
            .arg(&mut rms_out)
            .arg(&inp)
            .arg(&wgt)
            .arg(&epsilon)
            .arg(&(hidden as i32))
            .launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .unwrap();

    // gemm_row
    let m32 = rows as i32;
    let n32 = n_out as i32;
    let k32 = hidden as i32;
    unsafe {
        stream
            .launch_builder(&func_gemm)
            .arg(&mut gemm_out)
            .arg(&rms_out)
            .arg(&b_gpu)
            .arg(&m32)
            .arg(&n32)
            .arg(&k32)
            .launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .unwrap();

    stream.synchronize().unwrap();
    stream.clone_dtoh(&gemm_out).unwrap()
}

/// Run the persistent fused kernel (single launch, 108 blocks, work queue).
fn run_persistent(
    ctx: &Arc<CudaContext>,
    input: &[f32],
    weight: &[f32],
    b_matrix: &[f32],
    rows: usize,
    hidden: usize,
    n_out: usize,
    epsilon: f32,
    num_blocks: u32,
) -> Vec<f32> {
    let stream = ctx.default_stream();

    std::fs::write("/tmp/persistent_rms_gemm.ptx", PERSISTENT_RMS_GEMM_PTX).ok();

    let module = ctx
        .load_module(Ptx::from_src(PERSISTENT_RMS_GEMM_PTX))
        .expect("persistent PTX should load");
    let func = module
        .load_function(&find_entry_name(PERSISTENT_RMS_GEMM_PTX))
        .expect("persistent entry should exist");

    let inp = stream.clone_htod(input).unwrap();
    let wgt = stream.clone_htod(weight).unwrap();
    let b_gpu = stream.clone_htod(b_matrix).unwrap();
    let mut gemm_out: CudaSlice<f32> = stream.alloc_zeros(rows * n_out).unwrap();

    // Tile counter — initialized to 0, reset by host between invocations
    let tile_counter: CudaSlice<u32> = stream.alloc_zeros(1).unwrap();

    // Persistent params: tile_counter, then same as fused
    // (tile_counter, input, weight, epsilon, hidden, C, B, M, N, K)
    let m32 = rows as i32;
    let n32 = n_out as i32;
    let k32 = hidden as i32;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&tile_counter) // persistent counter
            .arg(&inp) // rms input
            .arg(&wgt) // rms weight
            .arg(&epsilon) // rms epsilon
            .arg(&(hidden as i32)) // rms hidden_size
            .arg(&mut gemm_out) // gemm C output
            .arg(&b_gpu) // gemm B matrix
            .arg(&m32) // gemm M (= total rows)
            .arg(&n32) // gemm N
            .arg(&k32) // gemm K
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    stream.clone_dtoh(&gemm_out).unwrap()
}

#[test]
fn persistent_ptx_passes_ptxas() {
    let path = "/tmp/persistent_rms_gemm_validate.ptx";
    std::fs::write(path, PERSISTENT_RMS_GEMM_PTX).unwrap();

    let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        for line in stderr.lines().take(20) {
            println!("ptxas: {line}");
        }
        panic!("ptxas FAILED on persistent kernel");
    }

    println!(
        "PASS: persistent kernel passes ptxas ({} lines)",
        PERSISTENT_RMS_GEMM_PTX.lines().count()
    );
}

#[test]
fn persistent_matches_separate() {
    let c = ctx();

    let rows = 256; // More rows than blocks — persistent loop matters
    let hidden = 128;
    let n_out = 64;
    let epsilon = 1e-5f32;
    let num_blocks = 108; // Fill L4

    let input: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.017 + 0.3).sin())
        .collect();
    let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.002).collect();
    let b_matrix: Vec<f32> = (0..n_out * hidden)
        .map(|i| ((i as f32) * 0.013 + 0.7).cos() * 0.1)
        .collect();

    let output_separate =
        run_separate(&c, &input, &weight, &b_matrix, rows, hidden, n_out, epsilon);
    let output_persistent = run_persistent(
        &c,
        &input,
        &weight,
        &b_matrix,
        rows,
        hidden,
        n_out,
        epsilon,
        num_blocks as u32,
    );

    println!("Separate:   {:?}...", &output_separate[..4]);
    println!("Persistent: {:?}...", &output_persistent[..4]);

    // Sanity
    let sum_sep: f32 = output_separate.iter().sum();
    assert!(sum_sep.abs() > 0.01, "separate output is zeros");
    let sum_pers: f32 = output_persistent.iter().sum();
    assert!(sum_pers.abs() > 0.01, "persistent output is zeros");

    // Compare
    let mut max_diff = 0.0f32;
    for (i, (&a, &b)) in output_separate
        .iter()
        .zip(output_persistent.iter())
        .enumerate()
    {
        let diff = (a - b).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff < 1e-3,
            "[{i}]: separate={a}, persistent={b}, diff={diff}"
        );
    }

    println!(
        "PASS: persistent rms_norm+gemm matches separate ({rows} rows, {num_blocks} blocks, \
         {hidden} hidden -> {n_out} out, max_diff={max_diff:.2e})"
    );
}

#[test]
fn persistent_benchmark() {
    let c = ctx();
    let stream = c.default_stream();

    let rows = 256;
    let hidden = 128;
    let n_out = 64;
    let epsilon = 1e-5f32;
    let num_blocks = 108u32;
    let warmup = 50;
    let iters = 200;

    let input: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.017 + 0.3).sin())
        .collect();
    let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.002).collect();
    let b_matrix: Vec<f32> = (0..n_out * hidden)
        .map(|i| ((i as f32) * 0.013 + 0.7).cos() * 0.1)
        .collect();

    // Pre-load modules
    let mod_rms = c.load_module(Ptx::from_src(RMS_NORM_F32_PTX)).unwrap();
    let func_rms = mod_rms
        .load_function(&find_entry_name(RMS_NORM_F32_PTX))
        .unwrap();
    let mod_gemm = c.load_module(Ptx::from_src(GEMM_ROW_PTX)).unwrap();
    let func_gemm = mod_gemm
        .load_function(&find_entry_name(GEMM_ROW_PTX))
        .unwrap();
    let mod_fused = c.load_module(Ptx::from_src(FUSED_RMS_GEMM_PTX)).unwrap();
    let func_fused = mod_fused.load_function("fused_rms_norm_gemm").unwrap();
    let mod_persistent = c
        .load_module(Ptx::from_src(PERSISTENT_RMS_GEMM_PTX))
        .unwrap();
    let func_persistent = mod_persistent
        .load_function(&find_entry_name(PERSISTENT_RMS_GEMM_PTX))
        .unwrap();

    let inp = stream.clone_htod(&input).unwrap();
    let wgt = stream.clone_htod(&weight).unwrap();
    let b_gpu = stream.clone_htod(&b_matrix).unwrap();
    let mut rms_out: CudaSlice<f32> = stream.alloc_zeros(rows * hidden).unwrap();
    let mut out: CudaSlice<f32> = stream.alloc_zeros(rows * n_out).unwrap();

    let m32 = rows as i32;
    let n32 = n_out as i32;
    let k32 = hidden as i32;

    let cfg_per_row = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let cfg_persistent = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // ── Benchmark: separate (2 launches) ──
    for _ in 0..warmup {
        unsafe {
            stream
                .launch_builder(&func_rms)
                .arg(&mut rms_out)
                .arg(&inp)
                .arg(&wgt)
                .arg(&epsilon)
                .arg(&(hidden as i32))
                .launch(cfg_per_row)
        }
        .unwrap();
        unsafe {
            stream
                .launch_builder(&func_gemm)
                .arg(&mut out)
                .arg(&rms_out)
                .arg(&b_gpu)
                .arg(&m32)
                .arg(&n32)
                .arg(&k32)
                .launch(cfg_per_row)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        unsafe {
            stream
                .launch_builder(&func_rms)
                .arg(&mut rms_out)
                .arg(&inp)
                .arg(&wgt)
                .arg(&epsilon)
                .arg(&(hidden as i32))
                .launch(cfg_per_row)
        }
        .unwrap();
        unsafe {
            stream
                .launch_builder(&func_gemm)
                .arg(&mut out)
                .arg(&rms_out)
                .arg(&b_gpu)
                .arg(&m32)
                .arg(&n32)
                .arg(&k32)
                .launch(cfg_per_row)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();
    let separate_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // ── Benchmark: fused (1 launch, grid=M) ──
    for _ in 0..warmup {
        unsafe {
            stream
                .launch_builder(&func_fused)
                .arg(&inp)
                .arg(&wgt)
                .arg(&epsilon)
                .arg(&(hidden as i32))
                .arg(&mut out)
                .arg(&b_gpu)
                .arg(&m32)
                .arg(&n32)
                .arg(&k32)
                .launch(cfg_per_row)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        unsafe {
            stream
                .launch_builder(&func_fused)
                .arg(&inp)
                .arg(&wgt)
                .arg(&epsilon)
                .arg(&(hidden as i32))
                .arg(&mut out)
                .arg(&b_gpu)
                .arg(&m32)
                .arg(&n32)
                .arg(&k32)
                .launch(cfg_per_row)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();
    let fused_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // ── Benchmark: persistent (1 launch, grid=108, work queue) ──
    let mut tile_counter: CudaSlice<u32> = stream.alloc_zeros(1).unwrap();
    for _ in 0..warmup {
        stream.memcpy_htod(&[0u32], &mut tile_counter).unwrap();
        unsafe {
            stream
                .launch_builder(&func_persistent)
                .arg(&tile_counter)
                .arg(&inp)
                .arg(&wgt)
                .arg(&epsilon)
                .arg(&(hidden as i32))
                .arg(&mut out)
                .arg(&b_gpu)
                .arg(&m32)
                .arg(&n32)
                .arg(&k32)
                .launch(cfg_persistent)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        stream.memcpy_htod(&[0u32], &mut tile_counter).unwrap();
        unsafe {
            stream
                .launch_builder(&func_persistent)
                .arg(&tile_counter)
                .arg(&inp)
                .arg(&wgt)
                .arg(&epsilon)
                .arg(&(hidden as i32))
                .arg(&mut out)
                .arg(&b_gpu)
                .arg(&m32)
                .arg(&n32)
                .arg(&k32)
                .launch(cfg_persistent)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();
    let persistent_us = t0.elapsed().as_micros() as f64 / iters as f64;

    let speedup_fused = separate_us / fused_us;
    let speedup_persistent = separate_us / persistent_us;

    println!();
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Persistent Kernel Benchmark (M={rows}, K={hidden}, N={n_out})  ║");
    println!("╠══════════════════════════════════════════════════════╣");
    println!("║  Separate  (2 launches, grid={rows}):  {separate_us:8.2} us/iter  ║");
    println!("║  Fused     (1 launch, grid={rows}):    {fused_us:8.2} us/iter  ║");
    println!("║  Persistent (1 launch, grid={num_blocks}):  {persistent_us:8.2} us/iter  ║");
    println!("║                                                      ║");
    println!("║  Fused vs separate:      {speedup_fused:5.2}x                     ║");
    println!("║  Persistent vs separate: {speedup_persistent:5.2}x                     ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();
}
