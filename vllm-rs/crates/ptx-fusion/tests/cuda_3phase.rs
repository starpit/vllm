//! Test: 3-phase MLP pipeline (norm -> GEMM+SiLU -> GEMM) in a persistent kernel.
//!
//! Proves multi-phase chaining: SMEM handoff between Phase 1-2, GMEM handoff
//! between Phase 2-3, SiLU epilogue injection, all in one persistent launch.
//!
//! Run with: cargo test -p ptx-fusion --features cuda --test cuda_3phase -- --nocapture

#![cfg(feature = "cuda")]
#![allow(clippy::too_many_arguments)]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ptx_fusion_macros::{extract_entry, fuse_3phase_mlp};
use std::sync::Arc;

// Standalone kernels for separate baseline
extract_entry!(
    "kernels/vllm_rms_norm.ptx",
    "rms_norm_kernelIfE",
    RMS_NORM_PTX
);
extract_entry!("kernels/gemm_row_f32.ptx", "gemm_row_f32", GEMM_ROW_PTX);

// 3-phase persistent MLP kernel
fuse_3phase_mlp!(
    "kernels/vllm_rms_norm.ptx",
    "rms_norm_kernelIfE",
    "kernels/gemm_row_f32.ptx",
    "gemm_row_f32",
    "kernels/gemm_row_f32.ptx",
    "gemm_row_f32",
    "fused_mlp",
    "param_0",
    "param_1", // Phase 1->2 SMEM binding
    "param_0",
    "param_1", // Phase 2->3 GMEM binding
    4096,
    FUSED_MLP_PTX
);

fn ctx() -> Arc<CudaContext> {
    CudaContext::new(0).unwrap()
}

fn find_entry(ptx: &str) -> String {
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
    panic!("no entry found");
}

/// SiLU activation on CPU
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Run 3 separate launches: rms_norm, GEMM+SiLU, GEMM
fn run_separate(
    ctx: &Arc<CudaContext>,
    input: &[f32],
    weight: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    rows: usize,
    hidden: usize,
    n_hidden: usize,
    n_out: usize,
    epsilon: f32,
) -> Vec<f32> {
    let stream = ctx.default_stream();

    let mod_rms = ctx.load_module(Ptx::from_src(RMS_NORM_PTX)).unwrap();
    let f_rms = mod_rms.load_function(&find_entry(RMS_NORM_PTX)).unwrap();
    let mod_gemm = ctx.load_module(Ptx::from_src(GEMM_ROW_PTX)).unwrap();
    let f_gemm = mod_gemm.load_function(&find_entry(GEMM_ROW_PTX)).unwrap();

    let inp = stream.clone_htod(input).unwrap();
    let wgt = stream.clone_htod(weight).unwrap();
    let w_up_gpu = stream.clone_htod(w_up).unwrap();
    let w_down_gpu = stream.clone_htod(w_down).unwrap();
    let mut norm_out: CudaSlice<f32> = stream.alloc_zeros(rows * hidden).unwrap();
    let mut up_out: CudaSlice<f32> = stream.alloc_zeros(rows * n_hidden).unwrap();
    let mut final_out: CudaSlice<f32> = stream.alloc_zeros(rows * n_out).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // Phase 1: rms_norm
    unsafe {
        stream
            .launch_builder(&f_rms)
            .arg(&mut norm_out)
            .arg(&inp)
            .arg(&wgt)
            .arg(&epsilon)
            .arg(&(hidden as i32))
            .launch(cfg)
    }
    .unwrap();

    // Phase 2: GEMM (norm_out @ W_up^T)
    let m = rows as i32;
    let n1 = n_hidden as i32;
    let k1 = hidden as i32;
    unsafe {
        stream
            .launch_builder(&f_gemm)
            .arg(&mut up_out)
            .arg(&norm_out)
            .arg(&w_up_gpu)
            .arg(&m)
            .arg(&n1)
            .arg(&k1)
            .launch(cfg)
    }
    .unwrap();

    // Apply SiLU on CPU (since we don't have a separate SiLU kernel)
    stream.synchronize().unwrap();
    let mut up_host = stream.clone_dtoh(&up_out).unwrap();
    for v in &mut up_host {
        *v = silu(*v);
    }
    stream.memcpy_htod(&up_host, &mut up_out).unwrap();

    // Phase 3: GEMM (silu_out @ W_down^T)
    let n2 = n_out as i32;
    let k2 = n_hidden as i32;
    unsafe {
        stream
            .launch_builder(&f_gemm)
            .arg(&mut final_out)
            .arg(&up_out)
            .arg(&w_down_gpu)
            .arg(&m)
            .arg(&n2)
            .arg(&k2)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    stream.clone_dtoh(&final_out).unwrap()
}

#[test]
fn three_phase_ptx_passes_ptxas() {
    let path = "/tmp/fused_mlp_3phase.ptx";
    std::fs::write(path, FUSED_MLP_PTX).unwrap();

    let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        for line in stderr.lines().take(20) {
            println!("ptxas: {line}");
        }
        panic!("ptxas FAILED on 3-phase MLP kernel");
    }

    println!(
        "PASS: 3-phase MLP kernel passes ptxas ({} lines)",
        FUSED_MLP_PTX.lines().count()
    );
}

#[test]
fn three_phase_matches_separate() {
    let c = ctx();
    let stream = c.default_stream();

    let rows = 256;
    let hidden = 128; // K
    let n_hidden = 64; // Phase 2 output / Phase 3 input
    let n_out = 32; // Final output
    let epsilon = 1e-5f32;
    let num_blocks = 108u32;

    let input: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.017 + 0.3).sin())
        .collect();
    let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.002).collect();
    let w_up: Vec<f32> = (0..n_hidden * hidden)
        .map(|i| ((i as f32) * 0.013 + 0.7).cos() * 0.1)
        .collect();
    let w_down: Vec<f32> = (0..n_out * n_hidden)
        .map(|i| ((i as f32) * 0.011 + 0.5).sin() * 0.1)
        .collect();

    // Separate: 3 launches + CPU SiLU
    let output_separate = run_separate(
        &c, &input, &weight, &w_up, &w_down, rows, hidden, n_hidden, n_out, epsilon,
    );

    // 3-phase persistent: single launch
    std::fs::write("/tmp/fused_mlp_3phase.ptx", FUSED_MLP_PTX).ok();

    let module = c
        .load_module(Ptx::from_src(FUSED_MLP_PTX))
        .expect("3-phase PTX should load");
    let func = module
        .load_function(&find_entry(FUSED_MLP_PTX))
        .expect("3-phase entry should exist");

    let inp = stream.clone_htod(&input).unwrap();
    let wgt = stream.clone_htod(&weight).unwrap();
    let w_up_gpu = stream.clone_htod(&w_up).unwrap();
    let w_down_gpu = stream.clone_htod(&w_down).unwrap();
    let mut up_buf: CudaSlice<f32> = stream.alloc_zeros(rows * n_hidden).unwrap();
    let mut final_out: CudaSlice<f32> = stream.alloc_zeros(rows * n_out).unwrap();
    let mut tile_counter: CudaSlice<u32> = stream.alloc_zeros(1).unwrap();

    // Persistent 3-phase params:
    // tile_counter,
    // rms: input, weight, epsilon, hidden
    // gemm1: C (=up_buf, shared with gemm2 A), B (=w_up), M, N, K
    // gemm2: C (=final_out), B (=w_down), M, N, K
    let m = rows as i32;
    let n1 = n_hidden as i32;
    let k1 = hidden as i32;
    let n2 = n_out as i32;
    let k2 = n_hidden as i32;

    stream.memcpy_htod(&[0u32], &mut tile_counter).unwrap();

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
            .arg(&mut up_buf) // gemm1 C output (shared: gemm2 A input)
            .arg(&w_up_gpu) // gemm1 B
            .arg(&m) // gemm1 M
            .arg(&n1) // gemm1 N
            .arg(&k1) // gemm1 K
            .arg(&mut final_out) // gemm2 C output
            .arg(&w_down_gpu) // gemm2 B
            .arg(&m) // gemm2 M
            .arg(&n2) // gemm2 N
            .arg(&k2) // gemm2 K
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let output_fused = stream.clone_dtoh(&final_out).unwrap();

    println!("Separate:  {:?}...", &output_separate[..4]);
    println!("3-phase:   {:?}...", &output_fused[..4]);

    // Sanity
    let sum_sep: f32 = output_separate.iter().sum();
    assert!(sum_sep.abs() > 0.001, "separate output is zeros");
    let sum_fused: f32 = output_fused.iter().sum();
    assert!(sum_fused.abs() > 0.001, "fused output is zeros");

    // Compare
    let mut max_diff = 0.0f32;
    for (i, (&a, &b)) in output_separate.iter().zip(output_fused.iter()).enumerate() {
        let diff = (a - b).abs();
        max_diff = max_diff.max(diff);
        assert!(diff < 1e-2, "[{i}]: separate={a}, fused={b}, diff={diff}");
    }

    println!(
        "PASS: 3-phase MLP matches separate ({rows} rows, {num_blocks} blocks, \
         {hidden}->{n_hidden}->{n_out}, max_diff={max_diff:.2e})"
    );
}

#[test]
fn three_phase_benchmark() {
    let c = ctx();
    let stream = c.default_stream();

    let rows = 256;
    let hidden = 128;
    let n_hidden = 64;
    let n_out = 32;
    let epsilon = 1e-5f32;
    let num_blocks = 108u32;
    let warmup = 50;
    let iters = 200;

    let input: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.017 + 0.3).sin())
        .collect();
    let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.002).collect();
    let w_up: Vec<f32> = (0..n_hidden * hidden)
        .map(|i| ((i as f32) * 0.013 + 0.7).cos() * 0.1)
        .collect();
    let w_down: Vec<f32> = (0..n_out * n_hidden)
        .map(|i| ((i as f32) * 0.011 + 0.5).sin() * 0.1)
        .collect();

    // Pre-load modules
    let mod_rms = c.load_module(Ptx::from_src(RMS_NORM_PTX)).unwrap();
    let f_rms = mod_rms.load_function(&find_entry(RMS_NORM_PTX)).unwrap();
    let mod_gemm = c.load_module(Ptx::from_src(GEMM_ROW_PTX)).unwrap();
    let f_gemm = mod_gemm.load_function(&find_entry(GEMM_ROW_PTX)).unwrap();
    let mod_3phase = c.load_module(Ptx::from_src(FUSED_MLP_PTX)).unwrap();
    let f_3phase = mod_3phase
        .load_function(&find_entry(FUSED_MLP_PTX))
        .unwrap();

    let inp = stream.clone_htod(&input).unwrap();
    let wgt = stream.clone_htod(&weight).unwrap();
    let w_up_gpu = stream.clone_htod(&w_up).unwrap();
    let w_down_gpu = stream.clone_htod(&w_down).unwrap();
    let mut norm_out: CudaSlice<f32> = stream.alloc_zeros(rows * hidden).unwrap();
    let mut up_out: CudaSlice<f32> = stream.alloc_zeros(rows * n_hidden).unwrap();
    let mut final_out: CudaSlice<f32> = stream.alloc_zeros(rows * n_out).unwrap();

    let cfg_rows = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let cfg_persistent = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let m = rows as i32;
    let n1 = n_hidden as i32;
    let k1 = hidden as i32;
    let n2 = n_out as i32;
    let k2 = n_hidden as i32;

    // ── Benchmark: separate (3 launches + host SiLU) ──
    // Note: host SiLU is unfairly slow (includes sync + memcpy), so we also
    // benchmark "separate GPU-only" which just times the 3 kernel launches
    // without the SiLU (to show launch overhead).
    for _ in 0..warmup {
        unsafe {
            stream
                .launch_builder(&f_rms)
                .arg(&mut norm_out)
                .arg(&inp)
                .arg(&wgt)
                .arg(&epsilon)
                .arg(&(hidden as i32))
                .launch(cfg_rows)
        }
        .unwrap();
        unsafe {
            stream
                .launch_builder(&f_gemm)
                .arg(&mut up_out)
                .arg(&norm_out)
                .arg(&w_up_gpu)
                .arg(&m)
                .arg(&n1)
                .arg(&k1)
                .launch(cfg_rows)
        }
        .unwrap();
        unsafe {
            stream
                .launch_builder(&f_gemm)
                .arg(&mut final_out)
                .arg(&up_out)
                .arg(&w_down_gpu)
                .arg(&m)
                .arg(&n2)
                .arg(&k2)
                .launch(cfg_rows)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        unsafe {
            stream
                .launch_builder(&f_rms)
                .arg(&mut norm_out)
                .arg(&inp)
                .arg(&wgt)
                .arg(&epsilon)
                .arg(&(hidden as i32))
                .launch(cfg_rows)
        }
        .unwrap();
        unsafe {
            stream
                .launch_builder(&f_gemm)
                .arg(&mut up_out)
                .arg(&norm_out)
                .arg(&w_up_gpu)
                .arg(&m)
                .arg(&n1)
                .arg(&k1)
                .launch(cfg_rows)
        }
        .unwrap();
        unsafe {
            stream
                .launch_builder(&f_gemm)
                .arg(&mut final_out)
                .arg(&up_out)
                .arg(&w_down_gpu)
                .arg(&m)
                .arg(&n2)
                .arg(&k2)
                .launch(cfg_rows)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();
    let separate_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // ── Benchmark: 3-phase persistent (1 launch) ──
    let mut tile_counter: CudaSlice<u32> = stream.alloc_zeros(1).unwrap();
    for _ in 0..warmup {
        stream.memcpy_htod(&[0u32], &mut tile_counter).unwrap();
        unsafe {
            stream
                .launch_builder(&f_3phase)
                .arg(&tile_counter)
                .arg(&inp)
                .arg(&wgt)
                .arg(&epsilon)
                .arg(&(hidden as i32))
                .arg(&mut up_out)
                .arg(&w_up_gpu)
                .arg(&m)
                .arg(&n1)
                .arg(&k1)
                .arg(&mut final_out)
                .arg(&w_down_gpu)
                .arg(&m)
                .arg(&n2)
                .arg(&k2)
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
                .launch_builder(&f_3phase)
                .arg(&tile_counter)
                .arg(&inp)
                .arg(&wgt)
                .arg(&epsilon)
                .arg(&(hidden as i32))
                .arg(&mut up_out)
                .arg(&w_up_gpu)
                .arg(&m)
                .arg(&n1)
                .arg(&k1)
                .arg(&mut final_out)
                .arg(&w_down_gpu)
                .arg(&m)
                .arg(&n2)
                .arg(&k2)
                .launch(cfg_persistent)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();
    let persistent_us = t0.elapsed().as_micros() as f64 / iters as f64;

    let speedup = separate_us / persistent_us;

    println!();
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  3-Phase MLP Benchmark (M={rows}, {hidden}->{n_hidden}->{n_out})            ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  Separate (3 launches, grid={rows}):     {separate_us:8.2} us/iter     ║");
    println!(
        "║  3-phase persistent (1 launch, grid={num_blocks}): {persistent_us:8.2} us/iter     ║"
    );
    println!("║  Speedup:                             {speedup:5.2}x               ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
}

/// Benchmark helper: run separate (3 launches) and persistent (1 launch) at given sizes.
fn bench_3phase(rows: usize, hidden: usize, n_hidden: usize, n_out: usize, num_blocks: u32) {
    let c = ctx();
    let stream = c.default_stream();
    let epsilon = 1e-5f32;
    let warmup = 30;
    let iters = 100;

    let input: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.017 + 0.3).sin())
        .collect();
    let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.002).collect();
    let w_up: Vec<f32> = (0..n_hidden * hidden)
        .map(|i| ((i as f32) * 0.013 + 0.7).cos() * 0.01)
        .collect();
    let w_down: Vec<f32> = (0..n_out * n_hidden)
        .map(|i| ((i as f32) * 0.011 + 0.5).sin() * 0.01)
        .collect();

    let mod_rms = c.load_module(Ptx::from_src(RMS_NORM_PTX)).unwrap();
    let f_rms = mod_rms.load_function(&find_entry(RMS_NORM_PTX)).unwrap();
    let mod_gemm = c.load_module(Ptx::from_src(GEMM_ROW_PTX)).unwrap();
    let f_gemm = mod_gemm.load_function(&find_entry(GEMM_ROW_PTX)).unwrap();
    let mod_3p = c.load_module(Ptx::from_src(FUSED_MLP_PTX)).unwrap();
    let f_3p = mod_3p.load_function(&find_entry(FUSED_MLP_PTX)).unwrap();

    let inp = stream.clone_htod(&input).unwrap();
    let wgt = stream.clone_htod(&weight).unwrap();
    let w_up_gpu = stream.clone_htod(&w_up).unwrap();
    let w_down_gpu = stream.clone_htod(&w_down).unwrap();
    let mut norm_out: CudaSlice<f32> = stream.alloc_zeros(rows * hidden).unwrap();
    let mut up_out: CudaSlice<f32> = stream.alloc_zeros(rows * n_hidden).unwrap();
    let mut final_out: CudaSlice<f32> = stream.alloc_zeros(rows * n_out).unwrap();

    let m = rows as i32;
    let n1 = n_hidden as i32;
    let k1 = hidden as i32;
    let n2 = n_out as i32;
    let k2 = n_hidden as i32;

    let cfg_rows = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let cfg_pers = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // Separate
    for _ in 0..warmup {
        unsafe {
            stream
                .launch_builder(&f_rms)
                .arg(&mut norm_out)
                .arg(&inp)
                .arg(&wgt)
                .arg(&epsilon)
                .arg(&(hidden as i32))
                .launch(cfg_rows)
                .unwrap();
            stream
                .launch_builder(&f_gemm)
                .arg(&mut up_out)
                .arg(&norm_out)
                .arg(&w_up_gpu)
                .arg(&m)
                .arg(&n1)
                .arg(&k1)
                .launch(cfg_rows)
                .unwrap();
            stream
                .launch_builder(&f_gemm)
                .arg(&mut final_out)
                .arg(&up_out)
                .arg(&w_down_gpu)
                .arg(&m)
                .arg(&n2)
                .arg(&k2)
                .launch(cfg_rows)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        unsafe {
            stream
                .launch_builder(&f_rms)
                .arg(&mut norm_out)
                .arg(&inp)
                .arg(&wgt)
                .arg(&epsilon)
                .arg(&(hidden as i32))
                .launch(cfg_rows)
                .unwrap();
            stream
                .launch_builder(&f_gemm)
                .arg(&mut up_out)
                .arg(&norm_out)
                .arg(&w_up_gpu)
                .arg(&m)
                .arg(&n1)
                .arg(&k1)
                .launch(cfg_rows)
                .unwrap();
            stream
                .launch_builder(&f_gemm)
                .arg(&mut final_out)
                .arg(&up_out)
                .arg(&w_down_gpu)
                .arg(&m)
                .arg(&n2)
                .arg(&k2)
                .launch(cfg_rows)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();
    let sep_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // Persistent
    let mut ctr: CudaSlice<u32> = stream.alloc_zeros(1).unwrap();
    for _ in 0..warmup {
        stream.memcpy_htod(&[0u32], &mut ctr).unwrap();
        unsafe {
            stream
                .launch_builder(&f_3p)
                .arg(&ctr)
                .arg(&inp)
                .arg(&wgt)
                .arg(&epsilon)
                .arg(&(hidden as i32))
                .arg(&mut up_out)
                .arg(&w_up_gpu)
                .arg(&m)
                .arg(&n1)
                .arg(&k1)
                .arg(&mut final_out)
                .arg(&w_down_gpu)
                .arg(&m)
                .arg(&n2)
                .arg(&k2)
                .launch(cfg_pers)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        stream.memcpy_htod(&[0u32], &mut ctr).unwrap();
        unsafe {
            stream
                .launch_builder(&f_3p)
                .arg(&ctr)
                .arg(&inp)
                .arg(&wgt)
                .arg(&epsilon)
                .arg(&(hidden as i32))
                .arg(&mut up_out)
                .arg(&w_up_gpu)
                .arg(&m)
                .arg(&n1)
                .arg(&k1)
                .arg(&mut final_out)
                .arg(&w_down_gpu)
                .arg(&m)
                .arg(&n2)
                .arg(&k2)
                .launch(cfg_pers)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();
    let pers_us = t0.elapsed().as_micros() as f64 / iters as f64;

    let speedup = sep_us / pers_us;
    println!(
        "  M={rows:4}, {hidden:4}->{n_hidden:4}->{n_out:4}  |  separate: {sep_us:8.1} us  |  persistent: {pers_us:8.1} us  |  {speedup:.2}x"
    );
}

#[test]
fn three_phase_scaling() {
    println!();
    println!("3-Phase MLP Scaling (108 persistent blocks)");
    println!("──────────────────────────────────────────────────────────────────────────────");
    bench_3phase(256, 128, 64, 32, 108);
    bench_3phase(256, 512, 256, 128, 108);
    bench_3phase(256, 1024, 512, 256, 108);
    bench_3phase(256, 2048, 1024, 512, 108);
    bench_3phase(512, 1024, 512, 256, 108);
    bench_3phase(1024, 1024, 512, 256, 108);
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!();
}
