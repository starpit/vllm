//! Test: fuse rms_norm -> row GEMM via SMEM handoff (prologue injection).
//!
//! This eliminates the GMEM round-trip between normalization and GEMM.
//! rms_norm writes to SMEM, barrier, then GEMM reads A from SMEM.
//!
//! Run with: cargo test -p ptx-fusion --features cuda --test cuda_gemm_prologue -- --nocapture

#![cfg(feature = "cuda")]
#![allow(clippy::too_many_arguments)]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ptx_fusion_macros::{extract_entry, fuse_real_kernels};
use std::sync::Arc;

// Extract rms_norm<float>
extract_entry!(
    "kernels/vllm_rms_norm.ptx",
    "rms_norm_kernelIfE",
    RMS_NORM_F32_PTX
);

// Extract row GEMM
extract_entry!("kernels/gemm_row_f32.ptx", "gemm_row_f32", GEMM_ROW_PTX);

// Fuse: rms_norm output (param_0) -> gemm_row A input (param_1)
fuse_real_kernels!(
    "kernels/vllm_rms_norm.ptx",
    "rms_norm_kernelIfE",
    "kernels/gemm_row_f32.ptx",
    "gemm_row_f32",
    "fused_rms_norm_gemm",
    "param_0", // rms_norm output
    "param_1", // gemm_row A matrix
    4096,      // SMEM buffer (supports up to hidden=4096)
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
                .or_else(|| t.find("fused_"))
                .or_else(|| t.find("gemm_"))
        {
            let end = t.find('(').unwrap_or(t.len());
            return t[start..end].trim().to_string();
        }
    }
    panic!("no entry found in PTX");
}

/// Run the row GEMM standalone (no fusion) to verify it works.
fn run_gemm_row(
    ctx: &Arc<CudaContext>,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let stream = ctx.default_stream();

    let module = ctx.load_module(Ptx::from_src(GEMM_ROW_PTX)).unwrap();
    let func = module
        .load_function(&find_entry_name(GEMM_ROW_PTX))
        .unwrap();

    let a_gpu = stream.clone_htod(a).unwrap();
    let b_gpu = stream.clone_htod(b).unwrap();
    let mut out: CudaSlice<f32> = stream.alloc_zeros(m * n).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (m as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let m32 = m as i32;
    let n32 = n as i32;
    let k32 = k as i32;
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&mut out)
            .arg(&a_gpu)
            .arg(&b_gpu)
            .arg(&m32)
            .arg(&n32)
            .arg(&k32)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    stream.clone_dtoh(&out).unwrap()
}

/// Run rms_norm then gemm_row as two separate launches.
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

    let inp = stream.clone_htod(input).unwrap();
    let wgt = stream.clone_htod(weight).unwrap();
    let b_gpu = stream.clone_htod(b_matrix).unwrap();
    let mut rms_out: CudaSlice<f32> = stream.alloc_zeros(rows * hidden).unwrap();
    let mut gemm_out: CudaSlice<f32> = stream.alloc_zeros(rows * n_out).unwrap();

    // rms_norm: one block per row
    let cfg_rms = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func_rms)
            .arg(&mut rms_out)
            .arg(&inp)
            .arg(&wgt)
            .arg(&epsilon)
            .arg(&(hidden as i32))
            .launch(cfg_rms)
    }
    .unwrap();

    // gemm_row: one block per row
    let mod_gemm = ctx.load_module(Ptx::from_src(GEMM_ROW_PTX)).unwrap();
    let func_gemm = mod_gemm
        .load_function(&find_entry_name(GEMM_ROW_PTX))
        .unwrap();
    let cfg_gemm = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
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
            .launch(cfg_gemm)
    }
    .unwrap();

    stream.synchronize().unwrap();
    stream.clone_dtoh(&gemm_out).unwrap()
}

/// Run the fused rms_norm+gemm kernel (single launch, SMEM handoff).
fn run_fused(
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

    std::fs::write("/tmp/fused_rms_gemm.ptx", FUSED_RMS_GEMM_PTX).ok();

    let module = ctx
        .load_module(Ptx::from_src(FUSED_RMS_GEMM_PTX))
        .expect("fused PTX should load");
    let func = module
        .load_function("fused_rms_norm_gemm")
        .expect("fused entry should exist");

    let inp = stream.clone_htod(input).unwrap();
    let wgt = stream.clone_htod(weight).unwrap();
    let b_gpu = stream.clone_htod(b_matrix).unwrap();
    let mut gemm_out: CudaSlice<f32> = stream.alloc_zeros(rows * n_out).unwrap();

    // Fused params = rms_norm (minus output) + gemm_row (minus A)
    // rms_norm: [out=removed], input, weight, epsilon, hidden_size
    // gemm_row: C, [A=removed], B, M, N, K
    // Merged: input, weight, epsilon, hidden_size, C, B, M, N, K
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let m32 = rows as i32;
    let n32 = n_out as i32;
    let k32 = hidden as i32;
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&inp) // rms input
            .arg(&wgt) // rms weight
            .arg(&epsilon) // rms epsilon
            .arg(&(hidden as i32)) // rms hidden_size
            .arg(&mut gemm_out) // gemm C output
            .arg(&b_gpu) // gemm B matrix
            .arg(&m32) // gemm M
            .arg(&n32) // gemm N
            .arg(&k32) // gemm K
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    stream.clone_dtoh(&gemm_out).unwrap()
}

#[test]
fn gemm_row_standalone() {
    let c = ctx();
    let m = 4;
    let n = 64;
    let k = 128;

    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.017 + 0.3).sin())
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.013 + 0.7).cos() * 0.1)
        .collect();

    let gpu_result = run_gemm_row(&c, &a, &b, m, n, k);

    // CPU reference: C[row, col] = sum_k A[row*K+k] * B[col*K+k]
    let mut expected = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a[row * k + ki] * b[col * k + ki];
            }
            expected[row * n + col] = sum;
        }
    }

    let mut max_diff = 0.0f32;
    for (i, (&g, &e)) in gpu_result.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        max_diff = max_diff.max(diff);
        assert!(diff < 1e-3, "[{i}]: gpu={g}, cpu={e}, diff={diff}");
    }

    println!("GPU sample: {:?}", &gpu_result[..4]);
    println!("CPU sample: {:?}", &expected[..4]);
    println!("PASS: gemm_row standalone correct (M={m}, N={n}, K={k}, max_diff={max_diff:.2e})");
}

#[test]
fn fused_rms_gemm_correctness() {
    let c = ctx();

    let rows = 4;
    let hidden = 128; // K dimension — must be divisible by 4 (v4 loads in rms_norm)
    let n_out = 64;
    let epsilon = 1e-5f32;

    let input: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.017 + 0.3).sin())
        .collect();
    let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.002).collect();
    let b_matrix: Vec<f32> = (0..n_out * hidden)
        .map(|i| ((i as f32) * 0.013 + 0.7).cos() * 0.1)
        .collect();

    let output_separate =
        run_separate(&c, &input, &weight, &b_matrix, rows, hidden, n_out, epsilon);
    let output_fused = run_fused(&c, &input, &weight, &b_matrix, rows, hidden, n_out, epsilon);

    println!("Separate: {:?}...", &output_separate[..4]);
    println!("Fused:    {:?}...", &output_fused[..4]);

    // Sanity
    let sum_sep: f32 = output_separate.iter().sum();
    assert!(sum_sep.abs() > 0.01, "separate output is zeros");
    let sum_fused: f32 = output_fused.iter().sum();
    assert!(sum_fused.abs() > 0.01, "fused output is zeros");

    // Compare
    let mut max_diff = 0.0f32;
    for (i, (&a, &b)) in output_separate.iter().zip(output_fused.iter()).enumerate() {
        let diff = (a - b).abs();
        max_diff = max_diff.max(diff);
        assert!(diff < 1e-3, "[{i}]: separate={a}, fused={b}, diff={diff}");
    }

    println!(
        "PASS: fused rms_norm+gemm_row matches separate ({rows} rows x {hidden} hidden -> {n_out} out, max_diff={max_diff:.2e})"
    );
}
