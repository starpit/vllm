//! GPU correctness test: GEMM with GELU injected into the epilogue.
//!
//! Compares three implementations:
//! 1. GEMM + CPU GELU (separate)
//! 2. GEMM with GELU hand-written in CUDA (nvcc reference)
//! 3. GEMM with GELU injected by Ferrite (PTX rewritten)
//!
//! Run with: cargo test -p ptx-fusion --features cuda --test cuda_gemm_gelu_correctness -- --nocapture

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ptx_fusion_macros::{extract_entry, inject_epilogue};
use std::sync::Arc;

// Original GEMM (no activation)
extract_entry!("kernels/simple_gemm_f32.ptx", "simple_gemm_f32", GEMM_PTX);

// nvcc-compiled GEMM+GELU reference
extract_entry!(
    "kernels/simple_gemm_f32.ptx",
    "simple_gemm_f32_gelu",
    GEMM_GELU_REF_PTX
);

// Ferrite-injected GEMM+GELU
inject_epilogue!(
    "kernels/simple_gemm_f32.ptx",
    "simple_gemm_f32",
    Gelu,
    GEMM_GELU_FERRITE_PTX
);

fn ctx() -> Arc<CudaContext> {
    CudaContext::new(0).unwrap()
}

fn find_entry(ptx: &str) -> String {
    for line in ptx.lines() {
        if line.contains(".entry")
            && line.contains('(')
            && let Some(start) = line.find("_Z").or_else(|| line.find("simple_"))
        {
            let end = line.find('(').unwrap_or(line.len());
            return line[start..end].trim().to_string();
        }
    }
    panic!("no entry found");
}

/// CPU reference: GELU tanh approximation
fn gelu_tanh_cpu(x: f32) -> f32 {
    let beta: f32 = 0.797_884_6;
    let kappa: f32 = 0.044715;
    let inner = beta * (x + kappa * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

#[test]
fn gemm_gelu_ferrite_matches_reference() {
    let c = ctx();
    let stream = c.default_stream();

    let m = 32u32;
    let n = 64u32;
    let k = 16u32;

    // Random-ish test data (same as SiLU test)
    let a: Vec<f32> = (0..(m * k) as usize)
        .map(|i| ((i as f32) * 0.037 - 0.5).sin() * 0.5)
        .collect();
    let b: Vec<f32> = (0..(n * k) as usize)
        .map(|i| ((i as f32) * 0.023 + 0.3).cos() * 0.5)
        .collect();

    let a_gpu = stream.clone_htod(&a).unwrap();
    let b_gpu = stream.clone_htod(&b).unwrap();

    let block = (16u32, 16u32, 1u32);
    let grid = (n.div_ceil(block.0), m.div_ceil(block.1), 1);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    };

    // 1. Run original GEMM
    let mod_gemm = c.load_module(Ptx::from_src(GEMM_PTX)).unwrap();
    let func_gemm = mod_gemm.load_function(&find_entry(GEMM_PTX)).unwrap();
    let mut out_gemm: CudaSlice<f32> = stream.alloc_zeros((m * n) as usize).unwrap();
    unsafe {
        stream
            .launch_builder(&func_gemm)
            .arg(&mut out_gemm)
            .arg(&a_gpu)
            .arg(&b_gpu)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let gemm_result = stream.clone_dtoh(&out_gemm).unwrap();

    // Apply GELU on CPU to get expected result
    let expected: Vec<f32> = gemm_result.iter().map(|&x| gelu_tanh_cpu(x)).collect();

    // 2. Run nvcc reference (GEMM + GELU in one kernel)
    let mod_ref = c.load_module(Ptx::from_src(GEMM_GELU_REF_PTX)).unwrap();
    let func_ref = mod_ref
        .load_function(&find_entry(GEMM_GELU_REF_PTX))
        .unwrap();
    let mut out_ref: CudaSlice<f32> = stream.alloc_zeros((m * n) as usize).unwrap();
    unsafe {
        stream
            .launch_builder(&func_ref)
            .arg(&mut out_ref)
            .arg(&a_gpu)
            .arg(&b_gpu)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let ref_result = stream.clone_dtoh(&out_ref).unwrap();

    // 3. Run Ferrite-injected (GEMM with GELU in epilogue via PTX rewriting)
    std::fs::write("/tmp/gemm_gelu_ferrite.ptx", GEMM_GELU_FERRITE_PTX).ok();
    let mod_ferrite = c
        .load_module(Ptx::from_src(GEMM_GELU_FERRITE_PTX))
        .expect("Ferrite-injected PTX should load");
    let func_ferrite = mod_ferrite
        .load_function(&find_entry(GEMM_GELU_FERRITE_PTX))
        .unwrap();
    let mut out_ferrite: CudaSlice<f32> = stream.alloc_zeros((m * n) as usize).unwrap();
    unsafe {
        stream
            .launch_builder(&func_ferrite)
            .arg(&mut out_ferrite)
            .arg(&a_gpu)
            .arg(&b_gpu)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let ferrite_result = stream.clone_dtoh(&out_ferrite).unwrap();

    // Sanity checks
    let sum_gemm: f32 = gemm_result.iter().sum();
    assert!(sum_gemm.abs() > 0.01, "GEMM output is all zeros");

    // Compare reference (nvcc) vs expected (CPU)
    let mut max_diff_ref = 0.0f32;
    for (i, (&r, &e)) in ref_result.iter().zip(expected.iter()).enumerate() {
        let diff = (r - e).abs();
        max_diff_ref = max_diff_ref.max(diff);
        assert!(diff < 1e-3, "ref[{i}]: nvcc={r}, cpu={e}, diff={diff}");
    }

    // Compare Ferrite vs expected (CPU)
    let mut max_diff_ferrite = 0.0f32;
    for (i, (&f, &e)) in ferrite_result.iter().zip(expected.iter()).enumerate() {
        let diff = (f - e).abs();
        max_diff_ferrite = max_diff_ferrite.max(diff);
        assert!(
            diff < 1e-2,
            "ferrite[{i}]: ferrite={f}, expected={e}, diff={diff}"
        );
    }

    // Compare Ferrite vs reference directly
    let mut max_diff_fvr = 0.0f32;
    for (i, (&f, &r)) in ferrite_result.iter().zip(ref_result.iter()).enumerate() {
        let diff = (f - r).abs();
        max_diff_fvr = max_diff_fvr.max(diff);
        assert!(
            diff < 1e-2,
            "ferrite vs ref [{i}]: ferrite={f}, ref={r}, diff={diff}"
        );
    }

    println!("GEMM output sample: {:?}", &gemm_result[..4]);
    println!("Expected (GELU):    {:?}", &expected[..4]);
    println!("nvcc reference:     {:?}", &ref_result[..4]);
    println!("Ferrite injected:   {:?}", &ferrite_result[..4]);
    println!(
        "PASS: Ferrite GEMM+GELU matches (M={m}, N={n}, K={k}, \
         max_diff vs cpu={max_diff_ferrite:.2e}, vs nvcc={max_diff_fvr:.2e})"
    );
}
