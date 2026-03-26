//! Step 2: Test parser and fusion against real nvcc-compiled PTX.
//!
//! The kernels in kernels/*_real.ptx were compiled with:
//!   nvcc -ptx -arch=sm_89 kernel.cu -o kernel_real.ptx
//!
//! Run with: cargo test -p ptx-fusion --features cuda --test cuda_nvcc -- --nocapture

#![cfg(feature = "cuda")]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::assertions_on_constants)]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ptx_fusion_macros::{analyze_kernel, rewrite_kernel};
use std::sync::Arc;

// ── Compile-time protocol extraction from nvcc PTX ──

analyze_kernel!("kernels/rms_norm_real.ptx");
analyze_kernel!("kernels/matvec_real.ptx");

// Register rewrite on nvcc-compiled PTX
rewrite_kernel!("kernels/rms_norm_real.ptx", {
    "%f2" => "%f20",
    "%r1" => "%r20"
});

const RMS_NORM_REAL_PTX: &str = include_str!("../kernels/rms_norm_real.ptx");
const MATVEC_REAL_PTX: &str = include_str!("../kernels/matvec_real.ptx");

fn ctx() -> Arc<CudaContext> {
    CudaContext::new(0).unwrap()
}

// ── Protocol extraction tests ──

#[test]
fn nvcc_rms_norm_protocol() {
    println!("{:#?}", RMS_NORM_REAL);

    // Should detect the kernel name
    assert_eq!(RMS_NORM_REAL.name, "rms_norm_real");

    // 5 params: input(ptr), output(ptr), weight(ptr), n(u32), epsilon(f32)
    assert_eq!(RMS_NORM_REAL.params.len(), 5);
    assert!(RMS_NORM_REAL.params[0].is_pointer); // input
    assert!(RMS_NORM_REAL.params[1].is_pointer); // output
    assert!(RMS_NORM_REAL.params[2].is_pointer); // weight
    assert!(!RMS_NORM_REAL.params[3].is_pointer); // n
    assert!(!RMS_NORM_REAL.params[4].is_pointer); // epsilon

    // 2 global loads (input + weight), 1 global store (output)
    assert_eq!(
        RMS_NORM_REAL.global_loads.len(),
        2,
        "expected 2 global loads"
    );
    assert_eq!(
        RMS_NORM_REAL.global_stores.len(),
        1,
        "expected 1 global store"
    );

    // Param tracing should resolve to actual param names (not ?%rdN)
    for load in RMS_NORM_REAL.global_loads {
        assert!(
            !load.param_name.starts_with('?'),
            "unresolved param trace: {}",
            load.param_name
        );
    }
    for store in RMS_NORM_REAL.global_stores {
        assert!(
            !store.param_name.starts_with('?'),
            "unresolved param trace: {}",
            store.param_name
        );
    }

    // Load params should be param_0 (input) and param_2 (weight)
    let load_params: Vec<&str> = RMS_NORM_REAL
        .global_loads
        .iter()
        .map(|l| l.param_name)
        .collect();
    assert!(
        load_params.contains(&"rms_norm_real_param_0"),
        "missing input load: {load_params:?}"
    );
    assert!(
        load_params.contains(&"rms_norm_real_param_2"),
        "missing weight load: {load_params:?}"
    );

    // Store param should be param_1 (output)
    assert_eq!(
        RMS_NORM_REAL.global_stores[0].param_name,
        "rms_norm_real_param_1"
    );

    // No SMEM, no MMA
    assert_eq!(RMS_NORM_REAL.total_smem_bytes, 0);
    assert!(!RMS_NORM_REAL.has_mma);

    println!("PASS: nvcc rms_norm protocol extraction correct");
}

#[test]
fn nvcc_matvec_protocol() {
    println!("{:#?}", MATVEC_REAL);

    assert_eq!(MATVEC_REAL.name, "matvec_real");
    assert_eq!(MATVEC_REAL.params.len(), 5);

    // Should have global loads (vec_in for SMEM load + matrix in dot product)
    assert!(
        MATVEC_REAL.global_loads.len() >= 2,
        "expected at least 2 global loads, got {}",
        MATVEC_REAL.global_loads.len()
    );

    // Should have 1 global store (output)
    assert_eq!(
        MATVEC_REAL.global_stores.len(),
        1,
        "expected 1 global store"
    );

    // Should detect barrier
    assert!(
        !MATVEC_REAL.barriers.is_empty(),
        "expected bar.sync in matvec"
    );

    // Should detect SMEM usage
    assert!(MATVEC_REAL.smem_loads > 0, "expected SMEM loads");
    assert!(MATVEC_REAL.smem_stores > 0, "expected SMEM stores");

    println!("PASS: nvcc matvec protocol extraction correct");
}

// ── CUDA correctness: run nvcc-compiled kernels ──

#[test]
fn nvcc_rms_norm_runs_correctly() {
    let c = ctx();
    let stream = c.default_stream();
    let n = 256u32;
    let epsilon = 1e-5f32;

    let input: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();

    let module = c.load_module(Ptx::from_src(RMS_NORM_REAL_PTX)).unwrap();
    let func = module.load_function("rms_norm_real").unwrap();

    let inp = stream.clone_htod(&input).unwrap();
    let wgt = stream.clone_htod(&weight).unwrap();
    let mut out: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();

    let cfg = LaunchConfig::for_num_elems(n);
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&inp)
            .arg(&mut out)
            .arg(&wgt)
            .arg(&n)
            .arg(&epsilon)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out).unwrap();

    // Sanity: not all zeros
    let sum: f32 = result.iter().sum();
    assert!(sum.abs() > 1.0, "output is zeros (sum={sum})");

    // Verify against CPU reference
    for i in 0..n as usize {
        let val = input[i];
        let rms = (val * val + epsilon).sqrt().recip();
        let expected = val * rms * weight[i];
        let diff = (result[i] - expected).abs();
        assert!(
            diff < 1e-3,
            "mismatch at [{i}]: gpu={}, cpu={expected}, diff={diff}",
            result[i]
        );
    }

    println!(
        "PASS: nvcc rms_norm produces correct output (n={n}), sample: {:?}",
        &result[..4]
    );
}

#[test]
fn nvcc_matvec_runs_correctly() {
    let c = ctx();
    let stream = c.default_stream();
    let m = 64u32;
    let k = 128u32;

    let matrix: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let vec_in: Vec<f32> = (0..k).map(|i| (i as f32 + 1.0) * 0.1).collect();

    let module = c.load_module(Ptx::from_src(MATVEC_REAL_PTX)).unwrap();
    let func = module.load_function("matvec_real").unwrap();

    let mat = stream.clone_htod(&matrix).unwrap();
    let vec_gpu = stream.clone_htod(&vec_in).unwrap();
    let mut out: CudaSlice<f32> = stream.alloc_zeros(m as usize).unwrap();

    let block = k.max(64);
    let grid = m.div_ceil(block);
    // Dynamic shared memory for vec_in
    let smem_bytes = k * 4;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: smem_bytes,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&mat)
            .arg(&vec_gpu)
            .arg(&mut out)
            .arg(&m)
            .arg(&k)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out).unwrap();

    // Sanity
    let sum: f32 = result.iter().sum();
    assert!(sum.abs() > 0.01, "output is zeros (sum={sum})");

    // CPU reference
    for i in 0..m as usize {
        let mut expected = 0.0f32;
        for j in 0..k as usize {
            expected += matrix[i * k as usize + j] * vec_in[j];
        }
        let diff = (result[i] - expected).abs();
        assert!(
            diff < 1e-2,
            "mismatch at [{i}]: gpu={}, cpu={expected}, diff={diff}",
            result[i]
        );
    }

    println!(
        "PASS: nvcc matvec produces correct output (M={m}, K={k}), sample: {:?}",
        &result[..4]
    );
}

// ── Register rewrite on nvcc-compiled PTX ──

#[test]
fn nvcc_rms_norm_rewrite_matches() {
    let c = ctx();
    let stream = c.default_stream();
    let n = 256u32;
    let epsilon = 1e-5f32;

    let input: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();

    // Run original
    let mod_orig = c.load_module(Ptx::from_src(RMS_NORM_REAL_PTX)).unwrap();
    let f_orig = mod_orig.load_function("rms_norm_real").unwrap();
    let inp = stream.clone_htod(&input).unwrap();
    let wgt = stream.clone_htod(&weight).unwrap();
    let mut out_orig: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe {
        stream
            .launch_builder(&f_orig)
            .arg(&inp)
            .arg(&mut out_orig)
            .arg(&wgt)
            .arg(&n)
            .arg(&epsilon)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let orig = stream.clone_dtoh(&out_orig).unwrap();

    // Run rewritten
    let mod_rew = c
        .load_module(Ptx::from_src(RMS_NORM_REAL_REWRITTEN))
        .unwrap();
    let f_rew = mod_rew.load_function("rms_norm_real").unwrap();
    let mut out_rew: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();
    unsafe {
        stream
            .launch_builder(&f_rew)
            .arg(&inp)
            .arg(&mut out_rew)
            .arg(&wgt)
            .arg(&n)
            .arg(&epsilon)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let rew = stream.clone_dtoh(&out_rew).unwrap();

    // Compare bitwise
    for (i, (a, b)) in orig.iter().zip(rew.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "mismatch at [{i}]: orig={a}, rewritten={b}"
        );
    }

    println!("PASS: nvcc rms_norm rewrite is bitwise identical (n={n})");
}
