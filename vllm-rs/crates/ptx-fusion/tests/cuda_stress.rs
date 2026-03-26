//! Stress and edge-case tests for kernel fusion.
//!
//! Run with: cargo test -p ptx-fusion --features cuda --test cuda_stress -- --nocapture

#![cfg(feature = "cuda")]
#![allow(clippy::too_many_arguments)]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ptx_fusion_macros::{fuse_kernels, regfuse_kernels};
use std::sync::Arc;

const RMS_NORM_PTX: &str = include_str!("../kernels/rms_norm.ptx");
const MATVEC_PTX: &str = include_str!("../kernels/matvec.ptx");
const SCALE_PTX: &str = include_str!("../kernels/scale.ptx");

fuse_kernels!(
    "kernels/rms_norm.ptx",
    "kernels/matvec.ptx",
    "fused_rms_norm_matvec",
    "output",
    "vec_in"
);
regfuse_kernels!(
    "kernels/rms_norm.ptx",
    "kernels/scale.ptx",
    "regfused_rms_norm_scale",
    "output",
    "input"
);

fn ctx() -> Arc<CudaContext> {
    CudaContext::new(0).unwrap()
}

fn assert_close(a: &[f32], b: &[f32], label: &str, tol: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(diff <= tol, "{label}[{i}]: {x} vs {y} (diff={diff})");
    }
}

// ── Register fusion: various sizes ──

fn run_regfuse_at_size(n: u32) {
    let c = ctx();
    let stream = c.default_stream();
    let epsilon = 1e-5f32;
    let scale_val = 3.7f32;

    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37 + 0.1).sin()).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.005).collect();

    // Separate
    let mod_rms = c.load_module(Ptx::from_src(RMS_NORM_PTX)).unwrap();
    let mod_sc = c.load_module(Ptx::from_src(SCALE_PTX)).unwrap();
    let f_rms = mod_rms.load_function("rms_norm").unwrap();
    let f_sc = mod_sc.load_function("scale").unwrap();

    let inp = stream.clone_htod(&input).unwrap();
    let wgt = stream.clone_htod(&weight).unwrap();
    let mut tmp: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();
    let mut out_sep: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();

    let cfg = LaunchConfig::for_num_elems(n);
    unsafe {
        stream
            .launch_builder(&f_rms)
            .arg(&inp)
            .arg(&mut tmp)
            .arg(&wgt)
            .arg(&n)
            .arg(&epsilon)
            .launch(cfg)
    }
    .unwrap();
    unsafe {
        stream
            .launch_builder(&f_sc)
            .arg(&tmp)
            .arg(&mut out_sep)
            .arg(&n)
            .arg(&scale_val)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let sep = stream.clone_dtoh(&out_sep).unwrap();

    // Fused
    let mod_f = c
        .load_module(Ptx::from_src(REGFUSED_RMS_NORM_SCALE))
        .unwrap();
    let f_fused = mod_f.load_function("regfused_rms_norm_scale").unwrap();
    let mut out_fused: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();
    unsafe {
        stream
            .launch_builder(&f_fused)
            .arg(&inp)
            .arg(&wgt)
            .arg(&n)
            .arg(&epsilon)
            .arg(&mut out_fused)
            .arg(&scale_val)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let fused = stream.clone_dtoh(&out_fused).unwrap();

    assert_close(&sep, &fused, &format!("regfuse n={n}"), 0.0);
}

#[test]
fn regfuse_n1() {
    run_regfuse_at_size(1);
}

#[test]
fn regfuse_n7() {
    run_regfuse_at_size(7);
}

#[test]
fn regfuse_n32() {
    run_regfuse_at_size(32);
}

#[test]
fn regfuse_n255() {
    run_regfuse_at_size(255);
}

#[test]
fn regfuse_n256() {
    run_regfuse_at_size(256);
}

#[test]
fn regfuse_n1024() {
    run_regfuse_at_size(1024);
}

#[test]
fn regfuse_n4096() {
    run_regfuse_at_size(4096);
}

// ── SMEM fusion: various sizes ──

fn run_smem_fuse_at_size(m: u32, k: u32) {
    let c = ctx();
    let stream = c.default_stream();
    let n = k;
    let epsilon = 1e-5f32;

    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.23 + 0.5).cos()).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.003).collect();
    let matrix: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();

    // Separate
    let mod_rms = c.load_module(Ptx::from_src(RMS_NORM_PTX)).unwrap();
    let mod_mv = c.load_module(Ptx::from_src(MATVEC_PTX)).unwrap();
    let f_rms = mod_rms.load_function("rms_norm").unwrap();
    let f_mv = mod_mv.load_function("matvec").unwrap();

    let inp = stream.clone_htod(&input).unwrap();
    let wgt = stream.clone_htod(&weight).unwrap();
    let mat = stream.clone_htod(&matrix).unwrap();
    let mut tmp: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();
    let mut out_sep: CudaSlice<f32> = stream.alloc_zeros(m as usize).unwrap();

    let cfg_rms = LaunchConfig::for_num_elems(n);
    let block = k.max(64);
    let grid = m.div_ceil(block);
    let cfg_mv = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&f_rms)
            .arg(&inp)
            .arg(&mut tmp)
            .arg(&wgt)
            .arg(&n)
            .arg(&epsilon)
            .launch(cfg_rms)
    }
    .unwrap();
    unsafe {
        stream
            .launch_builder(&f_mv)
            .arg(&mat)
            .arg(&tmp)
            .arg(&mut out_sep)
            .arg(&m)
            .arg(&k)
            .launch(cfg_mv)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let sep = stream.clone_dtoh(&out_sep).unwrap();

    // Fused (single block, n must fit in SMEM handoff buffer <= 1024)
    let mod_f = c.load_module(Ptx::from_src(FUSED_RMS_NORM_MATVEC)).unwrap();
    let f_fused = mod_f.load_function("fused_rms_norm_matvec").unwrap();
    let mut out_fused: CudaSlice<f32> = stream.alloc_zeros(m as usize).unwrap();
    let cfg_fused = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&f_fused)
            .arg(&inp)
            .arg(&wgt)
            .arg(&n)
            .arg(&epsilon)
            .arg(&mat)
            .arg(&mut out_fused)
            .arg(&m)
            .arg(&k)
            .launch(cfg_fused)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let fused = stream.clone_dtoh(&out_fused).unwrap();

    assert_close(&sep, &fused, &format!("smem_fuse m={m} k={k}"), 1e-4);
}

#[test]
fn smem_fuse_16x32() {
    run_smem_fuse_at_size(16, 32);
}

#[test]
fn smem_fuse_32x64() {
    run_smem_fuse_at_size(32, 64);
}

#[test]
fn smem_fuse_64x128() {
    run_smem_fuse_at_size(64, 128);
}

#[test]
fn smem_fuse_128x256() {
    run_smem_fuse_at_size(128, 256);
}

#[test]
fn smem_fuse_64x512() {
    run_smem_fuse_at_size(64, 512);
}

// ── Determinism: run fused kernel 100 times, check all identical ──

#[test]
fn regfuse_deterministic() {
    let c = ctx();
    let stream = c.default_stream();
    let n = 256u32;
    let epsilon = 1e-5f32;
    let scale_val = 2.5f32;

    let input: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();

    let mod_f = c
        .load_module(Ptx::from_src(REGFUSED_RMS_NORM_SCALE))
        .unwrap();
    let f = mod_f.load_function("regfused_rms_norm_scale").unwrap();

    let inp = stream.clone_htod(&input).unwrap();
    let wgt = stream.clone_htod(&weight).unwrap();
    let cfg = LaunchConfig::for_num_elems(n);

    let mut reference: Option<Vec<f32>> = None;
    for run in 0..100 {
        let mut out: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&inp)
                .arg(&wgt)
                .arg(&n)
                .arg(&epsilon)
                .arg(&mut out)
                .arg(&scale_val)
                .launch(cfg)
        }
        .unwrap();
        stream.synchronize().unwrap();
        let result = stream.clone_dtoh(&out).unwrap();

        match &reference {
            None => reference = Some(result),
            Some(ref_val) => assert_close(ref_val, &result, &format!("determinism run {run}"), 0.0),
        }
    }
    println!("PASS: regfused kernel is deterministic across 100 runs");
}
