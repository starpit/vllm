//! Test single-entry extraction from multi-entry PTX files.
//!
//! Run with: cargo test -p ptx-fusion --features cuda --test cuda_extract -- --nocapture

#![cfg(feature = "cuda")]
#![allow(clippy::too_many_arguments)]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ptx_fusion_macros::{analyze_kernel_as, extract_entry};
use std::sync::Arc;

// Extract just the float specialization from the multi-entry PTX
extract_entry!(
    "kernels/vllm_rms_norm.ptx",
    "rms_norm_kernelIfE",
    RMS_NORM_F32_PTX
);

extract_entry!(
    "kernels/vllm_silu_mul.ptx",
    "act_and_mul_kernelIXadL_Z4silufEEfE",
    SILU_MUL_F32_PTX
);

// Analyze the extracted entries
analyze_kernel_as!("kernels/vllm_rms_norm.ptx", RMS_NORM_F32_PROTO);
analyze_kernel_as!("kernels/vllm_silu_mul.ptx", SILU_MUL_F32_PROTO);

fn ctx() -> Arc<CudaContext> {
    CudaContext::new(0).unwrap()
}

#[test]
fn extracted_rms_norm_is_valid_ptx() {
    // The extracted PTX should contain only the float entry
    assert!(
        RMS_NORM_F32_PTX.contains("rms_norm_kernelIfE"),
        "should contain float entry"
    );
    // Should NOT contain half or bf16 entries
    let entry_count = RMS_NORM_F32_PTX.matches(".visible .entry").count();
    assert_eq!(
        entry_count, 1,
        "should have exactly 1 entry, got {entry_count}"
    );

    println!(
        "Extracted rms_norm<float> PTX: {} lines, {} bytes",
        RMS_NORM_F32_PTX.lines().count(),
        RMS_NORM_F32_PTX.len()
    );
}

#[test]
fn extracted_silu_mul_is_valid_ptx() {
    assert!(
        SILU_MUL_F32_PTX.contains("act_and_mul_kernel"),
        "should contain silu entry"
    );
    let entry_count = SILU_MUL_F32_PTX.matches(".visible .entry").count();
    assert_eq!(
        entry_count, 1,
        "should have exactly 1 entry, got {entry_count}"
    );

    println!(
        "Extracted silu_mul<float> PTX: {} lines, {} bytes",
        SILU_MUL_F32_PTX.lines().count(),
        SILU_MUL_F32_PTX.len()
    );
}

#[test]
fn extracted_rms_norm_runs_on_gpu() {
    let c = ctx();
    let stream = c.default_stream();

    let hidden = 128;
    let rows = 4;
    let n = rows * hidden;
    let epsilon = 1e-5f32;

    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017 + 0.3).sin()).collect();
    let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.002).collect();

    // Load the extracted single-entry PTX
    let module = c
        .load_module(Ptx::from_src(RMS_NORM_F32_PTX))
        .expect("extracted PTX should load");

    // The entry name is the full mangled name from the original
    let func_name = RMS_NORM_F32_PTX
        .lines()
        .find(|l| l.contains(".entry"))
        .and_then(|l| {
            let start = l.find("_Z")?;
            let end = l.find('(')?;
            Some(l[start..end].trim())
        })
        .expect("should find entry name");

    let func = module.load_function(func_name).unwrap();

    let inp = stream.clone_htod(&input).unwrap();
    let wgt = stream.clone_htod(&weight).unwrap();
    let mut out: CudaSlice<f32> = stream.alloc_zeros(n).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&mut out)
            .arg(&inp)
            .arg(&wgt)
            .arg(&epsilon)
            .arg(&(hidden as i32))
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out).unwrap();

    // Verify against CPU
    for row in 0..rows {
        let x = &input[row * hidden..(row + 1) * hidden];
        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv_rms = (ss + epsilon).sqrt().recip();
        for j in 0..hidden {
            let expected = x[j] * inv_rms * weight[j];
            let diff = (result[row * hidden + j] - expected).abs();
            assert!(
                diff < 1e-3,
                "row={row} col={j}: got={}, expected={expected}, diff={diff}",
                result[row * hidden + j]
            );
        }
    }

    println!("PASS: extracted rms_norm<float> runs correctly on GPU");
}

#[test]
fn extracted_silu_mul_runs_on_gpu() {
    let c = ctx();
    let stream = c.default_stream();

    let d = 128;
    let rows = 4;
    let n = rows * d;

    let gate: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.031 - 1.0).sin()).collect();
    let up: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.019 + 0.5).cos()).collect();

    let module = c
        .load_module(Ptx::from_src(SILU_MUL_F32_PTX))
        .expect("extracted PTX should load");

    let func_name = SILU_MUL_F32_PTX
        .lines()
        .find(|l| l.contains(".entry"))
        .and_then(|l| {
            let start = l.find("_Z")?;
            let end = l.find('(')?;
            Some(l[start..end].trim())
        })
        .expect("should find entry name");

    let func = module.load_function(func_name).unwrap();

    let gate_gpu = stream.clone_htod(&gate).unwrap();
    let up_gpu = stream.clone_htod(&up).unwrap();
    let mut out: CudaSlice<f32> = stream.alloc_zeros(n).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&mut out)
            .arg(&gate_gpu)
            .arg(&up_gpu)
            .arg(&(d as i32))
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out).unwrap();

    for i in 0..n {
        let g = gate[i];
        let silu = g / (1.0 + (-g).exp());
        let expected = silu * up[i];
        let diff = (result[i] - expected).abs();
        assert!(
            diff < 1e-3,
            "[{i}]: got={}, expected={expected}, diff={diff}",
            result[i]
        );
    }

    println!("PASS: extracted silu_mul<float> runs correctly on GPU");
}
