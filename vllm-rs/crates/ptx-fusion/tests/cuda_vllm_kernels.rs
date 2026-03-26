//! Tests for real vllm-cuda kernels compiled to PTX via build.rs.
//!
//! These kernels are the actual production kernels used in vllm-rs inference.
//! The build script compiles them from csrc/ to PTX at build time.
//!
//! Run with: cargo test -p ptx-fusion --features cuda --test cuda_vllm_kernels -- --nocapture

#![cfg(feature = "cuda")]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::assertions_on_constants)]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ptx_fusion_macros::analyze_kernel_as;
use std::sync::Arc;

// ── Compile-time protocol extraction from real vllm kernels ──

analyze_kernel_as!("kernels/vllm_rms_norm.ptx", VLLM_RMS_NORM);
analyze_kernel_as!("kernels/vllm_silu_mul.ptx", VLLM_SILU_MUL);

const VLLM_RMS_NORM_PTX: &str = include_str!("../kernels/vllm_rms_norm.ptx");
const VLLM_SILU_MUL_PTX: &str = include_str!("../kernels/vllm_silu_mul.ptx");

fn ctx() -> Arc<CudaContext> {
    CudaContext::new(0).unwrap()
}

// Mangled name for rms_norm_kernel<float>
const RMS_NORM_F32_MANGLED: &str = "_Z15rms_norm_kernelIfEvPT_PKS0_S3_fi";
// Mangled name for act_and_mul_kernel<silu, float>
const SILU_MUL_F32_MANGLED: &str = "_Z18act_and_mul_kernelIL5silu0EfEvPT0_PKS1_S4_i";

// ── Protocol extraction tests ──

#[test]
fn vllm_rms_norm_protocol() {
    // Parser extracts the first .entry (float specialization)
    println!("Kernel: {}", VLLM_RMS_NORM.name);
    println!("Params: {}", VLLM_RMS_NORM.params.len());
    println!("Global loads: {}", VLLM_RMS_NORM.global_loads.len());
    println!("Global stores: {}", VLLM_RMS_NORM.global_stores.len());
    println!("SMEM loads: {}", VLLM_RMS_NORM.smem_loads);
    println!("SMEM stores: {}", VLLM_RMS_NORM.smem_stores);
    println!("Barriers: {:?}", VLLM_RMS_NORM.barriers);

    // Should have 5 params: out, input, weight, epsilon, hidden_size
    assert_eq!(VLLM_RMS_NORM.params.len(), 5);

    // Should have global loads (input + weight, vectorized)
    assert!(
        VLLM_RMS_NORM.global_loads.len() >= 2,
        "expected at least 2 global loads, got {}",
        VLLM_RMS_NORM.global_loads.len()
    );

    // Should have global stores (output, vectorized)
    assert!(
        !VLLM_RMS_NORM.global_stores.is_empty(),
        "expected global stores"
    );

    // Should use shared memory (block_reduce_sum + s_inv_rms)
    assert!(VLLM_RMS_NORM.smem_loads > 0, "expected SMEM loads");
    assert!(VLLM_RMS_NORM.smem_stores > 0, "expected SMEM stores");

    // Should have barriers (syncthreads in block_reduce_sum)
    assert!(
        !VLLM_RMS_NORM.barriers.is_empty(),
        "expected barriers in rms_norm"
    );

    println!("PASS: vllm rms_norm protocol extraction");
}

#[test]
fn vllm_silu_mul_protocol() {
    println!("Kernel: {}", VLLM_SILU_MUL.name);
    println!("Params: {}", VLLM_SILU_MUL.params.len());
    println!("Global loads: {}", VLLM_SILU_MUL.global_loads.len());
    println!("Global stores: {}", VLLM_SILU_MUL.global_stores.len());

    // Should have 4 params: out, gate, up, d
    assert_eq!(VLLM_SILU_MUL.params.len(), 4);

    // Should have global loads (gate + up, vectorized)
    assert!(
        VLLM_SILU_MUL.global_loads.len() >= 2,
        "expected at least 2 global loads, got {}",
        VLLM_SILU_MUL.global_loads.len()
    );

    // Should have global stores (output)
    assert!(
        !VLLM_SILU_MUL.global_stores.is_empty(),
        "expected global stores"
    );

    // Should NOT use shared memory (purely elementwise)
    assert_eq!(VLLM_SILU_MUL.smem_loads, 0, "silu_mul should have no SMEM");
    assert_eq!(VLLM_SILU_MUL.smem_stores, 0);

    // No barriers
    assert!(
        VLLM_SILU_MUL.barriers.is_empty(),
        "silu_mul should have no barriers"
    );

    println!("PASS: vllm silu_mul protocol extraction");
}

// ── GPU correctness: run real vllm kernels ──

#[test]
fn vllm_rms_norm_gpu_correctness() {
    let c = ctx();
    let stream = c.default_stream();

    let hidden = 128; // must be divisible by 4 (VEC_SIZE for float)
    let rows = 4;
    let n = rows * hidden;
    let epsilon = 1e-5f32;

    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017 + 0.3).sin()).collect();
    let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.002).collect();

    let module = c.load_module(Ptx::from_src(VLLM_RMS_NORM_PTX)).unwrap();

    // Try to find the float specialization
    let func = module
        .load_function(RMS_NORM_F32_MANGLED)
        .expect("could not find rms_norm_kernel<float>");

    let inp = stream.clone_htod(&input).unwrap();
    let wgt = stream.clone_htod(&weight).unwrap();
    let mut out: CudaSlice<f32> = stream.alloc_zeros(n).unwrap();

    // One block per row, 256 threads per block
    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
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

    // CPU reference
    for row in 0..rows {
        let x = &input[row * hidden..(row + 1) * hidden];
        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv_rms = (ss + epsilon).sqrt().recip();

        for j in 0..hidden {
            let expected = x[j] * inv_rms * weight[j];
            let got = result[row * hidden + j];
            let diff = (expected - got).abs();
            assert!(
                diff < 1e-3,
                "row={row} col={j}: expected={expected}, got={got}, diff={diff}"
            );
        }
    }

    println!(
        "PASS: vllm rms_norm GPU output matches CPU reference ({rows} rows x {hidden} hidden)"
    );
}

#[test]
fn vllm_silu_mul_gpu_correctness() {
    let c = ctx();
    let stream = c.default_stream();

    let d = 128; // divisible by 4
    let rows = 4;
    let n = rows * d;

    let gate: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.031 - 1.0).sin()).collect();
    let up: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.019 + 0.5).cos()).collect();

    let module = c.load_module(Ptx::from_src(VLLM_SILU_MUL_PTX)).unwrap();

    // Find the mangled name — it may differ, let's try common patterns
    let func_name = find_silu_entry(&module);
    let func = module.load_function(&func_name).unwrap();

    let gate_gpu = stream.clone_htod(&gate).unwrap();
    let up_gpu = stream.clone_htod(&up).unwrap();
    let mut out: CudaSlice<f32> = stream.alloc_zeros(n).unwrap();

    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
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

    // CPU reference: out[i] = silu(gate[i]) * up[i]
    for i in 0..n {
        let g = gate[i];
        let silu = g / (1.0 + (-g).exp());
        let expected = silu * up[i];
        let got = result[i];
        let diff = (expected - got).abs();
        assert!(
            diff < 1e-3,
            "[{i}]: expected={expected}, got={got}, diff={diff}"
        );
    }

    println!("PASS: vllm silu_mul GPU output matches CPU reference ({rows} rows x {d} d)");
}

/// Find the silu entry point name in the module by trying common manglings.
fn find_silu_entry(module: &std::sync::Arc<cudarc::driver::CudaModule>) -> String {
    // The exact mangling depends on nvcc version. Try several patterns.
    let candidates = [
        SILU_MUL_F32_MANGLED,
        // nvcc might mangle the function pointer template arg differently
        "_Z18act_and_mul_kernelIXadL_Z4siluEEfEvPT0_PKS1_S4_i",
        "_Z18act_and_mul_kernelIPFffEfEvPT0_PKS2_S5_i",
    ];
    for name in &candidates {
        if module.load_function(name).is_ok() {
            return name.to_string();
        }
    }
    // Last resort: read the PTX to find the actual mangled name
    let ptx = VLLM_SILU_MUL_PTX;
    for line in ptx.lines() {
        let t = line.trim();
        if t.starts_with(".visible")
            && t.contains(".entry")
            && t.contains("act_and_mul")
            && let Some(start) = t.find("_Z")
        {
            let end = t.find('(').unwrap_or(t.len());
            let name = t[start..end].trim();
            if module.load_function(name).is_ok() {
                return name.to_string();
            }
        }
    }
    panic!("could not find silu entry point in PTX module");
}
