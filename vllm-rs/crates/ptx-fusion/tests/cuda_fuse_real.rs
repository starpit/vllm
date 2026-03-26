//! Test fusing real vllm-rs production kernels: rms_norm<float> → silu_mul<float>
//!
//! Run with: cargo test -p ptx-fusion --features cuda --test cuda_fuse_real -- --nocapture

#![cfg(feature = "cuda")]
#![allow(clippy::too_many_arguments)]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ptx_fusion_macros::{extract_entry, fuse_real_kernels};
use std::sync::Arc;

// Extract individual float entries for the "separate" baseline
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

// Fuse them: rms_norm output (param_0) → silu_mul gate input (param_1)
fuse_real_kernels!(
    "kernels/vllm_rms_norm.ptx",
    "rms_norm_kernelIfE",
    "kernels/vllm_silu_mul.ptx",
    "act_and_mul_kernelIXadL_Z4silufEEfE",
    "fused_rms_norm_silu_mul",
    "param_0", // rms_norm output
    "param_1", // silu_mul gate input
    4096,      // SMEM buffer elements (supports up to hidden=4096)
    FUSED_RMS_SILU_PTX
);

fn ctx() -> Arc<CudaContext> {
    CudaContext::new(0).unwrap()
}

fn find_entry_name(ptx: &str) -> String {
    for line in ptx.lines() {
        let t = line.trim();
        if t.starts_with(".visible")
            && t.contains(".entry")
            && let Some(start) = t.find("_Z").or_else(|| t.find("fused_"))
        {
            let end = t.find('(').unwrap_or(t.len());
            return t[start..end].trim().to_string();
        }
    }
    panic!("no entry found in PTX");
}

/// Run rms_norm then silu_mul as two separate launches.
fn run_separate(
    ctx: &Arc<CudaContext>,
    input: &[f32],
    weight: &[f32],
    up: &[f32],
    rows: usize,
    hidden: usize,
    epsilon: f32,
) -> Vec<f32> {
    let stream = ctx.default_stream();
    let n = rows * hidden;

    let mod_rms = ctx.load_module(Ptx::from_src(RMS_NORM_F32_PTX)).unwrap();
    let mod_silu = ctx.load_module(Ptx::from_src(SILU_MUL_F32_PTX)).unwrap();
    let f_rms = mod_rms
        .load_function(&find_entry_name(RMS_NORM_F32_PTX))
        .unwrap();
    let f_silu = mod_silu
        .load_function(&find_entry_name(SILU_MUL_F32_PTX))
        .unwrap();

    let inp = stream.clone_htod(input).unwrap();
    let wgt = stream.clone_htod(weight).unwrap();
    let up_gpu = stream.clone_htod(up).unwrap();
    let mut rms_out: CudaSlice<f32> = stream.alloc_zeros(n).unwrap();
    let mut final_out: CudaSlice<f32> = stream.alloc_zeros(n).unwrap();

    // rms_norm: one block per row
    let block = 256u32;
    let cfg_rms = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    // rms_norm params: out, input, weight, epsilon, hidden_size
    unsafe {
        stream
            .launch_builder(&f_rms)
            .arg(&mut rms_out)
            .arg(&inp)
            .arg(&wgt)
            .arg(&epsilon)
            .arg(&(hidden as i32))
            .launch(cfg_rms)
    }
    .unwrap();

    // silu_mul: one block per row
    // silu_mul params: out, gate, up, d
    let cfg_silu = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&f_silu)
            .arg(&mut final_out)
            .arg(&rms_out)
            .arg(&up_gpu)
            .arg(&(hidden as i32))
            .launch(cfg_silu)
    }
    .unwrap();

    stream.synchronize().unwrap();
    stream.clone_dtoh(&final_out).unwrap()
}

/// Run the fused kernel (single launch).
fn run_fused(
    ctx: &Arc<CudaContext>,
    input: &[f32],
    weight: &[f32],
    up: &[f32],
    rows: usize,
    hidden: usize,
    epsilon: f32,
) -> Vec<f32> {
    let stream = ctx.default_stream();
    let n = rows * hidden;

    // Dump fused PTX for debugging
    std::fs::write("/tmp/fused_rms_silu.ptx", FUSED_RMS_SILU_PTX).ok();

    let module = ctx
        .load_module(Ptx::from_src(FUSED_RMS_SILU_PTX))
        .expect("fused PTX should load");
    let func = module
        .load_function("fused_rms_norm_silu_mul")
        .expect("fused entry should exist");

    let inp = stream.clone_htod(input).unwrap();
    let wgt = stream.clone_htod(weight).unwrap();
    let up_gpu = stream.clone_htod(up).unwrap();
    let mut final_out: CudaSlice<f32> = stream.alloc_zeros(n).unwrap();

    // Fused params = rms_norm params (minus output) + silu_mul params (minus gate)
    // rms_norm: [out=removed], input, weight, epsilon, hidden_size
    // silu_mul: [out], [gate=removed], up, d
    // Merged: input, weight, epsilon, hidden_size, silu_out, up, d
    // But the exact order depends on merge_params — let me check what the macro produces.
    //
    // Actually, merge_params takes A's params minus output, then B's minus gate.
    // A (rms_norm) params: param_0(out), param_1(in), param_2(wgt), param_3(eps), param_4(hidden)
    // minus param_0 → param_1(in), param_2(wgt), param_3(eps), param_4(hidden)
    //
    // B (silu_mul) params: param_0(out), param_1(gate), param_2(up), param_3(d)
    // minus param_1 → param_0(out), param_2(up), param_3(d)
    //
    // Merged: param_1(in), param_2(wgt), param_3(eps), param_4(hidden),
    //         silu_param_0(out), silu_param_2(up), silu_param_3(d)

    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&inp) // rms input
            .arg(&wgt) // rms weight
            .arg(&epsilon) // rms epsilon
            .arg(&(hidden as i32)) // rms hidden_size
            .arg(&mut final_out) // silu output
            .arg(&up_gpu) // silu up
            .arg(&(hidden as i32)) // silu d
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    stream.clone_dtoh(&final_out).unwrap()
}

#[test]
fn fused_rms_silu_correctness() {
    let c = ctx();

    let rows = 4;
    let hidden = 128; // must be divisible by 4 (v4 loads)
    let epsilon = 1e-5f32;
    let n = rows * hidden;

    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017 + 0.3).sin()).collect();
    let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.002).collect();
    let up: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013 + 0.7).cos()).collect();

    let output_separate = run_separate(&c, &input, &weight, &up, rows, hidden, epsilon);
    let output_fused = run_fused(&c, &input, &weight, &up, rows, hidden, epsilon);

    println!("Separate: {:?}...", &output_separate[..4]);
    println!("Fused:    {:?}...", &output_fused[..4]);

    // Sanity
    let sum_sep: f32 = output_separate.iter().sum();
    assert!(sum_sep.abs() > 0.01, "separate output is zeros");
    let sum_fused: f32 = output_fused.iter().sum();
    assert!(sum_fused.abs() > 0.01, "fused output is zeros");

    // Compare
    let mut max_diff = 0.0f32;
    for (i, (a, b)) in output_separate.iter().zip(output_fused.iter()).enumerate() {
        let diff = (a - b).abs();
        max_diff = max_diff.max(diff);
        assert!(diff < 1e-3, "[{i}]: separate={a}, fused={b}, diff={diff}");
    }

    println!(
        "PASS: fused rms_norm+silu_mul matches separate ({rows} rows x {hidden} hidden, max_diff={max_diff:.2e})"
    );
}
