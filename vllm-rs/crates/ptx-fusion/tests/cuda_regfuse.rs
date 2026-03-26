//! CUDA test: register-level fusion (rms_norm -> scale, no SMEM).
//!
//! Run with: cargo test -p ptx-fusion --features cuda --test cuda_regfuse -- --nocapture

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ptx_fusion_macros::regfuse_kernels;
use std::sync::Arc;

const RMS_NORM_PTX: &str = include_str!("../kernels/rms_norm.ptx");
const SCALE_PTX: &str = include_str!("../kernels/scale.ptx");

// Register-fuse: rms_norm output feeds scale input, value stays in register
regfuse_kernels!(
    "kernels/rms_norm.ptx",
    "kernels/scale.ptx",
    "regfused_rms_norm_scale",
    "output",
    "input"
);

fn get_ctx() -> Arc<CudaContext> {
    CudaContext::new(0).expect("no CUDA device available")
}

fn assert_f32_close(a: &[f32], b: &[f32], label: &str, tol: f32) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff <= tol,
            "{label}: mismatch at [{i}]: separate={x}, fused={y}, diff={diff}"
        );
    }
}

/// Run rms_norm then scale as two separate launches.
fn run_separate(
    ctx: &Arc<CudaContext>,
    input: &[f32],
    weight: &[f32],
    n: u32,
    epsilon: f32,
    scale_val: f32,
) -> Vec<f32> {
    let stream = ctx.default_stream();

    let mod_rms = ctx.load_module(Ptx::from_src(RMS_NORM_PTX)).unwrap();
    let mod_scale = ctx.load_module(Ptx::from_src(SCALE_PTX)).unwrap();
    let func_rms = mod_rms.load_function("rms_norm").unwrap();
    let func_scale = mod_scale.load_function("scale").unwrap();

    let input_gpu = stream.clone_htod(input).unwrap();
    let weight_gpu = stream.clone_htod(weight).unwrap();
    let mut intermediate: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();
    let mut output: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();

    let cfg = LaunchConfig::for_num_elems(n);

    // rms_norm: input -> intermediate
    unsafe {
        stream
            .launch_builder(&func_rms)
            .arg(&input_gpu)
            .arg(&mut intermediate)
            .arg(&weight_gpu)
            .arg(&n)
            .arg(&epsilon)
            .launch(cfg)
    }
    .unwrap();

    // scale: intermediate -> output
    unsafe {
        stream
            .launch_builder(&func_scale)
            .arg(&intermediate)
            .arg(&mut output)
            .arg(&n)
            .arg(&scale_val)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    stream.clone_dtoh(&output).unwrap()
}

/// Run the register-fused kernel (single launch, no intermediate memory).
fn run_regfused(
    ctx: &Arc<CudaContext>,
    input: &[f32],
    weight: &[f32],
    n: u32,
    epsilon: f32,
    scale_val: f32,
) -> Vec<f32> {
    let stream = ctx.default_stream();

    let module = ctx
        .load_module(Ptx::from_src(REGFUSED_RMS_NORM_SCALE))
        .unwrap();
    let func = module.load_function("regfused_rms_norm_scale").unwrap();

    let input_gpu = stream.clone_htod(input).unwrap();
    let weight_gpu = stream.clone_htod(weight).unwrap();
    let mut output: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();

    let cfg = LaunchConfig::for_num_elems(n);

    // Fused params: input, weight, n, epsilon, output, n(scale), scale_val
    // Merged: input, weight, n, epsilon, output, scale_val
    // (n is shared between both kernels)
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&input_gpu)
            .arg(&weight_gpu)
            .arg(&n)
            .arg(&epsilon)
            .arg(&mut output)
            .arg(&scale_val)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    stream.clone_dtoh(&output).unwrap()
}

#[test]
fn regfused_correctness() {
    let ctx = get_ctx();
    let n = 256u32;
    let epsilon = 1e-5f32;
    let scale_val = 2.5f32;

    let input: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();

    let output_separate = run_separate(&ctx, &input, &weight, n, epsilon, scale_val);
    let output_fused = run_regfused(&ctx, &input, &weight, n, epsilon, scale_val);

    println!("Separate: {:?}...", &output_separate[..4]);
    println!("RegFused: {:?}...", &output_fused[..4]);

    let sum = output_separate.iter().sum::<f32>();
    assert!(sum.abs() > 1.0, "separate output is zeros (sum={sum})");
    let sum_f = output_fused.iter().sum::<f32>();
    assert!(sum_f.abs() > 1.0, "fused output is zeros (sum={sum_f})");

    assert_f32_close(&output_separate, &output_fused, "regfused_vs_separate", 0.0);
    println!("PASS: register-fused kernel matches separate launches (n={n}, bitwise identical)");
}

#[test]
fn regfused_benchmark() {
    let ctx = get_ctx();
    let stream = ctx.default_stream();
    let n = 256u32;
    let epsilon = 1e-5f32;
    let scale_val = 2.5f32;

    let input: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();

    let mod_rms = ctx.load_module(Ptx::from_src(RMS_NORM_PTX)).unwrap();
    let mod_scale = ctx.load_module(Ptx::from_src(SCALE_PTX)).unwrap();
    let mod_fused = ctx
        .load_module(Ptx::from_src(REGFUSED_RMS_NORM_SCALE))
        .unwrap();
    let func_rms = mod_rms.load_function("rms_norm").unwrap();
    let func_scale = mod_scale.load_function("scale").unwrap();
    let func_fused = mod_fused.load_function("regfused_rms_norm_scale").unwrap();

    let input_gpu = stream.clone_htod(&input).unwrap();
    let weight_gpu = stream.clone_htod(&weight).unwrap();
    let mut intermediate: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();
    let mut output_sep: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();
    let mut output_fused: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();

    let cfg = LaunchConfig::for_num_elems(n);
    let warmup = 100;
    let iters = 10_000;

    // Warmup
    for _ in 0..warmup {
        unsafe {
            stream
                .launch_builder(&func_rms)
                .arg(&input_gpu)
                .arg(&mut intermediate)
                .arg(&weight_gpu)
                .arg(&n)
                .arg(&epsilon)
                .launch(cfg)
        }
        .unwrap();
        unsafe {
            stream
                .launch_builder(&func_scale)
                .arg(&intermediate)
                .arg(&mut output_sep)
                .arg(&n)
                .arg(&scale_val)
                .launch(cfg)
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
                .arg(&mut output_fused)
                .arg(&scale_val)
                .launch(cfg)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();

    // Bench separate
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        unsafe {
            stream
                .launch_builder(&func_rms)
                .arg(&input_gpu)
                .arg(&mut intermediate)
                .arg(&weight_gpu)
                .arg(&n)
                .arg(&epsilon)
                .launch(cfg)
        }
        .unwrap();
        unsafe {
            stream
                .launch_builder(&func_scale)
                .arg(&intermediate)
                .arg(&mut output_sep)
                .arg(&n)
                .arg(&scale_val)
                .launch(cfg)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();
    let sep_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // Bench fused
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        unsafe {
            stream
                .launch_builder(&func_fused)
                .arg(&input_gpu)
                .arg(&weight_gpu)
                .arg(&n)
                .arg(&epsilon)
                .arg(&mut output_fused)
                .arg(&scale_val)
                .launch(cfg)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();
    let fused_us = t0.elapsed().as_micros() as f64 / iters as f64;

    let speedup = sep_us / fused_us;

    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║  Register Fusion Benchmark (n={n})               ║");
    println!("╠══════════════════════════════════════════════════╣");
    println!(
        "║  Separate (2 launches): {:8.2} us/iter          ║",
        sep_us
    );
    println!(
        "║  RegFused (1 launch):   {:8.2} us/iter          ║",
        fused_us
    );
    println!(
        "║  Speedup:               {:8.2}x                ║",
        speedup
    );
    println!(
        "║  Saved per iter:        {:8.2} us               ║",
        sep_us - fused_us
    );
    println!("║  No SMEM, no barrier, no intermediate GMEM       ║");
    println!("╚══════════════════════════════════════════════════╝\n");
}
