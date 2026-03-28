//! Test the general `fuse!` macro: fuse two real kernels via perimeter analysis.
//!
//! Fuses rms_norm + silu_mul using the general `fuse!` macro (not the
//! special-purpose `fuse_real_kernels!`) and verifies the output matches running
//! them separately across a range of problem sizes and data patterns.
//!
//! Run: cargo test -p ptx-fusion --features cuda --test cuda_fuse_general -- --nocapture

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ptx_fusion_macros::extract_entry;
use std::sync::Arc;

// ── Extract individual entries for the "separate" baseline ──

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

// ── The general fuse! macro ──
// Fuse rms_norm → silu_mul: rms_norm's param_0 (output) feeds silu_mul's param_1 (gate input).
// The macro only sees perimeters — it doesn't know these are norm/silu.
ptx_fusion::fuse!(
    a = "kernels/vllm_rms_norm.ptx",
    b = "kernels/vllm_silu_mul.ptx",
    bind = { a.param_0 => b.param_1 },
    name = "general_fused_norm_silu",
    const = GENERAL_FUSED_PTX,
);

// ── Infrastructure ──

fn ctx() -> Arc<CudaContext> {
    CudaContext::new(0).unwrap()
}

fn find_entry_name(ptx: &str) -> String {
    for line in ptx.lines() {
        let t = line.trim();
        if t.contains(".entry") && t.contains('(') {
            let after_entry = t.split(".entry").nth(1).unwrap().trim();
            let end = after_entry.find('(').unwrap_or(after_entry.len());
            return after_entry[..end].trim().to_string();
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

    let block = 256u32;

    // rms_norm params: out, input, weight, epsilon, hidden_size
    unsafe {
        stream
            .launch_builder(&f_rms)
            .arg(&mut rms_out)
            .arg(&inp)
            .arg(&wgt)
            .arg(&epsilon)
            .arg(&(hidden as i32))
            .launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    }

    // silu_mul params: out, gate_input(=rms_out), up_input, d(=hidden), num_tokens
    unsafe {
        stream
            .launch_builder(&f_silu)
            .arg(&mut final_out)
            .arg(&rms_out)
            .arg(&up_gpu)
            .arg(&(hidden as i32))
            .arg(&(rows as i32))
            .launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    }

    stream.synchronize().unwrap();
    stream.clone_dtoh(&final_out).unwrap()
}

/// Run the general-fused kernel (single launch).
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

    let module = ctx.load_module(Ptx::from_src(GENERAL_FUSED_PTX)).unwrap();
    let func = module.load_function("general_fused_norm_silu").unwrap();

    let inp = stream.clone_htod(input).unwrap();
    let wgt = stream.clone_htod(weight).unwrap();
    let up_gpu = stream.clone_htod(up).unwrap();
    let mut out: CudaSlice<f32> = stream.alloc_zeros(n).unwrap();

    let block = 256u32;

    // Merged params (bound pair eliminated):
    // A = rms_norm: [param_0(out), param_1(in), param_2(wgt), eps, hidden_size]
    // B = silu_mul: [param_0(out), param_1(gate), param_2(up), d, num_tokens]
    // Remove A.param_0 and B.param_1:
    // → [A.param_1, A.param_2, A.eps, A.hidden, B.param_0, B.param_2, B.d, B.num_tokens]
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&inp) // A.param_1 (rms_norm input)
            .arg(&wgt) // A.param_2 (rms_norm weight)
            .arg(&epsilon) // A.epsilon
            .arg(&(hidden as i32)) // A.hidden_size
            .arg(&mut out) // B.param_0 (silu_mul output)
            .arg(&up_gpu) // B.param_2 (silu_mul up_input)
            .arg(&(hidden as i32)) // B.d
            .arg(&(rows as i32)) // B.num_tokens
            .launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    }

    stream.synchronize().unwrap();
    stream.clone_dtoh(&out).unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Generate test data with a specific seed-like offset for variety.
fn gen_data(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.001 + seed).sin()).collect()
}

fn gen_weight(hidden: usize, seed: f32) -> Vec<f32> {
    (0..hidden)
        .map(|i| ((i as f32) * 0.01 + seed).cos())
        .collect()
}

/// Run one test case: fused vs separate at (rows, hidden).
fn check(ctx: &Arc<CudaContext>, rows: usize, hidden: usize, epsilon: f32) -> f32 {
    let input = gen_data(rows * hidden, 0.3);
    let weight = gen_weight(hidden, 0.1);
    let up = gen_data(rows * hidden, 0.7);

    let separate = run_separate(ctx, &input, &weight, &up, rows, hidden, epsilon);
    let fused = run_fused(ctx, &input, &weight, &up, rows, hidden, epsilon);

    // Sanity: outputs are not all zeros
    let sum: f32 = separate.iter().map(|x| x.abs()).sum();
    assert!(
        sum > 0.01,
        "separate output is all zeros at rows={rows} hidden={hidden}"
    );
    let sum: f32 = fused.iter().map(|x| x.abs()).sum();
    assert!(
        sum > 0.01,
        "fused output is all zeros at rows={rows} hidden={hidden}"
    );

    max_abs_diff(&separate, &fused)
}

// ── Tests ──

#[test]
fn ptxas_valid() {
    println!("=== General fuse!: ptxas validation ===");

    let path = "/tmp/general_fused_norm_silu.ptx";
    std::fs::write(path, GENERAL_FUSED_PTX).unwrap();

    let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        for line in stderr.lines().take(20) {
            println!("  ptxas: {line}");
        }
        for (i, line) in GENERAL_FUSED_PTX.lines().enumerate().take(50) {
            println!("{:4}: {line}", i + 1);
        }
        panic!("ptxas FAILED on general fused PTX");
    }
    println!("PASS: general fused PTX passes ptxas");
}

#[test]
fn basic_correctness() {
    println!("=== General fuse!: basic correctness (4 x 128) ===");
    let ctx = ctx();
    let diff = check(&ctx, 4, 128, 1e-5);
    println!("  4 x 128: max_abs_diff = {diff:.2e}");
    assert!(diff < 1e-5, "diff too large: {diff:.2e}");
    println!("PASS");
}

#[test]
fn single_row() {
    println!("=== General fuse!: single row ===");
    let ctx = ctx();
    let diff = check(&ctx, 1, 128, 1e-5);
    println!("  1 x 128: max_abs_diff = {diff:.2e}");
    assert!(diff < 1e-5, "diff too large: {diff:.2e}");
    println!("PASS");
}

#[test]
fn many_rows() {
    println!("=== General fuse!: many rows ===");
    let ctx = ctx();
    for rows in [8, 32, 64, 256] {
        let diff = check(&ctx, rows, 128, 1e-5);
        println!("  {rows} x 128: max_abs_diff = {diff:.2e}");
        assert!(diff < 1e-5, "{rows} x 128: diff too large: {diff:.2e}");
    }
    println!("PASS");
}

#[test]
fn hidden_sizes() {
    println!("=== General fuse!: various hidden sizes ===");
    let ctx = ctx();
    // hidden must be divisible by 4 (v4 loads in rms_norm/silu_mul)
    for hidden in [64, 256, 512, 1024, 2048] {
        let diff = check(&ctx, 4, hidden, 1e-5);
        println!("  4 x {hidden}: max_abs_diff = {diff:.2e}");
        assert!(diff < 1e-5, "4 x {hidden}: diff too large: {diff:.2e}");
    }
    println!("PASS");
}

#[test]
fn production_dims() {
    println!("=== General fuse!: production dimensions ===");
    let ctx = ctx();
    // Qwen2.5-3B: hidden=2560, intermediate=6912
    // These are the sizes that matter for the forward pass.
    let cases = [
        (1, 2560, "decode bs=1"),
        (4, 2560, "decode bs=4"),
        (32, 2560, "decode bs=32"),
        (1, 3456, "down_proj hidden"),
    ];
    for (rows, hidden, label) in cases {
        let diff = check(&ctx, rows, hidden, 1e-5);
        println!("  {label} ({rows} x {hidden}): max_abs_diff = {diff:.2e}");
        assert!(diff < 1e-5, "{label}: diff too large: {diff:.2e}");
    }
    println!("PASS");
}

#[test]
fn epsilon_values() {
    println!("=== General fuse!: different epsilon values ===");
    let ctx = ctx();
    for eps in [1e-5f32, 1e-6, 1e-8, 1e-3] {
        let diff = check(&ctx, 4, 256, eps);
        println!("  eps={eps:.0e}: max_abs_diff = {diff:.2e}");
        assert!(diff < 1e-5, "eps={eps:.0e}: diff too large: {diff:.2e}");
    }
    println!("PASS");
}

#[test]
fn determinism() {
    println!("=== General fuse!: determinism (10 runs) ===");
    let ctx = ctx();
    let rows = 8;
    let hidden = 256;
    let epsilon = 1e-5f32;
    let input = gen_data(rows * hidden, 0.3);
    let weight = gen_weight(hidden, 0.1);
    let up = gen_data(rows * hidden, 0.7);

    let reference = run_fused(&ctx, &input, &weight, &up, rows, hidden, epsilon);
    for run in 0..10 {
        let result = run_fused(&ctx, &input, &weight, &up, rows, hidden, epsilon);
        let diff = max_abs_diff(&reference, &result);
        assert!(diff == 0.0, "run {run}: not deterministic, diff={diff:.2e}");
    }
    println!("  10 runs: all bitwise identical");
    println!("PASS");
}

#[test]
fn large_values() {
    println!("=== General fuse!: large input values ===");
    let ctx = ctx();
    let rows = 4;
    let hidden = 128;
    // Input with large magnitudes to stress numerical precision
    let input: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.1).sin() * 100.0)
        .collect();
    let weight = gen_weight(hidden, 0.1);
    let up: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.07).cos() * 50.0)
        .collect();

    let separate = run_separate(&ctx, &input, &weight, &up, rows, hidden, 1e-5);
    let fused = run_fused(&ctx, &input, &weight, &up, rows, hidden, 1e-5);
    let diff = max_abs_diff(&separate, &fused);
    println!("  large values: max_abs_diff = {diff:.2e}");
    assert!(diff < 1e-3, "large values: diff too large: {diff:.2e}");
    println!("PASS");
}

#[test]
fn near_zero_input() {
    println!("=== General fuse!: near-zero input ===");
    let ctx = ctx();
    let rows = 4;
    let hidden = 128;
    // Very small inputs — tests epsilon stability
    let input: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.001).sin() * 1e-6)
        .collect();
    let weight: Vec<f32> = vec![1.0; hidden];
    let up: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.002 + 0.3).sin())
        .collect();

    let separate = run_separate(&ctx, &input, &weight, &up, rows, hidden, 1e-5);
    let fused = run_fused(&ctx, &input, &weight, &up, rows, hidden, 1e-5);
    let diff = max_abs_diff(&separate, &fused);
    println!("  near-zero: max_abs_diff = {diff:.2e}");
    assert!(diff < 1e-5, "near-zero: diff too large: {diff:.2e}");
    println!("PASS");
}

// ══════════════════════════════════════════════════════════════════════
// Register handoff tests: hand-written elementwise kernels (rms_norm + scale)
// The fuse! macro should auto-detect both are elementwise and use register
// handoff instead of SMEM — zero memory traffic for the intermediate.
// ══════════════════════════════════════════════════════════════════════

const RMS_NORM_HAND_PTX: &str = include_str!("../kernels/rms_norm.ptx");
const SCALE_HAND_PTX: &str = include_str!("../kernels/scale.ptx");

// General fuse! on hand-written elementwise kernels → should select register handoff
ptx_fusion::fuse!(
    a = "kernels/rms_norm.ptx",
    b = "kernels/scale.ptx",
    bind = { a.output => b.input },
    name = "general_regfused_norm_scale",
    const = REGFUSED_PTX,
);

/// Run rms_norm then scale as two separate launches.
fn run_separate_reg(
    ctx: &Arc<CudaContext>,
    input: &[f32],
    weight: &[f32],
    n: u32,
    epsilon: f32,
    scale_val: f32,
) -> Vec<f32> {
    let stream = ctx.default_stream();

    let mod_rms = ctx.load_module(Ptx::from_src(RMS_NORM_HAND_PTX)).unwrap();
    let mod_scale = ctx.load_module(Ptx::from_src(SCALE_HAND_PTX)).unwrap();
    let f_rms = mod_rms.load_function("rms_norm").unwrap();
    let f_scale = mod_scale.load_function("scale").unwrap();

    let inp = stream.clone_htod(input).unwrap();
    let wgt = stream.clone_htod(weight).unwrap();
    let mut rms_out: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();
    let mut scale_out: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (n, 1, 1),
        shared_mem_bytes: 0,
    };

    // rms_norm params: input, output, weight, n, epsilon
    unsafe {
        stream
            .launch_builder(&f_rms)
            .arg(&inp)
            .arg(&mut rms_out)
            .arg(&wgt)
            .arg(&n)
            .arg(&epsilon)
            .launch(cfg)
            .unwrap();
    }

    // scale params: input, output, n, scale_val
    unsafe {
        stream
            .launch_builder(&f_scale)
            .arg(&rms_out)
            .arg(&mut scale_out)
            .arg(&n)
            .arg(&scale_val)
            .launch(cfg)
            .unwrap();
    }

    stream.synchronize().unwrap();
    stream.clone_dtoh(&scale_out).unwrap()
}

/// Run the general-fused kernel (register handoff).
fn run_fused_reg(
    ctx: &Arc<CudaContext>,
    input: &[f32],
    weight: &[f32],
    n: u32,
    epsilon: f32,
    scale_val: f32,
) -> Vec<f32> {
    let stream = ctx.default_stream();

    let module = ctx.load_module(Ptx::from_src(REGFUSED_PTX)).unwrap();
    let func = module.load_function("general_regfused_norm_scale").unwrap();

    let inp = stream.clone_htod(input).unwrap();
    let wgt = stream.clone_htod(weight).unwrap();
    let mut out: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (n, 1, 1),
        shared_mem_bytes: 0,
    };

    // Merged params (bound pair eliminated):
    // A = rms_norm: [input, output(bound), weight, n, epsilon]
    // B = scale:    [input(bound), output, n, scale_val]
    // After removing A.output and B.input:
    // → [A.input, A.weight, A.n, A.epsilon, B.output, B.scale_val]
    // Note: B.n is deduped (same name as A.n)
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&inp) // A.input
            .arg(&wgt) // A.weight
            .arg(&n) // A.n (shared with B.n)
            .arg(&epsilon) // A.epsilon
            .arg(&mut out) // B.output
            .arg(&scale_val) // B.scale_val
            .launch(cfg)
            .unwrap();
    }

    stream.synchronize().unwrap();
    stream.clone_dtoh(&out).unwrap()
}

#[test]
fn regfuse_ptxas_valid() {
    println!("=== Register handoff: ptxas validation ===");

    // Verify it's actually using register handoff (not SMEM)
    assert!(
        REGFUSED_PTX.contains("register handoff"),
        "should use register handoff for elementwise pair"
    );
    assert!(
        !REGFUSED_PTX.contains("st.shared"),
        "should NOT use SMEM for elementwise pair"
    );

    let path = "/tmp/general_regfused_norm_scale.ptx";
    std::fs::write(path, REGFUSED_PTX).unwrap();

    let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        for line in stderr.lines().take(20) {
            println!("  ptxas: {line}");
        }
        for (i, line) in REGFUSED_PTX.lines().enumerate() {
            println!("{:4}: {line}", i + 1);
        }
        panic!("ptxas FAILED on register-fused PTX");
    }
    println!("PASS: register-fused PTX passes ptxas");
}

#[test]
fn regfuse_basic() {
    println!("=== Register handoff: basic correctness ===");
    let ctx = ctx();
    let n = 128u32;
    let epsilon = 1e-5f32;
    let scale_val = 2.5f32;
    let input: Vec<f32> = (0..n as usize)
        .map(|i| ((i as f32) * 0.017 + 0.3).sin())
        .collect();
    let weight: Vec<f32> = (0..n as usize).map(|i| 1.0 + (i as f32) * 0.002).collect();

    let separate = run_separate_reg(&ctx, &input, &weight, n, epsilon, scale_val);
    let fused = run_fused_reg(&ctx, &input, &weight, n, epsilon, scale_val);
    let diff = max_abs_diff(&separate, &fused);
    println!("  n={n}, scale={scale_val}: max_abs_diff = {diff:.2e}");
    assert!(diff < 1e-5, "regfuse diff too large: {diff:.2e}");
    println!("PASS");
}

#[test]
fn regfuse_sizes() {
    println!("=== Register handoff: size sweep ===");
    let ctx = ctx();
    let epsilon = 1e-5f32;
    let scale_val = 0.7f32;

    for n in [32u32, 64, 128, 256, 512, 1024] {
        let input: Vec<f32> = (0..n as usize)
            .map(|i| ((i as f32) * 0.013 + 0.5).sin())
            .collect();
        let weight: Vec<f32> = (0..n as usize)
            .map(|i| ((i as f32) * 0.007 + 0.1).cos())
            .collect();

        let separate = run_separate_reg(&ctx, &input, &weight, n, epsilon, scale_val);
        let fused = run_fused_reg(&ctx, &input, &weight, n, epsilon, scale_val);
        let diff = max_abs_diff(&separate, &fused);
        println!("  n={n}: max_abs_diff = {diff:.2e}");
        assert!(diff < 1e-5, "n={n}: regfuse diff too large: {diff:.2e}");
    }
    println!("PASS");
}

#[test]
fn regfuse_determinism() {
    println!("=== Register handoff: determinism ===");
    let ctx = ctx();
    let n = 256u32;
    let input: Vec<f32> = (0..n as usize)
        .map(|i| ((i as f32) * 0.017 + 0.3).sin())
        .collect();
    let weight: Vec<f32> = (0..n as usize).map(|i| 1.0 + (i as f32) * 0.002).collect();

    let reference = run_fused_reg(&ctx, &input, &weight, n, 1e-5, 1.5);
    for run in 0..10 {
        let result = run_fused_reg(&ctx, &input, &weight, n, 1e-5, 1.5);
        let diff = max_abs_diff(&reference, &result);
        assert!(diff == 0.0, "run {run}: not deterministic, diff={diff:.2e}");
    }
    println!("  10 runs: all bitwise identical");
    println!("PASS");
}

#[test]
fn regfuse_matches_special_purpose() {
    println!("=== Register handoff: general fuse! vs special-purpose regfuse_kernels! ===");
    // Compare our general fuse! output against the proven regfuse_kernels! macro
    ptx_fusion_macros::regfuse_kernels!(
        "kernels/rms_norm.ptx",
        "kernels/scale.ptx",
        "special_regfused",
        "output",
        "input"
    );

    let ctx = ctx();
    let n = 256u32;
    let input: Vec<f32> = (0..n as usize)
        .map(|i| ((i as f32) * 0.017 + 0.3).sin())
        .collect();
    let weight: Vec<f32> = (0..n as usize).map(|i| 1.0 + (i as f32) * 0.002).collect();
    let epsilon = 1e-5f32;
    let scale_val = 2.0f32;

    let stream = ctx.default_stream();
    let inp = stream.clone_htod(&input).unwrap();
    let wgt = stream.clone_htod(&weight).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (n, 1, 1),
        shared_mem_bytes: 0,
    };

    // Run the special-purpose regfused kernel
    let mod_special = ctx.load_module(Ptx::from_src(SPECIAL_REGFUSED)).unwrap();
    let f_special = mod_special.load_function("special_regfused").unwrap();
    let mut out_special: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();
    // Special-purpose params: input, weight, n, epsilon, output, scale_val
    unsafe {
        stream
            .launch_builder(&f_special)
            .arg(&inp)
            .arg(&wgt)
            .arg(&n)
            .arg(&epsilon)
            .arg(&mut out_special)
            .arg(&scale_val)
            .launch(cfg)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let result_special = stream.clone_dtoh(&out_special).unwrap();

    // Run the general fuse! kernel
    let result_general = run_fused_reg(&ctx, &input, &weight, n, epsilon, scale_val);

    let diff = max_abs_diff(&result_special, &result_general);
    println!("  general vs special-purpose: max_abs_diff = {diff:.2e}");
    assert!(
        diff == 0.0,
        "general and special-purpose should be bitwise identical, got {diff:.2e}"
    );
    println!("PASS");
}

// ══════════════════════════════════════════════════════════════════════
// GEMM epilogue injection tests: CUTLASS GEMM + pointwise consumer
// fuse! detects the GEMM from its MMA instructions and injects the
// consumer's computation into the epilogue before bf16 conversion.
// ══════════════════════════════════════════════════════════════════════

// Fuse flat-param CUTLASS GEMM + scale via general fuse!
// The GEMM output (ptr_D / ferrite_params) feeds scale's input.
// scale.ptx has params: (input, output, n, scale_val)
// The GEMM's output is traced to the D pointer via st.global.
//
// We bind GEMM.param_0 (A ptr traced to output stores) -- wait, the flat-param
// GEMM's output stores trace to the D pointer at ferrite_params offset 24.
// But find_param_by_substring matches on param names. The flat-param GEMM has
// a single param "ferrite_params", so we can't disambiguate A/B/C/D by name.
//
// For now, test with the ORIGINAL (non-flat-param) CUTLASS PTX which has named params.
// TODO: support flat-param GEMM binding by offset.

ptx_fusion::fuse!(
    a = "kernels/cutlass_gemm_bf16_sm89.ptx",
    b = "kernels/scale.ptx",
    bind = { a.param_0 => b.input },
    name = "gemm_scale_fused",
    const = GEMM_SCALE_PTX,
);

#[test]
fn gemm_epilogue_ptxas_valid() {
    println!("=== GEMM epilogue injection: ptxas validation ===");

    // Verify the macro chose GemmEpilogue
    assert!(
        GEMM_SCALE_PTX.contains("FERRITE: inject epilogue"),
        "should use epilogue injection for GEMM -> elementwise"
    );

    let path = "/tmp/gemm_scale_fused.ptx";
    std::fs::write(path, GEMM_SCALE_PTX).unwrap();

    let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        for line in stderr.lines().take(20) {
            println!("  ptxas: {line}");
        }
        // Print first/last 20 lines of fused PTX
        let ptx_lines: Vec<&str> = GEMM_SCALE_PTX.lines().collect();
        println!("--- First 20 lines ---");
        for (i, line) in ptx_lines.iter().enumerate().take(20) {
            println!("{:4}: {line}", i + 1);
        }
        if ptx_lines.len() > 40 {
            println!("--- Last 20 lines ---");
            for (i, line) in ptx_lines.iter().enumerate().skip(ptx_lines.len() - 20) {
                println!("{:4}: {line}", i + 1);
            }
        }
        panic!("ptxas FAILED on GEMM+scale fused PTX");
    }
    println!("PASS: GEMM+scale fused PTX passes ptxas");
}

// ══════════════════════════════════════════════════════════════════════
// GEMM prologue GPU correctness: identity fn at A-load sites
// The identity prologue replaces cp.async with explicit ld->unpack->repack->st.
// The bf16->f32->bf16 round-trip is lossless for bf16 values, so the output
// should be identical to the original flat-param GEMM.
// ══════════════════════════════════════════════════════════════════════

// Baseline: unmodified flat-param CUTLASS GEMM
ptx_fusion::replace_perimeter_macro!(
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "flat_gemm_base",
    FLAT_GEMM_BASE_PTX
);

// Identity prologue + flat-param: chains prologue_identity then replace_perimeter
ptx_fusion::prologue_identity_flat!(
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "flat_gemm_prologue_id",
    FLAT_GEMM_PROLOGUE_ID_PTX
);

fn build_flat_params(
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    d_ptr: u64,
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    ldd: u32,
    alpha: f32,
    beta: f32,
) -> [u8; 88] {
    let mut p = [0u8; 88];
    p[0..8].copy_from_slice(&a_ptr.to_le_bytes());
    p[8..16].copy_from_slice(&b_ptr.to_le_bytes());
    p[16..24].copy_from_slice(&c_ptr.to_le_bytes());
    p[24..32].copy_from_slice(&d_ptr.to_le_bytes());
    p[32..40].copy_from_slice(&(lda as u64).to_le_bytes());
    p[40..48].copy_from_slice(&(ldb as u64).to_le_bytes());
    p[48..56].copy_from_slice(&(ldc as u64).to_le_bytes());
    p[56..64].copy_from_slice(&(ldd as u64).to_le_bytes());
    p[64..68].copy_from_slice(&(m as i32).to_le_bytes());
    p[68..72].copy_from_slice(&(n as i32).to_le_bytes());
    p[72..76].copy_from_slice(&(k as i32).to_le_bytes());
    p[76..80].copy_from_slice(&alpha.to_le_bytes());
    p[80..84].copy_from_slice(&beta.to_le_bytes());
    p
}

fn compute_grid(m: u32, n: u32, tile_m: u32, tile_n: u32) -> (u32, u32, u32) {
    let grid_m = m.div_ceil(tile_m);
    let grid_n = n.div_ceil(tile_n);
    const SWIZZLE_N: i32 = 4;
    let swizzle_log = if SWIZZLE_N >= 8 && grid_n >= 6 {
        3
    } else if SWIZZLE_N >= 4 && grid_n >= 3 {
        2
    } else if SWIZZLE_N >= 2 && grid_n >= 2 {
        1
    } else {
        0
    };
    let tile = 1u32 << swizzle_log;
    (grid_m * tile, grid_n.div_ceil(tile), 1)
}

/// Run a flat-param CUTLASS GEMM and return the output.
fn run_flat_gemm(
    ctx: &Arc<CudaContext>,
    ptx: &str,
    entry: &str,
    h_a: &[half::bf16],
    h_b: &[half::bf16],
    m: u32,
    n: u32,
    k: u32,
) -> Vec<half::bf16> {
    let stream = ctx.default_stream();
    let d_a = stream.clone_htod(h_a).unwrap();
    let d_b = stream.clone_htod(h_b).unwrap();
    let d_c: CudaSlice<half::bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    let mut d_d: CudaSlice<half::bf16> = stream.alloc_zeros((m * n) as usize).unwrap();

    let module = ctx.load_module(Ptx::from_src(ptx)).unwrap();
    let func = module.load_function(entry).unwrap();

    let (a_ptr, _) = d_a.device_ptr(&stream);
    let (b_ptr, _) = d_b.device_ptr(&stream);
    let (c_ptr, _) = d_c.device_ptr(&stream);
    let (d_ptr, _) = d_d.device_ptr(&stream);

    let params = build_flat_params(
        a_ptr as u64,
        b_ptr as u64,
        c_ptr as u64,
        d_ptr as u64,
        m,
        n,
        k,
        k,
        k,
        n,
        n,
        1.0,
        0.0,
    );

    let (gx, gy, gz) = compute_grid(m, n, 64, 128);
    let cfg = LaunchConfig {
        grid_dim: (gx, gy, gz),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 36864,
    };

    unsafe { stream.launch_builder(&func).arg(&params).launch(cfg) }.unwrap();
    stream.synchronize().unwrap();
    stream.clone_dtoh(&d_d).unwrap()
}

#[test]
fn prologue_identity_gpu_correctness() {
    println!("=== GEMM prologue identity: GPU correctness ===");

    let ctx = ctx();

    // Test at multiple sizes
    let cases: &[(u32, u32, u32)] = &[
        (64, 128, 32),  // single tile
        (128, 256, 64), // multi-tile
        (1, 128, 32),   // decode bs=1
        (32, 128, 128), // larger K
    ];

    for &(m, n, k) in cases {
        let h_a: Vec<half::bf16> = (0..(m * k) as usize)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
            .collect();
        let h_b: Vec<half::bf16> = (0..(n * k) as usize)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.00023 + 0.3).cos() * 0.1))
            .collect();

        let out_base = run_flat_gemm(
            &ctx,
            FLAT_GEMM_BASE_PTX,
            "flat_gemm_base",
            &h_a,
            &h_b,
            m,
            n,
            k,
        );
        let out_prologue = run_flat_gemm(
            &ctx,
            FLAT_GEMM_PROLOGUE_ID_PTX,
            "flat_gemm_prologue_id",
            &h_a,
            &h_b,
            m,
            n,
            k,
        );

        let mut max_diff = 0.0f32;
        for (i, (a, b)) in out_base.iter().zip(out_prologue.iter()).enumerate() {
            let diff = (a.to_f32() - b.to_f32()).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 0.01 {
                panic!(
                    "M={m} N={n} K={k}: mismatch at [{i}]: base={}, prologue={}, diff={diff:.2e}",
                    a.to_f32(),
                    b.to_f32()
                );
            }
        }
        println!("  M={m}, N={n}, K={k}: max_abs_diff = {max_diff:.2e}");
    }
    println!("PASS: prologue identity matches base GEMM");
}

// ── Prologue scale-by-2: GPU correctness ──
// GEMM(A*2, B) should equal 2 * GEMM(A, B) since GEMM is linear in A.

ptx_fusion::prologue_scale2_flat!(
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "flat_gemm_prologue_s2",
    FLAT_GEMM_PROLOGUE_S2_PTX
);

#[test]
fn prologue_scale2_gpu_correctness() {
    println!("=== GEMM prologue scale*2: GPU correctness ===");

    let ctx = ctx();

    let cases: &[(u32, u32, u32)] = &[(64, 128, 32), (128, 256, 64), (1, 128, 32), (32, 128, 128)];

    for &(m, n, k) in cases {
        let h_a: Vec<half::bf16> = (0..(m * k) as usize)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
            .collect();
        let h_b: Vec<half::bf16> = (0..(n * k) as usize)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.00023 + 0.3).cos() * 0.1))
            .collect();

        // Baseline: GEMM(A, B)
        let out_base = run_flat_gemm(
            &ctx,
            FLAT_GEMM_BASE_PTX,
            "flat_gemm_base",
            &h_a,
            &h_b,
            m,
            n,
            k,
        );
        // Prologue scale*2: GEMM(A*2, B)
        let out_scaled = run_flat_gemm(
            &ctx,
            FLAT_GEMM_PROLOGUE_S2_PTX,
            "flat_gemm_prologue_s2",
            &h_a,
            &h_b,
            m,
            n,
            k,
        );

        // Expect: out_scaled[i] ≈ 2 * out_base[i]
        let mut max_diff = 0.0f32;
        for (i, (base, scaled)) in out_base.iter().zip(out_scaled.iter()).enumerate() {
            let expected = base.to_f32() * 2.0;
            let actual = scaled.to_f32();
            let diff = (actual - expected).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 0.1 && expected.abs() > 1e-6 {
                panic!(
                    "M={m} N={n} K={k}: [{i}] expected {expected:.4}, got {actual:.4}, diff={diff:.2e}"
                );
            }
        }
        println!("  M={m}, N={n}, K={k}: max_abs_diff(scaled - 2*base) = {max_diff:.2e}");
    }
    println!("PASS: prologue scale*2 matches 2 * base GEMM");
}

// ══════════════════════════════════════════════════════════════════════
// rms_norm + GEMM fusion via intrinsic: GPU correctness
// The fused kernel normalizes the A-matrix inline at each cp.async site.
// Compare against: rms_norm(input, weight, eps) on CPU, then GEMM(normalized, B).
// ══════════════════════════════════════════════════════════════════════

ptx_fusion::fuse_rms_norm_gemm_flat!(
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "fused_rms_norm_gemm",
    FUSED_RMS_NORM_GEMM_PTX
);

/// CPU rms_norm reference: normalize each row of A by its RMS.
fn cpu_rms_norm(
    input: &[half::bf16],
    weight: &[half::bf16],
    m: usize,
    k: usize,
    eps: f32,
) -> Vec<half::bf16> {
    let mut out = vec![half::bf16::ZERO; m * k];
    for row in 0..m {
        let mut sum_sq = 0.0f32;
        for col in 0..k {
            let v = input[row * k + col].to_f32();
            sum_sq += v * v;
        }
        let inv_rms = (sum_sq / k as f32 + eps).sqrt().recip();
        for col in 0..k {
            let v = input[row * k + col].to_f32();
            let w = weight[col].to_f32();
            out[row * k + col] = half::bf16::from_f32(v * w * inv_rms);
        }
    }
    out
}

#[test]
fn rms_norm_gemm_gpu_correctness() {
    println!("=== rms_norm + GEMM intrinsic: GPU correctness ===");

    // Verify the fused PTX has rms_norm markers
    assert!(
        FUSED_RMS_NORM_GEMM_PTX.contains("FERRITE: rms_norm prologue"),
        "should have rms_norm prologue marker"
    );
    assert!(
        FUSED_RMS_NORM_GEMM_PTX.contains("ferrite_params[88]"),
        "should have flat params"
    );

    // ptxas check first
    let path = "/tmp/fused_rms_norm_gemm.ptx";
    std::fs::write(path, FUSED_RMS_NORM_GEMM_PTX).unwrap();
    let ptxas_out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");
    if !ptxas_out.status.success() {
        let stderr = String::from_utf8_lossy(&ptxas_out.stderr);
        for line in stderr.lines().take(20) {
            println!("  ptxas: {line}");
        }
        panic!("ptxas FAILED on fused rms_norm+GEMM PTX");
    }
    println!("  ptxas: PASS");

    let ctx = ctx();
    let m = 64u32;
    let k = 128u32; // hidden_size = K for rms_norm
    let n = 128u32;
    let eps = 1e-5f32;

    // Generate test data
    let h_input: Vec<half::bf16> = (0..(m * k) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.3))
        .collect();
    let h_weight: Vec<half::bf16> = (0..k as usize)
        .map(|i| half::bf16::from_f32(1.0 + (i as f32) * 0.001))
        .collect();
    let h_b: Vec<half::bf16> = (0..(n * k) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00023 + 0.3).cos() * 0.1))
        .collect();

    // Reference: CPU rms_norm then GPU GEMM
    let h_normed = cpu_rms_norm(&h_input, &h_weight, m as usize, k as usize, eps);
    let ref_out = run_flat_gemm(
        &ctx,
        FLAT_GEMM_BASE_PTX,
        "flat_gemm_base",
        &h_normed,
        &h_b,
        m,
        n,
        k,
    );

    // Fused: rms_norm + GEMM in one kernel
    // Params: _ferrite_rms_weight (u64), _ferrite_rms_epsilon (f32),
    //         _ferrite_rms_hidden (u32), ferrite_params[88]
    // A_ptr in ferrite_params = input_ptr (raw, not normalized)
    let stream = ctx.default_stream();
    let d_input = stream.clone_htod(&h_input).unwrap();
    let d_weight = stream.clone_htod(&h_weight).unwrap();
    let d_b = stream.clone_htod(&h_b).unwrap();
    let d_c: CudaSlice<half::bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    let mut d_d: CudaSlice<half::bf16> = stream.alloc_zeros((m * n) as usize).unwrap();

    let module = ctx
        .load_module(Ptx::from_src(FUSED_RMS_NORM_GEMM_PTX))
        .unwrap();
    let func = module.load_function("fused_rms_norm_gemm").unwrap();

    let (input_ptr, _) = d_input.device_ptr(&stream);
    let (weight_ptr, _) = d_weight.device_ptr(&stream);
    let (b_ptr, _) = d_b.device_ptr(&stream);
    let (c_ptr, _) = d_c.device_ptr(&stream);
    let (d_ptr, _) = d_d.device_ptr(&stream);

    let gemm_params = build_flat_params(
        input_ptr as u64, // A_ptr = raw input (norm happens inline)
        b_ptr as u64,
        c_ptr as u64,
        d_ptr as u64,
        m,
        n,
        k,
        k,
        k,
        n,
        n,
        1.0,
        0.0,
    );

    let weight_ptr_val = weight_ptr as u64;
    let hidden_val = k;
    let a_ptr_val = input_ptr as u64;
    let a_stride_val = k as u64; // stride in elements (= K for row-major A)

    let (gx, gy, gz) = compute_grid(m, n, 64, 128);
    let cfg = LaunchConfig {
        grid_dim: (gx, gy, gz),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 36864,
    };

    // 6 separate params: weight_ptr, epsilon, hidden, a_ptr, a_stride, ferrite_params[88]
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&weight_ptr_val)
            .arg(&eps)
            .arg(&hidden_val)
            .arg(&a_ptr_val)
            .arg(&a_stride_val)
            .arg(&gemm_params)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let fused_out = stream.clone_dtoh(&d_d).unwrap();

    // Compare
    let mut max_diff = 0.0f32;
    for (i, (r, f)) in ref_out.iter().zip(fused_out.iter()).enumerate() {
        let diff = (r.to_f32() - f.to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0.5 && r.to_f32().abs() > 1e-4 {
            println!(
                "  MISMATCH [{i}]: ref={:.4}, fused={:.4}, diff={diff:.2e}",
                r.to_f32(),
                f.to_f32()
            );
            if i > 10 {
                break;
            }
        }
    }
    println!("  M={m}, N={n}, K={k}: max_abs_diff = {max_diff:.2e}");
    // bf16 rounding in CPU vs GPU rms_norm will differ, allow some tolerance
    assert!(
        max_diff < 0.1,
        "rms_norm+GEMM fused diff too large: {max_diff:.2e}"
    );
    println!("PASS: rms_norm + GEMM fused matches reference");
}
