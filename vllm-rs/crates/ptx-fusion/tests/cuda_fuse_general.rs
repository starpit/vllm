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
