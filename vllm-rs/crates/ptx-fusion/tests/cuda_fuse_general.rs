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
extract_entry!(
    "kernels/vllm_rms_norm.ptx",
    "_Z15rms_norm_kernelI13__nv_bfloat16EvPT_PKS1_S4_fi",
    RMS_NORM_BF16_PTX
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

const FUSED_RMS_NORM_GEMM_PTX: &str = ptx_fusion::fuse!(
    a = "intrinsic:rms_norm",
    b = "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    bind = { a.output => b.param_0 },
    perimeter = "kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    name = "fused_rms_norm_gemm",
);

// Same kernel via compile! — resolves gemm_64x128x32 from ferrite.toml
const COMPILED_NORM_GEMM: ptx_fusion::FeriteKernel = ptx_fusion::compile!(
    a = rms_norm,
    b = gemm_64x128x32,
    bind = { a.output => b.param_0 },
    name = "compiled_norm_gemm",
);

#[test]
fn compile_macro_metadata() {
    assert_eq!(COMPILED_NORM_GEMM.tile_m, 64);
    assert_eq!(COMPILED_NORM_GEMM.tile_n, 128);
    assert_eq!(COMPILED_NORM_GEMM.threads, 128);
    assert_eq!(COMPILED_NORM_GEMM.smem_bytes, 36864);
    assert_eq!(COMPILED_NORM_GEMM.extra_param_bytes, 32); // rms_norm prefix
    assert!(
        COMPILED_NORM_GEMM
            .ptx
            .contains("FERRITE: rms_norm prologue")
    );
    assert!(COMPILED_NORM_GEMM.ptx.contains("ferrite_params[88]"));
    assert!(COMPILED_NORM_GEMM.ptx.contains("compiled_norm_gemm"));
    println!(
        "compile! metadata: tile={}x{}, threads={}, smem={}, extra_params={}B",
        COMPILED_NORM_GEMM.tile_m,
        COMPILED_NORM_GEMM.tile_n,
        COMPILED_NORM_GEMM.threads,
        COMPILED_NORM_GEMM.smem_bytes,
        COMPILED_NORM_GEMM.extra_param_bytes
    );
}

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

    let input_ptr_val = input_ptr as u64;
    let weight_ptr_val = weight_ptr as u64;
    let hidden_val = k;
    let a_stride_val = k as u64; // stride in elements (= K for row-major A)

    let (gx, gy, gz) = compute_grid(m, n, 64, 128);
    let cfg = LaunchConfig {
        grid_dim: (gx, gy, gz),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 36864,
    };

    // Pipeline compiler param order: input, weight, eps, hidden, a_stride, ferrite_params[88]
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&input_ptr_val)
            .arg(&weight_ptr_val)
            .arg(&eps)
            .arg(&hidden_val)
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

/// Helper: run fused rms_norm+GEMM at given dimensions, compare against CPU norm + GPU GEMM.
fn run_fused_norm_gemm_test(m: u32, n: u32, k: u32) -> f32 {
    let eps = 1e-5f32;
    let ctx = ctx();
    let stream = ctx.default_stream();

    let h_input: Vec<half::bf16> = (0..(m * k) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.3))
        .collect();
    let h_weight: Vec<half::bf16> = (0..k as usize)
        .map(|i| half::bf16::from_f32(1.0 + (i as f32) * 0.001))
        .collect();
    let h_b: Vec<half::bf16> = (0..(n * k) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00023 + 0.3).cos() * 0.1))
        .collect();

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
        input_ptr as u64,
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

    let input_ptr_val = input_ptr as u64;
    let weight_ptr_val = weight_ptr as u64;
    let hidden_val = k;
    let a_stride_val = k as u64;

    let (gx, gy, gz) = compute_grid(m, n, 64, 128);
    let cfg = LaunchConfig {
        grid_dim: (gx, gy, gz),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 36864,
    };

    // Pipeline compiler param order: input, weight, eps, hidden, a_stride, ferrite_params[88]
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&input_ptr_val)
            .arg(&weight_ptr_val)
            .arg(&eps)
            .arg(&hidden_val)
            .arg(&a_stride_val)
            .arg(&gemm_params)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let fused_out = stream.clone_dtoh(&d_d).unwrap();

    let mut max_diff = 0.0f32;
    let mut worst_i = 0;
    for (i, (r, f)) in ref_out.iter().zip(fused_out.iter()).enumerate() {
        let diff = (r.to_f32() - f.to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
            worst_i = i;
        }
    }
    if max_diff > 0.05 {
        let r = ref_out[worst_i].to_f32();
        let f = fused_out[worst_i].to_f32();
        println!(
            "  M={m}, N={n}, K={k}: max_abs_diff = {max_diff:.2e} at [{worst_i}] ref={r:.6} fused={f:.6}"
        );
    } else {
        println!("  M={m}, N={n}, K={k}: max_abs_diff = {max_diff:.2e}");
    }
    max_diff
}

#[test]
fn fused_norm_gemm_multi_tile_n() {
    // N > tile_n (128) → multiple N-tiles → swizzle kicks in
    println!("=== fused norm+GEMM: multi-tile N (swizzle test) ===");
    let diff = run_fused_norm_gemm_test(64, 256, 128);
    assert!(diff < 0.1, "M=64,N=256,K=128: diff={diff:.2e}");
    let diff = run_fused_norm_gemm_test(64, 384, 128);
    assert!(diff < 0.1, "M=64,N=384,K=128: diff={diff:.2e}");
    let diff = run_fused_norm_gemm_test(64, 512, 128);
    assert!(diff < 0.1, "M=64,N=512,K=128: diff={diff:.2e}");
    println!("PASS");
}

#[test]
fn fused_norm_gemm_multi_tile_m() {
    // M > tile_m (64) → multiple M-tiles + swizzle
    println!("=== fused norm+GEMM: multi-tile M ===");
    let diff = run_fused_norm_gemm_test(128, 128, 128);
    assert!(diff < 0.1, "M=128,N=128,K=128: diff={diff:.2e}");
    let diff = run_fused_norm_gemm_test(256, 128, 128);
    assert!(diff < 0.1, "M=256,N=128,K=128: diff={diff:.2e}");
    println!("PASS");
}

#[test]
fn fused_norm_gemm_multi_tile_mn() {
    // Both M and N multi-tile → full swizzle
    println!("=== fused norm+GEMM: multi-tile M×N ===");
    let diff = run_fused_norm_gemm_test(128, 256, 128);
    assert!(diff < 0.1, "M=128,N=256,K=128: diff={diff:.2e}");
    let diff = run_fused_norm_gemm_test(256, 384, 128);
    assert!(diff < 0.1, "M=256,N=384,K=128: diff={diff:.2e}");
    println!("PASS");
}

#[test]
fn fused_norm_gemm_large_k() {
    // K > tile_k (32) → multiple K-loop iterations
    println!("=== fused norm+GEMM: large K ===");
    let diff = run_fused_norm_gemm_test(64, 128, 256);
    assert!(diff < 0.1, "M=64,N=128,K=256: diff={diff:.2e}");
    let diff = run_fused_norm_gemm_test(64, 128, 512);
    assert!(diff < 0.1, "M=64,N=128,K=512: diff={diff:.2e}");
    println!("PASS");
}

#[test]
fn fused_norm_gemm_model_dims() {
    println!("=== fused norm+GEMM: sweep dimensions ===");
    let mut failures = Vec::new();

    for &(m, n, k) in &[
        // Single tile (baseline)
        (64, 128, 32),
        (64, 128, 64),
        (64, 128, 128),
        // Increasing K
        (64, 128, 256),
        (64, 128, 512),
        (64, 128, 1024),
        (64, 128, 2048),
        // Multi-tile N with small K
        (64, 256, 128),
        (64, 384, 128),
        (64, 512, 128),
        (64, 2560, 128),
        // Multi-tile M
        (128, 128, 128),
        (256, 128, 128),
        // Multi-tile M+N
        (128, 256, 128),
        (256, 256, 128),
        // Real model dims
        (64, 2560, 2048),
        (256, 2560, 2048),
    ] {
        let diff = run_fused_norm_gemm_test(m, n, k);
        // Tolerance: bf16 rms_norm vs f32 CPU rms_norm accumulates error proportional
        // to K (each of K products has bf16 rounding). Allow ~1 ULP per 512 K elements.
        let tol = 0.1 + (k as f32 / 512.0).ceil();
        if diff > tol {
            failures.push((m, n, k, diff));
        }
    }

    if !failures.is_empty() {
        println!("\nFAILURES:");
        for (m, n, k, diff) in &failures {
            println!("  M={m}, N={n}, K={k}: diff={diff:.2e}");
        }
        panic!("{} dimension(s) failed", failures.len());
    }
    println!("PASS: all dimensions");
}

// ══════════════════════════════════════════════════════════════════════
// Pipeline-compiled rms_norm + GEMM: ptxas + GPU correctness
// Uses the new pipeline compiler (not intrinsics) — everything derived
// from PTX analysis of the real rms_norm and CUTLASS kernels.
// ══════════════════════════════════════════════════════════════════════

const PIPELINE_FUSED_NORM_GEMM_PTX: &str = ptx_fusion_macros::pipeline_fuse!(
    "kernels/vllm_rms_norm.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "pipeline_fused_norm_gemm",
    "bfloat16"
);

#[test]
fn pipeline_fused_norm_gemm_ptxas() {
    println!("=== Pipeline-compiled rms_norm+GEMM: ptxas ===");

    // Verify the fused PTX has pipeline markers
    assert!(
        PIPELINE_FUSED_NORM_GEMM_PTX.contains("rms_norm prologue"),
        "should have rms_norm prologue"
    );
    assert!(
        PIPELINE_FUSED_NORM_GEMM_PTX.contains("mma.sync"),
        "should preserve GEMM MMA"
    );

    let path = "/tmp/pipeline_fused_norm_gemm.ptx";
    std::fs::write(path, PIPELINE_FUSED_NORM_GEMM_PTX).unwrap();

    let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        for line in stderr.lines().take(20) {
            println!("  ptxas: {line}");
        }
        panic!("ptxas FAILED on pipeline-compiled fused PTX");
    }
    println!("PASS: pipeline-compiled fused PTX passes ptxas");
}

#[test]
fn pipeline_fused_norm_gemm_gpu() {
    println!("=== Pipeline-compiled rms_norm+GEMM: GPU correctness ===");

    let ctx = ctx();
    let m = 64u32;
    let k = 128u32;
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

    // Fused: pipeline-compiled rms_norm + GEMM
    let stream = ctx.default_stream();
    let d_input = stream.clone_htod(&h_input).unwrap();
    let d_weight = stream.clone_htod(&h_weight).unwrap();
    let d_b = stream.clone_htod(&h_b).unwrap();
    let d_c: CudaSlice<half::bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    let mut d_d: CudaSlice<half::bf16> = stream.alloc_zeros((m * n) as usize).unwrap();

    let module = ctx
        .load_module(Ptx::from_src(PIPELINE_FUSED_NORM_GEMM_PTX))
        .unwrap();
    let func = module.load_function("pipeline_fused_norm_gemm").unwrap();

    let (input_ptr, _) = d_input.device_ptr(&stream);
    let (weight_ptr, _) = d_weight.device_ptr(&stream);
    let (b_ptr, _) = d_b.device_ptr(&stream);
    let (c_ptr, _) = d_c.device_ptr(&stream);
    let (d_ptr, _) = d_d.device_ptr(&stream);

    // Build flat GEMM params (A_ptr = raw input, norm happens inline)
    let gemm_params = build_flat_params(
        input_ptr as u64, // A_ptr
        b_ptr as u64,
        c_ptr as u64,
        d_ptr as u64,
        m,
        n,
        k,
        k, // lda
        k, // ldb (column-major B, stride = K)
        n, // ldc
        n, // ldd
        1.0,
        0.0,
    );

    let weight_ptr_val = weight_ptr as u64;
    let hidden_val = k;
    let a_ptr_val = input_ptr as u64;
    let a_stride_val = k as u64;

    let (gx, gy, gz) = compute_grid(m, n, 64, 128);
    let cfg = LaunchConfig {
        grid_dim: (gx, gy, gz),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 36864 + 512, // GEMM SMEM + inv_rms array
    };

    // Params: rms_input(u64), rms_weight(u64), rms_epsilon(f32), rms_hidden(u32),
    //         rms_a_stride(u64), ferrite_params[88]
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&a_ptr_val) // _ferrite_rms_input
            .arg(&weight_ptr_val) // _ferrite_rms_weight
            .arg(&eps) // _ferrite_rms_epsilon
            .arg(&hidden_val) // _ferrite_rms_hidden
            .arg(&a_stride_val) // _ferrite_rms_a_stride
            .arg(&gemm_params) // ferrite_params[88]
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
        if diff > 0.5 && i < 10 {
            println!(
                "  MISMATCH [{i}]: ref={:.4}, fused={:.4}, diff={diff:.2e}",
                r.to_f32(),
                f.to_f32()
            );
        }
    }
    println!("  M={m}, N={n}, K={k}: max_abs_diff = {max_diff:.2e}");
    assert!(
        max_diff < 0.1,
        "pipeline norm+GEMM diff too large: {max_diff:.2e}"
    );
    println!("PASS: pipeline-compiled rms_norm+GEMM matches reference");
}

// ══════════════════════════════════════════════════════════════════════
// M-sweep test: pipeline-compiled rms_norm+GEMM at ALL M values.
// This is the Phase 1 gate — the tile index extraction must produce
// correct m_tile at every M, including M>128 (multiple m_tiles).
// ══════════════════════════════════════════════════════════════════════

#[test]
fn pipeline_fused_norm_gemm_m_sweep() {
    println!("=== Pipeline-compiled rms_norm+GEMM: M-sweep ===");

    let ctx = ctx();
    let stream = ctx.default_stream();
    let k = 896u32; // Qwen 0.5B hidden dim
    let n = 128u32;
    let eps = 1e-5f32;

    // Weight vector (shared across all M values)
    let h_weight: Vec<half::bf16> = (0..k as usize)
        .map(|i| half::bf16::from_f32(1.0 + (i as f32) * 0.001))
        .collect();
    // B matrix (shared)
    let h_b: Vec<half::bf16> = (0..(n * k) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00023 + 0.3).cos() * 0.1))
        .collect();
    let d_weight = stream.clone_htod(&h_weight).unwrap();
    let d_b = stream.clone_htod(&h_b).unwrap();

    let module = ctx
        .load_module(Ptx::from_src(PIPELINE_FUSED_NORM_GEMM_PTX))
        .unwrap();
    let func = module.load_function("pipeline_fused_norm_gemm").unwrap();

    // Load the separate rms_norm kernel for reference
    let rms_module = ctx.load_module(Ptx::from_src(RMS_NORM_BF16_PTX)).unwrap();
    let rms_func = rms_module
        .load_function("_Z15rms_norm_kernelI13__nv_bfloat16EvPT_PKS1_S4_fi")
        .unwrap();

    let m_values = [1u32, 2, 4, 8, 16, 32, 64, 65, 128, 256, 512];
    let mut failures = Vec::new();

    for &m in &m_values {
        // Generate input for this M
        let h_input: Vec<half::bf16> = (0..(m * k) as usize)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.3))
            .collect();

        // Reference: GPU rms_norm then GPU GEMM (separate launches)
        let d_ref_input = stream.clone_htod(&h_input).unwrap();
        let mut d_normed: CudaSlice<half::bf16> = stream.alloc_zeros((m * k) as usize).unwrap();
        let (ref_inp_p, _) = d_ref_input.device_ptr(&stream);
        let (ref_nw_p, _) = d_weight.device_ptr(&stream);
        let (normed_p, _) = d_normed.device_ptr(&stream);
        unsafe {
            stream
                .launch_builder(&rms_func)
                .arg(&normed_p) // output
                .arg(&ref_inp_p) // input
                .arg(&ref_nw_p) // weight
                .arg(&eps)
                .arg(&(k as i32))
                .launch(LaunchConfig {
                    grid_dim: (m, 1, 1),
                    block_dim: (k.min(1024), 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .unwrap();
        stream.synchronize().unwrap();
        let h_normed = stream.clone_dtoh(&d_normed).unwrap();

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

        // Fused path
        let d_input = stream.clone_htod(&h_input).unwrap();
        let d_c: CudaSlice<half::bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
        let mut d_d: CudaSlice<half::bf16> = stream.alloc_zeros((m * n) as usize).unwrap();

        let (input_ptr, _) = d_input.device_ptr(&stream);
        let (weight_ptr, _) = d_weight.device_ptr(&stream);
        let (b_ptr, _) = d_b.device_ptr(&stream);
        let (c_ptr, _) = d_c.device_ptr(&stream);
        let (d_ptr, _) = d_d.device_ptr(&stream);

        let gemm_params = build_flat_params(
            input_ptr as u64,
            b_ptr as u64,
            c_ptr as u64,
            d_ptr as u64,
            m,
            n,
            k,
            k, // lda
            k, // ldb
            n, // ldc
            n, // ldd
            1.0,
            0.0,
        );

        let weight_ptr_val = weight_ptr as u64;
        let hidden_val = k;
        let a_ptr_val = input_ptr as u64;
        let a_stride_val = k as u64;

        let (gx, gy, gz) = compute_grid(m, n, 64, 128);
        let cfg = LaunchConfig {
            grid_dim: (gx, gy, gz),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 36864 + 512,
        };

        unsafe {
            stream
                .launch_builder(&func)
                .arg(&a_ptr_val)
                .arg(&weight_ptr_val)
                .arg(&eps)
                .arg(&hidden_val)
                .arg(&a_stride_val)
                .arg(&gemm_params)
                .launch(cfg)
        }
        .unwrap();
        stream.synchronize().unwrap();
        let fused_out = stream.clone_dtoh(&d_d).unwrap();

        // Compare
        let mut max_diff = 0.0f32;
        for (r, f) in ref_out.iter().zip(fused_out.iter()) {
            let diff = (r.to_f32() - f.to_f32()).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        let status = if max_diff < 0.1 { "PASS" } else { "FAIL" };
        println!("  M={m:>4} grid=({gx},{gy},{gz}) max_diff={max_diff:.2e} {status}");
        if max_diff >= 0.1 {
            failures.push((m, max_diff));
        }
    }

    assert!(failures.is_empty(), "M-sweep failures: {:?}", failures);
    println!("PASS: all M values correct");
}

// ══════════════════════════════════════════════════════════════════════
// Prologue isolation test: run JUST the rms_norm reduction prologue
// and verify inv_rms values against CPU reference.
// ══════════════════════════════════════════════════════════════════════

const PROLOGUE_TEST_PTX: &str = r#"
.version 8.8
.target sm_89
.address_size 64

.visible .entry test_prologue(
    .param .u64 _ferrite_rms_input,
    .param .f32 _ferrite_rms_epsilon,
    .param .u32 _ferrite_rms_hidden,
    .param .u64 _ferrite_rms_a_stride,
    .param .u32 _param_nrows,
    .param .u64 _param_out
)
{
    .reg .f32 %f_rms_inv;
    .reg .f32 %f_rms_sq, %f_rms_sum;
    .reg .f32 %f_rms_eps, %f_rms_hdnf;
    .reg .f32 %f_rms_t0, %f_rms_t1;
    .reg .b32 %r_rms_k, %r_rms_hdn, %r_rms_step;
    .reg .b32 %r_rms_row, %r_rms_nrows;
    .reg .b64 %rd_rms_in, %rd_rms_str;
    .reg .b64 %rd_rms_rowbase, %rd_rms_cur;
    .reg .b16 %h_rms_a;
    .reg .pred %p_rms_lp, %p_rms_row;
    .reg .b64 %rd_out;
    .reg .b32 %r_tmp;
    .shared .align 4 .f32 _ferrite_inv_rms[128];
    .shared .align 4 .f32 _ferrite_warp_scratch[4];

    ld.param.u64    %rd_rms_in, [_ferrite_rms_input];
    cvta.to.global.u64  %rd_rms_in, %rd_rms_in;
    ld.param.f32    %f_rms_eps, [_ferrite_rms_epsilon];
    ld.param.u32    %r_rms_hdn, [_ferrite_rms_hidden];
    cvt.rn.f32.u32  %f_rms_hdnf, %r_rms_hdn;
    ld.param.u64    %rd_rms_str, [_ferrite_rms_a_stride];
    ld.param.u32    %r_rms_nrows, [_param_nrows];
    ld.param.u64    %rd_out, [_param_out];
    cvta.to.global.u64  %rd_out, %rd_out;

    // -- Same prologue as pipeline_compile.rs generates --
    mov.u32     %r_rms_row, 0;
$L_rms_row_loop:
    mov.u32     %r_rms_step, %ctaid.x;
    mul.lo.s32  %r_rms_step, %r_rms_step, %r_rms_nrows;
    add.u32     %r_rms_step, %r_rms_step, %r_rms_row;
    cvt.s64.s32 %rd_rms_rowbase, %r_rms_step;
    mul.lo.s64  %rd_rms_rowbase, %rd_rms_rowbase, %rd_rms_str;
    shl.b64     %rd_rms_rowbase, %rd_rms_rowbase, 1;
    add.s64     %rd_rms_rowbase, %rd_rms_in, %rd_rms_rowbase;

    // K-loop: sum of squares
    mov.f32     %f_rms_sq, 0f00000000;
    mov.u32     %r_rms_k, %tid.x;
$L_rms_k_loop:
    setp.ge.u32 %p_rms_lp, %r_rms_k, %r_rms_hdn;
    @%p_rms_lp bra  $L_rms_k_done;
    cvt.u64.u32 %rd_rms_cur, %r_rms_k;
    shl.b64     %rd_rms_cur, %rd_rms_cur, 1;
    add.s64     %rd_rms_cur, %rd_rms_rowbase, %rd_rms_cur;
    ld.global.nc.b16    %h_rms_a, [%rd_rms_cur];
    cvt.f32.bf16    %f_rms_t0, %h_rms_a;
    fma.rn.f32  %f_rms_sq, %f_rms_t0, %f_rms_t0, %f_rms_sq;
    mov.u32     %r_rms_step, %ntid.x;
    add.u32     %r_rms_k, %r_rms_k, %r_rms_step;
    bra     $L_rms_k_loop;
$L_rms_k_done:

    // Warp shuffle reduction
    mov.b32     %r_rms_k, %f_rms_sq;
    shfl.sync.down.b32  %r_rms_step|%p_rms_lp, %r_rms_k, 16, 31, -1;
    mov.b32     %f_rms_t0, %r_rms_step;
    mov.b32     %f_rms_t1, %r_rms_k;
    add.f32     %f_rms_t1, %f_rms_t1, %f_rms_t0;
    mov.b32     %r_rms_k, %f_rms_t1;
    shfl.sync.down.b32  %r_rms_step|%p_rms_lp, %r_rms_k, 8, 31, -1;
    mov.b32     %f_rms_t0, %r_rms_step;
    mov.b32     %f_rms_t1, %r_rms_k;
    add.f32     %f_rms_t1, %f_rms_t1, %f_rms_t0;
    mov.b32     %r_rms_k, %f_rms_t1;
    shfl.sync.down.b32  %r_rms_step|%p_rms_lp, %r_rms_k, 4, 31, -1;
    mov.b32     %f_rms_t0, %r_rms_step;
    mov.b32     %f_rms_t1, %r_rms_k;
    add.f32     %f_rms_t1, %f_rms_t1, %f_rms_t0;
    mov.b32     %r_rms_k, %f_rms_t1;
    shfl.sync.down.b32  %r_rms_step|%p_rms_lp, %r_rms_k, 2, 31, -1;
    mov.b32     %f_rms_t0, %r_rms_step;
    mov.b32     %f_rms_t1, %r_rms_k;
    add.f32     %f_rms_t1, %f_rms_t1, %f_rms_t0;
    mov.b32     %r_rms_k, %f_rms_t1;
    shfl.sync.down.b32  %r_rms_step|%p_rms_lp, %r_rms_k, 1, 31, -1;
    mov.b32     %f_rms_t0, %r_rms_step;
    mov.b32     %f_rms_t1, %r_rms_k;
    add.f32     %f_rms_t1, %f_rms_t1, %f_rms_t0;
    mov.b32     %r_rms_k, %f_rms_t1;

    // SMEM reduce across warps
    mov.b32     %f_rms_sum, %r_rms_k;
    mov.u32     %r_rms_step, %tid.x;
    and.b32     %r_rms_step, %r_rms_step, 31;
    setp.ne.u32 %p_rms_lp, %r_rms_step, 0;
    @%p_rms_lp bra  $L_rms_warp_done;
    mov.u32     %r_rms_step, %tid.x;
    shr.u32     %r_rms_step, %r_rms_step, 5;
    shl.b32     %r_rms_step, %r_rms_step, 2;
    mov.u32     %r_rms_k, _ferrite_warp_scratch;
    add.s32     %r_rms_step, %r_rms_k, %r_rms_step;
    st.shared.f32   [%r_rms_step], %f_rms_sum;
$L_rms_warp_done:
    bar.sync    15;

    // Thread 0 sums warp contributions and computes inv_rms
    mov.u32     %r_rms_step, %tid.x;
    setp.ne.u32 %p_rms_lp, %r_rms_step, 0;
    @%p_rms_lp bra  $L_rms_reduce_done;
    mov.u32     %r_rms_k, _ferrite_warp_scratch;
    ld.shared.f32   %f_rms_sum, [%r_rms_k];
    ld.shared.f32   %f_rms_t0, [%r_rms_k+4];
    add.f32     %f_rms_sum, %f_rms_sum, %f_rms_t0;
    ld.shared.f32   %f_rms_t0, [%r_rms_k+8];
    add.f32     %f_rms_sum, %f_rms_sum, %f_rms_t0;
    ld.shared.f32   %f_rms_t0, [%r_rms_k+12];
    add.f32     %f_rms_sum, %f_rms_sum, %f_rms_t0;
    div.rn.f32  %f_rms_sum, %f_rms_sum, %f_rms_hdnf;
    add.f32     %f_rms_sum, %f_rms_sum, %f_rms_eps;
    rsqrt.approx.f32    %f_rms_sum, %f_rms_sum;
    mov.u32     %r_rms_k, _ferrite_inv_rms;
    shl.b32     %r_rms_step, %r_rms_row, 2;
    add.s32     %r_rms_step, %r_rms_k, %r_rms_step;
    st.shared.f32   [%r_rms_step], %f_rms_sum;
$L_rms_reduce_done:
    bar.sync    15;

    // Advance to next row
    add.u32     %r_rms_row, %r_rms_row, 1;
    setp.lt.u32 %p_rms_row, %r_rms_row, %r_rms_nrows;
    @%p_rms_row bra  $L_rms_row_loop;

    bar.sync    15;

    // -- Write inv_rms array to GMEM (thread 0 only) --
    mov.u32     %r_rms_step, %tid.x;
    setp.ne.u32 %p_rms_lp, %r_rms_step, 0;
    @%p_rms_lp bra  $L_done;

    mov.u32     %r_rms_k, _ferrite_inv_rms;
    mov.u32     %r_rms_row, 0;
$L_write_loop:
    setp.ge.u32 %p_rms_lp, %r_rms_row, %r_rms_nrows;
    @%p_rms_lp bra  $L_done;
    shl.b32     %r_rms_step, %r_rms_row, 2;
    add.s32     %r_tmp, %r_rms_k, %r_rms_step;
    ld.shared.f32   %f_rms_t0, [%r_tmp];
    cvt.u64.u32 %rd_rms_cur, %r_rms_step;
    add.s64     %rd_rms_cur, %rd_out, %rd_rms_cur;
    st.global.f32   [%rd_rms_cur], %f_rms_t0;
    add.u32     %r_rms_row, %r_rms_row, 1;
    bra     $L_write_loop;
$L_done:
    ret;
}
"#;

#[test]
fn prologue_isolation_inv_rms() {
    println!("=== Prologue isolation: verify inv_rms values ===");

    // ptxas check first
    let path = "/tmp/prologue_isolation.ptx";
    std::fs::write(path, PROLOGUE_TEST_PTX).unwrap();
    let ptxas_out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");
    if !ptxas_out.status.success() {
        let stderr = String::from_utf8_lossy(&ptxas_out.stderr);
        for line in stderr.lines().take(20) {
            println!("  ptxas: {line}");
        }
        panic!("ptxas FAILED on prologue isolation PTX");
    }
    println!("  ptxas: PASS");

    let ctx = ctx();
    let stream = ctx.default_stream();

    let m = 4u32; // small number of rows for quick test
    let k = 128u32;
    let eps = 1e-5f32;

    // Generate bf16 input (same pattern as GPU test)
    let h_input: Vec<half::bf16> = (0..(m * k) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.3))
        .collect();

    // CPU reference: compute inv_rms per row
    let mut cpu_inv_rms = vec![0.0f32; m as usize];
    for row in 0..m as usize {
        let mut sum_sq = 0.0f32;
        for col in 0..k as usize {
            let v = h_input[row * k as usize + col].to_f32();
            sum_sq += v * v;
        }
        cpu_inv_rms[row] = (sum_sq / k as f32 + eps).sqrt().recip();
    }

    // Launch prologue kernel
    let d_input = stream.clone_htod(&h_input).unwrap();
    let d_out: CudaSlice<f32> = stream.alloc_zeros(m as usize).unwrap();

    let module = ctx.load_module(Ptx::from_src(PROLOGUE_TEST_PTX)).unwrap();
    let func = module.load_function("test_prologue").unwrap();

    let (input_ptr, _) = d_input.device_ptr(&stream);
    let (out_ptr, _) = d_out.device_ptr(&stream);
    let input_ptr_val = input_ptr as u64;
    let stride_val = k as u64; // stride in elements
    let out_ptr_val = out_ptr as u64;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&input_ptr_val)
            .arg(&eps)
            .arg(&k)
            .arg(&stride_val)
            .arg(&m)
            .arg(&out_ptr_val)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let gpu_inv_rms = stream.clone_dtoh(&d_out).unwrap();

    // Compare
    println!("  Row | CPU inv_rms | GPU inv_rms | diff");
    let mut max_diff = 0.0f32;
    for row in 0..m as usize {
        let diff = (cpu_inv_rms[row] - gpu_inv_rms[row]).abs();
        let rel_diff = diff / cpu_inv_rms[row].abs().max(1e-10);
        println!(
            "  {:3} | {:11.6} | {:11.6} | {:.2e} (rel {:.2e})",
            row, cpu_inv_rms[row], gpu_inv_rms[row], diff, rel_diff
        );
        if diff > max_diff {
            max_diff = diff;
        }
    }
    // Allow small tolerance for bf16 rounding in sum-of-squares
    assert!(max_diff < 0.5, "inv_rms mismatch too large: {max_diff:.2e}");
    println!("PASS: prologue produces correct inv_rms (max_diff = {max_diff:.2e})");
}

// ══════════════════════════════════════════════════════════════════════
// SiLU+mul fused into GEMM_down: ptxas + GPU correctness
// Tests the Pointwise→TiledGemm pipeline pattern.
// ══════════════════════════════════════════════════════════════════════

const SILU_MUL_FUSED_GEMM_PTX: &str = ptx_fusion::compile! {
    a = silu_mul,
    b = gemm_64x128x32,
    bind = { a.output => b.param_0 },
    name = "silu_mul_fused_gemm",
}
.ptx;

/// CPU SiLU: x / (1 + exp(-x))
fn cpu_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
fn silu_mul_fused_gemm_ptxas() {
    println!("=== SiLU+mul fused GEMM: ptxas ===");

    assert!(
        SILU_MUL_FUSED_GEMM_PTX.contains("ex2.approx"),
        "should contain SiLU exp approximation"
    );
    assert!(
        SILU_MUL_FUSED_GEMM_PTX.contains("_ferrite_intermediate_bytes"),
        "should have intermediate offset param"
    );
    assert!(
        SILU_MUL_FUSED_GEMM_PTX.contains("mma.sync"),
        "should preserve GEMM MMA instructions"
    );

    let path = "/tmp/silu_mul_fused_gemm.ptx";
    std::fs::write(path, SILU_MUL_FUSED_GEMM_PTX).unwrap();
    let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        for line in stderr.lines().take(20) {
            println!("  ptxas: {line}");
        }
        panic!("ptxas FAILED on SiLU+mul fused GEMM PTX");
    }
    println!("PASS: SiLU+mul fused GEMM passes ptxas");
}

#[test]
fn silu_mul_fused_gemm_gpu() {
    println!("=== SiLU+mul fused GEMM: GPU correctness ===");
    let ctx = ctx();
    let stream = ctx.default_stream();

    let m = 64u32;
    let intermediate = 128u32; // intermediate_size
    let n_down = 128u32; // output hidden dim of down GEMM
    let k_down = intermediate; // K for down GEMM = intermediate

    // Generate gate_up data: [M, 2*intermediate] bf16
    let gate_up_cols = 2 * intermediate;
    let h_gate_up: Vec<half::bf16> = (0..(m * gate_up_cols) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00031 - 0.4).sin() * 0.2))
        .collect();

    // Generate down weight: [N_down, intermediate] bf16
    let h_b_down: Vec<half::bf16> = (0..(n_down * k_down) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00019 + 0.2).cos() * 0.1))
        .collect();

    // ── Reference: CPU SiLU+mul then GPU GEMM ──
    // activated[i,j] = SiLU(gate_up[i, j]) * gate_up[i, j + intermediate]
    let h_activated: Vec<half::bf16> = (0..(m * intermediate) as usize)
        .map(|idx| {
            let row = idx / intermediate as usize;
            let col = idx % intermediate as usize;
            let gate = h_gate_up[row * gate_up_cols as usize + col].to_f32();
            let up = h_gate_up[row * gate_up_cols as usize + intermediate as usize + col].to_f32();
            half::bf16::from_f32(cpu_silu(gate) * up)
        })
        .collect();

    let ref_out = run_flat_gemm(
        &ctx,
        FLAT_GEMM_BASE_PTX,
        "flat_gemm_base",
        &h_activated,
        &h_b_down,
        m,
        n_down,
        k_down,
    );

    // ── Fused: SiLU+mul inline at GEMM A-loads ──
    // A_ptr = gate_up_buf, lda = 2*intermediate (so A-loads naturally read gate columns)
    let d_gate_up = stream.clone_htod(&h_gate_up).unwrap();
    let d_b_down = stream.clone_htod(&h_b_down).unwrap();
    let d_c: CudaSlice<half::bf16> = stream.alloc_zeros((m * n_down) as usize).unwrap();
    let d_d: CudaSlice<half::bf16> = stream.alloc_zeros((m * n_down) as usize).unwrap();

    let module = ctx
        .load_module(Ptx::from_src(SILU_MUL_FUSED_GEMM_PTX))
        .unwrap();
    let func = module.load_function("silu_mul_fused_gemm").unwrap();

    let (gate_up_ptr, _) = d_gate_up.device_ptr(&stream);
    let (b_ptr, _) = d_b_down.device_ptr(&stream);
    let (c_ptr, _) = d_c.device_ptr(&stream);
    let (d_ptr, _) = d_d.device_ptr(&stream);

    // A_ptr = gate_up_ptr, lda = 2*intermediate (stride includes both gate + up columns)
    let gemm_params = build_flat_params(
        gate_up_ptr as u64,
        b_ptr as u64,
        c_ptr as u64,
        d_ptr as u64,
        m,
        n_down,
        k_down,
        gate_up_cols, // lda = 2*intermediate (wider stride)
        k_down,
        n_down,
        n_down,
        1.0,
        0.0,
    );

    // Extra param: intermediate_bytes = intermediate * 2 (bf16)
    let intermediate_bytes = (intermediate as u64) * 2;

    let (gx, gy, gz) = compute_grid(m, n_down, 64, 128);
    let cfg = LaunchConfig {
        grid_dim: (gx, gy, gz),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 36864,
    };

    // Param order: _ferrite_intermediate_bytes (u64), ferrite_params[88]
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&intermediate_bytes)
            .arg(&gemm_params)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let fused_out = stream.clone_dtoh(&d_d).unwrap();

    // Compare
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for (i, (r, f)) in ref_out.iter().zip(fused_out.iter()).enumerate() {
        let diff = (r.to_f32() - f.to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }
    println!(
        "  M={m}, N={n_down}, K={k_down}, intermediate={intermediate}: max_abs_diff = {max_diff:.2e} at [{max_idx}] ref={:.6} fused={:.6}",
        ref_out[max_idx].to_f32(),
        fused_out[max_idx].to_f32()
    );
    // SiLU has bf16 rounding at multiple stages, allow some tolerance
    assert!(
        max_diff < 1.0,
        "SiLU+mul fused GEMM diff too large: {max_diff:.2e}"
    );
    println!("PASS: SiLU+mul fused GEMM matches reference");
}

/// Debug: compare fused norm+GEMM (compile!) vs GPU rms_norm + GPU GEMM at model dims.
/// Uses the REAL vllm rms_norm kernel on GPU (not CPU reference) to isolate
/// whether the problem is the norm fusion or the GEMM param handoff.
#[test]
fn debug_norm_gemm_vs_separate_gpu() {
    println!("=== DEBUG: fused vs separate (both GPU) at model dims ===");
    let ctx = ctx();
    let stream = ctx.default_stream();

    let m = 4u32;
    let k = 2560u32; // hidden
    let n = 2560u32;
    let eps = 1e-5f32;

    // Test data
    let h_input: Vec<half::bf16> = (0..(m * k) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.3))
        .collect();
    let h_norm_wt: Vec<half::bf16> = (0..k as usize)
        .map(|i| half::bf16::from_f32(1.0 + (i as f32) * 0.001))
        .collect();
    let h_b: Vec<half::bf16> = (0..(n * k) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00023 + 0.3).cos() * 0.1))
        .collect();

    // ── Path A: GPU rms_norm (vllm kernel) then GPU GEMM (flat-param CUTLASS) ──
    let d_input = stream.clone_htod(&h_input).unwrap();
    let d_norm_wt = stream.clone_htod(&h_norm_wt).unwrap();
    let d_b = stream.clone_htod(&h_b).unwrap();
    let mut d_normed: CudaSlice<half::bf16> = stream.alloc_zeros((m * k) as usize).unwrap();

    // Run the real vllm rms_norm kernel
    let rms_module = ctx.load_module(Ptx::from_src(RMS_NORM_BF16_PTX)).unwrap();
    let rms_func = rms_module
        .load_function("_Z15rms_norm_kernelI13__nv_bfloat16EvPT_PKS1_S4_fi")
        .unwrap();
    let (inp_p, _) = d_input.device_ptr(&stream);
    let (nw_p, _) = d_norm_wt.device_ptr(&stream);
    let (normed_p, _) = d_normed.device_ptr(&stream);
    unsafe {
        stream
            .launch_builder(&rms_func)
            .arg(&normed_p) // output
            .arg(&inp_p) // input
            .arg(&nw_p) // weight
            .arg(&eps)
            .arg(&(k as i32)) // hidden_size
            .launch(LaunchConfig {
                grid_dim: (m, 1, 1),
                block_dim: (256.min(k), 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .unwrap();

    // Dump first 8 normed values
    stream.synchronize().unwrap();
    let h_normed_gpu = stream.clone_dtoh(&d_normed).unwrap();
    print!("  GPU rms_norm first 8: ");
    for i in 0..8.min(h_normed_gpu.len()) {
        print!("{:.4} ", h_normed_gpu[i].to_f32());
    }
    println!();

    // CPU rms_norm for comparison
    let h_normed_cpu = cpu_rms_norm(&h_input, &h_norm_wt, m as usize, k as usize, eps);
    print!("  CPU rms_norm first 8: ");
    for i in 0..8.min(h_normed_cpu.len()) {
        print!("{:.4} ", h_normed_cpu[i].to_f32());
    }
    println!();

    // Run flat-param GEMM on GPU-normed data
    let ref_out = run_flat_gemm(
        &ctx,
        FLAT_GEMM_BASE_PTX,
        "flat_gemm_base",
        &h_normed_gpu,
        &h_b,
        m,
        n,
        k,
    );

    // ── Path B: fused norm+GEMM via compile! kernel (per-arg launch) ──
    let d_c: CudaSlice<half::bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    let d_d: CudaSlice<half::bf16> = stream.alloc_zeros((m * n) as usize).unwrap();

    let module = ctx
        .load_module(Ptx::from_src(COMPILED_NORM_GEMM.ptx))
        .unwrap();
    let func = module.load_function(COMPILED_NORM_GEMM.entry).unwrap();

    let (bp, _) = d_b.device_ptr(&stream);
    let (cp, _) = d_c.device_ptr(&stream);
    let (dp, _) = d_d.device_ptr(&stream);

    let gemm_params = build_flat_params(
        inp_p as u64,
        bp as u64,
        cp as u64,
        dp as u64,
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
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&(inp_p as u64))
            .arg(&(nw_p as u64))
            .arg(&eps)
            .arg(&k)
            .arg(&(k as u64))
            .arg(&gemm_params)
            .launch(LaunchConfig {
                grid_dim: (gx, gy, gz),
                block_dim: (128, 1, 1),
                shared_mem_bytes: COMPILED_NORM_GEMM.smem_bytes,
            })
    }
    .unwrap();
    stream.synchronize().unwrap();
    let fused_out = stream.clone_dtoh(&d_d).unwrap();

    // Compare
    let mut max_diff = 0.0f32;
    let mut worst_i = 0;
    for (i, (r, f)) in ref_out.iter().zip(fused_out.iter()).enumerate() {
        let d = (r.to_f32() - f.to_f32()).abs();
        if d > max_diff {
            max_diff = d;
            worst_i = i;
        }
    }
    println!("  separate(GPU norm+GEMM) vs fused: max_diff={max_diff:.2e} at [{worst_i}]");
    print!("  separate first 8: ");
    for i in 0..8.min(ref_out.len()) {
        print!("{:.4} ", ref_out[i].to_f32());
    }
    println!();
    print!("  fused first 8:    ");
    for i in 0..8.min(fused_out.len()) {
        print!("{:.4} ", fused_out[i].to_f32());
    }
    println!();

    let tol = 1.0 + (k as f32 / 512.0).ceil();
    assert!(
        max_diff < tol,
        "fused vs separate GPU diff too large: {max_diff:.2e}"
    );
    println!("PASS");
}

/// Test compile! kernel (COMPILED_NORM_GEMM) at model-scale dimensions.
#[test]
fn norm_gemm_compile_model_dims() {
    println!("=== compile! norm+GEMM at model dims ===");
    let ctx = ctx();
    let stream = ctx.default_stream();

    for &(m, n, k) in &[(4u32, 2560u32, 2560u32), (1, 2560, 2560), (64, 13824, 2560)] {
        let eps = 1e-5f32;
        let h_input: Vec<half::bf16> = (0..(m * k) as usize)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.3))
            .collect();
        let h_norm_wt: Vec<half::bf16> = (0..k as usize)
            .map(|i| half::bf16::from_f32(1.0 + (i as f32) * 0.001))
            .collect();
        let h_b: Vec<half::bf16> = (0..(n * k) as usize)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.00023 + 0.3).cos() * 0.1))
            .collect();

        let h_normed = cpu_rms_norm(&h_input, &h_norm_wt, m as usize, k as usize, eps);
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

        let d_input = stream.clone_htod(&h_input).unwrap();
        let d_norm_wt = stream.clone_htod(&h_norm_wt).unwrap();
        let d_b = stream.clone_htod(&h_b).unwrap();
        let d_c: CudaSlice<half::bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
        let d_d: CudaSlice<half::bf16> = stream.alloc_zeros((m * n) as usize).unwrap();

        let module = ctx
            .load_module(Ptx::from_src(COMPILED_NORM_GEMM.ptx))
            .unwrap();
        let func = module.load_function(COMPILED_NORM_GEMM.entry).unwrap();

        let (inp, _) = d_input.device_ptr(&stream);
        let (nw, _) = d_norm_wt.device_ptr(&stream);
        let (bp, _) = d_b.device_ptr(&stream);
        let (cp, _) = d_c.device_ptr(&stream);
        let (dp, _) = d_d.device_ptr(&stream);

        let gemm_params = build_flat_params(
            inp as u64, bp as u64, cp as u64, dp as u64, m, n, k, k, k, n, n, 1.0, 0.0,
        );

        let (gx, gy, gz) = compute_grid(m, n, 64, 128);
        let cfg = LaunchConfig {
            grid_dim: (gx, gy, gz),
            block_dim: (128, 1, 1),
            shared_mem_bytes: COMPILED_NORM_GEMM.smem_bytes,
        };

        unsafe {
            stream
                .launch_builder(&func)
                .arg(&(inp as u64))
                .arg(&(nw as u64))
                .arg(&eps)
                .arg(&k)
                .arg(&(k as u64))
                .arg(&gemm_params)
                .launch(cfg)
        }
        .unwrap();
        stream.synchronize().unwrap();
        let fused_out = stream.clone_dtoh(&d_d).unwrap();

        let mut max_diff = 0.0f32;
        for (r, f) in ref_out.iter().zip(fused_out.iter()) {
            let d = (r.to_f32() - f.to_f32()).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        let tol = 1.0 + (k as f32 / 512.0).ceil();
        println!("  M={m}, N={n}, K={k}: max_diff={max_diff:.2e} (tol={tol:.1})");
        assert!(max_diff < tol, "compile! norm+GEMM failed: {max_diff:.2e}");
    }
    println!("PASS: compile! norm+GEMM correct at model dims");
}

// ══════════════════════════════════════════════════════════════════════
// Two-GEMM sequenced kernel: GPU correctness
// Verifies that two CUTLASS GEMMs in a single kernel produce the same
// output as running them as separate launches.
// ══════════════════════════════════════════════════════════════════════

const SEQUENCED_TWO_GEMM_PTX: &str = ptx_fusion_macros::sequence_gemms!(
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "sequenced_two_gemm"
);

#[test]
fn sequenced_two_gemm_gpu() {
    println!("=== Sequenced two-GEMM: GPU correctness ===");

    let ctx = ctx();
    let stream = ctx.default_stream();

    // Dimensions: A[M,K] × B1[N,K]^T → intermediate[M,N], then intermediate[M,N] × B2[P,N]^T → output[M,P]
    // Using same tile config (64x128x32) for both GEMMs.
    let m = 64u32;
    let k = 128u32; // K for GEMM_A
    let n = 128u32; // N for GEMM_A = K for GEMM_B
    let p = 128u32; // N for GEMM_B (output columns)

    // Test data
    let h_a: Vec<half::bf16> = (0..(m * k) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.3))
        .collect();
    let h_b1: Vec<half::bf16> = (0..(n * k) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00023 + 0.3).cos() * 0.1))
        .collect();
    let h_b2: Vec<half::bf16> = (0..(p * n) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00019 - 0.2).sin() * 0.15))
        .collect();

    // Reference: two separate GEMM launches
    let intermediate = run_flat_gemm(
        &ctx,
        FLAT_GEMM_BASE_PTX,
        "flat_gemm_base",
        &h_a,
        &h_b1,
        m,
        n,
        k,
    );
    let ref_output = run_flat_gemm(
        &ctx,
        FLAT_GEMM_BASE_PTX,
        "flat_gemm_base",
        &intermediate,
        &h_b2,
        m,
        p,
        n,
    );

    // Sequenced: both GEMMs in one kernel launch
    let d_a = stream.clone_htod(&h_a).unwrap();
    let d_b1 = stream.clone_htod(&h_b1).unwrap();
    let d_b2 = stream.clone_htod(&h_b2).unwrap();
    let d_inter: CudaSlice<half::bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    let d_c2: CudaSlice<half::bf16> = stream.alloc_zeros((m * p) as usize).unwrap();
    let mut d_out: CudaSlice<half::bf16> = stream.alloc_zeros((m * p) as usize).unwrap();

    let module = ctx
        .load_module(Ptx::from_src(SEQUENCED_TWO_GEMM_PTX))
        .unwrap();
    let func = module.load_function("sequenced_two_gemm").unwrap();

    let (a_ptr, _) = d_a.device_ptr(&stream);
    let (b1_ptr, _) = d_b1.device_ptr(&stream);
    let (inter_ptr, _) = d_inter.device_ptr(&stream);
    let (b2_ptr, _) = d_b2.device_ptr(&stream);
    let (c2_ptr, _) = d_c2.device_ptr(&stream);
    let (out_ptr, _) = d_out.device_ptr(&stream);

    // GEMM_A params: A[M,K] × B1[N,K]^T → intermediate[M,N]
    let params_a = build_flat_params(
        a_ptr as u64,
        b1_ptr as u64,
        inter_ptr as u64, // C (unused, beta=0)
        inter_ptr as u64, // D (output)
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
    // GEMM_B params: intermediate[M,N] × B2[P,N]^T → output[M,P]
    let params_b = build_flat_params(
        inter_ptr as u64, // A (reads GEMM_A's output)
        b2_ptr as u64,
        c2_ptr as u64,
        out_ptr as u64,
        m,
        p,
        n,
        n,
        n,
        p,
        p,
        1.0,
        0.0,
    );

    // Grid must be large enough for both GEMMs.
    // Both are M=64, N=128 with 64x128 tiles → 1 block each.
    let (gx, gy, gz) = compute_grid(m, n, 64, 128);
    let cfg = LaunchConfig {
        grid_dim: (gx, gy, gz),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 36864, // CUTLASS SMEM for 64x128x32
    };

    // Global barrier counter (1 barrier between 2 phases)
    let d_barriers: CudaSlice<u32> = stream.alloc_zeros(1).unwrap();
    let (bar_ptr, _) = d_barriers.device_ptr(&stream);
    let bar_ptr_val = bar_ptr as u64;

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&bar_ptr_val)
            .arg(&params_a)
            .arg(&params_b)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let seq_output = stream.clone_dtoh(&d_out).unwrap();

    // Compare
    let mut max_diff = 0.0f32;
    for (i, (r, s)) in ref_output.iter().zip(seq_output.iter()).enumerate() {
        let diff = (r.to_f32() - s.to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0.5 && i < 10 {
            println!(
                "  MISMATCH [{i}]: ref={:.4}, seq={:.4}, diff={diff:.2e}",
                r.to_f32(),
                s.to_f32()
            );
        }
    }
    println!("  M={m}, K={k}, N={n}, P={p}: max_diff={max_diff:.2e}");
    assert!(
        max_diff < 0.01,
        "sequenced two-GEMM diff too large: {max_diff:.2e}"
    );
    println!("PASS: sequenced two-GEMM matches separate launches");
}

// ══════════════════════════════════════════════════════════════════════
// MLP block: gate_up GEMM → SiLU+mul → down GEMM in one kernel
// This is the core of Segment B — the first multi-op CUTLASS bf16 fusion.
// ══════════════════════════════════════════════════════════════════════

const MLP_BLOCK_PTX: &str = ptx_fusion_macros::sequence_mlp_block!(
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "kernels/vllm_silu_mul.ptx",
    "mlp_fused"
);

#[test]
fn mlp_block_gpu() {
    println!("=== MLP block (gate_up → SiLU → down): GPU correctness ===");

    let ctx = ctx();
    let stream = ctx.default_stream();

    let m = 64u32;
    let hidden = 128u32; // input hidden dim
    let intermediate = 128u32; // intermediate size
    let gate_up_cols = 2 * intermediate; // gate_up output has gate + up halves

    // Test data: input[M, hidden], gate_up_weight[gate_up_cols, hidden], down_weight[hidden, intermediate]
    let h_input: Vec<half::bf16> = (0..(m * hidden) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.3))
        .collect();
    let h_gate_up_w: Vec<half::bf16> = (0..(gate_up_cols * hidden) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00023 + 0.3).cos() * 0.1))
        .collect();
    let h_down_w: Vec<half::bf16> = (0..(hidden * intermediate) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00019 - 0.2).sin() * 0.15))
        .collect();

    // Reference: 3 separate launches
    // 1. gate_up GEMM: input[M,hidden] × gate_up_w[gate_up_cols,hidden]^T → gate_up_out[M,gate_up_cols]
    let gate_up_out = run_flat_gemm(
        &ctx,
        FLAT_GEMM_BASE_PTX,
        "flat_gemm_base",
        &h_input,
        &h_gate_up_w,
        m,
        gate_up_cols,
        hidden,
    );

    // 2. CPU SiLU+mul: activated[i,j] = silu(gate_up_out[i,j]) * gate_up_out[i, j+intermediate]
    let h_activated: Vec<half::bf16> = (0..(m * intermediate) as usize)
        .map(|idx| {
            let row = idx / intermediate as usize;
            let col = idx % intermediate as usize;
            let gate = gate_up_out[row * gate_up_cols as usize + col].to_f32();
            let up =
                gate_up_out[row * gate_up_cols as usize + intermediate as usize + col].to_f32();
            half::bf16::from_f32(cpu_silu(gate) * up)
        })
        .collect();

    // 3. down GEMM: activated[M,intermediate] × down_w[hidden,intermediate]^T → output[M,hidden]
    let ref_output = run_flat_gemm(
        &ctx,
        FLAT_GEMM_BASE_PTX,
        "flat_gemm_base",
        &h_activated,
        &h_down_w,
        m,
        hidden,
        intermediate,
    );

    // ptxas pre-check
    let mlp_path = "/tmp/mlp_block_fused.ptx";
    std::fs::write(mlp_path, MLP_BLOCK_PTX).unwrap();
    let ptxas_out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", mlp_path])
        .output()
        .expect("ptxas");
    if !ptxas_out.status.success() {
        let stderr = String::from_utf8_lossy(&ptxas_out.stderr);
        eprintln!("ptxas errors:");
        for line in stderr.lines().take(20) {
            eprintln!("  {line}");
        }
        let ptx_lines: Vec<&str> = MLP_BLOCK_PTX.lines().collect();
        for line in stderr.lines() {
            if let Some(lnum) = line
                .split('(')
                .nth(1)
                .and_then(|s| s.split(')').next())
                .and_then(|s| s.parse::<usize>().ok())
            {
                let start = lnum.saturating_sub(2);
                let end = (lnum + 2).min(ptx_lines.len());
                for i in start..end {
                    let marker = if i + 1 == lnum { ">>>" } else { "   " };
                    eprintln!("{marker} {:4}: {}", i + 1, ptx_lines[i]);
                }
                eprintln!();
            }
        }
        panic!("ptxas FAILED on MLP block PTX");
    }
    println!("  ptxas: PASS ({} lines)", MLP_BLOCK_PTX.lines().count());

    // Fused: single kernel launch
    let d_input = stream.clone_htod(&h_input).unwrap();
    let d_gate_up_w = stream.clone_htod(&h_gate_up_w).unwrap();
    let d_down_w = stream.clone_htod(&h_down_w).unwrap();
    let d_gate_up_out: CudaSlice<half::bf16> =
        stream.alloc_zeros((m * gate_up_cols) as usize).unwrap();
    let d_c_down: CudaSlice<half::bf16> = stream.alloc_zeros((m * hidden) as usize).unwrap();
    let d_output: CudaSlice<half::bf16> = stream.alloc_zeros((m * hidden) as usize).unwrap();

    let module = ctx.load_module(Ptx::from_src(MLP_BLOCK_PTX)).unwrap();
    let func = module.load_function("mlp_fused").unwrap();

    let (inp_ptr, _) = d_input.device_ptr(&stream);
    let (guw_ptr, _) = d_gate_up_w.device_ptr(&stream);
    let (guo_ptr, _) = d_gate_up_out.device_ptr(&stream);
    let (dw_ptr, _) = d_down_w.device_ptr(&stream);
    let (cd_ptr, _) = d_c_down.device_ptr(&stream);
    let (out_ptr, _) = d_output.device_ptr(&stream);

    // gate_up GEMM params: input[M,hidden] × gate_up_w[gate_up_cols,hidden]^T → gate_up_out[M,gate_up_cols]
    let params_gate_up = build_flat_params(
        inp_ptr as u64,
        guw_ptr as u64,
        guo_ptr as u64,
        guo_ptr as u64,
        m,
        gate_up_cols,
        hidden,
        hidden,
        hidden,
        gate_up_cols,
        gate_up_cols,
        1.0,
        0.0,
    );

    // down GEMM params (SiLU-fused): reads gate_up_out with lda=2*intermediate
    let params_down = build_flat_params(
        guo_ptr as u64, // A_ptr = gate_up_out (SiLU reads gate half, per-site loads up half)
        dw_ptr as u64,
        cd_ptr as u64,
        out_ptr as u64,
        m,
        hidden,
        intermediate,
        gate_up_cols, // lda = 2*intermediate (stride covers both gate + up)
        intermediate,
        hidden,
        hidden,
        1.0,
        0.0,
    );

    // Extra param for down GEMM: intermediate_bytes
    let intermediate_bytes = (intermediate as u64) * 2; // bf16

    let (gx, gy, _gz) = compute_grid(m, gate_up_cols, 64, 128);
    let cfg = LaunchConfig {
        grid_dim: (gx, gy, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 36864,
    };

    // Global barrier counter (1 barrier between 2 phases)
    let d_barriers: CudaSlice<u32> = stream.alloc_zeros(1).unwrap();
    let (mlp_bar_ptr, _) = d_barriers.device_ptr(&stream);
    let mlp_bar_val = mlp_bar_ptr as u64;

    // Param order: _phase_barriers, ferrite_params (gate_up), _ferrite_intermediate_bytes_2 (down extra), ferrite_params_2 (down)
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&mlp_bar_val)
            .arg(&params_gate_up)
            .arg(&intermediate_bytes)
            .arg(&params_down)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let fused_output = stream.clone_dtoh(&d_output).unwrap();

    // Compare
    let mut max_diff = 0.0f32;
    for (i, (r, f)) in ref_output.iter().zip(fused_output.iter()).enumerate() {
        let diff = (r.to_f32() - f.to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0.5 && i < 10 {
            println!(
                "  MISMATCH [{i}]: ref={:.4}, fused={:.4}, diff={diff:.2e}",
                r.to_f32(),
                f.to_f32()
            );
        }
    }
    println!("  M={m}, hidden={hidden}, intermediate={intermediate}: max_diff={max_diff:.2e}");
    assert!(max_diff < 0.1, "MLP block diff too large: {max_diff:.2e}");
    println!("PASS: MLP block matches separate launches");
}

// ══════════════════════════════════════════════════════════════════════
// Segment B: O_proj → norm → gate_up+SiLU → down in ONE kernel launch
// This is the full Segment B from the tile pipeline architecture.
// ══════════════════════════════════════════════════════════════════════

const SEGMENT_B_PTX: &str = ptx_fusion_macros::sequence_segment_b!(
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "kernels/vllm_rms_norm.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "kernels/vllm_silu_mul.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "segment_b"
);

#[test]
fn segment_b_gpu() {
    println!("=== Segment B (O_proj → norm → gate_up+SiLU → down): GPU correctness ===");

    // ptxas pre-check
    let path = "/tmp/segment_b.ptx";
    std::fs::write(path, SEGMENT_B_PTX).unwrap();
    let ptxas_out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");
    if !ptxas_out.status.success() {
        let stderr = String::from_utf8_lossy(&ptxas_out.stderr);
        eprintln!(
            "ptxas errors ({} total lines):",
            SEGMENT_B_PTX.lines().count()
        );
        for line in stderr.lines().take(20) {
            eprintln!("  {line}");
        }
        panic!("ptxas FAILED on Segment B PTX");
    }
    println!("  ptxas: PASS ({} lines)", SEGMENT_B_PTX.lines().count());

    let ctx = ctx();
    let stream = ctx.default_stream();

    // Small dims for testing (all same tile config 64x128x32)
    let m = 64u32;
    let q_dim = 128u32; // O_proj K dimension
    let hidden = 128u32; // hidden dim (O_proj N, norm dim, gate_up K)
    let intermediate = 128u32; // intermediate (gate_up N/2, down K)
    let gate_up_cols = 2 * intermediate;

    // Test data
    let h_attn_out: Vec<half::bf16> = (0..(m * q_dim) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.3))
        .collect();
    let h_oproj_w: Vec<half::bf16> = (0..(hidden * q_dim) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00023 + 0.3).cos() * 0.1))
        .collect();
    let h_norm_w: Vec<half::bf16> = (0..hidden as usize)
        .map(|i| half::bf16::from_f32(1.0 + (i as f32) * 0.001))
        .collect();
    let h_gateup_w: Vec<half::bf16> = (0..(gate_up_cols * hidden) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00019 - 0.2).sin() * 0.08))
        .collect();
    let h_down_w: Vec<half::bf16> = (0..(hidden * intermediate) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00013 + 0.1).cos() * 0.12))
        .collect();

    // Reference: 5 separate operations
    // 1. O_proj GEMM
    let oproj_out = run_flat_gemm(
        &ctx,
        FLAT_GEMM_BASE_PTX,
        "flat_gemm_base",
        &h_attn_out,
        &h_oproj_w,
        m,
        hidden,
        q_dim,
    );
    // 2. rms_norm (GPU)
    let rms_module = ctx.load_module(Ptx::from_src(RMS_NORM_BF16_PTX)).unwrap();
    let rms_func = rms_module
        .load_function("_Z15rms_norm_kernelI13__nv_bfloat16EvPT_PKS1_S4_fi")
        .unwrap();
    let d_oproj = stream.clone_htod(&oproj_out).unwrap();
    let d_norm_w = stream.clone_htod(&h_norm_w).unwrap();
    let mut d_normed: CudaSlice<half::bf16> = stream.alloc_zeros((m * hidden) as usize).unwrap();
    let (oproj_p, _) = d_oproj.device_ptr(&stream);
    let (nw_p, _) = d_norm_w.device_ptr(&stream);
    let (normed_p, _) = d_normed.device_ptr(&stream);
    let eps = 1e-5f32;
    unsafe {
        stream
            .launch_builder(&rms_func)
            .arg(&normed_p)
            .arg(&oproj_p)
            .arg(&nw_p)
            .arg(&eps)
            .arg(&(hidden as i32))
            .launch(LaunchConfig {
                grid_dim: (m, 1, 1),
                block_dim: (hidden.min(1024), 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .unwrap();
    stream.synchronize().unwrap();
    let normed = stream.clone_dtoh(&d_normed).unwrap();
    // 3. gate_up GEMM
    let gateup_out = run_flat_gemm(
        &ctx,
        FLAT_GEMM_BASE_PTX,
        "flat_gemm_base",
        &normed,
        &h_gateup_w,
        m,
        gate_up_cols,
        hidden,
    );
    // 4. SiLU+mul
    let activated: Vec<half::bf16> = (0..(m * intermediate) as usize)
        .map(|idx| {
            let row = idx / intermediate as usize;
            let col = idx % intermediate as usize;
            let gate = gateup_out[row * gate_up_cols as usize + col].to_f32();
            let up = gateup_out[row * gate_up_cols as usize + intermediate as usize + col].to_f32();
            half::bf16::from_f32(cpu_silu(gate) * up)
        })
        .collect();
    // 5. down GEMM
    let ref_output = run_flat_gemm(
        &ctx,
        FLAT_GEMM_BASE_PTX,
        "flat_gemm_base",
        &activated,
        &h_down_w,
        m,
        hidden,
        intermediate,
    );

    // Fused: single kernel launch (Segment B)
    let d_attn = stream.clone_htod(&h_attn_out).unwrap();
    let d_oproj_w = stream.clone_htod(&h_oproj_w).unwrap();
    let d_gateup_w = stream.clone_htod(&h_gateup_w).unwrap();
    let d_down_w = stream.clone_htod(&h_down_w).unwrap();

    // Intermediate buffers
    let d_oproj_buf: CudaSlice<half::bf16> = stream.alloc_zeros((m * hidden) as usize).unwrap();
    let d_gateup_buf: CudaSlice<half::bf16> =
        stream.alloc_zeros((m * gate_up_cols) as usize).unwrap();
    let d_c_scratch: CudaSlice<half::bf16> = stream.alloc_zeros((m * hidden) as usize).unwrap();
    let d_output: CudaSlice<half::bf16> = stream.alloc_zeros((m * hidden) as usize).unwrap();

    let module = ctx.load_module(Ptx::from_src(SEGMENT_B_PTX)).unwrap();
    let func = module.load_function("segment_b").unwrap();

    let (attn_p, _) = d_attn.device_ptr(&stream);
    let (opw_p, _) = d_oproj_w.device_ptr(&stream);
    let (ob_p, _) = d_oproj_buf.device_ptr(&stream);
    let (nw2_p, _) = d_norm_w.device_ptr(&stream);
    let (guw_p, _) = d_gateup_w.device_ptr(&stream);
    let (gub_p, _) = d_gateup_buf.device_ptr(&stream);
    let (dw_p, _) = d_down_w.device_ptr(&stream);
    let (cs_p, _) = d_c_scratch.device_ptr(&stream);
    let (out_p, _) = d_output.device_ptr(&stream);

    // Phase 0: O_proj params
    let params_oproj = build_flat_params(
        attn_p as u64,
        opw_p as u64,
        ob_p as u64,
        ob_p as u64,
        m,
        hidden,
        q_dim,
        q_dim,
        q_dim,
        hidden,
        hidden,
        1.0,
        0.0,
    );
    // Phase 1: norm+gate_up params (extra rms params + ferrite_params)
    let rms_input = ob_p as u64; // reads O_proj output
    let rms_weight = nw2_p as u64;
    let rms_hidden = hidden;
    let rms_stride = hidden as u64;
    let params_gateup = build_flat_params(
        ob_p as u64,
        guw_p as u64,
        gub_p as u64,
        gub_p as u64,
        m,
        gate_up_cols,
        hidden,
        hidden,
        hidden,
        gate_up_cols,
        gate_up_cols,
        1.0,
        0.0,
    );
    // Phase 2: SiLU+down params
    let intermediate_bytes = (intermediate as u64) * 2;
    let params_down = build_flat_params(
        gub_p as u64,
        dw_p as u64,
        cs_p as u64,
        out_p as u64,
        m,
        hidden,
        intermediate,
        gate_up_cols,
        intermediate,
        hidden,
        hidden,
        1.0,
        0.0,
    );

    let (gx, gy, _) = compute_grid(m, gate_up_cols, 64, 128);
    let cfg = LaunchConfig {
        grid_dim: (gx, gy, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 36864 + 512, // GEMM SMEM + inv_rms array
    };

    // Global barrier counters (2 barriers between 3 phases)
    let d_barriers: CudaSlice<u32> = stream.alloc_zeros(2).unwrap();
    let (seg_bar_ptr, _) = d_barriers.device_ptr(&stream);
    let seg_bar_val = seg_bar_ptr as u64;

    // Param order: _phase_barriers, phase0(ferrite_params), phase1(rms_input, rms_weight, rms_eps,
    //   rms_hidden, rms_stride, ferrite_params_2), phase2(intermediate_bytes_3, ferrite_params_3)
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&seg_bar_val)
            .arg(&params_oproj)
            .arg(&rms_input)
            .arg(&rms_weight)
            .arg(&eps)
            .arg(&rms_hidden)
            .arg(&rms_stride)
            .arg(&params_gateup)
            .arg(&intermediate_bytes)
            .arg(&params_down)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let seg_output = stream.clone_dtoh(&d_output).unwrap();

    // Compare (note: norm uses 128 threads in fused vs hidden threads in separate,
    // so FP accumulation differs. We compare against GPU separate which also has
    // this difference at the norm step. The gate_up+SiLU+down chain amplifies it.)
    let mut max_diff = 0.0f32;
    for (i, (r, s)) in ref_output.iter().zip(seg_output.iter()).enumerate() {
        let diff = (r.to_f32() - s.to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 1.0 && i < 10 {
            println!(
                "  MISMATCH [{i}]: ref={:.4}, seg={:.4}, diff={diff:.2e}",
                r.to_f32(),
                s.to_f32()
            );
        }
    }
    println!("  M={m}: max_diff={max_diff:.2e}");
    // Tolerance: norm runs with 128 threads (GEMM block) vs 128 threads (hidden=128),
    // so accumulation order is the same here. Expect near-zero.
    assert!(max_diff < 0.5, "Segment B diff too large: {max_diff:.2e}");
    println!("PASS: Segment B matches separate launches");
}

#[test]
fn segment_b_m_sweep() {
    println!("=== Segment B: M-sweep (multi-tile) ===");

    let ctx = ctx();
    let stream = ctx.default_stream();

    let q_dim = 128u32;
    let hidden = 128u32;
    let intermediate = 128u32;
    let gate_up_cols = 2 * intermediate;
    let eps = 1e-5f32;

    let module = ctx.load_module(Ptx::from_src(SEGMENT_B_PTX)).unwrap();
    let func = module.load_function("segment_b").unwrap();
    let rms_module = ctx.load_module(Ptx::from_src(RMS_NORM_BF16_PTX)).unwrap();
    let rms_func = rms_module
        .load_function("_Z15rms_norm_kernelI13__nv_bfloat16EvPT_PKS1_S4_fi")
        .unwrap();

    // Shared weights
    let h_oproj_w: Vec<half::bf16> = (0..(hidden * q_dim) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00023 + 0.3).cos() * 0.1))
        .collect();
    let h_norm_w: Vec<half::bf16> = (0..hidden as usize)
        .map(|i| half::bf16::from_f32(1.0 + (i as f32) * 0.001))
        .collect();
    let h_gateup_w: Vec<half::bf16> = (0..(gate_up_cols * hidden) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00019 - 0.2).sin() * 0.08))
        .collect();
    let h_down_w: Vec<half::bf16> = (0..(hidden * intermediate) as usize)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.00013 + 0.1).cos() * 0.12))
        .collect();
    let d_oproj_w = stream.clone_htod(&h_oproj_w).unwrap();
    let d_norm_w = stream.clone_htod(&h_norm_w).unwrap();
    let d_gateup_w = stream.clone_htod(&h_gateup_w).unwrap();
    let d_down_w = stream.clone_htod(&h_down_w).unwrap();

    let m_values = [1u32, 8, 32, 64, 65, 128, 256];
    let mut failures = Vec::new();

    for &m in &m_values {
        let h_input: Vec<half::bf16> = (0..(m * q_dim) as usize)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.3))
            .collect();

        // Reference: separate operations
        let oproj_out = run_flat_gemm(
            &ctx,
            FLAT_GEMM_BASE_PTX,
            "flat_gemm_base",
            &h_input,
            &h_oproj_w,
            m,
            hidden,
            q_dim,
        );
        // GPU norm
        let d_ref_oproj = stream.clone_htod(&oproj_out).unwrap();
        let mut d_ref_normed: CudaSlice<half::bf16> =
            stream.alloc_zeros((m * hidden) as usize).unwrap();
        let (ref_op, _) = d_ref_oproj.device_ptr(&stream);
        let (ref_nw, _) = d_norm_w.device_ptr(&stream);
        let (ref_no, _) = d_ref_normed.device_ptr(&stream);
        unsafe {
            stream
                .launch_builder(&rms_func)
                .arg(&ref_no)
                .arg(&ref_op)
                .arg(&ref_nw)
                .arg(&eps)
                .arg(&(hidden as i32))
                .launch(LaunchConfig {
                    grid_dim: (m, 1, 1),
                    block_dim: (hidden.min(1024), 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .unwrap();
        stream.synchronize().unwrap();
        let normed = stream.clone_dtoh(&d_ref_normed).unwrap();
        let gateup_out = run_flat_gemm(
            &ctx,
            FLAT_GEMM_BASE_PTX,
            "flat_gemm_base",
            &normed,
            &h_gateup_w,
            m,
            gate_up_cols,
            hidden,
        );
        let activated: Vec<half::bf16> = (0..(m * intermediate) as usize)
            .map(|idx| {
                let row = idx / intermediate as usize;
                let col = idx % intermediate as usize;
                let gate = gateup_out[row * gate_up_cols as usize + col].to_f32();
                let up =
                    gateup_out[row * gate_up_cols as usize + intermediate as usize + col].to_f32();
                half::bf16::from_f32(cpu_silu(gate) * up)
            })
            .collect();
        let ref_output = run_flat_gemm(
            &ctx,
            FLAT_GEMM_BASE_PTX,
            "flat_gemm_base",
            &activated,
            &h_down_w,
            m,
            hidden,
            intermediate,
        );

        // Fused Segment B
        let d_input = stream.clone_htod(&h_input).unwrap();
        let d_oproj_buf: CudaSlice<half::bf16> = stream.alloc_zeros((m * hidden) as usize).unwrap();
        let d_gateup_buf: CudaSlice<half::bf16> =
            stream.alloc_zeros((m * gate_up_cols) as usize).unwrap();
        let d_c_scratch: CudaSlice<half::bf16> = stream.alloc_zeros((m * hidden) as usize).unwrap();
        let d_output: CudaSlice<half::bf16> = stream.alloc_zeros((m * hidden) as usize).unwrap();
        let d_barriers: CudaSlice<u32> = stream.alloc_zeros(2).unwrap();

        let (inp_p, _) = d_input.device_ptr(&stream);
        let (opw_p, _) = d_oproj_w.device_ptr(&stream);
        let (ob_p, _) = d_oproj_buf.device_ptr(&stream);
        let (nw_p, _) = d_norm_w.device_ptr(&stream);
        let (guw_p, _) = d_gateup_w.device_ptr(&stream);
        let (gub_p, _) = d_gateup_buf.device_ptr(&stream);
        let (dw_p, _) = d_down_w.device_ptr(&stream);
        let (cs_p, _) = d_c_scratch.device_ptr(&stream);
        let (out_p, _) = d_output.device_ptr(&stream);
        let (bar_p, _) = d_barriers.device_ptr(&stream);
        let bar_val = bar_p as u64;

        let params_oproj = build_flat_params(
            inp_p as u64,
            opw_p as u64,
            ob_p as u64,
            ob_p as u64,
            m,
            hidden,
            q_dim,
            q_dim,
            q_dim,
            hidden,
            hidden,
            1.0,
            0.0,
        );
        let rms_input_val = ob_p as u64;
        let rms_weight_val = nw_p as u64;
        let rms_hidden_val = hidden;
        let rms_stride_val = hidden as u64;
        let params_gateup = build_flat_params(
            ob_p as u64,
            guw_p as u64,
            gub_p as u64,
            gub_p as u64,
            m,
            gate_up_cols,
            hidden,
            hidden,
            hidden,
            gate_up_cols,
            gate_up_cols,
            1.0,
            0.0,
        );
        let intermediate_bytes = (intermediate as u64) * 2;
        let params_down = build_flat_params(
            gub_p as u64,
            dw_p as u64,
            cs_p as u64,
            out_p as u64,
            m,
            hidden,
            intermediate,
            gate_up_cols,
            intermediate,
            hidden,
            hidden,
            1.0,
            0.0,
        );

        let (gx, gy, _) = compute_grid(m, gate_up_cols, 64, 128);
        let cfg = LaunchConfig {
            grid_dim: (gx, gy, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 36864 + 512,
        };

        unsafe {
            stream
                .launch_builder(&func)
                .arg(&bar_val)
                .arg(&params_oproj)
                .arg(&rms_input_val)
                .arg(&rms_weight_val)
                .arg(&eps)
                .arg(&rms_hidden_val)
                .arg(&rms_stride_val)
                .arg(&params_gateup)
                .arg(&intermediate_bytes)
                .arg(&params_down)
                .launch(cfg)
        }
        .unwrap();
        stream.synchronize().unwrap();
        let seg_output = stream.clone_dtoh(&d_output).unwrap();

        let mut max_diff = 0.0f32;
        for (r, s) in ref_output.iter().zip(seg_output.iter()) {
            let diff = (r.to_f32() - s.to_f32()).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        let status = if max_diff < 0.5 { "PASS" } else { "FAIL" };
        println!("  M={m:>4} grid=({gx},{gy},1) max_diff={max_diff:.2e} {status}");
        if max_diff >= 0.5 {
            failures.push((m, max_diff));
        }
    }
    assert!(
        failures.is_empty(),
        "Segment B M-sweep failures: {:?}",
        failures
    );
    println!("PASS: Segment B all M values correct");
}

// ══════════════════════════════════════════════════════════════════════
// Persistent GEMM: work-queue loop with 2D ctaid dispatch
// ══════════════════════════════════════════════════════════════════════

const PERSISTENT_GEMM_PTX: &str = ptx_fusion_macros::persistent_gemm!(
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "persistent_gemm_test"
);

#[test]
fn persistent_gemm_gpu() {
    println!("=== Persistent GEMM: GPU correctness (2D ctaid) ===");

    // ptxas check
    let path = "/tmp/persistent_gemm.ptx";
    std::fs::write(path, PERSISTENT_GEMM_PTX).unwrap();
    let ptxas_out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");
    if !ptxas_out.status.success() {
        let stderr = String::from_utf8_lossy(&ptxas_out.stderr);
        eprintln!("ptxas errors:");
        for line in stderr.lines().take(20) {
            eprintln!("  {line}");
        }
        panic!("ptxas FAILED on persistent GEMM");
    }
    println!(
        "  ptxas: PASS ({} lines)",
        PERSISTENT_GEMM_PTX.lines().count()
    );

    let ctx = ctx();
    let stream = ctx.default_stream();

    let module = ctx.load_module(Ptx::from_src(PERSISTENT_GEMM_PTX)).unwrap();
    let func = module.load_function("persistent_gemm_test").unwrap();

    let num_sms = 58u32; // L4

    // Test at multiple (M, N) values to exercise both ctaid.x and ctaid.y
    let k = 128u32;
    let cases: &[(u32, u32)] = &[
        (1, 128),   // 1 tile, trivial
        (64, 128),  // 1 m-tile, 1 n-tile
        (64, 256),  // 1 m-tile, 2 n-tiles (swizzle_log=1, gy=1)
        (128, 256), // 2 m-tiles, 2 n-tiles
        (64, 640),  // 1 m-tile, 5 n-tiles (swizzle_log=2, gy=2 -- exercises ctaid.y!)
        (256, 640), // 4 m-tiles, 5 n-tiles (gy=2, many tiles)
    ];
    let mut failures = Vec::new();

    for &(m, n) in cases {
        let h_a: Vec<half::bf16> = (0..(m * k) as usize)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.3))
            .collect();
        let h_b: Vec<half::bf16> = (0..(n * k) as usize)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.00023 + 0.3).cos() * 0.1))
            .collect();

        // Reference: standard flat GEMM
        let ref_out = run_flat_gemm(
            &ctx,
            FLAT_GEMM_BASE_PTX,
            "flat_gemm_base",
            &h_a,
            &h_b,
            m,
            n,
            k,
        );

        // Persistent GEMM
        let d_a = stream.clone_htod(&h_a).unwrap();
        let d_b = stream.clone_htod(&h_b).unwrap();
        let d_c: CudaSlice<half::bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
        let d_d: CudaSlice<half::bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
        let d_counter: CudaSlice<u32> = stream.alloc_zeros(1).unwrap();

        let (a_ptr, _) = d_a.device_ptr(&stream);
        let (b_ptr, _) = d_b.device_ptr(&stream);
        let (c_ptr, _) = d_c.device_ptr(&stream);
        let (d_ptr, _) = d_d.device_ptr(&stream);
        let (ctr_ptr, _) = d_counter.device_ptr(&stream);

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

        let (gx, gy, _) = compute_grid(m, n, 64, 128);
        let total_tiles = gx * gy;
        let num_blocks = total_tiles.min(num_sms);
        let ctr_val = ctr_ptr as u64;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 36864,
        };

        // Param order: _persistent_counter, _persistent_grid_x, _persistent_total, ferrite_params
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&ctr_val)
                .arg(&gx)
                .arg(&total_tiles)
                .arg(&params)
                .launch(cfg)
        }
        .unwrap();
        stream.synchronize().unwrap();
        let pers_out = stream.clone_dtoh(&d_d).unwrap();

        let mut max_diff = 0.0f32;
        for (r, p) in ref_out.iter().zip(pers_out.iter()) {
            let diff = (r.to_f32() - p.to_f32()).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        let status = if max_diff < 0.01 { "PASS" } else { "FAIL" };
        println!(
            "  M={m:>4} grid=({gx},{gy}) tiles={total_tiles} blocks={num_blocks} max_diff={max_diff:.2e} {status}"
        );
        if max_diff >= 0.01 {
            failures.push((m, max_diff));
        }
    }
    assert!(
        failures.is_empty(),
        "Persistent GEMM failures: {:?}",
        failures
    );
    println!("PASS: persistent GEMM matches standard at all M values");
}

// ══════════════════════════════════════════════════════════════════════
// Persistent MLP block: gate_up + SiLU + down with per-M-tile barriers
// ══════════════════════════════════════════════════════════════════════

const PERSISTENT_MLP_PTX: &str = ptx_fusion_macros::persistent_mlp_block!(
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "kernels/vllm_silu_mul.ptx",
    "persistent_mlp"
);

#[test]
fn persistent_mlp_block_gpu() {
    println!("=== Persistent MLP block: GPU correctness ===");

    let path = "/tmp/persistent_mlp.ptx";
    std::fs::write(path, PERSISTENT_MLP_PTX).unwrap();
    let ptxas_out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");
    if !ptxas_out.status.success() {
        let stderr = String::from_utf8_lossy(&ptxas_out.stderr);
        eprintln!("ptxas errors:");
        for line in stderr.lines().take(20) {
            eprintln!("  {line}");
        }
        panic!("ptxas FAILED on persistent MLP");
    }
    println!(
        "  ptxas: PASS ({} lines)",
        PERSISTENT_MLP_PTX.lines().count()
    );

    let ctx = ctx();
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_src(PERSISTENT_MLP_PTX)).unwrap();
    let func = module.load_function("persistent_mlp").unwrap();

    let num_sms = 58u32;
    let hidden = 128u32;
    let intermediate = 128u32;
    let gate_up_cols = 2 * intermediate;

    let m_values = [1u32, 8, 64, 128, 256];
    let mut failures = Vec::new();

    for &m in &m_values {
        let h_input: Vec<half::bf16> = (0..(m * hidden) as usize)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.3))
            .collect();
        let h_gateup_w: Vec<half::bf16> = (0..(gate_up_cols * hidden) as usize)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.00019 - 0.2).sin() * 0.08))
            .collect();
        let h_down_w: Vec<half::bf16> = (0..(hidden * intermediate) as usize)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.00013 + 0.1).cos() * 0.12))
            .collect();

        // Reference: 3 separate launches
        let gate_up_out = run_flat_gemm(
            &ctx,
            FLAT_GEMM_BASE_PTX,
            "flat_gemm_base",
            &h_input,
            &h_gateup_w,
            m,
            gate_up_cols,
            hidden,
        );
        let activated: Vec<half::bf16> = (0..(m * intermediate) as usize)
            .map(|idx| {
                let row = idx / intermediate as usize;
                let col = idx % intermediate as usize;
                let gate = gate_up_out[row * gate_up_cols as usize + col].to_f32();
                let up =
                    gate_up_out[row * gate_up_cols as usize + intermediate as usize + col].to_f32();
                half::bf16::from_f32(cpu_silu(gate) * up)
            })
            .collect();
        let ref_output = run_flat_gemm(
            &ctx,
            FLAT_GEMM_BASE_PTX,
            "flat_gemm_base",
            &activated,
            &h_down_w,
            m,
            hidden,
            intermediate,
        );

        // Persistent fused MLP
        let d_input = stream.clone_htod(&h_input).unwrap();
        let d_gateup_w = stream.clone_htod(&h_gateup_w).unwrap();
        let d_down_w = stream.clone_htod(&h_down_w).unwrap();
        let d_gateup_buf: CudaSlice<half::bf16> =
            stream.alloc_zeros((m * gate_up_cols) as usize).unwrap();
        let d_output: CudaSlice<half::bf16> = stream.alloc_zeros((m * hidden) as usize).unwrap();
        let d_counter: CudaSlice<u32> = stream.alloc_zeros(1).unwrap();
        // M-tile completion counters (one per M-tile)
        let max_mtiles = m.div_ceil(64);
        let d_mtile_done: CudaSlice<u32> = stream.alloc_zeros(max_mtiles as usize).unwrap();

        let (inp_p, _) = d_input.device_ptr(&stream);
        let (guw_p, _) = d_gateup_w.device_ptr(&stream);
        let (gub_p, _) = d_gateup_buf.device_ptr(&stream);
        let (dw_p, _) = d_down_w.device_ptr(&stream);
        let (out_p, _) = d_output.device_ptr(&stream);
        let (ctr_p, _) = d_counter.device_ptr(&stream);
        let (mtd_p, _) = d_mtile_done.device_ptr(&stream);

        // Phase 0: gate_up params
        let params_gate_up = build_flat_params(
            inp_p as u64,
            guw_p as u64,
            gub_p as u64,
            gub_p as u64,
            m,
            gate_up_cols,
            hidden,
            hidden,
            hidden,
            gate_up_cols,
            gate_up_cols,
            1.0,
            0.0,
        );
        // Phase 1: SiLU-fused down params
        let intermediate_bytes = (intermediate as u64) * 2;
        let params_down = build_flat_params(
            gub_p as u64,
            dw_p as u64,
            out_p as u64,
            out_p as u64,
            m,
            hidden,
            intermediate,
            gate_up_cols,
            intermediate,
            hidden,
            hidden,
            1.0,
            0.0,
        );

        // Grid dims for each phase
        let (gx0, gy0, _) = compute_grid(m, gate_up_cols, 64, 128);
        let (gx1, gy1, _) = compute_grid(m, hidden, 64, 128);
        let total_phase0 = gx0 * gy0;
        let total_phase1 = gx1 * gy1;
        let total_tiles = total_phase0 + total_phase1;
        let num_blocks = total_tiles.min(num_sms);

        // N-tiles per M-tile for phase 0 barrier
        let grid_n_0 = gate_up_cols.div_ceil(128);
        let swizzle_log_0 = compute_swizzle_log(grid_n_0);
        let swizzle_tile_0 = 1u32 << swizzle_log_0;
        let ntiles_per_m_0 = swizzle_tile_0 * gy0; // tiles in one M-tile's column strip

        let ctr_val = ctr_p as u64;
        let mtd_val = mtd_p as u64;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 36864 + 512,
        };

        // Param order: persistent params, then phase params
        // _persistent_counter, _persistent_total, _persistent_grid_x_0,
        // _persistent_phase0_tiles, _persistent_grid_x_1,
        // _persistent_ntiles_per_m_0, _persistent_mtile_done,
        // ferrite_params (gate_up), _ferrite_intermediate_bytes_2, ferrite_params_2 (down)
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&ctr_val)
                .arg(&total_tiles)
                .arg(&gx0)
                .arg(&total_phase0)
                .arg(&gx1)
                .arg(&ntiles_per_m_0)
                .arg(&mtd_val)
                .arg(&params_gate_up)
                .arg(&intermediate_bytes)
                .arg(&params_down)
                .launch(cfg)
        }
        .unwrap();
        stream.synchronize().unwrap();
        let pers_output = stream.clone_dtoh(&d_output).unwrap();

        let mut max_diff = 0.0f32;
        for (r, p) in ref_output.iter().zip(pers_output.iter()) {
            let diff = (r.to_f32() - p.to_f32()).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        let status = if max_diff < 0.1 { "PASS" } else { "FAIL" };
        println!(
            "  M={m:>4} tiles=({total_phase0}+{total_phase1}={total_tiles}) blocks={num_blocks} max_diff={max_diff:.2e} {status}"
        );
        if max_diff >= 0.1 {
            failures.push((m, max_diff));
        }
    }
    assert!(
        failures.is_empty(),
        "Persistent MLP failures: {:?}",
        failures
    );
    println!("PASS: persistent MLP block matches separate launches");
}

fn compute_swizzle_log(grid_n: u32) -> u32 {
    const SWIZZLE_N: u32 = 4;
    if SWIZZLE_N >= 8 && grid_n >= 6 {
        3
    } else if SWIZZLE_N >= 4 && grid_n >= 3 {
        2
    } else if SWIZZLE_N >= 2 && grid_n >= 2 {
        1
    } else {
        0
    }
}
