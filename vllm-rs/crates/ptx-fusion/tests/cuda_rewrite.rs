//! CUDA tests: verify that register-rewritten PTX produces identical results.
//!
//! Run with: cargo test -p ptx-fusion --features cuda -- --nocapture

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ptx_fusion_macros::rewrite_kernel;
use std::sync::Arc;

// ── Compile-time: extract original + rewritten PTX ──────────────────

// Original PTX as const strings
const RMS_NORM_PTX: &str = include_str!("../kernels/rms_norm.ptx");
const MATVEC_PTX: &str = include_str!("../kernels/matvec.ptx");

// Rewritten PTX (register renames)
rewrite_kernel!("kernels/rms_norm.ptx", {
    "%f3" => "%f30",
    "%r3" => "%r30",
    "%rd3" => "%rd30"
});
// ^ emits RMS_NORM_REWRITTEN: &str and RMS_NORM_REWRITTEN_PROTOCOL

rewrite_kernel!("kernels/matvec.ptx", {
    "%f1" => "%f50",
    "%r5" => "%r50",
    "%rd3" => "%rd50"
});
// ^ emits MATVEC_REWRITTEN: &str and MATVEC_REWRITTEN_PROTOCOL

// ── Helpers ─────────────────────────────────────────────────────────

fn get_ctx() -> Arc<CudaContext> {
    CudaContext::new(0).expect("no CUDA device available")
}

fn assert_f32_eq(a: &[f32], b: &[f32], kernel_name: &str, tol: f32) {
    assert_eq!(a.len(), b.len(), "{kernel_name}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff <= tol,
            "{kernel_name}: mismatch at [{i}]: original={x}, rewritten={y}, diff={diff}"
        );
    }
}

// ── Test: RMSNorm original vs rewritten ─────────────────────────────

#[test]
fn rms_norm_rewrite_matches() {
    let ctx = get_ctx();
    let n = 256u32;
    let epsilon = 1e-5f32;

    // Create test data
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let weight_data: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();

    // Run original kernel
    let output_orig = run_rms_norm(&ctx, RMS_NORM_PTX, &input_data, &weight_data, n, epsilon);

    // Run rewritten kernel
    let output_rewritten = run_rms_norm(
        &ctx,
        RMS_NORM_REWRITTEN,
        &input_data,
        &weight_data,
        n,
        epsilon,
    );

    assert_f32_eq(&output_orig, &output_rewritten, "rms_norm", 0.0);
    println!("PASS: rms_norm rewritten output matches original (n={n})");

    // Sanity: check the output isn't all zeros
    let sum: f32 = output_orig.iter().sum();
    assert!(
        sum.abs() > 1.0,
        "rms_norm: output looks like all zeros (sum={sum})"
    );
    println!("  output sample: {:?}...", &output_orig[..4]);
}

fn run_rms_norm(
    ctx: &Arc<CudaContext>,
    ptx: &str,
    input: &[f32],
    weight: &[f32],
    n: u32,
    epsilon: f32,
) -> Vec<f32> {
    let stream = ctx.default_stream();
    let module = ctx
        .load_module(Ptx::from_src(ptx))
        .expect("failed to load PTX");
    let func = module.load_function("rms_norm").unwrap();

    let input_gpu = stream.clone_htod(input).unwrap();
    let weight_gpu = stream.clone_htod(weight).unwrap();
    let mut output_gpu: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();

    let cfg = LaunchConfig::for_num_elems(n);
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&input_gpu)
            .arg(&mut output_gpu)
            .arg(&weight_gpu)
            .arg(&n)
            .arg(&epsilon)
            .launch(cfg)
    }
    .expect("kernel launch failed");

    stream.synchronize().unwrap();
    stream.clone_dtoh(&output_gpu).unwrap()
}

// ── Test: Matvec original vs rewritten ──────────────────────────────

#[test]
fn matvec_rewrite_matches() {
    let ctx = get_ctx();
    let m = 64u32; // output rows
    let k = 128u32; // inner dimension

    // Create test data
    let matrix_data: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let vec_data: Vec<f32> = (0..k).map(|i| (i as f32 + 1.0) * 0.1).collect();

    // Run original kernel
    let output_orig = run_matvec(&ctx, MATVEC_PTX, &matrix_data, &vec_data, m, k);

    // Run rewritten kernel
    let output_rewritten = run_matvec(&ctx, MATVEC_REWRITTEN, &matrix_data, &vec_data, m, k);

    // FMA ordering might differ very slightly, use small tolerance
    assert_f32_eq(&output_orig, &output_rewritten, "matvec", 0.0);
    println!("PASS: matvec rewritten output matches original (M={m}, K={k})");

    // Sanity: check the output isn't all zeros
    let sum: f32 = output_orig.iter().sum();
    assert!(
        sum.abs() > 0.1,
        "matvec: output looks like all zeros (sum={sum})"
    );
    println!("  output sample: {:?}...", &output_orig[..4]);
}

fn run_matvec(
    ctx: &Arc<CudaContext>,
    ptx: &str,
    matrix: &[f32],
    vec_in: &[f32],
    m: u32,
    k: u32,
) -> Vec<f32> {
    let stream = ctx.default_stream();
    let module = ctx
        .load_module(Ptx::from_src(ptx))
        .expect("failed to load PTX");
    let func = module.load_function("matvec").unwrap();

    let matrix_gpu = stream.clone_htod(matrix).unwrap();
    let vec_gpu = stream.clone_htod(vec_in).unwrap();
    let mut output_gpu: CudaSlice<f32> = stream.alloc_zeros(m as usize).unwrap();

    // Need at least K threads per block for the shared memory vector load
    let block_size = k.max(64);
    let grid_size = m.div_ceil(block_size);
    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0, // statically allocated in PTX
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&matrix_gpu)
            .arg(&vec_gpu)
            .arg(&mut output_gpu)
            .arg(&m)
            .arg(&k)
            .launch(cfg)
    }
    .expect("kernel launch failed");

    stream.synchronize().unwrap();
    stream.clone_dtoh(&output_gpu).unwrap()
}
