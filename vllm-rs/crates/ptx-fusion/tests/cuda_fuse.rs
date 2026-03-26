//! CUDA test: verify that fused kernel produces same output as running two kernels sequentially.
//!
//! Run with: cargo test -p ptx-fusion --features cuda -- --nocapture

#![cfg(feature = "cuda")]
#![allow(clippy::too_many_arguments)]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ptx_fusion_macros::fuse_kernels;
use std::sync::Arc;

const RMS_NORM_PTX: &str = include_str!("../kernels/rms_norm.ptx");
const MATVEC_PTX: &str = include_str!("../kernels/matvec.ptx");

// Fuse rms_norm -> matvec: output of rms_norm feeds vec_in of matvec via SMEM
fuse_kernels!(
    "kernels/rms_norm.ptx",
    "kernels/matvec.ptx",
    "fused_rms_norm_matvec",
    "output",
    "vec_in"
);
// ^ emits FUSED_RMS_NORM_MATVEC: &str

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

/// Run rms_norm and matvec as two separate kernel launches.
/// Returns the final output of matvec.
fn run_separate(
    ctx: &Arc<CudaContext>,
    input: &[f32],
    weight: &[f32],
    matrix: &[f32],
    n: u32,
    epsilon: f32,
    m: u32,
    k: u32,
) -> Vec<f32> {
    let stream = ctx.default_stream();

    // Load both modules
    let mod_rms = ctx
        .load_module(Ptx::from_src(RMS_NORM_PTX))
        .expect("load rms_norm");
    let mod_mv = ctx
        .load_module(Ptx::from_src(MATVEC_PTX))
        .expect("load matvec");
    let func_rms = mod_rms.load_function("rms_norm").unwrap();
    let func_mv = mod_mv.load_function("matvec").unwrap();

    // Allocate
    let input_gpu = stream.clone_htod(input).unwrap();
    let weight_gpu = stream.clone_htod(weight).unwrap();
    let matrix_gpu = stream.clone_htod(matrix).unwrap();
    let mut intermediate_gpu: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();
    let mut output_gpu: CudaSlice<f32> = stream.alloc_zeros(m as usize).unwrap();

    // Launch rms_norm: input -> intermediate
    let cfg_rms = LaunchConfig::for_num_elems(n);
    unsafe {
        stream
            .launch_builder(&func_rms)
            .arg(&input_gpu)
            .arg(&mut intermediate_gpu)
            .arg(&weight_gpu)
            .arg(&n)
            .arg(&epsilon)
            .launch(cfg_rms)
    }
    .expect("rms_norm launch");

    // Launch matvec: matrix * intermediate -> output
    let block_size = k.max(64);
    let grid_size = m.div_ceil(block_size);
    let cfg_mv = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func_mv)
            .arg(&matrix_gpu)
            .arg(&intermediate_gpu)
            .arg(&mut output_gpu)
            .arg(&m)
            .arg(&k)
            .launch(cfg_mv)
    }
    .expect("matvec launch");

    stream.synchronize().unwrap();
    stream.clone_dtoh(&output_gpu).unwrap()
}

/// Run the fused kernel (single launch).
fn run_fused(
    ctx: &Arc<CudaContext>,
    input: &[f32],
    weight: &[f32],
    matrix: &[f32],
    n: u32,
    epsilon: f32,
    m: u32,
    k: u32,
) -> Vec<f32> {
    let stream = ctx.default_stream();

    let module = ctx
        .load_module(Ptx::from_src(FUSED_RMS_NORM_MATVEC))
        .expect("load fused kernel");
    let func = module.load_function("fused_rms_norm_matvec").unwrap();

    let input_gpu = stream.clone_htod(input).unwrap();
    let weight_gpu = stream.clone_htod(weight).unwrap();
    let matrix_gpu = stream.clone_htod(matrix).unwrap();
    let mut output_gpu: CudaSlice<f32> = stream.alloc_zeros(m as usize).unwrap();

    // Fused params: input, weight, n, epsilon, matrix, vec_out, M, K
    // We need the block to have at least K threads for the matvec SMEM load
    let block_size = k.max(64);
    let grid_size = 1u32; // single block for this test (n <= block_size)
    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&input_gpu)
            .arg(&weight_gpu)
            .arg(&n)
            .arg(&epsilon)
            .arg(&matrix_gpu)
            .arg(&mut output_gpu)
            .arg(&m)
            .arg(&k)
            .launch(cfg)
    }
    .expect("fused kernel launch");

    stream.synchronize().unwrap();
    stream.clone_dtoh(&output_gpu).unwrap()
}

#[test]
fn fused_matches_separate() {
    let ctx = get_ctx();

    // Dimensions: rms_norm operates on n elements, matvec does M×K matrix * K-vec → M-vec
    // The intermediate (rms_norm output = matvec vec_in) has K elements.
    let k = 128u32; // inner dim = rms_norm n
    let m = 64u32; // output rows
    let n = k; // rms_norm operates on K elements
    let epsilon = 1e-5f32;

    // Test data
    let input: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let matrix: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();

    // Run separate
    let output_separate = run_separate(&ctx, &input, &weight, &matrix, n, epsilon, m, k);

    // Run fused
    let output_fused = run_fused(&ctx, &input, &weight, &matrix, n, epsilon, m, k);

    println!("Separate output: {:?}...", &output_separate[..4]);
    println!("Fused output:    {:?}...", &output_fused[..4]);

    // Sanity: output isn't all zeros
    let sum_sep: f32 = output_separate.iter().sum();
    assert!(
        sum_sep.abs() > 0.01,
        "separate output is all zeros (sum={sum_sep})"
    );
    let sum_fused: f32 = output_fused.iter().sum();
    assert!(
        sum_fused.abs() > 0.01,
        "fused output is all zeros (sum={sum_fused})"
    );

    // Compare: should match within FP tolerance
    assert_f32_close(&output_separate, &output_fused, "fused_vs_separate", 1e-4);

    println!("PASS: fused kernel matches separate launches (M={m}, K={k}, tol=1e-4)");
}
