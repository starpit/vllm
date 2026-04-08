// SPDX-License-Identifier: Apache-2.0
//! Phase 3b/3c — end-to-end validation of the scheduled megakernel.
//!
//! Tests:
//!   1. `scheduled_megakernel_executes_topologically` (Phase 3b validation)
//!      — every node executed exactly once, ticks respect topological order.
//!   2. `tile_attn_norm_matches_cpu_golden` (Phase 3c step 1)
//!      — the real RMS-norm tile body produces output matching a CPU reference
//!      for every (layer, row) tile.

#![cfg(feature = "cuda")]

use cudarc::driver::result;
use half::bf16;
use vllm_tk_macros_core::{
    SCHEDULED_PREFILL_TINY_CTAS, reified_dag::ReifiedDag, reified_dag::TileSizes,
    scheduled_prefill_tiny_dims,
};
use vllm_tk_test_harness::ffi;

fn init_cuda() {
    result::init().expect("cuInit failed");
    let device = result::device::get(0).expect("cuDeviceGet failed");
    let ctx = unsafe { result::primary_ctx::retain(device) }.expect("cuCtxRetain failed");
    unsafe { result::ctx::set_current(ctx) }.expect("cuCtxSetCurrent failed");
}

fn gpu_alloc_zeros(bytes: usize) -> u64 {
    unsafe {
        let dptr = result::malloc_sync(bytes).expect("cuMemAlloc failed");
        result::memset_d8_sync(dptr, 0, bytes).expect("cuMemsetD8 failed");
        dptr
    }
}

fn gpu_alloc_zeros_u32(count: usize) -> *mut u32 {
    gpu_alloc_zeros(count * 4) as *mut u32
}

fn gpu_upload_bf16(host: &[bf16]) -> u64 {
    unsafe {
        let bytes = host.len() * 2;
        let dptr = result::malloc_sync(bytes).expect("cuMemAlloc failed");
        // Reinterpret as u16 for the cudarc upload helper.
        let host_u16: &[u16] = std::slice::from_raw_parts(host.as_ptr() as *const u16, host.len());
        result::memcpy_htod_sync(dptr, host_u16).expect("cuMemcpyHtoD failed");
        dptr
    }
}

fn gpu_download_bf16(dptr: u64, count: usize) -> Vec<bf16> {
    unsafe {
        let mut host_u16 = vec![0u16; count];
        result::memcpy_dtoh_sync(&mut host_u16, dptr as cudarc::driver::sys::CUdeviceptr)
            .expect("cuMemcpyDtoH failed");
        host_u16.into_iter().map(|u| bf16::from_bits(u)).collect()
    }
}

fn gpu_read_u32(ptr: *const u32, count: usize) -> Vec<u32> {
    let mut host = vec![0u32; count];
    unsafe {
        result::memcpy_dtoh_sync(&mut host, ptr as cudarc::driver::sys::CUdeviceptr)
            .expect("cuMemcpyDtoH failed");
    }
    host
}

/// Deterministic random bf16 fill — same seed → same output, useful for
/// reproducibility across runs.
fn random_bf16(n: usize, seed: u64, scale: f32) -> Vec<bf16> {
    let mut rng = seed;
    (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = ((rng >> 33) as i32 as f32) / (i32::MAX as f32) * scale;
            bf16::from_f32(v)
        })
        .collect()
}

/// CPU reference for RMS norm. y[c] = x[c] * w[c] / sqrt(mean(x^2) + eps).
/// Computes in fp32 to match the GPU path.
fn cpu_rms_norm(x: &[bf16], w: &[bf16], eps: f32) -> Vec<bf16> {
    let n = x.len();
    let mean_sq: f32 = x
        .iter()
        .map(|v| {
            let f = v.to_f32();
            f * f
        })
        .sum::<f32>()
        / n as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(w.iter())
        .map(|(xv, wv)| bf16::from_f32(xv.to_f32() * scale * wv.to_f32()))
        .collect()
}

struct TestBuffers {
    hidden_states: u64,
    rms_rope: u64,
    qkv: u64,
    attn_out: u64,
    rms_gate: u64,
    silu_out: u64,
    attn_norm_w: u64,
    mlp_norm_w: u64,
}

fn launch_with_buffers(b: &TestBuffers, eps: f32) -> (Vec<u32>, u32) {
    let dims = scheduled_prefill_tiny_dims();
    let dag = ReifiedDag::reify_llama(dims, TileSizes::default_v1());
    let n = dag.nodes.len();

    let flags = gpu_alloc_zeros_u32(n);
    let tick = gpu_alloc_zeros_u32(1);
    let barrier = gpu_alloc_zeros_u32(1);

    unsafe {
        ffi::launch_scheduled_megakernel(
            b.hidden_states as *mut _,
            b.rms_rope as *mut _,
            b.qkv as *mut _,
            b.attn_out as *mut _,
            b.rms_gate as *mut _,
            b.silu_out as *mut _,
            b.attn_norm_w as *mut _,
            b.mlp_norm_w as *mut _,
            eps,
            flags,
            tick,
            barrier,
            std::ptr::null_mut(),
        );
        result::stream::synchronize(std::ptr::null_mut()).expect("stream sync failed");
    }

    let ticks = gpu_read_u32(flags, n);
    let final_tick = gpu_read_u32(tick, 1)[0];
    (ticks, final_tick)
}

#[test]
#[ignore = "needs GPU"]
fn scheduled_megakernel_executes_topologically() {
    init_cuda();
    let dims = scheduled_prefill_tiny_dims();
    let dag = ReifiedDag::reify_llama(dims, TileSizes::default_v1());
    let n = dag.nodes.len();

    let kernel_n = unsafe { ffi::scheduled_megakernel_num_nodes() } as usize;
    assert_eq!(n, kernel_n);
    let kernel_ctas = unsafe { ffi::scheduled_megakernel_num_ctas() };
    assert_eq!(kernel_ctas, SCHEDULED_PREFILL_TINY_CTAS);
    let kernel_waves = unsafe { ffi::scheduled_megakernel_num_waves() };
    eprintln!("scheduled_megakernel: {n} nodes, {kernel_waves} waves, {kernel_ctas} CTAs");

    let (b, _) = build_test_buffers();
    let (ticks, final_tick) = launch_with_buffers(&b, 1e-5);
    eprintln!("final tick = {final_tick}");

    // Invariant 1: every node executed exactly once.
    let unexecuted: Vec<usize> = ticks
        .iter()
        .enumerate()
        .filter_map(|(i, &t)| if t == 0 { Some(i) } else { None })
        .collect();
    assert!(
        unexecuted.is_empty(),
        "{} nodes did not execute",
        unexecuted.len()
    );
    assert_eq!(final_tick as usize, n);

    // Invariant 2: topological order.
    for nd in &dag.nodes {
        let my_tick = ticks[nd.id.0 as usize];
        for d in &nd.deps {
            let dep_tick = ticks[d.0 as usize];
            assert!(
                my_tick > dep_tick,
                "topological violation: node {} (tick={}) ran before dep {} (tick={})",
                nd.id.0,
                my_tick,
                d.0,
                dep_tick
            );
        }
    }
}

/// Helper: validate that GPU output matches the CPU golden row-by-row.
/// The golden is computed by `golden_fn(row_idx)` so each test can supply
/// its own per-row reference.
fn assert_matches_golden<F: Fn(usize) -> Vec<bf16>>(
    gpu: &[bf16],
    seq: usize,
    hd: usize,
    label: &str,
    golden_fn: F,
) {
    let mut max_abs_err = 0.0_f32;
    let mut max_rel_err = 0.0_f32;
    for r in 0..seq {
        let golden = golden_fn(r);
        for c in 0..hd {
            let g = golden[c].to_f32();
            let k = gpu[r * hd + c].to_f32();
            let abs = (g - k).abs();
            let rel = if g.abs() > 1e-3 { abs / g.abs() } else { 0.0 };
            if abs > max_abs_err {
                max_abs_err = abs;
            }
            if rel > max_rel_err {
                max_rel_err = rel;
            }
        }
    }
    eprintln!(
        "{label}: max_abs_err={max_abs_err:.5}, max_rel_err={:.4}%",
        max_rel_err * 100.0
    );
    assert!(
        max_abs_err < 0.05,
        "{label}: max abs err {max_abs_err} too large"
    );
    assert!(
        max_rel_err < 0.05,
        "{label}: max rel err {max_rel_err} too large"
    );
}

/// Initial inputs the host fills before launching, so each tile body has
/// non-zero data to operate on regardless of which upstream tiles are still
/// placeholders.
struct TestInputs {
    h_data: Vec<bf16>,
    an_w_data: Vec<bf16>,
    mn_w_data: Vec<bf16>,
    attn_out_data: Vec<bf16>,
    qkv_data: Vec<bf16>,
}

/// Build the standard set of test buffers for the tiny model.
fn build_test_buffers() -> (TestBuffers, TestInputs) {
    let dims = scheduled_prefill_tiny_dims();
    let hd = dims.hidden_dim as usize;
    let seq = dims.seq_len as usize;
    let nl = dims.num_layers as usize;
    let id = dims.intermediate_dim as usize;
    let qkv_dim = ((dims.num_attn_heads + 2 * dims.num_kv_heads) * dims.head_dim) as usize;

    let h_data = random_bf16(seq * hd, 1, 0.5);
    let an_w_data = random_bf16(nl * hd, 2, 1.0);
    let mn_w_data = random_bf16(nl * hd, 3, 1.0);
    let attn_out_data = random_bf16(seq * hd, 4, 0.5);
    let qkv_data = random_bf16(seq * qkv_dim, 5, 0.5);

    let b = TestBuffers {
        hidden_states: gpu_upload_bf16(&h_data),
        rms_rope: gpu_alloc_zeros(seq * hd * 2),
        qkv: gpu_upload_bf16(&qkv_data),
        attn_out: gpu_upload_bf16(&attn_out_data),
        rms_gate: gpu_alloc_zeros(seq * hd * 2),
        silu_out: gpu_alloc_zeros(seq * id * 2),
        attn_norm_w: gpu_upload_bf16(&an_w_data),
        mlp_norm_w: gpu_upload_bf16(&mn_w_data),
    };
    (
        b,
        TestInputs {
            h_data,
            an_w_data,
            mn_w_data,
            attn_out_data,
            qkv_data,
        },
    )
}

#[test]
#[ignore = "needs GPU"]
fn tile_attn_norm_matches_cpu_golden() {
    init_cuda();
    let dims = scheduled_prefill_tiny_dims();
    let hd = dims.hidden_dim as usize;
    let seq = dims.seq_len as usize;
    let nl = dims.num_layers as usize;
    let eps: f32 = 1e-5;

    let (b, inp) = build_test_buffers();
    let _ = launch_with_buffers(&b, eps);

    // The wave schedule executes layers in order, so layer (NL-1)'s attn_norm
    // is the last writer for each row in rms_rope. Validate against that.
    let rms_rope_gpu = gpu_download_bf16(b.rms_rope, seq * hd);
    let last_layer_w = &inp.an_w_data[(nl - 1) * hd..nl * hd];
    assert_matches_golden(&rms_rope_gpu, seq, hd, "tile_attn_norm", |r| {
        cpu_rms_norm(&inp.h_data[r * hd..(r + 1) * hd], last_layer_w, eps)
    });
}

#[test]
#[ignore = "needs GPU"]
fn tile_mlp_norm_matches_cpu_golden() {
    init_cuda();
    let dims = scheduled_prefill_tiny_dims();
    let hd = dims.hidden_dim as usize;
    let seq = dims.seq_len as usize;
    let nl = dims.num_layers as usize;
    let eps: f32 = 1e-5;

    let (b, inp) = build_test_buffers();
    let _ = launch_with_buffers(&b, eps);

    // Last writer wins: layer (NL-1)'s mlp_norm. attn_out is non-zero
    // (pre-filled by build_test_buffers) so the validation is meaningful.
    let rms_gate_gpu = gpu_download_bf16(b.rms_gate, seq * hd);
    let last_layer_w = &inp.mn_w_data[(nl - 1) * hd..nl * hd];
    assert_matches_golden(&rms_gate_gpu, seq, hd, "tile_mlp_norm", |r| {
        cpu_rms_norm(&inp.attn_out_data[r * hd..(r + 1) * hd], last_layer_w, eps)
    });
}

/// CPU reference for one application of LLaMA-style split-half RoPE on Q/K
/// portions of a qkv row, in place. V heads pass through.
fn cpu_rope_one(
    qkv_row: &mut [bf16],
    position: usize,
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
) {
    let half = head_dim / 2;
    let theta_base: f32 = 10000.0;
    let total_rotated_heads = num_q_heads + num_kv_heads;
    for h in 0..total_rotated_heads {
        let head_off = h * head_dim;
        for c in 0..half {
            let exp = (2 * c) as f32 / head_dim as f32;
            let inv_freq = theta_base.powf(-exp);
            let ang = position as f32 * inv_freq;
            let cos_v = ang.cos();
            let sin_v = ang.sin();
            let x0 = qkv_row[head_off + c].to_f32();
            let x1 = qkv_row[head_off + c + half].to_f32();
            qkv_row[head_off + c] = bf16::from_f32(x0 * cos_v - x1 * sin_v);
            qkv_row[head_off + c + half] = bf16::from_f32(x0 * sin_v + x1 * cos_v);
        }
    }
}

#[test]
#[ignore = "needs GPU"]
fn tile_rope_matches_cpu_golden() {
    init_cuda();
    let dims = scheduled_prefill_tiny_dims();
    let qkv_dim = ((dims.num_attn_heads + 2 * dims.num_kv_heads) * dims.head_dim) as usize;
    let seq = dims.seq_len as usize;
    let nl = dims.num_layers as usize;
    let hdm = dims.head_dim as usize;
    let nah = dims.num_attn_heads as usize;
    let nkh = dims.num_kv_heads as usize;
    let eps: f32 = 1e-5;

    let (b, inp) = build_test_buffers();
    let _ = launch_with_buffers(&b, eps);

    // RoPE is in-place on the qkv buffer. The wave schedule applies RoPE
    // once per layer using the same parameters, so the buffer is rotated
    // NL times in total. CPU golden does the same.
    let qkv_gpu = gpu_download_bf16(b.qkv, seq * qkv_dim);
    let mut qkv_golden: Vec<bf16> = inp.qkv_data.clone();
    for r in 0..seq {
        for _ in 0..nl {
            let row = &mut qkv_golden[r * qkv_dim..(r + 1) * qkv_dim];
            cpu_rope_one(row, r, hdm, nah, nkh);
        }
    }

    let mut max_abs_err = 0.0_f32;
    let mut max_rel_err = 0.0_f32;
    for i in 0..seq * qkv_dim {
        let g = qkv_golden[i].to_f32();
        let k = qkv_gpu[i].to_f32();
        let abs = (g - k).abs();
        let rel = if g.abs() > 1e-3 { abs / g.abs() } else { 0.0 };
        if abs > max_abs_err {
            max_abs_err = abs;
        }
        if rel > max_rel_err {
            max_rel_err = rel;
        }
    }
    eprintln!(
        "tile_rope: max_abs_err={max_abs_err:.5}, max_rel_err={:.4}%",
        max_rel_err * 100.0
    );
    // RoPE accumulates rounding error each layer. NL=2 layers + bf16 → tolerance
    // of 0.05 abs / 5% rel covers it comfortably.
    assert!(max_abs_err < 0.05, "tile_rope: max abs err {max_abs_err}");
    assert!(max_rel_err < 0.05, "tile_rope: max rel err {max_rel_err}");
}
