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
    qkv_w: u64,
    o_w: u64,
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
            b.qkv_w as *mut _,
            b.o_w as *mut _,
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
    qkv_w_data: Vec<bf16>,
    o_w_data: Vec<bf16>,
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
    // Small weight scale (0.05) to keep accumulator products in range —
    // bf16 GEMM with hd=256 over (-0.5,0.5) inputs and full-magnitude
    // weights would push individual outputs into the tens, blowing
    // through bf16's relative precision.
    let qkv_w_data = random_bf16(nl * qkv_dim * hd, 6, 0.05);
    // o_w is HD×HD per layer; same small-scale weight to keep accumulators sane.
    let o_w_data = random_bf16(nl * hd * hd, 7, 0.05);

    let b = TestBuffers {
        hidden_states: gpu_upload_bf16(&h_data),
        rms_rope: gpu_alloc_zeros(seq * hd * 2),
        qkv: gpu_alloc_zeros(seq * qkv_dim * 2),
        attn_out: gpu_upload_bf16(&attn_out_data),
        rms_gate: gpu_alloc_zeros(seq * hd * 2),
        silu_out: gpu_alloc_zeros(seq * id * 2),
        attn_norm_w: gpu_upload_bf16(&an_w_data),
        mlp_norm_w: gpu_upload_bf16(&mn_w_data),
        qkv_w: gpu_upload_bf16(&qkv_w_data),
        o_w: gpu_upload_bf16(&o_w_data),
    };
    (
        b,
        TestInputs {
            h_data,
            an_w_data,
            mn_w_data,
            attn_out_data,
            qkv_w_data,
            o_w_data,
        },
    )
}

/// Compute the running hidden_states value as the kernel would see it at the
/// START of layer L's attn_norm — i.e. with all earlier layers' o_proj +
/// down residual contributions accumulated. With down still placeholder,
/// only o_proj contributes.
fn cpu_hidden_at_start_of_layer(
    inp: &TestInputs,
    layer: usize,
    hd: usize,
    seq: usize,
) -> Vec<bf16> {
    let mut h = inp.h_data.clone();
    for l in 0..layer {
        // o_proj residual: h += gemm(attn_out_data, o_w[l])
        let o_layer_w = &inp.o_w_data[l * hd * hd..(l + 1) * hd * hd];
        let contrib = cpu_gemm(&inp.attn_out_data, o_layer_w, seq, hd, hd);
        for i in 0..h.len() {
            h[i] = bf16::from_f32(h[i].to_f32() + contrib[i].to_f32());
        }
    }
    h
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

    // Last writer for rms_rope is layer (NL-1)'s attn_norm. By that time,
    // hidden_states has been residual-updated by all earlier o_proj passes.
    let rms_rope_gpu = gpu_download_bf16(b.rms_rope, seq * hd);
    let h_at_last_layer = cpu_hidden_at_start_of_layer(&inp, nl - 1, hd, seq);
    let last_layer_w = &inp.an_w_data[(nl - 1) * hd..nl * hd];
    assert_matches_golden(&rms_rope_gpu, seq, hd, "tile_attn_norm", |r| {
        cpu_rms_norm(&h_at_last_layer[r * hd..(r + 1) * hd], last_layer_w, eps)
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

    // mlp_norm reads attn_out which is still placeholder (pre-filled). It's
    // independent of the o_proj residual chain — attn_out doesn't change.
    let rms_gate_gpu = gpu_download_bf16(b.rms_gate, seq * hd);
    let last_layer_w = &inp.mn_w_data[(nl - 1) * hd..nl * hd];
    assert_matches_golden(&rms_gate_gpu, seq, hd, "tile_mlp_norm", |r| {
        cpu_rms_norm(&inp.attn_out_data[r * hd..(r + 1) * hd], last_layer_w, eps)
    });
}

#[test]
#[ignore = "needs GPU"]
fn tile_o_proj_residual_chain_matches_cpu_golden() {
    init_cuda();
    let dims = scheduled_prefill_tiny_dims();
    let hd = dims.hidden_dim as usize;
    let seq = dims.seq_len as usize;
    let nl = dims.num_layers as usize;
    let eps: f32 = 1e-5;

    let (b, inp) = build_test_buffers();
    let _ = launch_with_buffers(&b, eps);

    // o_proj is the only thing that updates hidden_states (down still
    // placeholder). After NL layers:
    //   hidden_states = h_data + Σ_{L=0..NL-1} gemm(attn_out_data, o_w[L])
    // CPU computes the full residual chain.
    let h_gpu = gpu_download_bf16(b.hidden_states, seq * hd);
    let h_cpu = cpu_hidden_at_start_of_layer(&inp, nl, hd, seq);

    let mut max_abs_err = 0.0_f32;
    let mut max_rel_err = 0.0_f32;
    for i in 0..seq * hd {
        let g = h_cpu[i].to_f32();
        let k = h_gpu[i].to_f32();
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
        "tile_o_proj_residual_chain: max_abs_err={max_abs_err:.5}, max_rel_err={:.4}%",
        max_rel_err * 100.0
    );
    assert!(max_abs_err < 0.15, "o_proj chain abs err {max_abs_err}");
    assert!(max_rel_err < 0.05, "o_proj chain rel err {max_rel_err}");
}

/// CPU reference for a single GEMM tile: out[m,n] = sum_k a[m,k] * b[n,k].
/// `a` is [m_dim, k_dim], `b` is [n_dim, k_dim] (B is "stored transposed"),
/// `out` is [m_dim, n_dim]. Computes in fp32, rounds to bf16 on store.
fn cpu_gemm(a: &[bf16], b: &[bf16], m_dim: usize, n_dim: usize, k_dim: usize) -> Vec<bf16> {
    let mut out = vec![bf16::from_f32(0.0); m_dim * n_dim];
    for m in 0..m_dim {
        for n in 0..n_dim {
            let mut acc = 0.0_f32;
            for k in 0..k_dim {
                acc += a[m * k_dim + k].to_f32() * b[n * k_dim + k].to_f32();
            }
            out[m * n_dim + n] = bf16::from_f32(acc);
        }
    }
    out
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

/// Phase 3c step 4: validates the full norm → gemm → rope chain through the
/// last layer. Once tile_qkv is real it overwrites the qkv buffer on every
/// layer's qkv pass before rope re-rotates it, so the only meaningful
/// validation point is the qkv buffer end-state, which equals
///
///     rope( gemm( norm( hidden_states, an_w[NL-1] ),  qkv_w[NL-1] ),  pos )
///
/// applied once (not NL times — each layer's qkv overwrites the previous).
/// Implicitly validates tile_qkv AND tile_rope.
#[test]
#[ignore = "needs GPU"]
fn qkv_chain_matches_cpu_golden() {
    init_cuda();
    let dims = scheduled_prefill_tiny_dims();
    let hd = dims.hidden_dim as usize;
    let qkv_dim = ((dims.num_attn_heads + 2 * dims.num_kv_heads) * dims.head_dim) as usize;
    let seq = dims.seq_len as usize;
    let nl = dims.num_layers as usize;
    let hdm = dims.head_dim as usize;
    let nah = dims.num_attn_heads as usize;
    let nkh = dims.num_kv_heads as usize;
    let eps: f32 = 1e-5;

    let (b, inp) = build_test_buffers();
    let _ = launch_with_buffers(&b, eps);

    let qkv_gpu = gpu_download_bf16(b.qkv, seq * qkv_dim);

    // CPU chain through layer NL-1 only (last writer wins). The chain input
    // is hidden_states *as the kernel sees it* at the start of layer NL-1,
    // which has all earlier o_proj residual contributions baked in.
    let an_w_last = &inp.an_w_data[(nl - 1) * hd..nl * hd];
    let qkv_w_last = &inp.qkv_w_data[(nl - 1) * qkv_dim * hd..nl * qkv_dim * hd];
    let h_at_last_layer = cpu_hidden_at_start_of_layer(&inp, nl - 1, hd, seq);

    let mut max_abs_err = 0.0_f32;
    let mut max_rel_err = 0.0_f32;
    for r in 0..seq {
        // 1. norm
        let normed = cpu_rms_norm(&h_at_last_layer[r * hd..(r + 1) * hd], an_w_last, eps);
        // 2. gemm — produces a single row of length qkv_dim
        let mut row = cpu_gemm(&normed, qkv_w_last, 1, qkv_dim, hd);
        // 3. rope (in-place on Q/K heads)
        cpu_rope_one(&mut row, r, hdm, nah, nkh);

        for n in 0..qkv_dim {
            let g = row[n].to_f32();
            let k = qkv_gpu[r * qkv_dim + n].to_f32();
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
        "qkv_chain: max_abs_err={max_abs_err:.5}, max_rel_err={:.4}%",
        max_rel_err * 100.0
    );
    // The chain (residual + norm + gemm + rope) accumulates bf16 rounding,
    // and per-element rel err is unreliable on near-zero outputs. Abs err
    // is the meaningful gate; rel tolerance is loose by design.
    assert!(max_abs_err < 0.10, "qkv_chain: max abs err {max_abs_err}");
    assert!(max_rel_err < 0.30, "qkv_chain: max rel err {max_rel_err}");
}
