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
    SCHEDULED_PREFILL_KV_PAGE_SIZE, SCHEDULED_PREFILL_TINY_CTAS, reified_dag::ReifiedDag,
    reified_dag::TileSizes, scheduled_prefill_tiny_dims,
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

fn gpu_upload_i32(host: &[i32]) -> u64 {
    unsafe {
        let bytes = host.len() * 4;
        let dptr = result::malloc_sync(bytes).expect("cuMemAlloc failed");
        result::memcpy_htod_sync(dptr, host).expect("cuMemcpyHtoD failed");
        dptr
    }
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
    q_post_rope: u64,
    attn_out: u64,
    rms_gate: u64,
    silu_out: u64,
    k_cache: u64,
    v_cache: u64,
    prefill_kv_indices: u64,
    prefill_kv_indptr: u64,
    prefill_qo_indptr: u64,
    attn_norm_w: u64,
    mlp_norm_w: u64,
    qkv_w: u64,
    o_w: u64,
    gate_w: u64,
    up_w: u64,
    down_w: u64,
}

fn launch_with_buffers(b: &TestBuffers, eps: f32) -> (Vec<u32>, u32) {
    let dims = scheduled_prefill_tiny_dims();
    let dag = ReifiedDag::reify_llama(dims, TileSizes::default_v1());
    let n = dag.nodes.len();
    let attn_scale = 1.0 / (dims.head_dim as f32).sqrt();

    let flags = gpu_alloc_zeros_u32(n);
    let tick = gpu_alloc_zeros_u32(1);
    let barrier = gpu_alloc_zeros_u32(1);

    unsafe {
        ffi::launch_scheduled_megakernel(
            b.hidden_states as *mut _,
            b.rms_rope as *mut _,
            b.qkv as *mut _,
            b.q_post_rope as *mut _,
            b.attn_out as *mut _,
            b.rms_gate as *mut _,
            b.silu_out as *mut _,
            b.k_cache as *mut _,
            b.v_cache as *mut _,
            b.prefill_kv_indices as *const i32,
            b.prefill_kv_indptr as *const i32,
            b.prefill_qo_indptr as *const i32,
            b.attn_norm_w as *mut _,
            b.mlp_norm_w as *mut _,
            b.qkv_w as *mut _,
            b.o_w as *mut _,
            b.gate_w as *mut _,
            b.up_w as *mut _,
            b.down_w as *mut _,
            eps,
            attn_scale,
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
    gate_w_data: Vec<bf16>,
    up_w_data: Vec<bf16>,
    down_w_data: Vec<bf16>,
}

/// Build the standard set of test buffers for the tiny model.
fn build_test_buffers() -> (TestBuffers, TestInputs) {
    let dims = scheduled_prefill_tiny_dims();
    let hd = dims.hidden_dim as usize;
    let seq = dims.seq_len as usize;
    let nl = dims.num_layers as usize;
    let id = dims.intermediate_dim as usize;
    let nah = dims.num_attn_heads as usize;
    let nkh = dims.num_kv_heads as usize;
    let hdm = dims.head_dim as usize;
    let qkv_dim = (nah + 2 * nkh) * hdm;
    // Paged KV cache geometry.
    let page_size = SCHEDULED_PREFILL_KV_PAGE_SIZE as usize;
    let pages_per_layer = seq.div_ceil(page_size);
    let total_pages = nl * pages_per_layer;
    // bf16, [total_pages, page_size, num_kv_heads, head_dim]
    let cache_bytes_per_layer = pages_per_layer * page_size * nkh * hdm * 2;
    let cache_bytes_total = nl * cache_bytes_per_layer;

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
    // gate_w / up_w are ID×HD per layer.
    let gate_w_data = random_bf16(nl * id * hd, 8, 0.05);
    let up_w_data = random_bf16(nl * id * hd, 9, 0.05);
    let down_w_data = random_bf16(nl * hd * id, 10, 0.02);

    // Block table: identity mapping for the single test sequence.
    let prefill_kv_indices: Vec<i32> = (0..pages_per_layer as i32).collect();
    // Per-sequence indptrs (one sequence): [0, pages_per_layer], [0, seq].
    let prefill_kv_indptr_data: Vec<i32> = vec![0, pages_per_layer as i32];
    let prefill_qo_indptr_data: Vec<i32> = vec![0, seq as i32];

    let _ = total_pages; // (sized via cache_bytes_total)

    let b = TestBuffers {
        hidden_states: gpu_upload_bf16(&h_data),
        rms_rope: gpu_alloc_zeros(seq * hd * 2),
        qkv: gpu_alloc_zeros(seq * qkv_dim * 2),
        q_post_rope: gpu_alloc_zeros(seq * nah * hdm * 2),
        attn_out: gpu_upload_bf16(&attn_out_data),
        rms_gate: gpu_alloc_zeros(seq * hd * 2),
        silu_out: gpu_alloc_zeros(seq * id * 2),
        k_cache: gpu_alloc_zeros(cache_bytes_total),
        v_cache: gpu_alloc_zeros(cache_bytes_total),
        prefill_kv_indices: gpu_upload_i32(&prefill_kv_indices),
        prefill_kv_indptr: gpu_upload_i32(&prefill_kv_indptr_data),
        prefill_qo_indptr: gpu_upload_i32(&prefill_qo_indptr_data),
        attn_norm_w: gpu_upload_bf16(&an_w_data),
        mlp_norm_w: gpu_upload_bf16(&mn_w_data),
        qkv_w: gpu_upload_bf16(&qkv_w_data),
        o_w: gpu_upload_bf16(&o_w_data),
        gate_w: gpu_upload_bf16(&gate_w_data),
        up_w: gpu_upload_bf16(&up_w_data),
        down_w: gpu_upload_bf16(&down_w_data),
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
            gate_w_data,
            up_w_data,
            down_w_data,
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
    id: usize,
    seq: usize,
    eps: f32,
) -> Vec<bf16> {
    let mut h = inp.h_data.clone();
    for l in 0..layer {
        // 1. o_proj residual: h += gemm(attn_out_data, o_w[l])
        let o_layer_w = &inp.o_w_data[l * hd * hd..(l + 1) * hd * hd];
        let o_contrib = cpu_gemm(&inp.attn_out_data, o_layer_w, seq, hd, hd);
        for i in 0..h.len() {
            h[i] = bf16::from_f32(h[i].to_f32() + o_contrib[i].to_f32());
        }
        // 2. down residual: h += gemm(silu_out_l, down_w[l])
        //    silu_out_l = silu(gemm(rms_gate_l, gate_w[l])) * gemm(rms_gate_l, up_w[l])
        //    rms_gate_l = norm(attn_out_data, mn_w[l])
        // With tile_attention still placeholder, attn_out_data is the
        // mlp_norm input — independent of the cascade.
        let mn_layer_w = &inp.mn_w_data[l * hd..(l + 1) * hd];
        let gate_layer_w = &inp.gate_w_data[l * id * hd..(l + 1) * id * hd];
        let up_layer_w = &inp.up_w_data[l * id * hd..(l + 1) * id * hd];
        let down_layer_w = &inp.down_w_data[l * hd * id..(l + 1) * hd * id];

        let mut silu_out_l = vec![bf16::from_f32(0.0); seq * id];
        for r in 0..seq {
            let normed = cpu_rms_norm(&inp.attn_out_data[r * hd..(r + 1) * hd], mn_layer_w, eps);
            let g_row = cpu_gemm(&normed, gate_layer_w, 1, id, hd);
            let u_row = cpu_gemm(&normed, up_layer_w, 1, id, hd);
            for n in 0..id {
                let gv = g_row[n].to_f32();
                let uv = u_row[n].to_f32();
                let silu_g = gv / (1.0 + (-gv).exp());
                silu_out_l[r * id + n] = bf16::from_f32(silu_g * uv);
            }
        }
        let down_contrib = cpu_gemm(&silu_out_l, down_layer_w, seq, hd, id);
        for i in 0..h.len() {
            h[i] = bf16::from_f32(h[i].to_f32() + down_contrib[i].to_f32());
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
    let id = dims.intermediate_dim as usize;
    let seq = dims.seq_len as usize;
    let nl = dims.num_layers as usize;
    let eps: f32 = 1e-5;

    let (b, inp) = build_test_buffers();
    let _ = launch_with_buffers(&b, eps);

    // Last writer for rms_rope is layer (NL-1)'s attn_norm. By that time,
    // hidden_states has been residual-updated by all earlier o_proj + down passes.
    let rms_rope_gpu = gpu_download_bf16(b.rms_rope, seq * hd);
    let h_at_last_layer = cpu_hidden_at_start_of_layer(&inp, nl - 1, hd, id, seq, eps);
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
fn full_residual_chain_matches_cpu_golden() {
    init_cuda();
    let dims = scheduled_prefill_tiny_dims();
    let hd = dims.hidden_dim as usize;
    let id = dims.intermediate_dim as usize;
    let seq = dims.seq_len as usize;
    let nl = dims.num_layers as usize;
    let eps: f32 = 1e-5;

    let (b, inp) = build_test_buffers();
    let _ = launch_with_buffers(&b, eps);

    // After NL layers, hidden_states has accumulated o_proj + down residuals
    // for every layer. cpu_hidden_at_start_of_layer(inp, nl, ...) is exactly
    // that end-state.
    let h_gpu = gpu_download_bf16(b.hidden_states, seq * hd);
    let h_cpu = cpu_hidden_at_start_of_layer(&inp, nl, hd, id, seq, eps);

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
        "full_residual_chain: max_abs_err={max_abs_err:.5}, max_rel_err={:.4}%",
        max_rel_err * 100.0
    );
    // The full residual chain accumulates 2 layers × (o_proj + down) of
    // bf16-rounded contributions. Accumulator magnitudes can reach ~50-100
    // where bf16 ULP is ~0.5. Relative error stays small; absolute is the
    // noisier metric. Both checks loosened to accommodate.
    assert!(max_abs_err < 2.5, "full chain abs err {max_abs_err}");
    assert!(max_rel_err < 0.05, "full chain rel err {max_rel_err}");
}

#[test]
#[ignore = "needs GPU"]
fn gate_up_chain_matches_cpu_golden() {
    init_cuda();
    let dims = scheduled_prefill_tiny_dims();
    let hd = dims.hidden_dim as usize;
    let id = dims.intermediate_dim as usize;
    let seq = dims.seq_len as usize;
    let nl = dims.num_layers as usize;
    let eps: f32 = 1e-5;

    let (b, inp) = build_test_buffers();
    let _ = launch_with_buffers(&b, eps);

    // gate_up reads rms_gate (output of mlp_norm). mlp_norm reads attn_out
    // which is still placeholder (pre-filled), so rms_gate is independent
    // of the o_proj residual chain. Last writer for silu_out is layer NL-1.
    let silu_out_gpu = gpu_download_bf16(b.silu_out, seq * id);
    let mn_w_last = &inp.mn_w_data[(nl - 1) * hd..nl * hd];
    let gate_w_last = &inp.gate_w_data[(nl - 1) * id * hd..nl * id * hd];
    let up_w_last = &inp.up_w_data[(nl - 1) * id * hd..nl * id * hd];

    let mut max_abs_err = 0.0_f32;
    let mut max_rel_err = 0.0_f32;
    for r in 0..seq {
        let normed = cpu_rms_norm(&inp.attn_out_data[r * hd..(r + 1) * hd], mn_w_last, eps);
        let g_row = cpu_gemm(&normed, gate_w_last, 1, id, hd);
        let u_row = cpu_gemm(&normed, up_w_last, 1, id, hd);
        for n in 0..id {
            let gv = g_row[n].to_f32();
            let uv = u_row[n].to_f32();
            let silu_g = gv / (1.0 + (-gv).exp());
            let golden = silu_g * uv;
            let k = silu_out_gpu[r * id + n].to_f32();
            let abs = (golden - k).abs();
            let rel = if golden.abs() > 1e-3 {
                abs / golden.abs()
            } else {
                0.0
            };
            if abs > max_abs_err {
                max_abs_err = abs;
            }
            if rel > max_rel_err {
                max_rel_err = rel;
            }
        }
    }
    eprintln!(
        "gate_up_chain: max_abs_err={max_abs_err:.5}, max_rel_err={:.4}%",
        max_rel_err * 100.0
    );
    assert!(max_abs_err < 0.15, "gate_up chain abs err {max_abs_err}");
    assert!(max_rel_err < 0.30, "gate_up chain rel err {max_rel_err}");
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

/// Returns the paged-cache offset for a (layer, row, kv_head, d) tuple,
/// matching the layout the GPU writes:
///
///   page_idx = prefill_kv_indices[row / page_size] + layer * pages_per_layer
///   slot     = row % page_size
///   offset   = ((page_idx * page_size + slot) * num_kv_heads + kv_head) * head_dim + d
///
/// For the test fixture, prefill_kv_indices is the identity mapping, so
/// physical_page = layer * pages_per_layer + (row / page_size).
fn paged_kv_offset(
    layer: usize,
    row: usize,
    kv_head: usize,
    d: usize,
    page_size: usize,
    pages_per_layer: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> usize {
    let logical_page = row / page_size;
    let slot = row % page_size;
    let physical_page = layer * pages_per_layer + logical_page;
    ((physical_page * page_size + slot) * num_kv_heads + kv_head) * head_dim + d
}

/// Phase 3c.8 step 2: validates that tile_rope correctly fans out into:
///   - q_post_rope (rotated Q for the last layer; previous layers overwritten)
///   - k_cache (rotated K for ALL layers, addressed by paged offsets)
///   - v_cache (passthrough V for ALL layers, addressed by paged offsets)
///
/// The CPU golden simulates the layer-by-layer chain (norm → gemm → rope
/// fan-out), walks the same paged-cache offsets, and compares.
#[test]
#[ignore = "needs GPU"]
fn rope_fanout_chain_matches_cpu_golden() {
    init_cuda();
    let dims = scheduled_prefill_tiny_dims();
    let hd = dims.hidden_dim as usize;
    let id = dims.intermediate_dim as usize;
    let qkv_dim = ((dims.num_attn_heads + 2 * dims.num_kv_heads) * dims.head_dim) as usize;
    let seq = dims.seq_len as usize;
    let nl = dims.num_layers as usize;
    let hdm = dims.head_dim as usize;
    let nah = dims.num_attn_heads as usize;
    let nkh = dims.num_kv_heads as usize;
    let q_dim = nah * hdm;
    let page_size = SCHEDULED_PREFILL_KV_PAGE_SIZE as usize;
    let pages_per_layer = seq.div_ceil(page_size);
    let cache_total = nl * pages_per_layer * page_size * nkh * hdm;
    let eps: f32 = 1e-5;

    let (b, inp) = build_test_buffers();
    let _ = launch_with_buffers(&b, eps);

    let q_post_gpu = gpu_download_bf16(b.q_post_rope, seq * q_dim);
    let k_cache_gpu = gpu_download_bf16(b.k_cache, cache_total);
    let v_cache_gpu = gpu_download_bf16(b.v_cache, cache_total);

    // CPU walks each layer, computes the qkv row, applies rope-fanout, and
    // stores the results in CPU mirrors of q_post_rope / k_cache / v_cache.
    let mut q_post_cpu = vec![bf16::from_f32(0.0); seq * q_dim];
    let mut k_cache_cpu = vec![bf16::from_f32(0.0); cache_total];
    let mut v_cache_cpu = vec![bf16::from_f32(0.0); cache_total];

    for l in 0..nl {
        let h = cpu_hidden_at_start_of_layer(&inp, l, hd, id, seq, eps);
        let an_w_l = &inp.an_w_data[l * hd..(l + 1) * hd];
        let qkv_w_l = &inp.qkv_w_data[l * qkv_dim * hd..(l + 1) * qkv_dim * hd];

        for r in 0..seq {
            // 1. norm
            let normed = cpu_rms_norm(&h[r * hd..(r + 1) * hd], an_w_l, eps);
            // 2. raw qkv gemm (one row of length qkv_dim)
            let mut row = cpu_gemm(&normed, qkv_w_l, 1, qkv_dim, hd);
            // 3. rope on Q+K head pairs (V untouched)
            cpu_rope_one(&mut row, r, hdm, nah, nkh);

            // Q region → q_post_rope (last writer wins, so layer NL-1's value
            // is what survives in the GPU buffer).
            for c in 0..q_dim {
                q_post_cpu[r * q_dim + c] = row[c];
            }

            // K region → paged k_cache slot for (l, r, kv_head, d)
            let k_off_in_row = nah * hdm;
            for kvh in 0..nkh {
                for d in 0..hdm {
                    let off = paged_kv_offset(l, r, kvh, d, page_size, pages_per_layer, nkh, hdm);
                    k_cache_cpu[off] = row[k_off_in_row + kvh * hdm + d];
                }
            }
            // V region → paged v_cache (passthrough, no rotation; read from
            // the gemm output `row` *before* the rope rewrote it — but rope
            // only touches Q and K, so V indices are still the gemm output).
            let v_off_in_row = (nah + nkh) * hdm;
            for kvh in 0..nkh {
                for d in 0..hdm {
                    let off = paged_kv_offset(l, r, kvh, d, page_size, pages_per_layer, nkh, hdm);
                    v_cache_cpu[off] = row[v_off_in_row + kvh * hdm + d];
                }
            }
        }
    }

    // ── Validate Q (last layer only — earlier overwritten in q_post_rope) ──
    let mut q_max_abs = 0.0_f32;
    for i in 0..seq * q_dim {
        let g = q_post_cpu[i].to_f32();
        let k = q_post_gpu[i].to_f32();
        let abs = (g - k).abs();
        if abs > q_max_abs {
            q_max_abs = abs;
        }
    }
    eprintln!("q_post_rope: max_abs_err={q_max_abs:.5}");
    assert!(q_max_abs < 0.10, "q_post_rope abs err {q_max_abs}");

    // ── Validate paged K cache (all layers, all rows) ──
    let mut k_max_abs = 0.0_f32;
    for i in 0..cache_total {
        let g = k_cache_cpu[i].to_f32();
        let k = k_cache_gpu[i].to_f32();
        let abs = (g - k).abs();
        if abs > k_max_abs {
            k_max_abs = abs;
        }
    }
    eprintln!("k_cache: max_abs_err={k_max_abs:.5}");
    assert!(k_max_abs < 0.10, "k_cache abs err {k_max_abs}");

    // ── Validate paged V cache (all layers, all rows) ──
    let mut v_max_abs = 0.0_f32;
    for i in 0..cache_total {
        let g = v_cache_cpu[i].to_f32();
        let k = v_cache_gpu[i].to_f32();
        let abs = (g - k).abs();
        if abs > v_max_abs {
            v_max_abs = abs;
        }
    }
    eprintln!("v_cache: max_abs_err={v_max_abs:.5}");
    assert!(v_max_abs < 0.10, "v_cache abs err {v_max_abs}");
}
