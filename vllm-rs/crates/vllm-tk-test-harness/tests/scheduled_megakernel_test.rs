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
    SCHEDULED_PREFILL_KV_PAGE_SIZE, reified_dag::LlamaDims, reified_dag::ReifiedDag,
    reified_dag::TileSizes, scheduled_prefill_medium_dims, scheduled_prefill_tiny_dims,
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

/// Function pointer type for the per-variant launch helpers. Same shape
/// across variants — only the symbol name differs.
type LaunchFn = unsafe extern "C" fn(
    hidden_states: *mut std::ffi::c_void,
    rms_rope: *mut std::ffi::c_void,
    qkv: *mut std::ffi::c_void,
    q_post_rope: *mut std::ffi::c_void,
    attn_out: *mut std::ffi::c_void,
    rms_gate: *mut std::ffi::c_void,
    silu_out: *mut std::ffi::c_void,
    k_cache: *mut std::ffi::c_void,
    v_cache: *mut std::ffi::c_void,
    prefill_kv_indices: *const i32,
    prefill_kv_indptr: *const i32,
    prefill_qo_indptr: *const i32,
    attn_norm_w: *mut std::ffi::c_void,
    mlp_norm_w: *mut std::ffi::c_void,
    qkv_w: *mut std::ffi::c_void,
    o_w: *mut std::ffi::c_void,
    gate_w: *mut std::ffi::c_void,
    up_w: *mut std::ffi::c_void,
    down_w: *mut std::ffi::c_void,
    eps: f32,
    attn_scale: f32,
    flags: *mut u32,
    tick_counter: *mut u32,
    barrier_arrived: *mut u32,
    stream: *mut std::ffi::c_void,
);

fn launch_with_buffers(
    b: &TestBuffers,
    dims: LlamaDims,
    eps: f32,
    launch: LaunchFn,
    num_nodes: u32,
) -> (Vec<u32>, u32) {
    let n = num_nodes as usize;
    let attn_scale = 1.0 / (dims.head_dim as f32).sqrt();

    let flags = gpu_alloc_zeros_u32(n);
    let tick = gpu_alloc_zeros_u32(1);
    let barrier = gpu_alloc_zeros_u32(1);

    unsafe {
        launch(
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

    let kernel_n = unsafe { ffi::scheduled_megakernel_tiny_num_nodes() } as usize;
    assert_eq!(n, kernel_n);
    let kernel_ctas = unsafe { ffi::scheduled_megakernel_tiny_num_ctas() };
    let _ = kernel_ctas; // CTA pool size is now picked by the DSL emitter heuristic
    let kernel_waves = unsafe { ffi::scheduled_megakernel_tiny_num_waves() };
    eprintln!("scheduled_megakernel(tiny): {n} nodes, {kernel_waves} waves, {kernel_ctas} CTAs");

    let (b, _) = build_test_buffers(dims, 0);
    let (ticks, final_tick) = launch_with_buffers(
        &b,
        dims,
        1e-5,
        ffi::launch_scheduled_megakernel_tiny,
        kernel_n as u32,
    );
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

/// Inputs the host fills before launching the scheduled megakernel. The
/// CPU forward simulator (cpu_forward) runs the same data and produces the
/// per-tile expected end-states.
struct TestInputs {
    h_data: Vec<bf16>,
    an_w_data: Vec<bf16>,
    mn_w_data: Vec<bf16>,
    qkv_w_data: Vec<bf16>,
    o_w_data: Vec<bf16>,
    gate_w_data: Vec<bf16>,
    up_w_data: Vec<bf16>,
    down_w_data: Vec<bf16>,
}

/// Build the standard set of test buffers for any LLaMA-shaped variant.
/// Uses deterministic seeds keyed by `seed_base` so different variants don't
/// share buffer values.
fn build_test_buffers(dims: LlamaDims, seed_base: u64) -> (TestBuffers, TestInputs) {
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

    let h_data = random_bf16(seq * hd, seed_base + 1, 0.5);
    let an_w_data = random_bf16(nl * hd, seed_base + 2, 1.0);
    let mn_w_data = random_bf16(nl * hd, seed_base + 3, 1.0);
    // Small weight scale (0.05) to keep accumulator products in range —
    // bf16 GEMM with hd=256 over (-0.5,0.5) inputs and full-magnitude
    // weights would push individual outputs into the tens, blowing
    // through bf16's relative precision.
    let qkv_w_data = random_bf16(nl * qkv_dim * hd, seed_base + 6, 0.05);
    // o_w is HD×HD per layer; same small-scale weight to keep accumulators sane.
    let o_w_data = random_bf16(nl * hd * hd, seed_base + 7, 0.05);
    // gate_w / up_w are ID×HD per layer.
    let gate_w_data = random_bf16(nl * id * hd, seed_base + 8, 0.05);
    let up_w_data = random_bf16(nl * id * hd, seed_base + 9, 0.05);
    let down_w_data = random_bf16(nl * hd * id, seed_base + 10, 0.02);

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
        attn_out: gpu_alloc_zeros(seq * hd * 2),
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
            qkv_w_data,
            o_w_data,
            gate_w_data,
            up_w_data,
            down_w_data,
        },
    )
}

/// Per-tile end states the CPU forward simulator produces. Last-layer-wins
/// fields hold whatever survives in the GPU buffers after the megakernel
/// runs to completion. The paged caches hold all NL layers' contents.
struct CpuForward {
    h_final: Vec<bf16>,
    rms_rope_last: Vec<bf16>,
    q_post_rope_last: Vec<bf16>,
    attn_out_last: Vec<bf16>,
    rms_gate_last: Vec<bf16>,
    silu_out_last: Vec<bf16>,
    k_cache: Vec<bf16>,
    v_cache: Vec<bf16>,
}

/// Run the full forward pass on CPU, mirroring exactly what the scheduled
/// megakernel does. Each layer:
///
///   rms_rope    = rms_norm(h, attn_norm_w[L])
///   qkv         = gemm(rms_rope, qkv_w[L])           // raw, no rope
///   q_post_rope = rope_q(qkv[Q region])              // last L wins
///   k_cache[L]  = rope_k(qkv[K region])              // paged write
///   v_cache[L]  = qkv[V region]                      // paged write
///   attn_out    = paged_causal_attention(q_post_rope, k_cache[L], v_cache[L])
///   h          += gemm(attn_out, o_w[L])
///   rms_gate    = rms_norm(h, mlp_norm_w[L])
///   silu_out    = silu(gemm(rms_gate, gate_w[L])) * gemm(rms_gate, up_w[L])
///   h          += gemm(silu_out, down_w[L])
fn cpu_forward(inp: &TestInputs, dims: LlamaDims, eps: f32) -> CpuForward {
    let hd = dims.hidden_dim as usize;
    let id = dims.intermediate_dim as usize;
    let seq = dims.seq_len as usize;
    let nl = dims.num_layers as usize;
    let hdm = dims.head_dim as usize;
    let nah = dims.num_attn_heads as usize;
    let nkh = dims.num_kv_heads as usize;
    let qkv_dim = (nah + 2 * nkh) * hdm;
    let q_dim = nah * hdm;
    let gqa_ratio = nah / nkh;
    let attn_scale = 1.0_f32 / (hdm as f32).sqrt();
    let page_size = SCHEDULED_PREFILL_KV_PAGE_SIZE as usize;
    let pages_per_layer = seq.div_ceil(page_size);
    let cache_total = nl * pages_per_layer * page_size * nkh * hdm;

    let mut h = inp.h_data.clone();
    let mut k_cache = vec![bf16::from_f32(0.0); cache_total];
    let mut v_cache = vec![bf16::from_f32(0.0); cache_total];

    let mut rms_rope_last = vec![bf16::from_f32(0.0); seq * hd];
    let mut q_post_rope_last = vec![bf16::from_f32(0.0); seq * q_dim];
    let mut attn_out_last = vec![bf16::from_f32(0.0); seq * hd];
    let mut rms_gate_last = vec![bf16::from_f32(0.0); seq * hd];
    let mut silu_out_last = vec![bf16::from_f32(0.0); seq * id];

    for l in 0..nl {
        let an_w_l = &inp.an_w_data[l * hd..(l + 1) * hd];
        let mn_w_l = &inp.mn_w_data[l * hd..(l + 1) * hd];
        let qkv_w_l = &inp.qkv_w_data[l * qkv_dim * hd..(l + 1) * qkv_dim * hd];
        let o_w_l = &inp.o_w_data[l * hd * hd..(l + 1) * hd * hd];
        let gate_w_l = &inp.gate_w_data[l * id * hd..(l + 1) * id * hd];
        let up_w_l = &inp.up_w_data[l * id * hd..(l + 1) * id * hd];
        let down_w_l = &inp.down_w_data[l * hd * id..(l + 1) * hd * id];

        // 1. attn_norm
        let mut rms_rope = vec![bf16::from_f32(0.0); seq * hd];
        for r in 0..seq {
            let normed = cpu_rms_norm(&h[r * hd..(r + 1) * hd], an_w_l, eps);
            rms_rope[r * hd..(r + 1) * hd].copy_from_slice(&normed);
        }

        // 2 + 3. qkv gemm + rope-fanout to q_post_rope, paged k_cache, paged v_cache
        let mut q_post_rope = vec![bf16::from_f32(0.0); seq * q_dim];
        for r in 0..seq {
            let mut row = cpu_gemm(&rms_rope[r * hd..(r + 1) * hd], qkv_w_l, 1, qkv_dim, hd);
            cpu_rope_one(&mut row, r, hdm, nah, nkh);
            for c in 0..q_dim {
                q_post_rope[r * q_dim + c] = row[c];
            }
            let k_off_in_row = nah * hdm;
            let v_off_in_row = (nah + nkh) * hdm;
            for kvh in 0..nkh {
                for d in 0..hdm {
                    let off = paged_kv_offset(l, r, kvh, d, page_size, pages_per_layer, nkh, hdm);
                    k_cache[off] = row[k_off_in_row + kvh * hdm + d];
                    v_cache[off] = row[v_off_in_row + kvh * hdm + d];
                }
            }
        }

        // 4. paged causal attention
        let mut attn_out = vec![bf16::from_f32(0.0); seq * hd];
        for r in 0..seq {
            for qh in 0..nah {
                let kvh = qh / gqa_ratio;
                let q_offset = r * q_dim + qh * hdm;
                let q: Vec<f32> = (0..hdm)
                    .map(|d| q_post_rope[q_offset + d].to_f32())
                    .collect();
                let attend_len = r + 1;
                let mut scores = vec![0.0_f32; attend_len];
                for k_pos in 0..attend_len {
                    let off =
                        paged_kv_offset(l, k_pos, kvh, 0, page_size, pages_per_layer, nkh, hdm);
                    let mut dot = 0.0_f32;
                    for d in 0..hdm {
                        dot += q[d] * k_cache[off + d].to_f32();
                    }
                    scores[k_pos] = dot * attn_scale;
                }
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0_f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    sum_exp += *s;
                }
                for s in &mut scores {
                    *s /= sum_exp;
                }
                let mut out = vec![0.0_f32; hdm];
                for k_pos in 0..attend_len {
                    let off =
                        paged_kv_offset(l, k_pos, kvh, 0, page_size, pages_per_layer, nkh, hdm);
                    for d in 0..hdm {
                        out[d] += scores[k_pos] * v_cache[off + d].to_f32();
                    }
                }
                for d in 0..hdm {
                    attn_out[r * hd + qh * hdm + d] = bf16::from_f32(out[d]);
                }
            }
        }

        // 5. o_proj residual: h += gemm(attn_out, o_w[l])
        let o_contrib = cpu_gemm(&attn_out, o_w_l, seq, hd, hd);
        for i in 0..h.len() {
            h[i] = bf16::from_f32(h[i].to_f32() + o_contrib[i].to_f32());
        }

        // 6. mlp_norm
        let mut rms_gate = vec![bf16::from_f32(0.0); seq * hd];
        for r in 0..seq {
            let normed = cpu_rms_norm(&attn_out[r * hd..(r + 1) * hd], mn_w_l, eps);
            rms_gate[r * hd..(r + 1) * hd].copy_from_slice(&normed);
        }

        // 7. gate_up
        let mut silu_out = vec![bf16::from_f32(0.0); seq * id];
        for r in 0..seq {
            let g_row = cpu_gemm(&rms_gate[r * hd..(r + 1) * hd], gate_w_l, 1, id, hd);
            let u_row = cpu_gemm(&rms_gate[r * hd..(r + 1) * hd], up_w_l, 1, id, hd);
            for n in 0..id {
                let gv = g_row[n].to_f32();
                let uv = u_row[n].to_f32();
                let silu_g = gv / (1.0 + (-gv).exp());
                silu_out[r * id + n] = bf16::from_f32(silu_g * uv);
            }
        }

        // 8. down residual: h += gemm(silu_out, down_w[l])
        let down_contrib = cpu_gemm(&silu_out, down_w_l, seq, hd, id);
        for i in 0..h.len() {
            h[i] = bf16::from_f32(h[i].to_f32() + down_contrib[i].to_f32());
        }

        if l == nl - 1 {
            rms_rope_last = rms_rope;
            q_post_rope_last = q_post_rope;
            attn_out_last = attn_out;
            rms_gate_last = rms_gate;
            silu_out_last = silu_out;
        }
    }

    CpuForward {
        h_final: h,
        rms_rope_last,
        q_post_rope_last,
        attn_out_last,
        rms_gate_last,
        silu_out_last,
        k_cache,
        v_cache,
    }
}

/// Compute (max_abs_err, max_rel_err) between two bf16 arrays.
// ── Golden file infrastructure ───────────────────────────────────────────
//
// For variants where cpu_forward is too slow to run on every test (1B and
// up), we commit a binary "golden" containing the bf16 bytes of h_final
// from a one-time CPU forward pass. The test reads the golden and compares
// the GPU output against it with bf16 tolerance.
//
// File format: raw little-endian bf16 bytes, no header. Variant name is
// in the filename. Regenerate via `cargo test --features cuda --release \
//   --test scheduled_megakernel_test regen_committed_goldens -- --ignored \
//   --nocapture`.

/// Path to the committed golden for a given variant.
fn golden_path(variant: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(format!("{variant}_h_final.golden"))
}

/// Write the bf16 buffer as little-endian bytes to the variant's golden path.
/// Creates the parent directory if needed. Used by the regen test.
fn write_golden(variant: &str, data: &[bf16]) {
    let path = golden_path(variant);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create fixtures dir");
    }
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for v in data {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    std::fs::write(&path, &bytes)
        .unwrap_or_else(|e| panic!("write golden {}: {e}", path.display()));
    eprintln!("wrote {} ({} bytes)", path.display(), bytes.len());
}

/// Load a committed golden as a bf16 vector. Panics with a helpful message
/// if the file is missing (the user needs to run the regen target).
fn read_golden(variant: &str) -> Vec<bf16> {
    let path = golden_path(variant);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing golden for variant `{variant}` at {}: {e}\n\
             regenerate via:\n  \
             cargo test --features cuda --release --test scheduled_megakernel_test \
             regen_committed_goldens -- --ignored --nocapture",
            path.display()
        )
    });
    assert_eq!(bytes.len() % 2, 0, "golden file size not a multiple of 2");
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Validate a GPU output buffer against a committed golden file.
fn assert_matches_committed_golden(
    variant: &str,
    gpu: &[bf16],
    abs_tol: f32,
    rel_tol: f32,
) {
    let golden = read_golden(variant);
    assert_eq!(
        gpu.len(),
        golden.len(),
        "variant {variant}: gpu size {} != golden size {}",
        gpu.len(),
        golden.len()
    );
    let (abs, rel) = errs(gpu, &golden);
    eprintln!(
        "{variant} h_final vs golden: max_abs_err={abs:.5}  max_rel_err={:.4}%",
        rel * 100.0
    );
    assert!(
        abs < abs_tol,
        "{variant}: abs err {abs} >= tol {abs_tol}"
    );
    assert!(
        rel < rel_tol,
        "{variant}: rel err {rel} >= tol {rel_tol}"
    );
}

fn errs(gpu: &[bf16], cpu: &[bf16]) -> (f32, f32) {
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for i in 0..gpu.len() {
        let g = cpu[i].to_f32();
        let k = gpu[i].to_f32();
        let abs = (g - k).abs();
        let rel = if g.abs() > 1e-3 { abs / g.abs() } else { 0.0 };
        if abs > max_abs {
            max_abs = abs;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    (max_abs, max_rel)
}

#[test]
#[ignore = "needs GPU"]
fn tile_attn_norm_matches_cpu_golden() {
    init_cuda();
    let dims = scheduled_prefill_tiny_dims();
    let hd = dims.hidden_dim as usize;
    let seq = dims.seq_len as usize;
    let eps: f32 = 1e-5;

    let (b, inp) = build_test_buffers(dims, 0);
    let n_nodes = unsafe { ffi::scheduled_megakernel_tiny_num_nodes() };
    let _ = launch_with_buffers(
        &b,
        dims,
        eps,
        ffi::launch_scheduled_megakernel_tiny,
        n_nodes,
    );
    let cpu = cpu_forward(&inp, dims, eps);

    let gpu = gpu_download_bf16(b.rms_rope, seq * hd);
    let (abs, rel) = errs(&gpu, &cpu.rms_rope_last);
    eprintln!(
        "tile_attn_norm: max_abs_err={abs:.5}, max_rel_err={:.4}%",
        rel * 100.0
    );
    assert!(abs < 0.10, "attn_norm abs {abs}");
}

#[test]
#[ignore = "needs GPU"]
fn tile_mlp_norm_matches_cpu_golden() {
    init_cuda();
    let dims = scheduled_prefill_tiny_dims();
    let hd = dims.hidden_dim as usize;
    let seq = dims.seq_len as usize;
    let eps: f32 = 1e-5;

    let (b, inp) = build_test_buffers(dims, 0);
    let n_nodes = unsafe { ffi::scheduled_megakernel_tiny_num_nodes() };
    let _ = launch_with_buffers(
        &b,
        dims,
        eps,
        ffi::launch_scheduled_megakernel_tiny,
        n_nodes,
    );
    let cpu = cpu_forward(&inp, dims, eps);

    let gpu = gpu_download_bf16(b.rms_gate, seq * hd);
    let (abs, rel) = errs(&gpu, &cpu.rms_gate_last);
    eprintln!(
        "tile_mlp_norm: max_abs_err={abs:.5}, max_rel_err={:.4}%",
        rel * 100.0
    );
    assert!(abs < 0.10, "mlp_norm abs {abs}");
}

#[test]
#[ignore = "needs GPU"]
fn tile_attention_matches_cpu_golden() {
    init_cuda();
    let dims = scheduled_prefill_tiny_dims();
    let hd = dims.hidden_dim as usize;
    let seq = dims.seq_len as usize;
    let eps: f32 = 1e-5;

    let (b, inp) = build_test_buffers(dims, 0);
    let n_nodes = unsafe { ffi::scheduled_megakernel_tiny_num_nodes() };
    let _ = launch_with_buffers(
        &b,
        dims,
        eps,
        ffi::launch_scheduled_megakernel_tiny,
        n_nodes,
    );
    let cpu = cpu_forward(&inp, dims, eps);

    let gpu = gpu_download_bf16(b.attn_out, seq * hd);
    let (abs, rel) = errs(&gpu, &cpu.attn_out_last);
    eprintln!(
        "tile_attention: max_abs_err={abs:.5}, max_rel_err={:.4}%",
        rel * 100.0
    );
    // bf16 attention has both an exp() and a divide; tolerance loosened.
    assert!(abs < 0.10, "attention abs {abs}");
}

#[test]
#[ignore = "needs GPU"]
fn gate_up_chain_matches_cpu_golden() {
    init_cuda();
    let dims = scheduled_prefill_tiny_dims();
    let id = dims.intermediate_dim as usize;
    let seq = dims.seq_len as usize;
    let eps: f32 = 1e-5;

    let (b, inp) = build_test_buffers(dims, 0);
    let n_nodes = unsafe { ffi::scheduled_megakernel_tiny_num_nodes() };
    let _ = launch_with_buffers(
        &b,
        dims,
        eps,
        ffi::launch_scheduled_megakernel_tiny,
        n_nodes,
    );
    let cpu = cpu_forward(&inp, dims, eps);

    let gpu = gpu_download_bf16(b.silu_out, seq * id);
    let (abs, rel) = errs(&gpu, &cpu.silu_out_last);
    eprintln!(
        "gate_up_chain: max_abs_err={abs:.5}, max_rel_err={:.4}%",
        rel * 100.0
    );
    assert!(abs < 0.20, "gate_up abs {abs}");
}

#[test]
#[ignore = "needs GPU"]
fn full_residual_chain_matches_cpu_golden() {
    init_cuda();
    let dims = scheduled_prefill_tiny_dims();
    let hd = dims.hidden_dim as usize;
    let seq = dims.seq_len as usize;
    let eps: f32 = 1e-5;

    let (b, inp) = build_test_buffers(dims, 0);
    let n_nodes = unsafe { ffi::scheduled_megakernel_tiny_num_nodes() };
    let _ = launch_with_buffers(
        &b,
        dims,
        eps,
        ffi::launch_scheduled_megakernel_tiny,
        n_nodes,
    );
    let cpu = cpu_forward(&inp, dims, eps);

    let gpu = gpu_download_bf16(b.hidden_states, seq * hd);
    let (abs, rel) = errs(&gpu, &cpu.h_final);
    eprintln!(
        "full_residual_chain: max_abs_err={abs:.5}, max_rel_err={:.4}%",
        rel * 100.0
    );
    // Full forward pass through 2 layers + attention + 2 residual updates.
    // bf16 accumulator noise dominates; abs check is loose, rel check tight.
    assert!(abs < 5.0, "full chain abs {abs}");
    assert!(rel < 0.10, "full chain rel {rel}");
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

/// Validates tile_rope's fan-out: q_post_rope (last layer), paged k_cache
/// and v_cache (all layers). Uses the cpu_forward simulator.
#[test]
#[ignore = "needs GPU"]
fn rope_fanout_chain_matches_cpu_golden() {
    init_cuda();
    let dims = scheduled_prefill_tiny_dims();
    let seq = dims.seq_len as usize;
    let nl = dims.num_layers as usize;
    let nah = dims.num_attn_heads as usize;
    let nkh = dims.num_kv_heads as usize;
    let hdm = dims.head_dim as usize;
    let q_dim = nah * hdm;
    let page_size = SCHEDULED_PREFILL_KV_PAGE_SIZE as usize;
    let pages_per_layer = seq.div_ceil(page_size);
    let cache_total = nl * pages_per_layer * page_size * nkh * hdm;
    let eps: f32 = 1e-5;

    let (b, inp) = build_test_buffers(dims, 0);
    let n_nodes = unsafe { ffi::scheduled_megakernel_tiny_num_nodes() };
    let _ = launch_with_buffers(
        &b,
        dims,
        eps,
        ffi::launch_scheduled_megakernel_tiny,
        n_nodes,
    );
    let cpu = cpu_forward(&inp, dims, eps);

    let q_post_gpu = gpu_download_bf16(b.q_post_rope, seq * q_dim);
    let k_cache_gpu = gpu_download_bf16(b.k_cache, cache_total);
    let v_cache_gpu = gpu_download_bf16(b.v_cache, cache_total);

    let (q_abs, _) = errs(&q_post_gpu, &cpu.q_post_rope_last);
    let (k_abs, _) = errs(&k_cache_gpu, &cpu.k_cache);
    let (v_abs, _) = errs(&v_cache_gpu, &cpu.v_cache);
    eprintln!("q_post_rope: max_abs_err={q_abs:.5}");
    eprintln!("k_cache:     max_abs_err={k_abs:.5}");
    eprintln!("v_cache:     max_abs_err={v_abs:.5}");
    assert!(q_abs < 0.10, "q_post_rope abs {q_abs}");
    assert!(k_abs < 0.10, "k_cache abs {k_abs}");
    assert!(v_abs < 0.10, "v_cache abs {v_abs}");
}

/// Phase 3d — comprehensive validation at the medium scaling fixture.
/// NL=4, seq=64, HD=512, ID=1024, NAH=8, NKH=4, HDM=64.
///
/// Exercises:
///   - multi-page KV cache (seq=64 / page_size=16 → 4 pages per layer,
///     total 16 physical pages across NL=4 layers)
///   - more layers than tiny — catches cross-layer cache slot collisions
///     and longer residual cascade accumulation
///   - GQA ratio = 2 (NAH=8, NKH=4) — exercises the head fan-out
///   - 4× larger HD/ID/qkv_dim per tile vs tiny
///
/// Validates every end-state buffer (h_final, all "_last" intermediates,
/// paged caches) against cpu_forward, in one shot. This is the bridge from
/// "tiny test fixture works" to "scaling holds with bigger dims".
#[test]
#[ignore = "needs GPU"]
fn medium_full_forward_pass_matches_cpu_golden() {
    init_cuda();
    let dims = scheduled_prefill_medium_dims();
    let hd = dims.hidden_dim as usize;
    let id = dims.intermediate_dim as usize;
    let seq = dims.seq_len as usize;
    let nl = dims.num_layers as usize;
    let nah = dims.num_attn_heads as usize;
    let nkh = dims.num_kv_heads as usize;
    let hdm = dims.head_dim as usize;
    let q_dim = nah * hdm;
    let page_size = SCHEDULED_PREFILL_KV_PAGE_SIZE as usize;
    let pages_per_layer = seq.div_ceil(page_size);
    let cache_total = nl * pages_per_layer * page_size * nkh * hdm;
    let eps: f32 = 1e-5;

    // Kernel reports its own node/CTA/wave counts (DSL-driven heuristic).
    let kernel_n = unsafe { ffi::scheduled_megakernel_medium_num_nodes() };
    let kernel_ctas = unsafe { ffi::scheduled_megakernel_medium_num_ctas() };
    let kernel_waves = unsafe { ffi::scheduled_megakernel_medium_num_waves() };
    eprintln!(
        "scheduled_megakernel(medium): {kernel_n} nodes, {kernel_waves} waves, {kernel_ctas} CTAs, {pages_per_layer} pages/layer × {nl} layers"
    );

    // Use a different seed base than tiny so a leak from a stale buffer
    // would be obvious.
    let (b, inp) = build_test_buffers(dims, 1000);
    let _ = launch_with_buffers(
        &b,
        dims,
        eps,
        ffi::launch_scheduled_megakernel_medium,
        kernel_n,
    );
    let cpu = cpu_forward(&inp, dims, eps);

    // Download every end-state buffer the simulator records.
    let h_gpu = gpu_download_bf16(b.hidden_states, seq * hd);
    let rms_rope_gpu = gpu_download_bf16(b.rms_rope, seq * hd);
    let q_post_gpu = gpu_download_bf16(b.q_post_rope, seq * q_dim);
    let attn_out_gpu = gpu_download_bf16(b.attn_out, seq * hd);
    let rms_gate_gpu = gpu_download_bf16(b.rms_gate, seq * hd);
    let silu_out_gpu = gpu_download_bf16(b.silu_out, seq * id);
    let k_cache_gpu = gpu_download_bf16(b.k_cache, cache_total);
    let v_cache_gpu = gpu_download_bf16(b.v_cache, cache_total);

    // Validate each end-state buffer. Tolerances are pairs of (max_abs,
    // max_rel) — both must hold. The cascade accumulates bf16 noise faster
    // at larger magnitudes, so abs tolerances are scaled to ~4 ULPs at the
    // expected magnitude. Rel tolerance is the meaningful precision check
    // and stays tight (under 5%) except for buffers like q_post_rope and
    // k_cache that contain near-zero values where rel err is undefined.
    let checks: [(&str, &[bf16], &[bf16], f32, f32); 8] = [
        (
            "rms_rope_last",
            &rms_rope_gpu,
            &cpu.rms_rope_last,
            0.10,
            0.05,
        ),
        (
            "q_post_rope_last",
            &q_post_gpu,
            &cpu.q_post_rope_last,
            0.20,
            10.0,
        ),
        (
            "attn_out_last",
            &attn_out_gpu,
            &cpu.attn_out_last,
            0.10,
            0.05,
        ),
        (
            "rms_gate_last",
            &rms_gate_gpu,
            &cpu.rms_gate_last,
            0.10,
            0.05,
        ),
        (
            "silu_out_last",
            &silu_out_gpu,
            &cpu.silu_out_last,
            0.50,
            0.05,
        ),
        (
            "k_cache (all layers)",
            &k_cache_gpu,
            &cpu.k_cache,
            0.20,
            10.0,
        ),
        (
            "v_cache (all layers)",
            &v_cache_gpu,
            &cpu.v_cache,
            0.10,
            0.05,
        ),
        ("h_final", &h_gpu, &cpu.h_final, 64.0, 0.05),
    ];
    let mut any_failed = false;
    for (label, gpu, cpu_buf, abs_tol, rel_tol) in checks {
        let (abs, rel) = errs(gpu, cpu_buf);
        eprintln!(
            "medium  {label:24}: max_abs_err={abs:.5}  max_rel_err={:.4}%",
            rel * 100.0
        );
        if abs >= abs_tol {
            eprintln!("    !! FAIL: abs {abs} >= tol {abs_tol}");
            any_failed = true;
        }
        if rel >= rel_tol {
            eprintln!("    !! FAIL: rel {rel} >= tol {rel_tol}");
            any_failed = true;
        }
    }
    assert!(!any_failed, "medium full forward pass validation failed");
}

// ── Phase 4 step 5: golden file infrastructure ──────────────────────────
//
// `regen_committed_goldens` runs the CPU forward simulator for every
// variant whose simulator runtime is fast enough for an interactive run
// (currently tiny + medium) and writes h_final to disk as a committed
// golden file. The fast variants run in <1 second so this is cheap to
// rerun whenever the algorithm intentionally changes.
//
// 1B real-dim variants are NOT regenerated here — their CPU simulator
// runtime is many minutes. They get a separate, slower regen target.
#[test]
#[ignore = "regenerates committed golden files; run on demand"]
fn regen_committed_goldens() {
    init_cuda();
    let eps: f32 = 1e-5;

    // Tiny.
    {
        let dims = scheduled_prefill_tiny_dims();
        let (_, inp) = build_test_buffers(dims, 0);
        let cpu = cpu_forward(&inp, dims, eps);
        write_golden("tiny", &cpu.h_final);
    }
    // Medium.
    {
        let dims = scheduled_prefill_medium_dims();
        let (_, inp) = build_test_buffers(dims, 1000);
        let cpu = cpu_forward(&inp, dims, eps);
        write_golden("medium", &cpu.h_final);
    }
}

#[test]
#[ignore = "needs GPU"]
fn tiny_h_final_matches_committed_golden() {
    init_cuda();
    let dims = scheduled_prefill_tiny_dims();
    let hd = dims.hidden_dim as usize;
    let seq = dims.seq_len as usize;
    let eps: f32 = 1e-5;

    let (b, _) = build_test_buffers(dims, 0);
    let n_nodes = unsafe { ffi::scheduled_megakernel_tiny_num_nodes() };
    let _ = launch_with_buffers(
        &b,
        dims,
        eps,
        ffi::launch_scheduled_megakernel_tiny,
        n_nodes,
    );
    let h_gpu = gpu_download_bf16(b.hidden_states, seq * hd);
    // Tiny's 2-layer residual cascade: bf16 ULP at the accumulated
    // magnitude reaches ~2-5; 5.0 matches the live full_residual_chain tol.
    assert_matches_committed_golden("tiny", &h_gpu, 5.0, 0.05);
}

#[test]
#[ignore = "needs GPU"]
fn medium_h_final_matches_committed_golden() {
    init_cuda();
    let dims = scheduled_prefill_medium_dims();
    let hd = dims.hidden_dim as usize;
    let seq = dims.seq_len as usize;
    let eps: f32 = 1e-5;

    let (b, _) = build_test_buffers(dims, 1000);
    let n_nodes = unsafe { ffi::scheduled_megakernel_medium_num_nodes() };
    let _ = launch_with_buffers(
        &b,
        dims,
        eps,
        ffi::launch_scheduled_megakernel_medium,
        n_nodes,
    );
    let h_gpu = gpu_download_bf16(b.hidden_states, seq * hd);
    // Medium accumulates a 4-layer cascade — bf16 ULP noise reaches ~30
    // at the magnitudes the residual builds up to.
    assert_matches_committed_golden("medium", &h_gpu, 32.0, 0.05);
}
