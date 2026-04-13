// SPDX-License-Identifier: Apache-2.0
//! Phase A.3 — standalone GPU smoke test for the FlashInfer attention shim.
//!
//! This test does NOT touch the scheduled megakernel. It exercises the
//! `run_flashinfer_attention_smoke` C++ shim
//! (`csrc/flashinfer_attention_shim.cu`) directly with synthetic inputs at
//! LLaMA-3.2-1B head dims, and compares the output against a naive
//! O(N²) CPU attention reference.
//!
//! Purpose: prove that
//!   - FlashInfer's `BatchPagedAttentionPersistent` runner runs cleanly
//!     on this build (cuda 12.9, sm_89, our shim TU)
//!   - the `PersistentParams` we hand-build matches what the runner reads
//!   - the NHD layout we use for Q / paged K / paged V is what the runner
//!     expects
//!   - the output is numerically correct for our exact LLaMA-1B config
//!     (head_dim=64, num_qo=32, num_kv=8, page_size=16)
//! all BEFORE we replace `tile_attention` with the FlashInfer runner.

#![cfg(feature = "cuda")]

use cudarc::driver::result;
use cudarc::driver::sys;
use ferrite_solver::target_profile::TargetProfile;
use ferrite_test_harness::ffi;
use half::bf16;

// ── Test dims (small enough that the CPU reference runs in milliseconds) ──
const SEQ_LEN: usize = 64;
const NUM_QO_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const PAGE_SIZE: usize = 16;
// SEQ_LEN / PAGE_SIZE = 4 pages.
const NUM_PAGES: usize = SEQ_LEN.div_ceil(PAGE_SIZE);

fn init_cuda() -> sys::CUdevice {
    result::init().expect("cuInit failed");
    let device = result::device::get(0).expect("cuDeviceGet failed");
    let ctx = unsafe { result::primary_ctx::retain(device) }.expect("cuCtxRetain failed");
    unsafe { result::ctx::set_current(ctx) }.expect("cuCtxSetCurrent failed");
    device
}

/// Query the device for `MultiProcessorCount`. Used to size FlashInfer
/// workspaces and to pass `num_sm` into the forked planner, so both
/// sides agree on the cluster count regardless of which GPU we're on.
fn device_num_sms(device: sys::CUdevice) -> u32 {
    let n = unsafe {
        result::device::get_attribute(
            device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
    }
    .expect("cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) failed");
    assert!(n > 0, "device reported MultiProcessorCount = {n}");
    n as u32
}

fn gpu_upload_bf16(host: &[bf16]) -> u64 {
    unsafe {
        let bytes = host.len() * 2;
        let dptr = result::malloc_sync(bytes).expect("cuMemAlloc failed");
        let host_u16: &[u16] = std::slice::from_raw_parts(host.as_ptr() as *const u16, host.len());
        result::memcpy_htod_sync(dptr, host_u16).expect("cuMemcpyHtoD failed");
        dptr
    }
}

fn gpu_upload_i32(host: &[i32]) -> u64 {
    unsafe {
        let bytes = host.len() * 4;
        let dptr = result::malloc_sync(bytes).expect("cuMemAlloc failed");
        result::memcpy_htod_sync(dptr, host).expect("cuMemcpyHtoD failed");
        dptr
    }
}

fn gpu_alloc_bf16(count: usize) -> u64 {
    unsafe {
        let bytes = count * 2;
        let dptr = result::malloc_sync(bytes).expect("cuMemAlloc failed");
        result::memset_d8_sync(dptr, 0, bytes).expect("cuMemsetD8 failed");
        dptr
    }
}

fn gpu_download_bf16(dptr: u64, count: usize) -> Vec<bf16> {
    unsafe {
        let mut host_u16 = vec![0u16; count];
        result::memcpy_dtoh_sync(&mut host_u16, dptr as cudarc::driver::sys::CUdeviceptr)
            .expect("cuMemcpyDtoH failed");
        host_u16.into_iter().map(bf16::from_bits).collect()
    }
}

/// Deterministic small bf16 fill — keeps softmax inputs in a sane range so
/// the CPU reference and bf16 GPU path agree to ~1e-2.
fn det_bf16_fill(n: usize, seed: u64) -> Vec<bf16> {
    let mut rng = seed;
    (0..n)
        .map(|_| {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map to [-0.25, 0.25]. Small enough that Q·K stays in [-1, 1]
            // even after the head_dim=64 dot product.
            let v = (((rng >> 33) as i32) as f32 / (i32::MAX as f32)) * 0.25;
            bf16::from_f32(v)
        })
        .collect()
}

/// Lay K (or V) out in the FlashInfer NHD paged format:
///   `[num_pages, page_size, num_kv_heads, head_dim]`
/// from a flat `[seq_len, num_kv_heads, head_dim]` source.
fn pack_paged(
    flat: &[bf16],
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    page_size: usize,
    num_pages: usize,
) -> Vec<bf16> {
    let mut out = vec![bf16::ZERO; num_pages * page_size * num_kv_heads * head_dim];
    for k in 0..seq_len {
        let page = k / page_size;
        let slot = k % page_size;
        for h in 0..num_kv_heads {
            for d in 0..head_dim {
                let src = (k * num_kv_heads + h) * head_dim + d;
                let dst = ((page * page_size + slot) * num_kv_heads + h) * head_dim + d;
                out[dst] = flat[src];
            }
        }
    }
    out
}

/// Naive O(N²) causal multi-head attention with GQA. Single sequence.
/// Q: [seq, num_qo, hd]    K, V: [seq, num_kv, hd]    O: [seq, num_qo, hd]
/// Computes everything in fp32 then casts back to bf16. Causal mask:
/// q at index i can attend to k indices [0..=i].
fn cpu_attention_ref(
    q: &[bf16],
    k: &[bf16],
    v: &[bf16],
    seq_len: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sm_scale: f32,
) -> Vec<bf16> {
    let gqa_ratio = num_qo_heads / num_kv_heads;
    let mut out = vec![bf16::ZERO; seq_len * num_qo_heads * head_dim];

    for q_idx in 0..seq_len {
        for h in 0..num_qo_heads {
            let kv_h = h / gqa_ratio;

            // ── Q · K^T for k in 0..=q_idx ──
            let mut scores = vec![0.0f32; q_idx + 1];
            for k_idx in 0..=q_idx {
                let mut s = 0.0f32;
                for d in 0..head_dim {
                    let qv = q[(q_idx * num_qo_heads + h) * head_dim + d].to_f32();
                    let kv = k[(k_idx * num_kv_heads + kv_h) * head_dim + d].to_f32();
                    s += qv * kv;
                }
                scores[k_idx] = s * sm_scale;
            }

            // ── softmax (numerically stable) ──
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                exp_sum += *s;
            }
            for s in &mut scores {
                *s /= exp_sum;
            }

            // ── weighted sum of V ──
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for k_idx in 0..=q_idx {
                    let vv = v[(k_idx * num_kv_heads + kv_h) * head_dim + d].to_f32();
                    acc += scores[k_idx] * vv;
                }
                out[(q_idx * num_qo_heads + h) * head_dim + d] = bf16::from_f32(acc);
            }
        }
    }

    out
}

fn errs(a: &[bf16], b: &[bf16]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = x.to_f32();
        let yf = y.to_f32();
        let abs = (xf - yf).abs();
        if abs > max_abs {
            max_abs = abs;
        }
        let denom = yf.abs().max(1e-6);
        let rel = abs / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    (max_abs, max_rel)
}

#[test]
#[ignore = "needs GPU"]
fn flashinfer_attention_runner_smoke() {
    let device = init_cuda();

    // ── Inputs (host) ──
    let q_flat = det_bf16_fill(SEQ_LEN * NUM_QO_HEADS * HEAD_DIM, /*seed=*/ 1);
    let k_flat = det_bf16_fill(SEQ_LEN * NUM_KV_HEADS * HEAD_DIM, /*seed=*/ 2);
    let v_flat = det_bf16_fill(SEQ_LEN * NUM_KV_HEADS * HEAD_DIM, /*seed=*/ 3);

    let sm_scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    // ── CPU reference ──
    let cpu_o = cpu_attention_ref(
        &q_flat,
        &k_flat,
        &v_flat,
        SEQ_LEN,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        sm_scale,
    );

    // ── Pack K / V into the FlashInfer paged NHD layout ──
    let k_paged = pack_paged(
        &k_flat,
        SEQ_LEN,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        NUM_PAGES,
    );
    let v_paged = pack_paged(
        &v_flat,
        SEQ_LEN,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        NUM_PAGES,
    );

    // Identity page table: page i lives at slot i in the cache.
    let kv_indices: Vec<i32> = (0..NUM_PAGES as i32).collect();

    // ── Upload to GPU ──
    let q_d = gpu_upload_bf16(&q_flat);
    let k_d = gpu_upload_bf16(&k_paged);
    let v_d = gpu_upload_bf16(&v_paged);
    let kv_indices_d = gpu_upload_i32(&kv_indices);
    let o_d = gpu_alloc_bf16(SEQ_LEN * NUM_QO_HEADS * HEAD_DIM);

    // ── Workspace sizes from a device-derived target profile. ──
    //
    // The test queries the running GPU for its SM count and patches
    // the L4 profile's `num_sm` field with the real value. Both the
    // workspace calculation and the `target_num_clusters` argument
    // passed into the shim's forked planner (`TwoStageHolisticPlanWithNumSm`)
    // are derived from the SAME number, so they cannot disagree on
    // how big the planner's bump-allocator scratch needs to be. This
    // makes the test portable across any sm80+ card (L4, L40S, A100,
    // H100, ...) without maintaining per-GPU `TargetProfile`
    // constructors just for the smoke test. A proper per-GPU
    // `TargetProfile` is still useful for the solver's cost model
    // (see SOLVER_HANDOFF.md TODO "Multi-GPU cost CSVs"), but it's
    // orthogonal to this test.
    let num_sm = device_num_sms(device);
    let mut profile = TargetProfile::l4_sm89();
    profile.num_sm = num_sm;
    let num_clusters = profile.cooperative_grid_size();
    let float_ws_bytes =
        profile.flashinfer_float_workspace_bytes(HEAD_DIM as u32, NUM_KV_HEADS as u32);
    let int_ws_bytes = profile.flashinfer_int_workspace_bytes();

    // ── Run shim ──
    let status = unsafe {
        ffi::run_flashinfer_attention_smoke(
            q_d as *mut u16,
            k_d as *mut u16,
            v_d as *mut u16,
            kv_indices_d as *mut i32,
            o_d as *mut u16,
            SEQ_LEN as i32,
            NUM_QO_HEADS as i32,
            NUM_KV_HEADS as i32,
            HEAD_DIM as i32,
            PAGE_SIZE as i32,
            NUM_PAGES as i32,
            float_ws_bytes,
            int_ws_bytes,
            num_clusters as i32,
            sm_scale,
            /*stream=*/ 0,
        )
    };
    assert_eq!(
        status, 0,
        "run_flashinfer_attention_smoke failed (status={status})"
    );

    // ── Download GPU output ──
    let gpu_o = gpu_download_bf16(o_d, SEQ_LEN * NUM_QO_HEADS * HEAD_DIM);

    let (max_abs, max_rel) = errs(&gpu_o, &cpu_o);
    eprintln!(
        "flashinfer_attention_smoke: max_abs_err={max_abs:.5}, max_rel_err={:.4}%",
        max_rel * 100.0
    );

    // bf16 noise + mma rounding tolerance. The CPU reference is fp32 inside,
    // bf16 outside. Per-element disagreement up to ~1e-2 is normal at these
    // dims for a single matmul + softmax + matmul.
    assert!(
        max_abs < 5e-2,
        "flashinfer attention abs err {max_abs} > 5e-2"
    );
}
