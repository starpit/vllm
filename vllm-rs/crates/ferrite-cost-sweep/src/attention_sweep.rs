// SPDX-License-Identifier: Apache-2.0
//! Attention sweep — FA2 vs FlashInfer across `(num_tokens, sk) × per-tuple`.
//!
//! For every head_dim in [`ATTN_SHAPES`] and every `FlashInferConfig` in
//! `FLASHINFER_CONFIG_SET`, times the paged attention step at each
//! `(m, sk)` grid point and emits CSV rows in the solver's
//! `(kernel, M, N, K, cost_us)` schema with axis overloading:
//!
//! ```text
//!   M = num_tokens
//!   N = sk_bucket (KV-cache span in tokens)
//!   K = head_dim
//! ```
//!
//! Kernel names: `fa2_attn_bf16_h{h}` (FA2 is softcap-agnostic — the
//! capping is a runtime float arg, not a template) and
//! `flashinfer_attn_bf16_h{h}_{no}softcap` (FI specializes on softcap at
//! compile time). `(num_qo_heads, num_kv_heads)` is a runtime parameter
//! of both kernels — the GQA ratio doesn't appear in the kernel name,
//! so the same row serves every model at that head_dim.
//!
//! Task #6's FI Impls consume these rows via
//! `profile.cost_us_for(fi_csv_name(head_dim, softcap), ctx.num_tokens(),
//!                      ctx.sk_bucket(), head_dim)` and gate compatibility
//! on CSV-row presence AND [`dispatch_for`] availability.
//!
//! Grid cells with `m > sk` are skipped — physically the prefill invariant
//! is `m ≤ sk` and decode is always `m == 1`. An absent row causes the
//! solver to route through FA2's fallback path at that workload.

#![cfg(feature = "cuda")]

use core::ffi::c_void;

use cudarc::driver::sys;
use ferrite_cuda_builder::flashinfer_config::{DType as BuilderDType, FLASHINFER_CONFIG_SET};
use ferrite_kernels::flashinfer::{
    FiDType, FlashInferConfig as RtFlashInferConfig, dispatch_for, workspace_bytes,
};

use crate::util::{bench_kernel, gpu_alloc_fill, gpu_alloc_zeros, h2d_copy, query_num_sm};

// FA2 paged attention — raw extern-C symbol from `libvllm_flash_attn.a`.
// Mirrors `ferrite_kernels::kernels::mha_varlen_fwd` exactly. Re-declared
// here so the sweep times the kernel at the extern boundary (no
// `OwnedTensor` / `CachingAllocator` noise in the measurement).
#[allow(clippy::too_many_arguments)]
unsafe extern "C" {
    fn mha_varlen_fwd(
        q_ptr: *mut c_void,
        k_ptr: *mut c_void,
        v_ptr: *mut c_void,
        out_ptr: *mut c_void,
        softmax_lse_ptr: *mut c_void,

        cu_seqlens_q: *const i32,
        cu_seqlens_k: *const i32,
        seqused_k: *const i32,

        block_table: *const i32,
        block_table_batch_stride: i32,

        batch_size: i32,
        max_seqlen_q: i32,
        max_seqlen_k: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_size: i32,
        page_block_size: i32,

        q_row_stride: i64,
        q_head_stride: i64,
        k_batch_stride: i64,
        k_row_stride: i64,
        k_head_stride: i64,
        o_row_stride: i64,
        o_head_stride: i64,

        softmax_scale: f32,
        is_causal: i32,
        window_size_left: i32,
        window_size_right: i32,
        softcap: f32,
        is_bf16: i32,
        num_splits: i32,

        softmax_lse_accum_ptr: *mut c_void,
        out_accum_ptr: *mut c_void,
        seqlenq_ngroups_swapped: i32,
        total_q: i32,

        rotary_cos_ptr: *const c_void,
        rotary_sin_ptr: *const c_void,
        rotary_dim: i32,
        rotate_cached_k: i32,
        is_rotary_interleaved: i32,

        block_unrotated_flags: *const u8,

        stream: sys::CUstream,
    );
}

/// Per-cell iteration counts. Attention is much heavier than a GEMM row,
/// so fewer iterations — but still enough to average out variance. Matches
/// the shape of `attn_bench.cu`'s `warmup=20 / iters=500` at the tighter
/// budget a CSV calibration tolerates.
const WARMUP: u32 = 10;
const ITERS: u32 = 50;

/// Calibration shapes — one per head_dim present in `FLASHINFER_CONFIG_SET`.
/// Cost rows are keyed only on `(head_dim, softcap)` (the compile-time FI
/// specialization dims); `(num_qo_heads, num_kv_heads)` is a runtime
/// parameter of the compiled kernel, so a single canonical `(q, k)` per
/// head_dim gives a cost table that serves every GQA ratio. Picked to
/// match common Llama/Qwen shapes (32, 8 for h64/h128; 16, 2 for h256)
/// so the first row measured is representative.
const ATTN_SHAPES: &[(u32, u32, u32)] = &[
    (64, 32, 8),  // Llama-3.2-1B / Qwen2.5-0.5B (any q/k works at runtime)
    (128, 32, 8), // Llama-3.1-8B / Qwen2.5-7B
    (256, 16, 2), // Gemma/DeepSeek-style large head_dim
];

/// Page size for paged-KV. Fixed at vLLM's default block size — matches
/// `attn_bench.cu` and every in-tree model's `KvCachePool::block_size`.
const PAGE_SIZE: u32 = 16;

/// `num_tokens` grid — same layout as `gemm_sweep::M_VALUES`.
const M_VALUES: &[u32] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

/// `sk_bucket` grid — matches `attention_helpers::sk_bucket_for` (and the
/// optional `sk_buckets = [...]` arg on `#[forward]`).
const SK_VALUES: &[u32] = &[128, 256, 512, 1024, 2048, 4096, 8192];

/// Softmax scale. Value is arbitrary for timing purposes — FA2 and FI
/// both apply it as a single fused multiply inside the inner loop.
const SM_SCALE: f32 = 0.125;

/// bf16 byte pattern producing a small positive value (~0.0097). Mirrors
/// `attn_bench.cu:79-81` — avoids the degenerate all-zeros softmax where
/// every attention score is equal. Timing-only, so the exact value is
/// unimportant beyond "finite, non-denormal".
const BF16_FILL: u8 = 0x3c;

pub fn run(launch_overhead_us: f64) {
    let stream: sys::CUstream = std::ptr::null_mut();
    let num_sm = query_num_sm();

    for &(head_dim, num_qo_heads, num_kv_heads) in ATTN_SHAPES {
        assert_eq!(
            num_qo_heads % num_kv_heads,
            0,
            "GQA invariant: num_qo_heads must be divisible by num_kv_heads"
        );
        for &sk in SK_VALUES {
            for &m in M_VALUES {
                if m > sk {
                    // Skip physically invalid cells — prefill requires
                    // `m ≤ sk`, decode has `m == 1`. Solver handles
                    // missing rows via its existing fallback logic.
                    continue;
                }
                bench_cell(
                    stream,
                    launch_overhead_us,
                    num_sm,
                    head_dim,
                    num_qo_heads,
                    num_kv_heads,
                    PAGE_SIZE,
                    m,
                    sk,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn bench_cell(
    stream: sys::CUstream,
    launch_overhead_us: f64,
    num_sm: i32,
    head_dim: u32,
    num_qo_heads: u32,
    num_kv_heads: u32,
    page_size: u32,
    m: u32,
    sk: u32,
) {
    let num_pages = sk.div_ceil(page_size);
    let head_dim_us = head_dim as usize;

    // Buffer sizes (all bf16 = 2 bytes/elem unless noted).
    let q_bytes = (m * num_qo_heads * head_dim) as usize * 2;
    let kv_bytes = (num_pages * page_size * num_kv_heads * head_dim) as usize * 2;
    let lse_bytes = (m * num_qo_heads) as usize * 4; // f32

    let q = gpu_alloc_fill(q_bytes, BF16_FILL);
    let k = gpu_alloc_fill(kv_bytes, BF16_FILL);
    let v = gpu_alloc_fill(kv_bytes, BF16_FILL);
    let o = gpu_alloc_zeros(q_bytes);
    let softmax_lse = gpu_alloc_zeros(lse_bytes);

    // Small index buffers — all batch=1.
    let cu_seqlens_q = gpu_alloc_zeros(2 * 4);
    let cu_seqlens_k = gpu_alloc_zeros(2 * 4);
    let seqused_k = gpu_alloc_zeros(4);
    let block_table = gpu_alloc_zeros(num_pages as usize * 4);
    let kv_indices = gpu_alloc_zeros(num_pages as usize * 4);

    // Upload small host-side setup once per cell.
    unsafe {
        let cu_q: [i32; 2] = [0, m as i32];
        let cu_k: [i32; 2] = [0, sk as i32];
        let sused: [i32; 1] = [sk as i32];
        h2d_copy(cu_seqlens_q, cu_q.as_ptr() as *const u8, 2 * 4);
        h2d_copy(cu_seqlens_k, cu_k.as_ptr() as *const u8, 2 * 4);
        h2d_copy(seqused_k, sused.as_ptr() as *const u8, 4);

        let indices: Vec<i32> = (0..num_pages as i32).collect();
        h2d_copy(
            block_table,
            indices.as_ptr() as *const u8,
            num_pages as usize * 4,
        );
        h2d_copy(
            kv_indices,
            indices.as_ptr() as *const u8,
            num_pages as usize * 4,
        );
    }

    // ── FA2 ──
    // Strides match kernels.rs::flash_attn_paged_ext's paged layout.
    let q_row_stride = (num_qo_heads * head_dim) as i64;
    let q_head_stride = head_dim as i64;
    let k_batch_stride = (page_size * num_kv_heads * head_dim) as i64;
    let k_row_stride = (num_kv_heads * head_dim) as i64;
    let k_head_stride = head_dim as i64;

    let fa2_launch = || unsafe {
        mha_varlen_fwd(
            q as *mut c_void,
            k as *mut c_void,
            v as *mut c_void,
            o as *mut c_void,
            softmax_lse as *mut c_void,
            cu_seqlens_q as *const i32,
            cu_seqlens_k as *const i32,
            seqused_k as *const i32,
            block_table as *const i32,
            num_pages as i32, // block_table_batch_stride
            1,                // batch_size
            m as i32,
            sk as i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            q_row_stride,
            q_head_stride,
            k_batch_stride,
            k_row_stride,
            k_head_stride,
            q_row_stride,  // o_row_stride
            q_head_stride, // o_head_stride
            SM_SCALE,
            1,   // is_causal
            -1,  // window_size_left (disabled)
            0,   // window_size_right (causal)
            0.0, // softcap — runtime; kernel is not specialized on it
            1,   // is_bf16
            1,   // num_splits
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,        // seqlenq_ngroups_swapped
            m as i32, // total_q
            std::ptr::null(),
            std::ptr::null(),
            0,
            0,
            0,
            std::ptr::null(),
            stream,
        )
    };
    let us = (bench_kernel(stream, WARMUP, ITERS, fa2_launch) - launch_overhead_us).max(0.0);
    println!("fa2_attn_bf16_h{head_dim},{m},{sk},{head_dim},{us:.2}");

    // ── FlashInfer (one row per FLASHINFER_CONFIG_SET tuple matching this head_dim) ──
    let (float_ws, int_ws) = workspace_bytes(num_sm, head_dim_us, num_kv_heads as usize);

    for cfg in FLASHINFER_CONFIG_SET.iter().copied() {
        if cfg.head_dim != head_dim {
            continue;
        }
        // Only bf16 tuples compile today — the Fp16 variant is a
        // placeholder (see FLASHINFER_CONFIG_SET doc comment).
        let rt_cfg = match cfg.dtype {
            BuilderDType::Bf16 => RtFlashInferConfig {
                dtype: FiDType::Bf16,
                head_dim,
                use_logits_soft_cap: cfg.use_logits_soft_cap,
            },
            BuilderDType::Fp16 => continue,
        };
        let Some(dispatch) = dispatch_for(rt_cfg) else {
            eprintln!("attention_sweep: no dispatch for {:?} (skipping)", rt_cfg);
            continue;
        };

        let mut rc: i32 = 0;
        // `logits_soft_cap` is a template-specialized constant in the FI
        // shim; the runtime float arg is only consulted when the template
        // has softcap enabled. Pass 0.0 otherwise — matches the bench's
        // convention and the prior-session benchmark.
        let soft_cap_arg = if cfg.use_logits_soft_cap { 30.0 } else { 0.0 };
        let handle = unsafe {
            (dispatch.plan_new)(
                q as *const c_void,
                k as *const c_void,
                v as *const c_void,
                kv_indices as *const i32,
                o as *mut c_void,
                m as i32,
                sk as i32,
                num_qo_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                page_size as i32,
                num_pages as i32,
                num_sm, // target_num_clusters
                float_ws,
                int_ws,
                SM_SCALE,
                soft_cap_arg,
                stream,
                &mut rc as *mut i32,
            )
        };
        if handle.is_null() {
            eprintln!(
                "fi_plan_new failed for cfg={:?} m={m} sk={sk} rc={rc}",
                rt_cfg
            );
            continue;
        }

        // Trial launch — detects runtime errors the planner can't catch
        // (e.g. sm_89 vs head_dim=256 exceeds the 100 KB max dynamic smem
        // budget via cudaFuncSetAttribute). If the kernel fails here we
        // skip row emission so the solver's CostTable doesn't ingest
        // fast-fail timings as if they were legitimate costs. The loss
        // of one failed cell means the Impl falls back to FA2 at that
        // workload — a graceful degradation, not a correctness hole.
        let trial_rc = unsafe { (dispatch.run)(handle, stream) };
        if trial_rc != 0 {
            eprintln!(
                "fi_run trial rc={trial_rc} for cfg={:?} m={m} sk={sk} — skipping row",
                rt_cfg
            );
            unsafe { (dispatch.plan_delete)(handle) };
            continue;
        }

        let fi_launch = || {
            let _ = unsafe { (dispatch.run)(handle, stream) };
        };
        let us = (bench_kernel(stream, WARMUP, ITERS, fi_launch) - launch_overhead_us).max(0.0);
        let softcap_tok = if cfg.use_logits_soft_cap {
            "softcap"
        } else {
            "nosoftcap"
        };
        println!("flashinfer_attn_bf16_h{head_dim}_{softcap_tok},{m},{sk},{head_dim},{us:.2}");

        unsafe { (dispatch.plan_delete)(handle) };
    }

    unsafe {
        sys::cuMemFree_v2(q);
        sys::cuMemFree_v2(k);
        sys::cuMemFree_v2(v);
        sys::cuMemFree_v2(o);
        sys::cuMemFree_v2(softmax_lse);
        sys::cuMemFree_v2(cu_seqlens_q);
        sys::cuMemFree_v2(cu_seqlens_k);
        sys::cuMemFree_v2(seqused_k);
        sys::cuMemFree_v2(block_table);
        sys::cuMemFree_v2(kv_indices);
    }
}
