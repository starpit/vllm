// SPDX-License-Identifier: Apache-2.0
//! Shared attention helpers for FP8 KV cache support.
//!
//! All model attention modules call these instead of directly invoking
//! reshape_and_cache/flash_attn to get FP8 quantize-on-write and
//! dequant-on-read behavior transparently.

use crate::flashinfer::{self, FlashInferConfig, FlashInferPlanCache};
use crate::kernels;
use crate::kv_cache::KvCachePool;
use ferrite_cuda_core::alloc::{CachingAllocator, OwnedTensor};
use ferrite_cuda_core::dtype::DType;
use ferrite_cuda_core::tensor::{GpuTensor, TensorView};

type CUstream = cudarc::driver::sys::CUstream;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Write K/V into the paged cache, handling both BF16 and FP8 paths.
///
/// When the cache is FP8, quantizes BF16/F16 K/V using per-layer scales.
/// When BF16/F16, performs a direct copy (existing path).
#[allow(clippy::too_many_arguments)]
pub unsafe fn write_kv_cache(
    k: TensorView<'_>,
    v: TensorView<'_>,
    slot_mapping: TensorView<'_>,
    kv_cache: &KvCachePool,
    layer_idx: usize,
    stream: CUstream,
) {
    if kv_cache.is_fp8() {
        kernels::reshape_and_cache_fp8(
            *k,
            *v,
            *kv_cache.k_cache(layer_idx),
            *kv_cache.v_cache(layer_idx),
            *slot_mapping,
            kv_cache.k_scale_ptr(layer_idx),
            kv_cache.v_scale_ptr(layer_idx),
            kv_cache.block_size,
            stream,
        );
    } else {
        kernels::reshape_and_cache(
            *k,
            *v,
            *kv_cache.k_cache(layer_idx),
            *kv_cache.v_cache(layer_idx),
            *slot_mapping,
            kv_cache.block_size,
            stream,
        );
    }
}

/// Run attention: fresh prefill (contiguous) or decode (paged/dequant).
///
/// Standard path (no softcap, no sliding window). Used by LLaMA, CommandR,
/// DeepSeek, Qwen3Next.
///
/// When FP8 decode: dequantizes KV from cache pages into contiguous BF16
/// buffers, then runs contiguous FA2. Fresh prefill always uses BF16 K/V
/// from the QKV projection directly.
#[allow(clippy::too_many_arguments)]
/// `cos_sin_cache_ptr`: optional pointer to `[max_pos, rotary_dim]` cos/sin cache
///   for fused RoPE on cached K (spans). Pass `std::ptr::null()` when not using spans.
pub unsafe fn attention_standard(
    q: TensorView<'_>,
    k: TensorView<'_>,
    v: TensorView<'_>,
    cu_seqlens_q: TensorView<'_>,
    seqused_k: TensorView<'_>,
    block_table: TensorView<'_>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    scale: f32,
    kv_cache: &KvCachePool,
    layer_idx: usize,
    num_sm: i32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
    cos_sin_cache_ptr: *const u8,
    rotary_dim: usize,
    is_rotary_interleaved: bool,
) -> OwnedTensor {
    let fresh_prefill = max_seqlen_q > 1 && max_seqlen_q == max_seqlen_k;

    if fresh_prefill {
        // Fresh prefill: use BF16 K/V from QKV projection (no cache read).
        // Pass cos_sin_cache so FA2 can apply RoPE to contiguous K if needed.
        kernels::flash_attn_contiguous(
            *q,
            *k,
            *v,
            *cu_seqlens_q,
            *cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            true,
            0.0,
            -1,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
        )
    } else if kv_cache.is_fp8() {
        // FP8 decode: dequant pages → contiguous FA2 with fused RoPE.
        fp8_decode_attention(
            *q,
            *cu_seqlens_q,
            *seqused_k,
            *block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            0.0,
            -1,
            kv_cache,
            layer_idx,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
        )
    } else {
        // BF16 paged FA2 — with optional fused RoPE for spans.
        kernels::flash_attn_paged_ext(
            *q,
            *kv_cache.k_cache(layer_idx),
            *kv_cache.v_cache(layer_idx),
            *cu_seqlens_q,
            *seqused_k,
            *block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            true,
            0.0,
            -1,
            kv_cache.block_size,
            num_sm,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
            kv_cache.block_unrotated_gpu(),
        )
    }
}

/// Decode-only attention from cache (K/V already written by fused kernel).
///
/// Handles both BF16 paged and FP8 dequant paths. Does NOT support prefill
/// (caller must ensure max_seqlen_q == 1).
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_decode_from_cache(
    q: TensorView<'_>,
    cu_seqlens_q: TensorView<'_>,
    seqused_k: TensorView<'_>,
    block_table: TensorView<'_>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    scale: f32,
    softcap: f32,
    window_size_left: i32,
    kv_cache: &KvCachePool,
    layer_idx: usize,
    num_sm: i32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
    cos_sin_cache_ptr: *const u8,
    rotary_dim: usize,
    is_rotary_interleaved: bool,
) -> OwnedTensor {
    if kv_cache.is_fp8() {
        fp8_decode_attention(
            *q,
            *cu_seqlens_q,
            *seqused_k,
            *block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            softcap,
            window_size_left,
            kv_cache,
            layer_idx,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
        )
    } else {
        kernels::flash_attn_paged_ext(
            *q,
            *kv_cache.k_cache(layer_idx),
            *kv_cache.v_cache(layer_idx),
            *cu_seqlens_q,
            *seqused_k,
            *block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            true,
            softcap,
            window_size_left,
            kv_cache.block_size,
            num_sm,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
            kv_cache.block_unrotated_gpu(),
        )
    }
}

/// Run attention with softcap and/or sliding window (Gemma2/Gemma3).
///
/// Uses `flash_attn_paged_ext` for the BF16 decode path.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_ext(
    q: TensorView<'_>,
    k: TensorView<'_>,
    v: TensorView<'_>,
    cu_seqlens_q: TensorView<'_>,
    seqused_k: TensorView<'_>,
    block_table: TensorView<'_>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    scale: f32,
    softcap: f32,
    window_size_left: i32,
    kv_cache: &KvCachePool,
    layer_idx: usize,
    num_sm: i32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
    cos_sin_cache_ptr: *const u8,
    rotary_dim: usize,
    is_rotary_interleaved: bool,
) -> OwnedTensor {
    let fresh_prefill = max_seqlen_q > 1 && max_seqlen_q == max_seqlen_k;

    if fresh_prefill {
        kernels::flash_attn_contiguous(
            *q,
            *k,
            *v,
            *cu_seqlens_q,
            *cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            true,
            softcap,
            window_size_left,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
        )
    } else if kv_cache.is_fp8() {
        fp8_decode_attention(
            *q,
            *cu_seqlens_q,
            *seqused_k,
            *block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            softcap,
            window_size_left,
            kv_cache,
            layer_idx,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
        )
    } else {
        kernels::flash_attn_paged_ext(
            *q,
            *kv_cache.k_cache(layer_idx),
            *kv_cache.v_cache(layer_idx),
            *cu_seqlens_q,
            *seqused_k,
            *block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            true,
            softcap,
            window_size_left,
            kv_cache.block_size,
            num_sm,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
            kv_cache.block_unrotated_gpu(),
        )
    }
}

// ---------------------------------------------------------------------------
// FP8 decode internals
// ---------------------------------------------------------------------------

/// FP8 decode path: dequant KV pages → contiguous BF16 → FA2.
#[allow(clippy::too_many_arguments)]
unsafe fn fp8_decode_attention(
    q: GpuTensor,
    cu_seqlens_q: GpuTensor,
    seqused_k: GpuTensor,
    block_table: GpuTensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    scale: f32,
    softcap: f32,
    window_size_left: i32,
    kv_cache: &KvCachePool,
    layer_idx: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
    cos_sin_cache_ptr: *const u8,
    rotary_dim: usize,
    is_rotary_interleaved: bool,
) -> OwnedTensor {
    let batch_size = seqused_k.dim(0);
    let num_kv_heads = kv_cache.num_kv_heads;
    let head_dim = kv_cache.head_dim;
    let output_dtype = q.dtype(); // BF16 or F16

    // D2H seqused_k to build cu_seqlens_k on CPU.
    let mut seq_lens = vec![0i32; batch_size];
    ferrite_cuda_core::driver::memcpy_dtoh_async(
        seq_lens.as_mut_ptr() as *mut u8,
        seqused_k.raw_ptr() as *const u8,
        batch_size * 4,
        stream,
    )
    .expect("D2H seqused_k");
    ferrite_cuda_core::driver::stream_synchronize(stream).expect("sync seqused_k");

    // Build cu_seqlens_k prefix sum: [0, s0, s0+s1, ...]
    let mut cu_seqlens_k_host = vec![0i32; batch_size + 1];
    let mut total_kv_tokens: usize = 0;
    for i in 0..batch_size {
        total_kv_tokens += seq_lens[i] as usize;
        cu_seqlens_k_host[i + 1] = total_kv_tokens as i32;
    }

    if total_kv_tokens == 0 {
        // Edge case: no KV tokens (shouldn't happen in practice).
        return kernels::flash_attn_contiguous(
            q,
            q,
            q, // dummy, won't actually run
            cu_seqlens_q,
            cu_seqlens_q,
            max_seqlen_q,
            0,
            scale,
            true,
            softcap,
            window_size_left,
            alloc,
            stream,
            std::ptr::null(),
            0,
            false,
        );
    }

    // Upload cu_seqlens_k to GPU.
    let cu_seqlens_k_gpu = alloc.alloc_tensor(&[batch_size + 1], DType::I32);
    ferrite_cuda_core::driver::memcpy_htod_async(
        cu_seqlens_k_gpu.as_gpu_tensor().raw_ptr() as *mut u8,
        cu_seqlens_k_host.as_ptr() as *const u8,
        (batch_size + 1) * 4,
        stream,
    )
    .expect("H2D cu_seqlens_k");

    // Read K scale and V scale from GPU (single f32 each).
    let mut k_scale_host: f32 = 1.0;
    let mut v_scale_host: f32 = 1.0;
    ferrite_cuda_core::driver::memcpy_dtoh_async(
        &mut k_scale_host as *mut f32 as *mut u8,
        kv_cache.k_scale_ptr(layer_idx) as *const u8,
        4,
        stream,
    )
    .expect("D2H k_scale");
    ferrite_cuda_core::driver::memcpy_dtoh_async(
        &mut v_scale_host as *mut f32 as *mut u8,
        kv_cache.v_scale_ptr(layer_idx) as *const u8,
        4,
        stream,
    )
    .expect("D2H v_scale");
    ferrite_cuda_core::driver::stream_synchronize(stream).expect("sync scales");

    // Dequant+gather K and V from FP8 cache pages to contiguous BF16.
    let k_contiguous = kernels::dequant_gather_pages(
        *kv_cache.k_cache(layer_idx),
        block_table,
        cu_seqlens_k_gpu.as_gpu_tensor(),
        k_scale_host,
        total_kv_tokens,
        num_kv_heads,
        head_dim,
        kv_cache.block_size,
        output_dtype,
        alloc,
        stream,
    );
    let v_contiguous = kernels::dequant_gather_pages(
        *kv_cache.v_cache(layer_idx),
        block_table,
        cu_seqlens_k_gpu.as_gpu_tensor(),
        v_scale_host,
        total_kv_tokens,
        num_kv_heads,
        head_dim,
        kv_cache.block_size,
        output_dtype,
        alloc,
        stream,
    );

    // Contiguous FA2 on the dequantized BF16 K/V.
    kernels::flash_attn_contiguous(
        q,
        k_contiguous.as_gpu_tensor(),
        v_contiguous.as_gpu_tensor(),
        cu_seqlens_q,
        cu_seqlens_k_gpu.as_gpu_tensor(),
        max_seqlen_q,
        max_seqlen_k,
        scale,
        true,
        softcap,
        window_size_left,
        alloc,
        stream,
        cos_sin_cache_ptr,
        rotary_dim,
        is_rotary_interleaved,
    )
}

// ---------------------------------------------------------------------------
// Spans: pre/post rotation helpers
// ---------------------------------------------------------------------------

/// Rotate span blocks in the KV cache before attention, run the attention
/// function, then un-rotate span blocks after. This is a no-op when spans
/// are disabled (both GPU flag pointers are null).
///
/// All model architectures should use this instead of calling attention
/// functions directly when spans may be active.
///
/// `cos_sin_cache`: `[max_pos, rotary_dim]` — the model's rotary cache.
/// `attn_fn`: closure that runs the actual attention (any variant).
#[allow(clippy::too_many_arguments)]
pub unsafe fn with_span_rotation<F>(
    kv_cache: &KvCachePool,
    layer_idx: usize,
    cos_sin_cache: TensorView<'_>,
    cu_seqlens_q: TensorView<'_>,
    seqused_k: TensorView<'_>,
    block_table: TensorView<'_>,
    max_seqlen_k: usize,
    stream: CUstream,
    attn_fn: F,
) -> OwnedTensor
where
    F: FnOnce() -> OwnedTensor,
{
    let unrotated_flags = kv_cache.block_unrotated_gpu();
    let span_flags = kv_cache.block_span_gpu();
    let has_spans = !unrotated_flags.is_null() && !span_flags.is_null();

    if has_spans {
        let batch_size = cu_seqlens_q.dim(0) as usize - 1;
        let max_blocks = block_table.dim(1) as usize;

        // Pre-attention: rotate blocks that are currently unrotated.
        kernels::rotary_paged_k_cache(
            *kv_cache.k_cache(layer_idx),
            *cos_sin_cache,
            *block_table,
            *seqused_k,
            unrotated_flags,
            batch_size,
            max_seqlen_k,
            max_blocks,
            kv_cache.block_size,
            kv_cache.num_kv_heads,
            kv_cache.head_dim,
            false, // forward rotation
            stream,
        );
    }

    let result = attn_fn();

    if has_spans {
        let batch_size = cu_seqlens_q.dim(0) as usize - 1;
        let max_blocks = block_table.dim(1) as usize;

        // Post-attention: un-rotate all span blocks.
        kernels::rotary_paged_k_cache(
            *kv_cache.k_cache(layer_idx),
            *cos_sin_cache,
            *block_table,
            *seqused_k,
            span_flags,
            batch_size,
            max_seqlen_k,
            max_blocks,
            kv_cache.block_size,
            kv_cache.num_kv_heads,
            kv_cache.head_dim,
            true, // inverse rotation
            stream,
        );
    }

    result
}

// ---------------------------------------------------------------------------
// FlashInfer attention
// ---------------------------------------------------------------------------

/// Process-global FlashInfer plan cache.
///
/// Intentionally NOT `thread_local!`: vllm's executor captures the forward
/// pass on one worker thread (where [`flashinfer_attention`] builds the
/// plan) but runs graph replay (and our pre-replay
/// [`replan_fi_for_decode`]) on a different thread. A per-thread cache
/// splits that state and the replan call finds a null handle at replay.
///
/// The plan handle is opaque CUDA state; the FI shim doesn't hold any
/// locks of its own, so a Mutex around the cache is sufficient. Lock
/// contention is minimal — each forward step briefly holds it during
/// (1) build/replan, (2) 16 per-layer set_io+run pairs, and (3) the
/// pre-replay replan.
static FI_PLAN_CACHE: std::sync::Mutex<FlashInferPlanCache> =
    std::sync::Mutex::new(FlashInferPlanCache::new());

/// Round `max_seqlen_k` up to the nearest bucket used by the solver-calibrated
/// FI cost table. Matches `SK_BUCKETS` in ferrite-forward-macro.
///
/// The bucket is used as a plan-cache key: changing sk_bucket tears down the
/// previous plan and rebuilds workspaces. Keeping the mapping monotonic
/// ensures that a decode sweep from sk=128 → 8192 rebuilds at bucket
/// boundaries, not on every token.
pub fn sk_bucket_for(max_seqlen_k: usize) -> u32 {
    const BUCKETS: &[u32] = &[128, 256, 512, 1024, 2048, 4096, 8192];
    for &b in BUCKETS {
        if (max_seqlen_k as u32) <= b {
            return b;
        }
    }
    // Beyond the largest bucket — round up to next power-of-two to keep the
    // plan-cache key stable across similar lengths.
    (max_seqlen_k as u32).next_power_of_two()
}

/// Run one layer of FlashInfer paged attention. Called from
/// `FlashInferAttentionDecodeImpl` / `…PrefillImpl` codegen.
///
/// Expects `cu_seqlens_q.dim(0) == 2` (batch_size=1) — the unified shim is
/// single-sequence-only. When this precondition doesn't hold the caller
/// must use the FA2 fallback instead.
///
/// `cfg` selects the compiled tuple; if the tuple is not in
/// `FLASHINFER_CONFIG_SET` (see `ferrite-cuda-builder`), this returns
/// `None` and the caller must fall back.
///
/// `sk_bucket` is the monotonic ceiling of `max_seqlen_k` — see
/// [`sk_bucket_for`]. Changing it invalidates the cached plan.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flashinfer_attention(
    q: TensorView<'_>,
    _cu_seqlens_q: TensorView<'_>,
    _seqused_k: TensorView<'_>,
    block_table: TensorView<'_>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    scale: f32,
    softcap: f32,
    kv_cache: &KvCachePool,
    layer_idx: usize,
    num_sm: i32,
    cfg: FlashInferConfig,
    sk_bucket: u32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> Option<OwnedTensor> {
    debug_assert_eq!(
        q.ndim(),
        3,
        "FlashInfer expects Q shape [seq_len, num_qo_heads, head_dim]"
    );
    debug_assert_eq!(q.dim(2) as u32, cfg.head_dim, "Q head_dim != cfg.head_dim");

    let seq_len = q.dim(0);
    let num_qo_heads = q.dim(1);
    let num_kv_heads = kv_cache.num_kv_heads;
    let head_dim = kv_cache.head_dim;
    let page_size = kv_cache.block_size;
    let num_pages = max_seqlen_k.div_ceil(page_size);

    if std::env::var("FI_TRACE").is_ok() && layer_idx == 0 {
        eprintln!(
            "[FI] L0 seq_len={} max_k={} scale={} softcap={} num_qo={} num_kv={} hd={} ps={} np={}",
            seq_len,
            max_seqlen_k,
            scale,
            softcap,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            num_pages
        );
    }

    let o = alloc.alloc_tensor(&[seq_len, num_qo_heads, head_dim], q.dtype());
    let (float_ws, int_ws) = flashinfer::workspace_bytes(num_sm, head_dim, num_kv_heads);

    // block_table is [batch_size, max_pages]; batch=1 → row 0 is the kv_indices
    // array. Shim reads exactly `num_pages` entries.
    let kv_indices_ptr = block_table.raw_ptr() as *const i32;
    let k_layer = kv_cache.k_cache(layer_idx);
    let v_layer = kv_cache.v_cache(layer_idx);

    {
        let mut cache = FI_PLAN_CACHE.lock().expect("FI_PLAN_CACHE poisoned");
        let built = unsafe {
            cache
                .ensure(
                    cfg,
                    max_seqlen_q as u32,
                    sk_bucket,
                    q.raw_ptr() as *const core::ffi::c_void,
                    k_layer.raw_ptr() as *const core::ffi::c_void,
                    v_layer.raw_ptr() as *const core::ffi::c_void,
                    kv_indices_ptr,
                    o.as_gpu_tensor().raw_ptr() as *mut core::ffi::c_void,
                    max_seqlen_k as i32,
                    num_qo_heads as i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    page_size as i32,
                    num_pages as i32,
                    num_sm,
                    float_ws,
                    int_ws,
                    scale,
                    softcap,
                    stream,
                )
                .is_some()
        };
        if !built {
            tracing::error!(
                ?cfg,
                sk_bucket,
                "FlashInfer plan build failed — falling back"
            );
            return None;
        }
        // Re-run the scheduler with the current step's (seqlen_k, num_pages).
        // The captured memcpy inside TwoStageHolisticPlanWithNumSm reads from
        // pinned int_ws_h — at graph replay, the pre-launch `replan_fi_for_decode`
        // updates int_ws_h with that step's scheduling before the captured
        // memcpy re-executes, so int_ws_d ends up with fresh data.
        unsafe {
            let rc = cache.replan(
                seq_len as i32,
                max_seqlen_k as i32,
                num_pages as i32,
                stream,
            );
            if rc != 0 {
                tracing::error!(rc, "fi_replan returned nonzero");
            }
        }

        unsafe {
            cache.set_io(
                q.raw_ptr() as *const core::ffi::c_void,
                k_layer.raw_ptr() as *const core::ffi::c_void,
                v_layer.raw_ptr() as *const core::ffi::c_void,
                kv_indices_ptr,
                o.as_gpu_tensor().raw_ptr() as *mut core::ffi::c_void,
            );
            let rc = cache.run(stream);
            if rc != 0 {
                tracing::error!(rc, "fi_run returned nonzero");
            }
        }
    }
    Some(o)
}

/// Re-plan the thread-local FI plan cache for the current decode step.
/// The H2D memcpy issued here lands on `stream` and completes before any
/// subsequent kernel launch on the same stream.
///
/// No-op when no FI plan has been built (models without `sk_buckets`).
///
/// # Safety
/// `stream` must be the compute stream used by the graph runner.
pub unsafe fn replan_fi_for_decode(
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    page_size: usize,
    stream: CUstream,
) {
    let num_pages = max_seqlen_k.div_ceil(page_size) as i32;
    let mut cache = FI_PLAN_CACHE.lock().expect("FI_PLAN_CACHE poisoned");
    unsafe {
        cache.replan(max_seqlen_q as i32, max_seqlen_k as i32, num_pages, stream);
    }
}

/// Drop the global FlashInfer plan cache. Call on worker teardown to
/// free the planner's `cudaMalloc`ed workspaces deterministically.
pub fn reset_fi_plan_cache() {
    FI_PLAN_CACHE
        .lock()
        .expect("FI_PLAN_CACHE poisoned")
        .clear();
}
