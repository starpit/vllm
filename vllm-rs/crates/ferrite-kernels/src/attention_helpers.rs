// SPDX-License-Identifier: Apache-2.0
//! Shared attention helpers for FP8 KV cache support.
//!
//! All model attention modules call these instead of directly invoking
//! reshape_and_cache/flash_attn to get FP8 quantize-on-write and
//! dequant-on-read behavior transparently.
//!
//! During CUDA graph capture/replay, the [`Fp8GraphCtx`] thread-local provides
//! pre-allocated dequant buffers and cached scales, eliminating the D2H syncs
//! and variable-size allocations that would otherwise break graph capture.

use std::cell::Cell;

use crate::kernels;
use crate::kv_cache::KvCachePool;
use ferrite_cuda_core::alloc::{CachingAllocator, OwnedTensor};
use ferrite_cuda_core::dtype::DType;
use ferrite_cuda_core::tensor::{GpuTensor, TensorView};

type CUstream = cudarc::driver::sys::CUstream;

// ---------------------------------------------------------------------------
// FP8 CUDA graph context (thread-local)
// ---------------------------------------------------------------------------

/// Pre-allocated buffers and cached scales for FP8 decode during graph
/// capture/replay. Set via [`set_fp8_graph_ctx`] before capture, cleared after.
///
/// All pointer fields are persistent GPU allocations owned by [`CudaGraphRunner`].
#[derive(Clone, Copy)]
pub struct Fp8GraphCtx {
    /// `[max_total_kv, num_kv_heads, head_dim]` in model dtype (BF16/F16).
    pub k_buf: *mut u8,
    /// Same shape as `k_buf`.
    pub v_buf: *mut u8,
    /// `[max_batch + 1]` i32 — prefix-sum of per-sequence KV lengths.
    pub cu_seqlens_k: *mut u8,
    /// Grid launch size for dequant kernels (= buffer capacity in tokens).
    pub max_total_kv: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub output_dtype: DType,
    /// Host-side per-layer K scales (len = num_layers). Pointer into a Vec
    /// owned by CudaGraphRunner — valid for the duration of the graph ctx.
    pub k_scales: *const f32,
    /// Host-side per-layer V scales.
    pub v_scales: *const f32,
}

// Raw pointers are Send — the buffers are GPU-side and only accessed on the
// thread that owns the CUDA context.
unsafe impl Send for Fp8GraphCtx {}

thread_local! {
    static FP8_GRAPH_CTX: Cell<Option<Fp8GraphCtx>> = const { Cell::new(None) };
}

/// Set the FP8 graph context for the current thread. Call before graph capture.
pub fn set_fp8_graph_ctx(ctx: Fp8GraphCtx) {
    FP8_GRAPH_CTX.set(Some(ctx));
}

/// Clear the FP8 graph context. Call after graph capture completes.
pub fn clear_fp8_graph_ctx() {
    FP8_GRAPH_CTX.set(None);
}

/// Returns `true` if the FP8 graph context is active on this thread.
pub fn has_fp8_graph_ctx() -> bool {
    FP8_GRAPH_CTX.get().is_some()
}

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
///
/// Two modes:
/// 1. **Normal** (no graph ctx): D2H seqused_k + scales, allocate dequant buffers,
///    run dequant+FA2. Used during eager decode.
/// 2. **Graphed** (thread-local [`Fp8GraphCtx`] set): uses pre-allocated buffers
///    and cached scales — no D2H syncs, no dynamic allocations. Used during CUDA
///    graph capture.
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
    // Check for pre-allocated graph context.
    if let Some(ctx) = FP8_GRAPH_CTX.get() {
        return fp8_decode_attention_graphed(
            q,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            softcap,
            window_size_left,
            kv_cache,
            layer_idx,
            ctx,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
        );
    }

    // --- Normal (eager) path ---
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

/// FP8 decode path for CUDA graph capture: uses pre-allocated buffers and cached
/// scales. No D2H syncs, no dynamic allocations — all addresses are fixed.
///
/// The dequant kernel is launched with `ctx.max_total_kv` grid blocks. Blocks
/// beyond the actual total (determined by `cu_seqlens_k`) exit early via the
/// kernel's bounds check (`seq_idx >= batch_size`).
#[allow(clippy::too_many_arguments)]
unsafe fn fp8_decode_attention_graphed(
    q: GpuTensor,
    cu_seqlens_q: GpuTensor,
    _seqused_k: GpuTensor,
    block_table: GpuTensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    scale: f32,
    softcap: f32,
    window_size_left: i32,
    kv_cache: &KvCachePool,
    layer_idx: usize,
    ctx: Fp8GraphCtx,
    alloc: &mut CachingAllocator,
    stream: CUstream,
    cos_sin_cache_ptr: *const u8,
    rotary_dim: usize,
    is_rotary_interleaved: bool,
) -> OwnedTensor {
    let batch_size = cu_seqlens_q.dim(0) - 1;

    // Use cached host-side scales (no D2H sync needed).
    let k_scale = *ctx.k_scales.add(layer_idx);
    let v_scale = *ctx.v_scales.add(layer_idx);

    // cu_seqlens_k lives in the graph runner's persistent buffer.
    // During capture: filled with dummy prefix sums by fill_dummy_decode_fp8.
    // During replay: updated by cuda_worker before graph launch.
    let cu_seqlens_k_gpu = GpuTensor::new(ctx.cu_seqlens_k, &[batch_size + 1], DType::I32);

    // Dequant into pre-allocated buffers with over-sized grid.
    // Blocks beyond actual total_kv_tokens exit early (kernel bounds check).
    kernels::dequant_gather_pages_into(
        *kv_cache.k_cache(layer_idx),
        block_table,
        cu_seqlens_k_gpu,
        k_scale,
        ctx.max_total_kv,
        ctx.num_kv_heads,
        ctx.head_dim,
        kv_cache.block_size,
        ctx.output_dtype,
        ctx.k_buf,
        stream,
    );
    kernels::dequant_gather_pages_into(
        *kv_cache.v_cache(layer_idx),
        block_table,
        cu_seqlens_k_gpu,
        v_scale,
        ctx.max_total_kv,
        ctx.num_kv_heads,
        ctx.head_dim,
        kv_cache.block_size,
        ctx.output_dtype,
        ctx.v_buf,
        stream,
    );

    // Create tensor views into the pre-allocated buffers.
    let k_contiguous = GpuTensor::new(
        ctx.k_buf,
        &[ctx.max_total_kv, ctx.num_kv_heads, ctx.head_dim],
        ctx.output_dtype,
    );
    let v_contiguous = GpuTensor::new(
        ctx.v_buf,
        &[ctx.max_total_kv, ctx.num_kv_heads, ctx.head_dim],
        ctx.output_dtype,
    );

    // Contiguous FA2 on the dequantized K/V.
    kernels::flash_attn_contiguous(
        q,
        k_contiguous,
        v_contiguous,
        cu_seqlens_q,
        cu_seqlens_k_gpu,
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
