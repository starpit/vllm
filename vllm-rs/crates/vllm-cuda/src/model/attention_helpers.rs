// SPDX-License-Identifier: Apache-2.0
//! Shared attention helpers for FP8 KV cache support.
//!
//! All model attention modules call these instead of directly invoking
//! reshape_and_cache/flash_attn to get FP8 quantize-on-write and
//! dequant-on-read behavior transparently.

use crate::alloc::{CachingAllocator, OwnedTensor};
use crate::dtype::DType;
use crate::kernels;
use crate::kv_cache::KvCachePool;
use crate::tensor::GpuTensor;

type CUstream = cudarc::driver::sys::CUstream;

/// Write K/V into the paged cache, handling both BF16 and FP8 paths.
///
/// When the cache is FP8, quantizes BF16/F16 K/V using per-layer scales.
/// When BF16/F16, performs a direct copy (existing path).
#[allow(clippy::too_many_arguments)]
pub unsafe fn write_kv_cache(
    k: GpuTensor,
    v: GpuTensor,
    slot_mapping: GpuTensor,
    kv_cache: &KvCachePool,
    layer_idx: usize,
    stream: CUstream,
) {
    if kv_cache.is_fp8() {
        kernels::reshape_and_cache_fp8(
            k,
            v,
            kv_cache.k_cache(layer_idx),
            kv_cache.v_cache(layer_idx),
            slot_mapping,
            kv_cache.k_scale_ptr(layer_idx),
            kv_cache.v_scale_ptr(layer_idx),
            kv_cache.block_size,
            stream,
        );
    } else {
        kernels::reshape_and_cache(
            k,
            v,
            kv_cache.k_cache(layer_idx),
            kv_cache.v_cache(layer_idx),
            slot_mapping,
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
pub unsafe fn attention_standard(
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    cu_seqlens_q: GpuTensor,
    seqused_k: GpuTensor,
    block_table: GpuTensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    scale: f32,
    kv_cache: &KvCachePool,
    layer_idx: usize,
    num_sm: i32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let fresh_prefill = max_seqlen_q > 1 && max_seqlen_q == max_seqlen_k;

    if fresh_prefill {
        // Fresh prefill: use BF16 K/V from QKV projection (no cache read).
        kernels::flash_attn_contiguous(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            true,
            0.0,
            -1,
            alloc,
            stream,
        )
    } else if kv_cache.is_fp8() {
        // FP8 decode: dequant pages → contiguous FA2.
        fp8_decode_attention(
            q,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            0.0,
            -1,
            kv_cache,
            layer_idx,
            alloc,
            stream,
        )
    } else {
        // BF16 decode: paged FA2 directly on cache.
        kernels::flash_attn_paged(
            q,
            kv_cache.k_cache(layer_idx),
            kv_cache.v_cache(layer_idx),
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            true,
            kv_cache.block_size,
            num_sm,
            alloc,
            stream,
        )
    }
}

/// Run attention with softcap and/or sliding window (Gemma2/Gemma3).
///
/// Uses `flash_attn_paged_ext` for the BF16 decode path.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_ext(
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
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
    num_sm: i32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let fresh_prefill = max_seqlen_q > 1 && max_seqlen_q == max_seqlen_k;

    if fresh_prefill {
        kernels::flash_attn_contiguous(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            true,
            softcap,
            window_size_left,
            alloc,
            stream,
        )
    } else if kv_cache.is_fp8() {
        fp8_decode_attention(
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
            alloc,
            stream,
        )
    } else {
        kernels::flash_attn_paged_ext(
            q,
            kv_cache.k_cache(layer_idx),
            kv_cache.v_cache(layer_idx),
            cu_seqlens_q,
            seqused_k,
            block_table,
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
        )
    }
}

/// FP8 decode path: dequant KV pages → contiguous BF16 → FA2.
///
/// Builds cu_seqlens_k from seqused_k on CPU (small — batch sizes ≤ 512),
/// uploads to GPU, then runs dequant_gather_pages for K and V, followed
/// by contiguous flash attention.
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
) -> OwnedTensor {
    let batch_size = seqused_k.dim(0);
    let num_kv_heads = kv_cache.num_kv_heads;
    let head_dim = kv_cache.head_dim;
    let output_dtype = q.dtype(); // BF16 or F16

    // D2H seqused_k to build cu_seqlens_k on CPU.
    let mut seq_lens = vec![0i32; batch_size];
    crate::driver::memcpy_dtoh_async(
        seq_lens.as_mut_ptr() as *mut u8,
        seqused_k.raw_ptr() as *const u8,
        batch_size * 4,
        stream,
    )
    .expect("D2H seqused_k");
    crate::driver::stream_synchronize(stream).expect("sync seqused_k");

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
        );
    }

    // Upload cu_seqlens_k to GPU.
    let cu_seqlens_k_gpu = alloc.alloc_tensor(&[batch_size + 1], DType::I32);
    crate::driver::memcpy_htod_async(
        cu_seqlens_k_gpu.as_gpu_tensor().raw_ptr() as *mut u8,
        cu_seqlens_k_host.as_ptr() as *const u8,
        (batch_size + 1) * 4,
        stream,
    )
    .expect("H2D cu_seqlens_k");

    // Read K scale and V scale from GPU (single f32 each).
    let mut k_scale_host: f32 = 1.0;
    let mut v_scale_host: f32 = 1.0;
    crate::driver::memcpy_dtoh_async(
        &mut k_scale_host as *mut f32 as *mut u8,
        kv_cache.k_scale_ptr(layer_idx) as *const u8,
        4,
        stream,
    )
    .expect("D2H k_scale");
    crate::driver::memcpy_dtoh_async(
        &mut v_scale_host as *mut f32 as *mut u8,
        kv_cache.v_scale_ptr(layer_idx) as *const u8,
        4,
        stream,
    )
    .expect("D2H v_scale");
    crate::driver::stream_synchronize(stream).expect("sync scales");

    // Dequant+gather K and V from FP8 cache pages to contiguous BF16.
    let k_contiguous = kernels::dequant_gather_pages(
        kv_cache.k_cache(layer_idx),
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
        kv_cache.v_cache(layer_idx),
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
    )
}
