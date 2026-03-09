// SPDX-License-Identifier: Apache-2.0
//! Direct kernel dispatch for `GpuTensor` — no candle dependency.
//!
//! These wrap the same CUDA FFI functions from `vllm-kernels/csrc/` but
//! dispatch from `GpuTensor::as_ptr()` instead of extracting raw pointers
//! from candle `Tensor` (which takes ~10 lines per tensor). Here it's one line.

use core::ffi::{c_int, c_void};

use crate::alloc::{CachingAllocator, OwnedTensor};
use crate::dtype::DType;
use crate::tensor::GpuTensor;

// ---------------------------------------------------------------------------
// FFI declarations (same C symbols as vllm-kernels, linked from libvllm_cuda.a)
// ---------------------------------------------------------------------------

type CUstream = cudarc::driver::sys::CUstream;

unsafe extern "C" {
    // RMS norm
    fn rms_norm_f16(
        out: *mut u16,
        input: *const u16,
        weight: *const u16,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );
    fn rms_norm_bf16(
        out: *mut u16,
        input: *const u16,
        weight: *const u16,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );
    fn rms_norm_f32(
        out: *mut f32,
        input: *const f32,
        weight: *const f32,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );

    // Fused add + RMS norm (in-place: residual += input, then norm)
    fn fused_add_rms_norm_f16(
        input: *mut u16,
        residual: *mut u16,
        weight: *const u16,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );
    fn fused_add_rms_norm_bf16(
        input: *mut u16,
        residual: *mut u16,
        weight: *const u16,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );
    fn fused_add_rms_norm_f32(
        input: *mut f32,
        residual: *mut f32,
        weight: *const f32,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );

    // Fused SiLU(gate) * up from combined [num_tokens, 2*d]
    fn silu_and_mul_fused_f16(
        out: *mut u16,
        gate_up: *const u16,
        num_tokens: i32,
        d: i32,
        stream: CUstream,
    );
    fn silu_and_mul_fused_bf16(
        out: *mut u16,
        gate_up: *const u16,
        num_tokens: i32,
        d: i32,
        stream: CUstream,
    );
    fn silu_and_mul_fused_f32(
        out: *mut f32,
        gate_up: *const f32,
        num_tokens: i32,
        d: i32,
        stream: CUstream,
    );

    // Fused GELU(tanh)(gate) * up from combined [num_tokens, 2*d]
    fn gelu_and_mul_fused_f16(
        out: *mut u16,
        gate_up: *const u16,
        num_tokens: i32,
        d: i32,
        stream: CUstream,
    );
    fn gelu_and_mul_fused_bf16(
        out: *mut u16,
        gate_up: *const u16,
        num_tokens: i32,
        d: i32,
        stream: CUstream,
    );
    fn gelu_and_mul_fused_f32(
        out: *mut f32,
        gate_up: *const f32,
        num_tokens: i32,
        d: i32,
        stream: CUstream,
    );

    // Rotary embedding (in-place on q and k)
    fn rotary_embedding_f16(
        positions: *const u32,
        query: *mut u16,
        key: *mut u16,
        cos_sin_cache: *const u16,
        rotary_dim: i32,
        total_q_dim: i32,
        total_k_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn rotary_embedding_bf16(
        positions: *const u32,
        query: *mut u16,
        key: *mut u16,
        cos_sin_cache: *const u16,
        rotary_dim: i32,
        total_q_dim: i32,
        total_k_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn rotary_embedding_f32(
        positions: *const u32,
        query: *mut f32,
        key: *mut f32,
        cos_sin_cache: *const f32,
        rotary_dim: i32,
        total_q_dim: i32,
        total_k_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );

    // Embedding gather
    fn embedding_gather_f16(
        out: *mut u16,
        weight: *const u16,
        ids: *const u32,
        hidden_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn embedding_gather_bf16(
        out: *mut u16,
        weight: *const u16,
        ids: *const u32,
        hidden_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn embedding_gather_f32(
        out: *mut f32,
        weight: *const f32,
        ids: *const u32,
        hidden_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );

    // Update decode metadata in-place on GPU
    fn update_decode_metadata(
        positions: *mut u32,
        slot_mapping: *mut i64,
        seqused_k: *mut i32,
        block_table: *const i32,
        num_reqs: c_int,
        block_size: c_int,
        max_blocks_per_seq: c_int,
        stream: CUstream,
    );

    // Split fused QKV
    fn split_qkv_f16(
        q: *mut u16,
        k: *mut u16,
        v: *mut u16,
        qkv: *const u16,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn split_qkv_bf16(
        q: *mut u16,
        k: *mut u16,
        v: *mut u16,
        qkv: *const u16,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn split_qkv_f32(
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        qkv: *const f32,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        num_tokens: i32,
        stream: CUstream,
    );

    // Fused QKV split + RoPE (replaces split_qkv + rotary_embedding)
    fn fused_qkv_rope_f16(
        q: *mut u16,
        k: *mut u16,
        v: *mut u16,
        qkv: *const u16,
        positions: *const u32,
        cos_sin_cache: *const u16,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn fused_qkv_rope_bf16(
        q: *mut u16,
        k: *mut u16,
        v: *mut u16,
        qkv: *const u16,
        positions: *const u32,
        cos_sin_cache: *const u16,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn fused_qkv_rope_f32(
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        qkv: *const f32,
        positions: *const u32,
        cos_sin_cache: *const f32,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );
}

// ---------------------------------------------------------------------------
// RMS Norm
// ---------------------------------------------------------------------------

/// RMS normalization: `out = input / rms(input) * weight`
///
/// * `input`: `[num_tokens, hidden_size]`
/// * `weight`: `[hidden_size]`
/// * Returns: `[num_tokens, hidden_size]` allocated from arena.
pub unsafe fn rms_norm(
    input: GpuTensor,
    weight: GpuTensor,
    eps: f32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = input.dim(0) as i32;
    let hidden_size = input.dim(1) as i32;
    let out = alloc.alloc_tensor(&[num_tokens as usize, hidden_size as usize], input.dtype());

    match input.dtype() {
        DType::F16 => rms_norm_f16(
            out.as_mut_ptr(),
            input.as_ptr(),
            weight.as_ptr(),
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        DType::BF16 => rms_norm_bf16(
            out.as_mut_ptr(),
            input.as_ptr(),
            weight.as_ptr(),
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        DType::F32 => rms_norm_f32(
            out.as_mut_ptr(),
            input.as_ptr(),
            weight.as_ptr(),
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        _ => panic!("rms_norm: unsupported dtype {:?}", input.dtype()),
    }
    out
}

// ---------------------------------------------------------------------------
// Fused Add + RMS Norm
// ---------------------------------------------------------------------------

/// Fused add + RMS norm: `residual += input; normed = rms_norm(residual) * weight`
///
/// The CUDA kernel writes normed output into `input` (in-place) and updates
/// `residual` in-place. We allocate a fresh output from the arena and copy
/// the input there first so the caller's input isn't corrupted.
///
/// Actually — the vllm-kernels fused_add_rms_norm writes:
///   residual[i] += input[i]       (in-place on residual)
///   input[i] = norm(residual[i])  (in-place on input, which becomes the output)
///
/// So `input` becomes the normed output and `residual` is the updated residual.
/// Both are mutated in-place.
///
/// For arena-based flow: we allocate a copy of `input` for the normed output,
/// and the caller must own the residual buffer.
///
/// * `input`: `[num_tokens, hidden_size]` — will contain normed output after call
/// * `residual`: `[num_tokens, hidden_size]` — will be updated in-place (res += input)
/// * `weight`: `[hidden_size]`
/// * `eps`: normalization epsilon
///
/// Returns: `(normed, residual)` — normed is the same buffer as `input` (now overwritten),
/// residual is the same buffer (updated in-place).
pub unsafe fn fused_add_rms_norm_inplace(
    input: GpuTensor,
    residual: GpuTensor,
    weight: GpuTensor,
    eps: f32,
    stream: CUstream,
) -> (GpuTensor, GpuTensor) {
    let num_tokens = input.dim(0) as i32;
    let hidden_size = input.dim(1) as i32;

    match input.dtype() {
        DType::F16 => fused_add_rms_norm_f16(
            input.as_mut_ptr(),
            residual.as_mut_ptr(),
            weight.as_ptr(),
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        DType::BF16 => fused_add_rms_norm_bf16(
            input.as_mut_ptr(),
            residual.as_mut_ptr(),
            weight.as_ptr(),
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        DType::F32 => fused_add_rms_norm_f32(
            input.as_mut_ptr(),
            residual.as_mut_ptr(),
            weight.as_ptr(),
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        _ => panic!("fused_add_rms_norm: unsupported dtype {:?}", input.dtype()),
    }

    // input now contains normed output, residual is updated.
    (input, residual)
}

/// Fused add + RMS norm with arena-allocated output.
///
/// Copies `input` to an arena buffer first (so the original `input` is not mutated),
/// then runs the in-place fused kernel.
///
/// Returns `(normed, residual)` where normed is arena-allocated.
pub unsafe fn fused_add_rms_norm(
    input: GpuTensor,
    residual: GpuTensor,
    weight: GpuTensor,
    eps: f32,
    alloc: &mut CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> (GpuTensor, GpuTensor) {
    // Allocate a copy of input for the normed output.
    let normed_buf = alloc.alloc_tensor(&[input.dim(0), input.dim(1)], input.dtype());
    crate::driver::memcpy_dtod_async(
        normed_buf.as_gpu_tensor().raw_ptr(),
        input.raw_ptr() as *const u8,
        input.size_bytes(),
        stream,
    )
    .expect("fused_add_rms_norm: D2D copy failed");

    fused_add_rms_norm_inplace(normed_buf.into_gpu_tensor(), residual, weight, eps, stream)
}

// ---------------------------------------------------------------------------
// Fused SiLU-and-Mul
// ---------------------------------------------------------------------------

/// Fused SiLU(gate) * up from combined gate_up tensor.
///
/// * `gate_up`: `[num_tokens, 2 * intermediate_size]`
/// * `intermediate_size`: the "d" dimension
/// * Returns: `[num_tokens, intermediate_size]` from arena.
pub unsafe fn silu_and_mul_fused(
    gate_up: GpuTensor,
    intermediate_size: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = gate_up.dim(0) as i32;
    let d = intermediate_size as i32;
    let out = alloc.alloc_tensor(&[num_tokens as usize, intermediate_size], gate_up.dtype());

    match gate_up.dtype() {
        DType::F16 => {
            silu_and_mul_fused_f16(out.as_mut_ptr(), gate_up.as_ptr(), num_tokens, d, stream)
        }
        DType::BF16 => {
            silu_and_mul_fused_bf16(out.as_mut_ptr(), gate_up.as_ptr(), num_tokens, d, stream)
        }
        DType::F32 => {
            silu_and_mul_fused_f32(out.as_mut_ptr(), gate_up.as_ptr(), num_tokens, d, stream)
        }
        _ => panic!(
            "silu_and_mul_fused: unsupported dtype {:?}",
            gate_up.dtype()
        ),
    }
    out
}

// ---------------------------------------------------------------------------
// Fused GELU-and-Mul
// ---------------------------------------------------------------------------

/// Fused GELU(tanh)(gate) * up from combined gate_up tensor.
///
/// * `gate_up`: `[num_tokens, 2 * intermediate_size]`
/// * `intermediate_size`: the "d" dimension
/// * Returns: `[num_tokens, intermediate_size]` from arena.
pub unsafe fn gelu_and_mul_fused(
    gate_up: GpuTensor,
    intermediate_size: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = gate_up.dim(0) as i32;
    let d = intermediate_size as i32;
    let out = alloc.alloc_tensor(&[num_tokens as usize, intermediate_size], gate_up.dtype());

    match gate_up.dtype() {
        DType::F16 => {
            gelu_and_mul_fused_f16(out.as_mut_ptr(), gate_up.as_ptr(), num_tokens, d, stream)
        }
        DType::BF16 => {
            gelu_and_mul_fused_bf16(out.as_mut_ptr(), gate_up.as_ptr(), num_tokens, d, stream)
        }
        DType::F32 => {
            gelu_and_mul_fused_f32(out.as_mut_ptr(), gate_up.as_ptr(), num_tokens, d, stream)
        }
        _ => panic!(
            "gelu_and_mul_fused: unsupported dtype {:?}",
            gate_up.dtype()
        ),
    }
    out
}

// ---------------------------------------------------------------------------
// Rotary Embedding
// ---------------------------------------------------------------------------

/// Fused rotary embedding applied in-place to Q and K.
///
/// * `q`: `[num_tokens, num_q_heads * head_dim]` — mutated in-place
/// * `k`: `[num_tokens, num_kv_heads * head_dim]` — mutated in-place
/// * `positions`: `[num_tokens]` (U32)
/// * `cos_sin_cache`: `[max_pos, rotary_dim]`
/// * `head_dim`: dimension per head
pub unsafe fn rotary_embedding_inplace(
    q: GpuTensor,
    k: GpuTensor,
    positions: GpuTensor,
    cos_sin_cache: GpuTensor,
    head_dim: usize,
    stream: CUstream,
) {
    let num_tokens = q.dim(0) as i32;
    let total_q_dim = q.dim(1) as i32;
    let total_k_dim = k.dim(1) as i32;
    let rotary_dim = cos_sin_cache.dim(1) as i32;
    let head_size = head_dim as i32;

    match q.dtype() {
        DType::F16 => rotary_embedding_f16(
            positions.as_ptr(),
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            cos_sin_cache.as_ptr(),
            rotary_dim,
            total_q_dim,
            total_k_dim,
            head_size,
            num_tokens,
            stream,
        ),
        DType::BF16 => rotary_embedding_bf16(
            positions.as_ptr(),
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            cos_sin_cache.as_ptr(),
            rotary_dim,
            total_q_dim,
            total_k_dim,
            head_size,
            num_tokens,
            stream,
        ),
        DType::F32 => rotary_embedding_f32(
            positions.as_ptr(),
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            cos_sin_cache.as_ptr(),
            rotary_dim,
            total_q_dim,
            total_k_dim,
            head_size,
            num_tokens,
            stream,
        ),
        _ => panic!("rotary_embedding: unsupported dtype {:?}", q.dtype()),
    }
}

// ---------------------------------------------------------------------------
// Embedding Gather
// ---------------------------------------------------------------------------

/// Embedding gather: out[i] = weight[input_ids[i]]
///
/// * `weight`: `[vocab_size, hidden_size]`
/// * `input_ids`: `[num_tokens]` (U32)
/// * Returns: `[num_tokens, hidden_size]` from arena.
pub unsafe fn embedding_gather(
    weight: GpuTensor,
    input_ids: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = input_ids.dim(0);
    let hidden_size = weight.dim(1);
    let out = alloc.alloc_tensor(&[num_tokens, hidden_size], weight.dtype());

    match weight.dtype() {
        DType::F16 => embedding_gather_f16(
            out.as_mut_ptr(),
            weight.as_ptr(),
            input_ids.as_ptr(),
            hidden_size as i32,
            num_tokens as i32,
            stream,
        ),
        DType::BF16 => embedding_gather_bf16(
            out.as_mut_ptr(),
            weight.as_ptr(),
            input_ids.as_ptr(),
            hidden_size as i32,
            num_tokens as i32,
            stream,
        ),
        DType::F32 => embedding_gather_f32(
            out.as_mut_ptr(),
            weight.as_ptr(),
            input_ids.as_ptr(),
            hidden_size as i32,
            num_tokens as i32,
            stream,
        ),
        _ => panic!("embedding_gather: unsupported dtype {:?}", weight.dtype()),
    }
    out
}

// ---------------------------------------------------------------------------
// Update decode metadata on GPU (persistent buffers)
// ---------------------------------------------------------------------------

/// Increment positions, recompute slot_mapping from block_table, and
/// increment seqused_k — all in one kernel launch on the GPU.
///
/// This replaces 3 CPU Vec builds + 3 H2D copies per decode step.
///
/// # Safety
/// All pointers must be valid GPU memory. `positions` and `slot_mapping`
/// must have at least `num_reqs` elements. `seqused_k` must have
/// `num_reqs` elements. `block_table` must be `[num_reqs, max_blocks_per_seq]`.
pub unsafe fn update_decode_metadata_gpu(
    positions: *mut u8,
    slot_mapping: *mut u8,
    seqused_k: *mut u8,
    block_table: *const u8,
    num_reqs: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
    stream: CUstream,
) {
    update_decode_metadata(
        positions as *mut u32,
        slot_mapping as *mut i64,
        seqused_k as *mut i32,
        block_table as *const i32,
        num_reqs as c_int,
        block_size as c_int,
        max_blocks_per_seq as c_int,
        stream,
    );
}

// ---------------------------------------------------------------------------
// Reshape and Cache (write new K/V tokens into paged KV cache)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn reshape_and_cache_f16(
        key: *const u16,
        value: *const u16,
        key_cache: *mut u16,
        value_cache: *mut u16,
        slot_mapping: *const i64,
        num_tokens: i32,
        num_heads: i32,
        head_dim: i32,
        block_size: i32,
        stream: CUstream,
    );
    fn reshape_and_cache_bf16(
        key: *const u16,
        value: *const u16,
        key_cache: *mut u16,
        value_cache: *mut u16,
        slot_mapping: *const i64,
        num_tokens: i32,
        num_heads: i32,
        head_dim: i32,
        block_size: i32,
        stream: CUstream,
    );
    fn reshape_and_cache_f32(
        key: *const f32,
        value: *const f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        slot_mapping: *const i64,
        num_tokens: i32,
        num_heads: i32,
        head_dim: i32,
        block_size: i32,
        stream: CUstream,
    );
}

/// Write new K/V tokens into the paged KV cache at the given slot positions.
///
/// * `key`: `[num_tokens, num_kv_heads, head_dim]`
/// * `value`: `[num_tokens, num_kv_heads, head_dim]`
/// * `key_cache`: `[num_blocks, block_size, num_kv_heads, head_dim]`
/// * `value_cache`: same layout
/// * `slot_mapping`: `[num_tokens]` (I64) — absolute slot index for each token
pub unsafe fn reshape_and_cache(
    key: GpuTensor,
    value: GpuTensor,
    key_cache: GpuTensor,
    value_cache: GpuTensor,
    slot_mapping: GpuTensor,
    block_size: usize,
    stream: CUstream,
) {
    let num_tokens = key.dim(0) as i32;
    let num_heads = key.dim(1) as i32;
    let head_dim = key.dim(2) as i32;
    let bs = block_size as i32;

    match key.dtype() {
        DType::F16 => reshape_and_cache_f16(
            key.as_ptr(),
            value.as_ptr(),
            key_cache.as_mut_ptr(),
            value_cache.as_mut_ptr(),
            slot_mapping.as_ptr(),
            num_tokens,
            num_heads,
            head_dim,
            bs,
            stream,
        ),
        DType::BF16 => reshape_and_cache_bf16(
            key.as_ptr(),
            value.as_ptr(),
            key_cache.as_mut_ptr(),
            value_cache.as_mut_ptr(),
            slot_mapping.as_ptr(),
            num_tokens,
            num_heads,
            head_dim,
            bs,
            stream,
        ),
        DType::F32 => reshape_and_cache_f32(
            key.as_ptr(),
            value.as_ptr(),
            key_cache.as_mut_ptr(),
            value_cache.as_mut_ptr(),
            slot_mapping.as_ptr(),
            num_tokens,
            num_heads,
            head_dim,
            bs,
            stream,
        ),
        _ => panic!("reshape_and_cache: unsupported dtype {:?}", key.dtype()),
    }
}

// ---------------------------------------------------------------------------
// FlashAttention-2 Paged (raw FFI — no candle dependency)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// FFI to mha_varlen_fwd in flash-attn-shim/ffi_shim.cu.
    /// Mirrors upstream vllm-project/flash-attention mha_varlen_fwd() logic.
    /// Handles both paged and non-paged KV, forces splitkv kernel for paged.
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
        stream: CUstream,
    );
}

fn round_multiple(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

/// Non-paged FlashAttention-2 forward pass (contiguous K/V).
///
/// Used for **prefill** where K/V come directly from the current forward pass
/// (not from the paged block cache). Matches Python vLLM's prefill attention.
///
/// * `q`: `[total_q_tokens, num_heads, head_dim]`
/// * `k`: `[total_k_tokens, num_kv_heads, head_dim]`
/// * `v`: `[total_k_tokens, num_kv_heads, head_dim]`
/// * `cu_seqlens_q/k`: `[batch_size + 1]` cumulative sequence lengths
///
/// Returns: `[total_q_tokens, num_heads, head_dim]` output tensor from arena.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_contiguous(
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    cu_seqlens_q: GpuTensor,
    cu_seqlens_k: GpuTensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    is_causal: bool,
    softcap: f32,
    window_size_left: i32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let total_q = q.dim(0);
    let num_heads = q.dim(1);
    let head_dim = q.dim(2);
    let num_kv_heads = k.dim(1);

    let batch_size = cu_seqlens_q.dim(0) - 1;

    let _head_size_rounded = round_multiple(head_dim, 32);
    let _seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    let _seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    // Allocate output and softmax_lse from arena.
    let out = alloc.alloc_tensor(&[total_q, num_heads, head_dim], q.dtype());
    let softmax_lse = alloc.alloc_tensor(&[num_heads * total_q], DType::F32);

    // Use mha_varlen_fwd with null block_table for contiguous (non-paged) path
    mha_varlen_fwd(
        q.raw_ptr() as *mut c_void,
        k.raw_ptr() as *mut c_void,
        v.raw_ptr() as *mut c_void,
        out.raw_ptr() as *mut c_void,
        softmax_lse.raw_ptr() as *mut c_void,
        cu_seqlens_q.as_ptr::<i32>() as *const i32,
        cu_seqlens_k.as_ptr::<i32>() as *const i32,
        std::ptr::null(), // seqused_k = null for contiguous
        std::ptr::null(), // block_table = null for contiguous
        0,                // block_table_batch_stride
        batch_size as i32,
        max_seqlen_q as i32,
        max_seqlen_k as i32,
        num_heads as i32,
        num_kv_heads as i32,
        head_dim as i32,
        0,                                // page_block_size (unused for contiguous)
        (num_heads * head_dim) as i64,    // q_row_stride
        head_dim as i64,                  // q_head_stride
        0,                                // k_batch_stride
        (num_kv_heads * head_dim) as i64, // k_row_stride
        head_dim as i64,                  // k_head_stride
        (num_heads * head_dim) as i64,    // o_row_stride
        head_dim as i64,                  // o_head_stride
        softmax_scale,
        if is_causal { 1 } else { 0 },
        window_size_left,
        if is_causal { 0 } else { -1 },
        softcap,
        if q.dtype() == DType::BF16 { 1 } else { 0 },
        1, // num_splits
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        stream,
    );

    out
}

/// Paged FlashAttention-2 forward pass.
///
/// Uses upstream vllm-project/flash-attention kernel via `mha_varlen_fwd` shim.
/// For paged KV, forces the splitkv kernel which correctly handles block_table.
///
/// * `q`: `[total_q_tokens, num_heads, head_dim]` (contiguous)
/// * `k_cache`: `[num_blocks, block_size, num_kv_heads, head_dim]`
/// * `v_cache`: same layout
/// * `cu_seqlens_q`: `[batch_size + 1]` (I32 on GPU, cumulative seq lengths for Q)
/// * `seqused_k`: `[batch_size]` (I32 on GPU, actual K length per sequence)
/// * `block_table`: `[batch_size, max_blocks_per_seq]` (I32 on GPU)
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_paged(
    q: GpuTensor,
    k_cache: GpuTensor,
    v_cache: GpuTensor,
    cu_seqlens_q: GpuTensor,
    seqused_k: GpuTensor,
    block_table: GpuTensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    is_causal: bool,
    block_size: usize,
    num_sm: i32,
    alloc: &mut CachingAllocator,
    _stream: CUstream,
) -> OwnedTensor {
    flash_attn_paged_ext(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        seqused_k,
        block_table,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        is_causal,
        0.0,
        -1,
        block_size,
        num_sm,
        alloc,
        _stream,
    )
}

/// Paged FlashAttention-2 with softcap and sliding window support.
///
/// Calls upstream mha_varlen_fwd which forces the splitkv kernel for paged KV.
/// Matches Python vLLM's flash_attn_varlen_func calling convention exactly.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_paged_ext(
    q: GpuTensor,
    k_cache: GpuTensor,
    v_cache: GpuTensor,
    cu_seqlens_q: GpuTensor,
    seqused_k: GpuTensor,
    block_table: GpuTensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    is_causal: bool,
    softcap: f32,
    window_size_left: i32,
    block_size: usize,
    _num_sm: i32,
    alloc: &mut CachingAllocator,
    _stream: CUstream,
) -> OwnedTensor {
    let total_q = q.dim(0);
    let num_heads = q.dim(1);
    let head_dim = q.dim(2);
    let num_kv_heads = k_cache.dim(2);

    let batch_size = cu_seqlens_q.dim(0) - 1;
    let max_blocks_per_seq = if block_table.ndim() == 2 {
        block_table.dim(1)
    } else {
        0
    };

    // Allocate output and softmax_lse from arena.
    let out = alloc.alloc_tensor(&[total_q, num_heads, head_dim], q.dtype());
    let softmax_lse = alloc.alloc_tensor(&[num_heads * total_q], DType::F32);

    // Q/O: [total_q, num_heads, head_dim] contiguous
    let q_row_stride = (num_heads * head_dim) as i64;
    let q_head_stride = head_dim as i64;

    // K/V cache: [num_blocks, block_size, num_kv_heads, head_dim]
    let kv_block_stride = (block_size * num_kv_heads * head_dim) as i64;
    let kv_row_stride = (num_kv_heads * head_dim) as i64;
    let kv_head_stride = head_dim as i64;

    // Python passes dummy zeros for cu_seqlens_k when using paged KV + seqused_k.
    // We allocate a zero-filled buffer from the arena for this.
    let dummy_cu_seqlens_k = alloc.alloc_tensor(&[batch_size + 1], DType::I32);
    // Arena memory is NOT guaranteed to be zeroed. Zero it explicitly.
    crate::driver::memset_d8(
        dummy_cu_seqlens_k.raw_ptr(),
        0,
        (batch_size + 1) * 4,
        _stream,
    )
    .expect("memset dummy_cu_seqlens_k");

    // Always use num_splits=1, matching Python vLLM's FA2 behavior.
    // The upstream vllm-flash-attn C code always sets params.num_splits = 1.
    // Our shim previously computed multi-split values via a heuristic, but
    // Python FA2 never uses num_splits > 1.
    let num_splits = 1;
    let lse_accum_ptr: *mut c_void = std::ptr::null_mut();
    let out_accum_ptr: *mut c_void = std::ptr::null_mut();

    mha_varlen_fwd(
        q.raw_ptr() as *mut c_void,
        k_cache.raw_ptr() as *mut c_void,
        v_cache.raw_ptr() as *mut c_void,
        out.raw_ptr() as *mut c_void,
        softmax_lse.raw_ptr() as *mut c_void,
        cu_seqlens_q.as_ptr::<i32>(),
        dummy_cu_seqlens_k.as_ptr::<i32>(),
        seqused_k.as_ptr::<i32>(),
        block_table.as_ptr::<i32>(),
        max_blocks_per_seq as i32,
        batch_size as i32,
        max_seqlen_q as i32,
        max_seqlen_k as i32,
        num_heads as i32,
        num_kv_heads as i32,
        head_dim as i32,
        block_size as i32,
        q_row_stride,
        q_head_stride,
        kv_block_stride,
        kv_row_stride,
        kv_head_stride,
        q_row_stride,  // o_row_stride = same as q
        q_head_stride, // o_head_stride = same as q
        softmax_scale,
        if is_causal { 1 } else { 0 },
        window_size_left,
        if is_causal { 0 } else { -1 },
        softcap,
        if q.dtype() == DType::BF16 { 1 } else { 0 },
        num_splits as i32,
        lse_accum_ptr,
        out_accum_ptr,
        _stream,
    );

    out
}

// ---------------------------------------------------------------------------
// Scalar multiply (in-place via cuBLAS)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn cublasScalEx(
        handle: cudarc::cublas::sys::cublasHandle_t,
        n: c_int,
        alpha: *const c_void,
        alphaType: cudarc::cublas::sys::cudaDataType_t,
        x: *mut c_void,
        xType: cudarc::cublas::sys::cudaDataType_t,
        incx: c_int,
        executionType: cudarc::cublas::sys::cudaDataType_t,
    ) -> cudarc::cublas::sys::cublasStatus_t;
}

/// Multiply every element of a tensor by a scalar, in-place.
///
/// * `x`: any contiguous tensor (F16, BF16, or F32)
/// * `scale`: the scalar multiplier (always f32)
/// * `cublas`: cuBLAS handle on the compute stream
pub unsafe fn scale_inplace(x: GpuTensor, scale: f32, cublas: &crate::cublas::CublasHandle) {
    use cudarc::cublas::sys::cudaDataType_t;
    let n = x.numel() as c_int;
    let x_type = match x.dtype() {
        DType::F16 => cudaDataType_t::CUDA_R_16F,
        DType::BF16 => cudaDataType_t::CUDA_R_16BF,
        DType::F32 => cudaDataType_t::CUDA_R_32F,
        _ => panic!("scale_inplace: unsupported dtype {:?}", x.dtype()),
    };
    let status = cublasScalEx(
        cublas.raw_handle(),
        n,
        &scale as *const f32 as *const c_void,
        cudaDataType_t::CUDA_R_32F,
        x.raw_ptr() as *mut c_void,
        x_type,
        1,
        cudaDataType_t::CUDA_R_32F,
    );
    assert_eq!(
        status,
        cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
    );
}

// ---------------------------------------------------------------------------
// Embedding pooling (GPU-side)
// ---------------------------------------------------------------------------

/// Extract a single row from `[num_tokens, hidden_size]` → `[hidden_size]` on GPU.
/// Used for Last-token and CLS pooling. Just a D2D memcpy of one row.
///
/// # Safety
/// `hidden_states` must be a valid 2D GPU tensor. `row_idx < num_tokens`.
pub unsafe fn pool_select_row(
    hidden_states: GpuTensor,
    row_idx: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let hidden_size = hidden_states.dim(1);
    let elem_bytes = hidden_states.dtype().size_bytes();
    let row_bytes = hidden_size * elem_bytes;

    let out = alloc.alloc_tensor(&[hidden_size], hidden_states.dtype());
    let src = hidden_states.raw_ptr().add(row_idx * row_bytes);
    crate::driver::memcpy_dtod_async(out.raw_ptr(), src, row_bytes, stream)
        .expect("pool_select_row: D2D copy failed");
    out
}

/// Mean pool: average all rows of `[num_tokens, hidden_size]` → `[hidden_size]`.
///
/// Uses cuBLAS gemv: `out = (1/N) * A^T * ones_vec` where A = `[N, H]` (row-major).
/// The ones vector is allocated from the caching allocator and filled via memset.
///
/// # Safety
/// `hidden_states` must be a valid 2D GPU tensor with dtype F32.
/// For bf16/f16 inputs, caller must cast to f32 first.
pub unsafe fn pool_mean_f32(
    hidden_states: GpuTensor,
    cublas: &crate::cublas::CublasHandle,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    use cudarc::cublas::sys::cublasOperation_t;

    assert_eq!(
        hidden_states.dtype(),
        DType::F32,
        "pool_mean_f32 requires f32 input"
    );
    let num_tokens = hidden_states.dim(0);
    let hidden_size = hidden_states.dim(1);

    if num_tokens == 1 {
        // Single token: just copy the row.
        return pool_select_row(hidden_states, 0, alloc, stream);
    }

    // Allocate ones vector [num_tokens] filled with 1.0f32.
    let ones = alloc.alloc_tensor(&[num_tokens], DType::F32);
    // Fill with 1.0f32 (bit pattern 0x3F800000). Use a kernel-free approach:
    // set all bytes to 0, then use cuBLAS to set to 1.0 would be circular.
    // Instead, write 1.0f32 via a small H2D.
    let ones_host: Vec<f32> = vec![1.0f32; num_tokens];
    crate::driver::memcpy_htod_async(
        ones.raw_ptr(),
        ones_host.as_ptr() as *const u8,
        num_tokens * 4,
        stream,
    )
    .expect("pool_mean: H2D ones vector");

    // Allocate output [hidden_size].
    let out = alloc.alloc_tensor(&[hidden_size], DType::F32);

    // cuBLAS gemv: out = alpha * A^T * x + beta * out
    // A is [N, H] in row-major = [H, N] in column-major.
    // We want A^T * x = [H, N]^T * [N] = [H] (sum of rows).
    // In column-major: A is [H, N], op=N means no-transpose, so y = A * x = [H].
    let alpha = 1.0f32 / num_tokens as f32;
    let beta = 0.0f32;

    unsafe extern "C" {
        fn cublasSgemv_v2(
            handle: cudarc::cublas::sys::cublasHandle_t,
            trans: cublasOperation_t,
            m: c_int,
            n: c_int,
            alpha: *const f32,
            a: *const f32,
            lda: c_int,
            x: *const f32,
            incx: c_int,
            beta: *const f32,
            y: *mut f32,
            incy: c_int,
        ) -> cudarc::cublas::sys::cublasStatus_t;
    }

    // Row-major [N, H] is column-major [H, N]. We want sum of rows = A^T * ones
    // In column-major: A = [H, N], trans=T → A^T * x = [N, H] * [N] — wrong dims.
    // Actually: row-major [N, H] stored as contiguous memory.
    // In cuBLAS column-major convention, this is a [H, N] matrix (columns are rows).
    // We want: out[h] = sum_n A[n, h] / N = (1/N) * A^T_colmajor * ones
    // A_colmajor = [H, N], transpose it → [N, H], multiply by ones[N] → [N] — wrong.
    // No: A_colmajor = [H, N], no transpose: y = A * x = [H, N] * [N, 1] = [H, 1]. Correct!
    let status = cublasSgemv_v2(
        cublas.raw_handle(),
        cublasOperation_t::CUBLAS_OP_N, // no transpose
        hidden_size as c_int,           // m = H
        num_tokens as c_int,            // n = N
        &alpha,
        hidden_states.as_ptr::<f32>(),
        hidden_size as c_int, // lda = H (column-major leading dim)
        ones.as_ptr::<f32>(),
        1,
        &beta,
        out.as_mut_ptr::<f32>(),
        1,
    );
    assert_eq!(
        status,
        cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        "pool_mean_f32: cublasSgemv failed"
    );

    out
}

// ---------------------------------------------------------------------------
// QKV Split (zero-copy on dim 1 via pointer arithmetic + D2D copy)
// ---------------------------------------------------------------------------

/// Split fused QKV output into separate Q, K, V tensors via CUDA kernel.
///
/// * `qkv`: `[num_tokens, q_size + 2 * kv_size]` (contiguous)
/// * Returns: `(q, k, v)` each `[num_tokens, heads, head_dim]`
///
/// Uses a single kernel launch instead of per-token D2D memcpy calls,
/// eliminating O(num_tokens * 3) driver API overhead per layer.
pub unsafe fn split_qkv(
    qkv: GpuTensor,
    q_size: usize,
    kv_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    alloc: &mut CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> (OwnedTensor, OwnedTensor, OwnedTensor) {
    let num_tokens = qkv.dim(0);
    let total_dim = qkv.dim(1);

    debug_assert_eq!(total_dim, q_size + 2 * kv_size);

    let q = alloc.alloc_tensor(&[num_tokens, num_q_heads, head_dim], qkv.dtype());
    let k = alloc.alloc_tensor(&[num_tokens, num_kv_heads, head_dim], qkv.dtype());
    let v = alloc.alloc_tensor(&[num_tokens, num_kv_heads, head_dim], qkv.dtype());

    match qkv.dtype() {
        DType::F16 => split_qkv_f16(
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            v.as_mut_ptr(),
            qkv.as_ptr(),
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            num_tokens as i32,
            stream,
        ),
        DType::BF16 => split_qkv_bf16(
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            v.as_mut_ptr(),
            qkv.as_ptr(),
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            num_tokens as i32,
            stream,
        ),
        DType::F32 => split_qkv_f32(
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            v.as_mut_ptr(),
            qkv.as_ptr(),
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            num_tokens as i32,
            stream,
        ),
        _ => panic!("split_qkv: unsupported dtype {:?}", qkv.dtype()),
    }

    (q, k, v)
}

/// Fused QKV split + RoPE: reads from the fused QKV GEMM output, applies
/// rotary position embedding to Q and K, copies V, and writes contiguous
/// outputs. Replaces separate `split_qkv` + `rotary_embedding_inplace` calls,
/// saving 1 kernel launch per layer per step.
///
/// * `qkv`: `[num_tokens, q_size + 2*kv_size]` — fused QKV GEMM output
/// * `positions`: `[num_tokens]` u32
/// * `cos_sin_cache`: `[max_pos, rotary_dim]`
/// * Returns: `(q, k, v)` where q is `[num_tokens, num_q_heads, head_dim]`,
///   k and v are `[num_tokens, num_kv_heads, head_dim]`.
pub unsafe fn fused_qkv_rope(
    qkv: GpuTensor,
    positions: GpuTensor,
    cos_sin_cache: GpuTensor,
    q_size: usize,
    kv_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> (OwnedTensor, OwnedTensor, OwnedTensor) {
    let num_tokens = qkv.dim(0);
    let total_dim = qkv.dim(1);
    let rotary_dim = cos_sin_cache.dim(1) as i32;

    debug_assert_eq!(total_dim, q_size + 2 * kv_size);

    let q = alloc.alloc_tensor(&[num_tokens, num_q_heads, head_dim], qkv.dtype());
    let k = alloc.alloc_tensor(&[num_tokens, num_kv_heads, head_dim], qkv.dtype());
    let v = alloc.alloc_tensor(&[num_tokens, num_kv_heads, head_dim], qkv.dtype());

    match qkv.dtype() {
        DType::F16 => fused_qkv_rope_f16(
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            v.as_mut_ptr(),
            qkv.as_ptr(),
            positions.as_ptr(),
            cos_sin_cache.as_ptr(),
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            rotary_dim,
            head_dim as i32,
            num_tokens as i32,
            stream,
        ),
        DType::BF16 => fused_qkv_rope_bf16(
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            v.as_mut_ptr(),
            qkv.as_ptr(),
            positions.as_ptr(),
            cos_sin_cache.as_ptr(),
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            rotary_dim,
            head_dim as i32,
            num_tokens as i32,
            stream,
        ),
        DType::F32 => fused_qkv_rope_f32(
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            v.as_mut_ptr(),
            qkv.as_ptr(),
            positions.as_ptr(),
            cos_sin_cache.as_ptr(),
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            rotary_dim,
            head_dim as i32,
            num_tokens as i32,
            stream,
        ),
        _ => panic!("fused_qkv_rope: unsupported dtype {:?}", qkv.dtype()),
    }

    (q, k, v)
}

// ---------------------------------------------------------------------------
// Batched argmax (GPU greedy sampling — avoids D2H of full logits)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn argmax_batched_f16(
        output: *mut u32,
        logits: *const u16,
        vocab_size: c_int,
        batch_size: c_int,
        stream: CUstream,
    );
    fn argmax_batched_bf16(
        output: *mut u32,
        logits: *const u16,
        vocab_size: c_int,
        batch_size: c_int,
        stream: CUstream,
    );
    fn argmax_batched_f32(
        output: *mut u32,
        logits: *const f32,
        vocab_size: c_int,
        batch_size: c_int,
        stream: CUstream,
    );

    fn sample_gumbel_batched_f16(
        output: *mut u32,
        logits: *const u16,
        vocab_size: c_int,
        batch_size: c_int,
        temperatures: *const f32,
        uniform_randoms: *const f32,
        stream: CUstream,
    );
    fn sample_gumbel_batched_bf16(
        output: *mut u32,
        logits: *const u16,
        vocab_size: c_int,
        batch_size: c_int,
        temperatures: *const f32,
        uniform_randoms: *const f32,
        stream: CUstream,
    );
    fn sample_gumbel_batched_f32(
        output: *mut u32,
        logits: *const f32,
        vocab_size: c_int,
        batch_size: c_int,
        temperatures: *const f32,
        uniform_randoms: *const f32,
        stream: CUstream,
    );

    fn sample_batched_f16(
        output: *mut u32,
        logits: *const u16,
        vocab_size: c_int,
        batch_size: c_int,
        temperatures: *const f32,
        top_ks: *const c_int,
        top_ps: *const f32,
        min_ps: *const f32,
        uniform_randoms: *const f32,
        stream: CUstream,
    );
    fn sample_batched_bf16(
        output: *mut u32,
        logits: *const u16,
        vocab_size: c_int,
        batch_size: c_int,
        temperatures: *const f32,
        top_ks: *const c_int,
        top_ps: *const f32,
        min_ps: *const f32,
        uniform_randoms: *const f32,
        stream: CUstream,
    );
    fn sample_batched_f32(
        output: *mut u32,
        logits: *const f32,
        vocab_size: c_int,
        batch_size: c_int,
        temperatures: *const f32,
        top_ks: *const c_int,
        top_ps: *const f32,
        min_ps: *const f32,
        uniform_randoms: *const f32,
        stream: CUstream,
    );
}

/// Batched argmax over logits `[batch_size, vocab_size]`.
///
/// Returns `[batch_size]` u32 tensor of token IDs, allocated from arena.
/// Only copies `batch_size * 4` bytes D2H instead of `batch_size * vocab_size * dtype_size`.
pub unsafe fn argmax_batched(
    logits: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let batch_size = logits.dim(0) as c_int;
    let vocab_size = logits.dim(1) as c_int;
    let out = alloc.alloc_tensor(&[batch_size as usize], DType::U32);

    match logits.dtype() {
        DType::F16 => argmax_batched_f16(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const u16,
            vocab_size,
            batch_size,
            stream,
        ),
        DType::BF16 => argmax_batched_bf16(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const u16,
            vocab_size,
            batch_size,
            stream,
        ),
        DType::F32 => argmax_batched_f32(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const f32,
            vocab_size,
            batch_size,
            stream,
        ),
        _ => panic!("argmax_batched: unsupported dtype {:?}", logits.dtype()),
    }
    out
}

/// Fast batched sampling via the Gumbel-max trick.
///
/// Equivalent to sampling from `softmax(logits / temperature)` but uses only
/// a single argmax-like pass (no radix select, no sort). Requires that no
/// top-k, top-p, or min-p filtering is needed.
///
/// * `logits`: `[batch_size, vocab_size]`
/// * `temperatures`: `[batch_size]` (F32, on GPU)
/// * `uniform_randoms`: `[batch_size]` (F32, on GPU) — seeds for per-element noise
///
/// Returns `[batch_size]` u32 tensor of sampled token IDs, allocated from arena.
pub unsafe fn sample_gumbel_batched(
    logits: GpuTensor,
    temperatures: GpuTensor,
    uniform_randoms: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let batch_size = logits.dim(0) as c_int;
    let vocab_size = logits.dim(1) as c_int;
    let out = alloc.alloc_tensor(&[batch_size as usize], DType::U32);

    match logits.dtype() {
        DType::F16 => sample_gumbel_batched_f16(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const u16,
            vocab_size,
            batch_size,
            temperatures.as_ptr(),
            uniform_randoms.as_ptr(),
            stream,
        ),
        DType::BF16 => sample_gumbel_batched_bf16(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const u16,
            vocab_size,
            batch_size,
            temperatures.as_ptr(),
            uniform_randoms.as_ptr(),
            stream,
        ),
        DType::F32 => sample_gumbel_batched_f32(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const f32,
            vocab_size,
            batch_size,
            temperatures.as_ptr(),
            uniform_randoms.as_ptr(),
            stream,
        ),
        _ => panic!(
            "sample_gumbel_batched: unsupported dtype {:?}",
            logits.dtype()
        ),
    }
    out
}

/// Batched GPU sampling with top-k/top-p/min-p over logits `[batch_size, vocab_size]`.
///
/// * `logits`: `[batch_size, vocab_size]`
/// * `temperatures`: `[batch_size]` (F32, on GPU)
/// * `top_ks`: `[batch_size]` (I32, on GPU)
/// * `top_ps`: `[batch_size]` (F32, on GPU)
/// * `min_ps`: `[batch_size]` (F32, on GPU)
/// * `uniform_randoms`: `[batch_size]` (F32, on GPU) — pre-generated U[0,1) random values
///
/// Returns `[batch_size]` u32 tensor of sampled token IDs, allocated from arena.
#[allow(clippy::too_many_arguments)]
pub unsafe fn sample_batched(
    logits: GpuTensor,
    temperatures: GpuTensor,
    top_ks: GpuTensor,
    top_ps: GpuTensor,
    min_ps: GpuTensor,
    uniform_randoms: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let batch_size = logits.dim(0) as c_int;
    let vocab_size = logits.dim(1) as c_int;
    let out = alloc.alloc_tensor(&[batch_size as usize], DType::U32);

    match logits.dtype() {
        DType::F16 => sample_batched_f16(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const u16,
            vocab_size,
            batch_size,
            temperatures.as_ptr(),
            top_ks.as_ptr(),
            top_ps.as_ptr(),
            min_ps.as_ptr(),
            uniform_randoms.as_ptr(),
            stream,
        ),
        DType::BF16 => sample_batched_bf16(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const u16,
            vocab_size,
            batch_size,
            temperatures.as_ptr(),
            top_ks.as_ptr(),
            top_ps.as_ptr(),
            min_ps.as_ptr(),
            uniform_randoms.as_ptr(),
            stream,
        ),
        DType::F32 => sample_batched_f32(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const f32,
            vocab_size,
            batch_size,
            temperatures.as_ptr(),
            top_ks.as_ptr(),
            top_ps.as_ptr(),
            min_ps.as_ptr(),
            uniform_randoms.as_ptr(),
            stream,
        ),
        _ => panic!("sample_batched: unsupported dtype {:?}", logits.dtype()),
    }
    out
}

// ---------------------------------------------------------------------------
// Cast to f32 (for logit modification kernels that require f32)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn cast_to_f32_f16(output: *mut f32, input: *const u16, n: c_int, stream: CUstream);
    fn cast_to_f32_bf16(output: *mut f32, input: *const u16, n: c_int, stream: CUstream);
}

/// Cast logits from any dtype to f32 on GPU.
/// If already f32, returns a view (no copy). Otherwise allocates from arena.
pub unsafe fn cast_logits_to_f32(
    logits: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let n = logits.numel() as c_int;
    let shape: Vec<usize> = logits.shape().iter().map(|&d| d as usize).collect();
    let out = alloc.alloc_tensor(&shape, DType::F32);
    match logits.dtype() {
        DType::F32 => {
            // Just D2D copy.
            crate::driver::memcpy_dtod_async(
                out.as_gpu_tensor().raw_ptr(),
                logits.raw_ptr() as *const u8,
                logits.size_bytes(),
                stream,
            )
            .expect("cast_logits_to_f32: D2D copy failed");
        }
        DType::F16 => cast_to_f32_f16(
            out.as_mut_ptr() as *mut f32,
            logits.as_ptr() as *const u16,
            n,
            stream,
        ),
        DType::BF16 => cast_to_f32_bf16(
            out.as_mut_ptr() as *mut f32,
            logits.as_ptr() as *const u16,
            n,
            stream,
        ),
        _ => panic!("cast_logits_to_f32: unsupported dtype {:?}", logits.dtype()),
    }
    out
}

// ---------------------------------------------------------------------------
// Penalties / Logit bias / Grammar mask / Log-softmax top-k
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn apply_penalties_inplace(
        logits: *mut f32,
        output_token_ids: *const c_int,
        prompt_token_ids: *const c_int,
        rep_penalties: *const f32,
        freq_penalties: *const f32,
        pres_penalties: *const f32,
        vocab_size: c_int,
        batch_size: c_int,
        max_output_len: c_int,
        max_prompt_len: c_int,
        stream: CUstream,
    );

    fn apply_logit_bias_inplace(
        logits: *mut f32,
        bias_token_ids: *const c_int,
        bias_values: *const f32,
        bias_offsets: *const c_int,
        vocab_size: c_int,
        batch_size: c_int,
        stream: CUstream,
    );

    fn apply_grammar_mask_inplace(
        logits: *mut f32,
        logits_backup: *const f32,
        allowed_ids: *const c_int,
        allowed_offsets: *const c_int,
        req_indices: *const c_int,
        vocab_size: c_int,
        num_grammar_reqs: c_int,
        stream: CUstream,
    );

    fn log_softmax_topk(
        logits: *const f32,
        sampled_ids: *const u32,
        vocab_size: c_int,
        batch_size: c_int,
        num_logprobs: c_int,
        out_logprobs: *mut f32,
        out_indices: *mut c_int,
        out_ranks: *mut u32,
        stream: CUstream,
    );
}

/// Apply repetition, frequency, and presence penalties to f32 logits in-place.
///
/// * `logits`: `[batch_size, vocab_size]` f32 on GPU — modified in-place
/// * `output_token_ids`: `[batch_size, max_output_len]` i32 on GPU, padded with `vocab_size`
/// * `prompt_token_ids`: `[batch_size, max_prompt_len]` i32 on GPU, padded with `vocab_size`
/// * `rep_penalties`, `freq_penalties`, `pres_penalties`: `[batch_size]` f32 on GPU
#[allow(clippy::too_many_arguments)]
pub unsafe fn apply_penalties(
    logits: GpuTensor,
    output_token_ids: GpuTensor,
    prompt_token_ids: GpuTensor,
    rep_penalties: GpuTensor,
    freq_penalties: GpuTensor,
    pres_penalties: GpuTensor,
    stream: CUstream,
) {
    let batch_size = logits.dim(0) as c_int;
    let vocab_size = logits.dim(1) as c_int;
    let max_output_len = output_token_ids.dim(1) as c_int;
    let max_prompt_len = prompt_token_ids.dim(1) as c_int;

    apply_penalties_inplace(
        logits.as_mut_ptr() as *mut f32,
        output_token_ids.as_ptr() as *const c_int,
        prompt_token_ids.as_ptr() as *const c_int,
        rep_penalties.as_ptr(),
        freq_penalties.as_ptr(),
        pres_penalties.as_ptr(),
        vocab_size,
        batch_size,
        max_output_len,
        max_prompt_len,
        stream,
    );
}

/// Apply sparse logit bias via CSR-packed scatter-add on f32 logits in-place.
///
/// * `logits`: `[batch_size, vocab_size]` f32 on GPU
/// * `bias_token_ids`: `[total_biases]` i32 on GPU
/// * `bias_values`: `[total_biases]` f32 on GPU
/// * `bias_offsets`: `[batch_size + 1]` i32 on GPU (CSR offsets)
pub unsafe fn apply_logit_bias(
    logits: GpuTensor,
    bias_token_ids: GpuTensor,
    bias_values: GpuTensor,
    bias_offsets: GpuTensor,
    stream: CUstream,
) {
    let batch_size = logits.dim(0) as c_int;
    let vocab_size = logits.dim(1) as c_int;

    apply_logit_bias_inplace(
        logits.as_mut_ptr() as *mut f32,
        bias_token_ids.as_ptr() as *const c_int,
        bias_values.as_ptr(),
        bias_offsets.as_ptr() as *const c_int,
        vocab_size,
        batch_size,
        stream,
    );
}

/// Apply grammar mask: set disallowed tokens to -inf using CSR-packed allow-list.
///
/// * `logits`: `[batch_size, vocab_size]` f32 on GPU — modified in-place
/// * `logits_backup`: `[batch_size, vocab_size]` f32 on GPU — pristine copy for restoring allowed tokens
/// * `allowed_ids`: `[total_allowed]` i32 on GPU
/// * `allowed_offsets`: `[num_grammar_reqs + 1]` i32 on GPU
/// * `req_indices`: `[num_grammar_reqs]` i32 on GPU — maps grammar index to batch row
pub unsafe fn apply_grammar_mask(
    logits: GpuTensor,
    logits_backup: GpuTensor,
    allowed_ids: GpuTensor,
    allowed_offsets: GpuTensor,
    req_indices: GpuTensor,
    stream: CUstream,
) {
    let vocab_size = logits.dim(1) as c_int;
    let num_grammar_reqs = req_indices.dim(0) as c_int;

    apply_grammar_mask_inplace(
        logits.as_mut_ptr() as *mut f32,
        logits_backup.as_ptr(),
        allowed_ids.as_ptr() as *const c_int,
        allowed_offsets.as_ptr() as *const c_int,
        req_indices.as_ptr() as *const c_int,
        vocab_size,
        num_grammar_reqs,
        stream,
    );
}

/// Fused log-softmax + top-k on f32 logits. Returns per-request top-k logprobs.
///
/// * `logits`: `[batch_size, vocab_size]` f32 on GPU (raw logits before penalties)
/// * `sampled_ids`: `[batch_size]` u32 on GPU
/// * `num_logprobs`: number of top logprobs to return (slot 0 = sampled token)
///
/// Returns `(logprobs, indices, ranks)`:
/// * `logprobs`: `[batch_size, num_logprobs + 1]` f32
/// * `indices`: `[batch_size, num_logprobs + 1]` i32
/// * `ranks`: `[batch_size, num_logprobs + 1]` u32
pub unsafe fn log_softmax_topk_gather(
    logits: GpuTensor,
    sampled_ids: GpuTensor,
    num_logprobs: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> (OwnedTensor, OwnedTensor, OwnedTensor) {
    let batch_size = logits.dim(0);
    let vocab_size = logits.dim(1) as c_int;
    let k = num_logprobs + 1;

    let out_lp = alloc.alloc_tensor(&[batch_size, k], DType::F32);
    let out_idx = alloc.alloc_tensor(&[batch_size, k], DType::U32);
    let out_ranks = alloc.alloc_tensor(&[batch_size, k], DType::U32);

    log_softmax_topk(
        logits.as_ptr(),
        sampled_ids.as_ptr() as *const u32,
        vocab_size,
        batch_size as c_int,
        num_logprobs as c_int,
        out_lp.as_mut_ptr() as *mut f32,
        out_idx.as_mut_ptr() as *mut c_int,
        out_ranks.as_mut_ptr() as *mut u32,
        stream,
    );

    (out_lp, out_idx, out_ranks)
}

// ---------------------------------------------------------------------------
// Tests for FlashAttention-2 FFI shim (mha_varlen_fwd)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests_flash_attn {
    use super::*;
    use crate::driver;

    /// Initialize CUDA context for test. Returns a stream.
    unsafe fn test_init() -> cudarc::driver::sys::CUstream {
        driver::init().expect("CUDA init");
        let dev = driver::device_get(0).expect("device 0");
        let _ctx = driver::ctx_create(dev).expect("ctx_create");
        driver::stream_create().expect("stream_create")
    }

    /// Upload a &[T] to GPU, return raw pointer.
    unsafe fn upload<T: Copy>(data: &[T], stream: cudarc::driver::sys::CUstream) -> *mut u8 {
        let bytes = data.len() * std::mem::size_of::<T>();
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream)
            .expect("memcpy_htod");
        driver::stream_synchronize(stream).expect("sync");
        ptr
    }

    /// Download GPU data to host Vec<T>.
    unsafe fn download<T: Copy + Default>(
        ptr: *mut u8,
        count: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Vec<T> {
        let bytes = count * std::mem::size_of::<T>();
        let mut host = vec![T::default(); count];
        driver::memcpy_dtoh_async(host.as_mut_ptr() as *mut u8, ptr, bytes, stream)
            .expect("memcpy_dtoh");
        driver::stream_synchronize(stream).expect("sync");
        host
    }

    /// Non-paged (contiguous) basic test: Q=K=V=0.1, verify output ~0.1.
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_mha_varlen_fwd_contiguous_basic() {
        unsafe {
            let stream = test_init();
            let (batch, seqlen, heads, head_dim) = (1, 4, 2, 64);
            let total_q = batch * seqlen;
            let nelems = total_q * heads * head_dim;
            let val = half::bf16::from_f32(0.1);
            let data: Vec<u16> = vec![val.to_bits(); nelems];

            let q_ptr = upload(&data, stream);
            let k_ptr = upload(&data, stream);
            let v_ptr = upload(&data, stream);
            let out_ptr = driver::mem_alloc(nelems * 2).expect("out alloc");
            driver::memset_d8(out_ptr, 0, nelems * 2, stream).expect("memset");
            let lse_ptr = driver::mem_alloc(heads * total_q * 4).expect("lse alloc");
            let cu_q_ptr = upload(&[0i32, seqlen as i32], stream);
            let cu_k_ptr = upload(&[0i32, seqlen as i32], stream);

            mha_varlen_fwd(
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                out_ptr as *mut c_void,
                lse_ptr as *mut c_void,
                cu_q_ptr as *const i32,
                cu_k_ptr as *const i32,
                std::ptr::null(), // seqused_k
                std::ptr::null(), // block_table (non-paged)
                0,
                batch as i32,
                seqlen as i32,
                seqlen as i32,
                heads as i32,
                heads as i32,
                head_dim as i32,
                1,
                (heads * head_dim) as i64,
                head_dim as i64,
                0,
                (heads * head_dim) as i64,
                head_dim as i64,
                (heads * head_dim) as i64,
                head_dim as i64,
                1.0 / (head_dim as f32).sqrt(),
                1,
                -1,
                0,
                0.0,
                1,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                stream,
            );
            driver::stream_synchronize(stream).expect("sync");

            let out = download::<u16>(out_ptr, nelems, stream);
            for (i, &bits) in out.iter().enumerate() {
                let f = half::bf16::from_bits(bits).to_f32();
                assert!(
                    (f - 0.1).abs() < 0.05,
                    "contiguous out[{}]={} (want ~0.1)",
                    i,
                    f
                );
            }
            for p in [q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr, cu_q_ptr, cu_k_ptr] {
                let _ = driver::mem_free(p);
            }
        }
    }

    /// Paged basic test: single block, single sequence.
    #[test]
    #[ignore]
    fn test_mha_varlen_fwd_paged_basic() {
        unsafe {
            let stream = test_init();
            let (batch, q_len, kv_len, heads, head_dim, block_size) = (1, 1, 4, 2, 64, 16);
            let val = half::bf16::from_f32(0.1);

            let q_ptr = upload(&vec![val.to_bits(); q_len * heads * head_dim], stream);
            let cache_elems = 1 * block_size * heads * head_dim;
            let k_ptr = upload(&vec![val.to_bits(); cache_elems], stream);
            let v_ptr = upload(&vec![val.to_bits(); cache_elems], stream);
            let out_elems = q_len * heads * head_dim;
            let out_ptr = driver::mem_alloc(out_elems * 2).expect("out");
            let lse_ptr = driver::mem_alloc(heads * q_len * 4).expect("lse");
            let cu_q_ptr = upload(&[0i32, 1], stream);
            let cu_k_ptr = upload(&[0i32, 0], stream); // dummy
            let seqused_ptr = upload(&[kv_len as i32], stream);
            let bt_ptr = upload(&[0i32], stream);

            mha_varlen_fwd(
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                out_ptr as *mut c_void,
                lse_ptr as *mut c_void,
                cu_q_ptr as *const i32,
                cu_k_ptr as *const i32,
                seqused_ptr as *const i32,
                bt_ptr as *const i32,
                1,
                batch as i32,
                q_len as i32,
                kv_len as i32,
                heads as i32,
                heads as i32,
                head_dim as i32,
                block_size as i32,
                (heads * head_dim) as i64,
                head_dim as i64,
                (block_size * heads * head_dim) as i64,
                (heads * head_dim) as i64,
                head_dim as i64,
                (heads * head_dim) as i64,
                head_dim as i64,
                1.0 / (head_dim as f32).sqrt(),
                1,
                -1,
                0,
                0.0,
                1,
                1, // num_splits=1 (paged requires explicit value)
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                stream,
            );
            driver::stream_synchronize(stream).expect("sync");

            let out = download::<u16>(out_ptr, out_elems, stream);
            for (i, &bits) in out.iter().enumerate() {
                let f = half::bf16::from_bits(bits).to_f32();
                assert!((f - 0.1).abs() < 0.05, "paged out[{}]={} (want ~0.1)", i, f);
            }
            for p in [
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                lse_ptr,
                cu_q_ptr,
                cu_k_ptr,
                seqused_ptr,
                bt_ptr,
            ] {
                let _ = driver::mem_free(p);
            }
        }
    }

    /// THE critical test: non-contiguous block allocation.
    /// block_table=[2, 0] — first tokens in block 2, later tokens in block 0.
    /// Block 1 is garbage. The old standard kernel would read linearly and hit garbage.
    #[test]
    #[ignore]
    fn test_mha_varlen_fwd_paged_noncontiguous_blocks() {
        unsafe {
            let stream = test_init();
            // block_size must be multiple of 16 (upstream requirement)
            let (batch, q_len, kv_len, heads, head_dim, block_size) = (1, 1, 32, 2, 64, 16);
            let num_blocks = 3;
            let val_a = half::bf16::from_f32(0.2);
            let val_b = half::bf16::from_f32(0.4);

            let q_ptr = upload(
                &vec![half::bf16::from_f32(0.1).to_bits(); q_len * heads * head_dim],
                stream,
            );

            // Build cache: block 0=val_b, block 1=garbage, block 2=val_a
            let epb = block_size * heads * head_dim;
            let mut cache: Vec<u16> = vec![0u16; num_blocks * epb];
            for i in 0..epb {
                cache[i] = val_b.to_bits();
            } // block 0
            for i in 0..epb {
                cache[epb + i] = 0xDEAD;
            } // block 1 (garbage)
            for i in 0..epb {
                cache[2 * epb + i] = val_a.to_bits();
            } // block 2
            let k_ptr = upload(&cache, stream);
            let v_ptr = upload(&cache, stream);

            let out_elems = q_len * heads * head_dim;
            let out_ptr = driver::mem_alloc(out_elems * 2).expect("out");
            let lse_ptr = driver::mem_alloc(heads * q_len * 4).expect("lse");
            let cu_q_ptr = upload(&[0i32, 1], stream);
            let cu_k_ptr = upload(&[0i32, 0], stream);
            let seqused_ptr = upload(&[kv_len as i32], stream);
            // block_table = [2, 0]: tokens 0-15 in block 2, tokens 16-31 in block 0
            let bt_ptr = upload(&[2i32, 0], stream);

            mha_varlen_fwd(
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                out_ptr as *mut c_void,
                lse_ptr as *mut c_void,
                cu_q_ptr as *const i32,
                cu_k_ptr as *const i32,
                seqused_ptr as *const i32,
                bt_ptr as *const i32,
                2,
                batch as i32,
                q_len as i32,
                kv_len as i32,
                heads as i32,
                heads as i32,
                head_dim as i32,
                block_size as i32,
                (heads * head_dim) as i64,
                head_dim as i64,
                (block_size * heads * head_dim) as i64,
                (heads * head_dim) as i64,
                head_dim as i64,
                (heads * head_dim) as i64,
                head_dim as i64,
                1.0 / (head_dim as f32).sqrt(),
                1,
                -1,
                0,
                0.0,
                1,
                1, // num_splits=1 (paged requires explicit value)
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                stream,
            );
            driver::stream_synchronize(stream).expect("sync");

            // Expected: uniform attention over 4*0.2 + 4*0.4 = avg 0.3
            let out = download::<u16>(out_ptr, out_elems, stream);
            for (i, &bits) in out.iter().enumerate() {
                let f = half::bf16::from_bits(bits).to_f32();
                assert!(
                    (f - 0.3).abs() < 0.1,
                    "noncontig out[{}]={} (want ~0.3)",
                    i,
                    f
                );
            }
            for p in [
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                lse_ptr,
                cu_q_ptr,
                cu_k_ptr,
                seqused_ptr,
                bt_ptr,
            ] {
                let _ = driver::mem_free(p);
            }
        }
    }

    /// Paged with batch_size=2 — verifies second batch element isn't corrupted.
    #[test]
    #[ignore]
    fn test_mha_varlen_fwd_paged_batch2() {
        unsafe {
            let stream = test_init();
            let (batch, heads, head_dim, block_size) = (2, 2, 64, 16);
            let kv_lens = [4i32, 6i32];
            let total_q = 2;
            let val = half::bf16::from_f32(0.1);

            let q_ptr = upload(&vec![val.to_bits(); total_q * heads * head_dim], stream);
            let num_blocks = 2;
            let epb = block_size * heads * head_dim;
            let k_ptr = upload(&vec![val.to_bits(); num_blocks * epb], stream);
            let v_ptr = upload(&vec![val.to_bits(); num_blocks * epb], stream);

            let out_elems = total_q * heads * head_dim;
            let out_ptr = driver::mem_alloc(out_elems * 2).expect("out");
            let lse_ptr = driver::mem_alloc(heads * total_q * 4).expect("lse");
            let cu_q_ptr = upload(&[0i32, 1, 2], stream);
            let cu_k_ptr = upload(&[0i32, 0, 0], stream);
            let seqused_ptr = upload(&kv_lens, stream);
            let bt_ptr = upload(&[0i32, -1, 1, -1], stream); // seq0→block0, seq1→block1

            mha_varlen_fwd(
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                out_ptr as *mut c_void,
                lse_ptr as *mut c_void,
                cu_q_ptr as *const i32,
                cu_k_ptr as *const i32,
                seqused_ptr as *const i32,
                bt_ptr as *const i32,
                2,
                batch as i32,
                1,
                6, // max_seqlen_q=1, max_seqlen_k=6
                heads as i32,
                heads as i32,
                head_dim as i32,
                block_size as i32,
                (heads * head_dim) as i64,
                head_dim as i64,
                (block_size * heads * head_dim) as i64,
                (heads * head_dim) as i64,
                head_dim as i64,
                (heads * head_dim) as i64,
                head_dim as i64,
                1.0 / (head_dim as f32).sqrt(),
                1,
                -1,
                0,
                0.0,
                1,
                1, // num_splits=1 (paged requires explicit value)
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                stream,
            );
            driver::stream_synchronize(stream).expect("sync");

            let out = download::<u16>(out_ptr, out_elems, stream);
            for (i, &bits) in out.iter().enumerate() {
                let f = half::bf16::from_bits(bits).to_f32();
                assert!(
                    (f - 0.1).abs() < 0.05,
                    "batch2 out[{}]={} (want ~0.1)",
                    i,
                    f
                );
            }
            for p in [
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                lse_ptr,
                cu_q_ptr,
                cu_k_ptr,
                seqused_ptr,
                bt_ptr,
            ] {
                let _ = driver::mem_free(p);
            }
        }
    }

    /// GQA test: num_heads=4, num_kv_heads=2.
    #[test]
    #[ignore]
    fn test_mha_varlen_fwd_paged_gqa() {
        unsafe {
            let stream = test_init();
            let (batch, q_len, kv_len, num_heads, num_kv_heads, head_dim, block_size) =
                (1, 1, 4, 4, 2, 64, 16);
            let val = half::bf16::from_f32(0.1);

            let q_ptr = upload(&vec![val.to_bits(); q_len * num_heads * head_dim], stream);
            let epb = block_size * num_kv_heads * head_dim;
            let k_ptr = upload(&vec![val.to_bits(); epb], stream);
            let v_ptr = upload(&vec![val.to_bits(); epb], stream);

            let out_elems = q_len * num_heads * head_dim;
            let out_ptr = driver::mem_alloc(out_elems * 2).expect("out");
            let lse_ptr = driver::mem_alloc(num_heads * q_len * 4).expect("lse");
            let cu_q_ptr = upload(&[0i32, 1], stream);
            let cu_k_ptr = upload(&[0i32, 0], stream);
            let seqused_ptr = upload(&[kv_len as i32], stream);
            let bt_ptr = upload(&[0i32], stream);

            mha_varlen_fwd(
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                out_ptr as *mut c_void,
                lse_ptr as *mut c_void,
                cu_q_ptr as *const i32,
                cu_k_ptr as *const i32,
                seqused_ptr as *const i32,
                bt_ptr as *const i32,
                1,
                batch as i32,
                q_len as i32,
                kv_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                block_size as i32,
                (num_heads * head_dim) as i64,
                head_dim as i64,
                (block_size * num_kv_heads * head_dim) as i64,
                (num_kv_heads * head_dim) as i64,
                head_dim as i64,
                (num_heads * head_dim) as i64,
                head_dim as i64,
                1.0 / (head_dim as f32).sqrt(),
                1,
                -1,
                0,
                0.0,
                1,
                1, // num_splits=1 (paged requires explicit value)
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                stream,
            );
            driver::stream_synchronize(stream).expect("sync");

            let out = download::<u16>(out_ptr, out_elems, stream);
            for (i, &bits) in out.iter().enumerate() {
                let f = half::bf16::from_bits(bits).to_f32();
                assert!((f - 0.1).abs() < 0.05, "GQA out[{}]={} (want ~0.1)", i, f);
            }
            for p in [
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                lse_ptr,
                cu_q_ptr,
                cu_k_ptr,
                seqused_ptr,
                bt_ptr,
            ] {
                let _ = driver::mem_free(p);
            }
        }
    }

    /// Compare paged FA2 with contiguous block_table [0,1,2] vs shuffled [2,0,1].
    /// Same KV data in blocks, just different block_table mapping.
    /// If outputs differ, the paged kernel mishandles non-contiguous block mappings.
    #[test]
    #[ignore]
    fn test_paged_vs_contiguous_gqa() {
        unsafe {
            let stream = test_init();
            // Realistic GQA config matching Qwen2.5-0.5B
            let (num_heads, num_kv_heads, head_dim, block_size) = (14, 2, 64, 16);
            let kv_len = 37; // spans 3 blocks (16+16+5)
            let scale = 1.0 / (head_dim as f32).sqrt();

            // Generate deterministic pseudo-random bf16 data
            let mut rng_state: u32 = 42;
            let mut next_bf16 = || -> u16 {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let f = (rng_state >> 16) as f32 / 65536.0 - 0.5;
                half::bf16::from_f32(f).to_bits()
            };

            // Q: [1, num_heads, head_dim]
            let q_data: Vec<u16> = (0..num_heads * head_dim).map(|_| next_bf16()).collect();
            let q_ptr = upload(&q_data, stream);

            let row_bytes = num_kv_heads * head_dim; // elements per row
            let block_elems = block_size * row_bytes;

            // Generate 3 blocks of KV data (virtual blocks 0, 1, 2)
            let kv0: Vec<u16> = (0..block_elems).map(|_| next_bf16()).collect();
            let kv1: Vec<u16> = (0..block_elems).map(|_| next_bf16()).collect();
            let kv2: Vec<u16> = (0..block_elems).map(|_| next_bf16()).collect();
            let v0: Vec<u16> = (0..block_elems).map(|_| next_bf16()).collect();
            let v1: Vec<u16> = (0..block_elems).map(|_| next_bf16()).collect();
            let v2: Vec<u16> = (0..block_elems).map(|_| next_bf16()).collect();

            // --- Layout A: physical [0,1,2] = virtual [0,1,2], block_table=[0,1,2] ---
            let mut k_a = Vec::with_capacity(3 * block_elems);
            k_a.extend_from_slice(&kv0);
            k_a.extend_from_slice(&kv1);
            k_a.extend_from_slice(&kv2);
            let mut v_a = Vec::with_capacity(3 * block_elems);
            v_a.extend_from_slice(&v0);
            v_a.extend_from_slice(&v1);
            v_a.extend_from_slice(&v2);
            let k_a_ptr = upload(&k_a, stream);
            let v_a_ptr = upload(&v_a, stream);
            let bt_a = upload(&[0i32, 1, 2], stream);

            // --- Layout B: physical [0,1,2] = virtual [2,0,1], block_table=[1,2,0] ---
            // Physical 0 has virtual 2's data, physical 1 has virtual 0's, physical 2 has virtual 1's
            let mut k_b = Vec::with_capacity(3 * block_elems);
            k_b.extend_from_slice(&kv2); // physical 0 = virtual 2
            k_b.extend_from_slice(&kv0); // physical 1 = virtual 0
            k_b.extend_from_slice(&kv1); // physical 2 = virtual 1
            let mut v_b = Vec::with_capacity(3 * block_elems);
            v_b.extend_from_slice(&v2);
            v_b.extend_from_slice(&v0);
            v_b.extend_from_slice(&v1);
            let k_b_ptr = upload(&k_b, stream);
            let v_b_ptr = upload(&v_b, stream);
            let bt_b = upload(&[1i32, 2, 0], stream); // virtual 0→phys 1, virtual 1→phys 2, virtual 2→phys 0

            // Shared metadata
            let cu_q = upload(&[0i32, 1], stream);
            let cu_k_dummy = upload(&[0i32, 0], stream);
            let seqused = upload(&[kv_len as i32], stream);

            let kv_block_stride = (block_size * num_kv_heads * head_dim) as i64;
            let kv_row_stride = (num_kv_heads * head_dim) as i64;
            let kv_head_stride = head_dim as i64;

            let call_paged = |k_ptr, v_ptr, bt_ptr, out_ptr, lse_ptr| {
                mha_varlen_fwd(
                    q_ptr as *mut c_void,
                    k_ptr as *mut c_void,
                    v_ptr as *mut c_void,
                    out_ptr as *mut c_void,
                    lse_ptr as *mut c_void,
                    cu_q as *const i32,
                    cu_k_dummy as *const i32,
                    seqused as *const i32,
                    bt_ptr as *const i32,
                    3, // block_table_batch_stride
                    1, // batch_size
                    1, // max_seqlen_q
                    kv_len as i32,
                    num_heads as i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    block_size as i32,
                    (num_heads * head_dim) as i64,
                    head_dim as i64,
                    kv_block_stride,
                    kv_row_stride,
                    kv_head_stride,
                    (num_heads * head_dim) as i64,
                    head_dim as i64,
                    scale,
                    1,
                    -1,
                    0,
                    0.0,
                    1,
                    1, // num_splits
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    stream,
                );
            };

            let out_a = driver::mem_alloc(num_heads * head_dim * 2).expect("out a");
            let lse_a = driver::mem_alloc(num_heads * 4).expect("lse a");
            call_paged(k_a_ptr, v_a_ptr, bt_a, out_a, lse_a);
            driver::stream_synchronize(stream).expect("sync a");

            let out_b = driver::mem_alloc(num_heads * head_dim * 2).expect("out b");
            let lse_b = driver::mem_alloc(num_heads * 4).expect("lse b");
            call_paged(k_b_ptr, v_b_ptr, bt_b, out_b, lse_b);
            driver::stream_synchronize(stream).expect("sync b");

            // --- Compare ---
            let res_a = download::<u16>(out_a, num_heads * head_dim, stream);
            let res_b = download::<u16>(out_b, num_heads * head_dim, stream);

            let mut max_diff: f32 = 0.0;
            let mut num_bad = 0;
            for i in 0..res_a.len() {
                let a = half::bf16::from_bits(res_a[i]).to_f32();
                let b = half::bf16::from_bits(res_b[i]).to_f32();
                let diff = (a - b).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > 0.01 {
                    num_bad += 1;
                }
            }
            eprintln!(
                "contiguous [0,1,2] vs shuffled [1,2,0]: max_diff={:.6}, num_bad={}/{} (threshold=0.01)",
                max_diff,
                num_bad,
                res_a.len()
            );

            // Also print first few values from each
            for i in 0..5.min(res_a.len()) {
                let a = half::bf16::from_bits(res_a[i]).to_f32();
                let b = half::bf16::from_bits(res_b[i]).to_f32();
                eprintln!("  [{i}] a={a:.6} b={b:.6} diff={:.6}", (a - b).abs());
            }

            assert!(
                max_diff < 0.02,
                "paged FA2 shuffled blocks diverge from contiguous: max_diff={:.6}, bad={}/{}",
                max_diff,
                num_bad,
                res_a.len()
            );

            // Cleanup
            for p in [
                q_ptr, k_a_ptr, v_a_ptr, k_b_ptr, v_b_ptr, bt_a, bt_b, out_a, lse_a, out_b, lse_b,
                cu_q, cu_k_dummy, seqused,
            ] {
                let _ = driver::mem_free(p);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Parameterized paged FA2 correctness: contiguous vs shuffled blocks.
    // Covers various GQA ratios, head dims, KV lengths, and batch sizes.
    // -----------------------------------------------------------------------

    /// Helper: run paged FA2 with contiguous block mapping [0,1,2,...] and a
    /// shuffled mapping, then assert outputs match within tolerance.
    unsafe fn assert_paged_shuffle_equivalent(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        kv_len: usize,
        batch_size: usize,
        label: &str,
    ) {
        let stream = test_init();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let num_blocks_per_seq = (kv_len + block_size - 1) / block_size;
        let total_physical_blocks = num_blocks_per_seq + 1; // +1 for shuffle room

        let mut rng_state: u32 = 0xBEEF_u32
            .wrapping_mul(num_heads as u32 + 1)
            .wrapping_add(kv_len as u32)
            .wrapping_add(batch_size as u32);
        let mut next_bf16 = || -> u16 {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let f = (rng_state >> 16) as f32 / 65536.0 - 0.5;
            half::bf16::from_f32(f).to_bits()
        };

        let row_elems = num_kv_heads * head_dim;
        let block_elems = block_size * row_elems;
        let total_cache_blocks = total_physical_blocks * batch_size;

        // Allocate Q for the batch: [batch_size, num_heads, head_dim]
        let q_elems = batch_size * num_heads * head_dim;
        let q_data: Vec<u16> = (0..q_elems).map(|_| next_bf16()).collect();
        let q_ptr = upload(&q_data, stream);

        // Allocate KV blocks: enough physical blocks for both layouts
        let cache_elems = total_cache_blocks * block_elems;
        let k_data: Vec<u16> = (0..cache_elems).map(|_| next_bf16()).collect();
        let v_data: Vec<u16> = (0..cache_elems).map(|_| next_bf16()).collect();

        // Layout A: block_table = [0, 1, 2, ...] per batch
        let k_a_ptr = upload(&k_data, stream);
        let v_a_ptr = upload(&v_data, stream);
        let bt_a_data: Vec<i32> = (0..batch_size)
            .flat_map(|b| {
                let base = (b * total_physical_blocks) as i32;
                (0..num_blocks_per_seq).map(move |i| base + i as i32)
            })
            .collect();
        let bt_a = upload(&bt_a_data, stream);

        // Layout B: shuffled — rotate blocks by 1 within each batch element
        // Physical block mapping: virtual[i] -> physical[(i+1) % (num_blocks+1)]
        let mut k_b_data = vec![0u16; cache_elems];
        let mut v_b_data = vec![0u16; cache_elems];
        let mut bt_b_data = Vec::with_capacity(batch_size * num_blocks_per_seq);

        for b in 0..batch_size {
            let phys_base = b * total_physical_blocks;
            for virt_idx in 0..num_blocks_per_seq {
                let phys_a = phys_base + virt_idx;
                let phys_b = phys_base + ((virt_idx + 1) % total_physical_blocks);
                // Copy block data from A's physical location to B's
                let src_off = phys_a * block_elems;
                let dst_off = phys_b * block_elems;
                k_b_data[dst_off..dst_off + block_elems]
                    .copy_from_slice(&k_data[src_off..src_off + block_elems]);
                v_b_data[dst_off..dst_off + block_elems]
                    .copy_from_slice(&v_data[src_off..src_off + block_elems]);
                bt_b_data.push(phys_b as i32);
            }
        }
        let k_b_ptr = upload(&k_b_data, stream);
        let v_b_ptr = upload(&v_b_data, stream);
        let bt_b = upload(&bt_b_data, stream);

        // Sequence metadata
        let cu_q_data: Vec<i32> = (0..=batch_size as i32).collect();
        let cu_q = upload(&cu_q_data, stream);
        let cu_k_dummy: Vec<i32> = vec![0i32; batch_size + 1];
        let cu_k = upload(&cu_k_dummy, stream);
        let seqused_data: Vec<i32> = vec![kv_len as i32; batch_size];
        let seqused = upload(&seqused_data, stream);

        let kv_block_stride = (block_size * num_kv_heads * head_dim) as i64;
        let kv_row_stride = (num_kv_heads * head_dim) as i64;
        let kv_head_stride = head_dim as i64;
        let q_row_stride = (num_heads * head_dim) as i64;
        let q_head_stride = head_dim as i64;

        let call = |k_ptr, v_ptr, bt_ptr, out_ptr, lse_ptr| {
            mha_varlen_fwd(
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                out_ptr as *mut c_void,
                lse_ptr as *mut c_void,
                cu_q as *const i32,
                cu_k as *const i32,
                seqused as *const i32,
                bt_ptr as *const i32,
                num_blocks_per_seq as i32,
                batch_size as i32,
                1, // max_seqlen_q
                kv_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                block_size as i32,
                q_row_stride,
                q_head_stride,
                kv_block_stride,
                kv_row_stride,
                kv_head_stride,
                q_row_stride, // o_row_stride = q_row_stride
                q_head_stride,
                scale,
                1,
                -1,
                0,
                0.0,
                1,
                1, // num_splits
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                stream,
            );
        };

        let out_elems = batch_size * num_heads * head_dim;
        let out_a = driver::mem_alloc(out_elems * 2).expect("out a");
        let lse_a = driver::mem_alloc(batch_size * num_heads * 4).expect("lse a");
        call(k_a_ptr, v_a_ptr, bt_a, out_a, lse_a);
        driver::stream_synchronize(stream).expect("sync a");

        let out_b = driver::mem_alloc(out_elems * 2).expect("out b");
        let lse_b = driver::mem_alloc(batch_size * num_heads * 4).expect("lse b");
        call(k_b_ptr, v_b_ptr, bt_b, out_b, lse_b);
        driver::stream_synchronize(stream).expect("sync b");

        let res_a = download::<u16>(out_a, out_elems, stream);
        let res_b = download::<u16>(out_b, out_elems, stream);

        let mut max_diff: f32 = 0.0;
        let mut num_bad = 0;
        for i in 0..res_a.len() {
            let a = half::bf16::from_bits(res_a[i]).to_f32();
            let b = half::bf16::from_bits(res_b[i]).to_f32();
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 0.01 {
                num_bad += 1;
            }
        }
        eprintln!(
            "[{label}] h={num_heads} hk={num_kv_heads} d={head_dim} bs={block_size} kv={kv_len} batch={batch_size}: max_diff={max_diff:.6}, bad={num_bad}/{}",
            res_a.len()
        );
        assert!(
            max_diff < 0.02,
            "[{label}] paged shuffle diverges: max_diff={max_diff:.6}, bad={num_bad}/{}",
            res_a.len()
        );

        for p in [
            q_ptr, k_a_ptr, v_a_ptr, k_b_ptr, v_b_ptr, bt_a, bt_b, out_a, lse_a, out_b, lse_b,
            cu_q, cu_k, seqused,
        ] {
            let _ = driver::mem_free(p);
        }
    }

    /// GQA ratio 7:1 (Qwen2.5-0.5B config) — the exact bug scenario
    #[test]
    #[ignore]
    fn test_paged_shuffle_gqa_7to1_hdim64() {
        unsafe {
            assert_paged_shuffle_equivalent(14, 2, 64, 16, 37, 1, "gqa_7to1_d64");
        }
    }

    /// GQA ratio 4:1 (common config)
    #[test]
    #[ignore]
    fn test_paged_shuffle_gqa_4to1_hdim128() {
        unsafe {
            assert_paged_shuffle_equivalent(32, 8, 128, 16, 50, 1, "gqa_4to1_d128");
        }
    }

    /// MHA (no GQA, ratio 1:1) — should also work with splitkv
    #[test]
    #[ignore]
    fn test_paged_shuffle_mha_hdim64() {
        unsafe {
            assert_paged_shuffle_equivalent(8, 8, 64, 16, 48, 1, "mha_d64");
        }
    }

    /// Small head_dim=32
    #[test]
    #[ignore]
    fn test_paged_shuffle_hdim32() {
        unsafe {
            assert_paged_shuffle_equivalent(8, 2, 32, 16, 64, 1, "gqa_d32");
        }
    }

    /// Large head_dim=128 with GQA
    #[test]
    #[ignore]
    fn test_paged_shuffle_hdim128_gqa() {
        unsafe {
            assert_paged_shuffle_equivalent(32, 4, 128, 16, 80, 1, "gqa_d128");
        }
    }

    /// Single block — all tokens fit in one page
    #[test]
    #[ignore]
    fn test_paged_shuffle_single_block() {
        unsafe {
            assert_paged_shuffle_equivalent(14, 2, 64, 16, 16, 1, "single_block");
        }
    }

    /// Many blocks (5 blocks, kv_len=80)
    #[test]
    #[ignore]
    fn test_paged_shuffle_many_blocks() {
        unsafe {
            assert_paged_shuffle_equivalent(14, 2, 64, 16, 80, 1, "many_blocks");
        }
    }

    /// Large kv_len spanning multiple kBlockN tiles (kBlockN=128 for hdim=64)
    #[test]
    #[ignore]
    fn test_paged_shuffle_multi_tile() {
        unsafe {
            // 200 tokens = 13 blocks of 16, spans 2 kBlockN tiles
            assert_paged_shuffle_equivalent(14, 2, 64, 16, 200, 1, "multi_tile");
        }
    }

    /// Batch size > 1 with shuffled blocks
    #[test]
    #[ignore]
    fn test_paged_shuffle_batch4() {
        unsafe {
            assert_paged_shuffle_equivalent(14, 2, 64, 16, 48, 4, "batch4");
        }
    }

    /// Batch size 2 with large GQA ratio and hdim=128
    #[test]
    #[ignore]
    fn test_paged_shuffle_batch2_gqa_hdim128() {
        unsafe {
            assert_paged_shuffle_equivalent(32, 4, 128, 16, 64, 2, "batch2_gqa_d128");
        }
    }

    /// Contiguous path (non-paged) through mha_varlen_fwd with null block_table
    #[test]
    #[ignore]
    fn test_contiguous_varlen_fwd() {
        unsafe {
            let stream = test_init();
            let (num_heads, num_kv_heads, head_dim) = (8, 2, 64);
            let kv_len = 32;
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut rng_state: u32 = 12345;
            let mut next_bf16 = || -> u16 {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                half::bf16::from_f32((rng_state >> 16) as f32 / 65536.0 - 0.5).to_bits()
            };

            let q: Vec<u16> = (0..num_heads * head_dim).map(|_| next_bf16()).collect();
            let k: Vec<u16> = (0..kv_len * num_kv_heads * head_dim)
                .map(|_| next_bf16())
                .collect();
            let v: Vec<u16> = (0..kv_len * num_kv_heads * head_dim)
                .map(|_| next_bf16())
                .collect();

            let q_ptr = upload(&q, stream);
            let k_ptr = upload(&k, stream);
            let v_ptr = upload(&v, stream);
            let out_ptr = driver::mem_alloc(num_heads * head_dim * 2).expect("out");
            let lse_ptr = driver::mem_alloc(num_heads * 4).expect("lse");
            let cu_q = upload(&[0i32, 1], stream);
            let cu_k = upload(&[0i32, kv_len as i32], stream);

            mha_varlen_fwd(
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                out_ptr as *mut c_void,
                lse_ptr as *mut c_void,
                cu_q as *const i32,
                cu_k as *const i32,
                std::ptr::null(), // seqused_k = null
                std::ptr::null(), // block_table = null (contiguous)
                0,
                1,
                1,
                kv_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                0, // page_block_size unused
                (num_heads * head_dim) as i64,
                head_dim as i64,
                0, // k_batch_stride unused for contiguous
                (num_kv_heads * head_dim) as i64,
                head_dim as i64,
                (num_heads * head_dim) as i64,
                head_dim as i64,
                scale,
                1,
                -1,
                0,
                0.0,
                1,
                1,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                stream,
            );
            driver::stream_synchronize(stream).expect("sync");

            let out = download::<u16>(out_ptr, num_heads * head_dim, stream);
            // Verify output is not all zeros or NaN
            let mut has_nonzero = false;
            for &bits in &out {
                let f = half::bf16::from_bits(bits).to_f32();
                assert!(!f.is_nan(), "contiguous FA2 produced NaN");
                if f.abs() > 1e-6 {
                    has_nonzero = true;
                }
            }
            assert!(has_nonzero, "contiguous FA2 output is all zeros");

            for p in [q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr, cu_q, cu_k] {
                let _ = driver::mem_free(p);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: GPU pooling kernels
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests_pooling {
    use super::*;
    use crate::alloc::CachingAllocator;
    use crate::driver;

    unsafe fn test_init() -> cudarc::driver::sys::CUstream {
        driver::init().expect("CUDA init");
        let dev = driver::device_get(0).expect("device 0");
        let _ctx = driver::ctx_create(dev).expect("ctx_create");
        driver::stream_create().expect("stream_create")
    }

    unsafe fn upload_f32(data: &[f32], stream: cudarc::driver::sys::CUstream) -> *mut u8 {
        let bytes = data.len() * 4;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("H2D");
        ptr
    }

    unsafe fn download_f32(
        ptr: *mut u8,
        count: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Vec<f32> {
        let mut buf = vec![0.0f32; count];
        driver::memcpy_dtoh_async(buf.as_mut_ptr() as *mut u8, ptr, count * 4, stream)
            .expect("D2H");
        driver::stream_synchronize(stream).expect("sync");
        buf
    }

    // -- pool_select_row tests --

    #[test]
    #[ignore]
    fn test_pool_select_row_first() {
        unsafe {
            let stream = test_init();
            let mut alloc = CachingAllocator::new();

            // 3 tokens, hidden_size=4
            let data: Vec<f32> = vec![
                1.0, 2.0, 3.0, 4.0, // row 0
                5.0, 6.0, 7.0, 8.0, // row 1
                9.0, 10.0, 11.0, 12.0, // row 2
            ];
            let ptr = upload_f32(&data, stream);
            let hs = GpuTensor::new(ptr, &[3, 4], DType::F32);

            let out = pool_select_row(hs, 0, &mut alloc, stream);
            let result = download_f32(out.as_gpu_tensor().raw_ptr(), 4, stream);
            assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);

            let _ = driver::mem_free(ptr);
        }
    }

    #[test]
    #[ignore]
    fn test_pool_select_row_last() {
        unsafe {
            let stream = test_init();
            let mut alloc = CachingAllocator::new();

            let data: Vec<f32> = vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ];
            let ptr = upload_f32(&data, stream);
            let hs = GpuTensor::new(ptr, &[3, 4], DType::F32);

            let out = pool_select_row(hs, 2, &mut alloc, stream);
            let result = download_f32(out.as_gpu_tensor().raw_ptr(), 4, stream);
            assert_eq!(result, vec![9.0, 10.0, 11.0, 12.0]);

            let _ = driver::mem_free(ptr);
        }
    }

    #[test]
    #[ignore]
    fn test_pool_select_row_middle() {
        unsafe {
            let stream = test_init();
            let mut alloc = CachingAllocator::new();

            let data: Vec<f32> = vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ];
            let ptr = upload_f32(&data, stream);
            let hs = GpuTensor::new(ptr, &[3, 4], DType::F32);

            let out = pool_select_row(hs, 1, &mut alloc, stream);
            let result = download_f32(out.as_gpu_tensor().raw_ptr(), 4, stream);
            assert_eq!(result, vec![5.0, 6.0, 7.0, 8.0]);

            let _ = driver::mem_free(ptr);
        }
    }

    #[test]
    #[ignore]
    fn test_pool_select_row_bf16() {
        unsafe {
            let stream = test_init();
            let mut alloc = CachingAllocator::new();

            // 2 tokens, hidden_size=3, bf16
            let row0: Vec<u16> = [1.0f32, 2.0, 3.0]
                .iter()
                .map(|&v| half::bf16::from_f32(v).to_bits())
                .collect();
            let row1: Vec<u16> = [4.0f32, 5.0, 6.0]
                .iter()
                .map(|&v| half::bf16::from_f32(v).to_bits())
                .collect();
            let data: Vec<u16> = [row0, row1].concat();
            let bytes = data.len() * 2;
            let ptr = driver::mem_alloc(bytes).expect("alloc");
            driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("H2D");

            let hs = GpuTensor::new(ptr, &[2, 3], DType::BF16);
            let out = pool_select_row(hs, 1, &mut alloc, stream);

            let mut buf = vec![0u16; 3];
            driver::memcpy_dtoh_async(
                buf.as_mut_ptr() as *mut u8,
                out.as_gpu_tensor().raw_ptr(),
                6,
                stream,
            )
            .expect("D2H");
            driver::stream_synchronize(stream).expect("sync");
            let vals: Vec<f32> = buf
                .iter()
                .map(|&b| half::bf16::from_bits(b).to_f32())
                .collect();
            assert!((vals[0] - 4.0).abs() < 0.1);
            assert!((vals[1] - 5.0).abs() < 0.1);
            assert!((vals[2] - 6.0).abs() < 0.1);

            let _ = driver::mem_free(ptr);
        }
    }

    // -- pool_mean_f32 tests --

    #[test]
    #[ignore]
    fn test_pool_mean_basic() {
        unsafe {
            let stream = test_init();
            let mut alloc = CachingAllocator::new();
            let cublas = crate::cublas::CublasHandle::new(stream).expect("cublas");

            // 3 tokens, hidden_size=3
            // row0=[1,2,3], row1=[3,4,5], row2=[5,6,7] → mean=[3,4,5]
            let data: Vec<f32> = vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0];
            let ptr = upload_f32(&data, stream);
            let hs = GpuTensor::new(ptr, &[3, 3], DType::F32);

            let out = pool_mean_f32(hs, &cublas, &mut alloc, stream);
            driver::stream_synchronize(stream).expect("sync");
            let result = download_f32(out.as_gpu_tensor().raw_ptr(), 3, stream);
            assert!((result[0] - 3.0).abs() < 1e-5, "got {}", result[0]);
            assert!((result[1] - 4.0).abs() < 1e-5, "got {}", result[1]);
            assert!((result[2] - 5.0).abs() < 1e-5, "got {}", result[2]);

            let _ = driver::mem_free(ptr);
        }
    }

    #[test]
    #[ignore]
    fn test_pool_mean_single_row() {
        unsafe {
            let stream = test_init();
            let mut alloc = CachingAllocator::new();
            let cublas = crate::cublas::CublasHandle::new(stream).expect("cublas");

            let data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0];
            let ptr = upload_f32(&data, stream);
            let hs = GpuTensor::new(ptr, &[1, 4], DType::F32);

            let out = pool_mean_f32(hs, &cublas, &mut alloc, stream);
            driver::stream_synchronize(stream).expect("sync");
            let result = download_f32(out.as_gpu_tensor().raw_ptr(), 4, stream);
            assert_eq!(result, vec![7.0, 8.0, 9.0, 10.0]);

            let _ = driver::mem_free(ptr);
        }
    }

    #[test]
    #[ignore]
    fn test_pool_mean_two_rows() {
        unsafe {
            let stream = test_init();
            let mut alloc = CachingAllocator::new();
            let cublas = crate::cublas::CublasHandle::new(stream).expect("cublas");

            // [0, 4] and [2, 6] → mean [1, 5]
            let data: Vec<f32> = vec![0.0, 4.0, 2.0, 6.0];
            let ptr = upload_f32(&data, stream);
            let hs = GpuTensor::new(ptr, &[2, 2], DType::F32);

            let out = pool_mean_f32(hs, &cublas, &mut alloc, stream);
            driver::stream_synchronize(stream).expect("sync");
            let result = download_f32(out.as_gpu_tensor().raw_ptr(), 2, stream);
            assert!((result[0] - 1.0).abs() < 1e-5);
            assert!((result[1] - 5.0).abs() < 1e-5);

            let _ = driver::mem_free(ptr);
        }
    }
}
