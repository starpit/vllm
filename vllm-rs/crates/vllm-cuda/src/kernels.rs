// SPDX-License-Identifier: Apache-2.0
//! Direct kernel dispatch for `GpuTensor` — no candle dependency.
//!
//! These wrap the same CUDA FFI functions from `vllm-kernels/csrc/` but
//! dispatch from `GpuTensor::as_ptr()` instead of extracting raw pointers
//! from candle `Tensor` (which takes ~10 lines per tensor). Here it's one line.

use core::ffi::{c_int, c_void};

use crate::arena::ScratchArena;
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
        cu_seqlens_k: *mut u32,
        block_table: *const u32,
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
    arena: &mut ScratchArena,
    stream: CUstream,
) -> GpuTensor {
    let num_tokens = input.dim(0) as i32;
    let hidden_size = input.dim(1) as i32;
    let out = arena.alloc(&[num_tokens as usize, hidden_size as usize], input.dtype());

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
    arena: &mut ScratchArena,
    stream: cudarc::driver::sys::CUstream,
) -> (GpuTensor, GpuTensor) {
    // Allocate a copy of input in the arena for the normed output.
    let normed_buf = arena.alloc(&[input.dim(0), input.dim(1)], input.dtype());
    crate::driver::memcpy_dtod_async(
        normed_buf.raw_ptr(),
        input.raw_ptr() as *const u8,
        input.size_bytes(),
        stream,
    )
    .expect("fused_add_rms_norm: D2D copy failed");

    fused_add_rms_norm_inplace(normed_buf, residual, weight, eps, stream)
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
    arena: &mut ScratchArena,
    stream: CUstream,
) -> GpuTensor {
    let num_tokens = gate_up.dim(0) as i32;
    let d = intermediate_size as i32;
    let out = arena.alloc(&[num_tokens as usize, intermediate_size], gate_up.dtype());

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
    arena: &mut ScratchArena,
    stream: CUstream,
) -> GpuTensor {
    let num_tokens = gate_up.dim(0) as i32;
    let d = intermediate_size as i32;
    let out = arena.alloc(&[num_tokens as usize, intermediate_size], gate_up.dtype());

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
    arena: &mut ScratchArena,
    stream: CUstream,
) -> GpuTensor {
    let num_tokens = input_ids.dim(0);
    let hidden_size = weight.dim(1);
    let out = arena.alloc(&[num_tokens, hidden_size], weight.dtype());

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
/// increment cu_seqlens_k — all in one kernel launch on the GPU.
///
/// This replaces 3 CPU Vec builds + 3 H2D copies per decode step.
///
/// # Safety
/// All pointers must be valid GPU memory. `positions` and `slot_mapping`
/// must have at least `num_reqs` elements. `cu_seqlens_k` must have
/// `num_reqs + 1` elements. `block_table` must be `[num_reqs, max_blocks_per_seq]`.
pub unsafe fn update_decode_metadata_gpu(
    positions: *mut u8,
    slot_mapping: *mut u8,
    cu_seqlens_k: *mut u8,
    block_table: *const u8,
    num_reqs: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
    stream: CUstream,
) {
    update_decode_metadata(
        positions as *mut u32,
        slot_mapping as *mut i64,
        cu_seqlens_k as *mut u32,
        block_table as *const u32,
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
    fn run_mha_paged(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        o_ptr: *const c_void,
        softmax_lse_ptr: *const c_void,
        alibi_slopes_ptr: *const c_void,

        cu_seqlens_q_ptr: *const i32,
        cu_seqlens_k_ptr: *const i32,

        q_batch_stride: u32,
        k_batch_stride: u32,
        v_batch_stride: u32,
        o_batch_stride: u32,
        alibi_slopes_batch_stride: u32,

        q_row_stride: u32,
        k_row_stride: u32,
        v_row_stride: u32,
        o_row_stride: u32,

        q_head_stride: u32,
        k_head_stride: u32,
        v_head_stride: u32,
        o_head_stride: u32,

        b: u32,
        h: u32,
        h_k: u32,
        d: u32,
        d_rounded: u32,
        softmax_scale: f32,

        seqlen_q: u32,
        seqlen_k: u32,
        seqlen_q_rounded: u32,
        seqlen_k_rounded: u32,

        is_bf16: c_int,
        is_causal: c_int,
        unpadded_lse: c_int,

        window_size_left: c_int,
        window_size_right: c_int,

        softcap: f32,

        block_table_ptr: *const i32,
        block_table_batch_stride: i64,
        page_block_size: c_int,
        num_splits: c_int,
        cuda_stream: CUstream,
    );
}

fn round_multiple(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

/// Paged FlashAttention-2 forward pass.
///
/// * `q`: `[total_q_tokens, num_heads, head_dim]` (contiguous)
/// * `k_cache`: `[num_blocks, block_size, num_kv_heads, head_dim]`
/// * `v_cache`: same layout
/// * `cu_seqlens_q`: `[batch_size + 1]` (U32 on GPU, cumulative seq lengths for Q)
/// * `cu_seqlens_k`: `[batch_size + 1]` (U32 on GPU, cumulative seq lengths for K)
/// * `block_table`: `[batch_size, max_blocks_per_seq]` (U32 on GPU)
/// * `max_seqlen_q`: maximum Q sequence length in this batch
/// * `max_seqlen_k`: maximum K sequence length in this batch
/// * `softmax_scale`: typically `1.0 / sqrt(head_dim)`
/// * `is_causal`: whether to apply causal mask
/// * `softcap`: attention logit soft capping (0.0 = disabled)
/// * `window_size_left`: sliding window size (-1 = unlimited)
/// * `arena`: scratch arena for output + softmax_lse allocation
///
/// Returns: `[total_q_tokens, num_heads, head_dim]` output tensor from arena.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_paged(
    q: GpuTensor,
    k_cache: GpuTensor,
    v_cache: GpuTensor,
    cu_seqlens_q: GpuTensor,
    cu_seqlens_k: GpuTensor,
    block_table: GpuTensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    is_causal: bool,
    block_size: usize,
    arena: &mut ScratchArena,
    stream: CUstream,
) -> GpuTensor {
    flash_attn_paged_ext(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        block_table,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        is_causal,
        0.0,
        -1,
        block_size,
        arena,
        stream,
    )
}

/// Paged FlashAttention-2 with softcap and sliding window support.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_paged_ext(
    q: GpuTensor,
    k_cache: GpuTensor,
    v_cache: GpuTensor,
    cu_seqlens_q: GpuTensor,
    cu_seqlens_k: GpuTensor,
    block_table: GpuTensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    is_causal: bool,
    softcap: f32,
    window_size_left: i32,
    block_size: usize,
    arena: &mut ScratchArena,
    stream: CUstream,
) -> GpuTensor {
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

    let head_size_rounded = round_multiple(head_dim, 32);
    let seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    let seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    // Allocate output and softmax_lse from arena.
    let out = arena.alloc(&[total_q, num_heads, head_dim], q.dtype());
    let softmax_lse = arena.alloc(&[num_heads * total_q], DType::F32);

    let is_bf16_flag: c_int = if q.dtype() == DType::BF16 { 1 } else { 0 };
    let causal_flag: c_int = if is_causal { 1 } else { 0 };

    // Q is [total_q, num_heads, head_dim] contiguous:
    //   q_row_stride = num_heads * head_dim
    //   q_head_stride = head_dim
    let q_row_stride = (num_heads * head_dim) as u32;
    let q_head_stride = head_dim as u32;

    // K/V cache: [num_blocks, block_size, num_kv_heads, head_dim]
    let kv_block_stride = (block_size * num_kv_heads * head_dim) as u32;
    let kv_row_stride = (num_kv_heads * head_dim) as u32;
    let kv_head_stride = head_dim as u32;

    run_mha_paged(
        q.raw_ptr() as *const c_void,
        k_cache.raw_ptr() as *const c_void,
        v_cache.raw_ptr() as *const c_void,
        out.raw_ptr() as *const c_void,
        softmax_lse.raw_ptr() as *const c_void,
        /* alibi_slopes */ std::ptr::null(),
        cu_seqlens_q.as_ptr::<i32>(),
        cu_seqlens_k.as_ptr::<i32>(),
        /* q_batch_stride */ 0,
        /* k_batch_stride */ kv_block_stride,
        /* v_batch_stride */ kv_block_stride,
        /* o_batch_stride */ 0,
        /* alibi_slopes_batch_stride */ 0,
        q_row_stride,
        kv_row_stride,
        kv_row_stride,
        /* o_row_stride */ q_row_stride,
        q_head_stride,
        kv_head_stride,
        kv_head_stride,
        /* o_head_stride */ q_head_stride,
        batch_size as u32,
        num_heads as u32,
        num_kv_heads as u32,
        head_dim as u32,
        head_size_rounded as u32,
        softmax_scale,
        max_seqlen_q as u32,
        max_seqlen_k as u32,
        seqlen_q_rounded as u32,
        seqlen_k_rounded as u32,
        is_bf16_flag,
        causal_flag,
        /* unpadded_lse */ 1,
        window_size_left,
        /* window_size_right */ if is_causal { 0 } else { -1 },
        softcap,
        block_table.as_ptr::<i32>(),
        max_blocks_per_seq as i64,
        block_size as c_int,
        /* num_splits */ 0,
        stream,
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
    arena: &mut ScratchArena,
    stream: cudarc::driver::sys::CUstream,
) -> (GpuTensor, GpuTensor, GpuTensor) {
    let num_tokens = qkv.dim(0);
    let total_dim = qkv.dim(1);

    debug_assert_eq!(total_dim, q_size + 2 * kv_size);

    let q = arena.alloc(&[num_tokens, num_q_heads, head_dim], qkv.dtype());
    let k = arena.alloc(&[num_tokens, num_kv_heads, head_dim], qkv.dtype());
    let v = arena.alloc(&[num_tokens, num_kv_heads, head_dim], qkv.dtype());

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
    arena: &mut ScratchArena,
    stream: CUstream,
) -> (GpuTensor, GpuTensor, GpuTensor) {
    let num_tokens = qkv.dim(0);
    let total_dim = qkv.dim(1);
    let rotary_dim = cos_sin_cache.dim(1) as i32;

    debug_assert_eq!(total_dim, q_size + 2 * kv_size);

    let q = arena.alloc(&[num_tokens, num_q_heads, head_dim], qkv.dtype());
    let k = arena.alloc(&[num_tokens, num_kv_heads, head_dim], qkv.dtype());
    let v = arena.alloc(&[num_tokens, num_kv_heads, head_dim], qkv.dtype());

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
    arena: &mut ScratchArena,
    stream: CUstream,
) -> GpuTensor {
    let batch_size = logits.dim(0) as c_int;
    let vocab_size = logits.dim(1) as c_int;
    let out = arena.alloc(&[batch_size as usize], DType::U32);

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
    arena: &mut ScratchArena,
    stream: CUstream,
) -> GpuTensor {
    let batch_size = logits.dim(0) as c_int;
    let vocab_size = logits.dim(1) as c_int;
    let out = arena.alloc(&[batch_size as usize], DType::U32);

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
