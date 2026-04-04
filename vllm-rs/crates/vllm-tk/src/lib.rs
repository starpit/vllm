// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]
//! ThunderKittens KVM megakernel for sm89 (L40S / RTX 4090).
//!
//! Compiles the TK kernel via cudaforge in build.rs and exposes a safe Rust
//! launch wrapper. With `cuda` feature, depends on `vllm-cuda` for `GpuTensor`,
//! `GpuWeights`, and weight loading.

pub mod ffi;
pub mod scheduler;
#[cfg(feature = "cuda")]
pub mod weights;
pub mod worker;

pub use ffi::TkTensorArg;

/// Launch the TK KVM LLaMA-1B megakernel (full forward pass).
///
/// All `TkTensorArg` values must wrap valid device pointers with shapes
/// matching the kernel's expectations. `stream` must be a valid CUDA stream
/// (pass 0 / null for the default stream).
///
/// # Safety
/// Caller must ensure all device pointers are valid and the stream is live.
#[allow(clippy::too_many_arguments)]
pub unsafe fn tk_llama_forward(
    // VM state
    bar: TkTensorArg,
    instructions: TkTensorArg,
    timings: TkTensorArg,
    // Weights
    qkv_w: TkTensorArg,
    attn_norm_w: TkTensorArg,
    o_w: TkTensorArg,
    mlp_norm_w: TkTensorArg,
    up_w: TkTensorArg,
    gate_w: TkTensorArg,
    down_w: TkTensorArg,
    lm_norm_w: TkTensorArg,
    lm_w: TkTensorArg,
    // KV cache
    k_cache: TkTensorArg,
    v_cache: TkTensorArg,
    // RoPE
    rope_cos: TkTensorArg,
    rope_sin: TkTensorArg,
    // Activations
    hidden: TkTensorArg,
    rms_rope: TkTensorArg,
    rms_gate: TkTensorArg,
    q_post: TkTensorArg,
    attn_out: TkTensorArg,
    silu: TkTensorArg,
    rms_lm: TkTensorArg,
    logits: TkTensorArg,
    // Paged KV metadata — decode
    pos_ids: TkTensorArg,
    kv_indptr: TkTensorArg,
    kv_indices: TkTensorArg,
    kv_last_page: TkTensorArg,
    kv_append: TkTensorArg,
    // Paged KV metadata — prefill
    prefill_qo_indptr: TkTensorArg,
    prefill_kv_indptr: TkTensorArg,
    prefill_kv_indices: TkTensorArg,
    prefill_kv_last_page_len: TkTensorArg,
    // Scalars
    attn_scale: f32,
    rms_norm_eps: f32,
    num_pages: i32,
    batch_size: i32,
    num_prefill_tokens: i32,
    // Stream
    stream: u64,
) {
    unsafe {
        ffi::tk_llama_1b_launch(
            bar,
            instructions,
            timings,
            qkv_w,
            attn_norm_w,
            o_w,
            mlp_norm_w,
            up_w,
            gate_w,
            down_w,
            lm_norm_w,
            lm_w,
            k_cache,
            v_cache,
            rope_cos,
            rope_sin,
            hidden,
            rms_rope,
            rms_gate,
            q_post,
            attn_out,
            silu,
            rms_lm,
            logits,
            pos_ids,
            kv_indptr,
            kv_indices,
            kv_last_page,
            kv_append,
            prefill_qo_indptr,
            prefill_kv_indptr,
            prefill_kv_indices,
            prefill_kv_last_page_len,
            attn_scale,
            rms_norm_eps,
            num_pages,
            batch_size,
            num_prefill_tokens,
            stream,
        );
    }
}
