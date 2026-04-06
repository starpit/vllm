// SPDX-License-Identifier: Apache-2.0
//! TK op-level test harness.
//!
//! Each TK op (rms_norm, gemm, attention, etc.) gets a standalone CUDA kernel
//! that can be launched independently for testing. The build.rs generates per-op
//! `.cu` files and compiles them into a static library.
//!
//! With `--features cuda`:
//! - FFI declarations for `test_{op_name}_launch(...)` are available
//! - GPU tests in `tests/op_tests.rs` exercise each op in isolation

/// The flat tensor descriptor passed to CUDA launch wrappers via FFI.
/// Must match the `TkTensorArg` struct in the generated CUDA code exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkTensorArg {
    pub ptr: u64,
    pub b: i32,
    pub d: i32,
    pub r: i32,
    pub c: i32,
}

impl TkTensorArg {
    /// Create a TkTensorArg from a device pointer and shape.
    pub fn new(ptr: u64, shape: &[usize]) -> Self {
        let (b, d, r, c) = match shape.len() {
            1 => (1, 1, 1, shape[0] as i32),
            2 => (1, 1, shape[0] as i32, shape[1] as i32),
            3 => (1, shape[0] as i32, shape[1] as i32, shape[2] as i32),
            4 => (
                shape[0] as i32,
                shape[1] as i32,
                shape[2] as i32,
                shape[3] as i32,
            ),
            _ => panic!("TkTensorArg: unsupported shape rank {}", shape.len()),
        };
        Self { ptr, b, d, r, c }
    }

    /// Create a zero/null tensor arg (for unused slots).
    pub fn null() -> Self {
        Self {
            ptr: 0,
            b: 1,
            d: 1,
            r: 1,
            c: 1,
        }
    }
}

/// Op names that have compiled test kernels.
pub const OP_NAMES: &[&str] = &[
    "attn_norm",
    "qkv_rope_append",
    "attention_decode",
    "o_proj_residual",
    "mlp_norm",
    "gate_silu",
    "up_matmul",
    "down_proj_residual",
    "lm_head_norm",
    "lm_head",
    "attention_prefill",
];

#[cfg(feature = "cuda")]
pub mod ffi {
    use super::TkTensorArg;

    // FFI launch functions for each op's test kernel.
    // Each has the same signature as the full megakernel launch wrapper.
    macro_rules! declare_test_launch {
        ($name:ident) => {
            unsafe extern "C" {
                pub fn $name(
                    // VM state (3 tensors)
                    bar: TkTensorArg,
                    instructions: TkTensorArg,
                    timings: TkTensorArg,
                    // Weights (9 tensors)
                    qkv_w: TkTensorArg,
                    attn_norm_w: TkTensorArg,
                    o_w: TkTensorArg,
                    mlp_norm_w: TkTensorArg,
                    up_w: TkTensorArg,
                    gate_w: TkTensorArg,
                    down_w: TkTensorArg,
                    lm_norm_w: TkTensorArg,
                    lm_w: TkTensorArg,
                    // KV cache (2 tensors)
                    k_cache: TkTensorArg,
                    v_cache: TkTensorArg,
                    // RoPE (2 tensors)
                    rope_cos: TkTensorArg,
                    rope_sin: TkTensorArg,
                    // Activations (8 tensors)
                    hidden: TkTensorArg,
                    rms_rope: TkTensorArg,
                    rms_gate: TkTensorArg,
                    q_post: TkTensorArg,
                    attn_out: TkTensorArg,
                    silu: TkTensorArg,
                    rms_lm: TkTensorArg,
                    logits_arg: TkTensorArg,
                    // Paged KV metadata — decode (5 tensors)
                    pos_ids: TkTensorArg,
                    kv_indptr: TkTensorArg,
                    kv_indices: TkTensorArg,
                    kv_last_page: TkTensorArg,
                    kv_append: TkTensorArg,
                    // Paged KV metadata — prefill (4 tensors)
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
                    num_layers: i32,
                    // CUDA stream
                    stream: u64,
                ) -> i32;
            }
        };
    }

    declare_test_launch!(test_attn_norm_launch);
    declare_test_launch!(test_mlp_norm_launch);
    declare_test_launch!(test_lm_head_norm_launch);
    declare_test_launch!(test_qkv_rope_append_launch);
    declare_test_launch!(test_attention_decode_launch);
    declare_test_launch!(test_attention_prefill_launch);
    declare_test_launch!(test_o_proj_residual_launch);
    declare_test_launch!(test_gate_silu_launch);
    declare_test_launch!(test_up_matmul_launch);
    declare_test_launch!(test_down_proj_residual_launch);
    declare_test_launch!(test_lm_head_launch);

    // Inline kernels (no KVM protocol — static tile pipeline)
    declare_test_launch!(inline_rmsnorm_launch);
    declare_test_launch!(inline_gemm_launch);
}
