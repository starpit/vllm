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
///
/// **Do not construct directly** — use the typed wrappers below which enforce
/// the correct `(b, d, r, c)` mapping for each GL type at compile time.
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
    /// Escape hatch for VM state tensors (barriers, instructions, timings)
    /// that don't map to a standard GL type.
    pub fn raw(ptr: u64, shape: &[usize]) -> Self {
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

// ── Typed tensor descriptors ──
//
// Each type mirrors one GL type from llama_sm89.cuh. They are
// #[repr(transparent)] over TkTensorArg so the ABI is identical,
// but the Rust type system prevents passing a WeightArg where a
// NormWeightArg is expected.

macro_rules! typed_tensor_arg {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug)]
        pub struct $name(TkTensorArg);

        // Allow conversion to raw TkTensorArg when needed
        impl From<$name> for TkTensorArg {
            fn from(t: $name) -> Self { t.0 }
        }
    };
}

typed_tensor_arg!(
    /// `weights_t = gl<bf16, 1, -1, -1, hidden_dim>`
    /// Also used for `weights_big_t` (same shape, different C dim).
    WeightArg
);
impl WeightArg {
    /// Create from `[1, num_layers, output_dim, input_dim]`.
    pub fn new(ptr: u64, num_layers: usize, output_dim: usize, input_dim: usize) -> Self {
        Self(TkTensorArg {
            ptr, b: 1,
            d: num_layers as i32,
            r: output_dim as i32,
            c: input_dim as i32,
        })
    }
}

typed_tensor_arg!(
    /// `norm_weights_t = gl<bf16, 1, 1, -1, hidden_dim>`
    NormWeightArg
);
impl NormWeightArg {
    /// Create from `[1, 1, num_layers, hidden_dim]`.
    pub fn new(ptr: u64, num_layers: usize, hidden_dim: usize) -> Self {
        Self(TkTensorArg {
            ptr, b: 1, d: 1,
            r: num_layers as i32,
            c: hidden_dim as i32,
        })
    }
}

typed_tensor_arg!(
    /// `activations_t = gl<bf16, 1, 1, -1, dim>`
    /// Also used for `activations_big_t`.
    ActivationArg
);
impl ActivationArg {
    /// Create from `[1, 1, batch, dim]`.
    pub fn new(ptr: u64, batch: usize, dim: usize) -> Self {
        Self(TkTensorArg {
            ptr, b: 1, d: 1,
            r: batch as i32,
            c: dim as i32,
        })
    }
}

typed_tensor_arg!(
    /// `logits_t = gl<bf16, 1, 1, -1, -1>`
    LogitsArg
);
impl LogitsArg {
    /// Create from `[1, 1, batch, vocab_size]`.
    pub fn new(ptr: u64, batch: usize, vocab_size: usize) -> Self {
        Self(TkTensorArg {
            ptr, b: 1, d: 1,
            r: batch as i32,
            c: vocab_size as i32,
        })
    }
}

typed_tensor_arg!(
    /// `kv_cache_t = gl<bf16, -1, -1, num_kv_heads, head_dim>`
    KvCacheArg
);
impl KvCacheArg {
    /// Create from `[total_pages, page_size, num_kv_heads, head_dim]`.
    pub fn new(
        ptr: u64, total_pages: usize, page_size: usize,
        num_kv_heads: usize, head_dim: usize,
    ) -> Self {
        Self(TkTensorArg {
            ptr,
            b: total_pages as i32,
            d: page_size as i32,
            r: num_kv_heads as i32,
            c: head_dim as i32,
        })
    }
}

typed_tensor_arg!(
    /// `rope_table_t = gl<float, 1, 1, -1, head_dim>`
    RopeArg
);
impl RopeArg {
    /// Create from `[1, 1, max_positions, head_dim]`.
    pub fn new(ptr: u64, max_positions: usize, head_dim: usize) -> Self {
        Self(TkTensorArg {
            ptr, b: 1, d: 1,
            r: max_positions as i32,
            c: head_dim as i32,
        })
    }
}

typed_tensor_arg!(
    /// `int32_vector_t = gl<int, 1, 1, 1, -1>`
    IntVecArg
);
impl IntVecArg {
    /// Create from `[1, 1, 1, len]`.
    pub fn new(ptr: u64, len: usize) -> Self {
        Self(TkTensorArg {
            ptr, b: 1, d: 1, r: 1,
            c: len as i32,
        })
    }
}

typed_tensor_arg!(
    /// `barriers = gl<uint, -1, -1, -1, -1>`
    BarrierArg
);
impl BarrierArg {
    /// Create from `[num_layers, num_ops, batch_blocks, cols]`.
    pub fn new(
        ptr: u64, num_layers: usize, num_ops: usize,
        batch_blocks: usize, cols: usize,
    ) -> Self {
        Self(TkTensorArg {
            ptr,
            b: num_layers as i32,
            d: num_ops as i32,
            r: batch_blocks as i32,
            c: cols as i32,
        })
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
    use super::*;

    // FFI launch functions for each op's test kernel.
    // Parameters use typed wrappers — the compiler rejects mismatched types.
    macro_rules! declare_test_launch {
        ($name:ident) => {
            unsafe extern "C" {
                pub fn $name(
                    // VM state (3 tensors)
                    bar: BarrierArg,
                    instructions: TkTensorArg, // no standard GL type
                    timings: TkTensorArg,       // no standard GL type
                    // Weights (9 tensors)
                    qkv_w: WeightArg,
                    attn_norm_w: NormWeightArg,
                    o_w: WeightArg,
                    mlp_norm_w: NormWeightArg,
                    up_w: WeightArg,
                    gate_w: WeightArg,
                    down_w: WeightArg, // weights_big_t, same repr
                    lm_norm_w: NormWeightArg,
                    lm_w: WeightArg,
                    // KV cache (2 tensors)
                    k_cache: KvCacheArg,
                    v_cache: KvCacheArg,
                    // RoPE (2 tensors)
                    rope_cos: RopeArg,
                    rope_sin: RopeArg,
                    // Activations (8 tensors)
                    hidden: ActivationArg,
                    rms_rope: ActivationArg,
                    rms_gate: ActivationArg,
                    q_post: ActivationArg,
                    attn_out: ActivationArg,
                    silu: ActivationArg, // activations_big_t, same repr
                    rms_lm: ActivationArg,
                    logits_arg: LogitsArg,
                    // Paged KV metadata — decode (5 tensors)
                    pos_ids: IntVecArg,
                    kv_indptr: IntVecArg,
                    kv_indices: IntVecArg,
                    kv_last_page: IntVecArg,
                    kv_append: IntVecArg,
                    // Paged KV metadata — prefill (4 tensors)
                    prefill_qo_indptr: IntVecArg,
                    prefill_kv_indptr: IntVecArg,
                    prefill_kv_indices: IntVecArg,
                    prefill_kv_last_page_len: IntVecArg,
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
    declare_test_launch!(fused_rmsnorm_gemm_launch);
    declare_test_launch!(fused_mlp_launch);
    declare_test_launch!(fused_layer_launch);
    declare_test_launch!(fused_full_layer_launch);
    declare_test_launch!(fused_multi_layer_launch);
    declare_test_launch!(inline_attention_decode_launch);
    declare_test_launch!(fused_multi_sm_launch);
}
