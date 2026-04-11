// SPDX-License-Identifier: Apache-2.0
//! Solver-driven forward pass dispatch.
//!
//! The `forward!` macro parses the model's forward structure from the
//! DSL body, runs the constraint solver at compile time, and emits
//! `solver_forward_layer()`.

use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::kernels;
use crate::kv_cache::KvCachePool;
use crate::model::llama::{LlamaDecoderLayer, RotaryCache};
use crate::tensor::{GpuTensor, TensorView};

// ── CUTLASS standalone GEMM FFI ─────────────────────────────────

macro_rules! cutlass_gemm_ffi {
    ($($name:ident),* $(,)?) => {
        #[cfg(feature = "cuda")]
        unsafe extern "C" {
            $(
                pub fn $name(
                    c: *mut u16, a: *const u16, b: *const u16,
                    m: i32, n: i32, k: i32,
                    alpha: f32, beta: f32, stream: u64,
                ) -> i32;
            )*
        }
    };
}

// Every (TB_M × TB_N, stages) config from cutlass_standalone_gemm.cu.
// The solver codegen constructs the name: cutlass_gemm_{M}x{N}_s{S}_launch.
cutlass_gemm_ffi!(
    cutlass_gemm_32x64_s4_launch,
    cutlass_gemm_32x64_s3_launch,
    cutlass_gemm_32x128_s4_launch,
    cutlass_gemm_32x128_s3_launch,
    cutlass_gemm_32x256_s3_launch,
    cutlass_gemm_64x64_s4_launch,
    cutlass_gemm_64x64_s3_launch,
    cutlass_gemm_64x128_s4_launch,
    cutlass_gemm_64x128_s3_launch,
    cutlass_gemm_128x64_s4_launch,
    cutlass_gemm_128x64_s3_launch,
    cutlass_gemm_128x128_s4_launch,
    cutlass_gemm_128x128_s3_launch,
    cutlass_gemm_128x256_s3_launch,
    cutlass_gemm_256x64_s4_launch,
    cutlass_gemm_256x64_s3_launch,
    // CUTLASS GEMV (M=1 specialization)
    cutlass_gemv_launch,
    // Legacy aliases (backward compat)
    cutlass_gemm_128x128_launch,
    cutlass_gemm_64x64_launch,
);

// ── TK fused MLP FFI ───────────────────────────────────────────

#[cfg(feature = "cuda")]
unsafe extern "C" {
    /// Grid-dispatched TK fused MLP: norm → gate GEMM+SiLU → up GEMM×gate → down GEMM+residual.
    /// One CTA per row (blockIdx.x), called once per layer with per-layer weight pointers.
    pub fn cp5_fused_mlp_solver_launch(
        hidden_ptr: u64,     // bf16 [batch_size, HD] — in/out (residual add)
        rms_gate_ptr: u64,   // bf16 [batch_size, HD] — scratch for normed activations
        silu_ptr: u64,       // bf16 [batch_size, ID] — scratch for gate*up
        mlp_norm_w_ptr: u64, // bf16 [1, HD] — norm weight (single layer)
        gate_w_ptr: u64,     // bf16 [ID, HD] — gate weight (single layer)
        up_w_ptr: u64,       // bf16 [ID, HD] — up weight (single layer)
        down_w_ptr: u64,     // bf16 [HD, ID] — down weight (single layer)
        rms_norm_eps: f32,
        batch_size: i32,
        stream: u64,
    ) -> i32;
}

// ── forward! expansion ──────────────────────────────────────────
//
// The DSL body describes the model structure. The solver runs at
// compile time for each model in the `models:` list and emits
// per-bucket dispatch functions.

use crate::layers::LinearLayer;

vllm_tk_macros::forward! {
    for layer in 0..NL {
        let normed = rmsnorm(hidden_states, attn_norm[layer]);
        let qkv = gemm(normed, qkv_weights[layer]);
        let (q, k, v) = rope_append(qkv, positions, kv_cache[layer]);
        let attn = attention_decode(q, k, v, kv_cache[layer], block_table);
        hidden_states = gemm_add(attn, o_proj[layer], hidden_states);

        let normed2 = rmsnorm(hidden_states, mlp_norm[layer]);
        let gate = silu(gemm(normed2, gate_weights[layer]));
        let up = gemm(normed2, up_weights[layer]);
        hidden_states = gemm_add(gate * up, down_proj[layer], hidden_states);
    }
    // Caller applies the final norm via fused_add_rms_norm_inplace
    // (already cuBLAS-free) and passes the normed hidden_states in.
    // The solver picks the optimal kernel for the [seq, vocab] projection.
    logits = gemm(hidden_states, lm_head);

    models: [
        { layers: 28, hidden: 3072, intermediate: 8192, heads: 24, kv_heads: 8, head_dim: 128, vocab: 128256 },
    ],
    target: l4_sm89,
    workloads: [1..4096],
}
