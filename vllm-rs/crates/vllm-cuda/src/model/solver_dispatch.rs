// SPDX-License-Identifier: Apache-2.0
//! Solver-driven forward pass dispatch.
//!
//! The `forward!` macro parses the model's forward structure from the
//! DSL body, runs the constraint solver at compile time, and emits
//! `solver_forward_layer()`.

use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::kv_cache::KvCachePool;
use crate::model::llama::{LlamaDecoderLayer, RotaryCache};
use crate::tensor::{GpuTensor, TensorView};
use crate::{kernels, layers::LinearLayer};

// ── CUTLASS standalone GEMM FFI ─────────────────────────────────

type CUstream = cudarc::driver::sys::CUstream;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    pub fn cutlass_gemm_128x128_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;

    pub fn cutlass_gemm_64x64_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
}

// ── forward! expansion ──────────────────────────────────────────
//
// The DSL body describes the model structure. The solver runs at
// compile time for each model in the `models:` list and emits
// per-bucket dispatch functions.

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

    models: [
        { layers: 28, hidden: 3072, intermediate: 8192, heads: 24, kv_heads: 8, head_dim: 128 },
    ],
    target: l4_sm89,
    workloads: [1..4096],
}
