// SPDX-License-Identifier: Apache-2.0
//! Solver-driven forward pass dispatch.
//!
//! The `forward!` macro runs the constraint solver at compile time
//! and emits `solver_forward_layer()` — a per-layer dispatch function
//! that selects the optimal kernel mix based on `num_tokens`.
//!
//! The generated code uses the same types as the existing eager
//! forward pass: `OwnedTensor`, `GpuTensor`, `CublasHandle`, etc.
//! No FFI shim layer, no raw pointer juggling.

use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::kv_cache::KvCachePool;
use crate::model::llama::{LlamaDecoderLayer, RotaryCache};
use crate::tensor::TensorView;
use crate::{kernels, layers::LinearLayer};

// ── CUTLASS standalone GEMM FFI ─────────────────────────────────
//
// Linked from libcutlass_standalone_gemm.a, compiled by
// vllm-kernels-cuda's build.rs from csrc/cutlass_standalone_gemm.cu.

type CUstream = cudarc::driver::sys::CUstream;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    /// CUTLASS 128×128×32 bf16 GEMM: C = alpha*A@B^T + beta*C.
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

    /// CUTLASS 64×64×32 bf16 GEMM: C = alpha*A@B^T + beta*C.
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

// The forward! macro expands here, emitting:
// - solver_forward_layer() function
// - solver_layer_bucket_N() functions (one per workload bucket)
vllm_tk_macros::forward! {
    model: llama_3_2_1b,
    target: l4_sm89,
    workloads: [1..4096],
}
