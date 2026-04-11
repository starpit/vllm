// SPDX-License-Identifier: Apache-2.0
//! `forward!` proc macro — the compile-time entrypoint to Ferrite.
//!
//! At `cargo build` time, `forward!` parses a DSL describing the model's
//! forward pass, builds a typed [`ModelDag`](vllm_tk_macros_core::dag::ModelDag),
//! decomposes it into a tile graph, runs the constraint solver against
//! the implementation library, and emits Rust that calls individual
//! kernels via FFI.
//!
//! The runtime does not see the solver. The generated
//! `solver_forward_layer()` / `solver_forward_lm_head()` functions are
//! straight-line kernel dispatches behind a `match` on `num_tokens`.

extern crate proc_macro;

use proc_macro::TokenStream;
use vllm_tk_macros_core::lowering::backend::{codegen, compile_dsl};

/// Generates a solver-driven forward function.
///
/// Emits `solver_forward_layer()` + `solver_forward_lm_head()` —
/// drop-in replacements for the per-layer and post-loop GEMM paths
/// that use the constraint solver's optimal kernel mix for each
/// workload bucket.
///
/// ```ignore
/// // Fully specialized: solver runs at compile time, zero startup cost.
/// forward! {
///     for layer in 0..NL {
///         let normed = rmsnorm(hidden_states, attn_norm[layer]);
///         // ... full forward body ...
///     }
///     logits = gemm(hidden_states, lm_head);
///     models: [{ layers: 28, hidden: 3072, intermediate: 8192, ... }],
///     target: l4_sm89,
///     workloads: [1..4096],
/// }
/// ```
///
/// The generated per-layer function matches on `num_tokens` and
/// dispatches to per-bucket implementations with solver-selected
/// kernels (CUTLASS, GEMV, FlashInfer, vllm-rs fused ops).
#[proc_macro]
pub fn forward(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();

    let def: compile_dsl::ForwardDef = match syn::parse2(input2) {
        Ok(d) => d,
        Err(e) => return e.to_compile_error().into(),
    };

    codegen::generate(&def).into()
}
