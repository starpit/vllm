// SPDX-License-Identifier: Apache-2.0
//! Rust launch wrappers for CUDA kernels the ferrite-stencil emitter
//! produces (and, at step 3a, the hand-picked smoke kernel that
//! proves the build plumbing).
//!
//! Step 3b: exposes `launch_smoke_sm89` — a single fixed-shape FA2
//! prefill with inputs `Q [SEQ_Q, HD]`, `K [SEQ_K, HD]`, `V [SEQ_K, HD]`
//! and output `O [SEQ_Q, HD]`, with `SEQ_Q = 8`, `SEQ_K = 16`,
//! `HD = 8`. Matches the constants in
//! `crates/ferrite-stencil/csrc/stencil_smoke_sm89.cu`. Later steps
//! replace this hand-picked shape with emitter-generated variants.

#![cfg_attr(not(feature = "cuda"), allow(unused))]

#[cfg(feature = "cuda")]
mod ffi;
#[cfg(feature = "cuda")]
pub use ffi::*;
