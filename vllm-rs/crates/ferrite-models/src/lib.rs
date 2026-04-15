// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Ferrite model architectures.
//!
//! `llama.rs` uses the new `#[forward]` macro (ferrite-forward).
//!
//! `qwen2.rs` is gated off: its `ferrite_macros::forward!{}` runs
//! the pre-DP backtrack-CP solver which dominates compile time
//! (minutes per edit). It's blocked on HANDOFF.md Step C
//! (`CublasFusedQkvGemmWithBiasImpl`, gap #5) before it can move
//! to `#[forward]`. Until then, vllm-cuda routes Qwen2 through the
//! hand-written `Qwen2ForCausalLM` path — nothing consumes this
//! file's output.

pub mod llama;
// pub mod qwen2;  // re-enable when Step C lands.
