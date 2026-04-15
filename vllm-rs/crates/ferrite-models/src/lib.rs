// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Ferrite model architectures.
//!
//! `llama.rs` uses the new `#[forward]` macro (ferrite-forward).
//! `qwen2.rs` still uses the legacy `ferrite_macros::forward!{}`;
//! it's blocked on Step C (bias-fused QKV Impl) per HANDOFF.md.

pub mod llama;
#[cfg(feature = "cuda")]
pub mod qwen2;
