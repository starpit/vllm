// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Qwen2-VL vision tower. The text decoder is shared with plain Qwen2 and
//! lives in `ferrite-model-qwen2` (registered for arch
//! `Qwen2VLForConditionalGeneration` via that crate's `configs/qwen2-vl-2b.json`).
//! This crate contributes only the vision-side `MultimodalForward`
//! registration via [`vision`]'s `inventory::submit!` rows.
//!
//! Qwen2.5-VL has materially different vision math (window attention,
//! RMSNorm in vision blocks, revised 2D RoPE) and lives in its own crate
//! (`ferrite-model-qwen2-5-vl`) when that work lands. Both VL crates rely
//! on `ferrite-model-qwen2` for the shared text decoder.

#[cfg(feature = "cuda")]
pub mod vision;
