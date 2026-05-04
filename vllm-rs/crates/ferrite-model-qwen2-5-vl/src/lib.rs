// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Qwen2.5-VL vision tower. The text decoder is shared with plain Qwen2 /
//! Qwen2.5 / Qwen2-VL and lives in `ferrite-model-qwen2` (registered for
//! arch `Qwen2_5_VLForConditionalGeneration` via that crate's
//! `configs/qwen2.5-vl-3b.json`). This crate contributes only the
//! vision-side `MultimodalForward` registration via [`vision`]'s
//! `inventory::submit!` rows.
//!
//! Vision math deltas vs Qwen2-VL (sibling crate `ferrite-model-qwen2-vl`):
//!
//! - block norms + merger.ln_q: LayerNorm (weight + bias) → RMSNorm (weight only).
//! - block MLP: `fc1 → QuickGELU → fc2` → SwiGLU (`gate_proj`, `up_proj`,
//!   `down_proj` with bias=True), explicit `intermediate_size` (not embed×4).
//! - per-layer attention: 4-of-32 layers (`fullatt_block_indexes=[7,15,23,31]`)
//!   use full image-frame `cu_seqlens`; the other 28 use `cu_window_seqlens`
//!   bucketed by 112-px spatial windows.
//! - tokens are gather-permuted into window order on entry (and unpermuted
//!   after the merger) so window/full layers share the same flat tensor.
//!
//! Patch-flatten (pixels → `[L, C·T·P²]`) is identical, so [`vision::patches_from_normalized_chw`]
//! re-uses the qwen2-vl ordering.

#[cfg(feature = "cuda")]
pub mod vision;
