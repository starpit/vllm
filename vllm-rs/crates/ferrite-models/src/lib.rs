// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Ferrite model architectures. Each module carries one DSL body
//! via `#[forward]`; the compiler fans out per-model specializations
//! across `model_architectures/<arch>/*.json`.
//!
//! Qwen2's body is identical to Llama's — the bias on Qwen2's QKV
//! projections is handled at weight-load time by
//! `LinearLayer::load_dense_concat`, which auto-detects per-source
//! `.bias` tensors and packs them into the fused `LinearLayer`;
//! `Linear::forward` then lights up `cublas.gemm_bias`'s epilog
//! automatically. No DSL-level `bias_add` op is needed, and no
//! per-arch Impl addition is required.

pub mod gemma2;
pub mod llama;
pub mod qwen2;
