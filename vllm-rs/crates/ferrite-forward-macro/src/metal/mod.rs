// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal implementation adapters organized by kernel type.
//!
//! This module provides a modular structure for Metal `Implementation` trait adapters,
//! replacing the monolithic `metal_bridge.rs` with separate files per kernel category.

pub mod activation;
pub mod add;
pub mod affine_embed;
pub mod affine_qmm;
pub mod attention;
pub mod bias_add;
pub mod dataflow;
pub mod embed;
pub mod fused_kernels;
pub mod gemm;
pub mod moe;
pub mod mul;
pub mod nvfp4_qmm;
pub mod reshape;
pub mod rmsnorm;
pub mod rope;
pub mod scalar_mul;
pub mod softcap;
pub mod sub;
pub mod synth_gate_up_silu_mul;
pub mod synth_mlp_pre_down;
pub mod synth_pre_attn;

// Re-export the main implementation types
pub use activation::MetalActivationImpl;
pub use add::MetalAddImpl;
pub use affine_embed::MetalAffineEmbedImpl;
pub use affine_qmm::MetalAffineQmmImpl;
pub use attention::MetalAttentionImpl;
pub use bias_add::MetalBiasAddImpl;
pub use embed::MetalEmbedImpl;
pub use moe::{MetalFusedMoeImpl, MetalSharedFusedMoeImpl};
pub use mul::MetalMulImpl;
pub use nvfp4_qmm::MetalNvfp4QmmImpl;
pub use reshape::MetalReshapeImpl;
pub use rope::{MetalRopeAppendImpl, MetalRopeAppendInterleavedImpl};
pub use scalar_mul::MetalScalarMulImpl;
pub use softcap::MetalTanhSoftCapImpl;
pub use sub::MetalSubImpl;
