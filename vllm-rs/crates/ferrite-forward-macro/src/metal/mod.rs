// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal implementation adapters organized by kernel type.
//!
//! This module provides a modular structure for Metal `Implementation` trait adapters,
//! replacing the monolithic `metal_bridge.rs` with separate files per kernel category.

pub mod attention;
pub mod activation;
pub mod awq;
pub mod rmsnorm;
pub mod gemm;
pub mod fused_kernels;
pub mod reshape;
pub mod add;
pub mod scalar_mul;
pub mod embed;
pub mod rope;
pub mod mul;
pub mod bias_add;
pub mod softcap;
pub mod sub;

// Re-export the main implementation types
pub use attention::MetalAttentionImpl;
pub use activation::MetalActivationImpl;
pub use awq::MetalAwqImpl;
pub use rmsnorm::MetalRmsNormImpl;
pub use gemm::MetalGemmImpl;
pub use fused_kernels::{MetalFusedAddRmsNormImpl, MetalFusedGateUpSiluMulImpl};
pub use reshape::MetalReshapeImpl;
pub use add::MetalAddImpl;
pub use scalar_mul::MetalScalarMulImpl;
pub use embed::MetalEmbedImpl;
pub use rope::{MetalRopeAppendImpl, MetalRopeAppendInterleavedImpl};
pub use mul::MetalMulImpl;
pub use bias_add::MetalBiasAddImpl;
pub use softcap::MetalTanhSoftCapImpl;
pub use sub::MetalSubImpl;
