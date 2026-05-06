// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal implementation adapters organized by kernel type.
//!
//! This module provides a modular structure for Metal `Implementation` trait adapters,
//! replacing the monolithic `metal_bridge.rs` with separate files per kernel category.

pub mod activation;
pub mod add;
pub mod attention;
pub mod awq;
pub mod bias_add;
pub mod embed;
pub mod fused_kernels;
pub mod gemm;
pub mod mul;
pub mod reshape;
pub mod rmsnorm;
pub mod rope;
pub mod scalar_mul;
pub mod softcap;
pub mod sub;

// Re-export the main implementation types
pub use activation::MetalActivationImpl;
pub use add::MetalAddImpl;
pub use attention::MetalAttentionImpl;
pub use awq::MetalAwqImpl;
pub use bias_add::MetalBiasAddImpl;
pub use embed::MetalEmbedImpl;
pub use fused_kernels::{MetalFusedAddRmsNormImpl, MetalFusedGateUpSiluMulImpl};
pub use gemm::MetalGemmImpl;
pub use mul::MetalMulImpl;
pub use reshape::MetalReshapeImpl;
pub use rmsnorm::MetalRmsNormImpl;
pub use rope::{MetalRopeAppendImpl, MetalRopeAppendInterleavedImpl};
pub use scalar_mul::MetalScalarMulImpl;
pub use softcap::MetalTanhSoftCapImpl;
pub use sub::MetalSubImpl;
