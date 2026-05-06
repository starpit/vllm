// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Bridge between Metal implementations and ferrite's Implementation trait.
//!
//! **DEPRECATED:** This module is being phased out in favor of the modular structure in `metal/`.
//!
//! All implementations have been moved to separate files:
//! - `metal/rmsnorm.rs` - MetalRmsNormImpl
//! - `metal/gemm.rs` - MetalGemmImpl
//! - `metal/fused_kernels.rs` - MetalFusedAddRmsNormImpl, MetalFusedGateUpSiluMulImpl
//! - `metal/attention.rs` - MetalAttentionImpl
//! - `metal/activation.rs` - MetalActivationImpl
//! - `metal/awq.rs` - MetalAwqImpl
//!
//! This file now serves as a compatibility shim, re-exporting implementations from their
//! new locations. It will be removed in a future version once all references are updated.

// Re-export all implementations from the modular structure
#[deprecated(since = "0.1.0", note = "Use `crate::metal::MetalRmsNormImpl` instead")]
pub use crate::metal::rmsnorm::MetalRmsNormImpl;

#[deprecated(since = "0.1.0", note = "Use `crate::metal::MetalGemmImpl` instead")]
pub use crate::metal::gemm::MetalGemmImpl;

#[deprecated(
    since = "0.1.0",
    note = "Use `crate::metal::MetalFusedAddRmsNormImpl` instead"
)]
pub use crate::metal::fused_kernels::MetalFusedAddRmsNormImpl;

#[deprecated(
    since = "0.1.0",
    note = "Use `crate::metal::MetalFusedGateUpSiluMulImpl` instead"
)]
pub use crate::metal::fused_kernels::MetalFusedGateUpSiluMulImpl;

#[deprecated(
    since = "0.1.0",
    note = "Use `crate::metal::MetalAttentionImpl` instead"
)]
pub use crate::metal::attention::MetalAttentionImpl;

#[deprecated(
    since = "0.1.0",
    note = "Use `crate::metal::MetalActivationImpl` instead"
)]
pub use crate::metal::activation::MetalActivationImpl;

#[deprecated(since = "0.1.0", note = "Use `crate::metal::MetalAwqImpl` instead")]
pub use crate::metal::awq::MetalAwqImpl;
