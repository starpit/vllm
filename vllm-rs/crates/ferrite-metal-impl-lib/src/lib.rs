// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal Implementation Library - Reserved for Future Use
//!
//! This crate is currently a placeholder. Metal `Implementation` trait adapters
//! are located in `ferrite-forward-macro/src/metal_bridge.rs` due to dependency
//! constraints (proc-macro crates cannot be dependencies of regular crates).
//!
//! ## Why This Crate Exists
//!
//! Originally intended to house Metal implementations separately from the proc-macro,
//! but blocked by circular dependency:
//! - Metal impls need `ferrite-forward-macro::Implementation` trait
//! - But proc-macro crates can't be dependencies
//!
//! ## Future Use Cases
//!
//! This crate may be repurposed for:
//! 1. Shared Metal utilities (device management, buffer helpers)
//! 2. Metal-specific type conversions
//! 3. Common Metal kernel infrastructure
//!
//! ## Current Implementation Location
//!
//! All Metal `Implementation` trait adapters are in:
//! - `ferrite-forward-macro/src/metal_bridge.rs` (11 implementations)
//! - Registered in `ferrite-forward-macro/src/impl_lib.rs::starter_library()`
//!
//! See `ferrite-forward-macro/src/metal_bridge.rs` for:
//! - MetalRmsNormImpl (fp16, bf16)
//! - MetalGemmImpl (fp16, fp32)
//! - MetalFusedAddRmsNormImpl (fp16, bf16)
//! - MetalFusedGateUpSiluMulImpl (fp16, bf16, gelu_fp16)

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_test() {
        // This crate is currently unused but reserved for future Metal utilities
        assert!(true);
    }
}
