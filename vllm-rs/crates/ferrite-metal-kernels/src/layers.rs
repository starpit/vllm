// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal layer types (Phase 5.6: minimal stubs for compilation)
//!
//! These are placeholder types to allow the macro-generated code to compile.
//! Actual implementations will be added in later phases.

use metal::Buffer;

/// Embedding layer (stub)
#[derive(Debug)]
pub struct Embedding {
    pub weight: Buffer,
}

/// RMS normalization layer (stub)
#[derive(Debug)]
pub struct RmsNorm {
    pub weight: Buffer,
    pub eps: f32,
}

/// Layer normalization with bias (stub)
#[derive(Debug)]
pub struct LayerNorm {
    pub weight: Buffer,
    pub bias: Option<Buffer>,
    pub eps: f32,
}

/// Dense linear layer (stub)
#[derive(Debug)]
pub struct LinearLayer {
    pub weight: Buffer,
    pub bias: Option<Buffer>,
}

/// Marlin quantized linear layer (stub)
#[derive(Debug)]
pub struct MarlinLinear {
    pub weight: Buffer,
}

/// BNB 4-bit quantized linear layer (stub)
#[derive(Debug)]
pub struct Bnb4bitLinear {
    pub weight: Buffer,
}

/// FP8 linear layer (stub)
#[derive(Debug)]
pub struct Fp8Linear {
    pub weight: Buffer,
}

/// FP8 block linear layer (stub)
#[derive(Debug)]
pub struct Fp8BlockLinear {
    pub weight: Buffer,
}

/// DeepSeek V2 MoE layer (stub)
#[derive(Debug)]
pub struct DeepSeekV2MoELayer {
    pub gate: Buffer,
}

/// DeepSeek V2 FP8 block MoE layer (stub)
#[derive(Debug)]
pub struct DeepSeekV2Fp8BlockMoELayer {
    pub gate: Buffer,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_types_compile() {
        // Placeholder test - actual layer operations require Metal device
        assert!(true, "Metal layer types compile");
    }
}