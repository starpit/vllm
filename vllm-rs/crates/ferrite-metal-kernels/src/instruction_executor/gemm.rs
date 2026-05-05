// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! GEMM instruction recording for Metal ICB.
//!
//! Records Metal Performance Shaders (MPS) GEMM operations into ICB.

use super::RecordingContext;

/// Record a GEMM kernel dispatch into the ICB.
///
/// Uses Metal Performance Shaders (MPS) for matrix multiplication.
///
/// # Arguments
/// * `ctx` - Recording context with device and ICB
/// * `input_buffer` - Input activation buffer [M, K]
/// * `weight_buffer` - Weight matrix buffer [N, K] (transposed)
/// * `output_buffer` - Output buffer [M, N]
/// * `m` - Number of rows in input
/// * `k` - Inner dimension (input cols = weight cols)
/// * `n` - Number of rows in weight (output cols)
/// * `dtype` - Data type ("fp16" or "fp32")
///
/// # Returns
/// `Ok(())` on success, `Err(String)` on failure
pub fn record_gemm(
    ctx: &mut RecordingContext,
    input_buffer: &metal::Buffer,
    weight_buffer: &metal::Buffer,
    output_buffer: &metal::Buffer,
    m: usize,
    k: usize,
    n: usize,
    dtype: &str,
) -> Result<(), String> {
    // TODO: Implement MPS GEMM recording
    // MPS doesn't directly support ICB recording - need to investigate alternatives:
    // 1. Use custom Metal GEMM shader (less optimal but ICB-compatible)
    // 2. Record MPS command as a separate command buffer (breaks ICB model)
    // 3. Use Metal's built-in matrix multiplication (if available in ICB)

    Err("GEMM ICB recording not yet implemented".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "GEMM recording not yet implemented"]
    fn test_gemm_recording() {
        // TODO: Implement test once GEMM recording is ready
    }
}
