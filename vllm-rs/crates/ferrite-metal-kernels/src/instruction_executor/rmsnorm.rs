// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! RMSNorm instruction recording for Metal ICB.

use super::{RecordingContext, dispatch_1d};
use crate::shader_cache::ShaderCache;
use metal::{MTLSize, MTLResourceOptions};
use std::sync::Arc;

/// Record an RMSNorm kernel dispatch into the ICB.
///
/// # Arguments
/// * `ctx` - Recording context with device and ICB
/// * `input_buffer` - Input tensor buffer
/// * `output_buffer` - Output tensor buffer
/// * `weight_buffer` - RMSNorm weight buffer
/// * `num_tokens` - Number of tokens (M dimension)
/// * `hidden_size` - Hidden size (N dimension)
/// * `eps` - Epsilon for numerical stability
/// * `dtype` - Data type ("fp16" or "bf16")
///
/// # Returns
/// `Ok(())` on success, `Err(String)` on shader compilation failure
pub fn record_rmsnorm(
    ctx: &mut RecordingContext,
    input_buffer: &metal::Buffer,
    output_buffer: &metal::Buffer,
    weight_buffer: &metal::Buffer,
    num_tokens: usize,
    hidden_size: usize,
    eps: f32,
    dtype: &str,
) -> Result<(), String> {
    // Get or compile shader
    let shader_cache = ShaderCache::new((*ctx.device).clone())
        .map_err(|e| format!("Failed to create shader cache: {:?}", e))?;
    let kernel_name = match dtype {
        "fp16" => "rmsnorm_f16",
        "bf16" => "rmsnorm_bf16",
        _ => return Err(format!("Unsupported dtype: {}", dtype)),
    };
    
    let pipeline = shader_cache
        .get_pipeline(kernel_name)
        .map_err(|e| format!("Failed to compile RMSNorm shader: {:?}", e))?;

    // Calculate dispatch size
    // Each thread processes one token (row)
    let (threadgroups, threads_per_threadgroup) = dispatch_1d(num_tokens, 256);

    // Prepare constant buffer for eps and hidden_size
    let constants = [eps, hidden_size as f32];
    let constants_buffer = ctx.device.new_buffer_with_data(
        constants.as_ptr() as *const _,
        std::mem::size_of_val(&constants) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Record the dispatch
    ctx.record_compute_dispatch(
        &pipeline,
        &[
            (input_buffer, 0, 0),      // [[buffer(0)]]
            (output_buffer, 0, 1),     // [[buffer(1)]]
            (weight_buffer, 0, 2),     // [[buffer(2)]]
            (&constants_buffer, 0, 3), // [[buffer(3)]]
        ],
        threadgroups,
        threads_per_threadgroup,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect_device;

    #[test]
    fn test_rmsnorm_recording() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = RecordingContext::new(Arc::new(device.device.clone()), 10)
            .expect("Failed to create recording context");

        // Create dummy buffers
        let input = device.device.new_buffer(
            1024 * 4096 * 2, // 1024 tokens × 4096 hidden × 2 bytes (fp16)
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output = device.device.new_buffer(
            1024 * 4096 * 2,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let weight = device.device.new_buffer(
            4096 * 2, // 4096 hidden × 2 bytes
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Record RMSNorm dispatch
        let result = record_rmsnorm(
            &mut ctx,
            &input,
            &output,
            &weight,
            1024,  // num_tokens
            4096,  // hidden_size
            1e-6,  // eps
            "fp16",
        );

        assert!(result.is_ok(), "RMSNorm recording failed: {:?}", result);
        assert_eq!(ctx.command_count(), 1, "Should record exactly 1 command");
    }
}
