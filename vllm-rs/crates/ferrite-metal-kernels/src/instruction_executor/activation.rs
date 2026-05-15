// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Activation function instruction recording for Metal ICB.

use super::{dispatch_1d, RecordingContext};
use crate::shader_cache::ShaderCache;
use metal::MTLResourceOptions;
use std::sync::Arc;

/// Record an activation function kernel dispatch into the ICB.
///
/// # Arguments
/// * `ctx` - Recording context with device and ICB
/// * `input_buffer` - Input tensor buffer
/// * `output_buffer` - Output tensor buffer
/// * `num_elements` - Total number of elements to process
/// * `activation_type` - Type of activation ("silu", "gelu", "gelu_tanh", "gelu_quick", "fatrelu")
/// * `dtype` - Data type ("fp16" or "bf16")
/// * `threshold` - Threshold parameter (only used for FatReLU)
///
/// # Returns
/// `Ok(())` on success, `Err(String)` on shader compilation failure
pub fn record_activation(
    ctx: &mut RecordingContext,
    input_buffer: &metal::Buffer,
    output_buffer: &metal::Buffer,
    num_elements: u64,
    activation_type: &str,
    dtype: &str,
    threshold: f32,
) -> Result<(), String> {
    // Get or compile shader
    let shader_cache = ShaderCache::new(ctx.device.as_ref().clone())
        .map_err(|e| format!("Failed to create shader cache: {:?}", e))?;

    let kernel_name = match (activation_type, dtype) {
        ("silu", "fp16") => "silu_f16",
        ("silu", "bf16") => "silu_bf16",
        ("gelu", "fp16") => "gelu_f16",
        ("gelu", "bf16") => "gelu_bf16",
        ("gelu_tanh", "fp16") => "gelu_tanh_f16",
        ("gelu_tanh", "bf16") => "gelu_tanh_bf16",
        ("gelu_quick", "fp16") => "gelu_quick_f16",
        ("gelu_quick", "bf16") => "gelu_quick_bf16",
        ("fatrelu", "fp16") => "fatrelu_f16",
        ("fatrelu", "bf16") => "fatrelu_bf16",
        _ => {
            return Err(format!(
                "Unsupported activation/dtype: {}/{}",
                activation_type, dtype
            ))
        }
    };

    let pipeline = shader_cache
        .get_pipeline(kernel_name)
        .map_err(|e| format!("Failed to compile activation shader: {:?}", e))?;

    // Calculate dispatch size
    let (threadgroups, threads_per_threadgroup) = dispatch_1d(num_elements as usize, 256);

    // Prepare constant buffer for threshold (used by FatReLU)
    let constants = [threshold, num_elements as f32];
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
            (&constants_buffer, 0, 2), // [[buffer(2)]]
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
    fn test_silu_recording() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = RecordingContext::new(Arc::new(device.device.clone()), 10)
            .expect("Failed to create recording context");

        // Create dummy buffers
        let num_elements = 1024 * 4096;
        let input = device.device.new_buffer(
            num_elements * 2, // fp16
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output = device.device.new_buffer(
            num_elements * 2,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Record SiLU dispatch
        let result =
            record_activation(&mut ctx, &input, &output, num_elements, "silu", "fp16", 0.0);

        assert!(result.is_ok(), "SiLU recording failed: {:?}", result);
        assert_eq!(ctx.command_count(), 1, "Should record exactly 1 command");
    }
}
