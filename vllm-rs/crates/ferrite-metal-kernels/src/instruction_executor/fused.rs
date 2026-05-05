// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Fused kernel instruction recording for Metal ICB.

use super::{dispatch_1d, RecordingContext};
use crate::shader_cache::ShaderCache;

/// Record a fused Add+RMSNorm kernel dispatch into the ICB.
///
/// # Arguments
/// * `ctx` - Recording context with device and ICB
/// * `delta_buffer` - Delta tensor to add
/// * `residual_buffer` - Residual tensor (updated in-place)
/// * `weight_buffer` - RMSNorm weight buffer
/// * `num_tokens` - Number of tokens
/// * `hidden_size` - Hidden size
/// * `eps` - Epsilon for numerical stability
/// * `dtype` - Data type ("fp16" or "bf16")
///
/// # Returns
/// `Ok(())` on success, `Err(String)` on failure
pub fn record_fused_add_rmsnorm(
    ctx: &mut RecordingContext,
    delta_buffer: &metal::Buffer,
    residual_buffer: &metal::Buffer,
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
        "fp16" => "fused_add_rmsnorm_f16",
        "bf16" => "fused_add_rmsnorm_bf16",
        _ => return Err(format!("Unsupported dtype: {}", dtype)),
    };

    let pipeline = shader_cache
        .get_pipeline(kernel_name)
        .map_err(|e| format!("Failed to compile fused Add+RMSNorm shader: {:?}", e))?;

    // Calculate dispatch size
    let (threadgroups, threads_per_threadgroup) = dispatch_1d(num_tokens, 256);

    // Prepare constant buffer
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
            (delta_buffer, 0, 0),
            (residual_buffer, 0, 1),
            (weight_buffer, 0, 2),
            (&constants_buffer, 0, 3),
        ],
        threadgroups,
        threads_per_threadgroup,
    );

    Ok(())
}

/// Record a fused Gate-Up-SiLU-Mul kernel dispatch into the ICB.
///
/// # Arguments
/// * `ctx` - Recording context with device and ICB
/// * `input_buffer` - Input tensor
/// * `gate_up_buffer` - Gate+Up projection output (from GEMM)
/// * `output_buffer` - Output buffer
/// * `num_tokens` - Number of tokens
/// * `intermediate_size` - Intermediate size
/// * `dtype` - Data type ("fp16" or "bf16")
///
/// # Returns
/// `Ok(())` on success, `Err(String)` on failure
pub fn record_fused_gate_up_silu_mul(
    ctx: &mut RecordingContext,
    gate_up_buffer: &metal::Buffer,
    output_buffer: &metal::Buffer,
    num_tokens: usize,
    intermediate_size: usize,
    dtype: &str,
) -> Result<(), String> {
    // Get or compile shader
    let shader_cache = ShaderCache::new((*ctx.device).clone())
        .map_err(|e| format!("Failed to create shader cache: {:?}", e))?;
    let kernel_name = match dtype {
        "fp16" => "fused_gate_up_silu_mul_f16",
        "bf16" => "fused_gate_up_silu_mul_bf16",
        _ => return Err(format!("Unsupported dtype: {}", dtype)),
    };

    let pipeline = shader_cache
        .get_pipeline(kernel_name)
        .map_err(|e| format!("Failed to compile fused SwiGLU shader: {:?}", e))?;

    // Calculate dispatch size
    let total_elements = num_tokens * intermediate_size;
    let (threadgroups, threads_per_threadgroup) = dispatch_1d(total_elements, 256);

    // Prepare constant buffer
    let constants = [intermediate_size as u32];
    let constants_buffer = ctx.device.new_buffer_with_data(
        constants.as_ptr() as *const _,
        std::mem::size_of_val(&constants) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Record the dispatch
    ctx.record_compute_dispatch(
        &pipeline,
        &[
            (gate_up_buffer, 0, 0),
            (output_buffer, 0, 1),
            (&constants_buffer, 0, 2),
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
    use std::sync::Arc;

    #[test]
    fn test_fused_add_rmsnorm_recording() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = super::super::RecordingContext::new(Arc::new(device.device.clone()), 10)
            .expect("Failed to create recording context");

        // Create dummy buffers
        let delta = device.device.new_buffer(
            1024 * 4096 * 2,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let residual = device.device.new_buffer(
            1024 * 4096 * 2,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let weight = device
            .device
            .new_buffer(4096 * 2, metal::MTLResourceOptions::StorageModeShared);

        let result = record_fused_add_rmsnorm(
            &mut ctx, &delta, &residual, &weight, 1024, 4096, 1e-6, "fp16",
        );

        assert!(
            result.is_ok(),
            "Fused Add+RMSNorm recording failed: {:?}",
            result
        );
        assert_eq!(ctx.command_count(), 1);
    }
}
