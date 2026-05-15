// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Fused kernel instruction recording for Metal ICB.

use super::{dispatch_1d, Buffer, RecordingContext};
use crate::shader_cache::ShaderCache;
use objc2_metal::{MTLDevice, MTLResourceOptions};
use std::ffi::c_void;
use std::ptr::NonNull;

pub fn record_fused_add_rmsnorm(
    ctx: &mut RecordingContext,
    delta_buffer: &Buffer,
    residual_buffer: &Buffer,
    weight_buffer: &Buffer,
    num_tokens: usize,
    hidden_size: usize,
    eps: f32,
    dtype: &str,
) -> Result<(), String> {
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

    let (threadgroups, threads_per_threadgroup) = dispatch_1d(num_tokens, 256);

    let constants = [eps, hidden_size as f32];
    let constants_buffer = unsafe {
        ctx.device
            .newBufferWithBytes_length_options(
                NonNull::new(constants.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(&constants),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| "newBufferWithBytes returned nil".to_string())?
    };

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

pub fn record_fused_gate_up_silu_mul(
    ctx: &mut RecordingContext,
    gate_up_buffer: &Buffer,
    output_buffer: &Buffer,
    num_tokens: usize,
    intermediate_size: usize,
    dtype: &str,
) -> Result<(), String> {
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

    let total_elements = num_tokens * intermediate_size;
    let (threadgroups, threads_per_threadgroup) = dispatch_1d(total_elements, 256);

    let constants = [intermediate_size as u32];
    let constants_buffer = unsafe {
        ctx.device
            .newBufferWithBytes_length_options(
                NonNull::new(constants.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(&constants),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| "newBufferWithBytes returned nil".to_string())?
    };

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

        let delta = device
            .device
            .newBufferWithLength_options(1024 * 4096 * 2, MTLResourceOptions::StorageModeShared)
            .expect("delta");
        let residual = device
            .device
            .newBufferWithLength_options(1024 * 4096 * 2, MTLResourceOptions::StorageModeShared)
            .expect("residual");
        let weight = device
            .device
            .newBufferWithLength_options(4096 * 2, MTLResourceOptions::StorageModeShared)
            .expect("weight");

        let result = record_fused_add_rmsnorm(
            &mut ctx, &delta, &residual, &weight, 1024, 4096, 1e-6, "fp16",
        );

        assert!(result.is_ok(), "Recording failed: {:?}", result);
        assert_eq!(ctx.command_count(), 1);
    }
}
