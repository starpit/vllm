// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! RMSNorm instruction recording for Metal ICB.

use super::{dispatch_1d, Buffer, RecordingContext};
use crate::shader_cache::ShaderCache;
use objc2_metal::{MTLDevice, MTLResourceOptions};
use std::ffi::c_void;
use std::ptr::NonNull;

pub fn record_rmsnorm(
    ctx: &mut RecordingContext,
    input_buffer: &Buffer,
    output_buffer: &Buffer,
    weight_buffer: &Buffer,
    num_tokens: usize,
    hidden_size: usize,
    eps: f32,
    dtype: &str,
) -> Result<(), String> {
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
            (input_buffer, 0, 0),
            (output_buffer, 0, 1),
            (weight_buffer, 0, 2),
            (&constants_buffer, 0, 3),
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
    fn test_rmsnorm_recording() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = RecordingContext::new(Arc::new(device.device.clone()), 10)
            .expect("Failed to create recording context");

        let input = device
            .device
            .newBufferWithLength_options(1024 * 4096 * 2, MTLResourceOptions::StorageModeShared)
            .expect("input");
        let output = device
            .device
            .newBufferWithLength_options(1024 * 4096 * 2, MTLResourceOptions::StorageModeShared)
            .expect("output");
        let weight = device
            .device
            .newBufferWithLength_options(4096 * 2, MTLResourceOptions::StorageModeShared)
            .expect("weight");

        let result = record_rmsnorm(&mut ctx, &input, &output, &weight, 1024, 4096, 1e-6, "fp16");

        assert!(result.is_ok(), "RMSNorm recording failed: {:?}", result);
        assert_eq!(ctx.command_count(), 1);
    }
}
