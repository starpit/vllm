// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Elementwise operation instruction recording for Metal ICB.
//!
//! This module provides instruction recorders for elementwise binary operations:
//! - Add: out = a + b
//! - Mul: out = a * b
//! - Sub: out = a - b
//! - ScalarMul: out = scalar * input
//! - BiasAdd: out = input + bias (with broadcasting)

use super::{dispatch_1d, RecordingContext};
use crate::shader_cache::ShaderCache;
use metal::MTLResourceOptions;
use std::sync::Arc;

/// Record an Add kernel dispatch into the ICB.
///
/// Performs elementwise addition: out = a + b
///
/// # Arguments
/// * `ctx` - Recording context with device and ICB
/// * `a_buffer` - First input tensor buffer
/// * `b_buffer` - Second input tensor buffer
/// * `output_buffer` - Output tensor buffer
/// * `num_elements` - Total number of elements
/// * `dtype` - Data type ("fp16" or "bf16")
pub fn record_add(
    ctx: &mut RecordingContext,
    a_buffer: &metal::Buffer,
    b_buffer: &metal::Buffer,
    output_buffer: &metal::Buffer,
    num_elements: u64,
    dtype: &str,
) -> Result<(), String> {
    let shader_cache = ShaderCache::new(ctx.device.as_ref().clone())
        .map_err(|e| format!("Failed to create shader cache: {:?}", e))?;

    let kernel_name = match dtype {
        "fp16" => "add_f16",
        "bf16" => "add_bf16",
        _ => return Err(format!("Unsupported dtype: {}", dtype)),
    };

    let pipeline = shader_cache
        .get_pipeline(kernel_name)
        .map_err(|e| format!("Failed to compile Add shader: {:?}", e))?;

    let (threadgroups, threads_per_threadgroup) = dispatch_1d(num_elements as usize, 256);

    ctx.record_compute_dispatch(
        &pipeline,
        &[(a_buffer, 0, 0), (b_buffer, 0, 1), (output_buffer, 0, 2)],
        threadgroups,
        threads_per_threadgroup,
    );

    Ok(())
}

/// Record a Mul kernel dispatch into the ICB.
///
/// Performs elementwise multiplication: out = a * b
///
/// # Arguments
/// * `ctx` - Recording context with device and ICB
/// * `a_buffer` - First input tensor buffer
/// * `b_buffer` - Second input tensor buffer
/// * `output_buffer` - Output tensor buffer
/// * `num_elements` - Total number of elements
/// * `dtype` - Data type ("fp16" or "bf16")
pub fn record_mul(
    ctx: &mut RecordingContext,
    a_buffer: &metal::Buffer,
    b_buffer: &metal::Buffer,
    output_buffer: &metal::Buffer,
    num_elements: u64,
    dtype: &str,
) -> Result<(), String> {
    let shader_cache = ShaderCache::new(ctx.device.as_ref().clone())
        .map_err(|e| format!("Failed to create shader cache: {:?}", e))?;

    let kernel_name = match dtype {
        "fp16" => "mul_f16",
        "bf16" => "mul_bf16",
        _ => return Err(format!("Unsupported dtype: {}", dtype)),
    };

    let pipeline = shader_cache
        .get_pipeline(kernel_name)
        .map_err(|e| format!("Failed to compile Mul shader: {:?}", e))?;

    let (threadgroups, threads_per_threadgroup) = dispatch_1d(num_elements as usize, 256);

    ctx.record_compute_dispatch(
        &pipeline,
        &[(a_buffer, 0, 0), (b_buffer, 0, 1), (output_buffer, 0, 2)],
        threadgroups,
        threads_per_threadgroup,
    );

    Ok(())
}

/// Record a Sub kernel dispatch into the ICB.
///
/// Performs elementwise subtraction: out = a - b
///
/// # Arguments
/// * `ctx` - Recording context with device and ICB
/// * `a_buffer` - First input tensor buffer
/// * `b_buffer` - Second input tensor buffer
/// * `output_buffer` - Output tensor buffer
/// * `num_elements` - Total number of elements
/// * `dtype` - Data type ("fp16" or "bf16")
pub fn record_sub(
    ctx: &mut RecordingContext,
    a_buffer: &metal::Buffer,
    b_buffer: &metal::Buffer,
    output_buffer: &metal::Buffer,
    num_elements: u64,
    dtype: &str,
) -> Result<(), String> {
    let shader_cache = ShaderCache::new(ctx.device.as_ref().clone())
        .map_err(|e| format!("Failed to create shader cache: {:?}", e))?;

    let kernel_name = match dtype {
        "fp16" => "sub_f16",
        "bf16" => "sub_bf16",
        _ => return Err(format!("Unsupported dtype: {}", dtype)),
    };

    let pipeline = shader_cache
        .get_pipeline(kernel_name)
        .map_err(|e| format!("Failed to compile Sub shader: {:?}", e))?;

    let (threadgroups, threads_per_threadgroup) = dispatch_1d(num_elements as usize, 256);

    ctx.record_compute_dispatch(
        &pipeline,
        &[(a_buffer, 0, 0), (b_buffer, 0, 1), (output_buffer, 0, 2)],
        threadgroups,
        threads_per_threadgroup,
    );

    Ok(())
}

/// Record a ScalarMul kernel dispatch into the ICB.
///
/// Performs broadcast scalar multiplication: out = scalar * input
///
/// # Arguments
/// * `ctx` - Recording context with device and ICB
/// * `input_buffer` - Input tensor buffer
/// * `output_buffer` - Output tensor buffer
/// * `scalar` - Scalar value to multiply
/// * `num_elements` - Total number of elements
/// * `dtype` - Data type ("fp16" or "bf16")
pub fn record_scalar_mul(
    ctx: &mut RecordingContext,
    input_buffer: &metal::Buffer,
    output_buffer: &metal::Buffer,
    scalar: f32,
    num_elements: u64,
    dtype: &str,
) -> Result<(), String> {
    let shader_cache = ShaderCache::new(ctx.device.as_ref().clone())
        .map_err(|e| format!("Failed to create shader cache: {:?}", e))?;

    let kernel_name = match dtype {
        "fp16" => "scalar_mul_f16",
        "bf16" => "scalar_mul_bf16",
        _ => return Err(format!("Unsupported dtype: {}", dtype)),
    };

    let pipeline = shader_cache
        .get_pipeline(kernel_name)
        .map_err(|e| format!("Failed to compile ScalarMul shader: {:?}", e))?;

    let (threadgroups, threads_per_threadgroup) = dispatch_1d(num_elements as usize, 256);

    // Create constant buffer for scalar
    let constants = [scalar];
    let constants_buffer = ctx.device.new_buffer_with_data(
        constants.as_ptr() as *const _,
        std::mem::size_of_val(&constants) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    ctx.record_compute_dispatch(
        &pipeline,
        &[
            (input_buffer, 0, 0),
            (output_buffer, 0, 1),
            (&constants_buffer, 0, 2),
        ],
        threadgroups,
        threads_per_threadgroup,
    );

    Ok(())
}

/// Record a BiasAdd kernel dispatch into the ICB.
///
/// Performs broadcast addition: out = input + bias
/// Bias is broadcast across the last dimension.
///
/// # Arguments
/// * `ctx` - Recording context with device and ICB
/// * `input_buffer` - Input tensor buffer [M, N]
/// * `bias_buffer` - Bias vector buffer [N]
/// * `output_buffer` - Output tensor buffer [M, N]
/// * `num_rows` - Number of rows (M)
/// * `num_cols` - Number of columns (N)
/// * `dtype` - Data type ("fp16" or "bf16")
pub fn record_bias_add(
    ctx: &mut RecordingContext,
    input_buffer: &metal::Buffer,
    bias_buffer: &metal::Buffer,
    output_buffer: &metal::Buffer,
    num_rows: u32,
    num_cols: u32,
    dtype: &str,
) -> Result<(), String> {
    let shader_cache = ShaderCache::new(ctx.device.as_ref().clone())
        .map_err(|e| format!("Failed to create shader cache: {:?}", e))?;

    let kernel_name = match dtype {
        "fp16" => "bias_add_f16",
        "bf16" => "bias_add_bf16",
        _ => return Err(format!("Unsupported dtype: {}", dtype)),
    };

    let pipeline = shader_cache
        .get_pipeline(kernel_name)
        .map_err(|e| format!("Failed to compile BiasAdd shader: {:?}", e))?;

    let (threadgroups, threads_per_threadgroup) = dispatch_1d((num_rows * num_cols) as usize, 256);

    // Create constant buffer for dimensions
    let constants = [num_cols];
    let constants_buffer = ctx.device.new_buffer_with_data(
        constants.as_ptr() as *const _,
        std::mem::size_of_val(&constants) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    ctx.record_compute_dispatch(
        &pipeline,
        &[
            (input_buffer, 0, 0),
            (bias_buffer, 0, 1),
            (output_buffer, 0, 2),
            (&constants_buffer, 0, 3),
        ],
        threadgroups,
        threads_per_threadgroup,
    );

    Ok(())
}

/// Record a TanhSoftCap kernel dispatch into the ICB.
///
/// Performs logit capping: out = cap * tanh(input / cap)
/// Used in Gemma2 architecture for attention logit capping.
///
/// # Arguments
/// * `ctx` - Recording context with device and ICB
/// * `input_buffer` - Input tensor buffer
/// * `output_buffer` - Output tensor buffer
/// * `cap` - Soft cap value
/// * `num_elements` - Total number of elements
/// * `dtype` - Data type ("fp16" or "bf16")
pub fn record_tanh_soft_cap(
    ctx: &mut RecordingContext,
    input_buffer: &metal::Buffer,
    output_buffer: &metal::Buffer,
    cap: f32,
    num_elements: u64,
    dtype: &str,
) -> Result<(), String> {
    let shader_cache = ShaderCache::new(ctx.device.as_ref().clone())
        .map_err(|e| format!("Failed to create shader cache: {:?}", e))?;

    let kernel_name = match dtype {
        "fp16" => "tanh_soft_cap_f16",
        "bf16" => "tanh_soft_cap_bf16",
        _ => return Err(format!("Unsupported dtype: {}", dtype)),
    };

    let pipeline = shader_cache
        .get_pipeline(kernel_name)
        .map_err(|e| format!("Failed to compile TanhSoftCap shader: {:?}", e))?;

    let (threadgroups, threads_per_threadgroup) = dispatch_1d(num_elements as usize, 256);

    // Create constant buffer for cap
    let constants = [cap];
    let constants_buffer = ctx.device.new_buffer_with_data(
        constants.as_ptr() as *const _,
        std::mem::size_of_val(&constants) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    ctx.record_compute_dispatch(
        &pipeline,
        &[
            (input_buffer, 0, 0),
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

    #[test]
    fn test_add_recording() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = RecordingContext::new(Arc::new(device.device.clone()), 10)
            .expect("Failed to create recording context");

        let num_elements = 1024 * 4096;
        let buffer_size = num_elements * 2; // fp16

        let a = device
            .device
            .new_buffer(buffer_size as u64, MTLResourceOptions::StorageModeShared);
        let b = device
            .device
            .new_buffer(buffer_size as u64, MTLResourceOptions::StorageModeShared);
        let out = device
            .device
            .new_buffer(buffer_size as u64, MTLResourceOptions::StorageModeShared);

        let result = record_add(&mut ctx, &a, &b, &out, num_elements as u64, "fp16");
        assert!(result.is_ok(), "Add recording failed: {:?}", result);
        assert_eq!(ctx.command_count(), 1);
    }

    #[test]
    fn test_mul_recording() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = RecordingContext::new(Arc::new(device.device.clone()), 10)
            .expect("Failed to create recording context");

        let num_elements = 1024 * 4096;
        let buffer_size = num_elements * 2;

        let a = device
            .device
            .new_buffer(buffer_size as u64, MTLResourceOptions::StorageModeShared);
        let b = device
            .device
            .new_buffer(buffer_size as u64, MTLResourceOptions::StorageModeShared);
        let out = device
            .device
            .new_buffer(buffer_size as u64, MTLResourceOptions::StorageModeShared);

        let result = record_mul(&mut ctx, &a, &b, &out, num_elements as u64, "fp16");
        assert!(result.is_ok(), "Mul recording failed: {:?}", result);
        assert_eq!(ctx.command_count(), 1);
    }

    #[test]
    fn test_sub_recording() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = RecordingContext::new(Arc::new(device.device.clone()), 10)
            .expect("Failed to create recording context");

        let num_elements = 1024 * 4096;
        let buffer_size = num_elements * 2;

        let a = device
            .device
            .new_buffer(buffer_size as u64, MTLResourceOptions::StorageModeShared);
        let b = device
            .device
            .new_buffer(buffer_size as u64, MTLResourceOptions::StorageModeShared);
        let out = device
            .device
            .new_buffer(buffer_size as u64, MTLResourceOptions::StorageModeShared);

        let result = record_sub(&mut ctx, &a, &b, &out, num_elements as u64, "fp16");
        assert!(result.is_ok(), "Sub recording failed: {:?}", result);
        assert_eq!(ctx.command_count(), 1);
    }

    #[test]
    fn test_scalar_mul_recording() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = RecordingContext::new(Arc::new(device.device.clone()), 10)
            .expect("Failed to create recording context");

        let num_elements = 1024 * 4096;
        let buffer_size = num_elements * 2;

        let input = device
            .device
            .new_buffer(buffer_size as u64, MTLResourceOptions::StorageModeShared);
        let out = device
            .device
            .new_buffer(buffer_size as u64, MTLResourceOptions::StorageModeShared);

        let result = record_scalar_mul(&mut ctx, &input, &out, 2.5, num_elements as u64, "fp16");
        assert!(result.is_ok(), "ScalarMul recording failed: {:?}", result);
        assert_eq!(ctx.command_count(), 1);
    }

    #[test]
    fn test_bias_add_recording() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = RecordingContext::new(Arc::new(device.device.clone()), 10)
            .expect("Failed to create recording context");

        let num_rows = 1024;
        let num_cols = 4096;
        let input_size = num_rows * num_cols * 2;
        let bias_size = num_cols * 2;

        let input = device
            .device
            .new_buffer(input_size as u64, MTLResourceOptions::StorageModeShared);
        let bias = device
            .device
            .new_buffer(bias_size as u64, MTLResourceOptions::StorageModeShared);
        let out = device
            .device
            .new_buffer(input_size as u64, MTLResourceOptions::StorageModeShared);

        let result = record_bias_add(&mut ctx, &input, &bias, &out, num_rows, num_cols, "fp16");
        assert!(result.is_ok(), "BiasAdd recording failed: {:?}", result);
        assert_eq!(ctx.command_count(), 1);
    }

    #[test]
    fn test_tanh_soft_cap_recording() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = RecordingContext::new(Arc::new(device.device.clone()), 10)
            .expect("Failed to create recording context");

        let num_elements = 1024 * 4096;
        let buffer_size = num_elements * 2;

        let input = device
            .device
            .new_buffer(buffer_size as u64, MTLResourceOptions::StorageModeShared);
        let out = device
            .device
            .new_buffer(buffer_size as u64, MTLResourceOptions::StorageModeShared);

        let result =
            record_tanh_soft_cap(&mut ctx, &input, &out, 30.0, num_elements as u64, "fp16");
        assert!(result.is_ok(), "TanhSoftCap recording failed: {:?}", result);
        assert_eq!(ctx.command_count(), 1);
    }
}
