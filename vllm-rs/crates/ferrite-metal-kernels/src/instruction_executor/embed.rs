// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Embed instruction recording for Metal ICB.
//!
//! This module provides instruction recorders for embedding lookup operations.

use super::{dispatch_1d, RecordingContext};
use crate::shader_cache::ShaderCache;
use metal::MTLResourceOptions;
use std::sync::Arc;

/// Record an Embed kernel dispatch into the ICB.
///
/// Performs embedding lookup: out[i] = table[indices[i]]
///
/// # Arguments
/// * `ctx` - Recording context with device and ICB
/// * `table_buffer` - Embedding table buffer [vocab_size, hidden_size]
/// * `indices_buffer` - Input indices buffer [num_tokens] (i32)
/// * `output_buffer` - Output tensor buffer [num_tokens, hidden_size]
/// * `num_tokens` - Number of tokens to embed
/// * `hidden_size` - Hidden dimension size
/// * `dtype` - Data type ("fp16" or "bf16")
///
/// # Returns
/// `Ok(())` on success, `Err(String)` on shader compilation failure
pub fn record_embed(
    ctx: &mut RecordingContext,
    table_buffer: &metal::Buffer,
    indices_buffer: &metal::Buffer,
    output_buffer: &metal::Buffer,
    num_tokens: u32,
    hidden_size: u32,
    dtype: &str,
) -> Result<(), String> {
    let shader_cache = ShaderCache::new(ctx.device.as_ref().clone())
        .map_err(|e| format!("Failed to create shader cache: {:?}", e))?;

    let kernel_name = match dtype {
        "fp16" => "embed_f16",
        "bf16" => "embed_bf16",
        _ => return Err(format!("Unsupported dtype: {}", dtype)),
    };

    let pipeline = shader_cache
        .get_pipeline(kernel_name)
        .map_err(|e| format!("Failed to compile Embed shader: {:?}", e))?;

    // Each thread processes one token (copies one row from table to output)
    let (threadgroups, threads_per_threadgroup) = dispatch_1d(num_tokens as usize, 256);

    // Create constant buffer for hidden_size
    let constants = [hidden_size];
    let constants_buffer = ctx.device.new_buffer_with_data(
        constants.as_ptr() as *const _,
        std::mem::size_of_val(&constants) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    ctx.record_compute_dispatch(
        &pipeline,
        &[
            (table_buffer, 0, 0),      // [[buffer(0)]]
            (indices_buffer, 0, 1),    // [[buffer(1)]]
            (output_buffer, 0, 2),     // [[buffer(2)]]
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
    fn test_embed_recording() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = RecordingContext::new(Arc::new(device.device.clone()), 10)
            .expect("Failed to create recording context");

        let vocab_size = 32000;
        let hidden_size = 4096;
        let num_tokens = 128;

        // Create buffers
        let table = device.device.new_buffer(
            (vocab_size * hidden_size * 2) as u64, // fp16
            metal::MTLResourceOptions::StorageModeShared,
        );
        let indices = device.device.new_buffer(
            (num_tokens * 4) as u64, // i32
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output = device.device.new_buffer(
            (num_tokens * hidden_size * 2) as u64, // fp16
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Record Embed dispatch
        let result = record_embed(
            &mut ctx,
            &table,
            &indices,
            &output,
            num_tokens as u32,
            hidden_size as u32,
            "fp16",
        );

        assert!(result.is_ok(), "Embed recording failed: {:?}", result);
        assert_eq!(ctx.command_count(), 1, "Should record exactly 1 command");
    }
}
