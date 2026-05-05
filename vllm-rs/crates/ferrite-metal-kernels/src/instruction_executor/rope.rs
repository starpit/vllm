// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! RoPE (Rotary Positional Embedding) instruction recording for Metal ICB.

use super::{dispatch_1d, RecordingContext};
use crate::shader_cache::ShaderCache;
use metal::MTLResourceOptions;
use std::sync::Arc;

/// Record a RoPE kernel dispatch into the ICB.
///
/// # Arguments
/// * `ctx` - Recording context with device and ICB
/// * `q_buffer` - Query tensor buffer [num_tokens, num_heads, head_dim]
/// * `k_buffer` - Key tensor buffer [num_tokens, num_kv_heads, head_dim]
/// * `positions_buffer` - Position indices buffer [num_tokens]
/// * `cos_sin_buffer` - Precomputed cos/sin cache [max_positions, head_dim]
/// * `num_tokens` - Number of tokens
/// * `num_q_heads` - Number of query heads
/// * `num_kv_heads` - Number of key/value heads
/// * `head_dim` - Head dimension
/// * `interleaved` - Whether to use interleaved RoPE (GPT-J style) vs NeoX style
/// * `dtype` - Data type ("fp16" or "bf16")
///
/// # Returns
/// `Ok(())` on success, `Err(String)` on shader compilation failure
pub fn record_rope(
    ctx: &mut RecordingContext,
    q_buffer: &metal::Buffer,
    k_buffer: &metal::Buffer,
    positions_buffer: &metal::Buffer,
    cos_sin_buffer: &metal::Buffer,
    num_tokens: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    interleaved: bool,
    dtype: &str,
) -> Result<(), String> {
    // Get or compile shader
    let shader_cache = ShaderCache::new(ctx.device.as_ref().clone())
        .map_err(|e| format!("Failed to create shader cache: {:?}", e))?;

    let kernel_name = match (interleaved, dtype) {
        (false, "fp16") => "rope_neox_f16",
        (false, "bf16") => "rope_neox_bf16",
        (true, "fp16") => "rope_interleaved_f16",
        (true, "bf16") => "rope_interleaved_bf16",
        _ => {
            return Err(format!(
                "Unsupported interleaved/dtype: {}/{}",
                interleaved, dtype
            ))
        }
    };

    let pipeline = shader_cache
        .get_pipeline(kernel_name)
        .map_err(|e| format!("Failed to compile RoPE shader: {:?}", e))?;

    // Calculate dispatch size - one thread per token
    let (threadgroups, threads_per_threadgroup) = dispatch_1d(num_tokens as usize, 256);

    // Prepare constant buffer
    let constants = [
        num_tokens as u32,
        num_q_heads,
        num_kv_heads,
        head_dim,
    ];
    let constants_buffer = ctx.device.new_buffer_with_data(
        constants.as_ptr() as *const _,
        std::mem::size_of_val(&constants) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Record the dispatch
    ctx.record_compute_dispatch(
        &pipeline,
        &[
            (q_buffer, 0, 0),          // [[buffer(0)]]
            (k_buffer, 0, 1),          // [[buffer(1)]]
            (positions_buffer, 0, 2),  // [[buffer(2)]]
            (cos_sin_buffer, 0, 3),    // [[buffer(3)]]
            (&constants_buffer, 0, 4), // [[buffer(4)]]
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
    fn test_rope_recording() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = RecordingContext::new(Arc::new(device.device.clone()), 10)
            .expect("Failed to create recording context");

        let num_tokens = 128;
        let num_q_heads = 32;
        let num_kv_heads = 8;
        let head_dim = 128;

        // Create dummy buffers
        let q = device.device.new_buffer(
            num_tokens * num_q_heads * head_dim * 2, // fp16
            metal::MTLResourceOptions::StorageModeShared,
        );
        let k = device.device.new_buffer(
            num_tokens * num_kv_heads * head_dim * 2,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let positions = device.device.new_buffer(
            num_tokens * 4, // i32
            metal::MTLResourceOptions::StorageModeShared,
        );
        let cos_sin = device.device.new_buffer(
            2048 * head_dim * 2, // max_positions × head_dim × fp16
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Record RoPE dispatch
        let result = record_rope(
            &mut ctx,
            &q,
            &k,
            &positions,
            &cos_sin,
            num_tokens as u32,
            num_q_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            false, // NeoX style
            "fp16",
        );

        assert!(result.is_ok(), "RoPE recording failed: {:?}", result);
        assert_eq!(ctx.command_count(), 1, "Should record exactly 1 command");
    }
}
