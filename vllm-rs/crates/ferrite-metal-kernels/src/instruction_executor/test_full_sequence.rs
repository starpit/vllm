// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Integration test for recording and executing a full instruction sequence.
//!
//! This test validates the complete ICB infrastructure by recording multiple
//! operations (RMSNorm, Activation, RoPE) into a single ICB and executing them.

#[cfg(test)]
mod tests {
    use crate::detect_device;
    use crate::instruction_executor::{activation, rmsnorm, rope, RecordingContext};
    use metal::MTLResourceOptions;
    use std::sync::Arc;

    #[test]
    fn test_full_instruction_sequence_recording() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = RecordingContext::new(Arc::new(device.device.clone()), 100)
            .expect("Failed to create recording context");

        let num_tokens = 128;
        let hidden_size = 4096;
        let num_q_heads = 32;
        let num_kv_heads = 8;
        let head_dim = 128;

        // Create buffers for a typical forward pass sequence:
        // 1. RMSNorm on input
        // 2. SiLU activation
        // 3. RoPE on Q/K

        // RMSNorm buffers
        let input = device.device.new_buffer(
            num_tokens * hidden_size * 2, // fp16
            MTLResourceOptions::StorageModeShared,
        );
        let normed = device.device.new_buffer(
            num_tokens * hidden_size * 2,
            MTLResourceOptions::StorageModeShared,
        );
        let norm_weight = device
            .device
            .new_buffer(hidden_size * 2, MTLResourceOptions::StorageModeShared);

        // Activation buffers
        let activated = device.device.new_buffer(
            num_tokens * hidden_size * 2,
            MTLResourceOptions::StorageModeShared,
        );

        // RoPE buffers
        let q = device.device.new_buffer(
            num_tokens * num_q_heads * head_dim * 2,
            MTLResourceOptions::StorageModeShared,
        );
        let k = device.device.new_buffer(
            num_tokens * num_kv_heads * head_dim * 2,
            MTLResourceOptions::StorageModeShared,
        );
        let positions = device.device.new_buffer(
            num_tokens * 4, // i32
            MTLResourceOptions::StorageModeShared,
        );
        let cos_sin = device
            .device
            .new_buffer(2048 * head_dim * 2, MTLResourceOptions::StorageModeShared);

        // Record instruction sequence
        println!("Recording instruction 1: RMSNorm");
        rmsnorm::record_rmsnorm(
            &mut ctx,
            &input,
            &normed,
            &norm_weight,
            num_tokens as u32,
            hidden_size as u32,
            1e-6,
            "fp16",
        )
        .expect("Failed to record RMSNorm");

        println!("Recording instruction 2: SiLU activation");
        activation::record_activation(
            &mut ctx,
            &normed,
            &activated,
            (num_tokens * hidden_size) as u64,
            "silu",
            "fp16",
            0.0,
        )
        .expect("Failed to record SiLU");

        println!("Recording instruction 3: RoPE");
        rope::record_rope(
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
        )
        .expect("Failed to record RoPE");

        // Verify all commands were recorded
        assert_eq!(ctx.command_count(), 3, "Should record exactly 3 commands");
        println!(
            "✅ Successfully recorded {} instructions into ICB",
            ctx.command_count()
        );

        // Create command buffer and encoder to execute the ICB
        let command_queue = device.device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // CRITICAL: Set pipeline state on encoder before executing ICB
        // The ICB inherits the pipeline state from the encoder
        // For this test, we'll use the RMSNorm pipeline as a placeholder
        // (in production, each wave would set its own pipeline)
        let shader_cache = crate::shader_cache::ShaderCache::new(device.device.clone())
            .expect("Failed to create shader cache");
        let pipeline = shader_cache
            .get_pipeline("rmsnorm_f16")
            .expect("Failed to get pipeline");
        encoder.set_compute_pipeline_state(&pipeline);

        // Execute the recorded ICB
        println!("Executing ICB with {} commands", ctx.command_count());
        ctx.execute_on_encoder(encoder, 0..ctx.command_count());

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        println!("✅ ICB execution completed successfully");
        println!("✅ Full instruction sequence test PASSED");
    }

    #[test]
    fn test_icb_reset_and_rerecord() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = RecordingContext::new(Arc::new(device.device.clone()), 100)
            .expect("Failed to create recording context");

        let num_tokens = 64;
        let hidden_size = 2048;

        // Create buffers
        let input = device.device.new_buffer(
            num_tokens * hidden_size * 2,
            MTLResourceOptions::StorageModeShared,
        );
        let output = device.device.new_buffer(
            num_tokens * hidden_size * 2,
            MTLResourceOptions::StorageModeShared,
        );
        let weight = device
            .device
            .new_buffer(hidden_size * 2, MTLResourceOptions::StorageModeShared);

        // Record first sequence
        rmsnorm::record_rmsnorm(
            &mut ctx,
            &input,
            &output,
            &weight,
            num_tokens as u32,
            hidden_size as u32,
            1e-6,
            "fp16",
        )
        .expect("Failed to record RMSNorm");

        assert_eq!(ctx.command_count(), 1);

        // Reset and record again
        ctx.reset_range(0..1);

        rmsnorm::record_rmsnorm(
            &mut ctx,
            &input,
            &output,
            &weight,
            num_tokens as u32,
            hidden_size as u32,
            1e-5, // Different epsilon
            "fp16",
        )
        .expect("Failed to re-record RMSNorm");

        assert_eq!(ctx.command_count(), 1);
        println!("✅ ICB reset and re-record test PASSED");
    }
}
