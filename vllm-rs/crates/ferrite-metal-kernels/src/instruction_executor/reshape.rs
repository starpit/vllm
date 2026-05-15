// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Reshape instruction recording for Metal ICB.
//!
//! Reshape is a metadata-only operation that doesn't require GPU execution.
//! It simply reinterprets the buffer with different dimensions.

use super::RecordingContext;

/// Record a Reshape operation (metadata-only, no GPU dispatch needed).
///
/// Reshape is a view operation that doesn't copy data or require GPU execution.
/// This function exists for API completeness but doesn't record any ICB commands.
///
/// # Arguments
/// * `ctx` - Recording context (unused, kept for API consistency)
/// * `input_buffer` - Input tensor buffer (same as output)
/// * `output_buffer` - Output tensor buffer (same as input)
/// * `input_shape` - Input shape dimensions
/// * `output_shape` - Output shape dimensions
///
/// # Returns
/// Always returns `Ok(())` since no GPU work is needed
pub fn record_reshape(
    _ctx: &mut RecordingContext,
    _input_buffer: &metal::Buffer,
    _output_buffer: &metal::Buffer,
    _input_shape: &[usize],
    _output_shape: &[usize],
) -> Result<(), String> {
    // Reshape is a metadata-only operation - no GPU dispatch needed
    // The input and output buffers are the same, just with different shape metadata
    // This function exists for API completeness but doesn't record any ICB commands
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect_device;
    use std::sync::Arc;

    #[test]
    fn test_reshape_recording() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = RecordingContext::new(Arc::new(device.device.clone()), 10)
            .expect("Failed to create recording context");

        let buffer_size = 1024 * 4096 * 2; // fp16
        let buffer = device.device.new_buffer(
            buffer_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Reshape [1024, 4096] -> [4096, 1024]
        let result = record_reshape(&mut ctx, &buffer, &buffer, &[1024, 4096], &[4096, 1024]);

        assert!(result.is_ok(), "Reshape recording failed: {:?}", result);
        // Reshape doesn't record any commands (metadata-only)
        assert_eq!(
            ctx.command_count(),
            0,
            "Reshape should not record any commands"
        );
    }
}
