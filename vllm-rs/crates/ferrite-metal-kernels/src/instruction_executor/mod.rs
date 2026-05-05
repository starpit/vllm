// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Instruction recording infrastructure for Metal Indirect Command Buffers (ICB).
//!
//! This module provides the core infrastructure for recording ferrite-forward
//! instructions into Metal ICBs for efficient GPU execution.

pub mod activation;
pub mod attention;
pub mod elementwise;
pub mod embed;
pub mod fused;
pub mod gemm;
pub mod icb_ffi;
pub mod reshape;
pub mod rmsnorm;
pub mod rope;

#[cfg(test)]
mod test_minimal_recording;

#[cfg(test)]
mod test_icb_execution;

#[cfg(test)]
mod test_direct_vs_icb;

#[cfg(test)]
mod test_full_sequence;

use foreign_types::ForeignType;
use icb_ffi::{IndirectCommandBuffer, IndirectCommandBufferDescriptor, MTLIndirectCommandType};
use metal::{ComputePipelineState, Device, MTLSize};
use std::sync::Arc;

/// Context for recording instructions into an ICB.
///
/// Holds the Metal device, shader cache, and ICB being recorded into.
/// Passed by `&mut` to each instruction's recording method.
pub struct RecordingContext {
    pub device: Arc<Device>,
    pub icb: IndirectCommandBuffer,
    pub command_index: usize,
}

impl RecordingContext {
    /// Create a new recording context with an ICB of the given capacity.
    ///
    /// CRITICAL: Uses `inheritPipelineState = true` to avoid crashes on Apple Silicon.
    /// This means:
    /// - Pipeline state is set on the compute encoder, NOT on ICB commands
    /// - ICB commands only set buffers and dispatch parameters
    /// - All commands in the ICB share the same pipeline state from the encoder
    pub fn new(device: Arc<Device>, max_commands: usize) -> Result<Self, String> {
        let descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.set_command_types(MTLIndirectCommandType::ConcurrentDispatch as u64);
        descriptor.set_max_kernel_buffer_bind_count(31); // Max buffer bindings per kernel
        descriptor.set_inherit_pipeline_state(true); // CRITICAL: Must be true on Apple Silicon
        descriptor.set_inherit_buffers(false);

        let icb = IndirectCommandBuffer::new(&device, &descriptor, max_commands as u64, 0)?;

        Ok(Self {
            device,
            icb,
            command_index: 0,
        })
    }

    /// Get the number of commands recorded so far.
    pub fn command_count(&self) -> usize {
        self.command_index
    }

    /// Record a compute dispatch into the ICB at the current command index.
    ///
    /// IMPORTANT: This does NOT set the pipeline state on the ICB command.
    /// The pipeline state must be set on the compute encoder before executing the ICB.
    /// This is required to avoid crashes on Apple Silicon when using inheritPipelineState=true.
    ///
    /// # Arguments
    /// * `pipeline` - IGNORED - kept for API compatibility but not used
    /// * `buffers` - Slice of (buffer, offset, index) tuples to bind
    /// * `threadgroups` - Number of threadgroups to dispatch
    /// * `threads_per_threadgroup` - Threads per threadgroup
    pub fn record_compute_dispatch(
        &mut self,
        _pipeline: &ComputePipelineState, // Ignored - pipeline inherited from encoder
        buffers: &[(&metal::Buffer, u64, u64)],
        threadgroups: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        let command = self
            .icb
            .indirect_compute_command_at(self.command_index as u64);

        // Reset command before encoding
        command.reset();

        // Bind buffers
        for &(buffer, offset, index) in buffers {
            command.set_kernel_buffer(buffer.as_ptr() as *mut _, offset, index);
        }

        // Set dispatch size
        command.concurrent_dispatch_threadgroups(threadgroups, threads_per_threadgroup);

        self.command_index += 1;
    }

    /// Get a reference to the underlying ICB for execution.
    pub fn icb(&self) -> &IndirectCommandBuffer {
        &self.icb
    }

    /// Reset the ICB commands in the given range and reset command counter to start
    pub fn reset_range(&mut self, range: std::ops::Range<usize>) {
        self.icb
            .reset_with_range(range.start as u64..range.end as u64);
        // Reset command index to the start of the range so re-recording works correctly
        self.command_index = range.start;
    }

    /// Execute the recorded commands on a compute encoder
    ///
    /// CRITICAL: The pipeline state must be set on the encoder BEFORE calling this,
    /// as ICB commands inherit the pipeline state from the encoder.
    pub fn execute_on_encoder(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        range: std::ops::Range<usize>,
    ) {
        unsafe {
            use objc::{msg_send, sel, sel_impl};
            let ns_range = metal::NSRange {
                location: range.start as u64,
                length: (range.end - range.start) as u64,
            };
            let encoder_ptr = encoder as *const _ as *mut objc::runtime::Object;
            let _: () = msg_send![
                encoder_ptr,
                executeCommandsInBuffer: self.icb.as_ptr()
                withRange: ns_range
            ];
        }
    }
}

/// Helper to calculate 1D dispatch parameters.
///
/// # Arguments
/// * `total_threads` - Total number of threads needed
/// * `threads_per_group` - Preferred threads per threadgroup (typically 256)
///
/// # Returns
/// (threadgroups, threads_per_threadgroup) as MTLSize
pub fn dispatch_1d(total_threads: usize, threads_per_group: usize) -> (MTLSize, MTLSize) {
    let num_groups = total_threads.div_ceil(threads_per_group);
    (
        MTLSize {
            width: num_groups as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads_per_group as u64,
            height: 1,
            depth: 1,
        },
    )
}

/// Helper to calculate 2D dispatch parameters.
///
/// # Arguments
/// * `width` - Width dimension
/// * `height` - Height dimension
/// * `tile_width` - Tile width (typically 16 or 32)
/// * `tile_height` - Tile height (typically 16 or 32)
///
/// # Returns
/// (threadgroups, threads_per_threadgroup) as MTLSize
pub fn dispatch_2d(
    width: usize,
    height: usize,
    tile_width: usize,
    tile_height: usize,
) -> (MTLSize, MTLSize) {
    let groups_x = width.div_ceil(tile_width);
    let groups_y = height.div_ceil(tile_height);
    (
        MTLSize {
            width: groups_x as u64,
            height: groups_y as u64,
            depth: 1,
        },
        MTLSize {
            width: tile_width as u64,
            height: tile_height as u64,
            depth: 1,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recording_context_creation() {
        let device = crate::detect_device().expect("Metal device required");
        let ctx = RecordingContext::new(Arc::new(device.device.clone()), 100)
            .expect("Failed to create recording context");
        assert_eq!(ctx.command_count(), 0);
    }

    #[test]
    fn test_dispatch_1d_calculation() {
        let (tg, tpt) = dispatch_1d(1000, 256);
        assert_eq!(tg.width, 4); // ceil(1000 / 256) = 4
        assert_eq!(tg.height, 1);
        assert_eq!(tg.depth, 1);
        assert_eq!(tpt.width, 256);
        assert_eq!(tpt.height, 1);
        assert_eq!(tpt.depth, 1);
    }

    #[test]
    fn test_dispatch_2d_calculation() {
        let (tg, tpt) = dispatch_2d(1024, 768, 16, 16);
        assert_eq!(tg.width, 64); // ceil(1024 / 16) = 64
        assert_eq!(tg.height, 48); // ceil(768 / 16) = 48
        assert_eq!(tg.depth, 1);
        assert_eq!(tpt.width, 16);
        assert_eq!(tpt.height, 16);
        assert_eq!(tpt.depth, 1);
    }
}
