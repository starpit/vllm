// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Instruction recording infrastructure for Metal Indirect Command Buffers (ICB).

pub mod attention;
pub mod fused;
pub mod gemm;
pub mod icb_ffi;
pub mod rmsnorm;

#[cfg(test)]
mod test_minimal_recording;

#[cfg(test)]
mod test_icb_execution;

#[cfg(test)]
mod test_direct_vs_icb;

use icb_ffi::{IndirectCommandBuffer, IndirectCommandBufferDescriptor, MTLIndirectCommandType};
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{msg_send, sel};
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize,
};
use std::sync::Arc;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type ComputeCommandEncoderRef = ProtocolObject<dyn MTLComputeCommandEncoder>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;

/// Context for recording instructions into an ICB.
pub struct RecordingContext {
    pub device: Arc<Device>,
    pub icb: IndirectCommandBuffer,
    pub command_index: usize,
}

impl RecordingContext {
    pub fn new(device: Arc<Device>, max_commands: usize) -> Result<Self, String> {
        let descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.set_command_types(MTLIndirectCommandType::ConcurrentDispatch as u64);
        descriptor.set_max_kernel_buffer_bind_count(31);
        descriptor.set_inherit_pipeline_state(true);
        descriptor.set_inherit_buffers(false);

        let icb = IndirectCommandBuffer::new(&device, &descriptor, max_commands as u64, 0)?;

        Ok(Self {
            device,
            icb,
            command_index: 0,
        })
    }

    pub fn command_count(&self) -> usize {
        self.command_index
    }

    pub fn record_compute_dispatch(
        &mut self,
        _pipeline: &ComputePipelineState, // Pipeline inherited from encoder
        buffers: &[(&Buffer, u64, u64)],
        threadgroups: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        let command = self
            .icb
            .indirect_compute_command_at(self.command_index as u64);

        command.reset();

        for &(buffer, offset, index) in buffers {
            let buf_ptr: *mut AnyObject =
                Retained::as_ptr(buffer) as *const AnyObject as *mut AnyObject;
            command.set_kernel_buffer(buf_ptr, offset, index);
        }

        command.concurrent_dispatch_threadgroups(threadgroups, threads_per_threadgroup);

        self.command_index += 1;
    }

    pub fn icb(&self) -> &IndirectCommandBuffer {
        &self.icb
    }

    pub fn reset_range(&self, range: std::ops::Range<usize>) {
        self.icb
            .reset_with_range(range.start as u64..range.end as u64);
    }

    pub fn execute_on_encoder(
        &self,
        encoder: &ComputeCommandEncoderRef,
        range: std::ops::Range<usize>,
    ) {
        unsafe {
            let ns_range = NSRange {
                location: range.start,
                length: range.end - range.start,
            };
            let encoder_ptr: *mut AnyObject =
                encoder as *const ComputeCommandEncoderRef as *const AnyObject as *mut AnyObject;
            let _: () = msg_send![
                encoder_ptr,
                executeCommandsInBuffer: self.icb.as_ptr(),
                withRange: ns_range
            ];
        }
    }
}

pub fn dispatch_1d(total_threads: usize, threads_per_group: usize) -> (MTLSize, MTLSize) {
    let num_groups = (total_threads + threads_per_group - 1) / threads_per_group;
    (
        MTLSize {
            width: num_groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads_per_group,
            height: 1,
            depth: 1,
        },
    )
}

pub fn dispatch_2d(
    width: usize,
    height: usize,
    tile_width: usize,
    tile_height: usize,
) -> (MTLSize, MTLSize) {
    let groups_x = (width + tile_width - 1) / tile_width;
    let groups_y = (height + tile_height - 1) / tile_height;
    (
        MTLSize {
            width: groups_x,
            height: groups_y,
            depth: 1,
        },
        MTLSize {
            width: tile_width,
            height: tile_height,
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
        assert_eq!(tg.width, 4);
        assert_eq!(tg.height, 1);
        assert_eq!(tg.depth, 1);
        assert_eq!(tpt.width, 256);
        assert_eq!(tpt.height, 1);
        assert_eq!(tpt.depth, 1);
    }

    #[test]
    fn test_dispatch_2d_calculation() {
        let (tg, tpt) = dispatch_2d(1024, 768, 16, 16);
        assert_eq!(tg.width, 64);
        assert_eq!(tg.height, 48);
        assert_eq!(tg.depth, 1);
        assert_eq!(tpt.width, 16);
        assert_eq!(tpt.height, 16);
        assert_eq!(tpt.depth, 1);
    }
}
