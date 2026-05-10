// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Direct objc2 bindings for Metal Indirect Command Buffer API.
//!
//! objc2-metal 0.3 doesn't fully expose ICB recording APIs (the
//! per-command setBuffer/setBytes/dispatch methods); we go through
//! `msg_send!` for those.

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool, ProtocolObject};
use objc2::{class, msg_send, sel};
use objc2_foundation::NSRange;
use objc2_metal::{MTLDevice, MTLSize};

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;

/// MTLIndirectCommandType enum values
#[repr(u64)]
#[allow(dead_code)]
pub enum MTLIndirectCommandType {
    Draw = 1 << 0,
    DrawIndexed = 1 << 1,
    DrawPatches = 1 << 2,
    DrawIndexedPatches = 1 << 3,
    ConcurrentDispatch = 1 << 5,
    ConcurrentDispatchThreads = 1 << 6,
}

/// Wrapper for MTLIndirectCommandBufferDescriptor
pub struct IndirectCommandBufferDescriptor {
    descriptor: *mut AnyObject,
}

impl IndirectCommandBufferDescriptor {
    pub fn new() -> Self {
        unsafe {
            let descriptor: *mut AnyObject =
                msg_send![class!(MTLIndirectCommandBufferDescriptor), alloc];
            let descriptor: *mut AnyObject = msg_send![descriptor, init];
            Self { descriptor }
        }
    }

    pub fn set_command_types(&self, types: u64) {
        unsafe {
            let _: () = msg_send![self.descriptor, setCommandTypes: types];
        }
    }

    /// Set whether commands inherit pipeline state from the encoder.
    /// CRITICAL: On Apple Silicon, this MUST be set to true to avoid
    /// crashes when encoding ICB commands.
    pub fn set_inherit_pipeline_state(&self, inherit: bool) {
        unsafe {
            let inherit_bool = Bool::new(inherit);
            let _: () = msg_send![self.descriptor, setInheritPipelineState: inherit_bool];
        }
    }

    pub fn set_inherit_buffers(&self, inherit: bool) {
        unsafe {
            let inherit_bool = Bool::new(inherit);
            let _: () = msg_send![self.descriptor, setInheritBuffers: inherit_bool];
        }
    }

    pub fn set_max_vertex_buffer_bind_count(&self, count: u64) {
        unsafe {
            let _: () = msg_send![self.descriptor, setMaxVertexBufferBindCount: count];
        }
    }

    pub fn set_max_fragment_buffer_bind_count(&self, count: u64) {
        unsafe {
            let _: () = msg_send![self.descriptor, setMaxFragmentBufferBindCount: count];
        }
    }

    pub fn set_max_kernel_buffer_bind_count(&self, count: u64) {
        unsafe {
            let _: () = msg_send![self.descriptor, setMaxKernelBufferBindCount: count];
        }
    }

    pub fn as_ptr(&self) -> *mut AnyObject {
        self.descriptor
    }
}

// Note: Metal manages the descriptor's lifetime via ARC, no manual release needed.

/// Wrapper for MTLIndirectCommandBuffer
pub struct IndirectCommandBuffer {
    buffer: *mut AnyObject,
}

impl IndirectCommandBuffer {
    pub fn new(
        device: &Device,
        descriptor: &IndirectCommandBufferDescriptor,
        max_count: u64,
        options: u64,
    ) -> Result<Self, String> {
        unsafe {
            let device_ptr: *mut AnyObject =
                Retained::as_ptr(device) as *const AnyObject as *mut AnyObject;
            let buffer: *mut AnyObject = msg_send![
                device_ptr,
                newIndirectCommandBufferWithDescriptor: descriptor.as_ptr(),
                maxCommandCount: max_count,
                options: options
            ];

            if buffer.is_null() {
                return Err("Failed to create indirect command buffer".to_string());
            }

            Ok(Self { buffer })
        }
    }

    pub fn indirect_compute_command_at(&self, index: u64) -> IndirectComputeCommand {
        unsafe {
            let command: *mut AnyObject =
                msg_send![self.buffer, indirectComputeCommandAtIndex: index];
            IndirectComputeCommand { command }
        }
    }

    pub fn size(&self) -> u64 {
        unsafe { msg_send![self.buffer, size] }
    }

    pub fn gpu_resource_id(&self) -> u64 {
        unsafe { msg_send![self.buffer, gpuResourceID] }
    }

    pub fn as_ptr(&self) -> *mut AnyObject {
        self.buffer
    }

    pub fn reset_with_range(&self, range: std::ops::Range<u64>) {
        unsafe {
            let ns_range = NSRange {
                location: range.start as usize,
                length: (range.end - range.start) as usize,
            };
            let _: () = msg_send![self.buffer, resetWithRange: ns_range];
        }
    }
}

unsafe impl Send for IndirectCommandBuffer {}

/// Wrapper for MTLIndirectComputeCommand
pub struct IndirectComputeCommand {
    command: *mut AnyObject,
}

impl IndirectComputeCommand {
    pub fn set_compute_pipeline_state(&self, pipeline: *mut AnyObject) {
        unsafe {
            let _: () = msg_send![self.command, setComputePipelineState: pipeline];
        }
    }

    pub fn set_kernel_buffer(&self, buffer: *mut AnyObject, offset: u64, index: u64) {
        unsafe {
            let _: () = msg_send![
                self.command,
                setKernelBuffer: buffer,
                offset: offset,
                atIndex: index
            ];
        }
    }

    pub fn set_threadgroup_memory_length(&self, length: u64, index: u64) {
        unsafe {
            let _: () = msg_send![
                self.command,
                setThreadgroupMemoryLength: length,
                atIndex: index
            ];
        }
    }

    pub fn concurrent_dispatch_threadgroups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        unsafe {
            let _: () = msg_send![
                self.command,
                concurrentDispatchThreadgroups: threadgroups_per_grid,
                threadsPerThreadgroup: threads_per_threadgroup
            ];
        }
    }

    pub fn concurrent_dispatch_threads(
        &self,
        threads_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        unsafe {
            let _: () = msg_send![
                self.command,
                concurrentDispatchThreads: threads_per_grid,
                threadsPerThreadgroup: threads_per_threadgroup
            ];
        }
    }

    pub fn reset(&self) {
        unsafe {
            let _: () = msg_send![self.command, reset];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect_device;

    #[test]
    fn test_icb_descriptor_creation() {
        let desc = IndirectCommandBufferDescriptor::new();
        assert!(!desc.as_ptr().is_null());
        desc.set_command_types(MTLIndirectCommandType::ConcurrentDispatch as u64);
        desc.set_max_kernel_buffer_bind_count(31);
        desc.set_inherit_pipeline_state(false);
    }

    #[test]
    fn test_icb_creation() {
        let device = detect_device().expect("Metal device required");
        let desc = IndirectCommandBufferDescriptor::new();
        desc.set_command_types(MTLIndirectCommandType::ConcurrentDispatch as u64);
        desc.set_max_kernel_buffer_bind_count(31);

        let icb = IndirectCommandBuffer::new(&device.device, &desc, 100, 0);
        assert!(icb.is_ok(), "Failed to create ICB: {:?}", icb.err());
        let icb = icb.unwrap();
        assert!(icb.size() > 0);
    }
}
