// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Direct FFI bindings for Metal Indirect Command Buffer API.
//!
//! The metal-rs crate doesn't expose ICB APIs, so we bind them directly via objc.

use metal::{Device, MTLSize};
use objc::runtime::{Object, BOOL, NO, YES};
use objc::{class, msg_send, sel, sel_impl};

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
    descriptor: *mut Object,
}

impl IndirectCommandBufferDescriptor {
    /// Create a new indirect command buffer descriptor
    pub fn new() -> Self {
        unsafe {
            let descriptor: *mut Object =
                msg_send![class!(MTLIndirectCommandBufferDescriptor), alloc];
            let descriptor: *mut Object = msg_send![descriptor, init];
            Self { descriptor }
        }
    }

    /// Set the command types this ICB will contain
    pub fn set_command_types(&self, types: u64) {
        unsafe {
            let _: () = msg_send![self.descriptor, setCommandTypes: types];
        }
    }

    /// Set whether commands inherit pipeline state from the encoder
    /// CRITICAL: On Apple Silicon, this MUST be set to true to avoid crashes
    /// when encoding ICB commands. When true, pipeline state is set on the
    /// compute encoder, not on individual ICB commands.
    pub fn set_inherit_pipeline_state(&self, inherit: bool) {
        unsafe {
            let inherit_bool: BOOL = if inherit { YES } else { NO };
            let _: () = msg_send![self.descriptor, setInheritPipelineState: inherit_bool];
        }
    }

    /// Set whether commands inherit buffers
    pub fn set_inherit_buffers(&self, inherit: bool) {
        unsafe {
            let inherit_bool: BOOL = if inherit { YES } else { NO };
            let _: () = msg_send![self.descriptor, setInheritBuffers: inherit_bool];
        }
    }

    /// Set maximum vertex buffer bind count
    pub fn set_max_vertex_buffer_bind_count(&self, count: u64) {
        unsafe {
            let _: () = msg_send![self.descriptor, setMaxVertexBufferBindCount: count];
        }
    }

    /// Set maximum fragment buffer bind count
    pub fn set_max_fragment_buffer_bind_count(&self, count: u64) {
        unsafe {
            let _: () = msg_send![self.descriptor, setMaxFragmentBufferBindCount: count];
        }
    }

    /// Set maximum kernel buffer bind count (for compute)
    pub fn set_max_kernel_buffer_bind_count(&self, count: u64) {
        unsafe {
            let _: () = msg_send![self.descriptor, setMaxKernelBufferBindCount: count];
        }
    }

    /// Get the raw descriptor pointer
    pub fn as_ptr(&self) -> *mut Object {
        self.descriptor
    }
}

// Note: Metal manages the descriptor's lifetime via ARC, no manual release needed

/// Wrapper for MTLIndirectCommandBuffer
pub struct IndirectCommandBuffer {
    buffer: *mut Object,
}

impl IndirectCommandBuffer {
    /// Create a new indirect command buffer from a Metal device
    pub fn new(
        device: &Device,
        descriptor: &IndirectCommandBufferDescriptor,
        max_count: u64,
        options: u64,
    ) -> Result<Self, String> {
        use foreign_types::ForeignType;
        unsafe {
            let buffer: *mut Object = msg_send![
                device.as_ptr() as *mut Object,
                newIndirectCommandBufferWithDescriptor: descriptor.as_ptr()
                maxCommandCount: max_count
                options: options
            ];

            if buffer.is_null() {
                return Err("Failed to create indirect command buffer".to_string());
            }

            Ok(Self { buffer })
        }
    }

    /// Get an indirect compute command at the specified index
    pub fn indirect_compute_command_at(&self, index: u64) -> IndirectComputeCommand {
        unsafe {
            let command: *mut Object = msg_send![self.buffer, indirectComputeCommandAtIndex: index];
            IndirectComputeCommand { command }
        }
    }

    /// Get the size in bytes
    pub fn size(&self) -> u64 {
        unsafe { msg_send![self.buffer, size] }
    }

    /// Get the GPU resource ID
    pub fn gpu_resource_id(&self) -> u64 {
        unsafe { msg_send![self.buffer, gpuResourceID] }
    }

    /// Get the raw buffer pointer
    pub fn as_ptr(&self) -> *mut Object {
        self.buffer
    }

    /// Reset a range of commands in the ICB
    pub fn reset_with_range(&self, range: std::ops::Range<u64>) {
        unsafe {
            let ns_range = metal::NSRange {
                location: range.start,
                length: range.end - range.start,
            };
            let _: () = msg_send![self.buffer, resetWithRange: ns_range];
        }
    }
}

// Note: Metal manages the ICB's lifetime via ARC, no manual release needed

// MTLIndirectCommandBuffer is documented as thread-safe for refcounting and for
// recording-from-one-thread / executing-from-another. The ferrite-metal pool
// (Phase 5.D) needs to move a baked ICB between threads (a checked-out worker
// runs on a different thread than the one that recorded it). The pool's
// semaphore-bounded checkout guarantees a given ICB is touched by at most one
// thread at a time, which makes Send sound. Sync is intentionally NOT
// implemented — concurrent mutation of `command_index` / `reset_with_range`
// from multiple threads is not supported.
unsafe impl Send for IndirectCommandBuffer {}

/// Wrapper for MTLIndirectComputeCommand
pub struct IndirectComputeCommand {
    command: *mut Object,
}

impl IndirectComputeCommand {
    /// Set the compute pipeline state
    pub fn set_compute_pipeline_state(&self, pipeline: *mut Object) {
        unsafe {
            let _: () = msg_send![self.command, setComputePipelineState: pipeline];
        }
    }

    /// Set a kernel buffer at the specified index
    pub fn set_kernel_buffer(&self, buffer: *mut Object, offset: u64, index: u64) {
        unsafe {
            let _: () = msg_send![
                self.command,
                setKernelBuffer: buffer
                offset: offset
                atIndex: index
            ];
        }
    }

    /// Set threadgroup memory length at index
    pub fn set_threadgroup_memory_length(&self, length: u64, index: u64) {
        unsafe {
            let _: () = msg_send![
                self.command,
                setThreadgroupMemoryLength: length
                atIndex: index
            ];
        }
    }

    /// Dispatch threadgroups with concurrent execution
    pub fn concurrent_dispatch_threadgroups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        unsafe {
            msg_send![
                self.command,
                concurrentDispatchThreadgroups: threadgroups_per_grid
                threadsPerThreadgroup: threads_per_threadgroup
            ]
        }
    }

    /// Dispatch threads with concurrent execution
    pub fn concurrent_dispatch_threads(
        &self,
        threads_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        unsafe {
            let _: () = msg_send![
                self.command,
                concurrentDispatchThreads: threads_per_grid
                threadsPerThreadgroup: threads_per_threadgroup
            ];
        }
    }

    /// Reset the command
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

        // Set some properties
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

    #[test]
    fn test_indirect_compute_command() {
        let device = detect_device().expect("Metal device required");
        let desc = IndirectCommandBufferDescriptor::new();
        desc.set_command_types(MTLIndirectCommandType::ConcurrentDispatch as u64);
        desc.set_max_kernel_buffer_bind_count(31);

        let icb =
            IndirectCommandBuffer::new(&device.device, &desc, 10, 0).expect("Failed to create ICB");

        // Get a command at index 0
        let cmd = icb.indirect_compute_command_at(0);

        // Test that we can call methods on it (won't actually execute anything)
        cmd.reset();
    }
}
