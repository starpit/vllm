// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal kernel infrastructure for ferrite.
//!
//! Provides Metal device management, shader compilation, and kernel dispatch
//! primitives for Apple Silicon GPUs.

pub mod activation;
pub mod allocator;
pub mod argmax;
pub mod awq;
pub mod device;
pub mod fused_kernels;
pub mod gemm;
pub mod residency;
pub mod rope;
pub mod shader_cache;
pub mod specialized_pipeline_cache;
pub mod stream;

/// Re-export of the upstream `metal` crate so downstream callers
/// (notably `ferrite-forward::interpreter::metal::pipelines`) can
/// reach Metal types without taking their own `metal` dependency
/// — the version stays pinned here.
pub use metal;

// Instruction recording for ICB execution
pub mod instruction_executor;

use metal::{Device, MTLResourceOptions};

/// Embed a precompiled `.metallib` produced by `build.rs` from
/// `shaders/<name>.metal`. Returns a `&'static [u8]` suitable for
/// `Device::new_library_with_data`. Replaces the runtime-MSL-compile
/// path that called `Device::new_library_with_source(include_str!…)`
/// — the AoT version skips the MSL→AIR frontend on every process
/// start.
#[macro_export]
macro_rules! embedded_metallib {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".metallib"))
    };
}

pub use allocator::{AllocatorError, MetalAllocator, PooledBuffer};
pub use device::{detect_device, MetalDevice};
pub use stream::{wait_for_completion, MetalStream, MetalStreamError};

/// Metal buffer wrapper with automatic memory management
pub struct MetalBuffer {
    buffer: metal::Buffer,
    size: usize,
}

impl MetalBuffer {
    pub fn new(device: &Device, size: usize) -> Self {
        let buffer = device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
        Self { buffer, size }
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.buffer.contents()
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn metal_buffer(&self) -> &metal::Buffer {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_detection() {
        let device = detect_device();
        assert!(device.is_some(), "Should detect Metal device on macOS");
    }

    #[test]
    fn test_buffer_creation() {
        let device = detect_device().expect("Metal device required");
        let buffer = MetalBuffer::new(&device.device, 1024);
        assert_eq!(buffer.size(), 1024);
    }
}
