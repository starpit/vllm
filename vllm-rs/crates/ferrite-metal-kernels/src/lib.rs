// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal kernel infrastructure for ferrite.
//!
//! Provides Metal device management, shader compilation, and kernel dispatch
//! primitives for Apple Silicon GPUs.

// Kernel-launch / dispatch / record entry points pass the full set of
// buffers, dims, scales, and offsets positionally — bundling them into
// param structs would only obscure the 1:1 mapping to the MSL kernel
// signatures, so `too_many_arguments` is expected here. Likewise the
// CPU-reference helpers mirror those wide signatures. `type_complexity`
// is allowed for the same shader-table lookup return tuples.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

/// Compile-time-generated kernel instantiation tables. The
/// `STEEL_PAGED_HEAD_DIMS` slice and `steel_paged_symbol()` lookup
/// in here are emitted by `build.rs` from the same head-dim list
/// that produces `attention_steel_paged_instantiations.h`. The
/// dispatcher in `ferrite-forward/.../lowering.rs` must consume
/// these — never hand-list head dims at the call site.
pub mod steel_paged {
    include!(concat!(
        env!("OUT_DIR"),
        "/steel_paged_kernels_generated.rs"
    ));
}

pub mod activation;
pub mod allocator;
pub mod argmax;
pub mod argpartition;
pub mod chain_advance;
pub mod cpu_reference;
pub mod device;
pub mod fused_kernels;
pub mod gate_scale;
pub mod gemm;
pub mod moe_weighted_sum;
pub mod quantized;
pub mod residency;
pub mod rope;
pub mod shader_cache;
pub mod single_buffer_kv;
pub mod slice_trailing_cols;
pub mod softmax;
pub mod specialized_pipeline_cache;
pub mod stream;
pub mod take_along_axis;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// Embed a precompiled `.metallib` produced by `build.rs` from
/// `shaders/<name>.metal`. Returns a `&'static [u8]` suitable for
/// `device.newLibraryWithData_error`. Replaces the runtime-MSL-compile
/// path that called `newLibraryWithSource:options:error:` —
/// the AoT version skips the MSL→AIR frontend on every process start.
#[macro_export]
macro_rules! embedded_metallib {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".metallib"))
    };
}

pub use allocator::{AllocatorError, MetalAllocator, PooledBuffer};
pub use device::{detect_device, MetalDevice};
pub use stream::{wait_for_completion, MetalStream, MetalStreamError};
// Re-export the targets crate so downstream consumers (ferrite-forward
// runtime lowering) can name `MetalTargetProfile` without taking
// a direct dep on `ferrite-metal-targets`.
pub use ferrite_metal_targets;

/// Metal buffer wrapper with automatic memory management
pub struct MetalBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    size: usize,
}

impl MetalBuffer {
    pub fn new(device: &Retained<ProtocolObject<dyn MTLDevice>>, size: usize) -> Self {
        let buffer = device
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .expect("newBufferWithLength returned nil");
        Self { buffer, size }
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.buffer.contents().as_ptr()
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn metal_buffer(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
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
