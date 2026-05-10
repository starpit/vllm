// SPDX-License-Identifier: Apache-2.0
//! `GpuDevice` (metal arm).

#![cfg(feature = "metal")]

use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice};

use crate::MetalAllocator;

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

/// Metal-side `GpuDevice`. Mirrors the cuda struct's role —
/// "everything kernel launches need" — at the Apple-silicon
/// equivalents (Device + CommandQueue + MetalAllocator).
pub struct GpuDevice {
    pub device: Arc<Device>,
    pub queue: CommandQueue,
    pub allocator: Arc<MetalAllocator>,
}

impl GpuDevice {
    pub fn new(device: Arc<Device>, allocator: Arc<MetalAllocator>) -> Self {
        let queue = device
            .newCommandQueue()
            .expect("newCommandQueue returned nil");
        Self {
            device,
            queue,
            allocator,
        }
    }
}
