// SPDX-License-Identifier: Apache-2.0
//! `GpuDevice` (metal arm). Same NAME as the cuda struct in
//! `device.rs` so any downstream code that names `&mut GpuDevice` —
//! including the `FerriteWeights::forward` trait method — resolves
//! against this struct under `cfg(feature = "metal")` and against
//! the cuda one under `cfg(feature = "cuda")`. The two structs are
//! cfg-mutex'd: only one ever exists in a single build.
//!
//! Carries the metal-equivalent of cuda's "GPU runtime context":
//!   - `device`     ← `metal::Device` (analog of `CUcontext`).
//!   - `queue`      ← `metal::CommandQueue` (analog of `CUstream`).
//!   - `allocator`  ← `MetalAllocator` (analog of `CachingAllocator`).
//!
//! Per-arch `forward` impl bodies invoke pool dispatch via these;
//! `FerriteWorker(metal)` constructs and owns the instance.

#![cfg(feature = "metal")]

use std::sync::Arc;

use metal::{CommandQueue, Device};

use crate::MetalAllocator;

/// Metal-side `GpuDevice`. Mirrors the cuda struct's role —
/// "everything kernel launches need" — at the Apple-silicon
/// equivalents (Device + CommandQueue + MetalAllocator).
pub struct GpuDevice {
    pub device: Arc<Device>,
    pub queue: CommandQueue,
    pub allocator: Arc<MetalAllocator>,
}

impl GpuDevice {
    /// Wrap an already-detected metal device + allocator with a
    /// fresh command queue. The worker calls this once at
    /// `init_device` time.
    pub fn new(device: Arc<Device>, allocator: Arc<MetalAllocator>) -> Self {
        let queue = device.new_command_queue();
        Self {
            device,
            queue,
            allocator,
        }
    }
}
