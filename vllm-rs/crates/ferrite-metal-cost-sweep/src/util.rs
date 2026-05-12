// SPDX-License-Identifier: Apache-2.0
//! Utilities for Metal cost sweep: device init, timing, overhead measurement.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLResourceOptions,
};
use std::sync::OnceLock;
use std::time::Instant;

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type Buffer = Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>;

static DEVICE: OnceLock<Device> = OnceLock::new();

pub fn init_metal() {
    let dev = MTLCreateSystemDefaultDevice();
    if dev.is_none() {
        eprintln!("ERROR: No Metal device found. Are you running on Apple Silicon?");
        std::process::exit(1);
    }
    DEVICE.set(dev.unwrap()).ok();
    eprintln!("Metal device initialized: {}", device_name());
}

pub fn device() -> &'static Device {
    DEVICE
        .get()
        .expect("Metal device not initialized. Call init_metal() first.")
}

pub fn device_name() -> String {
    device().name().to_string()
}

pub fn new_command_queue() -> CommandQueue {
    device()
        .newCommandQueue()
        .expect("newCommandQueue returned nil")
}

/// Empty-command-buffer overhead — submitted commit + wait round trip
/// with no encoded work. Subtracted from all kernel timings so the CSV
/// reports compute-only cost per the convention in the CUDA sweep.
pub fn measure_launch_overhead() -> f64 {
    let queue = new_command_queue();
    for _ in 0..10 {
        let cb = queue.commandBuffer().expect("commandBuffer");
        cb.commit();
        cb.waitUntilCompleted();
    }
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let cb = queue.commandBuffer().expect("commandBuffer");
        cb.commit();
        cb.waitUntilCompleted();
    }
    let elapsed = start.elapsed();
    let overhead_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    eprintln!("Launch overhead: {overhead_us:.2} us");
    overhead_us
}

/// Run `f` `iterations` times after `warmup` warm-up calls; return the
/// average wall-time per call in microseconds, minus the empty-cmdbuf
/// `launch_overhead_us`. `f` is responsible for encoding + committing
/// + waiting on one command buffer.
pub fn time_kernel<F>(launch_overhead_us: f64, warmup: u32, iterations: u32, mut f: F) -> f64
where
    F: FnMut(),
{
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let total_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    (total_us - launch_overhead_us).max(0.0)
}

/// Allocate a zero-filled `StorageModeShared` MTLBuffer of `n_bytes`.
pub fn create_buffer(n_bytes: usize) -> Buffer {
    device()
        .newBufferWithLength_options(n_bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBufferWithLength returned nil")
}

/// Zero the contents of a `StorageModeShared` buffer. `create_buffer`
/// only allocates — Metal does not guarantee zero-init — so callers
/// that bind the buffer to a shader that *reads* it as index data
/// (positions, slot_mapping, etc.) must zero-fill first to avoid
/// OOB reads / writes from junk values.
pub fn zero_buffer(buf: &Buffer) {
    let len = buf.length();
    if len == 0 {
        return;
    }
    let ptr = buf.contents();
    unsafe {
        std::ptr::write_bytes(ptr.as_ptr() as *mut u8, 0u8, len);
    }
}
