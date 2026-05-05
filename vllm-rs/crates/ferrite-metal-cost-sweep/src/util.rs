// SPDX-License-Identifier: Apache-2.0
//! Utilities for Metal cost sweep: device init, timing, overhead measurement.

use metal::{Device, MTLResourceOptions};
use std::sync::OnceLock;
use std::time::Instant;

/// Global Metal device handle. Initialized once in `init_metal()`.
static DEVICE: OnceLock<Device> = OnceLock::new();

/// Initialize Metal device. Called once from `main()`.
pub fn init_metal() {
    let dev = Device::system_default();
    if dev.is_none() {
        eprintln!("ERROR: No Metal device found. Are you running on Apple Silicon?");
        std::process::exit(1);
    }
    DEVICE.set(dev.unwrap()).ok();
    eprintln!("Metal device initialized: {}", device_name());
}

/// Get the Metal device handle.
pub fn device() -> &'static Device {
    DEVICE
        .get()
        .expect("Metal device not initialized. Call init_metal() first.")
}

/// Get the device name for logging.
pub fn device_name() -> String {
    device().name().to_string()
}

/// Measure launch overhead by timing an empty command buffer.
/// Returns overhead in microseconds.
pub fn measure_launch_overhead() -> f64 {
    let device = device();
    let queue = device.new_command_queue();

    // Warm-up: run a few empty command buffers to stabilize timing
    for _ in 0..10 {
        let cmd_buffer = queue.new_command_buffer();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    // Measure: average over 100 runs
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let cmd_buffer = queue.new_command_buffer();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }
    let elapsed = start.elapsed();

    let overhead_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    eprintln!("Launch overhead: {overhead_us:.2} us");
    overhead_us
}

/// Time a Metal kernel execution.
/// Returns elapsed time in microseconds, with launch overhead subtracted.
pub fn time_kernel<F>(launch_overhead_us: f64, mut f: F) -> f64
where
    F: FnMut(),
{
    let _device = device();
    let _queue = _device.new_command_queue();

    // Warm-up: run kernel 3 times
    for _ in 0..3 {
        f();
    }

    // Measure: average over multiple runs
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();

    let total_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let compute_us = total_us - launch_overhead_us;
    compute_us.max(0.0) // Ensure non-negative
}

/// Create a Metal buffer with the given size in bytes.
pub fn create_buffer(size_bytes: usize) -> metal::Buffer {
    let device = device();
    device.new_buffer(size_bytes as u64, MTLResourceOptions::StorageModeShared)
}
