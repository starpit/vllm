// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Metal kernel execution.
//!
//! These tests verify the complete pipeline:
//! - Device detection and initialization
//! - Buffer allocation and management
//! - Shader compilation and pipeline creation
//! - Kernel execution with real GPU work
//! - Result verification

use ferrite_metal_kernels::{detect_device, MetalAllocator, MetalStream};
use metal::{CompileOptions, MTLResourceOptions};

/// Test basic Metal device initialization and buffer allocation
#[test]
fn test_device_and_buffer_allocation() {
    let device = detect_device().expect("Metal device required for integration tests");
    let allocator = MetalAllocator::new(&device.device);

    // Allocate a buffer
    let buffer = allocator
        .allocate(1024 * 1024)
        .expect("Should allocate 1MB buffer");
    assert!(buffer.size() >= 1024 * 1024);
    assert!(!buffer.contents().is_null());

    // Verify we can write to the buffer
    unsafe {
        let ptr = buffer.contents() as *mut f32;
        for i in 0..256 {
            *ptr.add(i) = i as f32;
        }

        // Verify the data
        for i in 0..256 {
            assert_eq!(*ptr.add(i), i as f32);
        }
    }
}

/// Test Metal stream creation and command buffer execution
#[test]
fn test_stream_execution() {
    let device = detect_device().expect("Metal device required");
    let mut stream = MetalStream::new(&device.device);

    // Execute a simple command buffer
    stream
        .with_command_buffer(|cmd_buf| {
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.end_encoding();
            Ok(())
        })
        .expect("Should execute command buffer");

    stream.synchronize().expect("Should synchronize");
}

/// Test simple compute shader execution (vector addition)
#[test]
fn test_simple_compute_shader() {
    let device = detect_device().expect("Metal device required");
    let allocator = MetalAllocator::new(&device.device);

    // Simple Metal shader for vector addition
    let shader_source = r#"
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void vector_add(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* result [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            result[id] = a[id] + b[id];
        }
    "#;

    // Compile shader
    let compile_options = CompileOptions::new();
    let library = device
        .device
        .new_library_with_source(shader_source, &compile_options)
        .expect("Should compile shader");

    let function = library
        .get_function("vector_add", None)
        .expect("Should find vector_add function");

    let pipeline = device
        .device
        .new_compute_pipeline_state_with_function(&function)
        .expect("Should create pipeline state");

    // Allocate buffers
    const SIZE: usize = 1024;
    let buffer_a = allocator
        .allocate(SIZE * 4)
        .expect("Should allocate buffer A");
    let buffer_b = allocator
        .allocate(SIZE * 4)
        .expect("Should allocate buffer B");
    let buffer_result = allocator
        .allocate(SIZE * 4)
        .expect("Should allocate result buffer");

    // Initialize input data
    unsafe {
        let ptr_a = buffer_a.contents() as *mut f32;
        let ptr_b = buffer_b.contents() as *mut f32;

        for i in 0..SIZE {
            *ptr_a.add(i) = i as f32;
            *ptr_b.add(i) = (i * 2) as f32;
        }
    }

    // Execute kernel
    let mut stream = MetalStream::new(&device.device);
    stream
        .with_command_buffer(|cmd_buf| {
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(buffer_a.buffer()), 0);
            encoder.set_buffer(1, Some(buffer_b.buffer()), 0);
            encoder.set_buffer(2, Some(buffer_result.buffer()), 0);

            let grid_size = metal::MTLSize::new(SIZE as u64, 1, 1);
            let threadgroup_size = metal::MTLSize::new(256, 1, 1);
            encoder.dispatch_threads(grid_size, threadgroup_size);

            encoder.end_encoding();
            Ok(())
        })
        .expect("Should execute kernel");

    stream.synchronize().expect("Should synchronize");

    // Verify results
    unsafe {
        let ptr_result = buffer_result.contents() as *const f32;
        for i in 0..SIZE {
            let expected = (i + i * 2) as f32;
            let actual = *ptr_result.add(i);
            assert_eq!(actual, expected, "Mismatch at index {}", i);
        }
    }
}

/// Test RMSNorm kernel execution (if shader exists)
#[test]
fn test_rmsnorm_kernel_execution() {
    let device = detect_device().expect("Metal device required");
    let allocator = MetalAllocator::new(&device.device);

    // RMSNorm shader (simplified version for testing)
    let shader_source = r#"
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void rmsnorm_f16(
            device const half* input [[buffer(0)]],
            device const half* weight [[buffer(1)]],
            device half* output [[buffer(2)]],
            constant uint& hidden_size [[buffer(3)]],
            constant float& eps [[buffer(4)]],
            uint token_id [[thread_position_in_grid]]
        ) {
            // Calculate RMS
            float sum_sq = 0.0f;
            device const half* token_input = input + token_id * hidden_size;
            
            for (uint i = 0; i < hidden_size; i++) {
                float val = float(token_input[i]);
                sum_sq += val * val;
            }
            
            float rms = sqrt(sum_sq / float(hidden_size) + eps);
            float inv_rms = 1.0f / rms;
            
            // Normalize and scale
            device half* token_output = output + token_id * hidden_size;
            for (uint i = 0; i < hidden_size; i++) {
                float normalized = float(token_input[i]) * inv_rms;
                token_output[i] = half(normalized * float(weight[i]));
            }
        }
    "#;

    // Compile shader
    let compile_options = CompileOptions::new();
    let library = device
        .device
        .new_library_with_source(shader_source, &compile_options)
        .expect("Should compile RMSNorm shader");

    let function = library
        .get_function("rmsnorm_f16", None)
        .expect("Should find rmsnorm_f16 function");

    let pipeline = device
        .device
        .new_compute_pipeline_state_with_function(&function)
        .expect("Should create pipeline state");

    // Test parameters
    const NUM_TOKENS: usize = 4;
    const HIDDEN_SIZE: usize = 128;
    const EPS: f32 = 1e-5;

    // Allocate buffers
    let input_size = NUM_TOKENS * HIDDEN_SIZE * 2; // f16 = 2 bytes
    let weight_size = HIDDEN_SIZE * 2;
    let output_size = NUM_TOKENS * HIDDEN_SIZE * 2;

    let buffer_input = allocator
        .allocate(input_size)
        .expect("Should allocate input");
    let buffer_weight = allocator
        .allocate(weight_size)
        .expect("Should allocate weight");
    let buffer_output = allocator
        .allocate(output_size)
        .expect("Should allocate output");

    // Initialize input data (using f32 for simplicity, will be converted to f16 by Metal)
    unsafe {
        let ptr_input = buffer_input.contents() as *mut u16; // f16 as u16
        let ptr_weight = buffer_weight.contents() as *mut u16;

        // Simple test pattern: input = 1.0, weight = 1.0
        // Expected output after RMSNorm: ~1.0 (since all inputs are same)
        for i in 0..(NUM_TOKENS * HIDDEN_SIZE) {
            *ptr_input.add(i) = 0x3C00; // f16(1.0) in hex
        }

        for i in 0..HIDDEN_SIZE {
            *ptr_weight.add(i) = 0x3C00; // f16(1.0) in hex
        }
    }

    // Create parameter buffers
    let hidden_size_buffer = device.device.new_buffer_with_data(
        &(HIDDEN_SIZE as u32) as *const u32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let eps_buffer = device.device.new_buffer_with_data(
        &EPS as *const f32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    // Execute kernel
    let mut stream = MetalStream::new(&device.device);
    stream
        .with_command_buffer(|cmd_buf| {
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(buffer_input.buffer()), 0);
            encoder.set_buffer(1, Some(buffer_weight.buffer()), 0);
            encoder.set_buffer(2, Some(buffer_output.buffer()), 0);
            encoder.set_buffer(3, Some(&hidden_size_buffer), 0);
            encoder.set_buffer(4, Some(&eps_buffer), 0);

            let grid_size = metal::MTLSize::new(NUM_TOKENS as u64, 1, 1);
            let threadgroup_size = metal::MTLSize::new(1, 1, 1);
            encoder.dispatch_threads(grid_size, threadgroup_size);

            encoder.end_encoding();
            Ok(())
        })
        .expect("Should execute RMSNorm kernel");

    stream.synchronize().expect("Should synchronize");

    // Verify output is not zero (basic sanity check)
    unsafe {
        let ptr_output = buffer_output.contents() as *const u16;
        let first_value = *ptr_output;
        assert_ne!(first_value, 0, "Output should not be zero");

        // All outputs should be similar since all inputs were the same
        for i in 0..(NUM_TOKENS * HIDDEN_SIZE) {
            let value = *ptr_output.add(i);
            assert_ne!(value, 0, "Output at index {} should not be zero", i);
        }
    }
}

/// Test buffer pooling and reuse
#[test]
fn test_buffer_pooling_integration() {
    let device = detect_device().expect("Metal device required");
    let allocator = MetalAllocator::new(&device.device);

    // Allocate and drop multiple buffers
    for _ in 0..10 {
        let buffer = allocator.allocate(4096).expect("Should allocate");
        // Buffer is automatically returned to pool on drop
        drop(buffer);
    }

    // Check that buffers are being pooled
    if let Some((total_bytes, pooled_count)) = allocator.stats() {
        assert!(pooled_count > 0, "Should have buffers in pool");
        assert!(total_bytes > 0, "Should have allocated memory");
    }

    // Allocate again - should reuse from pool
    let buffer = allocator.allocate(4096).expect("Should allocate from pool");
    assert!(buffer.size() >= 4096);
}

/// Test concurrent kernel execution (multiple streams)
#[test]
fn test_concurrent_streams() {
    let device = detect_device().expect("Metal device required");

    // Create multiple streams
    let mut stream1 = MetalStream::new(&device.device);
    let mut stream2 = MetalStream::new(&device.device);

    // Execute work on both streams
    stream1
        .with_command_buffer(|cmd_buf| {
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.end_encoding();
            Ok(())
        })
        .expect("Stream 1 should execute");

    stream2
        .with_command_buffer(|cmd_buf| {
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.end_encoding();
            Ok(())
        })
        .expect("Stream 2 should execute");

    // Both streams execute independently and commit their work
    // No explicit synchronization needed for this test
}

/// Test error handling for invalid operations
#[test]
fn test_error_handling() {
    let device = detect_device().expect("Metal device required");
    let allocator = MetalAllocator::new(&device.device);

    // Test invalid allocation size
    let result = allocator.allocate(0);
    assert!(result.is_err(), "Should fail to allocate zero bytes");
}
