// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Debug test for fused kernel parameter passing

use ferrite_metal_kernels::{detect_device, fused_kernels::*, MetalStream};
use half::f16;
use metal::MTLResourceOptions;

fn create_buffer_with_data<T: Copy>(device: &metal::Device, data: &[T]) -> metal::Buffer {
    let size = data.len() * std::mem::size_of::<T>();
    let buffer = device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);

    unsafe {
        let ptr = buffer.contents() as *mut T;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    }

    buffer
}

fn read_buffer_data<T: Copy>(buffer: &metal::Buffer, count: usize) -> Vec<T> {
    let mut result = vec![unsafe { std::mem::zeroed() }; count];
    unsafe {
        let ptr = buffer.contents() as *const T;
        std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), count);
    }
    result
}

#[test]
fn test_simple_fused_add_rmsnorm() {
    let metal_device = detect_device().expect("Metal device required");
    let mut stream = MetalStream::new(&metal_device.device);
    let fused_norm = FusedAddRmsNorm::new(std::sync::Arc::new(metal_device.clone()))
        .expect("Failed to create kernel");
    let device = &metal_device.device;

    // Very simple test: M=1, N=4
    let m = 1u32;
    let n = 4u32;
    let eps = 1e-5f32;

    // Simple values: input = [1, 2, 3, 4], residual = [0, 0, 0, 0], weight = [1, 1, 1, 1]
    // Expected: sum = [1, 2, 3, 4]
    // RMS = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5 + eps) ≈ 2.739
    // Output = [1/2.739, 2/2.739, 3/2.739, 4/2.739] ≈ [0.365, 0.730, 1.095, 1.460]
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let residual: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];
    let weight: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

    let input_f16: Vec<f16> = input.iter().map(|&x| f16::from_f32(x)).collect();
    let residual_f16: Vec<f16> = residual.iter().map(|&x| f16::from_f32(x)).collect();
    let weight_f16: Vec<f16> = weight.iter().map(|&x| f16::from_f32(x)).collect();

    let input_buf = create_buffer_with_data(device, &input_f16);
    let residual_buf = create_buffer_with_data(device, &residual_f16);
    let weight_buf = create_buffer_with_data(device, &weight_f16);
    let output_buf = device.new_buffer(
        (m * n * std::mem::size_of::<f16>() as u32) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    println!("Executing kernel with M={}, N={}, eps={}", m, n, eps);
    println!("Input: {:?}", input);
    println!("Residual: {:?}", residual);
    println!("Weight: {:?}", weight);

    fused_norm
        .execute(
            &mut stream,
            &input_buf,
            &residual_buf,
            &weight_buf,
            &output_buf,
            None,
            m,
            n,
            eps,
            true,
        )
        .expect("Kernel execution failed");

    stream.synchronize().expect("Synchronization failed");

    let output_f16: Vec<f16> = read_buffer_data(&output_buf, (m * n) as usize);
    let output: Vec<f32> = output_f16.iter().map(|&x| x.to_f32()).collect();

    println!("Output: {:?}", output);

    // Expected RMS = sqrt(30/4) = sqrt(7.5) ≈ 2.7386
    let expected_rms = ((1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0) / 4.0 + eps).sqrt();
    println!("Expected RMS: {}", expected_rms);

    let expected: Vec<f32> = input.iter().map(|&x| x / expected_rms).collect();
    println!("Expected output: {:?}", expected);

    // Check if output is reasonable
    for (i, &val) in output.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Output at index {} is not finite: {}",
            i,
            val
        );

        let diff = (val - expected[i]).abs();
        println!(
            "Index {}: got {}, expected {}, diff {}",
            i, val, expected[i], diff
        );
    }
}
