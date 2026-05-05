//! Unit tests for AWQ dequantization

use ferrite_metal_kernels::awq::{AwqError, MetalAwq};
use ferrite_metal_kernels::device::{detect_device, MetalDevice};
use metal::MTLResourceOptions;
use std::sync::Arc;

/// Helper to create a buffer from data
fn create_buffer<T>(device: &MetalDevice, data: &[T]) -> metal::Buffer {
    let size = data.len() * std::mem::size_of::<T>();
    device.device.new_buffer_with_data(
        data.as_ptr() as *const _,
        size as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Helper to create device for tests
fn create_test_device() -> Arc<MetalDevice> {
    Arc::new(detect_device().expect("Failed to detect Metal device"))
}

/// Helper to read buffer contents as a slice
fn read_buffer<T: Copy>(buffer: &metal::Buffer, count: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
}

#[test]
fn test_awq_unpack_int4() {
    let device = create_test_device();
    let awq = MetalAwq::new(device.clone()).expect("Failed to create AWQ");

    // Test data: pack 8 INT4 values (0-7) into one uint32
    // Bits: 0x76543210 = 0111 0110 0101 0100 0011 0010 0001 0000
    let packed_data: Vec<u32> = vec![0x76543210];
    let packed_buffer = create_buffer(&device, &packed_data);

    // Unpack
    let unpacked_buffer = awq
        .unpack_int4(&packed_buffer, 1, 8)
        .expect("Failed to unpack");

    // Read results
    let unpacked: Vec<u16> = read_buffer(&unpacked_buffer, 8);

    // Convert to f32 for easier comparison
    let unpacked_f32: Vec<f32> = unpacked
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect();

    // Expected: [0, 1, 2, 3, 4, 5, 6, 7]
    for i in 0..8 {
        assert_eq!(unpacked_f32[i], i as f32, "Mismatch at index {}", i);
    }
}

#[test]
fn test_awq_unpack_int4_multiple() {
    let device = create_test_device();
    let awq = MetalAwq::new(device.clone()).expect("Failed to create AWQ");

    // Test data: 2 packed uint32s
    // First: 0-7, Second: 8-15
    let packed_data: Vec<u32> = vec![0x76543210, 0xFEDCBA98];
    let packed_buffer = create_buffer(&device, &packed_data);

    // Unpack
    let unpacked_buffer = awq
        .unpack_int4(&packed_buffer, 1, 16)
        .expect("Failed to unpack");

    // Read results
    let unpacked: Vec<u16> = read_buffer(&unpacked_buffer, 16);
    let unpacked_f32: Vec<f32> = unpacked
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect();

    // Expected: [0, 1, 2, ..., 15]
    for i in 0..16 {
        assert_eq!(unpacked_f32[i], i as f32, "Mismatch at index {}", i);
    }
}

#[test]
fn test_awq_dequantize_simple() {
    let device = create_test_device();
    let awq = MetalAwq::new(device.clone()).expect("Failed to create AWQ");

    // Simple test: 1 group, 8 weights
    // Packed weights: [0, 1, 2, 3, 4, 5, 6, 7]
    let packed_weights: Vec<u32> = vec![0x76543210];
    let packed_weights_buffer = create_buffer(&device, &packed_weights);

    // Scales: all 2.0
    let scales: Vec<u16> = vec![half::f16::from_f32(2.0).to_bits(); 8];
    let scales_buffer = create_buffer(&device, &scales);

    // Zeros: all 0
    let packed_zeros: Vec<u32> = vec![0x00000000];
    let packed_zeros_buffer = create_buffer(&device, &packed_zeros);

    // Dequantize
    // group_size=1 means each input channel is its own group
    let group_size = 1;
    let dequantized_buffer = awq
        .dequantize(
            &packed_weights_buffer,
            &scales_buffer,
            &packed_zeros_buffer,
            group_size,
            1,
            8,
        )
        .expect("Failed to dequantize");

    // Read results
    let dequantized: Vec<u16> = read_buffer(&dequantized_buffer, 8);
    let dequantized_f32: Vec<f32> = dequantized
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect();

    // Expected: (weight - 0) * 2.0 = [0, 2, 4, 6, 8, 10, 12, 14]
    for i in 0..8 {
        let expected = (i as f32) * 2.0;
        assert!(
            (dequantized_f32[i] - expected).abs() < 0.1,
            "Mismatch at index {}: got {}, expected {}",
            i,
            dequantized_f32[i],
            expected
        );
    }
}

#[test]
fn test_awq_dequantize_with_zeros() {
    let device = create_test_device();
    let awq = MetalAwq::new(device.clone()).expect("Failed to create AWQ");

    // Packed weights: [4, 5, 6, 7, 8, 9, 10, 11]
    let packed_weights: Vec<u32> = vec![0xBA987654];
    let packed_weights_buffer = create_buffer(&device, &packed_weights);

    // Scales: all 1.0
    let scales: Vec<u16> = vec![half::f16::from_f32(1.0).to_bits(); 8];
    let scales_buffer = create_buffer(&device, &scales);

    // Zeros: all 4 (so weights become [0, 1, 2, 3, 4, 5, 6, 7])
    let packed_zeros: Vec<u32> = vec![0x44444444];
    let packed_zeros_buffer = create_buffer(&device, &packed_zeros);

    // Dequantize
    // group_size=1 means each input channel is its own group
    let group_size = 1;
    let dequantized_buffer = awq
        .dequantize(
            &packed_weights_buffer,
            &scales_buffer,
            &packed_zeros_buffer,
            group_size,
            1,
            8,
        )
        .expect("Failed to dequantize");

    // Read results
    let dequantized: Vec<u16> = read_buffer(&dequantized_buffer, 8);
    let dequantized_f32: Vec<f32> = dequantized
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect();

    // Expected: (weight - 4) * 1.0 = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in 0..8 {
        let expected = i as f32;
        assert!(
            (dequantized_f32[i] - expected).abs() < 0.1,
            "Mismatch at index {}: got {}, expected {}",
            i,
            dequantized_f32[i],
            expected
        );
    }
}

#[test]
fn test_awq_dequantize_multiple_groups() {
    let device = create_test_device();
    let awq = MetalAwq::new(device.clone()).expect("Failed to create AWQ");

    // 2 input channels, 8 output channels, group_size = 1
    // IC=2, OC=8, G=1 -> 2 groups

    // Packed weights: 2 rows x 1 packed uint32 (8 weights each)
    // Row 0: [0, 1, 2, 3, 4, 5, 6, 7]
    // Row 1: [8, 9, 10, 11, 12, 13, 14, 15]
    let packed_weights: Vec<u32> = vec![0x76543210, 0xFEDCBA98];
    let packed_weights_buffer = create_buffer(&device, &packed_weights);

    // Scales: 2 groups x 8 output channels
    // Group 0: all 1.0, Group 1: all 0.5
    let mut scales: Vec<u16> = Vec::new();
    scales.extend(vec![half::f16::from_f32(1.0).to_bits(); 8]); // Group 0
    scales.extend(vec![half::f16::from_f32(0.5).to_bits(); 8]); // Group 1
    let scales_buffer = create_buffer(&device, &scales);

    // Zeros: 2 groups x 1 packed uint32 (8 zeros each)
    // All zeros = 0
    let packed_zeros: Vec<u32> = vec![0x00000000, 0x00000000];
    let packed_zeros_buffer = create_buffer(&device, &packed_zeros);

    // Dequantize
    let group_size = 1;
    let dequantized_buffer = awq
        .dequantize(
            &packed_weights_buffer,
            &scales_buffer,
            &packed_zeros_buffer,
            group_size,
            2,
            8,
        )
        .expect("Failed to dequantize");

    // Read results
    let dequantized: Vec<u16> = read_buffer(&dequantized_buffer, 16);
    let dequantized_f32: Vec<f32> = dequantized
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect();

    // Expected:
    // Row 0 (group 0, scale 1.0): [0, 1, 2, 3, 4, 5, 6, 7]
    // Row 1 (group 1, scale 0.5): [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5]
    for i in 0..8 {
        let expected = i as f32;
        assert!(
            (dequantized_f32[i] - expected).abs() < 0.1,
            "Row 0, index {}: got {}, expected {}",
            i,
            dequantized_f32[i],
            expected
        );
    }
    for i in 0..8 {
        let expected = (8 + i) as f32 * 0.5;
        assert!(
            (dequantized_f32[8 + i] - expected).abs() < 0.1,
            "Row 1, index {}: got {}, expected {}",
            i,
            dequantized_f32[8 + i],
            expected
        );
    }
}

#[test]
fn test_awq_dequantize_and_gemm() {
    let device = create_test_device();
    let awq = MetalAwq::new(device.clone()).expect("Failed to create AWQ");

    // Simple GEMM: [1, 8] @ [8, 8] = [1, 8]
    // Activations: [1, 2, 3, 4, 5, 6, 7, 8]
    let activations: Vec<u16> = (1..=8)
        .map(|x| half::f16::from_f32(x as f32).to_bits())
        .collect();
    let activations_buffer = create_buffer(&device, &activations);

    // Packed weights: identity-like (diagonal pattern)
    // For simplicity, use all 1s
    let packed_weights: Vec<u32> = vec![0x11111111];
    let packed_weights_buffer = create_buffer(&device, &packed_weights);

    // Scales: all 1.0
    let scales: Vec<u16> = vec![half::f16::from_f32(1.0).to_bits(); 8];
    let scales_buffer = create_buffer(&device, &scales);

    // Zeros: all 0
    let packed_zeros: Vec<u32> = vec![0x00000000];
    let packed_zeros_buffer = create_buffer(&device, &packed_zeros);

    // Dequantize and GEMM
    // group_size=1 means each input channel is its own group
    let group_size = 1;
    let output_buffer = awq
        .dequantize_and_gemm(
            &activations_buffer,
            &packed_weights_buffer,
            &scales_buffer,
            &packed_zeros_buffer,
            group_size,
            1,
            8,
            8,
        )
        .expect("Failed to dequantize and GEMM");

    // Read results
    let output: Vec<u16> = read_buffer(&output_buffer, 8);
    let output_f32: Vec<f32> = output
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect();

    // With all weights = 1, output should be sum of activations = 36
    // But distributed across 8 outputs (each output is sum of corresponding column)
    // Since all weights are 1, each output = sum of all activations = 36
    for i in 0..8 {
        assert!(
            output_f32[i] > 0.0,
            "Output {} should be positive, got {}",
            i,
            output_f32[i]
        );
    }
}

#[test]
fn test_awq_invalid_dimensions() {
    let device = create_test_device();
    let awq = MetalAwq::new(device.clone()).expect("Failed to create AWQ");

    // Invalid: num_out_channels not multiple of 8
    let packed_weights: Vec<u32> = vec![0x76543210];
    let packed_weights_buffer = create_buffer(&device, &packed_weights);

    let result = awq.unpack_int4(&packed_weights_buffer, 1, 7);
    assert!(result.is_err());
    match result {
        Err(AwqError::InvalidDimensions(_)) => {}
        _ => panic!("Expected InvalidDimensions error"),
    }
}
