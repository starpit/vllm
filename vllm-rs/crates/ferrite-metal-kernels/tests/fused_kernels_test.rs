// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for fused kernel implementations

use ferrite_metal_kernels::{detect_device, fused_kernels::*, MetalStream};
use half::f16;
use metal::MTLResourceOptions;

const EPSILON: f32 = 1e-5;
const ATOL_F16: f32 = 1e-2; // Relaxed tolerance for FP16 with transcendental functions (1%)

/// Helper to create a buffer and fill with data
fn create_buffer_with_data<T: Copy>(device: &metal::Device, data: &[T]) -> metal::Buffer {
    let size = data.len() * std::mem::size_of::<T>();
    let buffer = device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);

    unsafe {
        let ptr = buffer.contents() as *mut T;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    }

    buffer
}

/// Helper to read buffer data
fn read_buffer_data<T: Copy>(buffer: &metal::Buffer, count: usize) -> Vec<T> {
    let mut result = vec![unsafe { std::mem::zeroed() }; count];
    unsafe {
        let ptr = buffer.contents() as *const T;
        std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), count);
    }
    result
}

/// Reference implementation: RMSNorm
fn rmsnorm_reference(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = weight.len();
    let m = input.len() / n;
    let mut output = vec![0.0f32; input.len()];

    for i in 0..m {
        let row = &input[i * n..(i + 1) * n];

        // Compute mean of squares
        let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / n as f32;
        let rms = (mean_sq + eps).sqrt();

        // Normalize and scale
        for j in 0..n {
            output[i * n + j] = (row[j] / rms) * weight[j];
        }
    }

    output
}

/// Reference implementation: Add + RMSNorm
fn add_rmsnorm_reference(
    input: &[f32],
    residual: &[f32],
    weight: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = weight.len();
    let m = input.len() / n;
    let mut output = vec![0.0f32; input.len()];
    let mut residual_out = vec![0.0f32; input.len()];

    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            residual_out[idx] = input[idx] + residual[idx];
        }

        let row = &residual_out[i * n..(i + 1) * n];
        let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / n as f32;
        let rms = (mean_sq + eps).sqrt();

        for j in 0..n {
            output[i * n + j] = (row[j] / rms) * weight[j];
        }
    }

    (output, residual_out)
}

/// Reference implementation: SiLU
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Reference implementation: Gate-Up-SiLU-Mul
fn gate_up_silu_mul_reference(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| silu(*g) * u)
        .collect()
}

#[test]
fn test_fused_add_rmsnorm_f16() {
    let metal_device = detect_device().expect("Metal device required");
    let mut stream = MetalStream::new(&metal_device.device);
    let fused_norm = FusedAddRmsNorm::new(std::sync::Arc::new(metal_device.clone()))
        .expect("Failed to create kernel");
    let device = &metal_device.device;

    // Test parameters
    let m = 4u32;
    let n = 128u32;
    let eps = 1e-5f32;

    // Generate test data
    let input: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01).collect();
    let residual: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.005).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();

    // Convert to f16
    let input_f16: Vec<f16> = input.iter().map(|&x| f16::from_f32(x)).collect();
    let residual_f16: Vec<f16> = residual.iter().map(|&x| f16::from_f32(x)).collect();
    let weight_f16: Vec<f16> = weight.iter().map(|&x| f16::from_f32(x)).collect();

    // Create buffers
    let input_buf = create_buffer_with_data(device, &input_f16);
    let residual_buf = create_buffer_with_data(device, &residual_f16);
    let weight_buf = create_buffer_with_data(device, &weight_f16);
    let output_buf = device.new_buffer(
        (m * n * std::mem::size_of::<f16>() as u32) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let residual_out_buf = device.new_buffer(
        (m * n * std::mem::size_of::<f16>() as u32) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Execute kernel
    println!("Executing kernel with M={}, N={}, eps={}", m, n, eps);
    println!(
        "Input buffer size: {}, Output buffer size: {}",
        input_buf.length(),
        output_buf.length()
    );

    fused_norm
        .execute(
            &mut stream,
            &input_buf,
            &residual_buf,
            &weight_buf,
            &output_buf,
            Some(&residual_out_buf),
            m,
            n,
            eps,
            true, // use_f16
        )
        .expect("Kernel execution failed");

    println!("Kernel executed, synchronizing...");
    stream.synchronize().expect("Synchronization failed");
    println!("Synchronization complete");

    // Read results
    let output_f16: Vec<f16> = read_buffer_data(&output_buf, (m * n) as usize);
    let residual_out_f16: Vec<f16> = read_buffer_data(&residual_out_buf, (m * n) as usize);

    let output: Vec<f32> = output_f16.iter().map(|&x| x.to_f32()).collect();

    // Debug: print first few values
    println!(
        "First 10 output values: {:?}",
        &output[0..10.min(output.len())]
    );
    println!("First 10 expected values from reference");
    let residual_out: Vec<f32> = residual_out_f16.iter().map(|&x| x.to_f32()).collect();

    // Compute reference
    let (expected_output, expected_residual) =
        add_rmsnorm_reference(&input, &residual, &weight, eps);

    // Verify results
    for i in 0..output.len() {
        let diff = (output[i] - expected_output[i]).abs();
        assert!(
            diff < ATOL_F16,
            "Output mismatch at index {}: got {}, expected {}, diff {}",
            i,
            output[i],
            expected_output[i],
            diff
        );
    }

    for i in 0..residual_out.len() {
        let diff = (residual_out[i] - expected_residual[i]).abs();
        assert!(
            diff < ATOL_F16,
            "Residual output mismatch at index {}: got {}, expected {}, diff {}",
            i,
            residual_out[i],
            expected_residual[i],
            diff
        );
    }

    println!("✓ Fused Add+RMSNorm F16: All values within tolerance");
}

#[test]
fn test_fused_add_rmsnorm_vectorized() {
    let metal_device = detect_device().expect("Metal device required");
    let mut stream = MetalStream::new(&metal_device.device);
    let fused_norm = FusedAddRmsNorm::new(std::sync::Arc::new(metal_device.clone()))
        .expect("Failed to create kernel");
    let device = &metal_device.device;

    // Test with N divisible by 4 for vectorization
    let m = 2u32;
    let n = 256u32; // Divisible by 4
    let eps = 1e-5f32;

    let input: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01).collect();
    let residual: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.005).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();

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

    let (expected_output, _) = add_rmsnorm_reference(&input, &residual, &weight, eps);

    for i in 0..output.len() {
        let diff = (output[i] - expected_output[i]).abs();
        assert!(
            diff < ATOL_F16,
            "Vectorized output mismatch at index {}: got {}, expected {}, diff {}",
            i,
            output[i],
            expected_output[i],
            diff
        );
    }

    println!("✓ Fused Add+RMSNorm Vectorized: All values within tolerance");
}

#[test]
fn test_fused_gate_up_silu_mul_separate() {
    let metal_device = detect_device().expect("Metal device required");
    let mut stream = MetalStream::new(&metal_device.device);
    let fused_silu = FusedGateUpSiluMul::new(std::sync::Arc::new(metal_device.clone()))
        .expect("Failed to create kernel");
    let device = &metal_device.device;

    let m = 4u32;
    let n = 128u32;

    // Generate test data
    let gate: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01 - 2.0).collect();
    let up: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.005 + 1.0).collect();

    let gate_f16: Vec<f16> = gate.iter().map(|&x| f16::from_f32(x)).collect();
    let up_f16: Vec<f16> = up.iter().map(|&x| f16::from_f32(x)).collect();

    let gate_buf = create_buffer_with_data(device, &gate_f16);
    let up_buf = create_buffer_with_data(device, &up_f16);
    let output_buf = device.new_buffer(
        (m * n * std::mem::size_of::<f16>() as u32) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    fused_silu
        .execute_separate(&mut stream, &gate_buf, &up_buf, &output_buf, m, n, true)
        .expect("Kernel execution failed");

    stream.synchronize().expect("Synchronization failed");

    let output_f16: Vec<f16> = read_buffer_data(&output_buf, (m * n) as usize);
    let output: Vec<f32> = output_f16.iter().map(|&x| x.to_f32()).collect();

    let expected = gate_up_silu_mul_reference(&gate, &up);

    for i in 0..output.len() {
        let diff = (output[i] - expected[i]).abs();
        assert!(
            diff < ATOL_F16,
            "SiLU output mismatch at index {}: got {}, expected {}, diff {}",
            i,
            output[i],
            expected[i],
            diff
        );
    }

    println!("✓ Fused Gate-Up-SiLU-Mul (separate): All values within tolerance");
}

#[test]
fn test_fused_gate_up_silu_mul_concat() {
    let metal_device = detect_device().expect("Metal device required");
    let mut stream = MetalStream::new(&metal_device.device);
    let fused_silu = FusedGateUpSiluMul::new(std::sync::Arc::new(metal_device.clone()))
        .expect("Failed to create kernel");
    let device = &metal_device.device;

    let m = 4u32;
    let n = 128u32;

    // Generate concatenated gate_up data [M, 2*N]
    let gate: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01 - 2.0).collect();
    let up: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.005 + 1.0).collect();

    // Concatenate: [gate | up] for each row
    let mut gate_up = Vec::with_capacity((m * n * 2) as usize);
    for i in 0..m as usize {
        gate_up.extend_from_slice(&gate[i * n as usize..(i + 1) * n as usize]);
        gate_up.extend_from_slice(&up[i * n as usize..(i + 1) * n as usize]);
    }

    let gate_up_f16: Vec<f16> = gate_up.iter().map(|&x| f16::from_f32(x)).collect();

    let gate_up_buf = create_buffer_with_data(device, &gate_up_f16);
    let output_buf = device.new_buffer(
        (m * n * std::mem::size_of::<f16>() as u32) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    fused_silu
        .execute_concat(&mut stream, &gate_up_buf, &output_buf, m, n, true)
        .expect("Kernel execution failed");

    stream.synchronize().expect("Synchronization failed");

    let output_f16: Vec<f16> = read_buffer_data(&output_buf, (m * n) as usize);
    let output: Vec<f32> = output_f16.iter().map(|&x| x.to_f32()).collect();

    let expected = gate_up_silu_mul_reference(&gate, &up);

    for i in 0..output.len() {
        let diff = (output[i] - expected[i]).abs();
        assert!(
            diff < ATOL_F16,
            "SiLU concat output mismatch at index {}: got {}, expected {}, diff {}",
            i,
            output[i],
            expected[i],
            diff
        );
    }

    println!("✓ Fused Gate-Up-SiLU-Mul (concat): All values within tolerance");
}

#[test]
fn test_fused_gate_up_gelu_mul() {
    let metal_device = detect_device().expect("Metal device required");
    let mut stream = MetalStream::new(&metal_device.device);
    let fused_silu = FusedGateUpSiluMul::new(std::sync::Arc::new(metal_device.clone()))
        .expect("Failed to create kernel");
    let device = &metal_device.device;

    let m = 2u32;
    let n = 64u32;

    let gate: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.02 - 1.0).collect();
    let up: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01 + 0.5).collect();

    let gate_f16: Vec<f16> = gate.iter().map(|&x| f16::from_f32(x)).collect();
    let up_f16: Vec<f16> = up.iter().map(|&x| f16::from_f32(x)).collect();

    let gate_buf = create_buffer_with_data(device, &gate_f16);
    let up_buf = create_buffer_with_data(device, &up_f16);
    let output_buf = device.new_buffer(
        (m * n * std::mem::size_of::<f16>() as u32) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Test approximate GELU
    fused_silu
        .execute_gelu(
            &mut stream,
            &gate_buf,
            &up_buf,
            &output_buf,
            m,
            n,
            false, // approximate
        )
        .expect("Kernel execution failed");

    stream.synchronize().expect("Synchronization failed");

    let output_f16: Vec<f16> = read_buffer_data(&output_buf, (m * n) as usize);
    let output: Vec<f32> = output_f16.iter().map(|&x| x.to_f32()).collect();

    // Verify output is reasonable (GELU should be in [-1, 1] range roughly)
    for &val in &output {
        assert!(
            val.is_finite(),
            "GELU output contains non-finite value: {}",
            val
        );
    }

    println!("✓ Fused Gate-Up-GELU-Mul: Output is finite and reasonable");
}

#[test]
fn test_fused_kernels_numerical_stability() {
    let metal_device = detect_device().expect("Metal device required");
    let mut stream = MetalStream::new(&metal_device.device);
    let fused_norm = FusedAddRmsNorm::new(std::sync::Arc::new(metal_device.clone()))
        .expect("Failed to create kernel");
    let device = &metal_device.device;

    // Test with large values to check numerical stability
    let m = 2u32;
    let n = 64u32;
    let eps = 1e-5f32;

    let input: Vec<f32> = vec![100.0; (m * n) as usize];
    let residual: Vec<f32> = vec![50.0; (m * n) as usize];
    let weight: Vec<f32> = vec![1.0; n as usize];

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

    // Verify no NaN or Inf
    for (i, &val) in output.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Output contains non-finite value at index {}: {}",
            i,
            val
        );
    }

    println!("✓ Fused kernels numerical stability: No NaN/Inf with large inputs");
}
