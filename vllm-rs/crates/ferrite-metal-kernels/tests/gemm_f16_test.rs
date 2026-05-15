//! Test GEMM with manually created FP16 buffers to isolate MPS data type issue

use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::gemm::MetalGemm;
use ferrite_metal_kernels::stream::MetalStream;
use metal::MTLResourceOptions;
use std::sync::Arc;

#[test]
fn test_gemm_f16_manual_buffers() {
    let device = Arc::new(detect_device().expect("Failed to detect Metal device"));
    let mut stream = MetalStream::new(&device.device);
    let gemm = MetalGemm::new(device.clone()).expect("Failed to create GEMM");

    // Create simple 2x2 FP16 matrices manually
    // A = [[1, 2], [3, 4]]
    // B = [[5, 6], [7, 8]]
    // C = A @ B = [[19, 22], [43, 50]]

    let a_data: Vec<u16> = vec![
        half::f16::from_f32(1.0).to_bits(),
        half::f16::from_f32(2.0).to_bits(),
        half::f16::from_f32(3.0).to_bits(),
        half::f16::from_f32(4.0).to_bits(),
    ];

    let b_data: Vec<u16> = vec![
        half::f16::from_f32(5.0).to_bits(),
        half::f16::from_f32(6.0).to_bits(),
        half::f16::from_f32(7.0).to_bits(),
        half::f16::from_f32(8.0).to_bits(),
    ];

    let c_data: Vec<u16> = vec![half::f16::ZERO.to_bits(); 4];

    // Create buffers
    let a_buf = device.device.new_buffer_with_data(
        a_data.as_ptr() as *const _,
        8,
        MTLResourceOptions::StorageModeShared,
    );

    let b_buf = device.device.new_buffer_with_data(
        b_data.as_ptr() as *const _,
        8,
        MTLResourceOptions::StorageModeShared,
    );

    let c_buf = device.device.new_buffer_with_data(
        c_data.as_ptr() as *const _,
        8,
        MTLResourceOptions::StorageModeShared,
    );

    // Execute GEMM
    gemm.execute(
        &mut stream,
        &a_buf,
        &b_buf,
        &c_buf,
        2,     // m
        2,     // n
        2,     // k
        1.0,   // alpha
        0.0,   // beta
        false, // transpose_a
        false, // transpose_b
        true,  // use_f16
    )
    .expect("GEMM execution failed");

    // Read results
    let ptr = c_buf.contents() as *const u16;
    let result: Vec<u16> = unsafe { std::slice::from_raw_parts(ptr, 4).to_vec() };
    let result_f32: Vec<f32> = result
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect();

    println!("Result: {:?}", result_f32);

    let expected = vec![19.0f32, 22.0, 43.0, 50.0];
    for i in 0..4 {
        assert!(
            (result_f32[i] - expected[i]).abs() < 0.1,
            "Mismatch at index {}: got {}, expected {}",
            i,
            result_f32[i],
            expected[i]
        );
    }
}
