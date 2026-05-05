use ferrite_metal_kernels::{MetalDevice, MetalStream};
use ferrite_metal_targets::MetalTargetProfile;
use metal::MTLResourceOptions;
use std::sync::Arc;

#[test]
fn test_basic_attention_single_head() {
    // Test configuration
    const HEAD_SIZE: usize = 64;
    const SEQ_LEN: usize = 128;
    let scale: f32 = 1.0 / (HEAD_SIZE as f32).sqrt();

    // Initialize Metal device and stream
    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs(); // Use M1 Max profile for testing
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let stream = MetalStream::new(&metal_device);

    // Create test data: Q, K, V (using f16 for Metal half precision)
    let mut q_data = vec![0.0f32; HEAD_SIZE];
    let mut k_data = vec![0.0f32; SEQ_LEN * HEAD_SIZE];
    let mut v_data = vec![0.0f32; SEQ_LEN * HEAD_SIZE];

    // Initialize with simple pattern for testing
    // Q: [1, 0, 0, 0, ...]
    q_data[0] = 1.0;

    // K: Each row is [1, 0, 0, ...] with slight variation
    // Last token (127) has highest dot product with Q
    for i in 0..SEQ_LEN {
        k_data[i * HEAD_SIZE] = 1.0 + (i as f32) * 0.01;
    }

    // V: First dimension = token index, rest = 0
    // This makes it easy to verify which tokens got attention
    for i in 0..SEQ_LEN {
        v_data[i * HEAD_SIZE] = i as f32;
        // Rest are 0
    }

    // Convert f32 to f16 (Metal half)
    let q_f16: Vec<u16> = q_data
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let k_f16: Vec<u16> = k_data
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let v_f16: Vec<u16> = v_data
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();

    // Create Metal buffers
    let q_buffer = device.device.new_buffer_with_data(
        q_f16.as_ptr() as *const _,
        (q_f16.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let k_buffer = device.device.new_buffer_with_data(
        k_f16.as_ptr() as *const _,
        (k_f16.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let v_buffer = device.device.new_buffer_with_data(
        v_f16.as_ptr() as *const _,
        (v_f16.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.device.new_buffer(
        (HEAD_SIZE * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create params buffer
    #[repr(C)]
    struct AttentionParams {
        seq_len: u32,
        head_size: u32,
        scale: f32,
        block_size: u32,
    }

    let params = AttentionParams {
        seq_len: SEQ_LEN as u32,
        head_size: HEAD_SIZE as u32,
        scale: scale,
        block_size: 16,
    };

    let params_buffer = device.device.new_buffer_with_data(
        &params as *const _ as *const _,
        std::mem::size_of::<AttentionParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Load shader library
    let library_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .join("attention.metal");

    let library_source =
        std::fs::read_to_string(&library_path).expect("Failed to read attention.metal");

    let library = device
        .device
        .new_library_with_source(&library_source, &metal::CompileOptions::new())
        .expect("Failed to compile shader library");

    let kernel_function = library
        .get_function("attention_single_head", None)
        .expect("Failed to get kernel function");

    let pipeline_state = device
        .device
        .new_compute_pipeline_state_with_function(&kernel_function)
        .expect("Failed to create pipeline state");

    // Execute kernel
    let command_buffer = stream.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&q_buffer), 0);
    encoder.set_buffer(1, Some(&k_buffer), 0);
    encoder.set_buffer(2, Some(&v_buffer), 0);
    encoder.set_buffer(3, Some(&output_buffer), 0);
    encoder.set_buffer(4, Some(&params_buffer), 0);

    // Allocate threadgroup memory for shared_logits
    let threadgroup_memory_length = (SEQ_LEN * 4) as u64; // float[seq_len]
    encoder.set_threadgroup_memory_length(0, threadgroup_memory_length);

    // Dispatch threads
    let threadgroup_size = metal::MTLSize::new(256, 1, 1);
    let grid_size = metal::MTLSize::new(256, 1, 1);

    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back results
    let output_ptr = output_buffer.contents() as *const u16;
    let output_f16: Vec<u16> =
        unsafe { std::slice::from_raw_parts(output_ptr, HEAD_SIZE).to_vec() };

    let output_f32: Vec<f32> = output_f16
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect();

    // Verify results
    // Expected: Attention should be highest for first token (K[0] has highest dot product with Q)
    // Output should be weighted average of V rows, dominated by V[0]

    println!("Output (first 10 elements): {:?}", &output_f32[..10]);

    // Basic sanity checks
    assert!(
        output_f32.iter().all(|&x: &f32| x.is_finite()),
        "Output contains non-finite values"
    );

    // The output should be a weighted average of V rows
    // Since Q·K[127] is highest (1.0 + 127*0.01 = 2.27), attention focuses on last token
    // V[127][0] = 127, so output[0] should be close to 127 (but softmax spreads it)
    // Expected: output[0] ≈ 60-80 (weighted average skewed toward token 127)
    let first_value = output_f32[0];
    assert!(
        first_value > 50.0 && first_value < 130.0,
        "Output[0] = {} is outside expected range [50, 130]",
        first_value
    );

    // Rest of output should be close to 0 (only first dim of V is non-zero)
    let max_rest = output_f32[1..]
        .iter()
        .map(|&x: &f32| x.abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_rest < 1.0,
        "Output[1..] should be near zero, but max = {}",
        max_rest
    );

    println!(
        "✓ Basic attention test passed - output[0] = {:.2}",
        first_value
    );
}

#[test]
fn test_attention_numerical_stability() {
    // Test with extreme values to verify numerical stability
    const HEAD_SIZE: usize = 64;
    const SEQ_LEN: usize = 32;
    let scale: f32 = 1.0 / (HEAD_SIZE as f32).sqrt();

    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs(); // Use M1 Max profile for testing
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let stream = MetalStream::new(&metal_device);

    // Create test data with large values
    let mut q_data = vec![0.0f32; HEAD_SIZE];
    let mut k_data = vec![0.0f32; SEQ_LEN * HEAD_SIZE];
    let mut v_data = vec![0.0f32; SEQ_LEN * HEAD_SIZE];

    // Q: Large values
    for i in 0..HEAD_SIZE {
        q_data[i] = 10.0;
    }

    // K: Large values with variation
    for i in 0..SEQ_LEN {
        for j in 0..HEAD_SIZE {
            k_data[i * HEAD_SIZE + j] = 10.0 + (i as f32) * 0.1;
        }
    }

    // V: Simple pattern
    for i in 0..SEQ_LEN {
        v_data[i * HEAD_SIZE] = 1.0;
    }

    // Convert and create buffers (similar to previous test)
    let q_f16: Vec<u16> = q_data
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let k_f16: Vec<u16> = k_data
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let v_f16: Vec<u16> = v_data
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();

    let q_buffer = device.device.new_buffer_with_data(
        q_f16.as_ptr() as *const _,
        (q_f16.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let k_buffer = device.device.new_buffer_with_data(
        k_f16.as_ptr() as *const _,
        (k_f16.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let v_buffer = device.device.new_buffer_with_data(
        v_f16.as_ptr() as *const _,
        (v_f16.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.device.new_buffer(
        (HEAD_SIZE * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    #[repr(C)]
    struct AttentionParams {
        seq_len: u32,
        head_size: u32,
        scale: f32,
        block_size: u32,
    }

    let params = AttentionParams {
        seq_len: SEQ_LEN as u32,
        head_size: HEAD_SIZE as u32,
        scale: scale,
        block_size: 16,
    };

    let params_buffer = device.device.new_buffer_with_data(
        &params as *const _ as *const _,
        std::mem::size_of::<AttentionParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Load and execute shader (similar to previous test)
    let library_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .join("attention.metal");

    let library_source =
        std::fs::read_to_string(&library_path).expect("Failed to read attention.metal");

    let library = device
        .device
        .new_library_with_source(&library_source, &metal::CompileOptions::new())
        .expect("Failed to compile shader library");

    let kernel_function = library
        .get_function("attention_single_head", None)
        .expect("Failed to get kernel function");

    let pipeline_state = device
        .device
        .new_compute_pipeline_state_with_function(&kernel_function)
        .expect("Failed to create pipeline state");

    let command_buffer = stream.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&q_buffer), 0);
    encoder.set_buffer(1, Some(&k_buffer), 0);
    encoder.set_buffer(2, Some(&v_buffer), 0);
    encoder.set_buffer(3, Some(&output_buffer), 0);
    encoder.set_buffer(4, Some(&params_buffer), 0);

    let threadgroup_memory_length = (SEQ_LEN * 4) as u64;
    encoder.set_threadgroup_memory_length(0, threadgroup_memory_length);

    let threadgroup_size = metal::MTLSize::new(256, 1, 1);
    let grid_size = metal::MTLSize::new(256, 1, 1);

    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back results
    let output_ptr = output_buffer.contents() as *const u16;
    let output_f16: Vec<u16> =
        unsafe { std::slice::from_raw_parts(output_ptr, HEAD_SIZE).to_vec() };

    let output_f32: Vec<f32> = output_f16
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect();

    // Verify numerical stability: no NaN or Inf
    assert!(
        output_f32.iter().all(|&x: &f32| x.is_finite()),
        "Output contains NaN or Inf despite large input values"
    );

    println!("✓ Numerical stability test passed");
}

#[test]
#[ignore] // Ignore until we have reference CUDA implementation
fn test_attention_vs_cuda_reference() {
    // TODO: Compare Metal attention output against CUDA reference
    // This will verify numerical accuracy (element-wise error < 1e-5)
}
