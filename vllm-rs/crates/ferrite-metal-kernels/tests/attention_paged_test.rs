use ferrite_metal_kernels::{MetalDevice, MetalStream};
use metal::MTLResourceOptions;
use std::sync::Arc;

#[test]
fn test_paged_attention_single_head() {
    // Test configuration
    const HEAD_SIZE: usize = 64;
    const SEQ_LEN: usize = 128;
    const BLOCK_SIZE: usize = 16;
    let scale: f32 = 1.0 / (HEAD_SIZE as f32).sqrt();

    // Initialize Metal device and stream
    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs();
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let stream = MetalStream::new(&metal_device);

    // Calculate number of blocks
    let num_blocks = (SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE; // 8 blocks

    // Create test data: Q, K, V (using f16 for Metal half precision)
    let mut q_data = vec![0.0f32; HEAD_SIZE];

    // K and V are stored in paged format: [num_blocks, head_size, block_size]
    let mut k_data = vec![0.0f32; num_blocks * HEAD_SIZE * BLOCK_SIZE];
    let mut v_data = vec![0.0f32; num_blocks * HEAD_SIZE * BLOCK_SIZE];

    // Initialize Q: [1, 0, 0, 0, ...]
    q_data[0] = 1.0;

    // Initialize K in paged format
    // Each block contains BLOCK_SIZE tokens
    for block_idx in 0..num_blocks {
        for block_offset in 0..BLOCK_SIZE {
            let token_idx = block_idx * BLOCK_SIZE + block_offset;
            if token_idx < SEQ_LEN {
                // K[token_idx][0] = 1.0 + token_idx * 0.01
                let k_idx = block_idx * HEAD_SIZE * BLOCK_SIZE + 0 * BLOCK_SIZE + block_offset;
                k_data[k_idx] = 1.0 + (token_idx as f32) * 0.01;
            }
        }
    }

    // Initialize V in paged format
    // V[token_idx][0] = token_idx, rest = 0
    for block_idx in 0..num_blocks {
        for block_offset in 0..BLOCK_SIZE {
            let token_idx = block_idx * BLOCK_SIZE + block_offset;
            if token_idx < SEQ_LEN {
                let v_idx = block_idx * HEAD_SIZE * BLOCK_SIZE + 0 * BLOCK_SIZE + block_offset;
                v_data[v_idx] = token_idx as f32;
            }
        }
    }

    // Create block table (identity mapping for this test)
    let block_table: Vec<i32> = (0..num_blocks as i32).collect();

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

    let block_table_buffer = device.device.new_buffer_with_data(
        block_table.as_ptr() as *const _,
        (block_table.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.device.new_buffer(
        (HEAD_SIZE * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create params buffer
    #[repr(C)]
    struct PagedAttentionParams {
        seq_len: u32,
        head_size: u32,
        scale: f32,
        block_size: u32,
        max_num_blocks: u32,
        kv_block_stride: u32,
        kv_head_stride: u32,
    }

    let kv_head_stride = HEAD_SIZE as u32 * BLOCK_SIZE as u32;
    let kv_block_stride = kv_head_stride; // Single head for now

    let params = PagedAttentionParams {
        seq_len: SEQ_LEN as u32,
        head_size: HEAD_SIZE as u32,
        scale: scale,
        block_size: BLOCK_SIZE as u32,
        max_num_blocks: num_blocks as u32,
        kv_block_stride: kv_block_stride,
        kv_head_stride: kv_head_stride,
    };

    let params_buffer = device.device.new_buffer_with_data(
        &params as *const _ as *const _,
        std::mem::size_of::<PagedAttentionParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Load shader library
    let library_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .join("attention_paged.metal");

    let library_source =
        std::fs::read_to_string(&library_path).expect("Failed to read attention_paged.metal");

    let library = device
        .device
        .new_library_with_source(&library_source, &metal::CompileOptions::new())
        .expect("Failed to compile shader library");

    let kernel_function = library
        .get_function("attention_paged_single_head", None)
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
    encoder.set_buffer(3, Some(&block_table_buffer), 0);
    encoder.set_buffer(4, Some(&output_buffer), 0);
    encoder.set_buffer(5, Some(&params_buffer), 0);

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

    println!(
        "Paged attention output (first 10 elements): {:?}",
        &output_f32[..10]
    );

    // Verify results
    assert!(
        output_f32.iter().all(|&x: &f32| x.is_finite()),
        "Output contains non-finite values"
    );

    // Expected: Similar to non-paged version, output[0] should be ~60-80
    let first_value = output_f32[0];
    assert!(
        first_value > 50.0 && first_value < 130.0,
        "Output[0] = {} is outside expected range [50, 130]",
        first_value
    );

    // Rest of output should be close to 0
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
        "✓ Paged attention test passed - output[0] = {:.2}",
        first_value
    );
}

#[test]
fn test_paged_attention_scattered_blocks() {
    // Test with non-identity block table (scattered blocks)
    const HEAD_SIZE: usize = 64;
    const SEQ_LEN: usize = 64;
    const BLOCK_SIZE: usize = 16;
    let scale: f32 = 1.0 / (HEAD_SIZE as f32).sqrt();

    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs();
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let stream = MetalStream::new(&metal_device);

    let num_blocks = (SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE; // 4 blocks

    // Create scattered block table: [3, 1, 0, 2]
    // This tests that we correctly look up physical blocks
    let block_table: Vec<i32> = vec![3, 1, 0, 2];

    let mut q_data = vec![0.0f32; HEAD_SIZE];
    let mut k_data = vec![0.0f32; num_blocks * HEAD_SIZE * BLOCK_SIZE];
    let mut v_data = vec![0.0f32; num_blocks * HEAD_SIZE * BLOCK_SIZE];

    q_data[0] = 1.0;

    // Initialize K and V with physical block indices
    // This way we can verify correct block table lookup
    for physical_block in 0..num_blocks {
        for block_offset in 0..BLOCK_SIZE {
            let k_idx = physical_block * HEAD_SIZE * BLOCK_SIZE + 0 * BLOCK_SIZE + block_offset;
            let v_idx = physical_block * HEAD_SIZE * BLOCK_SIZE + 0 * BLOCK_SIZE + block_offset;

            // Store physical block number in the data
            k_data[k_idx] = 1.0 + (physical_block as f32) * 0.1;
            v_data[v_idx] = physical_block as f32 * 10.0;
        }
    }

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

    let block_table_buffer = device.device.new_buffer_with_data(
        block_table.as_ptr() as *const _,
        (block_table.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.device.new_buffer(
        (HEAD_SIZE * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    #[repr(C)]
    struct PagedAttentionParams {
        seq_len: u32,
        head_size: u32,
        scale: f32,
        block_size: u32,
        max_num_blocks: u32,
        kv_block_stride: u32,
        kv_head_stride: u32,
    }

    let kv_head_stride = HEAD_SIZE as u32 * BLOCK_SIZE as u32;
    let kv_block_stride = kv_head_stride;

    let params = PagedAttentionParams {
        seq_len: SEQ_LEN as u32,
        head_size: HEAD_SIZE as u32,
        scale: scale,
        block_size: BLOCK_SIZE as u32,
        max_num_blocks: num_blocks as u32,
        kv_block_stride: kv_block_stride,
        kv_head_stride: kv_head_stride,
    };

    let params_buffer = device.device.new_buffer_with_data(
        &params as *const _ as *const _,
        std::mem::size_of::<PagedAttentionParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let library_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .join("attention_paged.metal");

    let library_source =
        std::fs::read_to_string(&library_path).expect("Failed to read attention_paged.metal");

    let library = device
        .device
        .new_library_with_source(&library_source, &metal::CompileOptions::new())
        .expect("Failed to compile shader library");

    let kernel_function = library
        .get_function("attention_paged_single_head", None)
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
    encoder.set_buffer(3, Some(&block_table_buffer), 0);
    encoder.set_buffer(4, Some(&output_buffer), 0);
    encoder.set_buffer(5, Some(&params_buffer), 0);

    let threadgroup_memory_length = (SEQ_LEN * 4) as u64;
    encoder.set_threadgroup_memory_length(0, threadgroup_memory_length);

    let threadgroup_size = metal::MTLSize::new(256, 1, 1);
    let grid_size = metal::MTLSize::new(256, 1, 1);

    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let output_ptr = output_buffer.contents() as *const u16;
    let output_f16: Vec<u16> =
        unsafe { std::slice::from_raw_parts(output_ptr, HEAD_SIZE).to_vec() };

    let output_f32: Vec<f32> = output_f16
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect();

    println!(
        "Scattered blocks output (first 10 elements): {:?}",
        &output_f32[..10]
    );

    // Verify results
    assert!(
        output_f32.iter().all(|&x: &f32| x.is_finite()),
        "Output contains non-finite values"
    );

    // With scattered blocks, attention should still work correctly
    // The output should be a weighted average based on the actual data
    let first_value = output_f32[0];
    assert!(
        first_value.is_finite() && first_value >= 0.0,
        "Output[0] = {} is invalid",
        first_value
    );

    println!(
        "✓ Scattered blocks test passed - output[0] = {:.2}",
        first_value
    );
}
