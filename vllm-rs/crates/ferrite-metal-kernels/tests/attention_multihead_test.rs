use ferrite_metal_kernels::{MetalDevice, MetalStream};
use metal::MTLResourceOptions;
use std::sync::Arc;

#[test]
fn test_multihead_attention_basic() {
    // Test basic multi-head attention with independent heads
    const HEAD_SIZE: usize = 64;
    const NUM_HEADS: usize = 4;
    const NUM_KV_HEADS: usize = 4; // Same as num_heads (no GQA)
    const SEQ_LEN: usize = 64;
    const BLOCK_SIZE: usize = 16;
    let scale: f32 = 1.0 / (HEAD_SIZE as f32).sqrt();

    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs();
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let stream = MetalStream::new(&metal_device);

    let num_blocks = (SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Create test data
    let mut q_data = vec![0.0f32; NUM_HEADS * HEAD_SIZE];
    let mut k_data = vec![0.0f32; NUM_KV_HEADS * num_blocks * HEAD_SIZE * BLOCK_SIZE];
    let mut v_data = vec![0.0f32; NUM_KV_HEADS * num_blocks * HEAD_SIZE * BLOCK_SIZE];

    // Initialize Q: Different pattern for each head
    for head_idx in 0..NUM_HEADS {
        let q_offset = head_idx * HEAD_SIZE;
        q_data[q_offset] = 1.0 + (head_idx as f32) * 0.1; // Q[head][0] = 1.0, 1.1, 1.2, 1.3
    }

    // Initialize K: Each head has different values
    for kv_head_idx in 0..NUM_KV_HEADS {
        for block_idx in 0..num_blocks {
            for block_offset in 0..BLOCK_SIZE {
                let token_idx = block_idx * BLOCK_SIZE + block_offset;
                if token_idx < SEQ_LEN {
                    let k_idx = kv_head_idx * num_blocks * HEAD_SIZE * BLOCK_SIZE
                        + block_idx * HEAD_SIZE * BLOCK_SIZE
                        + 0 * BLOCK_SIZE
                        + block_offset;
                    k_data[k_idx] = 1.0 + (kv_head_idx as f32) * 0.1 + (token_idx as f32) * 0.01;
                }
            }
        }
    }

    // Initialize V: Each head has different values
    for kv_head_idx in 0..NUM_KV_HEADS {
        for block_idx in 0..num_blocks {
            for block_offset in 0..BLOCK_SIZE {
                let token_idx = block_idx * BLOCK_SIZE + block_offset;
                if token_idx < SEQ_LEN {
                    let v_idx = kv_head_idx * num_blocks * HEAD_SIZE * BLOCK_SIZE
                        + block_idx * HEAD_SIZE * BLOCK_SIZE
                        + 0 * BLOCK_SIZE
                        + block_offset;
                    v_data[v_idx] = (kv_head_idx as f32) * 100.0 + (token_idx as f32);
                }
            }
        }
    }

    // Create block table (identity mapping)
    let block_table: Vec<i32> = (0..num_blocks as i32).collect();

    // Convert to f16
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
        (NUM_HEADS * HEAD_SIZE * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create params buffer
    #[repr(C)]
    struct MultiHeadAttentionParams {
        seq_len: u32,
        head_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        scale: f32,
        block_size: u32,
        max_num_blocks: u32,
        kv_block_stride: u32,
        kv_head_stride: u32,
    }

    let kv_head_stride = HEAD_SIZE as u32 * BLOCK_SIZE as u32;
    let kv_block_stride = kv_head_stride;

    let params = MultiHeadAttentionParams {
        seq_len: SEQ_LEN as u32,
        head_size: HEAD_SIZE as u32,
        num_heads: NUM_HEADS as u32,
        num_kv_heads: NUM_KV_HEADS as u32,
        scale: scale,
        block_size: BLOCK_SIZE as u32,
        max_num_blocks: num_blocks as u32,
        kv_block_stride: kv_block_stride,
        kv_head_stride: kv_head_stride,
    };

    let params_buffer = device.device.new_buffer_with_data(
        &params as *const _ as *const _,
        std::mem::size_of::<MultiHeadAttentionParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Load shader library
    let library_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .join("attention_multihead.metal");

    let library_source =
        std::fs::read_to_string(&library_path).expect("Failed to read attention_multihead.metal");

    let library = device
        .device
        .new_library_with_source(&library_source, &metal::CompileOptions::new())
        .expect("Failed to compile shader library");

    let kernel_function = library
        .get_function("attention_multihead_paged", None)
        .expect("Failed to get kernel function");

    let pipeline_state = device
        .device
        .new_compute_pipeline_state_with_function(&kernel_function)
        .expect("Failed to create pipeline state");

    // Execute kernel - one threadgroup per head
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
    let threadgroup_memory_length = (SEQ_LEN * 4) as u64;
    encoder.set_threadgroup_memory_length(0, threadgroup_memory_length);

    // Dispatch: one threadgroup per head
    let threadgroup_size = metal::MTLSize::new(256, 1, 1);
    let grid_size = metal::MTLSize::new(NUM_HEADS as u64, 1, 1);

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back results
    let output_ptr = output_buffer.contents() as *const u16;
    let output_f16: Vec<u16> =
        unsafe { std::slice::from_raw_parts(output_ptr, NUM_HEADS * HEAD_SIZE).to_vec() };

    let output_f32: Vec<f32> = output_f16
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect();

    println!("Multi-head attention output:");
    for head_idx in 0..NUM_HEADS {
        let head_offset = head_idx * HEAD_SIZE;
        println!(
            "  Head {}: [{:.2}, {:.2}, {:.2}, ...]",
            head_idx,
            output_f32[head_offset],
            output_f32[head_offset + 1],
            output_f32[head_offset + 2]
        );
    }

    // Verify results
    assert!(
        output_f32.iter().all(|&x: &f32| x.is_finite()),
        "Output contains non-finite values"
    );

    // Each head should have different output (since they have different Q, K, V)
    for head_idx in 0..NUM_HEADS {
        let head_offset = head_idx * HEAD_SIZE;
        let first_value = output_f32[head_offset];

        // Expected range based on V values: head_idx * 100 + weighted average of token indices
        let expected_min = (head_idx as f32) * 100.0 - 50.0;
        let expected_max = (head_idx as f32) * 100.0 + 100.0;

        assert!(
            first_value > expected_min && first_value < expected_max,
            "Head {} output[0] = {} is outside expected range [{}, {}]",
            head_idx,
            first_value,
            expected_min,
            expected_max
        );
    }

    println!("✓ Multi-head attention test passed");
}

#[test]
fn test_gqa_attention() {
    // Test Grouped Query Attention: 8 Q heads share 2 KV heads (4:1 ratio)
    const HEAD_SIZE: usize = 64;
    const NUM_HEADS: usize = 8;
    const NUM_KV_HEADS: usize = 2; // GQA: 8 Q heads / 2 KV heads = 4 Q heads per KV head
    const SEQ_LEN: usize = 64;
    const BLOCK_SIZE: usize = 16;
    let scale: f32 = 1.0 / (HEAD_SIZE as f32).sqrt();

    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs();
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let stream = MetalStream::new(&metal_device);

    let num_blocks = (SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Create test data
    let mut q_data = vec![0.0f32; NUM_HEADS * HEAD_SIZE];
    let mut k_data = vec![0.0f32; NUM_KV_HEADS * num_blocks * HEAD_SIZE * BLOCK_SIZE];
    let mut v_data = vec![0.0f32; NUM_KV_HEADS * num_blocks * HEAD_SIZE * BLOCK_SIZE];

    // Initialize Q: Different pattern for each head
    for head_idx in 0..NUM_HEADS {
        let q_offset = head_idx * HEAD_SIZE;
        q_data[q_offset] = 1.0 + (head_idx as f32) * 0.1;
    }

    // Initialize K: Only 2 KV heads
    for kv_head_idx in 0..NUM_KV_HEADS {
        for block_idx in 0..num_blocks {
            for block_offset in 0..BLOCK_SIZE {
                let token_idx = block_idx * BLOCK_SIZE + block_offset;
                if token_idx < SEQ_LEN {
                    let k_idx = kv_head_idx * num_blocks * HEAD_SIZE * BLOCK_SIZE
                        + block_idx * HEAD_SIZE * BLOCK_SIZE
                        + 0 * BLOCK_SIZE
                        + block_offset;
                    k_data[k_idx] = 1.0 + (kv_head_idx as f32) * 0.5 + (token_idx as f32) * 0.01;
                }
            }
        }
    }

    // Initialize V: Only 2 KV heads with distinct values
    for kv_head_idx in 0..NUM_KV_HEADS {
        for block_idx in 0..num_blocks {
            for block_offset in 0..BLOCK_SIZE {
                let token_idx = block_idx * BLOCK_SIZE + block_offset;
                if token_idx < SEQ_LEN {
                    let v_idx = kv_head_idx * num_blocks * HEAD_SIZE * BLOCK_SIZE
                        + block_idx * HEAD_SIZE * BLOCK_SIZE
                        + 0 * BLOCK_SIZE
                        + block_offset;
                    // KV head 0: values around 0-100
                    // KV head 1: values around 1000-1100
                    v_data[v_idx] = (kv_head_idx as f32) * 1000.0 + (token_idx as f32);
                }
            }
        }
    }

    let block_table: Vec<i32> = (0..num_blocks as i32).collect();

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
        (NUM_HEADS * HEAD_SIZE * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    #[repr(C)]
    struct MultiHeadAttentionParams {
        seq_len: u32,
        head_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        scale: f32,
        block_size: u32,
        max_num_blocks: u32,
        kv_block_stride: u32,
        kv_head_stride: u32,
    }

    let kv_head_stride = HEAD_SIZE as u32 * BLOCK_SIZE as u32;
    let kv_block_stride = kv_head_stride;

    let params = MultiHeadAttentionParams {
        seq_len: SEQ_LEN as u32,
        head_size: HEAD_SIZE as u32,
        num_heads: NUM_HEADS as u32,
        num_kv_heads: NUM_KV_HEADS as u32,
        scale: scale,
        block_size: BLOCK_SIZE as u32,
        max_num_blocks: num_blocks as u32,
        kv_block_stride: kv_block_stride,
        kv_head_stride: kv_head_stride,
    };

    let params_buffer = device.device.new_buffer_with_data(
        &params as *const _ as *const _,
        std::mem::size_of::<MultiHeadAttentionParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let library_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .join("attention_multihead.metal");

    let library_source =
        std::fs::read_to_string(&library_path).expect("Failed to read attention_multihead.metal");

    let library = device
        .device
        .new_library_with_source(&library_source, &metal::CompileOptions::new())
        .expect("Failed to compile shader library");

    let kernel_function = library
        .get_function("attention_multihead_paged", None)
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
    let grid_size = metal::MTLSize::new(NUM_HEADS as u64, 1, 1);

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let output_ptr = output_buffer.contents() as *const u16;
    let output_f16: Vec<u16> =
        unsafe { std::slice::from_raw_parts(output_ptr, NUM_HEADS * HEAD_SIZE).to_vec() };

    let output_f32: Vec<f32> = output_f16
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect();

    println!("GQA attention output (8 Q heads, 2 KV heads):");
    for head_idx in 0..NUM_HEADS {
        let head_offset = head_idx * HEAD_SIZE;
        let kv_head_idx = head_idx / 4; // 4 Q heads per KV head
        println!(
            "  Q head {} (uses KV head {}): [{:.2}, {:.2}, {:.2}, ...]",
            head_idx,
            kv_head_idx,
            output_f32[head_offset],
            output_f32[head_offset + 1],
            output_f32[head_offset + 2]
        );
    }

    // Verify results
    assert!(
        output_f32.iter().all(|&x: &f32| x.is_finite()),
        "Output contains non-finite values"
    );

    // Verify GQA grouping: heads 0-3 should use KV head 0, heads 4-7 should use KV head 1
    // KV head 0 has V values around 0-100
    // KV head 1 has V values around 1000-1100

    for head_idx in 0..4 {
        let head_offset = head_idx * HEAD_SIZE;
        let first_value = output_f32[head_offset];
        assert!(
            first_value >= 0.0 && first_value < 200.0,
            "Q head {} (KV head 0) output[0] = {} should be in range [0, 200]",
            head_idx,
            first_value
        );
    }

    for head_idx in 4..8 {
        let head_offset = head_idx * HEAD_SIZE;
        let first_value = output_f32[head_offset];
        assert!(
            first_value >= 900.0 && first_value < 1200.0,
            "Q head {} (KV head 1) output[0] = {} should be in range [900, 1200]",
            head_idx,
            first_value
        );
    }

    println!("✓ GQA attention test passed - verified 4:1 Q:KV head grouping");
}
