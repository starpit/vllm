use ferrite_metal_kernels::{MetalDevice, MetalStream};
use metal::MTLResourceOptions;
use std::sync::Arc;

#[test]
fn test_optimized_vs_baseline_correctness() {
    // Compare optimized vectorized version against baseline
    // HEAD_SIZE must be multiple of 4 for vectorization
    const HEAD_SIZE: usize = 128; // 128 / 4 = 32 vectors
    const NUM_HEADS: usize = 8;
    const NUM_KV_HEADS: usize = 2; // GQA: 4:1 ratio
    const SEQ_LEN: usize = 256;
    const BLOCK_SIZE: usize = 16;
    let scale: f32 = 1.0 / (HEAD_SIZE as f32).sqrt();

    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs();
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let stream = MetalStream::new(&metal_device);

    let num_blocks = (SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Create test data with known pattern
    let mut q_data = vec![0.0f32; NUM_HEADS * HEAD_SIZE];
    let mut k_data = vec![0.0f32; NUM_KV_HEADS * num_blocks * HEAD_SIZE * BLOCK_SIZE];
    let mut v_data = vec![0.0f32; NUM_KV_HEADS * num_blocks * HEAD_SIZE * BLOCK_SIZE];

    // Initialize with deterministic pattern
    for head_idx in 0..NUM_HEADS {
        for i in 0..HEAD_SIZE {
            q_data[head_idx * HEAD_SIZE + i] = ((head_idx + i) % 10) as f32 * 0.1;
        }
    }

    for kv_head_idx in 0..NUM_KV_HEADS {
        for block_idx in 0..num_blocks {
            for i in 0..HEAD_SIZE {
                for block_offset in 0..BLOCK_SIZE {
                    let token_idx = block_idx * BLOCK_SIZE + block_offset;
                    if token_idx < SEQ_LEN {
                        let k_idx = kv_head_idx * num_blocks * HEAD_SIZE * BLOCK_SIZE
                            + block_idx * HEAD_SIZE * BLOCK_SIZE
                            + i * BLOCK_SIZE
                            + block_offset;
                        let v_idx = k_idx;

                        k_data[k_idx] = ((kv_head_idx + i + token_idx) % 10) as f32 * 0.1;
                        v_data[v_idx] = ((kv_head_idx * 100 + token_idx) % 100) as f32;
                    }
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

    // Create buffers (shared for both kernels)
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

    let output_baseline = device.device.new_buffer(
        (NUM_HEADS * HEAD_SIZE * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_optimized = device.device.new_buffer(
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
    let params = MultiHeadAttentionParams {
        seq_len: SEQ_LEN as u32,
        head_size: HEAD_SIZE as u32,
        num_heads: NUM_HEADS as u32,
        num_kv_heads: NUM_KV_HEADS as u32,
        scale: scale,
        block_size: BLOCK_SIZE as u32,
        max_num_blocks: num_blocks as u32,
        kv_block_stride: kv_head_stride,
        kv_head_stride: kv_head_stride,
    };

    let params_buffer = device.device.new_buffer_with_data(
        &params as *const _ as *const _,
        std::mem::size_of::<MultiHeadAttentionParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Load both shader libraries
    let baseline_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .join("attention_multihead.metal");

    let optimized_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .join("attention_multihead_optimized.metal");

    let baseline_source =
        std::fs::read_to_string(&baseline_path).expect("Failed to read attention_multihead.metal");

    let optimized_source = std::fs::read_to_string(&optimized_path)
        .expect("Failed to read attention_multihead_optimized.metal");

    let baseline_library = device
        .device
        .new_library_with_source(&baseline_source, &metal::CompileOptions::new())
        .expect("Failed to compile baseline shader");

    let optimized_library = device
        .device
        .new_library_with_source(&optimized_source, &metal::CompileOptions::new())
        .expect("Failed to compile optimized shader");

    let baseline_function = baseline_library
        .get_function("attention_multihead_paged", None)
        .expect("Failed to get baseline kernel");

    let optimized_function = optimized_library
        .get_function("attention_multihead_paged_optimized", None)
        .expect("Failed to get optimized kernel");

    let baseline_pipeline = device
        .device
        .new_compute_pipeline_state_with_function(&baseline_function)
        .expect("Failed to create baseline pipeline");

    let optimized_pipeline = device
        .device
        .new_compute_pipeline_state_with_function(&optimized_function)
        .expect("Failed to create optimized pipeline");

    // Run baseline kernel
    {
        let command_buffer = stream.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&baseline_pipeline);
        encoder.set_buffer(0, Some(&q_buffer), 0);
        encoder.set_buffer(1, Some(&k_buffer), 0);
        encoder.set_buffer(2, Some(&v_buffer), 0);
        encoder.set_buffer(3, Some(&block_table_buffer), 0);
        encoder.set_buffer(4, Some(&output_baseline), 0);
        encoder.set_buffer(5, Some(&params_buffer), 0);

        let threadgroup_memory_length = (SEQ_LEN * 4) as u64;
        encoder.set_threadgroup_memory_length(0, threadgroup_memory_length);

        let threadgroup_size = metal::MTLSize::new(256, 1, 1);
        let grid_size = metal::MTLSize::new(NUM_HEADS as u64, 1, 1);

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // Run optimized kernel
    {
        let command_buffer = stream.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&optimized_pipeline);
        encoder.set_buffer(0, Some(&q_buffer), 0);
        encoder.set_buffer(1, Some(&k_buffer), 0);
        encoder.set_buffer(2, Some(&v_buffer), 0);
        encoder.set_buffer(3, Some(&block_table_buffer), 0);
        encoder.set_buffer(4, Some(&output_optimized), 0);
        encoder.set_buffer(5, Some(&params_buffer), 0);

        let threadgroup_memory_length = (SEQ_LEN * 4) as u64;
        encoder.set_threadgroup_memory_length(0, threadgroup_memory_length);

        let threadgroup_size = metal::MTLSize::new(256, 1, 1);
        let grid_size = metal::MTLSize::new(NUM_HEADS as u64, 1, 1);

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // Read back and compare results
    let baseline_ptr = output_baseline.contents() as *const u16;
    let optimized_ptr = output_optimized.contents() as *const u16;

    let baseline_f16: Vec<u16> =
        unsafe { std::slice::from_raw_parts(baseline_ptr, NUM_HEADS * HEAD_SIZE).to_vec() };

    let optimized_f16: Vec<u16> =
        unsafe { std::slice::from_raw_parts(optimized_ptr, NUM_HEADS * HEAD_SIZE).to_vec() };

    let baseline_f32: Vec<f32> = baseline_f16
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect();

    let optimized_f32: Vec<f32> = optimized_f16
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect();

    println!("Comparing baseline vs optimized attention:");
    println!(
        "  Baseline head 0: [{:.4}, {:.4}, {:.4}, ...]",
        baseline_f32[0], baseline_f32[1], baseline_f32[2]
    );
    println!(
        "  Optimized head 0: [{:.4}, {:.4}, {:.4}, ...]",
        optimized_f32[0], optimized_f32[1], optimized_f32[2]
    );

    // Verify numerical correctness (allow small FP16 error)
    let max_error = baseline_f32
        .iter()
        .zip(optimized_f32.iter())
        .map(|(b, o)| (b - o).abs())
        .fold(0.0f32, f32::max);

    println!("  Max absolute error: {:.6}", max_error);

    // FP16 precision: ~0.001 (0.1% relative error is acceptable)
    assert!(
        max_error < 0.1,
        "Optimized kernel differs from baseline by {:.6} (threshold: 0.1)",
        max_error
    );

    // Verify all outputs are finite
    assert!(
        baseline_f32.iter().all(|&x| x.is_finite()),
        "Baseline output contains non-finite values"
    );
    assert!(
        optimized_f32.iter().all(|&x| x.is_finite()),
        "Optimized output contains non-finite values"
    );

    println!(
        "✓ Optimized kernel matches baseline (max error: {:.6})",
        max_error
    );
}
