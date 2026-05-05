use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferrite_metal_kernels::{MetalDevice, MetalStream};
use metal::MTLResourceOptions;
use std::sync::Arc;
use std::time::Duration;

fn benchmark_single_head_attention(c: &mut Criterion) {
    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs();
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let stream = MetalStream::new(&metal_device);

    let mut group = c.benchmark_group("single_head_attention");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for seq_len in [64, 128, 256, 512, 1024, 2048].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            seq_len,
            |b, &seq_len| {
                const HEAD_SIZE: usize = 128;
                const BLOCK_SIZE: usize = 16;
                let scale: f32 = 1.0 / (HEAD_SIZE as f32).sqrt();
                let num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

                // Create test data
                let q_data = vec![1.0f32; HEAD_SIZE];
                let k_data = vec![1.0f32; num_blocks * HEAD_SIZE * BLOCK_SIZE];
                let v_data = vec![1.0f32; num_blocks * HEAD_SIZE * BLOCK_SIZE];
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
                let params = PagedAttentionParams {
                    seq_len: seq_len as u32,
                    head_size: HEAD_SIZE as u32,
                    scale: scale,
                    block_size: BLOCK_SIZE as u32,
                    max_num_blocks: num_blocks as u32,
                    kv_block_stride: kv_head_stride,
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

                let library_source = std::fs::read_to_string(&library_path)
                    .expect("Failed to read attention_paged.metal");

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

                b.iter(|| {
                    let command_buffer = stream.queue().new_command_buffer();
                    let encoder = command_buffer.new_compute_command_encoder();

                    encoder.set_compute_pipeline_state(&pipeline_state);
                    encoder.set_buffer(0, Some(&q_buffer), 0);
                    encoder.set_buffer(1, Some(&k_buffer), 0);
                    encoder.set_buffer(2, Some(&v_buffer), 0);
                    encoder.set_buffer(3, Some(&block_table_buffer), 0);
                    encoder.set_buffer(4, Some(&output_buffer), 0);
                    encoder.set_buffer(5, Some(&params_buffer), 0);

                    let threadgroup_memory_length = (seq_len * 4) as u64;
                    encoder.set_threadgroup_memory_length(0, threadgroup_memory_length);

                    let threadgroup_size = metal::MTLSize::new(256, 1, 1);
                    let grid_size = metal::MTLSize::new(256, 1, 1);

                    encoder.dispatch_threads(grid_size, threadgroup_size);
                    encoder.end_encoding();

                    command_buffer.commit();
                    command_buffer.wait_until_completed();

                    black_box(&output_buffer);
                });
            },
        );
    }
    group.finish();
}

fn benchmark_multihead_attention(c: &mut Criterion) {
    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs();
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let stream = MetalStream::new(&metal_device);

    let mut group = c.benchmark_group("multihead_attention");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for (num_heads, seq_len) in [(4, 512), (8, 512), (16, 512), (32, 512)].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}heads_{}seq", num_heads, seq_len)),
            &(*num_heads, *seq_len),
            |b, &(num_heads, seq_len)| {
                const HEAD_SIZE: usize = 128;
                const BLOCK_SIZE: usize = 16;
                let scale: f32 = 1.0 / (HEAD_SIZE as f32).sqrt();
                let num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
                let num_kv_heads = num_heads; // No GQA for this benchmark

                let q_data = vec![1.0f32; num_heads * HEAD_SIZE];
                let k_data = vec![1.0f32; num_kv_heads * num_blocks * HEAD_SIZE * BLOCK_SIZE];
                let v_data = vec![1.0f32; num_kv_heads * num_blocks * HEAD_SIZE * BLOCK_SIZE];
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
                    (num_heads * HEAD_SIZE * 2) as u64,
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
                    seq_len: seq_len as u32,
                    head_size: HEAD_SIZE as u32,
                    num_heads: num_heads as u32,
                    num_kv_heads: num_kv_heads as u32,
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

                let library_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("shaders")
                    .join("attention_multihead.metal");

                let library_source = std::fs::read_to_string(&library_path)
                    .expect("Failed to read attention_multihead.metal");

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

                b.iter(|| {
                    let command_buffer = stream.queue().new_command_buffer();
                    let encoder = command_buffer.new_compute_command_encoder();

                    encoder.set_compute_pipeline_state(&pipeline_state);
                    encoder.set_buffer(0, Some(&q_buffer), 0);
                    encoder.set_buffer(1, Some(&k_buffer), 0);
                    encoder.set_buffer(2, Some(&v_buffer), 0);
                    encoder.set_buffer(3, Some(&block_table_buffer), 0);
                    encoder.set_buffer(4, Some(&output_buffer), 0);
                    encoder.set_buffer(5, Some(&params_buffer), 0);

                    let threadgroup_memory_length = (seq_len * 4) as u64;
                    encoder.set_threadgroup_memory_length(0, threadgroup_memory_length);

                    let threadgroup_size = metal::MTLSize::new(256, 1, 1);
                    let grid_size = metal::MTLSize::new(num_heads as u64, 1, 1);

                    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
                    encoder.end_encoding();

                    command_buffer.commit();
                    command_buffer.wait_until_completed();

                    black_box(&output_buffer);
                });
            },
        );
    }
    group.finish();
}

fn benchmark_optimized_attention(c: &mut Criterion) {
    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs();
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let stream = MetalStream::new(&metal_device);

    let mut group = c.benchmark_group("optimized_vs_baseline");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for seq_len in [256, 512, 1024].iter() {
        // Baseline
        group.bench_with_input(
            BenchmarkId::new("baseline", seq_len),
            seq_len,
            |b, &seq_len| {
                const HEAD_SIZE: usize = 128;
                const NUM_HEADS: usize = 8;
                const NUM_KV_HEADS: usize = 2;
                const BLOCK_SIZE: usize = 16;
                let scale: f32 = 1.0 / (HEAD_SIZE as f32).sqrt();
                let num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

                let q_data = vec![1.0f32; NUM_HEADS * HEAD_SIZE];
                let k_data = vec![1.0f32; NUM_KV_HEADS * num_blocks * HEAD_SIZE * BLOCK_SIZE];
                let v_data = vec![1.0f32; NUM_KV_HEADS * num_blocks * HEAD_SIZE * BLOCK_SIZE];
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
                let params = MultiHeadAttentionParams {
                    seq_len: seq_len as u32,
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

                let library_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("shaders")
                    .join("attention_multihead.metal");

                let library_source = std::fs::read_to_string(&library_path)
                    .expect("Failed to read attention_multihead.metal");

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

                b.iter(|| {
                    let command_buffer = stream.queue().new_command_buffer();
                    let encoder = command_buffer.new_compute_command_encoder();

                    encoder.set_compute_pipeline_state(&pipeline_state);
                    encoder.set_buffer(0, Some(&q_buffer), 0);
                    encoder.set_buffer(1, Some(&k_buffer), 0);
                    encoder.set_buffer(2, Some(&v_buffer), 0);
                    encoder.set_buffer(3, Some(&block_table_buffer), 0);
                    encoder.set_buffer(4, Some(&output_buffer), 0);
                    encoder.set_buffer(5, Some(&params_buffer), 0);

                    let threadgroup_memory_length = (seq_len * 4) as u64;
                    encoder.set_threadgroup_memory_length(0, threadgroup_memory_length);

                    let threadgroup_size = metal::MTLSize::new(256, 1, 1);
                    let grid_size = metal::MTLSize::new(NUM_HEADS as u64, 1, 1);

                    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
                    encoder.end_encoding();

                    command_buffer.commit();
                    command_buffer.wait_until_completed();

                    black_box(&output_buffer);
                });
            },
        );

        // Optimized
        group.bench_with_input(
            BenchmarkId::new("optimized", seq_len),
            seq_len,
            |b, &seq_len| {
                const HEAD_SIZE: usize = 128;
                const NUM_HEADS: usize = 8;
                const NUM_KV_HEADS: usize = 2;
                const BLOCK_SIZE: usize = 16;
                let scale: f32 = 1.0 / (HEAD_SIZE as f32).sqrt();
                let num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

                let q_data = vec![1.0f32; NUM_HEADS * HEAD_SIZE];
                let k_data = vec![1.0f32; NUM_KV_HEADS * num_blocks * HEAD_SIZE * BLOCK_SIZE];
                let v_data = vec![1.0f32; NUM_KV_HEADS * num_blocks * HEAD_SIZE * BLOCK_SIZE];
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
                let params = MultiHeadAttentionParams {
                    seq_len: seq_len as u32,
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

                let library_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("shaders")
                    .join("attention_multihead_optimized.metal");

                let library_source = std::fs::read_to_string(&library_path)
                    .expect("Failed to read attention_multihead_optimized.metal");

                let library = device
                    .device
                    .new_library_with_source(&library_source, &metal::CompileOptions::new())
                    .expect("Failed to compile shader library");

                let kernel_function = library
                    .get_function("attention_multihead_paged_optimized", None)
                    .expect("Failed to get kernel function");

                let pipeline_state = device
                    .device
                    .new_compute_pipeline_state_with_function(&kernel_function)
                    .expect("Failed to create pipeline state");

                b.iter(|| {
                    let command_buffer = stream.queue().new_command_buffer();
                    let encoder = command_buffer.new_compute_command_encoder();

                    encoder.set_compute_pipeline_state(&pipeline_state);
                    encoder.set_buffer(0, Some(&q_buffer), 0);
                    encoder.set_buffer(1, Some(&k_buffer), 0);
                    encoder.set_buffer(2, Some(&v_buffer), 0);
                    encoder.set_buffer(3, Some(&block_table_buffer), 0);
                    encoder.set_buffer(4, Some(&output_buffer), 0);
                    encoder.set_buffer(5, Some(&params_buffer), 0);

                    let threadgroup_memory_length = (seq_len * 4) as u64;
                    encoder.set_threadgroup_memory_length(0, threadgroup_memory_length);

                    let threadgroup_size = metal::MTLSize::new(256, 1, 1);
                    let grid_size = metal::MTLSize::new(NUM_HEADS as u64, 1, 1);

                    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
                    encoder.end_encoding();

                    command_buffer.commit();
                    command_buffer.wait_until_completed();

                    black_box(&output_buffer);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_single_head_attention,
    benchmark_multihead_attention,
    benchmark_optimized_attention
);
criterion_main!(benches);
