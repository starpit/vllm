use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferrite_metal_kernels::{awq::MetalAwq, MetalDevice};
use metal::MTLResourceOptions;
use std::sync::Arc;
use std::time::Duration;

fn benchmark_awq_unpack(c: &mut Criterion) {
    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs();
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let awq = MetalAwq::new(device.clone()).expect("Failed to create AWQ");

    let mut group = c.benchmark_group("awq_unpack_int4");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    // Test different layer sizes: (IC, OC)
    // Typical sizes for Llama-7B: 4096x4096, 4096x11008, 11008x4096
    for (ic, oc) in [
        (4096, 4096),  // Self-attention Q/K/V projections
        (4096, 11008), // MLP up projection
        (11008, 4096), // MLP down projection
    ]
    .iter()
    {
        let size_label = format!("{}x{}", ic, oc);

        group.bench_with_input(
            BenchmarkId::from_parameter(&size_label),
            &(*ic, *oc),
            |b, &(ic, oc)| {
                // Create packed INT4 weights (8 per uint32)
                let num_packed = ic * (oc / 8);
                let packed_data: Vec<u32> = (0..num_packed)
                    .map(|i| {
                        // Pack 8 INT4 values (0-15) into one uint32
                        let mut packed = 0u32;
                        for j in 0..8 {
                            let val = ((i + j) % 16) as u32;
                            packed |= val << (j * 4);
                        }
                        packed
                    })
                    .collect();

                let packed_buf = device.device.new_buffer_with_data(
                    packed_data.as_ptr() as *const _,
                    (packed_data.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                b.iter(|| {
                    let output = awq.unpack_int4(&packed_buf, ic, oc).expect("Unpack failed");

                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

fn benchmark_awq_dequantize(c: &mut Criterion) {
    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs();
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let awq = MetalAwq::new(device.clone()).expect("Failed to create AWQ");

    let mut group = c.benchmark_group("awq_dequantize");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    let group_size = 128u32;

    for (ic, oc) in [(4096, 4096), (4096, 11008), (11008, 4096)].iter() {
        let size_label = format!("{}x{}", ic, oc);

        group.bench_with_input(
            BenchmarkId::from_parameter(&size_label),
            &(*ic, *oc),
            |b, &(ic, oc)| {
                // Create packed INT4 weights
                let num_packed = ic * (oc / 8);
                let packed_weights: Vec<u32> = (0..num_packed)
                    .map(|i| {
                        let mut packed = 0u32;
                        for j in 0..8 {
                            let val = ((i + j) % 16) as u32;
                            packed |= val << (j * 4);
                        }
                        packed
                    })
                    .collect();

                // Create scales (FP16)
                let num_groups = ic / group_size as usize;
                let scales: Vec<u16> = (0..num_groups * oc)
                    .map(|i| half::f16::from_f32(0.01 + (i as f32) * 0.0001).to_bits())
                    .collect();

                // Create packed zeros
                let num_packed_zeros = num_groups * (oc / 8);
                let packed_zeros: Vec<u32> = (0..num_packed_zeros)
                    .map(|i| {
                        let mut packed = 0u32;
                        for j in 0..8 {
                            let val = ((i + j) % 8) as u32; // Zeros typically 0-7
                            packed |= val << (j * 4);
                        }
                        packed
                    })
                    .collect();

                let packed_weights_buf = device.device.new_buffer_with_data(
                    packed_weights.as_ptr() as *const _,
                    (packed_weights.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let scales_buf = device.device.new_buffer_with_data(
                    scales.as_ptr() as *const _,
                    (scales.len() * 2) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let packed_zeros_buf = device.device.new_buffer_with_data(
                    packed_zeros.as_ptr() as *const _,
                    (packed_zeros.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                b.iter(|| {
                    let output = awq
                        .dequantize(
                            &packed_weights_buf,
                            &scales_buf,
                            &packed_zeros_buf,
                            group_size,
                            ic,
                            oc,
                        )
                        .expect("Dequantize failed");

                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

fn benchmark_awq_dequantize_vec4(c: &mut Criterion) {
    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs();
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let awq = MetalAwq::new(device.clone()).expect("Failed to create AWQ");

    let mut group = c.benchmark_group("awq_dequantize_vec4");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    let group_size = 128u32;

    for (ic, oc) in [(4096, 4096), (4096, 11008), (11008, 4096)].iter() {
        let size_label = format!("{}x{}", ic, oc);

        group.bench_with_input(
            BenchmarkId::from_parameter(&size_label),
            &(*ic, *oc),
            |b, &(ic, oc)| {
                // Create packed INT4 weights
                let num_packed = ic * (oc / 8);
                let packed_weights: Vec<u32> = (0..num_packed)
                    .map(|i| {
                        let mut packed = 0u32;
                        for j in 0..8 {
                            let val = ((i + j) % 16) as u32;
                            packed |= val << (j * 4);
                        }
                        packed
                    })
                    .collect();

                // Create scales (FP16)
                let num_groups = ic / group_size as usize;
                let scales: Vec<u16> = (0..num_groups * oc)
                    .map(|i| half::f16::from_f32(0.01 + (i as f32) * 0.0001).to_bits())
                    .collect();

                // Create packed zeros
                let num_packed_zeros = num_groups * (oc / 8);
                let packed_zeros: Vec<u32> = (0..num_packed_zeros)
                    .map(|i| {
                        let mut packed = 0u32;
                        for j in 0..8 {
                            let val = ((i + j) % 8) as u32;
                            packed |= val << (j * 4);
                        }
                        packed
                    })
                    .collect();

                let packed_weights_buf = device.device.new_buffer_with_data(
                    packed_weights.as_ptr() as *const _,
                    (packed_weights.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let scales_buf = device.device.new_buffer_with_data(
                    scales.as_ptr() as *const _,
                    (scales.len() * 2) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let packed_zeros_buf = device.device.new_buffer_with_data(
                    packed_zeros.as_ptr() as *const _,
                    (packed_zeros.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                b.iter(|| {
                    let output = awq
                        .dequantize_vec4(
                            &packed_weights_buf,
                            &scales_buf,
                            &packed_zeros_buf,
                            group_size,
                            ic,
                            oc,
                        )
                        .expect("Dequantize vec4 failed");

                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

fn benchmark_awq_dequantize_and_gemm(c: &mut Criterion) {
    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs();
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let awq = MetalAwq::new(device.clone()).expect("Failed to create AWQ");

    let mut group = c.benchmark_group("awq_dequantize_and_gemm");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    let group_size = 128u32;

    // Test with different batch sizes
    for (m, ic, oc) in [
        (1, 4096, 4096),  // Single token
        (4, 4096, 4096),  // Small batch
        (16, 4096, 4096), // Medium batch
        (1, 4096, 11008), // MLP up
        (1, 11008, 4096), // MLP down
    ]
    .iter()
    {
        let size_label = format!("{}x{}x{}", m, ic, oc);

        group.bench_with_input(
            BenchmarkId::from_parameter(&size_label),
            &(*m, *ic, *oc),
            |b, &(m, ic, oc)| {
                // Create activations (FP16)
                let activations: Vec<u16> = (0..m * ic)
                    .map(|i| half::f16::from_f32((i as f32) * 0.01).to_bits())
                    .collect();

                // Create packed INT4 weights
                let num_packed = ic * (oc / 8);
                let packed_weights: Vec<u32> = (0..num_packed)
                    .map(|i| {
                        let mut packed = 0u32;
                        for j in 0..8 {
                            let val = ((i + j) % 16) as u32;
                            packed |= val << (j * 4);
                        }
                        packed
                    })
                    .collect();

                // Create scales (FP16)
                let num_groups = ic / group_size as usize;
                let scales: Vec<u16> = (0..num_groups * oc)
                    .map(|i| half::f16::from_f32(0.01 + (i as f32) * 0.0001).to_bits())
                    .collect();

                // Create packed zeros
                let num_packed_zeros = num_groups * (oc / 8);
                let packed_zeros: Vec<u32> = (0..num_packed_zeros)
                    .map(|i| {
                        let mut packed = 0u32;
                        for j in 0..8 {
                            let val = ((i + j) % 8) as u32;
                            packed |= val << (j * 4);
                        }
                        packed
                    })
                    .collect();

                let activations_buf = device.device.new_buffer_with_data(
                    activations.as_ptr() as *const _,
                    (activations.len() * 2) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let packed_weights_buf = device.device.new_buffer_with_data(
                    packed_weights.as_ptr() as *const _,
                    (packed_weights.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let scales_buf = device.device.new_buffer_with_data(
                    scales.as_ptr() as *const _,
                    (scales.len() * 2) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let packed_zeros_buf = device.device.new_buffer_with_data(
                    packed_zeros.as_ptr() as *const _,
                    (packed_zeros.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                b.iter(|| {
                    let output = awq
                        .dequantize_and_gemm(
                            &activations_buf,
                            &packed_weights_buf,
                            &scales_buf,
                            &packed_zeros_buf,
                            group_size,
                            m,
                            ic,
                            oc,
                        )
                        .expect("Dequantize and GEMM failed");

                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

fn benchmark_awq_vectorization_comparison(c: &mut Criterion) {
    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs();
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let awq = MetalAwq::new(device.clone()).expect("Failed to create AWQ");

    let mut group = c.benchmark_group("awq_vectorization_comparison");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    let group_size = 128u32;
    let ic = 4096;
    let oc = 4096;

    // Create test data once
    let num_packed = ic * (oc / 8);
    let packed_weights: Vec<u32> = (0..num_packed)
        .map(|i| {
            let mut packed = 0u32;
            for j in 0..8 {
                let val = ((i + j) % 16) as u32;
                packed |= val << (j * 4);
            }
            packed
        })
        .collect();

    let num_groups = ic / group_size as usize;
    let scales: Vec<u16> = (0..num_groups * oc)
        .map(|i| half::f16::from_f32(0.01 + (i as f32) * 0.0001).to_bits())
        .collect();

    let num_packed_zeros = num_groups * (oc / 8);
    let packed_zeros: Vec<u32> = (0..num_packed_zeros)
        .map(|i| {
            let mut packed = 0u32;
            for j in 0..8 {
                let val = ((i + j) % 8) as u32;
                packed |= val << (j * 4);
            }
            packed
        })
        .collect();

    let packed_weights_buf = device.device.new_buffer_with_data(
        packed_weights.as_ptr() as *const _,
        (packed_weights.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let scales_buf = device.device.new_buffer_with_data(
        scales.as_ptr() as *const _,
        (scales.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let packed_zeros_buf = device.device.new_buffer_with_data(
        packed_zeros.as_ptr() as *const _,
        (packed_zeros.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Benchmark scalar version
    group.bench_function("scalar", |b| {
        b.iter(|| {
            let output = awq
                .dequantize(
                    &packed_weights_buf,
                    &scales_buf,
                    &packed_zeros_buf,
                    group_size,
                    ic,
                    oc,
                )
                .expect("Dequantize failed");

            black_box(&output);
        });
    });

    // Benchmark vectorized version
    group.bench_function("vec4", |b| {
        b.iter(|| {
            let output = awq
                .dequantize_vec4(
                    &packed_weights_buf,
                    &scales_buf,
                    &packed_zeros_buf,
                    group_size,
                    ic,
                    oc,
                )
                .expect("Dequantize vec4 failed");

            black_box(&output);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_awq_unpack,
    benchmark_awq_dequantize,
    benchmark_awq_dequantize_vec4,
    benchmark_awq_dequantize_and_gemm,
    benchmark_awq_vectorization_comparison
);
criterion_main!(benches);
