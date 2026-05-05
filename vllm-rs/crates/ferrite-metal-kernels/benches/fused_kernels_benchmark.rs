use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferrite_metal_kernels::{fused_kernels::*, MetalDevice, MetalStream};
use metal::MTLResourceOptions;
use std::sync::Arc;
use std::time::Duration;

fn benchmark_fused_add_rmsnorm(c: &mut Criterion) {
    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs();
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let mut stream = MetalStream::new(&metal_device);

    let mut group = c.benchmark_group("add_rmsnorm_fusion");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    // Test different sizes: (batch_size, hidden_dim)
    for (m, n) in [(1, 4096), (4, 4096), (16, 4096), (32, 4096), (64, 4096)].iter() {
        let size_label = format!("{}x{}", m, n);

        // Benchmark fused kernel
        group.bench_with_input(
            BenchmarkId::new("fused", &size_label),
            &(*m, *n),
            |b, &(m, n)| {
                let eps = 1e-5f32;

                // Create test data
                let input: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01).collect();
                let residual: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.005).collect();
                let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();

                let input_f16: Vec<u16> = input
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let residual_f16: Vec<u16> = residual
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let weight_f16: Vec<u16> = weight
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();

                let input_buf = device.device.new_buffer_with_data(
                    input_f16.as_ptr() as *const _,
                    (input_f16.len() * 2) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let residual_buf = device.device.new_buffer_with_data(
                    residual_f16.as_ptr() as *const _,
                    (residual_f16.len() * 2) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let weight_buf = device.device.new_buffer_with_data(
                    weight_f16.as_ptr() as *const _,
                    (weight_f16.len() * 2) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let output_buf = device
                    .device
                    .new_buffer((m * n * 2) as u64, MTLResourceOptions::StorageModeShared);

                let fused_norm =
                    FusedAddRmsNorm::new(device.clone()).expect("Failed to create fused kernel");

                b.iter(|| {
                    fused_norm
                        .execute(
                            &mut stream,
                            &input_buf,
                            &residual_buf,
                            &weight_buf,
                            &output_buf,
                            None,
                            m as u32,
                            n as u32,
                            eps,
                            true,
                        )
                        .expect("Kernel execution failed");

                    stream.synchronize().expect("Synchronization failed");
                    black_box(&output_buf);
                });
            },
        );

        // Benchmark separate kernels (simulated - would need actual separate implementations)
        // For now, we'll just measure the fused version to establish baseline
    }
    group.finish();
}

fn benchmark_fused_gate_up_silu_mul(c: &mut Criterion) {
    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs();
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let mut stream = MetalStream::new(&metal_device);

    let mut group = c.benchmark_group("gate_up_silu_mul_fusion");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    // Test different sizes: (batch_size, intermediate_dim)
    for (m, n) in [
        (1, 11008),
        (4, 11008),
        (16, 11008),
        (32, 11008),
        (64, 11008),
    ]
    .iter()
    {
        let size_label = format!("{}x{}", m, n);

        // Benchmark fused kernel (separate inputs)
        group.bench_with_input(
            BenchmarkId::new("fused_separate", &size_label),
            &(*m, *n),
            |b, &(m, n)| {
                let gate: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01 - 2.0).collect();
                let up: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.005 + 1.0).collect();

                let gate_f16: Vec<u16> = gate
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let up_f16: Vec<u16> = up
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();

                let gate_buf = device.device.new_buffer_with_data(
                    gate_f16.as_ptr() as *const _,
                    (gate_f16.len() * 2) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let up_buf = device.device.new_buffer_with_data(
                    up_f16.as_ptr() as *const _,
                    (up_f16.len() * 2) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let output_buf = device
                    .device
                    .new_buffer((m * n * 2) as u64, MTLResourceOptions::StorageModeShared);

                let fused_silu =
                    FusedGateUpSiluMul::new(device.clone()).expect("Failed to create fused kernel");

                b.iter(|| {
                    fused_silu
                        .execute_separate(
                            &mut stream,
                            &gate_buf,
                            &up_buf,
                            &output_buf,
                            m as u32,
                            n as u32,
                            true,
                        )
                        .expect("Kernel execution failed");

                    stream.synchronize().expect("Synchronization failed");
                    black_box(&output_buf);
                });
            },
        );

        // Benchmark fused kernel (concatenated inputs)
        group.bench_with_input(
            BenchmarkId::new("fused_concat", &size_label),
            &(*m, *n),
            |b, &(m, n)| {
                let gate: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01 - 2.0).collect();
                let up: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.005 + 1.0).collect();

                // Concatenate: [gate | up] for each row
                let mut gate_up = Vec::with_capacity((m * n * 2) as usize);
                for i in 0..m as usize {
                    gate_up.extend_from_slice(&gate[i * n as usize..(i + 1) * n as usize]);
                    gate_up.extend_from_slice(&up[i * n as usize..(i + 1) * n as usize]);
                }

                let gate_up_f16: Vec<u16> = gate_up
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();

                let gate_up_buf = device.device.new_buffer_with_data(
                    gate_up_f16.as_ptr() as *const _,
                    (gate_up_f16.len() * 2) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let output_buf = device
                    .device
                    .new_buffer((m * n * 2) as u64, MTLResourceOptions::StorageModeShared);

                let fused_silu =
                    FusedGateUpSiluMul::new(device.clone()).expect("Failed to create fused kernel");

                b.iter(|| {
                    fused_silu
                        .execute_concat(
                            &mut stream,
                            &gate_up_buf,
                            &output_buf,
                            m as u32,
                            n as u32,
                            true,
                        )
                        .expect("Kernel execution failed");

                    stream.synchronize().expect("Synchronization failed");
                    black_box(&output_buf);
                });
            },
        );
    }
    group.finish();
}

fn benchmark_vectorization_impact(c: &mut Criterion) {
    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let profile = ferrite_metal_targets::m1_max_with_costs();
    let device = Arc::new(MetalDevice::new(metal_device.clone(), profile));
    let mut stream = MetalStream::new(&metal_device);

    let mut group = c.benchmark_group("vectorization_impact");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    // Test with dimensions divisible by 4 (vectorizable) vs not
    for (m, n, label) in [(16, 4096, "vectorizable"), (16, 4095, "non_vectorizable")].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(*m, *n),
            |b, &(m, n)| {
                let eps = 1e-5f32;

                let input: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01).collect();
                let residual: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.005).collect();
                let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();

                let input_f16: Vec<u16> = input
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let residual_f16: Vec<u16> = residual
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let weight_f16: Vec<u16> = weight
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();

                let input_buf = device.device.new_buffer_with_data(
                    input_f16.as_ptr() as *const _,
                    (input_f16.len() * 2) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let residual_buf = device.device.new_buffer_with_data(
                    residual_f16.as_ptr() as *const _,
                    (residual_f16.len() * 2) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let weight_buf = device.device.new_buffer_with_data(
                    weight_f16.as_ptr() as *const _,
                    (weight_f16.len() * 2) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let output_buf = device
                    .device
                    .new_buffer((m * n * 2) as u64, MTLResourceOptions::StorageModeShared);

                let fused_norm =
                    FusedAddRmsNorm::new(device.clone()).expect("Failed to create fused kernel");

                b.iter(|| {
                    fused_norm
                        .execute(
                            &mut stream,
                            &input_buf,
                            &residual_buf,
                            &weight_buf,
                            &output_buf,
                            None,
                            m as u32,
                            n as u32,
                            eps,
                            true,
                        )
                        .expect("Kernel execution failed");

                    stream.synchronize().expect("Synchronization failed");
                    black_box(&output_buf);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_fused_add_rmsnorm,
    benchmark_fused_gate_up_silu_mul,
    benchmark_vectorization_impact
);
criterion_main!(benches);
