// SPDX-License-Identifier: Apache-2.0
//! RMSNorm cost sweep for Metal.
//!
//! Sweeps over typical hidden_size values (2048-8192) and sequence lengths
//! (1-8192 tokens) to calibrate the cost model for RMSNorm operations.

use crate::util;
use metal::MTLSize;

/// Run RMSNorm cost sweep and emit CSV rows.
pub fn run(launch_overhead_us: f64) {
    eprintln!("Starting RMSNorm sweep...");

    // Typical hidden_size values from popular models:
    // - Llama 7B: 4096
    // - Llama 13B: 5120
    // - Llama 70B: 8192
    // - Qwen 7B: 4096
    // - Mistral 7B: 4096
    let hidden_sizes = vec![2048, 3072, 4096, 5120, 6144, 7168, 8192];

    // Sequence lengths: powers of 2 from 1 to 8192
    let seq_lens = vec![
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    ];

    // Sweep F16 variant
    for &hidden_size in &hidden_sizes {
        for &seq_len in &seq_lens {
            let cost_us = benchmark_rmsnorm_f16(seq_len, hidden_size, launch_overhead_us);
            // CSV format: kernel,M,N,K,cost_us
            // For RMSNorm: M=seq_len, N=hidden_size, K=0 (unused)
            println!("metal_rmsnorm_f16,{seq_len},{hidden_size},0,{cost_us:.2}");
        }
    }

    // Sweep BF16 variant
    for &hidden_size in &hidden_sizes {
        for &seq_len in &seq_lens {
            let cost_us = benchmark_rmsnorm_bf16(seq_len, hidden_size, launch_overhead_us);
            println!("metal_rmsnorm_bf16,{seq_len},{hidden_size},0,{cost_us:.2}");
        }
    }

    eprintln!("RMSNorm sweep complete");
}

/// Benchmark RMSNorm F16 kernel.
fn benchmark_rmsnorm_f16(seq_len: usize, hidden_size: usize, launch_overhead_us: f64) -> f64 {
    let device = util::device();
    let queue = device.new_command_queue();

    // Allocate buffers
    let input_size = seq_len * hidden_size * 2; // F16 = 2 bytes
    let output_size = input_size;
    let weight_size = hidden_size * 2;

    let input_buf = util::create_buffer(input_size);
    let output_buf = util::create_buffer(output_size);
    let weight_buf = util::create_buffer(weight_size);

    // Load shader (placeholder - actual shader loading will be implemented)
    // For now, we'll use a simple compute shader that does minimal work
    let library_source = r#"
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void rmsnorm_f16(
            device const half* input [[buffer(0)]],
            device const half* weight [[buffer(1)]],
            device half* output [[buffer(2)]],
            constant uint& seq_len [[buffer(3)]],
            constant uint& hidden_size [[buffer(4)]],
            constant float& eps [[buffer(5)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= seq_len) return;
            
            // Compute RMS
            float sum_sq = 0.0f;
            uint offset = tid * hidden_size;
            for (uint i = 0; i < hidden_size; i++) {
                float val = float(input[offset + i]);
                sum_sq += val * val;
            }
            float rms = sqrt(sum_sq / float(hidden_size) + eps);
            float inv_rms = 1.0f / rms;
            
            // Normalize and scale
            for (uint i = 0; i < hidden_size; i++) {
                float val = float(input[offset + i]);
                float w = float(weight[i]);
                output[offset + i] = half(val * inv_rms * w);
            }
        }
    "#;

    let library = device
        .new_library_with_source(library_source, &metal::CompileOptions::new())
        .expect("Failed to compile Metal shader");
    let kernel = library
        .get_function("rmsnorm_f16", None)
        .expect("Failed to get kernel function");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("Failed to create pipeline state");

    // Warm-up
    for _ in 0..3 {
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input_buf), 0);
        encoder.set_buffer(1, Some(&weight_buf), 0);
        encoder.set_buffer(2, Some(&output_buf), 0);

        let threadgroup_size = MTLSize::new(256, 1, 1);
        let grid_size = MTLSize::new((((seq_len + 255) / 256) * 256) as u64, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    // Measure
    let iterations = 10;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input_buf), 0);
        encoder.set_buffer(1, Some(&weight_buf), 0);
        encoder.set_buffer(2, Some(&output_buf), 0);

        let threadgroup_size = MTLSize::new(256, 1, 1);
        let grid_size = MTLSize::new((((seq_len + 255) / 256) * 256) as u64, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }
    let elapsed = start.elapsed();

    let total_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let compute_us = total_us - launch_overhead_us;
    compute_us.max(0.0)
}

/// Benchmark RMSNorm BF16 kernel.
fn benchmark_rmsnorm_bf16(seq_len: usize, hidden_size: usize, launch_overhead_us: f64) -> f64 {
    // BF16 implementation - similar to F16 but with bfloat16 type
    // For now, use same implementation as F16 (Metal doesn't have native BF16 support,
    // so we'd need to implement it via bit manipulation)
    benchmark_rmsnorm_f16(seq_len, hidden_size, launch_overhead_us)
}
