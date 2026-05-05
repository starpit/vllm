// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

/// Fused Add + RMSNorm kernel: y = rmsnorm(x + residual, weight, eps)
///
/// This fusion eliminates one memory round-trip by computing the residual add
/// and normalization in a single pass. Critical for memory-bound Apple Silicon.
///
/// Pattern: x' = rmsnorm(x + residual, w, eps)
/// Used in: Pre-norm and post-norm residual patterns in transformer layers
///
/// Grid: (M, 1, 1) where M = batch_size
/// Threadgroup: (min(N, 1024), 1, 1) where N = hidden_size
///
/// Outputs:
/// - output: normalized result [M, N]
/// - residual_out: (x + residual) for next layer's residual [M, N]
kernel void fused_add_rmsnorm_f16(
    device const half* input [[buffer(0)]],
    device const half* residual [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device half* output [[buffer(3)]],
    device half* residual_out [[buffer(4)]],  // Optional: pass null if not needed
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    // Pass 1: Compute x + residual and sum of squares
    threadgroup float shared_sum[1024];
    
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float x_val = float(input[gid * N + i]);
        float r_val = float(residual[gid * N + i]);
        float sum_val = x_val + r_val;
        
        // Write out residual sum if output buffer provided
        if (residual_out != nullptr) {
            residual_out[gid * N + i] = half(sum_val);
        }
        
        local_sum += sum_val * sum_val;
    }
    shared_sum[tid] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction for sum of squares
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Compute RMS normalization factor
    float rms = sqrt(shared_sum[0] / float(N) + eps);
    
    // Pass 2: Normalize and scale with weight
    for (uint i = tid; i < N; i += tg_size) {
        float x_val = float(input[gid * N + i]);
        float r_val = float(residual[gid * N + i]);
        float sum_val = x_val + r_val;
        float w = float(weight[i]);
        output[gid * N + i] = half((sum_val / rms) * w);
    }
}

/// BF16 variant (uses float as Metal doesn't have native bfloat16)
kernel void fused_add_rmsnorm_bf16(
    device const float* input [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* residual_out [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    threadgroup float shared_sum[1024];
    
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float x_val = input[gid * N + i];
        float r_val = residual[gid * N + i];
        float sum_val = x_val + r_val;
        
        if (residual_out != nullptr) {
            residual_out[gid * N + i] = sum_val;
        }
        
        local_sum += sum_val * sum_val;
    }
    shared_sum[tid] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float rms = sqrt(shared_sum[0] / float(N) + eps);
    
    for (uint i = tid; i < N; i += tg_size) {
        float x_val = input[gid * N + i];
        float r_val = residual[gid * N + i];
        float sum_val = x_val + r_val;
        float w = weight[i];
        output[gid * N + i] = (sum_val / rms) * w;
    }
}

/// Optimized variant with vectorized loads (half4) for better memory bandwidth
/// Requires N to be multiple of 4
kernel void fused_add_rmsnorm_f16_vec4(
    device const half4* input [[buffer(0)]],
    device const half4* residual [[buffer(1)]],
    device const half4* weight [[buffer(2)]],
    device half4* output [[buffer(3)]],
    device half4* residual_out [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N_div4 [[buffer(6)]],  // N / 4
    constant float& eps [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    threadgroup float shared_sum[1024];
    
    float local_sum = 0.0f;
    for (uint i = tid; i < N_div4; i += tg_size) {
        half4 x_val = input[gid * N_div4 + i];
        half4 r_val = residual[gid * N_div4 + i];
        half4 sum_val = x_val + r_val;
        
        if (residual_out != nullptr) {
            residual_out[gid * N_div4 + i] = sum_val;
        }
        
        // Accumulate sum of squares for all 4 elements
        float4 sum_f = float4(sum_val);
        local_sum += dot(sum_f, sum_f);
    }
    shared_sum[tid] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float rms = sqrt(shared_sum[0] / float(N_div4 * 4) + eps);
    
    for (uint i = tid; i < N_div4; i += tg_size) {
        half4 x_val = input[gid * N_div4 + i];
        half4 r_val = residual[gid * N_div4 + i];
        half4 sum_val = x_val + r_val;
        half4 w = weight[i];
        output[gid * N_div4 + i] = half4((float4(sum_val) / rms) * float4(w));
    }
}
