// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

/// RMSNorm kernel: y = x * weight / sqrt(mean(x^2) + eps)
///
/// Grid: (M, 1, 1) where M = batch_size
/// Threadgroup: (min(N, 1024), 1, 1) where N = hidden_size
kernel void rmsnorm_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    // Compute mean of squares using threadgroup reduction
    threadgroup float shared_sum[1024];
    
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float val = float(input[gid * N + i]);
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction in shared memory
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Broadcast RMS to all threads
    float rms = sqrt(shared_sum[0] / float(N) + eps);
    
    // Normalize and scale
    for (uint i = tid; i < N; i += tg_size) {
        float val = float(input[gid * N + i]);
        float w = float(weight[i]);
        output[gid * N + i] = half((val / rms) * w);
    }
}

/// BF16 variant (uses float16 as Metal doesn't have native bfloat16)
kernel void rmsnorm_bf16(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    threadgroup float shared_sum[1024];
    
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float val = input[gid * N + i];
        local_sum += val * val;
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
        float val = input[gid * N + i];
        float w = weight[i];
        output[gid * N + i] = (val / rms) * w;
    }
}
