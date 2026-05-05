// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

/// Fused Gate-Up-SiLU-Mul kernel for SwiGLU activation
///
/// Pattern: silu(gate_proj(x)) * up_proj(x)
/// Where: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
///
/// This fusion eliminates memory round-trips by computing the activation
/// in a single pass. Critical for memory-bound MLP layers on Apple Silicon.
///
/// Two variants:
/// 1. Separate GEMMs: Takes gate_out and up_out as inputs
/// 2. Fused GEMM: Takes concatenated gate_up output [B, 2*I] as input
///
/// Grid: (M, 1, 1) where M = batch_size
/// Threadgroup: (min(N, 1024), 1, 1) where N = intermediate_size

/// SiLU activation: x * sigmoid(x)
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

/// Variant 1: Separate gate and up outputs
/// Input: gate_out [M, N], up_out [M, N]
/// Output: silu(gate_out) * up_out [M, N]
kernel void fused_gate_up_silu_mul_f16(
    device const half* gate_out [[buffer(0)]],
    device const half* up_out [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N; i += tg_size) {
        float gate = float(gate_out[gid * N + i]);
        float up = float(up_out[gid * N + i]);
        output[gid * N + i] = half(silu(gate) * up);
    }
}

/// Variant 2: Fused gate_up output (concatenated in column dimension)
/// Input: gate_up [M, 2*N] where first N columns are gate, next N are up
/// Output: silu(gate) * up [M, N]
kernel void fused_gate_up_silu_mul_concat_f16(
    device const half* gate_up [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],  // intermediate_size (not 2*N)
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N; i += tg_size) {
        float gate = float(gate_up[gid * (2 * N) + i]);
        float up = float(gate_up[gid * (2 * N) + N + i]);
        output[gid * N + i] = half(silu(gate) * up);
    }
}

/// BF16 variant - separate outputs
kernel void fused_gate_up_silu_mul_bf16(
    device const float* gate_out [[buffer(0)]],
    device const float* up_out [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N; i += tg_size) {
        float gate = gate_out[gid * N + i];
        float up = up_out[gid * N + i];
        output[gid * N + i] = silu(gate) * up;
    }
}

/// BF16 variant - concatenated input
kernel void fused_gate_up_silu_mul_concat_bf16(
    device const float* gate_up [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N; i += tg_size) {
        float gate = gate_up[gid * (2 * N) + i];
        float up = gate_up[gid * (2 * N) + N + i];
        output[gid * N + i] = silu(gate) * up;
    }
}

/// Vectorized variant (half4) for better memory bandwidth
/// Requires N to be multiple of 4
kernel void fused_gate_up_silu_mul_f16_vec4(
    device const half4* gate_out [[buffer(0)]],
    device const half4* up_out [[buffer(1)]],
    device half4* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N_div4 [[buffer(4)]],  // N / 4
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N_div4; i += tg_size) {
        half4 gate = gate_out[gid * N_div4 + i];
        half4 up = up_out[gid * N_div4 + i];
        
        // Apply SiLU element-wise
        float4 gate_f = float4(gate);
        float4 up_f = float4(up);
        float4 result;
        result.x = silu(gate_f.x) * up_f.x;
        result.y = silu(gate_f.y) * up_f.y;
        result.z = silu(gate_f.z) * up_f.z;
        result.w = silu(gate_f.w) * up_f.w;
        
        output[gid * N_div4 + i] = half4(result);
    }
}

/// Vectorized concatenated variant
kernel void fused_gate_up_silu_mul_concat_f16_vec4(
    device const half4* gate_up [[buffer(0)]],
    device half4* output [[buffer(1)]],
    constant uint& M [[buffer(2)]],
    constant uint& N_div4 [[buffer(3)]],  // N / 4
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N_div4; i += tg_size) {
        half4 gate = gate_up[gid * (2 * N_div4) + i];
        half4 up = gate_up[gid * (2 * N_div4) + N_div4 + i];
        
        float4 gate_f = float4(gate);
        float4 up_f = float4(up);
        float4 result;
        result.x = silu(gate_f.x) * up_f.x;
        result.y = silu(gate_f.y) * up_f.y;
        result.z = silu(gate_f.z) * up_f.z;
        result.w = silu(gate_f.w) * up_f.w;
        
        output[gid * N_div4 + i] = half4(result);
    }
}

/// GELU variant for Gemma2/3 models
/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
inline float gelu_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanh(inner));
}

/// Fused Gate-Up-GELU-Mul for Gemma models
kernel void fused_gate_up_gelu_mul_f16(
    device const half* gate_out [[buffer(0)]],
    device const half* up_out [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N; i += tg_size) {
        float gate = float(gate_out[gid * N + i]);
        float up = float(up_out[gid * N + i]);
        output[gid * N + i] = half(gelu_approx(gate) * up);
    }
}

// Note: Metal does not have erf() function, so exact GELU is not available
// Use approximate GELU instead (gelu_approx above)
