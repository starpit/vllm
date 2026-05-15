// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Add: out = a + b
// ============================================================================

kernel void add_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] + b[gid];
}

kernel void add_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] + b[gid];
}

// ============================================================================
// Mul: out = a * b
// ============================================================================

kernel void mul_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] * b[gid];
}

kernel void mul_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] * b[gid];
}

// ============================================================================
// Sub: out = a - b
// ============================================================================

kernel void sub_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] - b[gid];
}

kernel void sub_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] - b[gid];
}

// ============================================================================
// ScalarMul: out = scalar * input
// ============================================================================

kernel void scalar_mul_f16(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = half(scalar) * input[gid];
}

kernel void scalar_mul_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = bfloat(scalar) * input[gid];
}

// ============================================================================
// BiasAdd: out = input + bias (broadcast bias across last dimension)
// ============================================================================

kernel void bias_add_f16(
    device const half* input [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& num_cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint col = gid % num_cols;
    out[gid] = input[gid] + bias[col];
}

kernel void bias_add_bf16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* bias [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& num_cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint col = gid % num_cols;
    out[gid] = input[gid] + bias[col];
}

// ── Specialized BiasAdd (function-constant num_cols) ────────────────────────
//
// Pipeline-time num_cols binding so the Metal driver can constant-fold the
// modulus on dispatches of fixed projection width (Q/K/V each ship a separate
// specialized pipeline at lowering time — `function_constant(0)` is the only
// axis). Bound by `Instruction::MetalBiasAdd` via the
// `KernelId::BiasAdd` lowering arm.
constant uint BIAS_ADD_NUM_COLS [[function_constant(0)]];

kernel void bias_add_f16_specialized(
    device const half* input [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = input[gid] + bias[gid % BIAS_ADD_NUM_COLS];
}

kernel void bias_add_bf16_specialized(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* bias [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = input[gid] + bias[gid % BIAS_ADD_NUM_COLS];
}

// ============================================================================
// TanhSoftCap: out = cap * tanh(input / cap)
// Used in Gemma2 for attention logit capping
// ============================================================================

kernel void tanh_soft_cap_f16(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant float& cap [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = float(input[gid]);
    float result = cap * tanh(x / cap);
    out[gid] = half(result);
}

kernel void tanh_soft_cap_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant float& cap [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = float(input[gid]);
    float result = cap * tanh(x / cap);
    out[gid] = bfloat(result);
}
