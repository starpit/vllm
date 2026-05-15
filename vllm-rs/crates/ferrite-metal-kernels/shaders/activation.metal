#include <metal_stdlib>
using namespace metal;

// ============================================================================
// SiLU (Swish) Activation: x * sigmoid(x)
// ============================================================================

kernel void silu_f16(
    device half* output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    half x = input[gid];
    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    output[gid] = x / (1.0h + exp(-x));
}

kernel void silu_bf16(
    device bfloat* output [[buffer(0)]],
    device const bfloat* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = float(input[gid]);
    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    output[gid] = bfloat(x / (1.0f + exp(-x)));
}

kernel void silu_f32(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = input[gid];
    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    output[gid] = x / (1.0f + exp(-x));
}

// Vectorized SiLU (4 elements at a time)
kernel void silu_vec4_f16(
    device half4* output [[buffer(0)]],
    device const half4* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    half4 x = input[gid];
    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    output[gid] = x / (1.0h + exp(-x));
}

// ============================================================================
// GELU Activation (Tanh Approximation)
// Note: Metal doesn't have erf(), so we use tanh approximation
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ============================================================================

kernel void gelu_f16(
    device half* output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    half x = input[gid];
    // GELU tanh approximation
    constexpr half BETA = 0.7978845608h;  // sqrt(2/pi)
    constexpr half KAPPA = 0.044715h;
    half x_cube = x * x * x;
    half inner = BETA * (x + KAPPA * x_cube);
    output[gid] = 0.5h * x * (1.0h + tanh(inner));
}

kernel void gelu_bf16(
    device bfloat* output [[buffer(0)]],
    device const bfloat* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = float(input[gid]);
    // GELU tanh approximation
    constexpr float BETA = 0.7978845608f;  // sqrt(2/pi)
    constexpr float KAPPA = 0.044715f;
    float x_cube = x * x * x;
    float inner = BETA * (x + KAPPA * x_cube);
    output[gid] = bfloat(0.5f * x * (1.0f + tanh(inner)));
}

kernel void gelu_f32(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = input[gid];
    // GELU tanh approximation
    constexpr float BETA = 0.7978845608f;  // sqrt(2/pi)
    constexpr float KAPPA = 0.044715f;
    float x_cube = x * x * x;
    float inner = BETA * (x + KAPPA * x_cube);
    output[gid] = 0.5f * x * (1.0f + tanh(inner));
}

// ============================================================================
// GELU Tanh (explicit name for compatibility)
// ============================================================================

kernel void gelu_tanh_f16(
    device half* output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    half x = input[gid];
    constexpr half BETA = 0.7978845608h;
    constexpr half KAPPA = 0.044715h;
    half x_cube = x * x * x;
    half inner = BETA * (x + KAPPA * x_cube);
    output[gid] = 0.5h * x * (1.0h + tanh(inner));
}

kernel void gelu_tanh_bf16(
    device bfloat* output [[buffer(0)]],
    device const bfloat* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = float(input[gid]);
    constexpr float BETA = 0.7978845608f;
    constexpr float KAPPA = 0.044715f;
    float x_cube = x * x * x;
    float inner = BETA * (x + KAPPA * x_cube);
    output[gid] = bfloat(0.5f * x * (1.0f + tanh(inner)));
}

kernel void gelu_tanh_f32(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = input[gid];
    constexpr float BETA = 0.7978845608f;
    constexpr float KAPPA = 0.044715f;
    float x_cube = x * x * x;
    float inner = BETA * (x + KAPPA * x_cube);
    output[gid] = 0.5f * x * (1.0f + tanh(inner));
}

// ============================================================================
// GELU Quick Approximation: x * sigmoid(1.702 * x)
// ============================================================================

kernel void gelu_quick_f16(
    device half* output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    half x = input[gid];
    // GELU quick: x * sigmoid(1.702 * x)
    constexpr half ALPHA = 1.702h;
    output[gid] = x / (1.0h + exp(-ALPHA * x));
}

kernel void gelu_quick_bf16(
    device bfloat* output [[buffer(0)]],
    device const bfloat* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = float(input[gid]);
    // GELU quick: x * sigmoid(1.702 * x)
    constexpr float ALPHA = 1.702f;
    output[gid] = bfloat(x / (1.0f + exp(-ALPHA * x)));
}

kernel void gelu_quick_f32(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = input[gid];
    // GELU quick: x * sigmoid(1.702 * x)
    constexpr float ALPHA = 1.702f;
    output[gid] = x / (1.0f + exp(-ALPHA * x));
}

// ============================================================================
// FatReLU: max(0, x) with threshold
// ============================================================================

kernel void fatrelu_f16(
    device half* output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& threshold [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    half x = input[gid];
    half t = half(threshold);
    output[gid] = (x > t) ? x : 0.0h;
}

kernel void fatrelu_bf16(
    device bfloat* output [[buffer(0)]],
    device const bfloat* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& threshold [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = float(input[gid]);
    output[gid] = bfloat((x > threshold) ? x : 0.0f);
}

kernel void fatrelu_f32(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& threshold [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = input[gid];
    output[gid] = (x > threshold) ? x : 0.0f;
}