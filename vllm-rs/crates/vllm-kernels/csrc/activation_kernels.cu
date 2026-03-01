// SPDX-License-Identifier: Apache-2.0
// Fused activation + element-wise multiply CUDA kernels for vLLM Rust.
//
// Port of csrc/activation_kernels.cu from Python vLLM (simplified).
// One thread block per row (token). Each block processes the gate and up
// projections element-wise: out[i] = act(gate[i]) * up[i].
//
// Simplifications vs Python vLLM:
// - Scalar loads instead of vectorized int4/u32x8 loads
// - No packed half2/bfloat162 arithmetic path
// - Separate gate/up pointers (matching Rust trait API) instead of
//   concatenated [2*d] input

#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// Activation functions (scalar, in float)
// ---------------------------------------------------------------------------

// SiLU: x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// GELU (exact / erf): x * 0.5 * (1 + erf(x / sqrt(2)))
__device__ __forceinline__ float gelu_erf(float x) {
    constexpr float ALPHA = 0.7071067811865476f; // 1/sqrt(2)
    return x * 0.5f * (1.0f + erff(x * ALPHA));
}

// GELU (tanh approx): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu_tanh(float x) {
    constexpr float BETA = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float KAPPA = 0.044715f;
    float x3 = x * x * x;
    float inner = BETA * (x + KAPPA * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// ---------------------------------------------------------------------------
// Fused act-and-mul kernel: out[i] = act(gate[i]) * up[i]
// ---------------------------------------------------------------------------

template <float (*ACT_FN)(float), typename T>
__global__ void act_and_mul_kernel(
    T* __restrict__ out,
    const T* __restrict__ gate,
    const T* __restrict__ up,
    int d)
{
    const int row = blockIdx.x;
    const T* g = gate + row * d;
    const T* u = up + row * d;
    T* o = out + row * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float gv = ACT_FN(static_cast<float>(g[i]));
        float uv = static_cast<float>(u[i]);
        o[i] = static_cast<T>(gv * uv);
    }
}

// ---------------------------------------------------------------------------
// C entry points: silu_and_mul
// ---------------------------------------------------------------------------

extern "C" {

void silu_and_mul_f32(
    float* out, const float* gate, const float* up,
    int num_tokens, int d)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<silu, float><<<num_tokens, threads>>>(out, gate, up, d);
}

void silu_and_mul_f16(
    __half* out, const __half* gate, const __half* up,
    int num_tokens, int d)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<silu, __half><<<num_tokens, threads>>>(out, gate, up, d);
}

void silu_and_mul_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
    int num_tokens, int d)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<silu, __nv_bfloat16><<<num_tokens, threads>>>(out, gate, up, d);
}

// ---------------------------------------------------------------------------
// C entry points: gelu_and_mul (tanh approximation)
// ---------------------------------------------------------------------------

void gelu_and_mul_f32(
    float* out, const float* gate, const float* up,
    int num_tokens, int d)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_tanh, float><<<num_tokens, threads>>>(out, gate, up, d);
}

void gelu_and_mul_f16(
    __half* out, const __half* gate, const __half* up,
    int num_tokens, int d)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_tanh, __half><<<num_tokens, threads>>>(out, gate, up, d);
}

void gelu_and_mul_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
    int num_tokens, int d)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_tanh, __nv_bfloat16><<<num_tokens, threads>>>(out, gate, up, d);
}

// ---------------------------------------------------------------------------
// C entry points: gelu_new_and_mul (exact / erf)
// ---------------------------------------------------------------------------

void gelu_new_and_mul_f32(
    float* out, const float* gate, const float* up,
    int num_tokens, int d)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_erf, float><<<num_tokens, threads>>>(out, gate, up, d);
}

void gelu_new_and_mul_f16(
    __half* out, const __half* gate, const __half* up,
    int num_tokens, int d)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_erf, __half><<<num_tokens, threads>>>(out, gate, up, d);
}

void gelu_new_and_mul_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
    int num_tokens, int d)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_erf, __nv_bfloat16><<<num_tokens, threads>>>(out, gate, up, d);
}

} // extern "C"
