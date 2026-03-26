// SPDX-License-Identifier: Apache-2.0
// Fused activation + element-wise multiply CUDA kernels for vLLM Rust.
//
// Port of csrc/activation_kernels.cu from Python vLLM.
// One thread block per row (token). Each block processes the gate and up
// projections element-wise: out[i] = act(gate[i]) * up[i].
//
// Uses vectorized 128-bit loads/stores for throughput.

#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "vec_utils.cuh"

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
// Fused act-and-mul kernel (vectorized): out[i] = act(gate[i]) * up[i]
// ---------------------------------------------------------------------------

template <float (*ACT_FN)(float), typename T>
__global__ void act_and_mul_kernel(
    T* __restrict__ out,
    const T* __restrict__ gate,
    const T* __restrict__ up,
    int d)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const int row = blockIdx.x;
    const T* g = gate + row * d;
    const T* u = up + row * d;
    T* o = out + row * d;

    const int num_vecs = d / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    // Vectorized loop.
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float gbuf[VEC_SIZE], ubuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&g[vi * VEC_SIZE]), gbuf);
        unpack_vec<T>(vec_load(&u[vi * VEC_SIZE]), ubuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = ACT_FN(gbuf[j]) * ubuf[j];
        }
        vec_store(&o[vi * VEC_SIZE], pack_vec<T>(obuf));
    }

    // Scalar tail.
    for (int i = tail_start + threadIdx.x; i < d; i += blockDim.x) {
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
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<silu, float><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

void silu_and_mul_f16(
    __half* out, const __half* gate, const __half* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<silu, __half><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

void silu_and_mul_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<silu, __nv_bfloat16><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

// ---------------------------------------------------------------------------
// C entry points: gelu_and_mul (tanh approximation)
// ---------------------------------------------------------------------------

void gelu_and_mul_f32(
    float* out, const float* gate, const float* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_tanh, float><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

void gelu_and_mul_f16(
    __half* out, const __half* gate, const __half* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_tanh, __half><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

void gelu_and_mul_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_tanh, __nv_bfloat16><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

// ---------------------------------------------------------------------------
// C entry points: gelu_new_and_mul (exact / erf)
// ---------------------------------------------------------------------------

void gelu_new_and_mul_f32(
    float* out, const float* gate, const float* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_erf, float><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

void gelu_new_and_mul_f16(
    __half* out, const __half* gate, const __half* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_erf, __half><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

void gelu_new_and_mul_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_erf, __nv_bfloat16><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

} // extern "C"

// ---------------------------------------------------------------------------
// Fused act-and-mul from combined gate_up tensor [num_tokens, 2*d]
// Avoids two contiguous copy kernels + allocations per layer.
// ---------------------------------------------------------------------------

template <float (*ACT_FN)(float), typename T>
__global__ void act_and_mul_fused_kernel(
    T* __restrict__ out,
    const T* __restrict__ gate_up,
    int d)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const int row = blockIdx.x;
    const int stride = 2 * d;
    const T* g = gate_up + row * stride;       // gate half
    const T* u = gate_up + row * stride + d;   // up half
    T* o = out + row * d;

    const int num_vecs = d / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float gbuf[VEC_SIZE], ubuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&g[vi * VEC_SIZE]), gbuf);
        unpack_vec<T>(vec_load(&u[vi * VEC_SIZE]), ubuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = ACT_FN(gbuf[j]) * ubuf[j];
        }
        vec_store(&o[vi * VEC_SIZE], pack_vec<T>(obuf));
    }

    for (int i = tail_start + threadIdx.x; i < d; i += blockDim.x) {
        float gv = ACT_FN(static_cast<float>(g[i]));
        float uv = static_cast<float>(u[i]);
        o[i] = static_cast<T>(gv * uv);
    }
}

extern "C" {

// C entry points: silu_and_mul_fused (takes [num_tokens, 2*d])
void silu_and_mul_fused_f32(
    float* out, const float* gate_up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_fused_kernel<silu, float><<<num_tokens, threads, 0, stream>>>(out, gate_up, d);
}

void silu_and_mul_fused_f16(
    __half* out, const __half* gate_up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_fused_kernel<silu, __half><<<num_tokens, threads, 0, stream>>>(out, gate_up, d);
}

void silu_and_mul_fused_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* gate_up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_fused_kernel<silu, __nv_bfloat16><<<num_tokens, threads, 0, stream>>>(out, gate_up, d);
}

// C entry points: gelu_and_mul_fused (takes [num_tokens, 2*d])
void gelu_and_mul_fused_f32(
    float* out, const float* gate_up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_fused_kernel<gelu_tanh, float><<<num_tokens, threads, 0, stream>>>(out, gate_up, d);
}

void gelu_and_mul_fused_f16(
    __half* out, const __half* gate_up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_fused_kernel<gelu_tanh, __half><<<num_tokens, threads, 0, stream>>>(out, gate_up, d);
}

void gelu_and_mul_fused_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* gate_up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_fused_kernel<gelu_tanh, __nv_bfloat16><<<num_tokens, threads, 0, stream>>>(out, gate_up, d);
}

} // extern "C"

// ---------------------------------------------------------------------------
// Tanh softcap inplace: x[i] = cap * tanh(x[i] / cap)
// Used for Gemma 2 final logit softcapping.
// x: [n] total elements, cap: scalar float.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void tanh_softcap_inplace_kernel(
    T* __restrict__ x,
    float inv_cap,
    float cap,
    int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float v = static_cast<float>(x[i]) * inv_cap;
        x[i] = static_cast<T>(cap * tanhf(v));
    }
}

extern "C" {

void tanh_softcap_inplace_f32(
    float* x, float cap, int n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    tanh_softcap_inplace_kernel<float><<<blocks, threads, 0, stream>>>(
        x, 1.0f / cap, cap, n);
}

void tanh_softcap_inplace_f16(
    __half* x, float cap, int n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    tanh_softcap_inplace_kernel<__half><<<blocks, threads, 0, stream>>>(
        x, 1.0f / cap, cap, n);
}

void tanh_softcap_inplace_bf16(
    __nv_bfloat16* x, float cap, int n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    tanh_softcap_inplace_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        x, 1.0f / cap, cap, n);
}

} // extern "C"

// ---------------------------------------------------------------------------
// Broadcast multiply inplace: x[row, col] *= scale[col]
// x: [num_rows, d], scale: [d] — both same dtype.
// One block per row.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void broadcast_mul_inplace_kernel(
    T* __restrict__ x,
    const T* __restrict__ scale,
    int d)
{
    const int row = blockIdx.x;
    T* xr = x + row * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float xv = (float)xr[i];
        float sv = (float)scale[i];
        xr[i] = (T)(xv * sv);
    }
}

extern "C" {

void broadcast_mul_inplace_f32(
    float* x, const float* scale,
    int num_rows, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    broadcast_mul_inplace_kernel<float><<<num_rows, threads, 0, stream>>>(x, scale, d);
}

void broadcast_mul_inplace_f16(
    __half* x, const __half* scale,
    int num_rows, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    broadcast_mul_inplace_kernel<__half><<<num_rows, threads, 0, stream>>>(x, scale, d);
}

void broadcast_mul_inplace_bf16(
    __nv_bfloat16* x, const __nv_bfloat16* scale,
    int num_rows, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    broadcast_mul_inplace_kernel<__nv_bfloat16><<<num_rows, threads, 0, stream>>>(x, scale, d);
}

} // extern "C"
