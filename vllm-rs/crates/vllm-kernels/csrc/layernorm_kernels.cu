// SPDX-License-Identifier: Apache-2.0
// Fused RMS normalization CUDA kernel for vLLM Rust.
//
// Port of csrc/layernorm_kernels.cu from Python vLLM.
// One thread block per row (token). Each block cooperatively computes
// the variance, then applies weight * x * rsqrt(variance + eps).
//
// Uses vectorized 128-bit loads/stores for throughput.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "vec_utils.cuh"

// Warp-level reduce sum via shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduce sum via shared memory.
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32]; // one per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

// ---- RMS Norm (out-of-place, vectorized) ----
// out[row, :] = weight * input[row, :] * rsqrt(mean(input[row, :]^2) + eps)

template <typename T>
__global__ void rms_norm_kernel(
    T* __restrict__ out,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    float epsilon,
    int hidden_size)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;
    using VT = typename VecType<T>::Type;

    const int row = blockIdx.x;
    const T* x = input + row * hidden_size;
    T* y = out + row * hidden_size;

    const int num_vecs = hidden_size / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    // Pass 1: Compute sum of squares with vectorized loads.
    float ss = 0.0f;
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float buf[VEC_SIZE];
        unpack_vec<T>(vec_load(&x[vi * VEC_SIZE]), buf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            ss += buf[j] * buf[j];
        }
    }
    // Scalar tail.
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(x[i]);
        ss += v * v;
    }
    ss = block_reduce_sum(ss);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(ss / hidden_size + epsilon);
    }
    __syncthreads();

    // Pass 2: Apply normalization with vectorized loads/stores.
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float xbuf[VEC_SIZE], wbuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&x[vi * VEC_SIZE]), xbuf);
        unpack_vec<T>(vec_load(&weight[vi * VEC_SIZE]), wbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = xbuf[j] * s_inv_rms * wbuf[j];
        }
        vec_store(&y[vi * VEC_SIZE], pack_vec<T>(obuf));
    }
    // Scalar tail.
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(x[i]) * s_inv_rms;
        y[i] = static_cast<T>(v * static_cast<float>(weight[i]));
    }
}

// ---- Fused Add + RMS Norm (in-place on input, residual updated, vectorized) ----
// residual += input; input = weight * residual * rsqrt(mean(residual^2) + eps)

template <typename T>
__global__ void fused_add_rms_norm_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    float epsilon,
    int hidden_size)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;
    using VT = typename VecType<T>::Type;

    const int row = blockIdx.x;
    T* inp = input + row * hidden_size;
    T* res = residual + row * hidden_size;

    const int num_vecs = hidden_size / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    // Pass 1: Fused add + variance computation with vectorized loads/stores.
    float ss = 0.0f;
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float ibuf[VEC_SIZE], rbuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&inp[vi * VEC_SIZE]), ibuf);
        unpack_vec<T>(vec_load(&res[vi * VEC_SIZE]), rbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            rbuf[j] += ibuf[j];
            ss += rbuf[j] * rbuf[j];
        }
        vec_store(&res[vi * VEC_SIZE], pack_vec<T>(rbuf));
    }
    // Scalar tail.
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float r = static_cast<float>(res[i]) + static_cast<float>(inp[i]);
        res[i] = static_cast<T>(r);
        ss += r * r;
    }
    ss = block_reduce_sum(ss);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(ss / hidden_size + epsilon);
    }
    __syncthreads();

    // Pass 2: Normalize with vectorized loads/stores.
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float rbuf[VEC_SIZE], wbuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&res[vi * VEC_SIZE]), rbuf);
        unpack_vec<T>(vec_load(&weight[vi * VEC_SIZE]), wbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = rbuf[j] * s_inv_rms * wbuf[j];
        }
        vec_store(&inp[vi * VEC_SIZE], pack_vec<T>(obuf));
    }
    // Scalar tail.
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(res[i]) * s_inv_rms;
        inp[i] = static_cast<T>(v * static_cast<float>(weight[i]));
    }
}

// ---- C entry points ----

extern "C" {

// ---- RMS Norm entry points ----

void rms_norm_f32(
    float* out, const float* input, const float* weight,
    float epsilon, int num_tokens, int hidden_size)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    rms_norm_kernel<float><<<num_tokens, threads>>>(
        out, input, weight, epsilon, hidden_size);
}

void rms_norm_f16(
    __half* out, const __half* input, const __half* weight,
    float epsilon, int num_tokens, int hidden_size)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    rms_norm_kernel<__half><<<num_tokens, threads>>>(
        out, input, weight, epsilon, hidden_size);
}

void rms_norm_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* input, const __nv_bfloat16* weight,
    float epsilon, int num_tokens, int hidden_size)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    rms_norm_kernel<__nv_bfloat16><<<num_tokens, threads>>>(
        out, input, weight, epsilon, hidden_size);
}

// ---- Fused Add + RMS Norm entry points ----

void fused_add_rms_norm_f32(
    float* input, float* residual, const float* weight,
    float epsilon, int num_tokens, int hidden_size)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    fused_add_rms_norm_kernel<float><<<num_tokens, threads>>>(
        input, residual, weight, epsilon, hidden_size);
}

void fused_add_rms_norm_f16(
    __half* input, __half* residual, const __half* weight,
    float epsilon, int num_tokens, int hidden_size)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    fused_add_rms_norm_kernel<__half><<<num_tokens, threads>>>(
        input, residual, weight, epsilon, hidden_size);
}

void fused_add_rms_norm_bf16(
    __nv_bfloat16* input, __nv_bfloat16* residual, const __nv_bfloat16* weight,
    float epsilon, int num_tokens, int hidden_size)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    fused_add_rms_norm_kernel<__nv_bfloat16><<<num_tokens, threads>>>(
        input, residual, weight, epsilon, hidden_size);
}

} // extern "C"
