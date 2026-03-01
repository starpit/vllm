// SPDX-License-Identifier: Apache-2.0
// Fused RMS normalization CUDA kernel for vLLM Rust.
//
// Port of csrc/layernorm_kernels.cu from Python vLLM (simplified).
// One thread block per row (token). Each block cooperatively computes
// the variance, then applies weight * x * rsqrt(variance + eps).

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

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

// ---- RMS Norm (out-of-place) ----
// out[row, :] = weight * input[row, :] * rsqrt(mean(input[row, :]^2) + eps)

template <typename T>
__global__ void rms_norm_kernel(
    T* __restrict__ out,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    float epsilon,
    int hidden_size)
{
    const int row = blockIdx.x;
    const T* x = input + row * hidden_size;
    T* y = out + row * hidden_size;

    // Compute sum of squares.
    float ss = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(x[i]);
        ss += v * v;
    }
    ss = block_reduce_sum(ss);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(ss / hidden_size + epsilon);
    }
    __syncthreads();

    // Apply normalization.
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(x[i]) * s_inv_rms;
        y[i] = static_cast<T>(v * static_cast<float>(weight[i]));
    }
}

// ---- Fused Add + RMS Norm (in-place on input, residual updated) ----
// residual += input; input = weight * residual * rsqrt(mean(residual^2) + eps)

template <typename T>
__global__ void fused_add_rms_norm_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    float epsilon,
    int hidden_size)
{
    const int row = blockIdx.x;
    T* inp = input + row * hidden_size;
    T* res = residual + row * hidden_size;

    // Fused add + variance computation.
    float ss = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
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

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(res[i]) * s_inv_rms;
        inp[i] = static_cast<T>(v * static_cast<float>(weight[i]));
    }
}

// ---- C entry points ----

extern "C" {

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

} // extern "C"
