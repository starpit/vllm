// SPDX-License-Identifier: Apache-2.0
// Element-wise CUDA kernels for MoE operations:
// 1. sigmoid_mul_add: out = a + sigmoid(gate) * b  (shared expert gating)
// 2. add_inplace: a += b  (simple accumulation)
//
// Uses vectorized 128-bit loads/stores via vec_utils.cuh.

#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "vec_utils.cuh"

// ---------------------------------------------------------------------------
// sigmoid_mul_add: out[i] = a[i] + sigmoid(gate_val) * b[i]
// gate is [num_tokens, 1], broadcast across hidden dimension.
// a, b, out are [num_tokens, hidden_size].
// ---------------------------------------------------------------------------

template <typename T>
__global__ void sigmoid_mul_add_kernel(
    T* __restrict__ out,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const T* __restrict__ gate,  // [num_tokens, 1]
    int hidden_size)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const int row = blockIdx.x;
    const T* a_row = a + row * hidden_size;
    const T* b_row = b + row * hidden_size;
    T* o_row = out + row * hidden_size;

    // Load gate value for this row and compute sigmoid.
    float g = static_cast<float>(gate[row]);
    float sig = 1.0f / (1.0f + expf(-g));

    // Vectorized path.
    const int num_vecs = hidden_size / VEC_SIZE;
    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        float fa[VEC_SIZE], fb[VEC_SIZE], fo[VEC_SIZE];
        unpack_vec<T>(vec_load(&a_row[i * VEC_SIZE]), fa);
        unpack_vec<T>(vec_load(&b_row[i * VEC_SIZE]), fb);

        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            fo[j] = fa[j] + sig * fb[j];
        }
        vec_store(&o_row[i * VEC_SIZE], pack_vec<T>(fo));
    }

    // Scalar tail.
    const int tail_start = num_vecs * VEC_SIZE;
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float fa = static_cast<float>(a_row[i]);
        float fb = static_cast<float>(b_row[i]);
        o_row[i] = static_cast<T>(fa + sig * fb);
    }
}

// ---------------------------------------------------------------------------
// add_inplace: a[i] += b[i]
// a, b are [num_tokens, hidden_size]. Modifies a in-place.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void add_inplace_kernel(
    T* __restrict__ a,
    const T* __restrict__ b,
    int hidden_size)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const int row = blockIdx.x;
    T* a_row = a + row * hidden_size;
    const T* b_row = b + row * hidden_size;

    const int num_vecs = hidden_size / VEC_SIZE;
    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        float fa[VEC_SIZE], fb[VEC_SIZE];
        unpack_vec<T>(vec_load(&a_row[i * VEC_SIZE]), fa);
        unpack_vec<T>(vec_load(&b_row[i * VEC_SIZE]), fb);

        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            fa[j] += fb[j];
        }
        vec_store(&a_row[i * VEC_SIZE], pack_vec<T>(fa));
    }

    const int tail_start = num_vecs * VEC_SIZE;
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float fa = static_cast<float>(a_row[i]);
        float fb = static_cast<float>(b_row[i]);
        a_row[i] = static_cast<T>(fa + fb);
    }
}

// =====================================================================
// extern "C" launchers
// =====================================================================

static constexpr int BLOCK_SIZE = 256;

// sigmoid_mul_add: out = a + sigmoid(gate) * b
extern "C" void sigmoid_mul_add_bf16(
    void* out, const void* a, const void* b, const void* gate,
    int num_tokens, int hidden_size, cudaStream_t stream)
{
    sigmoid_mul_add_kernel<__nv_bfloat16><<<num_tokens, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(out),
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<const __nv_bfloat16*>(gate),
        hidden_size);
}

extern "C" void sigmoid_mul_add_f16(
    void* out, const void* a, const void* b, const void* gate,
    int num_tokens, int hidden_size, cudaStream_t stream)
{
    sigmoid_mul_add_kernel<__half><<<num_tokens, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<__half*>(out),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<const __half*>(gate),
        hidden_size);
}

extern "C" void sigmoid_mul_add_f32(
    void* out, const void* a, const void* b, const void* gate,
    int num_tokens, int hidden_size, cudaStream_t stream)
{
    sigmoid_mul_add_kernel<float><<<num_tokens, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<float*>(out),
        reinterpret_cast<const float*>(a),
        reinterpret_cast<const float*>(b),
        reinterpret_cast<const float*>(gate),
        hidden_size);
}

// add_inplace: a += b
extern "C" void add_inplace_bf16(
    void* a, const void* b,
    int num_tokens, int hidden_size, cudaStream_t stream)
{
    add_inplace_kernel<__nv_bfloat16><<<num_tokens, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        hidden_size);
}

extern "C" void add_inplace_f16(
    void* a, const void* b,
    int num_tokens, int hidden_size, cudaStream_t stream)
{
    add_inplace_kernel<__half><<<num_tokens, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<__half*>(a),
        reinterpret_cast<const __half*>(b),
        hidden_size);
}

extern "C" void add_inplace_f32(
    void* a, const void* b,
    int num_tokens, int hidden_size, cudaStream_t stream)
{
    add_inplace_kernel<float><<<num_tokens, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<float*>(a),
        reinterpret_cast<const float*>(b),
        hidden_size);
}
