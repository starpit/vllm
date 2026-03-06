// SPDX-License-Identifier: Apache-2.0
// Embedding gather kernel: out[i] = weight[input_ids[i]]
//
// Simple: one thread per (token, dim) element.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

template<typename T>
__global__ void embedding_gather_kernel(
    T* __restrict__ out,           // [num_tokens, hidden_size]
    const T* __restrict__ weight,  // [vocab_size, hidden_size]
    const uint32_t* __restrict__ ids,  // [num_tokens]
    int hidden_size,
    int num_tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * hidden_size;
    if (idx >= total) return;

    int token = idx / hidden_size;
    int dim = idx % hidden_size;
    uint32_t id = ids[token];

    out[token * hidden_size + dim] = weight[id * hidden_size + dim];
}

// Vectorized version: 128-bit loads (8 x f16/bf16 or 4 x f32)
template<typename T, int VEC_SIZE>
__global__ void embedding_gather_vec_kernel(
    T* __restrict__ out,
    const T* __restrict__ weight,
    const uint32_t* __restrict__ ids,
    int hidden_size,
    int num_tokens
) {
    using VecT = typename std::conditional<sizeof(T) * VEC_SIZE == 16, int4,
                  typename std::conditional<sizeof(T) * VEC_SIZE == 8, int2, int>::type>::type;

    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_hidden = hidden_size / VEC_SIZE;
    int total_vecs = num_tokens * vec_hidden;
    if (vec_idx >= total_vecs) return;

    int token = vec_idx / vec_hidden;
    int dim_vec = vec_idx % vec_hidden;
    uint32_t id = ids[token];

    const VecT* w_vec = reinterpret_cast<const VecT*>(weight + id * hidden_size);
    VecT* o_vec = reinterpret_cast<VecT*>(out + token * hidden_size);
    o_vec[dim_vec] = w_vec[dim_vec];
}

#define LAUNCH_GATHER(T, VEC)                                                  \
    do {                                                                        \
        int vec_hidden = hidden_size / (VEC);                                   \
        int total = num_tokens * vec_hidden;                                    \
        int threads = 256;                                                      \
        int blocks = (total + threads - 1) / threads;                           \
        if (hidden_size % (VEC) == 0) {                                         \
            embedding_gather_vec_kernel<T, VEC><<<blocks, threads, 0, stream>>>( \
                (T*)out, (const T*)weight, ids, hidden_size, num_tokens);        \
        } else {                                                                \
            total = num_tokens * hidden_size;                                    \
            blocks = (total + threads - 1) / threads;                           \
            embedding_gather_kernel<T><<<blocks, threads, 0, stream>>>(          \
                (T*)out, (const T*)weight, ids, hidden_size, num_tokens);        \
        }                                                                       \
    } while (0)

extern "C" {

void embedding_gather_f16(
    void* out, const void* weight, const uint32_t* ids,
    int hidden_size, int num_tokens, cudaStream_t stream
) {
    LAUNCH_GATHER(__half, 8);  // 8 x f16 = 128 bits
}

void embedding_gather_bf16(
    void* out, const void* weight, const uint32_t* ids,
    int hidden_size, int num_tokens, cudaStream_t stream
) {
    LAUNCH_GATHER(__nv_bfloat16, 8);
}

void embedding_gather_f32(
    void* out, const void* weight, const uint32_t* ids,
    int hidden_size, int num_tokens, cudaStream_t stream
) {
    LAUNCH_GATHER(float, 4);  // 4 x f32 = 128 bits
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Split fused QKV: [num_tokens, q_size + 2*kv_size] → Q, K, V contiguous
// One thread per element.
// ---------------------------------------------------------------------------

template<typename T>
__global__ void split_qkv_kernel(
    T* __restrict__ q_out,     // [num_tokens, q_size]
    T* __restrict__ k_out,     // [num_tokens, kv_size]
    T* __restrict__ v_out,     // [num_tokens, kv_size]
    const T* __restrict__ qkv, // [num_tokens, total_dim]
    int q_size,
    int kv_size,
    int total_dim,
    int num_tokens
) {
    // Grid covers all elements across Q, K, V outputs.
    // We map each thread to one element in one of the three outputs.
    int total_out = num_tokens * (q_size + 2 * kv_size);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    int token = idx / (q_size + 2 * kv_size);
    int within = idx % (q_size + 2 * kv_size);

    const T* src_row = qkv + token * total_dim;

    if (within < q_size) {
        q_out[token * q_size + within] = src_row[within];
    } else if (within < q_size + kv_size) {
        int k_offset = within - q_size;
        k_out[token * kv_size + k_offset] = src_row[q_size + k_offset];
    } else {
        int v_offset = within - q_size - kv_size;
        v_out[token * kv_size + v_offset] = src_row[q_size + kv_size + v_offset];
    }
}

#define LAUNCH_SPLIT_QKV(T)                                                    \
    do {                                                                        \
        int total = num_tokens * (q_size + 2 * kv_size);                       \
        int threads = 256;                                                      \
        int blocks = (total + threads - 1) / threads;                           \
        split_qkv_kernel<T><<<blocks, threads, 0, stream>>>(                   \
            (T*)q_out, (T*)k_out, (T*)v_out, (const T*)qkv,                   \
            q_size, kv_size, total_dim, num_tokens);                           \
    } while (0)

extern "C" {

void split_qkv_f16(
    void* q_out, void* k_out, void* v_out, const void* qkv,
    int q_size, int kv_size, int total_dim, int num_tokens,
    cudaStream_t stream
) {
    LAUNCH_SPLIT_QKV(__half);
}

void split_qkv_bf16(
    void* q_out, void* k_out, void* v_out, const void* qkv,
    int q_size, int kv_size, int total_dim, int num_tokens,
    cudaStream_t stream
) {
    LAUNCH_SPLIT_QKV(__nv_bfloat16);
}

void split_qkv_f32(
    void* q_out, void* k_out, void* v_out, const void* qkv,
    int q_size, int kv_size, int total_dim, int num_tokens,
    cudaStream_t stream
) {
    LAUNCH_SPLIT_QKV(float);
}

}  // extern "C"
