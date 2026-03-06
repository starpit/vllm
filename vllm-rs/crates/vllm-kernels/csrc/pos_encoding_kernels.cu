// SPDX-License-Identifier: Apache-2.0
// Fused rotary position embedding (RoPE) CUDA kernel for vLLM Rust.
//
// Port of csrc/pos_encoding_kernels.cu from Python vLLM.
// One thread block per token. Each thread handles one (head, rot_offset)
// pair, performing the NeoX-style rotation in-place.
//
// Uses vectorized 128-bit loads/stores where alignment allows.
// RoPE pairs (x, y) are at offsets [i] and [i + half], which may not be
// adjacent for vectorization, so we vectorize the cos/sin loads and
// process multiple rotation pairs per thread iteration.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "vec_utils.cuh"

// ---------------------------------------------------------------------------
// Fused RoPE kernel: lookup cos/sin from cache + rotate per head, in-place
// ---------------------------------------------------------------------------

template <typename T>
__global__ void rotary_embedding_kernel(
    const uint32_t* __restrict__ positions,  // [num_tokens]
    T* __restrict__ query,                   // [num_tokens, total_q_dim]
    T* __restrict__ key,                     // [num_tokens, total_k_dim]
    const T* __restrict__ cos_sin_cache,     // [max_pos, rotary_dim]
    int rotary_dim,                          // = 2 * half_rot
    int total_q_dim,                         // = num_q_heads * head_size
    int total_k_dim,                         // = num_kv_heads * head_size
    int head_size)                           // dimension per head
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const int token_idx = blockIdx.x;
    const int pos = static_cast<int>(positions[token_idx]);
    const int half_rot = rotary_dim / 2;

    // Pointer into the cos/sin cache for this position.
    const T* cos_ptr = cos_sin_cache + pos * rotary_dim;
    const T* sin_ptr = cos_ptr + half_rot;

    // Compute the rotary dimension per head (min of half_rot and head_size/2).
    const int half = (half_rot < head_size / 2) ? half_rot : (head_size / 2);

    // Number of vec-sized chunks in the rotary half-dimension.
    const int half_vecs = half / VEC_SIZE;
    const int half_tail = half_vecs * VEC_SIZE;

    // --- Rotate query heads ---
    const int num_q_heads = total_q_dim / head_size;
    T* q = query + token_idx * total_q_dim;

    // Vectorized: process VEC_SIZE rotation pairs at a time.
    for (int tid = threadIdx.x; tid < num_q_heads * half_vecs; tid += blockDim.x) {
        const int head = tid / half_vecs;
        const int vi = tid % half_vecs;
        const int base = head * head_size + vi * VEC_SIZE;

        float xbuf[VEC_SIZE], ybuf[VEC_SIZE], cbuf[VEC_SIZE], sbuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&q[base]), xbuf);
        unpack_vec<T>(vec_load(&q[base + half]), ybuf);
        unpack_vec<T>(vec_load(&cos_ptr[vi * VEC_SIZE]), cbuf);
        unpack_vec<T>(vec_load(&sin_ptr[vi * VEC_SIZE]), sbuf);

        float ox[VEC_SIZE], oy[VEC_SIZE];
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            ox[j] = xbuf[j] * cbuf[j] - ybuf[j] * sbuf[j];
            oy[j] = ybuf[j] * cbuf[j] + xbuf[j] * sbuf[j];
        }
        vec_store(&q[base], pack_vec<T>(ox));
        vec_store(&q[base + half], pack_vec<T>(oy));
    }

    // Scalar tail for remaining rotation pairs.
    for (int tid = threadIdx.x; tid < num_q_heads * (half - half_tail); tid += blockDim.x) {
        const int head = tid / (half - half_tail);
        const int rot_offset = half_tail + tid % (half - half_tail);
        const int base = head * head_size;

        float x = static_cast<float>(q[base + rot_offset]);
        float y = static_cast<float>(q[base + rot_offset + half]);
        float c = static_cast<float>(cos_ptr[rot_offset]);
        float s = static_cast<float>(sin_ptr[rot_offset]);

        q[base + rot_offset]        = static_cast<T>(x * c - y * s);
        q[base + rot_offset + half] = static_cast<T>(y * c + x * s);
    }

    // --- Rotate key heads ---
    if (total_k_dim > 0) {
        const int num_k_heads = total_k_dim / head_size;
        T* k = key + token_idx * total_k_dim;

        // Vectorized.
        for (int tid = threadIdx.x; tid < num_k_heads * half_vecs; tid += blockDim.x) {
            const int head = tid / half_vecs;
            const int vi = tid % half_vecs;
            const int base = head * head_size + vi * VEC_SIZE;

            float xbuf[VEC_SIZE], ybuf[VEC_SIZE], cbuf[VEC_SIZE], sbuf[VEC_SIZE];
            unpack_vec<T>(vec_load(&k[base]), xbuf);
            unpack_vec<T>(vec_load(&k[base + half]), ybuf);
            unpack_vec<T>(vec_load(&cos_ptr[vi * VEC_SIZE]), cbuf);
            unpack_vec<T>(vec_load(&sin_ptr[vi * VEC_SIZE]), sbuf);

            float ox[VEC_SIZE], oy[VEC_SIZE];
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                ox[j] = xbuf[j] * cbuf[j] - ybuf[j] * sbuf[j];
                oy[j] = ybuf[j] * cbuf[j] + xbuf[j] * sbuf[j];
            }
            vec_store(&k[base], pack_vec<T>(ox));
            vec_store(&k[base + half], pack_vec<T>(oy));
        }

        // Scalar tail.
        for (int tid = threadIdx.x; tid < num_k_heads * (half - half_tail); tid += blockDim.x) {
            const int head = tid / (half - half_tail);
            const int rot_offset = half_tail + tid % (half - half_tail);
            const int base = head * head_size;

            float x = static_cast<float>(k[base + rot_offset]);
            float y = static_cast<float>(k[base + rot_offset + half]);
            float c = static_cast<float>(cos_ptr[rot_offset]);
            float s = static_cast<float>(sin_ptr[rot_offset]);

            k[base + rot_offset]        = static_cast<T>(x * c - y * s);
            k[base + rot_offset + half] = static_cast<T>(y * c + x * s);
        }
    }
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" {

void rotary_embedding_f32(
    const uint32_t* positions, float* query, float* key,
    const float* cos_sin_cache,
    int rotary_dim, int total_q_dim, int total_k_dim,
    int head_size, int num_tokens, cudaStream_t stream)
{
    int half = rotary_dim / 2;
    int num_q_heads = total_q_dim / head_size;
    int work = num_q_heads * half;
    int threads = (work < 512) ? work : 512;
    if (threads < 1) threads = 1;
    rotary_embedding_kernel<float><<<num_tokens, threads, 0, stream>>>(
        positions, query, key, cos_sin_cache,
        rotary_dim, total_q_dim, total_k_dim, head_size);
}

void rotary_embedding_f16(
    const uint32_t* positions, __half* query, __half* key,
    const __half* cos_sin_cache,
    int rotary_dim, int total_q_dim, int total_k_dim,
    int head_size, int num_tokens, cudaStream_t stream)
{
    int half = rotary_dim / 2;
    int num_q_heads = total_q_dim / head_size;
    int work = num_q_heads * half;
    int threads = (work < 512) ? work : 512;
    if (threads < 1) threads = 1;
    rotary_embedding_kernel<__half><<<num_tokens, threads, 0, stream>>>(
        positions, query, key, cos_sin_cache,
        rotary_dim, total_q_dim, total_k_dim, head_size);
}

void rotary_embedding_bf16(
    const uint32_t* positions, __nv_bfloat16* query, __nv_bfloat16* key,
    const __nv_bfloat16* cos_sin_cache,
    int rotary_dim, int total_q_dim, int total_k_dim,
    int head_size, int num_tokens, cudaStream_t stream)
{
    int half = rotary_dim / 2;
    int num_q_heads = total_q_dim / head_size;
    int work = num_q_heads * half;
    int threads = (work < 512) ? work : 512;
    if (threads < 1) threads = 1;
    rotary_embedding_kernel<__nv_bfloat16><<<num_tokens, threads, 0, stream>>>(
        positions, query, key, cos_sin_cache,
        rotary_dim, total_q_dim, total_k_dim, head_size);
}

} // extern "C"
