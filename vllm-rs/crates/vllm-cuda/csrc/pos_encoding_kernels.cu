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
// Fused QKV split + RoPE kernel
//
// Reads from the fused QKV GEMM output [num_tokens, total_dim], applies
// RoPE to Q and K, and writes contiguous Q, K, V outputs.
// Replaces separate split_qkv + rotary_embedding kernels (saves 1 launch).
// ---------------------------------------------------------------------------

template <typename T>
__global__ void fused_qkv_rope_kernel(
    T* __restrict__ q_out,                   // [num_tokens, q_size]
    T* __restrict__ k_out,                   // [num_tokens, kv_size]
    T* __restrict__ v_out,                   // [num_tokens, kv_size]
    const T* __restrict__ qkv,               // [num_tokens, total_dim]
    const uint32_t* __restrict__ positions,  // [num_tokens]
    const T* __restrict__ cos_sin_cache,     // [max_pos, rotary_dim]
    int q_size,                              // = num_q_heads * head_size
    int kv_size,                             // = num_kv_heads * head_size
    int total_dim,                           // = q_size + 2 * kv_size
    int rotary_dim,                          // = 2 * half_rot
    int head_size)
{
    constexpr int VEC = VecType<T>::SIZE;

    const int token_idx = blockIdx.x;
    const int pos = static_cast<int>(positions[token_idx]);
    const int half_rot = rotary_dim / 2;
    const T* cos_ptr = cos_sin_cache + pos * rotary_dim;
    const T* sin_ptr = cos_ptr + half_rot;
    const int half = (half_rot < head_size / 2) ? half_rot : (head_size / 2);
    const int half_vecs = half / VEC;
    const int half_tail = half_vecs * VEC;

    const T* row = qkv + token_idx * total_dim;
    T* q = q_out + token_idx * q_size;
    T* k = k_out + token_idx * kv_size;
    T* v = v_out + token_idx * kv_size;

    const int nqh = q_size / head_size;
    const int nkh = kv_size / head_size;
    const int rot_dim_full = 2 * half;  // actual rotary portion per head
    const int non_rot = head_size - rot_dim_full;

    // --- Q heads: read from QKV, apply RoPE, write contiguous ---
    // Vectorized rotary pairs.
    for (int tid = threadIdx.x; tid < nqh * half_vecs; tid += blockDim.x) {
        const int h = tid / half_vecs;
        const int vi = tid % half_vecs;
        const int b = h * head_size + vi * VEC;

        float xbuf[VEC], ybuf[VEC], cbuf[VEC], sbuf[VEC];
        unpack_vec<T>(vec_load(&row[b]), xbuf);
        unpack_vec<T>(vec_load(&row[b + half]), ybuf);
        unpack_vec<T>(vec_load(&cos_ptr[vi * VEC]), cbuf);
        unpack_vec<T>(vec_load(&sin_ptr[vi * VEC]), sbuf);

        float ox[VEC], oy[VEC];
        #pragma unroll
        for (int j = 0; j < VEC; j++) {
            ox[j] = xbuf[j] * cbuf[j] - ybuf[j] * sbuf[j];
            oy[j] = ybuf[j] * cbuf[j] + xbuf[j] * sbuf[j];
        }
        vec_store(&q[b], pack_vec<T>(ox));
        vec_store(&q[b + half], pack_vec<T>(oy));
    }

    // Scalar tail for remaining rotary pairs.
    for (int tid = threadIdx.x; tid < nqh * (half - half_tail); tid += blockDim.x) {
        const int h = tid / (half - half_tail);
        const int r = half_tail + tid % (half - half_tail);
        const int b = h * head_size;

        float x = static_cast<float>(row[b + r]);
        float y = static_cast<float>(row[b + r + half]);
        float c = static_cast<float>(cos_ptr[r]);
        float s = static_cast<float>(sin_ptr[r]);

        q[b + r]        = static_cast<T>(x * c - y * s);
        q[b + r + half] = static_cast<T>(y * c + x * s);
    }

    // Copy non-rotary elements (when head_size > rotary_dim).
    for (int tid = threadIdx.x; tid < nqh * non_rot; tid += blockDim.x) {
        const int h = tid / non_rot;
        const int i = rot_dim_full + tid % non_rot;
        q[h * head_size + i] = row[h * head_size + i];
    }

    // --- K heads: read from QKV + q_size offset, apply RoPE, write contiguous ---
    const T* k_row = row + q_size;

    for (int tid = threadIdx.x; tid < nkh * half_vecs; tid += blockDim.x) {
        const int h = tid / half_vecs;
        const int vi = tid % half_vecs;
        const int b = h * head_size + vi * VEC;

        float xbuf[VEC], ybuf[VEC], cbuf[VEC], sbuf[VEC];
        unpack_vec<T>(vec_load(&k_row[b]), xbuf);
        unpack_vec<T>(vec_load(&k_row[b + half]), ybuf);
        unpack_vec<T>(vec_load(&cos_ptr[vi * VEC]), cbuf);
        unpack_vec<T>(vec_load(&sin_ptr[vi * VEC]), sbuf);

        float ox[VEC], oy[VEC];
        #pragma unroll
        for (int j = 0; j < VEC; j++) {
            ox[j] = xbuf[j] * cbuf[j] - ybuf[j] * sbuf[j];
            oy[j] = ybuf[j] * cbuf[j] + xbuf[j] * sbuf[j];
        }
        vec_store(&k[b], pack_vec<T>(ox));
        vec_store(&k[b + half], pack_vec<T>(oy));
    }

    for (int tid = threadIdx.x; tid < nkh * (half - half_tail); tid += blockDim.x) {
        const int h = tid / (half - half_tail);
        const int r = half_tail + tid % (half - half_tail);
        const int b = h * head_size;

        float x = static_cast<float>(k_row[b + r]);
        float y = static_cast<float>(k_row[b + r + half]);
        float c = static_cast<float>(cos_ptr[r]);
        float s = static_cast<float>(sin_ptr[r]);

        k[b + r]        = static_cast<T>(x * c - y * s);
        k[b + r + half] = static_cast<T>(y * c + x * s);
    }

    for (int tid = threadIdx.x; tid < nkh * non_rot; tid += blockDim.x) {
        const int h = tid / non_rot;
        const int i = rot_dim_full + tid % non_rot;
        k[h * head_size + i] = k_row[h * head_size + i];
    }

    // --- V: straight copy from QKV + q_size + kv_size offset (vectorized) ---
    const T* v_row = row + q_size + kv_size;
    const int v_vecs = kv_size / VEC;

    for (int tid = threadIdx.x; tid < v_vecs; tid += blockDim.x) {
        vec_store(&v[tid * VEC], vec_load(&v_row[tid * VEC]));
    }
    // Scalar tail for V.
    for (int tid = threadIdx.x; tid < kv_size - v_vecs * VEC; tid += blockDim.x) {
        v[v_vecs * VEC + tid] = v_row[v_vecs * VEC + tid];
    }
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

#define LAUNCH_FUSED_QKV_ROPE(T)                                               \
    do {                                                                        \
        int half = rotary_dim / 2;                                             \
        int nqh = q_size / head_size;                                          \
        int work = nqh * half;                                                 \
        int threads = (work < 512) ? work : 512;                               \
        if (threads < 1) threads = 1;                                          \
        fused_qkv_rope_kernel<T><<<num_tokens, threads, 0, stream>>>(          \
            (T*)q_out, (T*)k_out, (T*)v_out, (const T*)qkv,                   \
            (const uint32_t*)positions, (const T*)cos_sin_cache,               \
            q_size, kv_size, total_dim, rotary_dim, head_size);                \
    } while (0)

extern "C" {

void fused_qkv_rope_f32(
    void* q_out, void* k_out, void* v_out, const void* qkv,
    const void* positions, const void* cos_sin_cache,
    int q_size, int kv_size, int total_dim, int rotary_dim,
    int head_size, int num_tokens, cudaStream_t stream)
{
    LAUNCH_FUSED_QKV_ROPE(float);
}

void fused_qkv_rope_f16(
    void* q_out, void* k_out, void* v_out, const void* qkv,
    const void* positions, const void* cos_sin_cache,
    int q_size, int kv_size, int total_dim, int rotary_dim,
    int head_size, int num_tokens, cudaStream_t stream)
{
    LAUNCH_FUSED_QKV_ROPE(__half);
}

void fused_qkv_rope_bf16(
    void* q_out, void* k_out, void* v_out, const void* qkv,
    const void* positions, const void* cos_sin_cache,
    int q_size, int kv_size, int total_dim, int rotary_dim,
    int head_size, int num_tokens, cudaStream_t stream)
{
    LAUNCH_FUSED_QKV_ROPE(__nv_bfloat16);
}

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

// ---------------------------------------------------------------------------
// Interleaved RoPE kernel (Cohere convention)
//
// Pairs (2i, 2i+1) are rotated together, unlike NeoX which pairs (i, i+half).
// cos_sin_cache layout is the same: [max_pos, rotary_dim] where first half
// is cos, second half is sin. But we index differently when applying.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void fused_qkv_interleaved_rope_kernel(
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    T* __restrict__ v_out,
    const T* __restrict__ qkv,
    const uint32_t* __restrict__ positions,
    const T* __restrict__ cos_sin_cache,
    int q_size,
    int kv_size,
    int total_dim,
    int rotary_dim,
    int head_size)
{
    const int token_idx = blockIdx.x;
    const int pos = static_cast<int>(positions[token_idx]);
    const int half_rot = rotary_dim / 2;
    const T* cos_ptr = cos_sin_cache + pos * rotary_dim;
    const T* sin_ptr = cos_ptr + half_rot;
    // Number of interleaved pairs per head = min(half_rot, head_size/2)
    const int num_pairs = (half_rot < head_size / 2) ? half_rot : (head_size / 2);
    const int non_rot = head_size - 2 * num_pairs;

    const T* row = qkv + token_idx * total_dim;
    T* q = q_out + token_idx * q_size;
    T* k = k_out + token_idx * kv_size;
    T* v = v_out + token_idx * kv_size;

    const int nqh = q_size / head_size;
    const int nkh = kv_size / head_size;

    // --- Q heads: interleaved RoPE ---
    for (int tid = threadIdx.x; tid < nqh * num_pairs; tid += blockDim.x) {
        const int h = tid / num_pairs;
        const int p = tid % num_pairs;
        const int base = h * head_size + 2 * p;  // interleaved: pair at (2p, 2p+1)

        float x0 = static_cast<float>(row[base]);
        float x1 = static_cast<float>(row[base + 1]);
        float c = static_cast<float>(cos_ptr[p]);
        float s = static_cast<float>(sin_ptr[p]);

        q[base]     = static_cast<T>(x0 * c - x1 * s);
        q[base + 1] = static_cast<T>(x1 * c + x0 * s);
    }
    // Copy non-rotary elements.
    for (int tid = threadIdx.x; tid < nqh * non_rot; tid += blockDim.x) {
        const int h = tid / non_rot;
        const int i = 2 * num_pairs + tid % non_rot;
        q[h * head_size + i] = row[h * head_size + i];
    }

    // --- K heads: interleaved RoPE ---
    const T* k_row = row + q_size;
    for (int tid = threadIdx.x; tid < nkh * num_pairs; tid += blockDim.x) {
        const int h = tid / num_pairs;
        const int p = tid % num_pairs;
        const int base = h * head_size + 2 * p;

        float x0 = static_cast<float>(k_row[base]);
        float x1 = static_cast<float>(k_row[base + 1]);
        float c = static_cast<float>(cos_ptr[p]);
        float s = static_cast<float>(sin_ptr[p]);

        k[base]     = static_cast<T>(x0 * c - x1 * s);
        k[base + 1] = static_cast<T>(x1 * c + x0 * s);
    }
    for (int tid = threadIdx.x; tid < nkh * non_rot; tid += blockDim.x) {
        const int h = tid / non_rot;
        const int i = 2 * num_pairs + tid % non_rot;
        k[h * head_size + i] = k_row[h * head_size + i];
    }

    // --- V: straight copy ---
    constexpr int VEC = VecType<T>::SIZE;
    const T* v_row = row + q_size + kv_size;
    const int v_vecs = kv_size / VEC;
    for (int tid = threadIdx.x; tid < v_vecs; tid += blockDim.x) {
        vec_store(&v[tid * VEC], vec_load(&v_row[tid * VEC]));
    }
    for (int tid = threadIdx.x; tid < kv_size - v_vecs * VEC; tid += blockDim.x) {
        v[v_vecs * VEC + tid] = v_row[v_vecs * VEC + tid];
    }
}

#define LAUNCH_FUSED_QKV_INTERLEAVED_ROPE(T)                                    \
    do {                                                                         \
        int half = rotary_dim / 2;                                              \
        int nqh = q_size / head_size;                                           \
        int work = nqh * half;                                                  \
        int threads = (work < 512) ? work : 512;                                \
        if (threads < 1) threads = 1;                                           \
        fused_qkv_interleaved_rope_kernel<T><<<num_tokens, threads, 0, stream>>>( \
            (T*)q_out, (T*)k_out, (T*)v_out, (const T*)qkv,                    \
            (const uint32_t*)positions, (const T*)cos_sin_cache,                \
            q_size, kv_size, total_dim, rotary_dim, head_size);                 \
    } while (0)

extern "C" {

void fused_qkv_interleaved_rope_f32(
    void* q_out, void* k_out, void* v_out, const void* qkv,
    const void* positions, const void* cos_sin_cache,
    int q_size, int kv_size, int total_dim, int rotary_dim,
    int head_size, int num_tokens, cudaStream_t stream)
{
    LAUNCH_FUSED_QKV_INTERLEAVED_ROPE(float);
}

void fused_qkv_interleaved_rope_f16(
    void* q_out, void* k_out, void* v_out, const void* qkv,
    const void* positions, const void* cos_sin_cache,
    int q_size, int kv_size, int total_dim, int rotary_dim,
    int head_size, int num_tokens, cudaStream_t stream)
{
    LAUNCH_FUSED_QKV_INTERLEAVED_ROPE(__half);
}

void fused_qkv_interleaved_rope_bf16(
    void* q_out, void* k_out, void* v_out, const void* qkv,
    const void* positions, const void* cos_sin_cache,
    int q_size, int kv_size, int total_dim, int rotary_dim,
    int head_size, int num_tokens, cudaStream_t stream)
{
    LAUNCH_FUSED_QKV_INTERLEAVED_ROPE(__nv_bfloat16);
}

} // extern "C"
