// SPDX-License-Identifier: Apache-2.0
// Fused reshape_and_cache CUDA kernel for vLLM Rust.
//
// Port of csrc/cache_kernels.cu from Python vLLM.
// Scatters newly computed K/V tokens into the paged KV cache using
// a slot_mapping that maps each token to its target cache slot.
//
// Cache layout: [num_blocks, block_size, num_kv_heads, head_dim] (NHD)
// This matches the "flash" variant in Python vLLM.
//
// Uses vectorized 128-bit loads/stores for throughput.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "vec_utils.cuh"

// ---------------------------------------------------------------------------
// reshape_and_cache kernel (vectorized)
// ---------------------------------------------------------------------------
// Grid:  num_tokens blocks (one per token)
// Block: min(num_heads * head_dim, 1024) threads
//
// Each token's K/V data ([num_heads, head_dim] elements) is copied from
// the input tensors to the target slot in the paged cache.

template <typename T>
__global__ void reshape_and_cache_kernel(
    const T* __restrict__ key,           // [num_tokens, num_heads, head_dim]
    const T* __restrict__ value,         // [num_tokens, num_heads, head_dim]
    T* __restrict__ key_cache,           // [num_blocks, block_size, num_heads, head_dim]
    T* __restrict__ value_cache,         // [num_blocks, block_size, num_heads, head_dim]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    int num_heads,
    int head_dim,
    int block_size)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const int token_idx = blockIdx.x;
    const int64_t slot = slot_mapping[token_idx];

    // slot < 0 means padding token — skip.
    if (slot < 0) {
        return;
    }

    const int n_elems = num_heads * head_dim;
    const int num_vecs = n_elems / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    // Source: contiguous input at [token_idx, :, :]
    const T* key_src = key + token_idx * n_elems;
    const T* value_src = value + token_idx * n_elems;

    // Destination: cache[block_idx, block_offset, :, :]
    T* key_dst = key_cache + slot * n_elems;
    T* value_dst = value_cache + slot * n_elems;

    // Vectorized copy.
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        vec_store(&key_dst[vi * VEC_SIZE], vec_load(&key_src[vi * VEC_SIZE]));
        vec_store(&value_dst[vi * VEC_SIZE], vec_load(&value_src[vi * VEC_SIZE]));
    }

    // Scalar tail.
    for (int i = tail_start + threadIdx.x; i < n_elems; i += blockDim.x) {
        key_dst[i] = key_src[i];
        value_dst[i] = value_src[i];
    }
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" {

void reshape_and_cache_f32(
    const float* key, const float* value,
    float* key_cache, float* value_cache,
    const int64_t* slot_mapping,
    int num_tokens, int num_heads, int head_dim, int block_size)
{
    int n_elems = num_heads * head_dim;
    int threads = (n_elems < 1024) ? n_elems : 1024;
    reshape_and_cache_kernel<float><<<num_tokens, threads>>>(
        key, value, key_cache, value_cache, slot_mapping,
        num_heads, head_dim, block_size);
}

void reshape_and_cache_f16(
    const __half* key, const __half* value,
    __half* key_cache, __half* value_cache,
    const int64_t* slot_mapping,
    int num_tokens, int num_heads, int head_dim, int block_size)
{
    int n_elems = num_heads * head_dim;
    int threads = (n_elems < 1024) ? n_elems : 1024;
    reshape_and_cache_kernel<__half><<<num_tokens, threads>>>(
        key, value, key_cache, value_cache, slot_mapping,
        num_heads, head_dim, block_size);
}

void reshape_and_cache_bf16(
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* key_cache, __nv_bfloat16* value_cache,
    const int64_t* slot_mapping,
    int num_tokens, int num_heads, int head_dim, int block_size)
{
    int n_elems = num_heads * head_dim;
    int threads = (n_elems < 1024) ? n_elems : 1024;
    reshape_and_cache_kernel<__nv_bfloat16><<<num_tokens, threads>>>(
        key, value, key_cache, value_cache, slot_mapping,
        num_heads, head_dim, block_size);
}

} // extern "C"
