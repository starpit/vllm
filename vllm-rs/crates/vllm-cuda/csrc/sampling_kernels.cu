// SPDX-License-Identifier: Apache-2.0
// Fused top-k / top-p / min-p sampling CUDA kernel for vLLM Rust.
//
// One thread block per request. The kernel computes softmax probabilities
// on-the-fly from logits, uses radix select to find the top-K threshold,
// compacts survivors to shared memory, sorts them, applies top-p and min-p
// cutoffs, then samples a token — all without materializing probabilities
// in global memory. Only a single u32 token ID is written back.
//
// Algorithm:
//   1. Softmax prep (2 passes over L2-cached logits): max reduction, sum_exp
//   2. Radix select (32 passes): find K-th largest probability threshold
//   3. Compact survivors to shared memory (1 pass, atomic)
//   4. Bitonic sort in shared memory (descending by probability)
//   5. Top-p cutoff: prefix sum, find cumsum > top_p
//   6. Min-p filter: discard prob < min_p * max_prob
//   7. Re-normalize + sample from uniform random

#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define SAMPLING_BLOCK_SIZE 256
#define MAX_CANDIDATES 1024
#define WARP_SIZE 32
#define NUM_WARPS (SAMPLING_BLOCK_SIZE / WARP_SIZE)

// ---------------------------------------------------------------------------
// Type conversion helpers
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float to_float(T val);

template <>
__device__ __forceinline__ float to_float<float>(float val) { return val; }

template <>
__device__ __forceinline__ float to_float<__half>(__half val) { return __half2float(val); }

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

// ---------------------------------------------------------------------------
// Warp-level reductions
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ int warp_reduce_sum_int(int val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Block-level reductions (all threads get the result)
// ---------------------------------------------------------------------------

__device__ float block_reduce_sum(float val, float* warp_buf) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane_id == 0) warp_buf[warp_id] = val;
    __syncthreads();

    if (tid < WARP_SIZE) {
        float v = (tid < NUM_WARPS) ? warp_buf[tid] : 0.0f;
        v = warp_reduce_sum(v);
        if (tid == 0) warp_buf[0] = v;
    }
    __syncthreads();
    float result = warp_buf[0];
    __syncthreads();
    return result;
}

__device__ float block_reduce_max(float val, float* warp_buf) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    val = warp_reduce_max(val);
    if (lane_id == 0) warp_buf[warp_id] = val;
    __syncthreads();

    if (tid < WARP_SIZE) {
        float v = (tid < NUM_WARPS) ? warp_buf[tid] : -INFINITY;
        v = warp_reduce_max(v);
        if (tid == 0) warp_buf[0] = v;
    }
    __syncthreads();
    float result = warp_buf[0];
    __syncthreads();
    return result;
}

__device__ int block_reduce_sum_int(int val, int* warp_buf) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    val = warp_reduce_sum_int(val);
    if (lane_id == 0) warp_buf[warp_id] = val;
    __syncthreads();

    if (tid < WARP_SIZE) {
        int v = (tid < NUM_WARPS) ? warp_buf[tid] : 0;
        v = warp_reduce_sum_int(v);
        if (tid == 0) warp_buf[0] = v;
    }
    __syncthreads();
    int result = warp_buf[0];
    __syncthreads();
    return result;
}

// ---------------------------------------------------------------------------
// Next power of 2
// ---------------------------------------------------------------------------

__device__ __forceinline__ int next_pow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

// ---------------------------------------------------------------------------
// Core sampling logic (called by both single and batched kernels)
// ---------------------------------------------------------------------------

template <typename T>
__device__ void sample_top_k_top_p_core(
    uint32_t* __restrict__ output,
    const T* __restrict__ logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    float uniform_random)
{
    // Shared memory
    __shared__ float s_warp_buf[NUM_WARPS];
    __shared__ int s_warp_buf_int[NUM_WARPS];
    __shared__ float s_probs[MAX_CANDIDATES];
    __shared__ uint32_t s_indices[MAX_CANDIDATES];
    __shared__ int s_num_candidates;

    int tid = threadIdx.x;
    float inv_temp = 1.0f / temperature;

    // ===== Phase 1: Find max logit (1 pass) =====
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += SAMPLING_BLOCK_SIZE) {
        float val = to_float(logits[i]) * inv_temp;
        local_max = fmaxf(local_max, val);
    }
    float max_logit = block_reduce_max(local_max, s_warp_buf);

    // ===== Phase 2: Compute sum_exp for softmax denominator (1 pass) =====
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += SAMPLING_BLOCK_SIZE) {
        float val = to_float(logits[i]) * inv_temp;
        local_sum += expf(val - max_logit);
    }
    float sum_exp = block_reduce_sum(local_sum, s_warp_buf);
    float inv_sum_exp = 1.0f / sum_exp;

    // prob(i) = exp(logit(i)/T - max_logit) * inv_sum_exp

    // ===== Phase 3: Radix select for top-K threshold (32 passes) =====
    // For positive IEEE 754 floats, bit ordering preserves value ordering.
    // We find the largest threshold T (as u32 bits) such that
    // count(prob >= T) >= effective_k.
    int effective_k = (top_k > 0)
        ? min(top_k, vocab_size)
        : min(MAX_CANDIDATES, vocab_size);

    uint32_t threshold_bits = 0;

    #pragma unroll 1
    for (int bit = 31; bit >= 0; bit--) {
        uint32_t candidate = threshold_bits | (1u << bit);

        int local_count = 0;
        for (int i = tid; i < vocab_size; i += SAMPLING_BLOCK_SIZE) {
            float val = to_float(logits[i]) * inv_temp;
            float prob = expf(val - max_logit) * inv_sum_exp;
            uint32_t bits = __float_as_uint(prob);
            if (bits >= candidate) local_count++;
        }

        int total_count = block_reduce_sum_int(local_count, s_warp_buf_int);
        if (total_count >= effective_k) {
            threshold_bits = candidate;
        }
    }

    float threshold_prob = __uint_as_float(threshold_bits);

    // ===== Phase 3b: Apply min_p threshold =====
    if (min_p > 0.0f) {
        // The max probability is the first element that passes any threshold.
        // We can find it cheaply: it's exp(0) * inv_sum_exp = inv_sum_exp
        // (since max_logit corresponds to the token with highest logit/T,
        //  and exp(max_logit - max_logit) = 1.0).
        float max_prob = inv_sum_exp;
        float min_p_threshold = min_p * max_prob;
        threshold_prob = fmaxf(threshold_prob, min_p_threshold);
    }

    // ===== Phase 4: Two-phase compaction to shared memory =====
    // When many tokens tie at the threshold (e.g. uniform logits), a naive
    // single-pass atomic compaction can overflow MAX_CANDIDATES and lose
    // high-probability tokens. Fix: first compact tokens strictly above
    // threshold (guaranteed few), then fill remaining slots with tied tokens.
    uint32_t threshold_bits_u = __float_as_uint(threshold_prob);
    int cap = min(effective_k, MAX_CANDIDATES);

    if (tid == 0) s_num_candidates = 0;
    __syncthreads();

    // Phase 4a: tokens strictly above threshold (prob bits > threshold_bits).
    for (int i = tid; i < vocab_size; i += SAMPLING_BLOCK_SIZE) {
        float val = to_float(logits[i]) * inv_temp;
        float prob = expf(val - max_logit) * inv_sum_exp;
        uint32_t bits = __float_as_uint(prob);
        if (bits > threshold_bits_u) {
            int pos = atomicAdd(&s_num_candidates, 1);
            if (pos < cap) {
                s_probs[pos] = prob;
                s_indices[pos] = (uint32_t)i;
            }
        }
    }
    __syncthreads();

    int strict_count = min(s_num_candidates, cap);

    // Phase 4b: tokens at threshold (fill remaining slots).
    if (strict_count < cap) {
        for (int i = tid; i < vocab_size; i += SAMPLING_BLOCK_SIZE) {
            float val = to_float(logits[i]) * inv_temp;
            float prob = expf(val - max_logit) * inv_sum_exp;
            uint32_t bits = __float_as_uint(prob);
            if (bits == threshold_bits_u) {
                int pos = atomicAdd(&s_num_candidates, 1);
                if (pos < cap) {
                    s_probs[pos] = prob;
                    s_indices[pos] = (uint32_t)i;
                }
            }
        }
    }
    __syncthreads();

    int num_candidates = min(s_num_candidates, cap);
    if (num_candidates <= 0) {
        // Fallback: should not happen, but return argmax token.
        if (tid == 0) {
            float best = -INFINITY;
            uint32_t best_idx = 0;
            for (int i = 0; i < vocab_size; i++) {
                float val = to_float(logits[i]);
                if (val > best) { best = val; best_idx = (uint32_t)i; }
            }
            output[0] = best_idx;
        }
        return;
    }

    // ===== Phase 5: Bitonic sort in shared memory (descending by prob) =====
    int n_padded = next_pow2(num_candidates);

    // Pad with zeros.
    for (int i = tid + num_candidates; i < n_padded; i += SAMPLING_BLOCK_SIZE) {
        s_probs[i] = 0.0f;
        s_indices[i] = 0;
    }
    __syncthreads();

    // Bitonic sort: descending order.
    for (int k = 2; k <= n_padded; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < n_padded; i += SAMPLING_BLOCK_SIZE) {
                int ixj = i ^ j;
                if (ixj > i) {
                    // For descending: (i & k) == 0 means compare-and-swap
                    // so that larger values come first.
                    bool swap_if_less = ((i & k) == 0);
                    float pi = s_probs[i];
                    float pj = s_probs[ixj];
                    if (swap_if_less ? (pi < pj) : (pi > pj)) {
                        s_probs[i] = pj;
                        s_probs[ixj] = pi;
                        uint32_t tmp = s_indices[i];
                        s_indices[i] = s_indices[ixj];
                        s_indices[ixj] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }

    // ===== Phase 6: Top-p cutoff + min-p filter + sampling (thread 0) =====
    if (tid == 0) {
        // Find top-p cutoff: first index where cumulative sum > top_p.
        float cumsum = 0.0f;
        int cutoff = num_candidates;
        for (int i = 0; i < num_candidates; i++) {
            cumsum += s_probs[i];
            if (cumsum > top_p) {
                cutoff = i + 1;  // inclusive: keep this token
                break;
            }
        }

        // Re-normalize surviving probs and sample.
        float total = 0.0f;
        for (int i = 0; i < cutoff; i++) {
            total += s_probs[i];
        }

        float target = uniform_random * total;
        cumsum = 0.0f;
        uint32_t sampled = s_indices[cutoff - 1];  // fallback: last survivor
        for (int i = 0; i < cutoff; i++) {
            cumsum += s_probs[i];
            if (cumsum >= target) {
                sampled = s_indices[i];
                break;
            }
        }

        output[0] = sampled;
    }
}

// ---------------------------------------------------------------------------
// Single-request kernel wrapper (scalar params, <<<1, 256>>>)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void sample_top_k_top_p_kernel(
    uint32_t* __restrict__ output,
    const T* __restrict__ logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    float uniform_random)
{
    sample_top_k_top_p_core(output, logits, vocab_size,
                            temperature, top_k, top_p, min_p, uniform_random);
}

// ---------------------------------------------------------------------------
// Batched kernel wrapper (array params, <<<batch_size, 256>>>)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void sample_top_k_top_p_batched_kernel(
    uint32_t* __restrict__ output,
    const T* __restrict__ logits,
    int vocab_size,
    const float* __restrict__ temperatures,
    const int* __restrict__ top_ks,
    const float* __restrict__ top_ps,
    const float* __restrict__ min_ps,
    const float* __restrict__ uniform_randoms)
{
    int bid = blockIdx.x;
    sample_top_k_top_p_core(
        output + bid,
        logits + bid * vocab_size,
        vocab_size,
        temperatures[bid], top_ks[bid], top_ps[bid],
        min_ps[bid], uniform_randoms[bid]);
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" {

// ---------------------------------------------------------------------------
// Single-request entry points (backwards compatible, launch <<<1, 256>>>)
// ---------------------------------------------------------------------------

void sample_top_k_top_p_f32(
    uint32_t* output,
    const float* logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    float uniform_random)
{
    sample_top_k_top_p_kernel<float><<<1, SAMPLING_BLOCK_SIZE>>>(
        output, logits, vocab_size, temperature, top_k, top_p, min_p,
        uniform_random);
}

void sample_top_k_top_p_f16(
    uint32_t* output,
    const uint16_t* logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    float uniform_random)
{
    sample_top_k_top_p_kernel<__half><<<1, SAMPLING_BLOCK_SIZE>>>(
        output, reinterpret_cast<const __half*>(logits), vocab_size,
        temperature, top_k, top_p, min_p, uniform_random);
}

void sample_top_k_top_p_bf16(
    uint32_t* output,
    const uint16_t* logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    float uniform_random)
{
    sample_top_k_top_p_kernel<__nv_bfloat16><<<1, SAMPLING_BLOCK_SIZE>>>(
        output, reinterpret_cast<const __nv_bfloat16*>(logits), vocab_size,
        temperature, top_k, top_p, min_p, uniform_random);
}

// ---------------------------------------------------------------------------
// Batched entry points (launch <<<batch_size, 256>>>)
// All param arrays are device pointers of length batch_size.
// ---------------------------------------------------------------------------

void sample_batched_f32(
    uint32_t* output,
    const float* logits,
    int vocab_size,
    int batch_size,
    const float* temperatures,
    const int* top_ks,
    const float* top_ps,
    const float* min_ps,
    const float* uniform_randoms,
    cudaStream_t stream)
{
    if (batch_size > 0) {
        sample_top_k_top_p_batched_kernel<float><<<batch_size, SAMPLING_BLOCK_SIZE, 0, stream>>>(
            output, logits, vocab_size,
            temperatures, top_ks, top_ps, min_ps, uniform_randoms);
    }
}

void sample_batched_f16(
    uint32_t* output,
    const uint16_t* logits,
    int vocab_size,
    int batch_size,
    const float* temperatures,
    const int* top_ks,
    const float* top_ps,
    const float* min_ps,
    const float* uniform_randoms,
    cudaStream_t stream)
{
    if (batch_size > 0) {
        sample_top_k_top_p_batched_kernel<__half><<<batch_size, SAMPLING_BLOCK_SIZE, 0, stream>>>(
            output, reinterpret_cast<const __half*>(logits), vocab_size,
            temperatures, top_ks, top_ps, min_ps, uniform_randoms);
    }
}

void sample_batched_bf16(
    uint32_t* output,
    const uint16_t* logits,
    int vocab_size,
    int batch_size,
    const float* temperatures,
    const int* top_ks,
    const float* top_ps,
    const float* min_ps,
    const float* uniform_randoms,
    cudaStream_t stream)
{
    if (batch_size > 0) {
        sample_top_k_top_p_batched_kernel<__nv_bfloat16><<<batch_size, SAMPLING_BLOCK_SIZE, 0, stream>>>(
            output, reinterpret_cast<const __nv_bfloat16*>(logits), vocab_size,
            temperatures, top_ks, top_ps, min_ps, uniform_randoms);
    }
}

}  // extern "C" (close before templates)

// ---------------------------------------------------------------------------
// Gumbel-max sampling: equivalent to sampling from softmax(logit/T) but
// requires only a single argmax-like pass. Uses the identity:
//   sample ~ Categorical(softmax(logit/T))
//   <==>  sample = argmax_i(logit_i/T + Gumbel_i)
// where Gumbel_i = -log(-log(U_i)), U_i ~ Uniform(0,1).
//
// Per-element randomness is generated via a counter-based hash (murmurhash3
// finalizer), seeded from the per-request uniform_random value.
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t murmurhash3_finalize(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

__device__ __forceinline__ float hash_to_uniform(uint32_t seed, uint32_t idx) {
    // Combine seed and index, then hash to get a uniform float in (0, 1).
    uint32_t h = murmurhash3_finalize(seed ^ (idx * 2654435761u));
    // Map to (0, 1) — exclude 0 to avoid log(0).
    return (float)(h >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}

template <typename T>
__global__ void sample_gumbel_batched_kernel(
    uint32_t* __restrict__ output,
    const T* __restrict__ logits,
    int vocab_size,
    const float* __restrict__ temperatures,
    const float* __restrict__ uniform_randoms)
{
    int bid = blockIdx.x;
    const T* row = logits + bid * vocab_size;
    float inv_temp = 1.0f / temperatures[bid];

    // Use the uniform random as seed for per-element noise.
    uint32_t seed = __float_as_uint(uniform_randoms[bid]);

    __shared__ float s_warp_buf[NUM_WARPS];
    __shared__ int s_warp_idx_buf[NUM_WARPS];

    int tid = threadIdx.x;
    float best_val = -INFINITY;
    int best_idx = 0;

    for (int i = tid; i < vocab_size; i += SAMPLING_BLOCK_SIZE) {
        float logit = to_float(row[i]) * inv_temp;
        // Gumbel noise: -log(-log(u))
        float u = hash_to_uniform(seed, (uint32_t)i);
        float gumbel = -logf(-logf(u));
        float score = logit + gumbel;
        if (score > best_val) {
            best_val = score;
            best_idx = i;
        }
    }

    // Warp reduce max with index (same as argmax_kernel).
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
        int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    if (lane_id == 0) {
        s_warp_buf[warp_id] = best_val;
        s_warp_idx_buf[warp_id] = best_idx;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        best_val = (tid < NUM_WARPS) ? s_warp_buf[tid] : -INFINITY;
        best_idx = (tid < NUM_WARPS) ? s_warp_idx_buf[tid] : 0;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
            int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
            if (other_val > best_val) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }
        if (tid == 0) {
            output[bid] = (uint32_t)best_idx;
        }
    }
}

extern "C" {

void sample_gumbel_batched_f32(
    uint32_t* output, const float* logits, int vocab_size, int batch_size,
    const float* temperatures, const float* uniform_randoms, cudaStream_t stream) {
    if (batch_size > 0)
        sample_gumbel_batched_kernel<float><<<batch_size, SAMPLING_BLOCK_SIZE, 0, stream>>>(
            output, logits, vocab_size, temperatures, uniform_randoms);
}

void sample_gumbel_batched_f16(
    uint32_t* output, const uint16_t* logits, int vocab_size, int batch_size,
    const float* temperatures, const float* uniform_randoms, cudaStream_t stream) {
    if (batch_size > 0)
        sample_gumbel_batched_kernel<__half><<<batch_size, SAMPLING_BLOCK_SIZE, 0, stream>>>(
            output, reinterpret_cast<const __half*>(logits), vocab_size, temperatures, uniform_randoms);
}

void sample_gumbel_batched_bf16(
    uint32_t* output, const uint16_t* logits, int vocab_size, int batch_size,
    const float* temperatures, const float* uniform_randoms, cudaStream_t stream) {
    if (batch_size > 0)
        sample_gumbel_batched_kernel<__nv_bfloat16><<<batch_size, SAMPLING_BLOCK_SIZE, 0, stream>>>(
            output, reinterpret_cast<const __nv_bfloat16*>(logits), vocab_size, temperatures, uniform_randoms);
}

}  // extern "C" (close gumbel)

// ---------------------------------------------------------------------------
// Batched argmax: one thread block per row, writes u32 token ID.
// Much cheaper than the full sampling kernel for greedy decoding.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void argmax_kernel(
    uint32_t* __restrict__ output,
    const T* __restrict__ logits,
    int vocab_size)
{
    int bid = blockIdx.x;
    const T* row = logits + bid * vocab_size;

    __shared__ float s_warp_buf[NUM_WARPS];
    __shared__ int s_warp_idx_buf[NUM_WARPS];

    int tid = threadIdx.x;
    float best_val = -INFINITY;
    int best_idx = 0;
    for (int i = tid; i < vocab_size; i += SAMPLING_BLOCK_SIZE) {
        float val = to_float(row[i]);
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    // Warp reduce max with index.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
        int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    if (lane_id == 0) {
        s_warp_buf[warp_id] = best_val;
        s_warp_idx_buf[warp_id] = best_idx;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        best_val = (tid < NUM_WARPS) ? s_warp_buf[tid] : -INFINITY;
        best_idx = (tid < NUM_WARPS) ? s_warp_idx_buf[tid] : 0;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
            int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
            if (other_val > best_val) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }
        if (tid == 0) {
            output[bid] = (uint32_t)best_idx;
        }
    }
}

extern "C" {

void argmax_batched_f32(uint32_t* output, const float* logits, int vocab_size, int batch_size, cudaStream_t stream) {
    if (batch_size > 0)
        argmax_kernel<float><<<batch_size, SAMPLING_BLOCK_SIZE, 0, stream>>>(output, logits, vocab_size);
}

void argmax_batched_f16(uint32_t* output, const uint16_t* logits, int vocab_size, int batch_size, cudaStream_t stream) {
    if (batch_size > 0)
        argmax_kernel<__half><<<batch_size, SAMPLING_BLOCK_SIZE, 0, stream>>>(output, reinterpret_cast<const __half*>(logits), vocab_size);
}

void argmax_batched_bf16(uint32_t* output, const uint16_t* logits, int vocab_size, int batch_size, cudaStream_t stream) {
    if (batch_size > 0)
        argmax_kernel<__nv_bfloat16><<<batch_size, SAMPLING_BLOCK_SIZE, 0, stream>>>(output, reinterpret_cast<const __nv_bfloat16*>(logits), vocab_size);
}

}  // extern "C" (close argmax)

// ---------------------------------------------------------------------------
// Cast logits to f32: simple element-wise conversion.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void cast_to_f32_kernel(
    float* __restrict__ output,
    const T* __restrict__ input,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = to_float(input[idx]);
    }
}

// ---------------------------------------------------------------------------
// Apply penalties: repetition, frequency, presence — fused, one block per row.
// Matches Python vLLM's apply_penalties logic.
//
// For each token in vocab:
//   count = occurrences in output_token_ids + prompt_token_ids
//   if count > 0:
//     logit = logit > 0 ? logit / rep_penalty : logit * rep_penalty
//     logit -= freq_penalty * count + pres_penalty
//
// Token ID == vocab_size is used as padding (ignored).
// ---------------------------------------------------------------------------

__global__ void apply_penalties_kernel(
    float* __restrict__ logits,       // [batch, vocab]
    const int* __restrict__ output_token_ids, // [batch, max_output_len] padded w/ vocab_size
    const int* __restrict__ prompt_token_ids, // [batch, max_prompt_len] padded w/ vocab_size
    const float* __restrict__ rep_penalties,  // [batch]
    const float* __restrict__ freq_penalties, // [batch]
    const float* __restrict__ pres_penalties, // [batch]
    int vocab_size,
    int max_output_len,
    int max_prompt_len)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float rep_pen = rep_penalties[bid];
    float freq_pen = freq_penalties[bid];
    float pres_pen = pres_penalties[bid];

    float* row = logits + bid * vocab_size;
    const int* out_ids = output_token_ids + bid * max_output_len;
    const int* prompt_ids = prompt_token_ids + bid * max_prompt_len;

    // For each token in this thread's slice of vocab, count occurrences.
    for (int v = tid; v < vocab_size; v += blockDim.x) {
        int count = 0;
        for (int j = 0; j < max_output_len; j++) {
            if (out_ids[j] == v) count++;
        }
        for (int j = 0; j < max_prompt_len; j++) {
            if (prompt_ids[j] == v) count++;
        }

        if (count > 0) {
            float logit = row[v];
            // Repetition penalty: shrink toward 0.
            if (logit > 0.0f) {
                logit /= rep_pen;
            } else {
                logit *= rep_pen;
            }
            // Frequency + presence penalties.
            logit -= freq_pen * (float)count + pres_pen;
            row[v] = logit;
        }
    }
}

// ---------------------------------------------------------------------------
// Apply logit bias: CSR-packed sparse scatter-add.
// bias_offsets[i]..bias_offsets[i+1] are the bias entries for request i.
// ---------------------------------------------------------------------------

__global__ void apply_logit_bias_kernel(
    float* __restrict__ logits,          // [batch, vocab]
    const int* __restrict__ bias_token_ids,  // [total_biases]
    const float* __restrict__ bias_values,   // [total_biases]
    const int* __restrict__ bias_offsets,    // [batch+1]
    int vocab_size)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float* row = logits + bid * vocab_size;
    int start = bias_offsets[bid];
    int end = bias_offsets[bid + 1];
    for (int i = start + tid; i < end; i += blockDim.x) {
        int token_id = bias_token_ids[i];
        if (token_id >= 0 && token_id < vocab_size) {
            row[token_id] += bias_values[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Apply grammar mask: CSR-packed allow-list. For grammar requests, set all
// logits to -inf, then restore allowed tokens to their original values.
// req_indices maps dense grammar index to batch row index.
// ---------------------------------------------------------------------------

__global__ void apply_grammar_mask_kernel(
    float* __restrict__ logits,              // [batch, vocab]
    const float* __restrict__ logits_backup, // [batch, vocab] (copy before masking)
    const int* __restrict__ allowed_ids,     // [total_allowed]
    const int* __restrict__ allowed_offsets,  // [num_grammar_reqs+1]
    const int* __restrict__ req_indices,     // [num_grammar_reqs] -> batch row index
    int vocab_size)
{
    int gid = blockIdx.x;  // grammar request index
    int tid = threadIdx.x;
    int bid = req_indices[gid]; // actual batch row
    float* row = logits + bid * vocab_size;

    // Step 1: set all logits to -inf.
    for (int v = tid; v < vocab_size; v += blockDim.x) {
        row[v] = -INFINITY;
    }
    __syncthreads();

    // Step 2: restore allowed tokens from backup.
    const float* backup_row = logits_backup + bid * vocab_size;
    int start = allowed_offsets[gid];
    int end = allowed_offsets[gid + 1];
    for (int i = start + tid; i < end; i += blockDim.x) {
        int token_id = allowed_ids[i];
        if (token_id >= 0 && token_id < vocab_size) {
            row[token_id] = backup_row[token_id];
        }
    }
}

// ---------------------------------------------------------------------------
// Fused log-softmax + top-k: compute log-softmax, gather sampled token's
// logprob, then find top-K logprobs. One block per request.
//
// Output layout per request: [sampled_logprob, top1, top2, ..., topK]
// with corresponding indices.  Total: (num_logprobs + 1) entries.
// ---------------------------------------------------------------------------

__global__ void log_softmax_topk_kernel(
    const float* __restrict__ logits,   // [batch, vocab]
    const uint32_t* __restrict__ sampled_ids, // [batch]
    int vocab_size,
    int num_logprobs,
    float* __restrict__ out_logprobs,   // [batch, num_logprobs+1]
    int* __restrict__ out_indices,      // [batch, num_logprobs+1]
    uint32_t* __restrict__ out_ranks)   // [batch, num_logprobs+1]
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int k = num_logprobs + 1; // output width per request
    const float* row = logits + bid * vocab_size;
    uint32_t sampled_id = sampled_ids[bid];

    __shared__ float s_warp_buf[NUM_WARPS];

    // --- Pass 1: max reduction ---
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += SAMPLING_BLOCK_SIZE) {
        local_max = fmaxf(local_max, row[i]);
    }
    float max_val = block_reduce_max(local_max, s_warp_buf);

    // --- Pass 2: sum_exp ---
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += SAMPLING_BLOCK_SIZE) {
        local_sum += expf(row[i] - max_val);
    }
    float sum_exp = block_reduce_sum(local_sum, s_warp_buf);
    float log_sum_exp = logf(sum_exp);

    // log_softmax(i) = row[i] - max_val - log(sum_exp)

    // --- Slot 0: sampled token's logprob + rank ---
    float sampled_logit = row[sampled_id];
    if (tid == 0) {
        float sampled_lp = sampled_logit - max_val - log_sum_exp;
        out_logprobs[bid * k] = sampled_lp;
        out_indices[bid * k] = (int)sampled_id;
    }

    // Compute rank of sampled token (1-indexed).
    int local_higher = 0;
    for (int i = tid; i < vocab_size; i += SAMPLING_BLOCK_SIZE) {
        if (row[i] > sampled_logit) local_higher++;
    }
    int total_higher = block_reduce_sum_int(local_higher, (int*)s_warp_buf);
    if (tid == 0) {
        out_ranks[bid * k] = (uint32_t)(total_higher + 1);
    }

    // --- Slots 1..num_logprobs: top-K by iterative selection ---
    float prev_threshold = INFINITY;
    int prev_idx = -1;
    for (int slot = 1; slot < k; slot++) {
        float best_val = -INFINITY;
        int best_idx = vocab_size;
        for (int i = tid; i < vocab_size; i += SAMPLING_BLOCK_SIZE) {
            float v = row[i];
            bool candidate;
            if (slot == 1) {
                candidate = true;
            } else {
                candidate = (v < prev_threshold) ||
                            (v == prev_threshold && i > prev_idx);
            }
            if (candidate && (v > best_val || (v == best_val && i < best_idx))) {
                best_val = v;
                best_idx = i;
            }
        }

        // Block reduce: find global best (max value, min index on tie).
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
            int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
            if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }

        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        __shared__ float s_topk_val[NUM_WARPS];
        __shared__ int s_topk_idx[NUM_WARPS];
        if (lane_id == 0) {
            s_topk_val[warp_id] = best_val;
            s_topk_idx[warp_id] = best_idx;
        }
        __syncthreads();

        if (tid < WARP_SIZE) {
            best_val = (tid < NUM_WARPS) ? s_topk_val[tid] : -INFINITY;
            best_idx = (tid < NUM_WARPS) ? s_topk_idx[tid] : vocab_size;
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
                int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
                if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
                    best_val = other_val;
                    best_idx = other_idx;
                }
            }
        }
        // Broadcast from thread 0.
        best_val = __shfl_sync(0xffffffff, best_val, 0);
        best_idx = __shfl_sync(0xffffffff, best_idx, 0);

        if (tid == 0) {
            float lp = best_val - max_val - log_sum_exp;
            out_logprobs[bid * k + slot] = lp;
            out_indices[bid * k + slot] = best_idx;
        }

        // Compute rank of this top-k token.
        int local_h = 0;
        for (int i = tid; i < vocab_size; i += SAMPLING_BLOCK_SIZE) {
            if (row[i] > best_val) local_h++;
        }
        int total_h = block_reduce_sum_int(local_h, (int*)s_warp_buf);
        if (tid == 0) {
            out_ranks[bid * k + slot] = (uint32_t)(total_h + 1);
        }

        prev_threshold = best_val;
        prev_idx = best_idx;
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// C entry points for penalties / bias / grammar / logprobs kernels
// ---------------------------------------------------------------------------

extern "C" {

void cast_to_f32_f16(float* output, const uint16_t* input, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (n > 0)
        cast_to_f32_kernel<__half><<<blocks, threads, 0, stream>>>(
            output, reinterpret_cast<const __half*>(input), n);
}

void cast_to_f32_bf16(float* output, const uint16_t* input, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (n > 0)
        cast_to_f32_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            output, reinterpret_cast<const __nv_bfloat16*>(input), n);
}

void apply_penalties_inplace(
    float* logits,
    const int* output_token_ids,
    const int* prompt_token_ids,
    const float* rep_penalties,
    const float* freq_penalties,
    const float* pres_penalties,
    int vocab_size,
    int batch_size,
    int max_output_len,
    int max_prompt_len,
    cudaStream_t stream)
{
    if (batch_size > 0) {
        apply_penalties_kernel<<<batch_size, SAMPLING_BLOCK_SIZE, 0, stream>>>(
            logits, output_token_ids, prompt_token_ids,
            rep_penalties, freq_penalties, pres_penalties,
            vocab_size, max_output_len, max_prompt_len);
    }
}

void apply_logit_bias_inplace(
    float* logits,
    const int* bias_token_ids,
    const float* bias_values,
    const int* bias_offsets,
    int vocab_size,
    int batch_size,
    cudaStream_t stream)
{
    if (batch_size > 0) {
        apply_logit_bias_kernel<<<batch_size, SAMPLING_BLOCK_SIZE, 0, stream>>>(
            logits, bias_token_ids, bias_values, bias_offsets, vocab_size);
    }
}

void apply_grammar_mask_inplace(
    float* logits,
    const float* logits_backup,
    const int* allowed_ids,
    const int* allowed_offsets,
    const int* req_indices,
    int vocab_size,
    int num_grammar_reqs,
    cudaStream_t stream)
{
    if (num_grammar_reqs > 0) {
        apply_grammar_mask_kernel<<<num_grammar_reqs, SAMPLING_BLOCK_SIZE, 0, stream>>>(
            logits, logits_backup, allowed_ids, allowed_offsets, req_indices, vocab_size);
    }
}

void log_softmax_topk(
    const float* logits,
    const uint32_t* sampled_ids,
    int vocab_size,
    int batch_size,
    int num_logprobs,
    float* out_logprobs,
    int* out_indices,
    uint32_t* out_ranks,
    cudaStream_t stream)
{
    if (batch_size > 0) {
        log_softmax_topk_kernel<<<batch_size, SAMPLING_BLOCK_SIZE, 0, stream>>>(
            logits, sampled_ids, vocab_size, num_logprobs,
            out_logprobs, out_indices, out_ranks);
    }
}

}  // extern "C" (new kernels)
