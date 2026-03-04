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
// Main sampling kernel
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
// C entry points (one per logit dtype)
// ---------------------------------------------------------------------------

extern "C" {

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

}  // extern "C"
