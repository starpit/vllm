#include <metal_stdlib>
using namespace metal;

// Basic single-head attention kernel (Phase 4.1.1)
// Simplified version without paging, quantization, or advanced features
// Goal: Prove numerical correctness of Metal attention implementation

struct AttentionParams {
    uint seq_len;        // Length of key/value sequence
    uint head_size;      // Dimension of each attention head (e.g., 64, 128)
    float scale;         // Attention scale factor (1/sqrt(head_size))
    uint block_size;     // KV cache block size (e.g., 16)
};

// Phase 4.1.1: Basic attention without paging
// Input:  Q [head_size], K [seq_len, head_size], V [seq_len, head_size]
// Output: O [head_size]
// Algorithm:
//   1. Compute attention scores: scores[i] = Q · K[i] * scale
//   2. Softmax: weights[i] = exp(scores[i] - max) / sum(exp(scores - max))
//   3. Weighted sum: O = sum(weights[i] * V[i])

kernel void attention_single_head(
    device const half* q [[buffer(0)]],           // [head_size]
    device const half* k [[buffer(1)]],           // [seq_len, head_size]
    device const half* v [[buffer(2)]],           // [seq_len, head_size]
    device half* output [[buffer(3)]],            // [head_size]
    constant AttentionParams& params [[buffer(4)]],
    threadgroup float* shared_logits [[threadgroup(0)]],  // [seq_len]
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    const uint seq_len = params.seq_len;
    const uint head_size = params.head_size;
    const float scale = params.scale;
    
    // Step 1: Compute Q·K attention scores
    // Each thread computes scores for multiple tokens
    float max_logit = -INFINITY;
    
    for (uint token_idx = tid; token_idx < seq_len; token_idx += threadgroup_size) {
        // Compute dot product: Q · K[token_idx]
        float qk_dot = 0.0f;
        for (uint i = 0; i < head_size; i++) {
            qk_dot += float(q[i]) * float(k[token_idx * head_size + i]);
        }
        
        float logit = qk_dot * scale;
        shared_logits[token_idx] = logit;
        max_logit = max(max_logit, logit);
    }
    
    // Synchronize to ensure all logits are computed
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 2: Find global max logit (for numerical stability)
    // Reduce within simdgroup (32 threads)
    max_logit = simd_max(max_logit);
    
    // Reduce across simdgroups using threadgroup memory
    threadgroup float simdgroup_maxes[32];  // Max 32 simdgroups per threadgroup
    if (lane_id == 0) {
        simdgroup_maxes[simdgroup_id] = max_logit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction (only first simdgroup participates)
    if (simdgroup_id == 0) {
        float global_max = lane_id < (threadgroup_size / 32) ? simdgroup_maxes[lane_id] : -INFINITY;
        global_max = simd_max(global_max);
        if (lane_id == 0) {
            simdgroup_maxes[0] = global_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max_logit = simdgroup_maxes[0];
    
    // Step 3: Compute exp(logit - max) and sum
    float exp_sum = 0.0f;
    for (uint token_idx = tid; token_idx < seq_len; token_idx += threadgroup_size) {
        float logit = shared_logits[token_idx];
        float exp_val = exp(logit - global_max_logit);
        shared_logits[token_idx] = exp_val;
        exp_sum += exp_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 4: Reduce exp_sum across threads
    exp_sum = simd_sum(exp_sum);
    
    threadgroup float simdgroup_sums[32];
    if (lane_id == 0) {
        simdgroup_sums[simdgroup_id] = exp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simdgroup_id == 0) {
        float global_sum = lane_id < (threadgroup_size / 32) ? simdgroup_sums[lane_id] : 0.0f;
        global_sum = simd_sum(global_sum);
        if (lane_id == 0) {
            simdgroup_sums[0] = global_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_exp_sum = simdgroup_sums[0];
    
    // Step 5: Normalize to get softmax weights
    float inv_sum = 1.0f / (global_exp_sum + 1e-6f);
    for (uint token_idx = tid; token_idx < seq_len; token_idx += threadgroup_size) {
        shared_logits[token_idx] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 6: Compute weighted sum: output = sum(softmax[i] * V[i])
    // Each thread computes partial sums for different output dimensions
    for (uint dim_idx = tid; dim_idx < head_size; dim_idx += threadgroup_size) {
        float acc = 0.0f;
        for (uint token_idx = 0; token_idx < seq_len; token_idx++) {
            float weight = shared_logits[token_idx];
            float value = float(v[token_idx * head_size + dim_idx]);
            acc += weight * value;
        }
        output[dim_idx] = half(acc);
    }
}

// Phase 4.1.2: Attention with paged KV cache (to be implemented)
// This will handle non-contiguous memory access via block tables

// Phase 4.1.3: Multi-head attention (to be implemented)
// This will process multiple heads in parallel

// Phase 4.1.4: Grouped Query Attention (to be implemented)
// This will support multiple Q heads sharing KV heads

// Phase 4.1.5: Advanced features (to be implemented)
// - ALiBi positional bias
// - Block-sparse attention
// - FP8 quantization
// - Partitioned attention for long sequences
