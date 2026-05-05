#include <metal_stdlib>
using namespace metal;

// Phase 4.1.5: Multi-head attention with Grouped Query Attention (GQA)
// Extends paged attention to support multiple query heads and KV head sharing

struct MultiHeadAttentionParams {
    uint seq_len;              // Length of key/value sequence
    uint head_size;            // Dimension of each attention head (e.g., 64, 128)
    uint num_heads;            // Number of query heads
    uint num_kv_heads;         // Number of key/value heads (for GQA)
    float scale;               // Attention scale factor (1/sqrt(head_size))
    uint block_size;           // KV cache block size (e.g., 16)
    uint max_num_blocks;       // Maximum number of blocks per sequence
    uint kv_block_stride;      // Stride between blocks in KV cache
    uint kv_head_stride;       // Stride between heads in KV cache
};

// Multi-head paged attention kernel with GQA support
// Input:  Q [num_heads, head_size]
//         K [num_kv_heads, num_blocks, head_size/x, block_size, x] (paged)
//         V [num_kv_heads, num_blocks, head_size, block_size] (paged)
//         block_table [max_num_blocks] (maps logical -> physical blocks)
// Output: O [num_heads, head_size]
//
// GQA: Multiple query heads share the same KV heads
// Example: num_heads=32, num_kv_heads=8 -> 4 Q heads per KV head

kernel void attention_multihead_paged(
    device const half* q [[buffer(0)]],              // [num_heads, head_size]
    device const half* k_cache [[buffer(1)]],        // [num_kv_heads, num_blocks, head_size, block_size]
    device const half* v_cache [[buffer(2)]],        // [num_kv_heads, num_blocks, head_size, block_size]
    device const int* block_table [[buffer(3)]],     // [max_num_blocks]
    device half* output [[buffer(4)]],               // [num_heads, head_size]
    constant MultiHeadAttentionParams& params [[buffer(5)]],
    threadgroup float* shared_logits [[threadgroup(0)]],  // [seq_len]
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    uint head_idx [[threadgroup_position_in_grid]],  // Which head this threadgroup processes
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    const uint seq_len = params.seq_len;
    const uint head_size = params.head_size;
    const uint num_heads = params.num_heads;
    const uint num_kv_heads = params.num_kv_heads;
    const float scale = params.scale;
    const uint block_size = params.block_size;
    const uint kv_block_stride = params.kv_block_stride;
    const uint kv_head_stride = params.kv_head_stride;
    
    // GQA: Calculate which KV head this query head uses
    const uint num_queries_per_kv = num_heads / num_kv_heads;
    const uint kv_head_idx = head_idx / num_queries_per_kv;
    
    // Calculate number of blocks needed for this sequence
    const uint num_blocks = (seq_len + block_size - 1) / block_size;
    
    // Pointers to this head's Q and output
    device const half* q_head = q + head_idx * head_size;
    device half* output_head = output + head_idx * head_size;
    
    // Pointer to KV cache for this KV head
    device const half* k_cache_head = k_cache + kv_head_idx * kv_head_stride * num_blocks;
    device const half* v_cache_head = v_cache + kv_head_idx * kv_head_stride * num_blocks;
    
    // Step 1: Compute Q·K attention scores with paged access
    float max_logit = -INFINITY;
    
    // Iterate over blocks
    for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
        // Look up physical block number from block table
        const int physical_block_number = block_table[block_idx];
        
        // Iterate over tokens within this block
        for (uint block_offset = tid; block_offset < block_size; block_offset += threadgroup_size) {
            const uint token_idx = block_idx * block_size + block_offset;
            
            // Skip if beyond sequence length
            if (token_idx >= seq_len) {
                continue;
            }
            
            // Compute dot product: Q · K[token_idx]
            float qk_dot = 0.0f;
            for (uint i = 0; i < head_size; i++) {
                // K cache layout: [num_kv_heads, num_blocks, head_size, block_size]
                const uint k_idx = physical_block_number * kv_head_stride + i * block_size + block_offset;
                qk_dot += float(q_head[i]) * float(k_cache_head[k_idx]);
            }
            
            float logit = qk_dot * scale;
            shared_logits[token_idx] = logit;
            max_logit = max(max_logit, logit);
        }
    }
    
    // Synchronize to ensure all logits are computed
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 2: Find global max logit (for numerical stability)
    max_logit = simd_max(max_logit);
    
    threadgroup float simdgroup_maxes[32];
    if (lane_id == 0) {
        simdgroup_maxes[simdgroup_id] = max_logit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
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
    
    // Step 6: Compute weighted sum with paged V access: output = sum(softmax[i] * V[i])
    for (uint dim_idx = tid; dim_idx < head_size; dim_idx += threadgroup_size) {
        float acc = 0.0f;
        
        // Iterate over blocks
        for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
            const int physical_block_number = block_table[block_idx];
            
            // Iterate over tokens within this block
            for (uint block_offset = 0; block_offset < block_size; block_offset++) {
                const uint token_idx = block_idx * block_size + block_offset;
                
                // Skip if beyond sequence length
                if (token_idx >= seq_len) {
                    break;
                }
                
                float weight = shared_logits[token_idx];
                
                // V cache layout: [num_kv_heads, num_blocks, head_size, block_size]
                const uint v_idx = physical_block_number * kv_head_stride + dim_idx * block_size + block_offset;
                float value = float(v_cache_head[v_idx]);
                
                acc += weight * value;
            }
        }
        
        output_head[dim_idx] = half(acc);
    }
}

// Optimized version with vectorized loads (Phase 4.1.6 - future optimization)
// This will use float4/half4 for better memory bandwidth utilization
