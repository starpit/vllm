#include <metal_stdlib>
using namespace metal;

// Phase 4.1.4: Attention with paged KV cache
// Handles non-contiguous memory access via block tables
// Based on CUDA implementation in csrc/attention/attention_kernels.cuh

struct PagedAttentionParams {
    uint seq_len;              // Length of key/value sequence
    uint head_size;            // Dimension of each attention head (e.g., 64, 128)
    float scale;               // Attention scale factor (1/sqrt(head_size))
    uint block_size;           // KV cache block size (e.g., 16)
    uint max_num_blocks;       // Maximum number of blocks per sequence
    uint kv_block_stride;      // Stride between blocks in KV cache
    uint kv_head_stride;       // Stride between heads in KV cache
};

// Paged attention kernel with block table lookup
// Input:  Q [head_size]
//         K [num_blocks, head_size/x, block_size, x] (paged)
//         V [num_blocks, head_size, block_size] (paged)
//         block_table [max_num_blocks] (maps logical -> physical blocks)
// Output: O [head_size]

kernel void attention_paged_single_head(
    device const half* q [[buffer(0)]],              // [head_size]
    device const half* k_cache [[buffer(1)]],        // [num_blocks, head_size/x, block_size, x]
    device const half* v_cache [[buffer(2)]],        // [num_blocks, head_size, block_size]
    device const int* block_table [[buffer(3)]],     // [max_num_blocks]
    device half* output [[buffer(4)]],               // [head_size]
    constant PagedAttentionParams& params [[buffer(5)]],
    threadgroup float* shared_logits [[threadgroup(0)]],  // [seq_len]
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    const uint seq_len = params.seq_len;
    const uint head_size = params.head_size;
    const float scale = params.scale;
    const uint block_size = params.block_size;
    const uint kv_block_stride = params.kv_block_stride;
    const uint kv_head_stride = params.kv_head_stride;
    
    // Calculate number of blocks needed for this sequence
    const uint num_blocks = (seq_len + block_size - 1) / block_size;
    
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
            
            // Compute K pointer for this token in paged cache
            // Layout: k_cache[physical_block][head_dim][block_offset]
            const uint k_base = physical_block_number * kv_block_stride + block_offset;
            
            // Compute dot product: Q · K[token_idx]
            float qk_dot = 0.0f;
            for (uint i = 0; i < head_size; i++) {
                // K cache layout: [num_blocks, head_size, block_size]
                const uint k_idx = physical_block_number * kv_head_stride + i * block_size + block_offset;
                qk_dot += float(q[i]) * float(k_cache[k_idx]);
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
                
                // V cache layout: [num_blocks, head_size, block_size]
                const uint v_idx = physical_block_number * kv_head_stride + dim_idx * block_size + block_offset;
                float value = float(v_cache[v_idx]);
                
                acc += weight * value;
            }
        }
        
        output[dim_idx] = half(acc);
    }
}

// Multi-head paged attention (Phase 4.1.5 - to be implemented)
// This will process multiple heads in parallel

// Grouped Query Attention (Phase 4.1.5 - to be implemented)
// This will support multiple Q heads sharing KV heads
