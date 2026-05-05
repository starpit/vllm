#include <metal_stdlib>
using namespace metal;

// Phase 4.1.6: Optimized multi-head attention with vectorized loads
// Improvements over base version:
// 1. Vectorized Q·K dot product using half4
// 2. Vectorized V accumulation using half4
// 3. Better memory coalescing
// 4. Reduced register pressure

struct MultiHeadAttentionParams {
    uint seq_len;              // Length of key/value sequence
    uint head_size;            // Dimension of each attention head (must be multiple of 4)
    uint num_heads;            // Number of query heads
    uint num_kv_heads;         // Number of key/value heads (for GQA)
    float scale;               // Attention scale factor (1/sqrt(head_size))
    uint block_size;           // KV cache block size (e.g., 16)
    uint max_num_blocks;       // Maximum number of blocks per sequence
    uint kv_block_stride;      // Stride between blocks in KV cache
    uint kv_head_stride;       // Stride between heads in KV cache
};

// Optimized multi-head paged attention kernel with vectorized loads
// Requirements: head_size must be multiple of 4 for vectorization
kernel void attention_multihead_paged_optimized(
    device const half* q [[buffer(0)]],              // [num_heads, head_size]
    device const half* k_cache [[buffer(1)]],        // [num_kv_heads, num_blocks, head_size, block_size]
    device const half* v_cache [[buffer(2)]],        // [num_kv_heads, num_blocks, head_size, block_size]
    device const int* block_table [[buffer(3)]],     // [max_num_blocks]
    device half* output [[buffer(4)]],               // [num_heads, head_size]
    constant MultiHeadAttentionParams& params [[buffer(5)]],
    threadgroup float* shared_logits [[threadgroup(0)]],  // [seq_len]
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    uint head_idx [[threadgroup_position_in_grid]],
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
    
    const uint num_blocks = (seq_len + block_size - 1) / block_size;
    
    // Pointers to this head's Q and output
    device const half* q_head = q + head_idx * head_size;
    device half* output_head = output + head_idx * head_size;
    
    // Pointer to KV cache for this KV head
    device const half* k_cache_head = k_cache + kv_head_idx * kv_head_stride * num_blocks;
    device const half* v_cache_head = v_cache + kv_head_idx * kv_head_stride * num_blocks;
    
    // Vectorization factor (process 4 elements at a time)
    const uint vec_size = 4;
    const uint head_size_vec = head_size / vec_size;
    
    // Step 1: Compute Q·K attention scores with vectorized loads
    float max_logit = -INFINITY;
    
    for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int physical_block_number = block_table[block_idx];
        
        for (uint block_offset = tid; block_offset < block_size; block_offset += threadgroup_size) {
            const uint token_idx = block_idx * block_size + block_offset;
            
            if (token_idx >= seq_len) {
                continue;
            }
            
            // Vectorized Q·K dot product
            float qk_dot = 0.0f;
            
            // Process 4 elements at a time
            for (uint i = 0; i < head_size_vec; i++) {
                // Load Q vector (4 elements)
                device const half4* q_vec_ptr = (device const half4*)(q_head + i * vec_size);
                half4 q_vec = *q_vec_ptr;
                
                // Load K vector (4 elements) from paged cache
                const uint k_base_idx = physical_block_number * kv_head_stride + i * vec_size * block_size + block_offset;
                half4 k_vec;
                k_vec.x = k_cache_head[k_base_idx];
                k_vec.y = k_cache_head[k_base_idx + block_size];
                k_vec.z = k_cache_head[k_base_idx + 2 * block_size];
                k_vec.w = k_cache_head[k_base_idx + 3 * block_size];
                
                // Dot product of 4 elements
                float4 q_f = float4(q_vec);
                float4 k_f = float4(k_vec);
                qk_dot += dot(q_f, k_f);
            }
            
            float logit = qk_dot * scale;
            shared_logits[token_idx] = logit;
            max_logit = max(max_logit, logit);
        }
    }
    
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
    
    // Step 6: Compute weighted sum with vectorized V access
    // Process 4 output dimensions at a time
    for (uint dim_idx_vec = tid; dim_idx_vec < head_size_vec; dim_idx_vec += threadgroup_size) {
        float4 acc = float4(0.0f);
        
        for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
            const int physical_block_number = block_table[block_idx];
            
            for (uint block_offset = 0; block_offset < block_size; block_offset++) {
                const uint token_idx = block_idx * block_size + block_offset;
                
                if (token_idx >= seq_len) {
                    break;
                }
                
                float weight = shared_logits[token_idx];
                
                // Load V vector (4 elements) from paged cache
                const uint v_base_idx = physical_block_number * kv_head_stride + dim_idx_vec * vec_size * block_size + block_offset;
                half4 v_vec;
                v_vec.x = v_cache_head[v_base_idx];
                v_vec.y = v_cache_head[v_base_idx + block_size];
                v_vec.z = v_cache_head[v_base_idx + 2 * block_size];
                v_vec.w = v_cache_head[v_base_idx + 3 * block_size];
                
                float4 v_f = float4(v_vec);
                acc += weight * v_f;
            }
        }
        
        // Write output (4 elements)
        device half4* output_vec_ptr = (device half4*)(output_head + dim_idx_vec * vec_size);
        *output_vec_ptr = half4(acc);
    }
}

// Variant with tunable threadgroup size for different sequence lengths
// Small sequences (< 256): Use 128 threads
// Medium sequences (256-1024): Use 256 threads
// Large sequences (> 1024): Use 512 threads
