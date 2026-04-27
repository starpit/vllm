#pragma once

#include <vector>

namespace scheduler_cpp {

struct Globals {
    // Model configuration
    int num_hidden_layers;
    int num_attention_heads;
    int num_kv_heads;
    int head_dim;
    int hidden_size;
    int intermediate_size;
    int vocab_size;
    
    float attn_scale;
    float rms_norm_eps;
    
    int tp_size;
    int tp_rank;
    int global_batch_size;
    int matmul_batch_block_size;
    int matmul_output_block_size;
    int prefill_block_size;
    
    bool global_work_queue_enabled;
    int sm_count;
    
    // Paged KV cache parameters
    int page_size;
    int num_pages;
    bool timing_record_enabled;
    
    // Prefill chunk lengths (renamed from prefill_seq_lens to match Python)
    std::vector<int> prefill_chunk_lens;
    
    // Prefill extend offsets (for seq extension, e.g. after a prefill cache hit)
    std::vector<int> prefill_extend_offsets;
    
    // Device batch sizes
    std::vector<int> device_batch_size;
    
    // Derived properties
    int num_combined_heads() const {
        return num_attention_heads + num_kv_heads * 2;
    }
    
    int local_batch_size(int device_idx) const {
        if (device_idx < static_cast<int>(device_batch_size.size())) {
            return device_batch_size[device_idx];
        }
        // Fallback if device_batch_size not set
        return global_batch_size / tp_size;
    }
    
    int local_batch_blocks(int device_idx) const {
        int num_blocks = (global_batch_size + matmul_batch_block_size - 1) / matmul_batch_block_size;
        int div = num_blocks / tp_size;
        int remainder = num_blocks % tp_size;
        
        int num_blocks_for_this_rank = div + (device_idx < remainder ? 1 : 0);
        return num_blocks_for_this_rank;
    }
    
    int qkv_dim() const {
        return (num_attention_heads + num_kv_heads * 2) * head_dim;
    }
    
    int num_batch_blocks() const {
        return (global_batch_size + matmul_batch_block_size - 1) / matmul_batch_block_size;
    }
    
    int num_output_blocks() const {
        return hidden_size / matmul_output_block_size;
    }
    
    int num_intermediate_blocks() const {
        return intermediate_size / matmul_output_block_size;
    }
    
    int num_vocab_blocks() const {
        return vocab_size / matmul_output_block_size;
    }
    
    int num_prefill_tokens() const {
        int total = 0;
        for (int len : prefill_chunk_lens) {
            total += len;
        }
        return total;
    }
    
    int num_decode_seqs() const {
        return global_batch_size - num_prefill_tokens();
    }
};

} // namespace scheduler_cpp