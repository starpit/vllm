#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include "globs.hpp"

namespace scheduler_cpp {

enum class Pool : uint8_t { None = 0, Compute = 1, Memory = 2 };

static constexpr int INTS_PER_INSTRUCTION = 32;

class Instruction {
public:
    virtual ~Instruction() = default;
    virtual int opcode() const noexcept = 0;
    virtual int prev_opcode() const noexcept = 0;
    virtual Pool pool() const noexcept { return Pool::None; }
    virtual void write32(int* __restrict dst) const noexcept = 0;
    
    // Legacy methods for backwards compatibility - can be removed after full migration
    virtual std::unordered_map<std::string, std::string> tags() const { return {}; }
    virtual std::vector<int> serialize() const { return {}; }
};

class ComputeInstruction : public Instruction {
public:
    Pool pool() const noexcept override { return Pool::Compute; }
    std::unordered_map<std::string, std::string> tags() const override {
        return {{"pool", "compute"}};
    }
};

class MemoryInstruction : public Instruction {
public:
    Pool pool() const noexcept override { return Pool::Memory; }
    std::unordered_map<std::string, std::string> tags() const override {
        return {{"pool", "memory"}};
    }
};

class NormInstruction : public MemoryInstruction {
public:
    int layer_idx;
    std::vector<int> local_batch_indices;
    
    NormInstruction(int layer_idx, const std::vector<int>& local_batch_indices)
        : layer_idx(layer_idx), local_batch_indices(local_batch_indices) {}
    
    void write32(int* __restrict dst) const noexcept override {
        dst[0] = opcode();
        dst[1] = layer_idx;
        dst[2] = static_cast<int>(local_batch_indices.size());
        
        int n = std::min(static_cast<int>(local_batch_indices.size()), INTS_PER_INSTRUCTION - 3);
        for (int i = 0; i < n; i++) {
            dst[3 + i] = local_batch_indices[i];
        }
        // Remaining words left as zero (caller zeros the buffer)
    }
    
    // Legacy serialization for backwards compatibility
    std::vector<int> serialize() const override {
        std::vector<int> words = {opcode(), layer_idx, static_cast<int>(local_batch_indices.size())};
        words.insert(words.end(), local_batch_indices.begin(), local_batch_indices.end());
        return words;
    }
};

class MatMulInstruction : public ComputeInstruction {
public:
    int layer_idx;
    int local_batch_block_idx;
    int local_output_block_idx;
    int global_batch_block_idx;
    int global_output_block_idx;
    
    MatMulInstruction(int layer_idx, int local_batch_block_idx, int local_output_block_idx,
                      int global_batch_block_idx, int global_output_block_idx)
        : layer_idx(layer_idx), local_batch_block_idx(local_batch_block_idx),
          local_output_block_idx(local_output_block_idx),
          global_batch_block_idx(global_batch_block_idx),
          global_output_block_idx(global_output_block_idx) {}
    
    void write32(int* __restrict dst) const noexcept override {
        dst[0] = opcode();
        dst[1] = layer_idx;
        dst[2] = local_batch_block_idx;
        dst[3] = local_output_block_idx;
        dst[4] = global_batch_block_idx;
        dst[5] = global_output_block_idx;
        // Remaining words left as zero (caller zeros the buffer)
    }
    
    // Legacy serialization for backwards compatibility
    std::vector<int> serialize() const override {
        return {opcode(), layer_idx, local_batch_block_idx, local_output_block_idx, 
                global_batch_block_idx, global_output_block_idx};
    }
};

// Specific instruction types
class AttnNorm final : public NormInstruction {
public:
    AttnNorm(int layer_idx, const std::vector<int>& local_batch_indices)
        : NormInstruction(layer_idx, local_batch_indices) {}
    int opcode() const noexcept override { return 1; }
    int prev_opcode() const noexcept override { return 9; } // DownProjResidual
};

class QKV_RopeAppend final : public MatMulInstruction {
public:
    QKV_RopeAppend(int layer_idx, int local_batch_block_idx, int local_output_block_idx,
                   int global_batch_block_idx, int global_output_block_idx)
        : MatMulInstruction(layer_idx, local_batch_block_idx, local_output_block_idx,
                            global_batch_block_idx, global_output_block_idx) {}
    int opcode() const noexcept override { return 2; }
    int prev_opcode() const noexcept override { return 1; } // AttnNorm
};

class AttentionPrefill final : public ComputeInstruction {
public:
    int layer_idx;
    int prefill_seq_idx;
    int prefill_block_idx;
    int prefill_token_offset;
    int kv_head_idx;
    
    AttentionPrefill(int layer_idx, int prefill_seq_idx, int prefill_block_idx, 
                     int prefill_token_offset, int kv_head_idx)
        : layer_idx(layer_idx), prefill_seq_idx(prefill_seq_idx), 
          prefill_block_idx(prefill_block_idx), prefill_token_offset(prefill_token_offset),
          kv_head_idx(kv_head_idx) {}
    int opcode() const noexcept override { return 3; }
    int prev_opcode() const noexcept override { return 2; } // QKV_RopeAppend
    
    void write32(int* __restrict dst) const noexcept override {
        dst[0] = opcode();
        dst[1] = layer_idx;
        dst[2] = prefill_seq_idx;
        dst[3] = prefill_block_idx;
        dst[4] = prefill_token_offset;
        dst[5] = kv_head_idx;
        // Remaining words left as zero (caller zeros the buffer)
    }
    
    // Legacy serialization for backwards compatibility
    std::vector<int> serialize() const override {
        return {opcode(), layer_idx, prefill_seq_idx, prefill_block_idx, 
                prefill_token_offset, kv_head_idx};
    }
};

class AttentionDecode final : public ComputeInstruction {
public:
    int layer_idx;
    std::vector<int> data;
    
    AttentionDecode(int layer_idx, const std::vector<int>& data)
        : layer_idx(layer_idx), data(data) {}
    int opcode() const noexcept override { return 4; }
    int prev_opcode() const noexcept override { return 3; } // AttentionPrefill
    
    void write32(int* __restrict dst) const noexcept override {
        dst[0] = opcode();
        dst[1] = layer_idx;
        dst[2] = static_cast<int>(data.size());
        
        int n = std::min(static_cast<int>(data.size()), INTS_PER_INSTRUCTION - 3);
        for (int i = 0; i < n; i++) {
            dst[3 + i] = data[i];
        }
        // Remaining words left as zero (caller zeros the buffer)
    }
    
    // Legacy serialization for backwards compatibility
    std::vector<int> serialize() const override {
        std::vector<int> words = {opcode(), layer_idx, static_cast<int>(data.size())};
        words.insert(words.end(), data.begin(), data.end());
        return words;
    }
};

class O_ProjResidual final : public MatMulInstruction {
public:
    O_ProjResidual(int layer_idx, int local_batch_block_idx, int local_output_block_idx,
                   int global_batch_block_idx, int global_output_block_idx)
        : MatMulInstruction(layer_idx, local_batch_block_idx, local_output_block_idx,
                            global_batch_block_idx, global_output_block_idx) {}
    int opcode() const noexcept override { return 5; }
    int prev_opcode() const noexcept override { return 4; } // AttentionDecode
};

class MLP_Norm final : public NormInstruction {
public:
    MLP_Norm(int layer_idx, const std::vector<int>& local_batch_indices)
        : NormInstruction(layer_idx, local_batch_indices) {}
    int opcode() const noexcept override { return 6; }
    int prev_opcode() const noexcept override { return 5; } // O_ProjResidual
};

class GateSilu final : public MatMulInstruction {
public:
    GateSilu(int layer_idx, int local_batch_block_idx, int local_output_block_idx,
             int global_batch_block_idx, int global_output_block_idx)
        : MatMulInstruction(layer_idx, local_batch_block_idx, local_output_block_idx,
                            global_batch_block_idx, global_output_block_idx) {}
    int opcode() const noexcept override { return 7; }
    int prev_opcode() const noexcept override { return 6; } // MLP_Norm
};

class UpMatMul final : public MatMulInstruction {
public:
    UpMatMul(int layer_idx, int local_batch_block_idx, int local_output_block_idx,
             int global_batch_block_idx, int global_output_block_idx)
        : MatMulInstruction(layer_idx, local_batch_block_idx, local_output_block_idx,
                            global_batch_block_idx, global_output_block_idx) {}
    int opcode() const noexcept override { return 8; }
    int prev_opcode() const noexcept override { return 7; } // GateSilu
};

class DownProjResidual final : public MatMulInstruction {
public:
    DownProjResidual(int layer_idx, int local_batch_block_idx, int local_output_block_idx,
                     int global_batch_block_idx, int global_output_block_idx)
        : MatMulInstruction(layer_idx, local_batch_block_idx, local_output_block_idx,
                            global_batch_block_idx, global_output_block_idx) {}
    int opcode() const noexcept override { return 9; }
    int prev_opcode() const noexcept override { return 8; } // UpMatMul
};

class LM_Head_Norm final : public NormInstruction {
public:
    LM_Head_Norm(int layer_idx, const std::vector<int>& local_batch_indices)
        : NormInstruction(layer_idx, local_batch_indices) {}
    int opcode() const noexcept override { return 10; }
    int prev_opcode() const noexcept override { return 9; } // DownProjResidual
};

class LM_Head final : public MatMulInstruction {
public:
    LM_Head(int layer_idx, int local_batch_block_idx, int local_output_block_idx,
            int global_batch_block_idx, int global_output_block_idx)
        : MatMulInstruction(layer_idx, local_batch_block_idx, local_output_block_idx,
                            global_batch_block_idx, global_output_block_idx) {}
    int opcode() const noexcept override { return 11; }
    int prev_opcode() const noexcept override { return 10; } // LM_Head_Norm
};

class IncBarrier final : public MemoryInstruction {
public:
    int opcode() const noexcept override { return 12; }
    int prev_opcode() const noexcept override { return 0; } // No previous instruction needed
    
    void write32(int* __restrict dst) const noexcept override {
        dst[0] = opcode();
        // Remaining words left as zero (caller zeros the buffer)
    }
    
    // Legacy serialization for backwards compatibility
    std::vector<int> serialize() const override { return {opcode()}; }
};

class Die final : public ComputeInstruction {
public:
    int opcode() const noexcept override { return -1; }
    int prev_opcode() const noexcept override { return 0; } // No previous instruction needed
    
    void write32(int* __restrict dst) const noexcept override {
        dst[0] = opcode();
        // Remaining words left as zero (caller zeros the buffer)
    }
    
    // Legacy serialization for backwards compatibility
    std::vector<int> serialize() const override { return {opcode()}; }
};

class AllDeviceBarrier final : public MemoryInstruction {
public:
    int layer_idx;
    int bar_idx;
    
    AllDeviceBarrier(int layer_idx, int bar_idx)
        : layer_idx(layer_idx), bar_idx(bar_idx) {}
    
    int opcode() const noexcept override { return 13; }
    int prev_opcode() const noexcept override { return 0; } // No previous instruction needed
    
    void write32(int* __restrict dst) const noexcept override {
        dst[0] = opcode();
        dst[1] = layer_idx;
        dst[2] = bar_idx;
        // Remaining words left as zero (caller zeros the buffer)
    }
    
    // Legacy serialization for backwards compatibility
    std::vector<int> serialize() const override { 
        return {opcode(), layer_idx, bar_idx}; 
    }
};

class NoOp final : public Instruction {
public:
    int opcode() const noexcept override { return 0; }
    int prev_opcode() const noexcept override { return 0; }
    Pool pool() const noexcept override { return Pool::None; }
    
    void write32(int* __restrict dst) const noexcept override {
        dst[0] = 0;
        // Remaining words left as zero (caller zeros the buffer)
    }
    
    // Legacy methods for backwards compatibility
    std::unordered_map<std::string, std::string> tags() const override { return {}; }
    std::vector<int> serialize() const override { return {0}; }
};

} // namespace scheduler_cpp