#pragma once

#include "instructions.hpp"
#include "globs.hpp"
#include <torch/torch.h>
#include <functional>

namespace scheduler_cpp {

// Main scheduling functions
std::vector<std::unique_ptr<Instruction>> schedule_model(
    const Globals& globs,
    int device_idx,
    int layer_limit = -1,
    bool interleave_waves = false,
    int interleave_buffer_size = -1,
    const std::string& stop_after_op = "",
    bool disable_lm_head = false,
    bool add_final_sync = true,
    int max_oproj_instructions_per_gpu = -1,
    int num_threads = 1
);

torch::Tensor create_instruction_tensor(
    const Globals& globs,
    int device_idx,
    int layer_limit = -1,
    bool interleave_waves = false,
    int interleave_buffer_size = -1,
    bool move_to_gpu = true,
    const std::string& stop_after_op = "",
    bool zero_init = true,
    bool disable_lm_head = false,
    bool add_final_sync = true,
    int max_oproj_instructions_per_gpu = -1,
    int num_threads = 1
);

std::vector<torch::Tensor> create_all_instruction_tensors(
    const Globals& globs,
    int layer_limit = -1,
    bool interleave_waves = false,
    int interleave_buffer_size = -1,
    bool move_to_gpu = true,
    const std::string& stop_after_op = "",
    bool zero_init = true,
    bool disable_lm_head = false,
    bool add_final_sync = true,
    int max_oproj_instructions_per_gpu = -1,
    int num_threads = 1
);

// Schedule a single layer
std::pair<std::vector<std::vector<std::unique_ptr<Instruction>>>, std::vector<int>> schedule_layer(
    const Globals& globs,
    int device_idx,
    int layer_idx,
    bool interleave_waves,
    const std::string& stop_after_op = "",
    int max_oproj_instructions_per_gpu = -1
);

// Helper functions
torch::Tensor create_instruction_tensor_parallel(
    const Globals& globs,
    int device_idx,
    int layer_limit,
    bool interleave_waves,
    int interleave_buffer_size,
    bool move_to_gpu,
    const std::string& stop_after_op,
    bool zero_init,
    bool disable_lm_head,
    bool add_final_sync,
    int max_oproj_instructions_per_gpu,
    int threads_per_gpu
);

std::vector<std::vector<std::unique_ptr<Instruction>>> round_robin_assign_to_sms(
    std::vector<std::unique_ptr<Instruction>>& instructions, 
    int sm_count
);

torch::Tensor convert_instruction_queues_to_tensor(
    const std::vector<std::vector<std::unique_ptr<Instruction>>>& instruction_queues,
    const std::string& device,
    bool zero_init = true,
    int num_threads = 1
);

std::vector<int> serialize_and_pad(const Instruction& instruction);

// Scheduling subfunctions
std::pair<std::vector<std::unique_ptr<Instruction>>, int> schedule_norm(
    std::function<std::unique_ptr<Instruction>(int, const std::vector<int>&)> func,
    int layer_idx,
    int device_idx,
    const Globals& globs,
    bool interleave_waves
);

std::pair<std::vector<std::unique_ptr<Instruction>>, int> schedule_matmul(
    std::function<std::unique_ptr<Instruction>(int, int, int, int, int)> func,
    int layer_idx,
    int device_idx,
    const Globals& globs,
    int global_outdim,
    bool is_split_batch_dim,
    bool is_split_output_dim,
    int supergroup_size = 8
);

std::pair<std::vector<std::unique_ptr<Instruction>>, int> schedule_matmul_with_limit(
    std::function<std::unique_ptr<Instruction>(int, int, int, int, int)> func,
    int layer_idx,
    int device_idx,
    const Globals& globs,
    int global_outdim,
    bool is_split_batch_dim,
    bool is_split_output_dim,
    int supergroup_size = 8,
    int max_instructions = -1
);

std::pair<std::vector<std::unique_ptr<Instruction>>, int> schedule_attention_decode(
    const Globals& globs,
    int layer_idx
);

std::pair<std::vector<std::unique_ptr<Instruction>>, int> schedule_attention_prefill(
    const Globals& globs,
    int layer_idx,
    int max_instructions = -1
);

std::pair<std::vector<std::unique_ptr<Instruction>>, int> schedule_die(
    const Globals& globs,
    int device_idx
);

std::pair<std::vector<std::unique_ptr<Instruction>>, int> schedule_all_device_barrier(
    const Globals& globs,
    int layer_idx,
    int bar_idx
);

// Interleaving functions
std::vector<std::unique_ptr<Instruction>> interleave_instructions(
    std::vector<std::unique_ptr<Instruction>>& list_a,
    std::vector<std::unique_ptr<Instruction>>& list_b
);

std::vector<std::unique_ptr<Instruction>> interleave_instruction_waves(
    std::vector<std::vector<std::unique_ptr<Instruction>>>& instruction_waves,
    const std::vector<int>& wave_buffer_sizes,
    int overlap_buffer_size
);

} // namespace scheduler_cpp