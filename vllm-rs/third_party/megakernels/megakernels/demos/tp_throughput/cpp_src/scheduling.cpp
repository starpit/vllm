#include "scheduling.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <numeric>
#include <thread>

namespace scheduler_cpp {

constexpr int TIMING_SLOTS = 128;

std::vector<int> serialize_and_pad(const Instruction& instruction) {
    auto serialized = instruction.serialize();
    int num_padding =
        INTS_PER_INSTRUCTION - static_cast<int>(serialized.size());
    if (num_padding < 0) {
        throw std::runtime_error("Instruction serialization too long");
    }
    serialized.resize(INTS_PER_INSTRUCTION, 0);
    return serialized;
}

std::vector<std::pair<int, int>> make_supergroup(const std::vector<int>& iters1,
                                                 const std::vector<int>& iters2,
                                                 int group_size) {
    int num_iters1 = static_cast<int>(iters1.size());
    int num_iters2 = static_cast<int>(iters2.size());
    
    // Pre-reserve capacity to avoid reallocations
    int num_groups = (num_iters1 + group_size - 1) / group_size;
    int per_group = std::min(group_size, num_iters1) * num_iters2;
    std::vector<std::pair<int, int>> vals;
    vals.reserve(static_cast<size_t>(num_groups) * static_cast<size_t>(per_group));
    for (int group = 0; group < num_groups; ++group) {
        int start_in_group = group * group_size;
        int end_in_group = std::min(start_in_group + group_size, num_iters1);
        for (int j = 0; j < num_iters2; ++j) {
            for (int i = start_in_group; i < end_in_group; ++i) {
                vals.emplace_back(iters1[i], iters2[j]);
            }
        }
    }
    return vals;
}

std::pair<std::vector<std::unique_ptr<Instruction>>, int>
schedule_norm_single_wave(
    std::function<std::unique_ptr<Instruction>(int, const std::vector<int>&)>
        func,
    int layer_idx, int device_idx, const Globals& globs) {
    std::vector<std::vector<int>> round_robin_queues(globs.sm_count);

    for (int bidx = 0; bidx < globs.local_batch_size(device_idx); ++bidx) {
        int sm_idx = bidx % globs.sm_count;
        round_robin_queues[sm_idx].push_back(bidx);
    }

    int max_vecs_per_inst = std::min(6 * 32768 / (globs.hidden_size * 2), 12);

    std::vector<std::unique_ptr<Instruction>> instructions;
    for (int sm_idx = 0; sm_idx < globs.sm_count; ++sm_idx) {
        const auto& queue = round_robin_queues[sm_idx];
        int queue_size = static_cast<int>(queue.size());

        for (int start_idx = 0; start_idx < queue_size;
             start_idx += max_vecs_per_inst) {
            int end_idx = std::min(start_idx + max_vecs_per_inst, queue_size);

            std::vector<int> local_batch_indices(queue.begin() + start_idx,
                                                 queue.begin() + end_idx);
            instructions.push_back(func(layer_idx, local_batch_indices));
        }
    }

    return {std::move(instructions), globs.sm_count};
}

std::pair<std::vector<std::unique_ptr<Instruction>>, int>
schedule_norm_multi_wave(
    std::function<std::unique_ptr<Instruction>(int, const std::vector<int>&)>
        func,
    int layer_idx, int device_idx, const Globals& globs) {
    std::vector<std::unique_ptr<Instruction>> instructions;
    for (int batch_idx = 0; batch_idx < globs.local_batch_size(device_idx);
         ++batch_idx) {
        instructions.push_back(func(layer_idx, {batch_idx}));
    }
    return {std::move(instructions), globs.matmul_batch_block_size};
}

std::pair<std::vector<std::unique_ptr<Instruction>>, int> schedule_norm(
    std::function<std::unique_ptr<Instruction>(int, const std::vector<int>&)>
        func,
    int layer_idx, int device_idx, const Globals& globs,
    bool interleave_waves) {
    if (interleave_waves) {
        return schedule_norm_multi_wave(func, layer_idx, device_idx, globs);
    } else {
        return schedule_norm_single_wave(func, layer_idx, device_idx, globs);
    }
}

std::pair<std::vector<std::unique_ptr<Instruction>>, int> schedule_matmul(
    std::function<std::unique_ptr<Instruction>(int, int, int, int, int)> func,
    int layer_idx, int device_idx, const Globals& globs, int global_outdim,
    bool is_split_batch_dim, bool is_split_output_dim, int supergroup_size) {
    
    // Pre-calculate sizes for reservation
    int num_local_batch_blocks;
    if (is_split_batch_dim) {
        num_local_batch_blocks = globs.local_batch_blocks(device_idx);
    } else {
        num_local_batch_blocks = globs.num_batch_blocks();
    }
    
    int output_limit = global_outdim;
    if (is_split_output_dim) {
        output_limit = output_limit / globs.tp_size;
    }
    int num_local_output_blocks = output_limit / globs.matmul_output_block_size;
    
    std::vector<std::unique_ptr<Instruction>> instructions;
    instructions.reserve(static_cast<size_t>(num_local_batch_blocks) * static_cast<size_t>(num_local_output_blocks));


    std::vector<int> local_batch_idx_range;
    local_batch_idx_range.resize(num_local_batch_blocks);
    std::iota(local_batch_idx_range.begin(), local_batch_idx_range.end(), 0);

    std::vector<int> local_output_idx_range;
    local_output_idx_range.resize(num_local_output_blocks);
    std::iota(local_output_idx_range.begin(), local_output_idx_range.end(), 0);

    auto pairs = make_supergroup(local_batch_idx_range, local_output_idx_range,
                                 supergroup_size);

    for (const auto& [local_batch_idx, local_output_idx] : pairs) {
        int global_batch_block_idx;
        if (is_split_batch_dim) {
            global_batch_block_idx =
                num_local_batch_blocks * device_idx + local_batch_idx;
        } else {
            global_batch_block_idx = local_batch_idx;
        }

        int global_output_block_idx;
        if (is_split_output_dim) {
            global_output_block_idx =
                num_local_output_blocks * device_idx + local_output_idx;
        } else {
            global_output_block_idx = local_output_idx;
        }

        instructions.push_back(func(layer_idx, local_batch_idx,
                                    local_output_idx, global_batch_block_idx,
                                    global_output_block_idx));
    }

    int buffer_size = std::min(static_cast<int>(local_batch_idx_range.size()),
                               supergroup_size) *
                      static_cast<int>(local_output_idx_range.size());

    return {std::move(instructions), buffer_size};
}

std::pair<std::vector<std::unique_ptr<Instruction>>, int> schedule_matmul_with_limit(
    std::function<std::unique_ptr<Instruction>(int, int, int, int, int)> func,
    int layer_idx, int device_idx, const Globals& globs, int global_outdim,
    bool is_split_batch_dim, bool is_split_output_dim, int supergroup_size, int max_instructions) {

    // Pre-calculate sizes for reservation
    int num_local_batch_blocks;
    if (is_split_batch_dim) {
        num_local_batch_blocks = globs.local_batch_blocks(device_idx);
    } else {
        num_local_batch_blocks = globs.num_batch_blocks();
    }

    int output_limit = global_outdim;
    if (is_split_output_dim) {
        output_limit = output_limit / globs.tp_size;
    }
    int num_local_output_blocks = output_limit / globs.matmul_output_block_size;

    std::vector<std::unique_ptr<Instruction>> instructions;
    instructions.reserve(static_cast<size_t>(num_local_batch_blocks) * static_cast<size_t>(num_local_output_blocks));

    std::vector<int> local_batch_idx_range;
    local_batch_idx_range.resize(num_local_batch_blocks);
    std::iota(local_batch_idx_range.begin(), local_batch_idx_range.end(), 0);

    std::vector<int> local_output_idx_range;
    local_output_idx_range.resize(num_local_output_blocks);
    std::iota(local_output_idx_range.begin(), local_output_idx_range.end(), 0);

    auto pairs = make_supergroup(local_batch_idx_range, local_output_idx_range,
                                 supergroup_size);

    int instruction_count = 0;
    for (const auto& [local_batch_idx, local_output_idx] : pairs) {
        // Check if we've reached the maximum number of instructions
        if (max_instructions != -1 && instruction_count >= max_instructions) {
            break;
        }

        int global_batch_block_idx;
        if (is_split_batch_dim) {
            global_batch_block_idx =
                num_local_batch_blocks * device_idx + local_batch_idx;
        } else {
            global_batch_block_idx = local_batch_idx;
        }

        int global_output_block_idx;
        if (is_split_output_dim) {
            global_output_block_idx =
                num_local_output_blocks * device_idx + local_output_idx;
        } else {
            global_output_block_idx = local_output_idx;
        }

        instructions.push_back(func(layer_idx, local_batch_idx,
                                    local_output_idx, global_batch_block_idx,
                                    global_output_block_idx));
        instruction_count++;
    }

    int buffer_size = std::min(static_cast<int>(local_batch_idx_range.size()),
                               supergroup_size) *
                      static_cast<int>(local_output_idx_range.size());

    return {std::move(instructions), buffer_size};
}

std::pair<std::vector<std::unique_ptr<Instruction>>, int>
schedule_attention_decode(const Globals& globs, int layer_idx) {
    int local_kv_heads = globs.num_kv_heads / globs.tp_size;
    int group_size = 8;
    
    // Pre-reserve based on expected instruction count
    const int max_pairs = globs.num_decode_seqs() * local_kv_heads;
    std::vector<std::unique_ptr<Instruction>> instructions;
    instructions.reserve((max_pairs + group_size - 1) / group_size);

    std::vector<int> data;
    data.reserve(group_size * 2); // Each pair is 2 ints

    for (int seq_idx = 0; seq_idx < globs.num_decode_seqs(); ++seq_idx) {
        for (int kv_head_idx = 0; kv_head_idx < local_kv_heads; ++kv_head_idx) {
            data.push_back(seq_idx);
            data.push_back(kv_head_idx);

            if (static_cast<int>(data.size()) == group_size * 2) {
                instructions.push_back(
                    std::make_unique<AttentionDecode>(layer_idx, data));
                data.clear();
            }
        }
    }

    // Handle remaining data that doesn't fill a complete group
    if (!data.empty()) {
        instructions.push_back(
            std::make_unique<AttentionDecode>(layer_idx, data));
    }

    // TODO need to update to account for prefill
    int buffer_size = static_cast<int>(std::ceil(
        static_cast<double>(globs.matmul_batch_block_size * local_kv_heads) /
        group_size));

    return {std::move(instructions), buffer_size};
}

std::pair<std::vector<std::unique_ptr<Instruction>>, int>
schedule_attention_prefill(const Globals& globs, int layer_idx, int max_instructions) {
    std::vector<std::unique_ptr<Instruction>> instructions;

    int local_kv_heads = globs.num_kv_heads / globs.tp_size;
    int buffer_size = 0;
    int cumsum_tokens_processed = 0;

    int prefill_block_size = globs.prefill_block_size;

    for (size_t prefill_seq_idx = 0;
         prefill_seq_idx < globs.prefill_chunk_lens.size(); ++prefill_seq_idx) {
        int prefill_chunk_len = globs.prefill_chunk_lens[prefill_seq_idx];
        int prefill_extend_offset = (prefill_seq_idx < globs.prefill_extend_offsets.size()) 
            ? globs.prefill_extend_offsets[prefill_seq_idx] : 0;
        
        int num_prefill_blocks = static_cast<int>(std::ceil(
            static_cast<double>(prefill_chunk_len) / prefill_block_size));

        // TODO is this the best nesting of these two loops? (shouldn't matter
        // for L70B on 8GPUs here since theres only 1 kv head)
        for (int prefill_block_idx = 0; prefill_block_idx < num_prefill_blocks;
             ++prefill_block_idx) {
            int block_start_idx = prefill_block_idx * prefill_block_size;
            int block_end_idx =
                std::min(block_start_idx + prefill_block_size, prefill_chunk_len);

            int tokens_in_block = block_end_idx - block_start_idx;

            for (int kv_head_idx = 0; kv_head_idx < local_kv_heads;
                 ++kv_head_idx) {
                // Check if we've reached the maximum number of instructions
                if (max_instructions != -1 && static_cast<int>(instructions.size()) >= max_instructions) {
                    return {std::move(instructions), buffer_size};
                }

                instructions.push_back(std::make_unique<AttentionPrefill>(
                    layer_idx, static_cast<int>(prefill_seq_idx),
                    prefill_block_idx, prefill_extend_offset, kv_head_idx));

                if (cumsum_tokens_processed < globs.matmul_batch_block_size) {
                    buffer_size += 1;
                }
            }

            cumsum_tokens_processed += tokens_in_block;
        }
    }

    return {std::move(instructions), buffer_size};
}

std::pair<std::vector<std::unique_ptr<Instruction>>, int> schedule_die(
    const Globals& globs, int device_idx) {
    std::vector<std::unique_ptr<Instruction>> instructions;
    for (int sm_idx = 0; sm_idx < globs.sm_count; ++sm_idx) {
        instructions.push_back(std::make_unique<Die>());
    }
    return {std::move(instructions), static_cast<int>(instructions.size())};
}

std::pair<std::vector<std::unique_ptr<Instruction>>, int> schedule_all_device_barrier(
    const Globals& globs, int layer_idx, int bar_idx) {
    std::vector<std::unique_ptr<Instruction>> instructions;
    for (int sm_idx = 0; sm_idx < globs.sm_count; ++sm_idx) {
        instructions.push_back(std::make_unique<AllDeviceBarrier>(layer_idx, bar_idx));
    }
    return {std::move(instructions), static_cast<int>(instructions.size())};
}

// Interleave two instruction lists supporting variable lengths/ratios
std::vector<std::unique_ptr<Instruction>> interleave_instructions(
    std::vector<std::unique_ptr<Instruction>>& list_a,
    std::vector<std::unique_ptr<Instruction>>& list_b) {
    std::vector<std::unique_ptr<Instruction>>* shorter_list;
    std::vector<std::unique_ptr<Instruction>>* longer_list;

    if (list_a.size() < list_b.size()) {
        shorter_list = &list_a;
        longer_list = &list_b;
    } else {
        shorter_list = &list_b;
        longer_list = &list_a;
    }

    if (shorter_list->empty()) {
        std::vector<std::unique_ptr<Instruction>> result;
        result.reserve(longer_list->size());
        for (auto& inst : *longer_list) {
            result.push_back(std::move(inst));
        }
        longer_list->clear();
        return result;
    }

    std::vector<std::unique_ptr<Instruction>> combined_list;
    combined_list.reserve(list_a.size() + list_b.size());

    double ratio =
        static_cast<double>(longer_list->size()) / shorter_list->size();

    for (size_t i = 0; i < shorter_list->size(); ++i) {
        combined_list.push_back(std::move((*shorter_list)[i]));

        int longer_start = static_cast<int>(std::round(i * ratio));
        int longer_end = static_cast<int>(std::round((i + 1) * ratio));

        for (int j = longer_start;
             j < longer_end && j < static_cast<int>(longer_list->size()); ++j) {
            combined_list.push_back(std::move((*longer_list)[j]));
        }
    }

    int remaining_start =
        static_cast<int>(std::round(shorter_list->size() * ratio));
    for (int i = remaining_start; i < static_cast<int>(longer_list->size());
         ++i) {
        combined_list.push_back(std::move((*longer_list)[i]));
    }

    shorter_list->clear();
    longer_list->clear();

    return combined_list;
}

// Interleave instruction waves - complex algorithm for overlapping computation
std::vector<std::unique_ptr<Instruction>> interleave_instruction_waves(
    std::vector<std::vector<std::unique_ptr<Instruction>>>& instruction_waves,
    const std::vector<int>& wave_buffer_sizes, int overlap_buffer_size) {
    if (instruction_waves.empty()) {
        return {};
    }

    std::vector<int> wave_types;
    std::vector<int> wave_sizes;
    
    wave_types.reserve(instruction_waves.size());
    wave_sizes.reserve(instruction_waves.size());

    for (const auto& wave : instruction_waves) {
        if (!wave.empty()) {
            wave_types.push_back(static_cast<int>(wave[0]->pool()));
            wave_sizes.push_back(static_cast<int>(wave.size()));
        } else {
            wave_types.push_back(static_cast<int>(Pool::None));
            wave_sizes.push_back(0);
        }
    }

    int num_waves = static_cast<int>(instruction_waves.size());

    auto get_buffer_size = [&](int wave_idx) {
        return wave_buffer_sizes[wave_idx] + overlap_buffer_size;
    };

    // Calculate wave partitions
    std::vector<std::tuple<int, int, int>> wave_partitions;
    for (int wave_idx = 0; wave_idx < num_waves; ++wave_idx) {
        int wave_size = wave_sizes[wave_idx];
        int wave_buffer_size = get_buffer_size(wave_idx);

        int max_overlappable = std::max(0, wave_size - wave_buffer_size);
        if (max_overlappable == 0) {
            wave_partitions.emplace_back(0, wave_size, 0);
            continue;
        }

        int prev_wave_type =
            (wave_idx == 0) ? static_cast<int>(Pool::None) : wave_types[wave_idx - 1];
        int next_wave_type =
            (wave_idx == num_waves - 1) ? static_cast<int>(Pool::None) : wave_types[wave_idx + 1];
        int wave_type = wave_types[wave_idx];

        bool prev_wave_diff =
            (prev_wave_type != static_cast<int>(Pool::None)) && (prev_wave_type != wave_type);
        bool next_wave_diff =
            (next_wave_type != static_cast<int>(Pool::None)) && (next_wave_type != wave_type);

        int usable_prev_size = 0;
        if (prev_wave_diff) {
            int prev_size = wave_sizes[wave_idx - 1];
            int prev_buffer_size = get_buffer_size(wave_idx - 1);
            usable_prev_size = std::max(prev_size - prev_buffer_size, 0);
        }

        int usable_next_size = 0;
        if (next_wave_diff) {
            int next_size = wave_sizes[wave_idx + 1];
            int next_buffer_size = get_buffer_size(wave_idx + 1);
            usable_next_size = std::max(next_size - next_buffer_size, 0);
        }

        int denom = usable_prev_size + usable_next_size;
        if (denom == 0) denom = 1;

        int first_part = std::min(
            static_cast<int>(std::round(
                static_cast<double>(wave_size * usable_prev_size) / denom)),
            max_overlappable);
        int last_part = std::min(
            static_cast<int>(std::round(
                static_cast<double>(wave_size * usable_next_size) / denom)),
            max_overlappable);

        int middle_part = wave_size - first_part - last_part;

        wave_partitions.emplace_back(first_part, middle_part, last_part);
    }

    // Build serialized queue
    std::vector<std::unique_ptr<Instruction>> serialized_queue;
    std::vector<int> num_used_per_wave(num_waves, 0);

    auto extract_instructions =
        [&](int wave_idx,
            int num_instructions) -> std::vector<std::unique_ptr<Instruction>> {
        std::vector<std::unique_ptr<Instruction>> extracted;
        int num_used = num_used_per_wave[wave_idx];

        for (int i = 0; i < num_instructions; ++i) {
            extracted.push_back(
                std::move(instruction_waves[wave_idx][num_used + i]));
        }
        num_used_per_wave[wave_idx] += num_instructions;
        return extracted;
    };

    auto assign_instructions = [&](int wave_idx, int num_instructions) {
        auto extracted = extract_instructions(wave_idx, num_instructions);
        for (auto& inst : extracted) {
            serialized_queue.push_back(std::move(inst));
        }
    };

    auto assign_double = [&](int wave_idx1, int size1, int wave_idx2,
                             int size2) {
        auto extracted1 = extract_instructions(wave_idx1, size1);
        auto extracted2 = extract_instructions(wave_idx2, size2);
        auto interleaved = interleave_instructions(extracted1, extracted2);
        for (auto& inst : interleaved) {
            serialized_queue.push_back(std::move(inst));
        }
    };

    for (int wave_idx = 0; wave_idx < num_waves; ++wave_idx) {
        auto [amount_pre, amount_middle, amount_post] =
            wave_partitions[wave_idx];

        if (wave_idx == 0) {
            assign_instructions(wave_idx, amount_pre + amount_middle);
        } else {
            assign_instructions(wave_idx, amount_middle);
        }

        if (wave_idx < num_waves - 1) {
            auto [next_amount_pre, next_amount_middle, next_amount_post] =
                wave_partitions[wave_idx + 1];
            assign_double(wave_idx, amount_post, wave_idx + 1, next_amount_pre);
        } else {
            assign_instructions(wave_idx, amount_post);
        }
    }

    return serialized_queue;
}

// Schedule a single layer and return its waves and buffer sizes
std::pair<std::vector<std::vector<std::unique_ptr<Instruction>>>,
          std::vector<int>>
schedule_layer(const Globals& globs, int device_idx, int layer_idx,
               bool interleave_waves, const std::string& stop_after_op, int max_oproj_instructions_per_gpu) {
    std::vector<std::vector<std::unique_ptr<Instruction>>> layer_waves;
    std::vector<int> layer_buffer_sizes;

    // AttnNorm
    {
        auto norm_func = [](int layer_idx,
                            const std::vector<int>& local_batch_indices)
            -> std::unique_ptr<Instruction> {
            return std::make_unique<AttnNorm>(layer_idx, local_batch_indices);
        };
        auto [instructions, buffer_size] = schedule_norm(
            norm_func, layer_idx, device_idx, globs, interleave_waves);
        layer_waves.push_back(std::move(instructions));
        layer_buffer_sizes.push_back(buffer_size);
    }

    if (stop_after_op == "attn_norm") {
        return {std::move(layer_waves), std::move(layer_buffer_sizes)};
    }

    // QKV_RopeAppend
    {
        auto matmul_func =
            [](int layer_idx, int local_batch_block_idx,
               int local_output_block_idx, int global_batch_block_idx,
               int global_output_block_idx) -> std::unique_ptr<Instruction> {
            return std::make_unique<QKV_RopeAppend>(
                layer_idx, local_batch_block_idx, local_output_block_idx,
                global_batch_block_idx, global_output_block_idx);
        };
        int global_outdim =
            (globs.num_attention_heads + 2 * globs.num_kv_heads) *
            globs.head_dim;
        auto [instructions, buffer_size] =
            schedule_matmul(matmul_func, layer_idx, device_idx, globs,
                            global_outdim, false, true);
        layer_waves.push_back(std::move(instructions));
        layer_buffer_sizes.push_back(buffer_size);
    }

    if (stop_after_op == "qkv") {
        return {std::move(layer_waves), std::move(layer_buffer_sizes)};
    }

    // AttentionPrefill
    {
        auto [instructions, buffer_size] =
            schedule_attention_prefill(globs, layer_idx, -1);
        layer_waves.push_back(std::move(instructions));
        layer_buffer_sizes.push_back(buffer_size);
    }

    if (stop_after_op == "prefill") {
        return {std::move(layer_waves), std::move(layer_buffer_sizes)};
    }

    // AttentionDecode
    {
        auto [instructions, buffer_size] =
            schedule_attention_decode(globs, layer_idx);
        layer_waves.push_back(std::move(instructions));
        layer_buffer_sizes.push_back(buffer_size);
    }

    if (stop_after_op == "decode") {
        return {std::move(layer_waves), std::move(layer_buffer_sizes)};
    }

    // O_ProjResidual
    {
        auto matmul_func =
            [](int layer_idx, int local_batch_block_idx,
               int local_output_block_idx, int global_batch_block_idx,
               int global_output_block_idx) -> std::unique_ptr<Instruction> {
            return std::make_unique<O_ProjResidual>(
                layer_idx, local_batch_block_idx, local_output_block_idx,
                global_batch_block_idx, global_output_block_idx);
        };
        auto [instructions, buffer_size] =
            schedule_matmul_with_limit(matmul_func, layer_idx, device_idx, globs,
                            globs.hidden_size, true, false, 4, max_oproj_instructions_per_gpu); // use a smaller supergroup size for oproj to improve pipelining
        layer_waves.push_back(std::move(instructions));
        layer_buffer_sizes.push_back(buffer_size);
    }

    if (stop_after_op == "oproj") {
        return {std::move(layer_waves), std::move(layer_buffer_sizes)};
    }

    // MLP_Norm
    {
        auto norm_func = [](int layer_idx,
                            const std::vector<int>& local_batch_indices)
            -> std::unique_ptr<Instruction> {
            return std::make_unique<MLP_Norm>(layer_idx, local_batch_indices);
        };
        auto [instructions, buffer_size] = schedule_norm(
            norm_func, layer_idx, device_idx, globs, interleave_waves);
        layer_waves.push_back(std::move(instructions));
        layer_buffer_sizes.push_back(buffer_size);
    }

    if (stop_after_op == "mlp_norm") {
        return {std::move(layer_waves), std::move(layer_buffer_sizes)};
    }

    // GateSilu
    {
        auto matmul_func =
            [](int layer_idx, int local_batch_block_idx,
               int local_output_block_idx, int global_batch_block_idx,
               int global_output_block_idx) -> std::unique_ptr<Instruction> {
            return std::make_unique<GateSilu>(
                layer_idx, local_batch_block_idx, local_output_block_idx,
                global_batch_block_idx, global_output_block_idx);
        };
        auto [instructions, buffer_size] =
            schedule_matmul(matmul_func, layer_idx, device_idx, globs,
                            globs.intermediate_size, false, true);
        layer_waves.push_back(std::move(instructions));
        layer_buffer_sizes.push_back(buffer_size);
    }

    if (stop_after_op == "gate") {
        return {std::move(layer_waves), std::move(layer_buffer_sizes)};
    }

    // UpMatMul
    {
        auto matmul_func =
            [](int layer_idx, int local_batch_block_idx,
               int local_output_block_idx, int global_batch_block_idx,
               int global_output_block_idx) -> std::unique_ptr<Instruction> {
            return std::make_unique<UpMatMul>(
                layer_idx, local_batch_block_idx, local_output_block_idx,
                global_batch_block_idx, global_output_block_idx);
        };
        auto [instructions, buffer_size] =
            schedule_matmul(matmul_func, layer_idx, device_idx, globs,
                            globs.intermediate_size, false, true);
        layer_waves.push_back(std::move(instructions));
        layer_buffer_sizes.push_back(buffer_size);
    }

    if (stop_after_op == "up") {
        return {std::move(layer_waves), std::move(layer_buffer_sizes)};
    }

    // DownProjResidual
    {
        auto matmul_func =
            [](int layer_idx, int local_batch_block_idx,
               int local_output_block_idx, int global_batch_block_idx,
               int global_output_block_idx) -> std::unique_ptr<Instruction> {
            return std::make_unique<DownProjResidual>(
                layer_idx, local_batch_block_idx, local_output_block_idx,
                global_batch_block_idx, global_output_block_idx);
        };
        auto [instructions, buffer_size] =
            schedule_matmul(matmul_func, layer_idx, device_idx, globs,
                            globs.hidden_size, false, false);
        layer_waves.push_back(std::move(instructions));
        layer_buffer_sizes.push_back(buffer_size);
    }

    return {std::move(layer_waves), std::move(layer_buffer_sizes)};
}

std::vector<std::unique_ptr<Instruction>> schedule_model(
    const Globals& globs, int device_idx, int layer_limit,
    bool interleave_waves, int interleave_buffer_size,
    const std::string& stop_after_op, bool disable_lm_head, bool add_final_sync, int max_oproj_instructions_per_gpu, int num_threads) {
    // Validate stop_after_op parameter
    if (!stop_after_op.empty() && stop_after_op != "attn_norm" &&
        stop_after_op != "qkv" && stop_after_op != "decode" &&
        stop_after_op != "prefill" && stop_after_op != "oproj" &&
        stop_after_op != "mlp_norm" && stop_after_op != "gate" &&
        stop_after_op != "up" && stop_after_op != "lm_head_norm") {
        throw std::runtime_error("Invalid stop_after_op value: " +
                                 stop_after_op);
    }

    std::vector<std::vector<std::unique_ptr<Instruction>>>
        all_instructions_waves;
    std::vector<int> all_wave_buffer_sizes;

    // Add inc_barrier instruction at the start of model execution
    std::vector<std::unique_ptr<Instruction>> inc_barrier_wave;
    inc_barrier_wave.push_back(std::make_unique<IncBarrier>());
    all_instructions_waves.push_back(std::move(inc_barrier_wave));
    all_wave_buffer_sizes.push_back(1);

    int nlayers = (layer_limit == -1) ? globs.num_hidden_layers : layer_limit;

    // Use multi-threading if requested and there are multiple layers
    if (num_threads > 1 && nlayers > 1 && stop_after_op.empty()) {
        // Schedule layers in parallel
        std::vector<std::future<
            std::pair<std::vector<std::vector<std::unique_ptr<Instruction>>>,
                      std::vector<int>>>>
            futures;

        int layers_per_thread = (nlayers + num_threads - 1) / num_threads;

        for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            int start_layer = thread_idx * layers_per_thread;
            int end_layer = std::min(start_layer + layers_per_thread, nlayers);

            if (start_layer >= nlayers) break;

            futures.push_back(std::async(
                std::launch::async, [&globs, device_idx, interleave_waves,
                                     stop_after_op, start_layer, end_layer, max_oproj_instructions_per_gpu]() {
                    std::vector<std::vector<std::unique_ptr<Instruction>>>
                        thread_waves;
                    std::vector<int> thread_buffer_sizes;

                    for (int layer_idx = start_layer; layer_idx < end_layer;
                         ++layer_idx) {
                        auto [layer_waves, layer_buffer_sizes] =
                            schedule_layer(globs, device_idx, layer_idx,
                                           interleave_waves, stop_after_op, max_oproj_instructions_per_gpu);

                        thread_waves.insert(
                            thread_waves.end(),
                            std::make_move_iterator(layer_waves.begin()),
                            std::make_move_iterator(layer_waves.end()));
                        thread_buffer_sizes.insert(thread_buffer_sizes.end(),
                                                   layer_buffer_sizes.begin(),
                                                   layer_buffer_sizes.end());
                    }

                    return std::make_pair(std::move(thread_waves),
                                          std::move(thread_buffer_sizes));
                }));
        }

        // Collect results from all threads
        for (auto& future : futures) {
            auto [thread_waves, thread_buffer_sizes] = future.get();
            all_instructions_waves.insert(
                all_instructions_waves.end(),
                std::make_move_iterator(thread_waves.begin()),
                std::make_move_iterator(thread_waves.end()));
            all_wave_buffer_sizes.insert(all_wave_buffer_sizes.end(),
                                         thread_buffer_sizes.begin(),
                                         thread_buffer_sizes.end());
        }
    } else {
        // Single-threaded execution or early stop requested
        for (int layer_idx = 0; layer_idx < nlayers; ++layer_idx) {
            auto [layer_waves, layer_buffer_sizes] = schedule_layer(
                globs, device_idx, layer_idx, interleave_waves, stop_after_op, max_oproj_instructions_per_gpu);

            all_instructions_waves.insert(
                all_instructions_waves.end(),
                std::make_move_iterator(layer_waves.begin()),
                std::make_move_iterator(layer_waves.end()));
            all_wave_buffer_sizes.insert(all_wave_buffer_sizes.end(),
                                         layer_buffer_sizes.begin(),
                                         layer_buffer_sizes.end());

            // Check for early stop conditions using opcode comparison
            if (!stop_after_op.empty() && !layer_waves.empty()) {
                const auto& last_wave = layer_waves.back();
                if (!last_wave.empty()) {
                    int opcode = last_wave[0]->opcode();
                    bool should_stop = false;
                    
                    if (stop_after_op == "attn_norm" && opcode == 1) should_stop = true;
                    else if (stop_after_op == "qkv" && opcode == 2) should_stop = true;
                    else if (stop_after_op == "prefill" && opcode == 3) should_stop = true;
                    else if (stop_after_op == "decode" && opcode == 4) should_stop = true;
                    else if (stop_after_op == "oproj" && opcode == 5) should_stop = true;
                    else if (stop_after_op == "mlp_norm" && opcode == 6) should_stop = true;
                    else if (stop_after_op == "gate" && opcode == 7) should_stop = true;
                    else if (stop_after_op == "up" && opcode == 8) should_stop = true;
                    
                    if (should_stop) {
                        break;
                    }
                }
            }
        }
    }

    if (nlayers == globs.num_hidden_layers) {
        // LM_Head_Norm
        {
            auto norm_func = [](int layer_idx,
                                const std::vector<int>& local_batch_indices)
                -> std::unique_ptr<Instruction> {
                return std::make_unique<LM_Head_Norm>(layer_idx,
                                                      local_batch_indices);
            };
            auto [instructions, buffer_size] = schedule_norm(
                norm_func, 0, device_idx, globs, interleave_waves);
            all_instructions_waves.push_back(std::move(instructions));
            all_wave_buffer_sizes.push_back(buffer_size);
        }

        if (stop_after_op != "lm_head_norm" && !disable_lm_head) {
            // LM_Head
            {
                auto matmul_func = [](int layer_idx, int local_batch_block_idx,
                                      int local_output_block_idx,
                                      int global_batch_block_idx,
                                      int global_output_block_idx)
                    -> std::unique_ptr<Instruction> {
                    return std::make_unique<LM_Head>(
                        layer_idx, local_batch_block_idx,
                        local_output_block_idx, global_batch_block_idx,
                        global_output_block_idx);
                };
                auto [instructions, buffer_size] =
                    schedule_matmul(matmul_func, 0, device_idx, globs,
                                    globs.vocab_size, true, false);
                all_instructions_waves.push_back(std::move(instructions));
                all_wave_buffer_sizes.push_back(buffer_size);
            }
        }
    }

    if (add_final_sync) {
        // Add all device barrier instructions
        {
            auto [instructions, buffer_size] = schedule_all_device_barrier(globs, 0, 0);
            all_instructions_waves.push_back(std::move(instructions));
            all_wave_buffer_sizes.push_back(buffer_size);
        }
    }

    // Add die instructions (always at the end)
    {
        auto [instructions, buffer_size] = schedule_die(globs, device_idx);
        all_instructions_waves.push_back(std::move(instructions));
        all_wave_buffer_sizes.push_back(buffer_size);
    }

    if (interleave_waves) {
        return interleave_instruction_waves(all_instructions_waves,
                                            all_wave_buffer_sizes,
                                            interleave_buffer_size);
    } else {
        std::vector<std::unique_ptr<Instruction>> all_instructions;
        for (auto& wave : all_instructions_waves) {
            all_instructions.insert(all_instructions.end(),
                                    std::make_move_iterator(wave.begin()),
                                    std::make_move_iterator(wave.end()));
        }
        return all_instructions;
    }
}

std::vector<std::vector<std::unique_ptr<Instruction>>>
round_robin_assign_to_sms(
    std::vector<std::unique_ptr<Instruction>>& instructions, int sm_count) {
    std::vector<std::vector<std::unique_ptr<Instruction>>> buckets(sm_count);
    
    // Pre-reserve capacity for each bucket
    const size_t per = (instructions.size() + sm_count - 1) / sm_count;
    for (auto& b : buckets) {
        b.reserve(per);
    }

    for (size_t i = 0; i < instructions.size(); ++i) {
        buckets[i % sm_count].push_back(std::move(instructions[i]));
    }

    return buckets;
}

constexpr bool NON_BLOCKING = true;

torch::Tensor convert_instruction_queues_to_tensor(
    const std::vector<std::vector<std::unique_ptr<Instruction>>>&
        instruction_queues,
    const std::string& device, bool zero_init, int num_threads) {
    int num_sms = static_cast<int>(instruction_queues.size());

    // Find max queue length
    size_t max_queue_len = 0;
    for (const auto& queue : instruction_queues) {
        max_queue_len = std::max(max_queue_len, queue.size());
    }

    // Create tensor with pinned memory for fast H2D transfer
    auto host = torch::empty(
        {num_sms, static_cast<long>(max_queue_len), INTS_PER_INSTRUCTION},
        torch::dtype(torch::kInt32).pinned_memory(true));
    int* base = host.data_ptr<int>();

    // Zero once (fast) to guarantee padding for short encodings
    if (zero_init) {
        std::memset(base, 0, host.numel() * sizeof(int));
    }

    // Serialize instructions with multi-threading using fast write32 path
    if (num_threads > 1 && num_sms > 1) {
        std::vector<std::thread> threads;
        int sms_per_thread = (num_sms + num_threads - 1) / num_threads;

        for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            int start_sm = thread_idx * sms_per_thread;
            int end_sm = std::min(start_sm + sms_per_thread, num_sms);

            if (start_sm >= num_sms) break;

            threads.emplace_back([&instruction_queues, base,
                                  max_queue_len, start_sm, end_sm]() {
                for (int sm_idx = start_sm; sm_idx < end_sm; ++sm_idx) {
                    const auto& queue = instruction_queues[sm_idx];
                    int* dst = base + sm_idx * max_queue_len * INTS_PER_INSTRUCTION;

                    // Add actual instructions using fast write32
                    for (size_t i = 0; i < queue.size(); ++i) {
                        queue[i]->write32(dst + i * INTS_PER_INSTRUCTION);
                    }

                    // Add NoOp instructions for queue padding
                    NoOp noop;
                    for (size_t i = queue.size(); i < max_queue_len; ++i) {
                        noop.write32(dst + i * INTS_PER_INSTRUCTION);
                    }
                }
            });
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Single-threaded serialization using fast write32 path
        for (int sm_idx = 0; sm_idx < num_sms; ++sm_idx) {
            const auto& queue = instruction_queues[sm_idx];
            int* dst = base + sm_idx * max_queue_len * INTS_PER_INSTRUCTION;

            // Add actual instructions using fast write32
            for (size_t i = 0; i < queue.size(); ++i) {
                queue[i]->write32(dst + i * INTS_PER_INSTRUCTION);
            }

            // Add NoOp instructions for queue padding
            NoOp noop;
            for (size_t i = queue.size(); i < max_queue_len; ++i) {
                noop.write32(dst + i * INTS_PER_INSTRUCTION);
            }
        }
    }

    // Move to device once (non-blocking)
    auto serialized_tensor = host.to(device, NON_BLOCKING);
    return serialized_tensor;
}

torch::Tensor create_instruction_tensor(const Globals& globs, int device_idx,
                                        int layer_limit, bool interleave_waves,
                                        int interleave_buffer_size,
                                        bool move_to_gpu,
                                        const std::string& stop_after_op,
                                        bool zero_init, bool disable_lm_head,
                                        bool add_final_sync,
                                        int max_oproj_instructions_per_gpu,
                                        int num_threads) {
    auto schedule = schedule_model(globs, device_idx, layer_limit,
                                   interleave_waves, interleave_buffer_size,
                                   stop_after_op, disable_lm_head, add_final_sync, max_oproj_instructions_per_gpu, num_threads);

    std::string device_str;
    if (move_to_gpu) {
        device_str = "cuda:" + std::to_string(device_idx);
    } else {
        device_str = "cpu";
    }

    if (globs.global_work_queue_enabled) {
        // Global work queue path with pinned memory
        auto host = torch::empty(
            {static_cast<long>(schedule.size()), INTS_PER_INSTRUCTION},
            torch::dtype(torch::kInt32).pinned_memory(true));
        int* base = host.data_ptr<int>();

        // Zero once (fast) to guarantee padding for short encodings
        if (zero_init) {
            std::memset(base, 0, host.numel() * sizeof(int));
        }

        // Serialize instructions with multi-threading using fast write32 path
        if (num_threads > 1 &&
            schedule.size() > 100) {  // Only use threads for larger workloads
            std::vector<std::thread> threads;
            size_t items_per_thread =
                (schedule.size() + num_threads - 1) / num_threads;

            for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
                size_t start_idx = thread_idx * items_per_thread;
                size_t end_idx =
                    std::min(start_idx + items_per_thread, schedule.size());

                if (start_idx >= schedule.size()) break;

                threads.emplace_back(
                    [&schedule, base, start_idx, end_idx]() {
                        for (size_t i = start_idx; i < end_idx; ++i) {
                            schedule[i]->write32(base + i * INTS_PER_INSTRUCTION);
                        }
                    });
            }

            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }
        } else {
            // Single-threaded serialization using fast write32 path
            for (size_t i = 0; i < schedule.size(); ++i) {
                schedule[i]->write32(base + i * INTS_PER_INSTRUCTION);
            }
        }

        auto insts_tensor = host.to(device_str, NON_BLOCKING);
        return insts_tensor;

    } else {
        // Regular SM assignment path
        auto divided_across_sms =
            round_robin_assign_to_sms(schedule, globs.sm_count);
        return convert_instruction_queues_to_tensor(
            divided_across_sms, device_str, zero_init, num_threads);
    }
}

// Helper function for parallel instruction tensor creation with optimal thread usage
torch::Tensor create_instruction_tensor_parallel(
    const Globals& globs, int device_idx, int layer_limit,
    bool interleave_waves, int interleave_buffer_size,
    bool move_to_gpu, const std::string& stop_after_op,
    bool zero_init, bool disable_lm_head, bool add_final_sync,
    int max_oproj_instructions_per_gpu, int threads_per_gpu) {
    
    int nlayers = (layer_limit == -1) ? globs.num_hidden_layers : layer_limit;
    
    // Phase 1: Parallel layer generation
    std::vector<std::vector<std::vector<std::unique_ptr<Instruction>>>> all_layer_waves(nlayers);
    std::vector<std::vector<int>> all_buffer_sizes(nlayers);
    
    if (threads_per_gpu > 1 && nlayers > 1 && stop_after_op.empty()) {
        std::vector<std::future<void>> layer_futures;
        layer_futures.reserve(threads_per_gpu);
        
        // Use work-stealing pattern for better load balancing
        std::atomic<int> next_layer{0};
        
        for (int t = 0; t < threads_per_gpu; ++t) {
            layer_futures.push_back(std::async(std::launch::async,
                [&]() {
                    while (true) {
                        int layer_idx = next_layer.fetch_add(1);
                        if (layer_idx >= nlayers) break;
                        
                        auto [layer_waves, layer_buffer_sizes] = schedule_layer(
                            globs, device_idx, layer_idx, interleave_waves, stop_after_op, max_oproj_instructions_per_gpu);
                        
                        all_layer_waves[layer_idx] = std::move(layer_waves);
                        all_buffer_sizes[layer_idx] = std::move(layer_buffer_sizes);
                    }
                }));
        }
        
        // Wait for all layer generation to complete
        for (auto& future : layer_futures) {
            future.get();
        }
    } else {
        // Single-threaded layer generation (for early stop or small models)
        for (int layer_idx = 0; layer_idx < nlayers; ++layer_idx) {
            auto [layer_waves, layer_buffer_sizes] = schedule_layer(
                globs, device_idx, layer_idx, interleave_waves, stop_after_op, max_oproj_instructions_per_gpu);
            
            all_layer_waves[layer_idx] = std::move(layer_waves);
            all_buffer_sizes[layer_idx] = std::move(layer_buffer_sizes);
            
            // Handle early stop
            if (!stop_after_op.empty() && !layer_waves.empty()) {
                const auto& last_wave = layer_waves.back();
                if (!last_wave.empty()) {
                    int opcode = last_wave[0]->opcode();
                    bool should_stop = false;
                    
                    if (stop_after_op == "attn_norm" && opcode == 1) should_stop = true;
                    else if (stop_after_op == "qkv" && opcode == 2) should_stop = true;
                    else if (stop_after_op == "prefill" && opcode == 3) should_stop = true;
                    else if (stop_after_op == "decode" && opcode == 4) should_stop = true;
                    else if (stop_after_op == "oproj" && opcode == 5) should_stop = true;
                    else if (stop_after_op == "mlp_norm" && opcode == 6) should_stop = true;
                    else if (stop_after_op == "gate" && opcode == 7) should_stop = true;
                    else if (stop_after_op == "up" && opcode == 8) should_stop = true;
                    
                    if (should_stop) {
                        nlayers = layer_idx + 1;
                        all_layer_waves.resize(nlayers);
                        all_buffer_sizes.resize(nlayers);
                        break;
                    }
                }
            }
        }
    }
    
    // Flatten all waves and buffer sizes
    std::vector<std::vector<std::unique_ptr<Instruction>>> all_instructions_waves;
    std::vector<int> all_wave_buffer_sizes;
    
    // Add inc_barrier at the start
    std::vector<std::unique_ptr<Instruction>> inc_barrier_wave;
    inc_barrier_wave.push_back(std::make_unique<IncBarrier>());
    all_instructions_waves.push_back(std::move(inc_barrier_wave));
    all_wave_buffer_sizes.push_back(1);
    
    // Collect all layer waves
    for (int i = 0; i < nlayers; ++i) {
        for (auto& wave : all_layer_waves[i]) {
            all_instructions_waves.push_back(std::move(wave));
        }
        for (int buffer_size : all_buffer_sizes[i]) {
            all_wave_buffer_sizes.push_back(buffer_size);
        }
    }
    
    // Add final layers if needed
    if (nlayers == globs.num_hidden_layers) {
        // LM_Head_Norm
        {
            auto norm_func = [](int layer_idx,
                                const std::vector<int>& local_batch_indices)
                -> std::unique_ptr<Instruction> {
                return std::make_unique<LM_Head_Norm>(layer_idx,
                                                      local_batch_indices);
            };
            auto [instructions, buffer_size] = schedule_norm(
                norm_func, 0, device_idx, globs, interleave_waves);
            all_instructions_waves.push_back(std::move(instructions));
            all_wave_buffer_sizes.push_back(buffer_size);
        }
        
        if (stop_after_op != "lm_head_norm" && !disable_lm_head) {
            // LM_Head
            auto matmul_func = [](int layer_idx, int local_batch_block_idx,
                                  int local_output_block_idx,
                                  int global_batch_block_idx,
                                  int global_output_block_idx)
                -> std::unique_ptr<Instruction> {
                return std::make_unique<LM_Head>(
                    layer_idx, local_batch_block_idx,
                    local_output_block_idx, global_batch_block_idx,
                    global_output_block_idx);
            };
            auto [instructions, buffer_size] =
                schedule_matmul(matmul_func, 0, device_idx, globs,
                                globs.vocab_size, true, false);
            all_instructions_waves.push_back(std::move(instructions));
            all_wave_buffer_sizes.push_back(buffer_size);
        }
    }
    
    if (add_final_sync) {
        auto [instructions, buffer_size] = schedule_all_device_barrier(globs, 0, 0);
        all_instructions_waves.push_back(std::move(instructions));
        all_wave_buffer_sizes.push_back(buffer_size);
    }
    
    // Add die instructions
    {
        auto [instructions, buffer_size] = schedule_die(globs, device_idx);
        all_instructions_waves.push_back(std::move(instructions));
        all_wave_buffer_sizes.push_back(buffer_size);
    }
    
    // Phase 2: Serial wave interleaving (if requested)
    std::vector<std::unique_ptr<Instruction>> final_schedule;
    if (interleave_waves) {
        final_schedule = interleave_instruction_waves(
            all_instructions_waves, all_wave_buffer_sizes, interleave_buffer_size);
    } else {
        for (auto& wave : all_instructions_waves) {
            for (auto& inst : wave) {
                final_schedule.push_back(std::move(inst));
            }
        }
    }
    
    std::string device_str = move_to_gpu ? 
        "cuda:" + std::to_string(device_idx) : "cpu";
    
    // Phase 3: Parallel serialization
    if (globs.global_work_queue_enabled) {
        // Global work queue path
        auto host = torch::empty(
            {static_cast<long>(final_schedule.size()), INTS_PER_INSTRUCTION},
            torch::dtype(torch::kInt32).pinned_memory(true));
        int* base = host.data_ptr<int>();
        
        if (zero_init) {
            std::memset(base, 0, host.numel() * sizeof(int));
        }
        
        // Parallel serialization with multiple threads
        if (threads_per_gpu > 1 && final_schedule.size() > 100) {
            std::vector<std::future<void>> serialize_futures;
            size_t chunk_size = (final_schedule.size() + threads_per_gpu - 1) / threads_per_gpu;
            
            for (int t = 0; t < threads_per_gpu; ++t) {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, final_schedule.size());
                
                if (start >= final_schedule.size()) break;
                
                serialize_futures.push_back(std::async(std::launch::async,
                    [&final_schedule, base, start, end]() {
                        for (size_t i = start; i < end; ++i) {
                            final_schedule[i]->write32(base + i * INTS_PER_INSTRUCTION);
                        }
                    }));
            }
            
            for (auto& future : serialize_futures) {
                future.get();
            }
        } else {
            // Single-threaded serialization
            for (size_t i = 0; i < final_schedule.size(); ++i) {
                final_schedule[i]->write32(base + i * INTS_PER_INSTRUCTION);
            }
        }
        
        // Phase 4: Single copy to GPU
        return host.to(device_str, NON_BLOCKING);
        
    } else {
        // Regular SM assignment path
        auto divided_across_sms = round_robin_assign_to_sms(final_schedule, globs.sm_count);
        
        // Parallel serialization for SM queues
        // (reuse convert_instruction_queues_to_tensor which already has parallel support)
        return convert_instruction_queues_to_tensor(
            divided_across_sms, device_str, zero_init, threads_per_gpu);
    }
}

std::vector<torch::Tensor> create_all_instruction_tensors(
    const Globals& globs, int layer_limit, bool interleave_waves,
    int interleave_buffer_size, bool move_to_gpu,
    const std::string& stop_after_op, bool zero_init, bool disable_lm_head,
    bool add_final_sync, int max_oproj_instructions_per_gpu, int num_threads) {
    
    std::vector<torch::Tensor> tensors;
    tensors.reserve(globs.tp_size);
    
    // Optimal thread distribution: 8 threads per GPU
    constexpr int THREADS_PER_GPU = 8;
    
    // Only use multi-GPU parallelism if we have enough threads
    if (num_threads >= globs.tp_size * THREADS_PER_GPU) {
        // Parallel GPU processing with optimal thread distribution
        std::vector<std::future<torch::Tensor>> gpu_futures;
        gpu_futures.reserve(globs.tp_size);
        
        for (int dev_idx = 0; dev_idx < globs.tp_size; ++dev_idx) {
            gpu_futures.push_back(std::async(std::launch::async,
                [&globs, dev_idx, layer_limit, interleave_waves,
                 interleave_buffer_size, move_to_gpu, stop_after_op, zero_init,
                 disable_lm_head, add_final_sync, max_oproj_instructions_per_gpu]() {
                    return create_instruction_tensor_parallel(
                        globs, dev_idx, layer_limit, interleave_waves,
                        interleave_buffer_size, move_to_gpu, stop_after_op,
                        zero_init, disable_lm_head, add_final_sync, max_oproj_instructions_per_gpu, THREADS_PER_GPU);
                }));
        }
        
        // Collect results from all GPUs
        for (auto& future : gpu_futures) {
            tensors.push_back(future.get());
        }
    } else if (num_threads > 1 && globs.tp_size > 1) {
        // Fall back to simple parallel GPU processing
        std::vector<std::future<torch::Tensor>> futures;
        futures.reserve(globs.tp_size);
        
        int threads_per_gpu = std::max(1, num_threads / globs.tp_size);
        
        for (int dev_idx = 0; dev_idx < globs.tp_size; ++dev_idx) {
            futures.push_back(std::async(std::launch::async,
                [&globs, dev_idx, layer_limit, interleave_waves,
                 interleave_buffer_size, move_to_gpu, stop_after_op, zero_init,
                 disable_lm_head, add_final_sync, max_oproj_instructions_per_gpu, threads_per_gpu]() {
                    return create_instruction_tensor_parallel(
                        globs, dev_idx, layer_limit, interleave_waves,
                        interleave_buffer_size, move_to_gpu, stop_after_op,
                        zero_init, disable_lm_head, add_final_sync, max_oproj_instructions_per_gpu, threads_per_gpu);
                }));
        }
        
        for (auto& future : futures) {
            tensors.push_back(future.get());
        }
    } else {
        // Single-threaded execution
        for (int dev_idx = 0; dev_idx < globs.tp_size; ++dev_idx) {
            tensors.push_back(create_instruction_tensor(
                globs, dev_idx, layer_limit, interleave_waves,
                interleave_buffer_size, move_to_gpu, stop_after_op, zero_init,
                disable_lm_head, add_final_sync, max_oproj_instructions_per_gpu, 1));
        }
    }
    
    return tensors;
}

}  // namespace scheduler_cpp