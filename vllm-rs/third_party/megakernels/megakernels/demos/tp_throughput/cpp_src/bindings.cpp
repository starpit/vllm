#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "globs.hpp"
#include "instructions.hpp"
#include "scheduling.hpp"

namespace py = pybind11;
using namespace scheduler_cpp;

// Helper function to extract globals from Python object
Globals extract_globals_from_python(py::object globs_obj) {
    Globals globs;

    // Extract model configuration - these are stored directly on globs, not on
    // model_config
    globs.num_hidden_layers = globs_obj.attr("num_hidden_layers").cast<int>();
    globs.num_attention_heads =
        globs_obj.attr("num_attention_heads").cast<int>();
    globs.num_kv_heads = globs_obj.attr("num_kv_heads").cast<int>();
    globs.head_dim = globs_obj.attr("head_dim").cast<int>();
    globs.hidden_size = globs_obj.attr("hidden_size").cast<int>();
    globs.intermediate_size = globs_obj.attr("intermediate_size").cast<int>();
    globs.vocab_size = globs_obj.attr("vocab_size").cast<int>();

    // Extract other attributes
    globs.attn_scale = globs_obj.attr("attn_scale").cast<float>();
    globs.rms_norm_eps = globs_obj.attr("rms_norm_eps").cast<float>();
    globs.tp_size = globs_obj.attr("tp_size").cast<int>();
    globs.tp_rank = globs_obj.attr("tp_rank").cast<int>();
    globs.global_batch_size = globs_obj.attr("global_batch_size").cast<int>();
    globs.matmul_batch_block_size =
        globs_obj.attr("matmul_batch_block_size").cast<int>();
    globs.matmul_output_block_size =
        globs_obj.attr("matmul_output_block_size").cast<int>();
    globs.prefill_block_size = globs_obj.attr("prefill_block_size").cast<int>();
    globs.global_work_queue_enabled =
        globs_obj.attr("global_work_queue_enabled").cast<bool>();
    globs.sm_count = globs_obj.attr("sm_count").cast<int>();

    // Extract paged KV cache parameters
    globs.page_size = globs_obj.attr("page_size").cast<int>();
    globs.num_pages = globs_obj.attr("num_pages").cast<int>();
    globs.timing_record_enabled =
        globs_obj.attr("timing_record_enabled").cast<bool>();

    // Extract prefill_chunk_lens (renamed from prefill_seq_lens to match Python)
    globs.prefill_chunk_lens =
        globs_obj.attr("prefill_chunk_lens").cast<std::vector<int>>();
    
    // Extract prefill_extend_offsets
    globs.prefill_extend_offsets =
        globs_obj.attr("prefill_extend_offsets").cast<std::vector<int>>();

    // Extract device_batch_size if available
    if (py::hasattr(globs_obj, "device_batch_size")) {
        auto device_batch_size_obj = globs_obj.attr("device_batch_size");
        if (!device_batch_size_obj.is_none()) {
            globs.device_batch_size =
                device_batch_size_obj.cast<std::vector<int>>();
        }
    }

    return globs;
}

// Python wrapper function
torch::Tensor create_instruction_tensor_py(
    py::object globs_obj, int device_idx,
    py::object layer_limit_obj = py::none(), bool interleave_waves = false,
    py::object interleave_buffer_size_obj = py::none(), bool move_to_gpu = true,
    py::object stop_after_op_obj = py::none(), bool zero_init = true,
    bool disable_lm_head = false, bool add_final_sync = true,
    py::object max_oproj_instructions_per_gpu_obj = py::none(), int num_threads = 1) {
    auto globs = extract_globals_from_python(globs_obj);

    int layer_limit = -1;
    if (!layer_limit_obj.is_none()) {
        layer_limit = layer_limit_obj.cast<int>();
    }

    int interleave_buffer_size = -1;
    if (!interleave_buffer_size_obj.is_none()) {
        interleave_buffer_size = interleave_buffer_size_obj.cast<int>();
    }

    std::string stop_after_op = "";
    if (!stop_after_op_obj.is_none()) {
        stop_after_op = stop_after_op_obj.cast<std::string>();
    }

    int max_oproj_instructions = -1;
    if (!max_oproj_instructions_per_gpu_obj.is_none()) {
        max_oproj_instructions = max_oproj_instructions_per_gpu_obj.cast<int>();
    }

    return create_instruction_tensor(globs, device_idx, layer_limit,
                                     interleave_waves, interleave_buffer_size,
                                     move_to_gpu, stop_after_op, zero_init,
                                     disable_lm_head, add_final_sync,
                                     max_oproj_instructions, num_threads);
}

// Python wrapper function for create_all_instruction_tensors
std::vector<torch::Tensor> create_all_instruction_tensors_py(
    py::object globs_obj, py::object layer_limit_obj = py::none(),
    bool interleave_waves = false,
    py::object interleave_buffer_size_obj = py::none(), bool move_to_gpu = true,
    py::object stop_after_op_obj = py::none(), bool zero_init = true,
    bool disable_lm_head = false, bool add_final_sync = true,
    py::object max_oproj_instructions_per_gpu_obj = py::none(), int num_threads = 1) {
    auto globs = extract_globals_from_python(globs_obj);

    int layer_limit = -1;
    if (!layer_limit_obj.is_none()) {
        layer_limit = layer_limit_obj.cast<int>();
    }

    int interleave_buffer_size = -1;
    if (!interleave_buffer_size_obj.is_none()) {
        interleave_buffer_size = interleave_buffer_size_obj.cast<int>();
    }

    std::string stop_after_op = "";
    if (!stop_after_op_obj.is_none()) {
        stop_after_op = stop_after_op_obj.cast<std::string>();
    }

    int max_oproj_instructions = -1;
    if (!max_oproj_instructions_per_gpu_obj.is_none()) {
        max_oproj_instructions = max_oproj_instructions_per_gpu_obj.cast<int>();
    }

    return create_all_instruction_tensors(
        globs, layer_limit, interleave_waves, interleave_buffer_size,
        move_to_gpu, stop_after_op, zero_init, disable_lm_head, add_final_sync,
        max_oproj_instructions, num_threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "C++ implementation of scheduling functions.";

    m.def("create_instruction_tensor", &create_instruction_tensor_py,
          "Create instruction tensor", py::arg("globs"), py::arg("device_idx"),
          py::arg("layer_limit") = py::none(),
          py::arg("interleave_waves") = false,
          py::arg("interleave_buffer_size") = py::none(),
          py::arg("move_to_gpu") = true, py::arg("stop_after_op") = py::none(),
          py::arg("zero_init") = true, py::arg("disable_lm_head") = false,
          py::arg("add_final_sync") = true,
          py::arg("max_oproj_instructions_per_gpu") = py::none(),
          py::arg("num_threads") = 1);

    m.def("create_all_instruction_tensors", &create_all_instruction_tensors_py,
          "Create instruction tensors for all devices", py::arg("globs"),
          py::arg("layer_limit") = py::none(),
          py::arg("interleave_waves") = false,
          py::arg("interleave_buffer_size") = py::none(),
          py::arg("move_to_gpu") = true, py::arg("stop_after_op") = py::none(),
          py::arg("zero_init") = true, py::arg("disable_lm_head") = false,
          py::arg("add_final_sync") = true,
          py::arg("max_oproj_instructions_per_gpu") = py::none(),
          py::arg("num_threads") = 1);
}