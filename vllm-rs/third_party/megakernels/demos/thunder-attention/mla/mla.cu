#include "../../../include/megakernel.cuh"
#include "mla.cuh"
#include "mla_partial.cu"
#include "mla_reduction.cu"

using namespace kittens;
using namespace megakernel;

#include "pyutils/pyutils.cuh"

// Include the scheduler utility function from original
float get_quality(const std::vector<float>& next_times_input, int num_processors, int num_tokens, int seq_length) {
    int num_partial_steps = (seq_length + 31) / 32;

    if (next_times_input.size() > num_processors) {
        return -999999999.0f;
    }
    
    std::vector<float> next_times = next_times_input;
    std::sort(next_times.begin(), next_times.end(), std::greater<float>());

    std::vector<float> partial_times;
    for (int i = 0; i < num_processors; i++) {
        next_times[i%next_times.size()] -= 0.4f; // REDUCTION_COST_PER_STEP
        partial_times.push_back(next_times[i%next_times.size()]);
    }

    std::sort(partial_times.begin(), partial_times.end());
    
    for (size_t j = 0; j < next_times.size(); j++) {
        float actual_start_time = next_times[j] + 1.0f - 2.0f; // PRODUCER_LATENCY - STARTUP_TIME
        for (int k = 0; k < num_tokens; k++) {
            if (num_tokens * j + k < partial_times.size()) {
                partial_times[num_tokens * j + k] = actual_start_time;
            }
        }
    }
    
    std::sort(partial_times.begin(), partial_times.end(), std::greater<float>());
    
    float min_value = partial_times.back();
    for(int i = 0; i < partial_times.size(); i++) {
        if(num_partial_steps > 0) {
            int num_steps_alloc = std::min(num_partial_steps, (int)(round((partial_times[i]-min_value) / 1.49f))); // PARTIAL_COST_PER_STEP
            num_partial_steps -= num_steps_alloc;
            partial_times[i] -= num_steps_alloc * 1.49f;
            if(num_steps_alloc > 0) partial_times[i] -= (3.0f + 4.5f); // PARTIAL_OVERHEAD
        }
    }

    int full_passes = num_partial_steps / partial_times.size();
    int remainder = num_partial_steps - (full_passes * partial_times.size());

    std::sort(partial_times.begin(), partial_times.end(), std::greater<float>());
    min_value = 9999999999.0f;
    for(int i = 0; i < remainder; i++){
        float f = partial_times[i] - 1.49f * (full_passes+1);
        if(f < min_value) min_value = f;
    }
    for(int i = remainder; i < partial_times.size(); i++) {
        float f = partial_times[i] - 1.49f * full_passes;
        if(f < min_value) min_value = f;
    }

    return min_value;
}

PYBIND11_MODULE(mla_decode, m) {
    m.doc() = "MLA decode megakernel python module";
    
    // Bind the 16-head kernel
    kittens::py::bind_kernel< megakernel::mk<mla_config, mla_16_globals,
        mla_partial<16>,
        mla_reduction<16>
    >>(m, "mla_decode",
        &mla_16_globals::instructions,
        &mla_16_globals::timings,
        &mla_16_globals::global_instruction_index,
        &mla_16_globals::Qrot,
        &mla_16_globals::Qv,
        &mla_16_globals::Krot,
        &mla_16_globals::V,
        &mla_16_globals::Table,
        &mla_16_globals::O,
        &mla_16_globals::O_scratch,
        &mla_16_globals::Lvec_scratch,
        &mla_16_globals::completion_flag,
        &mla_16_globals::Softmax_scale,
        &mla_16_globals::tic
    );
    
//     // Bind the 8-head kernel
//     kittens::py::bind_kernel<megakernel::mk<mla_config, mla_8_globals, mla_partial<8>
//     //, mla_reduction<8>
//     >(m, "mla_decode_8_heads",
//         &mla_8_globals::instructions,
//         &mla_8_globals::timings,
//         &mla_8_globals::global_instruction_index,
//         &mla_8_globals::Qrot,
//         &mla_8_globals::Qv,
//         &mla_8_globals::Krot,
//         &mla_8_globals::V,
//         &mla_8_globals::Table,
//         &mla_8_globals::O,
//         &mla_8_globals::O_scratch,
//         &mla_8_globals::Lvec_scratch,
//         &mla_8_globals::completion_flag,
//         &mla_8_globals::Softmax_scale,
//         &mla_8_globals::tic
//     );
    
    // Bind scheduler utility
    m.def("__get_quality__", &get_quality, 
        "An internal utility function for generating efficient schedules.",
        pybind11::arg("next_times"), 
        pybind11::arg("num_processors"), 
        pybind11::arg("num_tokens"), 
        pybind11::arg("seq_length")
    );
}