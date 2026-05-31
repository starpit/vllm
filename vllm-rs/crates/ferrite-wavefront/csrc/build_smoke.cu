// SPDX-License-Identifier: Apache-2.0
//
// Phase 1 cross-shape compile gate. nvcc-compiles the megakernel
// scaffold + rms_lm_head IType against TWO concrete shape combos:
//
//   1. Llama-3.2-1B (HEAD_DIM=64, NUM_KV_HEADS=8, HIDDEN=2048,
//      INTER=8192, VOCAB=128256)
//   2. Llama-3.1-8B (HEAD_DIM=128, NUM_KV_HEADS=8, HIDDEN=4096,
//      INTER=14336, VOCAB=128256)
//
// Both must produce .ptx without errors. The point is to lock down
// shape-genericity at compile time — any IType template that compiles
// only for one shape is wrong by construction.
//
// Build (on H100 pod):
//   nvcc -arch=sm_90a -std=c++20 \
//        --expt-relaxed-constexpr --expt-extended-lambda \
//        -DKITTENS_HOPPER \
//        -I third_party/thunderkittens/include \
//        -I crates/ferrite-wavefront/csrc \
//        -I crates/ferrite-wavefront/csrc/megakernel \
//        -I crates/ferrite-wavefront/csrc/itypes/llama \
//        -ptx crates/ferrite-wavefront/csrc/build_smoke.cu \
//        -o /tmp/build_smoke.ptx

#include "kittens.cuh"
#include "config.cuh"
#include "megakernel.cuh"

// ferrite-vendored from ~/git/Megakernels/demos/low-latency-llama/
#include "llama.cuh"
#include "utils.cuh"
#include "matvec_pipeline.cuh"
#include "rms_lm_head.cu"
#include "rms_matvec_rope_append.cu"
#include "upgate.cu"
#include "matvec_adds.cu"
#include "attention_partial.cu"
#include "attention_reduction.cu"

namespace ferrite_smoke {

// ── Shape 1: Llama-3.2-1B ────────────────────────────────────────────
using globals_1b = globals_t<
    /*num_layers*/           16,
    /*hidden_dim*/         2048,
    /*intermediate_dim*/   8192,
    /*head_dim*/             64,
    /*num_attention_heads*/  32,
    /*num_kv_heads*/          8,
    /*kv_block_size*/        16,
    /*matvec_block_size*/    16,
    /*sm_count*/            132   // H100 SMs
>;

// ── Shape 2: Llama-3.1-8B ────────────────────────────────────────────
using globals_8b = globals_t<
    /*num_layers*/           32,
    /*hidden_dim*/         4096,
    /*intermediate_dim*/  14336,
    /*head_dim*/            128,
    /*num_attention_heads*/  32,
    /*num_kv_heads*/          8,
    /*kv_block_size*/        16,
    /*matvec_block_size*/    16,
    /*sm_count*/            132
>;

// Force template instantiation for both shapes. The struct definitions
// alone don't trigger codegen; using ::pipeline::loader_loop and
// ::consumer::run forces nvcc to instantiate the full template body.
//
// Both rms_lm_head<config, globals_1b> AND rms_lm_head<config, globals_8b>
// must compile clean. A hardcoded HEAD_DIM=64 in the template body
// would break globals_8b instantiation here.
template struct ::rms_lm_head<megakernel::default_config, globals_1b>;
template struct ::rms_lm_head<megakernel::default_config, globals_8b>;
template struct ::rms_qkv_rope_append<megakernel::default_config, globals_1b>;
template struct ::rms_qkv_rope_append<megakernel::default_config, globals_8b>;
template struct ::rms_upgate_silu<megakernel::default_config, globals_1b>;
template struct ::rms_upgate_silu<megakernel::default_config, globals_8b>;
template struct ::downproj<megakernel::default_config, globals_1b>;
template struct ::downproj<megakernel::default_config, globals_8b>;
template struct ::o_proj<megakernel::default_config, globals_1b>;
template struct ::o_proj<megakernel::default_config, globals_8b>;
template struct ::attention_partial<megakernel::default_config, globals_1b>;
template struct ::attention_partial<megakernel::default_config, globals_8b>;
template struct ::attention_reduction<megakernel::default_config, globals_1b>;
template struct ::attention_reduction<megakernel::default_config, globals_8b>;

}  // namespace ferrite_smoke

// nvcc requires a kernel symbol to actually emit ptx. Trivial entry:
extern "C" __global__ void ferrite_phase1_smoke_marker() {
    // Cross-shape gate: this kernel exists solely so nvcc walks the
    // template instantiations above and produces .ptx. The kernel
    // itself does nothing.
}
