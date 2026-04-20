// SPDX-License-Identifier: Apache-2.0
// Top-level for the vendored cross-gpu-llama megakernel — single-
// device adaptation. Replaces the original llama.cu's pybind/torch
// top-level with an `extern "C"` launcher Rust FFI can call.
//
// Op .cu / llama.cuh / matmul_pipeline.cuh come verbatim from
// ~/Megakernels/demos/cross-gpu-llama/. llama.cuh is patched (this
// session) to drop the obsolete 4-arg pgl signature so the file
// compiles against current ThunderKittens.
//
// Session 1 deliverable per MK_LLAMA_HANDOFF.md: this file builds.
// No real launcher logic yet — `launch_mk_llama_stub` is a placeholder
// so the static archive has at least one resolvable symbol.

// PCH equivalent: cross-gpu-llama relies on pch.cuh pulling in
// kittens.cuh + megakernel.cuh before any op .cu. With no PCH we
// include them up front so every op .cu sees ducks::, warpgroup::,
// tma::, megakernel::.
#include <iostream>
#include "kittens.cuh"
#include "megakernel.cuh"

#include "attention_decode.cu"
#include "attention_prefill.cu"
#include "batched_rms_norm.cu"
#include "gate_silu.cu"
#include "inc_barriers.cu"
#include "lm_head.cu"
#include "matmul_adds.cu"
#include "qkv_rope_append.cu"
#include "up_matmul.cu"
#include "all_device_barrier.cu"

using namespace kittens;
using namespace megakernel;

extern "C" cudaError_t launch_mk_llama_stub() {
    // Placeholder — real launcher (constructs globals_t + calls
    // mk<config, globals, ops...><<<>>>) lands in session 2 once
    // every op .cu compiles cleanly through the patched llama.cuh.
    return cudaSuccess;
}
