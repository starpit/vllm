// SPDX-License-Identifier: Apache-2.0
// Ferrite megakernel runtime trace infrastructure (Phase 3 step 8).
//
// The emitted mega kernel receives a runtime `trace_level` as a kernel
// arg, populated from the `FERRITE_MEGA_TRACE` env var read on the host
// side at launch time (see `emit_mega_forward_fn` in
// `ferrite-forward-macro/src/interpreter/mega.rs`). Per-op trace
// stanzas injected by the codegen (see `variant_cpp::WalkerBodies::push`)
// gate their `printf` on this level.
//
// Levels (cumulative):
//   0  off — every gate returns false; stanzas compile to roughly a
//            single compare + branch taken, so the cost on hot paths
//            is negligible.
//   1  per-op entry trace from CTA 0 thread 0 — one printf per op
//            invocation. Answers "what ops ran, in what order".
//   2  + first-bf16-values of the op's primary input activation slot.
//   3  + first-bf16-values of the op's primary output activation slot.
//   4+ reserved for per-warp detail and lm_head logit dumps.
//
// Design notes:
//   - printf from device code goes through CUDA's printf FIFO (default
//     ~1 MB, tunable via `cudaDeviceSetLimit(cudaLimitPrintfFifoSize,...)`).
//     Bursty level-1 output across a 16-layer decode emits ~hundreds of
//     lines per forward — fits comfortably.
//   - Gating on `blockIdx.x == 0 && threadIdx.x == 0` keeps per-op output
//     to one line per launch regardless of how many CTAs are resident;
//     level 4+ may relax this for per-warp debug.
//   - The helper is `__forceinline__` + constexpr-foldable on `level==0`
//     so the else-branch is effectively dead code in builds where the
//     user never sets the env var.
#pragma once

#include <cstdio>
#include <cuda_bf16.h>
#include "kittens.cuh"

namespace ferrite {

// Returns true if the caller should emit a level-`required` trace line.
// Gate shape picked to work in the storer role body: `kittens::laneid()
// == 0` picks thread 0 of whichever warp is executing (the storer body
// runs on one warp at `threadIdx.x in [128, 160)` for the baseline
// NUM_CONSUMER_WARPS=2 config — so `threadIdx.x == 0` would never
// fire). Adding `blockIdx.x == 0` further narrows to a single CTA so
// there is exactly one printf per op per launch.
//
// Why not emit into the consumer body? An earlier pass did, and it
// deadlocked the decode forward even with trace_level=0: the printf
// branch sits inside the heavily-syncing consumer warpgroup, and
// evidently interacted with the kernel's `__syncthreads` / `bar.sync`
// topology in a way that couldn't be unwound. The storer is a single
// low-register warp whose work is linear (wait → TMA store → advance)
// with no group barriers across roles, so the printf lives on a quiet
// side path.
__device__ __forceinline__ bool mega_trace_gate(int trace_level, int required) {
    return trace_level >= required
        && blockIdx.x == 0
        && blockIdx.y == 0
        && blockIdx.z == 0
        && kittens::laneid() == 0;
}

}  // namespace ferrite
