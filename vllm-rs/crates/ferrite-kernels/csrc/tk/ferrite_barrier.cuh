// SPDX-License-Identifier: Apache-2.0
// Ferrite-TK megakernel substrate — gmem cross-SM barriers.
//
// See FERRITE_TK_PLAN.md, "Subtile wavefront across SMs".
//
// For ops that span more SMs than a single CTA can cover
// (attention_reduction, lm_head over full vocab), ferrite splits
// the tile space across SMs and synchronizes via plain CUDA
// atomic counters in gmem. No TK `pgl<>`, no cuMulticast, no
// NCCL — TP=1 single-GPU is the scope of Phases 1-5.
//
// Ferrite codegen decides, per op, what barrier shape is needed
// (layer × batch-block × tile-group), sizes the barrier tensor
// accordingly, and zero-inits it on the host side in the
// launcher. Producer SMs atomicAdd their contribution count;
// consumer SMs spin-load until the expected total is reached.
//
// Phase 1 provides only the primitive helpers. Phase 3+
// (attention_reduction, lm_head) uses them for real.

#pragma once

#include "kittens.cuh"

namespace ferrite {

// Producer: bump a gmem counter by `count`. Release ordering so
// the data the producer wrote before this call is visible to a
// consumer that acquires on the same slot.
__device__ __forceinline__ void barrier_signal(int32_t* slot, int count) {
    // __threadfence() gives gpu-wide visibility of prior writes
    // before the atomic is observed. Matches a
    // `cuda::memory_order_release` store semantically.
    __threadfence();
    atomicAdd(slot, count);
}

// Consumer: spin until the gmem counter reaches `expected`.
// Uses a volatile load — we want to re-read memory every poll,
// not let the compiler hoist the load out of the loop. Issues a
// __threadfence() on exit so subsequent reads of producer data
// see the release'd writes.
__device__ __forceinline__ void barrier_wait(const int32_t* slot, int expected) {
    volatile const int32_t* vslot = slot;
    while (*vslot < expected) {
        __nanosleep(20);
    }
    __threadfence();
}

} // namespace ferrite
