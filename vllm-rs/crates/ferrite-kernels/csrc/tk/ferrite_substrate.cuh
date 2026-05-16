// SPDX-License-Identifier: Apache-2.0
// Ferrite-TK megakernel substrate — per-CTA SharedState.
//
// See FERRITE_TK_PLAN.md at the repo root.
//
// SharedState<Config> is the per-CTA shared-memory block that
// holds ferrite's page array, page-handoff semaphores, and
// scratch bytes. Ops talk to pages through this struct; codegen
// emits `stage` indices into pages[] and page_{ready,done}[] at
// compile time — there is no runtime page allocator.
//
// Config is a variant-specific struct emitted inline in each
// codegen'd .cu file (see ferrite_config.cuh template in the
// codegen prelude). It must define:
//
//   static constexpr int NUM_CONSUMER_WARPS;
//   static constexpr int NON_CONSUMER_REGISTERS;
//   static constexpr int CONSUMER_REGISTERS;
//   static constexpr int NUM_PAGES;
//   static constexpr int PAGE_SIZE;          // bytes per page
//   static constexpr int SCRATCH_BYTES;
//   static constexpr int INSTRUCTION_PIPE_STAGES;
//
// Deliberately minimal for Phase 1: the walker bodies are empty,
// so nothing actually reads the pages yet. The layout and init
// pattern are what matter — Phase 2 (rms_norm) exercises them
// for real.

#pragma once

#include "kittens.cuh"

namespace ferrite {

template <typename Config>
struct alignas(128) SharedState {
    // Handoff pages between loader / consumer / storer roles.
    // Each page is a raw byte buffer; op headers reinterpret
    // pages[stage] as whatever TK shared-tile type they need
    // (e.g. `kittens::st_bf<16,16>*`) via a reinterpret_cast.
    alignas(128) uint8_t pages[Config::NUM_PAGES][Config::PAGE_SIZE];

    // Page handoff mbarrier pairs, one per page slot.
    //   page_ready   [s] — loader arrives (TMA complete), consumer waits.
    //   page_done    [s] — consumer arrives (result ready), storer waits.
    //   page_consumed[s] — consumer arrives (input pages read), loader waits
    //                      before overwriting pages for the next instruction.
    //                      Initialized "consumed" (free) so the first
    //                      instruction's loader can proceed immediately.
    kittens::semaphore page_ready   [Config::NUM_PAGES];
    kittens::semaphore page_done    [Config::NUM_PAGES];
    kittens::semaphore page_consumed[Config::NUM_PAGES];

    // Per-variant scratch bytes. Ops that need a small fp32
    // accumulator, a softmax max/sum buffer, or a running-stat
    // reduction tile carve space out of this block; codegen
    // emits the offsets.
    alignas(128) uint8_t scratch[Config::SCRATCH_BYTES];
};

// Initialize every semaphore in a SharedState. Called once at
// kernel entry by thread 0 of the CTA; other threads wait on a
// CTA-wide barrier before first use.
//
// The transaction-count argument is 0 here — page_ready /
// page_done are used for thread/warp arrival signaling; ops
// that need TMA-transaction semaphores allocate them separately
// in scratch and init them with the tile's byte count.
template <typename Config>
__device__ __forceinline__ void init_shared_state(SharedState<Config>& ss) {
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < Config::NUM_PAGES; ++s) {
            kittens::init_semaphore(ss.page_ready   [s], 1);
            kittens::init_semaphore(ss.page_done    [s], 1);
            // page_consumed starts as "already consumed" (page is free).
            // Arriving immediately with count=1 flips the phase so the
            // first-instruction loader sees the page as available.
            kittens::init_semaphore(ss.page_consumed[s], 1);
            kittens::arrive(ss.page_consumed[s], 1);
        }
    }
    __syncthreads();
}

} // namespace ferrite
