// SPDX-License-Identifier: Apache-2.0
// Ferrite-owned in-kernel argmax over bf16 logits.
//
// Called from CTA 0 ONLY (blockIdx.x == 0), after the lm_head storer
// has completed writing all VOCAB_SIZE logits to gmem. All threads in
// the block participate in the parallel reduction.
//
// Two-phase design:
//  Phase A: Each thread scans its slice of logits, tracking local
//           (max_val, min_idx_at_max). Warp-level shfl-xor reduces
//           to per-warp best.
//  Phase B: Lane-0 of each warp writes to scratch; thread 0 sweeps
//           the per-warp results to find the global argmax.
//
// Scratch layout (from ss.scratch, offset 0):
//   float    warp_maxes[num_warps]         — per-warp max values
//   uint32_t warp_idxs [num_warps]         — per-warp argmax indices
//
// Bytes used: num_warps × 8. For NCW=2 (5 warps total), 40 bytes.
// SCRATCH_BYTES for M=1 llama-3.2-1b is 1024 — plenty of headroom.

#pragma once

#include "kittens.cuh"

namespace ferrite {
namespace argmax {

// Compute argmax over `VOCAB_SIZE` bf16 values.
// Called with ALL threads in the block active (blockIdx.x == 0).
// Returns the argmax index; meaningful only at threadIdx.x == 0.
// After __syncthreads() inside, ALL threads see the result written
// to output_token_ids[step] / *next_input_ids (if non-null).
template <int VOCAB_SIZE>
__device__ __forceinline__ void compute_and_write(
    const __nv_bfloat16* __restrict__ logits,  // [VOCAB_SIZE] in gmem
    float*    scratch,          // ≥ num_warps × 8 bytes from ss.scratch
    uint32_t* output_token_ids, // [num_steps] output — writes [step]
    uint32_t* next_input_ids,   // nullptr on last step; else step+1 slot
    int       step
) {
    const int tid      = threadIdx.x;
    const int num_warps = blockDim.x / 32;
    const int warp_id  = tid / 32;

    // --- Phase A: per-thread scan + warp reduction ---------------------

    float    local_max = -INFINITY;
    uint32_t local_idx = 0;

    for (int i = tid; i < VOCAB_SIZE; i += blockDim.x) {
        float v = __bfloat162float(logits[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = (uint32_t)i;
        }
    }

    // Warp-level reduce: keep (max_val, min_idx_at_max) — ties broken
    // by taking the lower token index, matching greedy-argmax semantics.
    for (int offset = 16; offset > 0; offset >>= 1) {
        float    other_max = __shfl_xor_sync(0xFFFFFFFF, local_max, offset);
        uint32_t other_idx = __shfl_xor_sync(0xFFFFFFFF, local_idx, offset);
        if (other_max > local_max ||
            (other_max == local_max && other_idx < local_idx)) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }

    // --- Phase B: cross-warp reduction via scratch --------------------

    float*    warp_maxes = scratch;
    uint32_t* warp_idxs  = reinterpret_cast<uint32_t*>(scratch + num_warps);

    if ((tid & 31) == 0) {  // lane 0 of each warp
        warp_maxes[warp_id] = local_max;
        warp_idxs [warp_id] = local_idx;
    }
    __syncthreads();

    if (tid == 0) {
        float    best_max = warp_maxes[0];
        uint32_t best_idx = warp_idxs [0];
        for (int w = 1; w < num_warps; ++w) {
            if (warp_maxes[w] > best_max ||
                (warp_maxes[w] == best_max && warp_idxs[w] < best_idx)) {
                best_max = warp_maxes[w];
                best_idx = warp_idxs[w];
            }
        }
        output_token_ids[step] = best_idx;
        if (next_input_ids != nullptr) {
            *next_input_ids = best_idx;
        }
    }
    __syncthreads();
}

}  // namespace argmax
}  // namespace ferrite
