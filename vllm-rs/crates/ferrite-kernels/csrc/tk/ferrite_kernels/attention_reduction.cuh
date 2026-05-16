// SPDX-License-Identifier: Apache-2.0
// Ferrite-owned TK 2.0 AttentionReduction op — 4 warp-role functions.
//
// Math: cross-split softmax rescale. Consumes `SPLITS` per-split
// partial attention outputs (produced by `attention_partial` with
// SPLITS > 1) plus their per-split log-sum-exp (LSE) statistics,
// and reduces them to the final per-(token, Q-head) attention
// output. For `s ∈ [0, SPLITS)`:
//   m_final = max_s m_partial[s]
//   l_final = sum_s exp(m_partial[s] - m_final) * l_partial[s]
//   O_final = sum_s exp(m_partial[s] - m_final) * l_partial[s] * O_partial[s]
//           / l_final
//
// In the `SPLITS == 1` configuration the math collapses to the
// identity: `O_final = O_partial[0]` and `l_final = l_partial[0]`,
// so the walker skips emitting this op's block entirely and uses
// the `attention_partial` output directly. This is the current
// Phase 3f-2d scope — ALL four role functions in this header are
// stubs fenced by `static_assert(SPLITS > 1, ...)`. The stub exists
// so:
//   1. `interpreter/variant_cpp::emit_op_block` has a stable op
//      name and role set to dispatch to when SPLITS > 1 lands.
//   2. The file's layout (page slots, bar IDs, template signature)
//      is committed in writing before any caller depends on it.
//   3. A walker that ever dispatches `AttentionReduction` at
//      `SPLITS == 1` fails to compile with a specific error,
//      not with a silent no-op that drops the op at codegen.
//
// Parallelism layout (target, for the SPLITS > 1 follow-up slice):
//   - Grid: one CTA per (token, Q head) — same shape as
//     `attention_partial` for SPLITS == 1, since this op fans IN
//     across splits for each head. Standalone walker would emit
//     `dim3(NUM_Q_HEADS, 1, 1)` and the bounds gate mirrors
//     `attention_partial.cuh`.
//   - Within CTA: four warp roles from ferrite_warp_roles.cuh.
//     - Loader warp: TMA-bulk loads the `SPLITS` O partials (each
//       HEAD_DIM bf16) + the `SPLITS` per-split (m, l) stats
//       (each fp32) into shared pages.
//     - Consumer warps: each warp owns HEAD_DIM/NUM_CONSUMER_WARPS
//       columns of O. Pass 1: every warp reduces across splits to
//       find m_final (scalar), l_final (scalar). Pass 2: each
//       warp rescales its O slice and divides by l_final.
//     - Launcher warp: empty.
//     - Storer warp: TMA-bulk-stores the finalized O row.
//
// Page layout (SPLITS > 1, target):
//   pages[base_stage + 0] — O_partials  [SPLITS, HEAD_DIM] bf16
//   pages[base_stage + 1] — m_partials  [SPLITS] fp32
//   pages[base_stage + 2] — l_partials  [SPLITS] fp32
//   pages[base_stage + 3] — O_final     [HEAD_DIM] bf16
//
// Semaphore handoff (target):
//   page_ready[base_stage + 0..2] — loader → consumer.
//   page_done [base_stage + 3]    — consumer → storer.
//
// bar IDs reserved for consumer-scoped syncs: 11, 12 (1..10 claimed
// by earlier ops — rms_norm 1/2, gemv/gemm 3/4, fused_add_rms_norm
// 5/6, fused_qkv_rope_cache 7/8, attention_partial 9/10).
//
// Scope caps for Phase 3f-2d-ii (this header):
//   SPLITS == 1  — identity op; reduction dispatch is skipped by
//                  the walker. A `SPLITS > 1` invocation here is
//                  the follow-up slice's job; the stub bodies
//                  fire a `static_assert` to catch any dispatch
//                  path that reaches them prematurely.
//   NUM_TOKENS == 1 — decode only, matching `attention_partial`.
//
// The template signature, page-slot constants, and bar IDs are
// committed here in the stub so 2d-iv can wire codegen dispatch
// symmetrically with `attention_partial`. Implementing the real
// bodies is a follow-up once a model / sequence length actually
// needs SPLITS > 1 (llama-3.2-1B at max seq 4096 / BLOCK_SIZE 16
// gives 256 pages — a single CTA is plenty; split-K only matters
// at long context / TP>1 where SM utilization drops below the
// work-queue).

#pragma once

#include "kittens.cuh"
#include "ferrite_globals.cuh"
#include "ferrite_substrate.cuh"

namespace ferrite {
namespace ops {
namespace attention_reduction {

// Page-slot offsets relative to `base_stage`. Reserved now so that
// walker code emitted by variant_cpp.rs can address them by name
// without guessing, even before the real bodies arrive.
constexpr int kOPartialsPageOff = 0;
constexpr int kMPartialsPageOff = 1;
constexpr int kLPartialsPageOff = 2;
constexpr int kOFinalPageOff    = 3;

// Consumer-scoped bar.sync IDs. Picked to avoid collision with
// bar 0 (__syncthreads) and with rms_norm (1/2), gemv/gemm (3/4),
// fused_add_rms_norm (5/6), fused_qkv_rope_cache (7/8),
// attention_partial (9/10).
constexpr int kConsumerBarPartial = 11;
constexpr int kConsumerBarPublish = 12;

// ---------- Loader --------------------------------------------------
//
// Stub. Real body (SPLITS > 1) will TMA-bulk-load O_partials +
// m_partials + l_partials from gmem into shared pages.
template <
    typename Config,
    int HEAD_DIM, int NUM_Q_HEADS,
    int NUM_TOKENS, int SPLITS
>
__device__ __forceinline__ void loader(
    ferrite::bf16_cptr     o_partials,   // [NUM_TOKENS, NUM_Q_HEADS, SPLITS, HEAD_DIM]
    ferrite::f32_cptr      m_partials,   // [NUM_TOKENS, NUM_Q_HEADS, SPLITS]
    ferrite::f32_cptr      l_partials,   // [NUM_TOKENS, NUM_Q_HEADS, SPLITS]
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    static_assert(SPLITS > 1,
                  "Phase 3f-2d-ii: AttentionReduction with SPLITS == 1 "
                  "is the identity — the walker must use the "
                  "attention_partial output directly and not dispatch "
                  "this op");
    static_assert(NUM_TOKENS == 1,
                  "Phase 3f-2d-ii: prefill (NUM_TOKENS > 1) not yet implemented");

    (void)o_partials;
    (void)m_partials;
    (void)l_partials;
    (void)ss;
    (void)base_stage;
    // Bounds gate kept for symmetry with the target implementation.
    if (blockIdx.x >= NUM_Q_HEADS) return;
    if (blockIdx.y >= 1) return;
}

// ---------- Consumer ------------------------------------------------
//
// Stub. Real body (SPLITS > 1) will:
//   1. Find m_final = max over splits.
//   2. Compute per-split weight `w_s = exp(m_s - m_final) * l_s`.
//   3. O_final[:] = sum_s w_s * O_partials[s, :] / sum_s w_s.
// Each consumer warp owns HEAD_DIM / NUM_CONSUMER_WARPS columns of
// O_final, so step 3 is embarrassingly parallel once step 1-2 have
// been broadcast through scratch.
template <
    typename Config,
    int HEAD_DIM, int NUM_Q_HEADS,
    int NUM_TOKENS, int SPLITS
>
__device__ __forceinline__ void consumer(
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int warp_in_role
) {
    static_assert(SPLITS > 1,
                  "Phase 3f-2d-ii: AttentionReduction with SPLITS == 1 "
                  "is the identity — the walker must use the "
                  "attention_partial output directly and not dispatch "
                  "this op");
    static_assert(NUM_TOKENS == 1,
                  "Phase 3f-2d-ii: prefill (NUM_TOKENS > 1) not yet implemented");

    (void)ss;
    (void)base_stage;
    (void)warp_in_role;
    if (blockIdx.x >= NUM_Q_HEADS) return;
    if (blockIdx.y >= 1) return;
}

// ---------- Launcher ------------------------------------------------
//
// Stub — empty on Hopper first-cut for role symmetry.
template <
    typename Config,
    int HEAD_DIM, int NUM_Q_HEADS,
    int NUM_TOKENS, int SPLITS
>
__device__ __forceinline__ void launcher(
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    static_assert(SPLITS > 1,
                  "Phase 3f-2d-ii: AttentionReduction with SPLITS == 1 "
                  "is the identity — the walker must use the "
                  "attention_partial output directly and not dispatch "
                  "this op");
    (void)ss;
    (void)base_stage;
    if (blockIdx.x >= NUM_Q_HEADS) return;
    if (blockIdx.y >= 1) return;
}

// ---------- Storer --------------------------------------------------
//
// Stub. Real body (SPLITS > 1) will TMA-bulk-store the finalized
// O row to `o_out[0, q_head, :]`.
template <
    typename Config,
    int HEAD_DIM, int NUM_Q_HEADS,
    int NUM_TOKENS, int SPLITS
>
__device__ __forceinline__ void storer(
    ferrite::bf16_ptr      o_out,         // [NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM]
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    static_assert(SPLITS > 1,
                  "Phase 3f-2d-ii: AttentionReduction with SPLITS == 1 "
                  "is the identity — the walker must use the "
                  "attention_partial output directly and not dispatch "
                  "this op");
    static_assert(NUM_TOKENS == 1,
                  "Phase 3f-2d-ii: prefill (NUM_TOKENS > 1) not yet implemented");

    (void)o_out;
    (void)ss;
    (void)base_stage;
    if (blockIdx.x >= NUM_Q_HEADS) return;
    if (blockIdx.y >= 1) return;
}

}  // namespace attention_reduction
}  // namespace ops
}  // namespace ferrite
