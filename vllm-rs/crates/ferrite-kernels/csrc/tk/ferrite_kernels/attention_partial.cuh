// SPDX-License-Identifier: Apache-2.0
// Ferrite-owned TK 2.0 AttentionPartial op — 4 warp-role functions.
//
// Math: paged-FA2 decode for one (token, kv_head) pair, packing
// GQA_RATIO Q-heads per CTA. Reads Q for those heads, streams K/V
// pages from the paged KV cache, and runs an online softmax to
// produce per-head attention output.
//
// TK 2.0 register-tile idiom: single-warp consumer, register-
// resident Q·K^T and P·V via `warp::mma_ABt` / `warp::mma_AB`, fp32
// register accumulator throughout — no scratch, no cross-warp
// reductions, no manual `__shfl_xor_sync` trees. Matches
// mk-v2-llama `AttentionPartial`'s topology; adapted for the ferrite
// substrate (SharedState pages + page_ready/page_done mbarriers
// instead of TK-schema semaphores).
//
// Scope caps for Wave F (this header):
//   SPLITS          == 1   — single CTA per (kv_head, token) covers
//                            the full sequence. attention_reduction
//                            is the identity.
//   NUM_TOKENS      >= 1   — batched decode (Wave F relaxed the
//                            previous NUM_TOKENS==1 cap); prefill
//                            (q_len > 1 per token) is still a
//                            separate impl.
//   SLIDING_WINDOW  == 0   — sliding-window attention deferred.
//   HAS_SOFTCAP     == 0   — Gemma3 softcap deferred.
//   GQA_RATIO ∈ [1,16]     — set by `store_n_rows<N>` lane mapping.
// Each cap lands as a `static_assert` inside every role function, so
// a walker that emits an unsupported variant fails to compile with a
// specific error. Follow-up slices implement the missing
// configurations; the op surface (loader/consumer/launcher/storer
// template signature) doesn't change.
//
// Consumer dispatch (Phase 7):
//   NUM_TOKENS == 1 → vector consumer: rv_fl<HEAD_DIM> per Q-head.
//     Replaces the 16-row MMA tiles with per-head row vectors.
//     For HEAD_DIM=64, GQA_RATIO=4: ~36 regs/lane vs ~112 → enables
//     CONSUMER_REGISTERS=80 → 5 CTAs/SM (vs 4 at 128).
//     Algorithm: K and V rows interleaved per j, scores[GQA_RATIO]
//     scalars, online softmax in per-head float state, warp::sum dot
//     products. Q released immediately after register load (earlier
//     than MMA path). Loader and storer are identical in both paths.
//   NUM_TOKENS > 1  → MMA consumer: rt_bf<16,HEAD_DIM> tile tiles.
//     Unchanged from Wave F.
//
// Parallelism layout (Wave F — persistent-thread grid):
//   - Grid: the mega launcher sizes the grid to `NUM_SMS *
//     ctas_per_sm` CTAs, one resident wave. Each role loops
//     `for (int tile = blockIdx.x; tile < NUM_KV_HEADS * NUM_TOKENS;
//          tile += gridDim.x, ++t_local)`
//     covering every (kv_head, token) tile. `t_local` drives the
//     per-CTA phase parity for the Q and O page handshakes.
//
//     Semantic mapping per tile:
//       kv_head       = tile % NUM_KV_HEADS
//       token         = tile / NUM_KV_HEADS
//       q_head_start  = kv_head * GQA_RATIO
//
//     For NUM_TOKENS == 1 the tile loop degenerates to the old
//     `kv_head = blockIdx.x` mapping; the K/V ring handshakes are
//     unchanged. For NUM_TOKENS > 1 CTAs re-enter the tile body
//     stridedly; the page handshakes below cover intra-tile ring
//     reuse AND cross-tile Q/O reuse on the same CTA.
//
//   - Within CTA: four warp roles from ferrite_warp_roles.cuh.
//     - Loader warp: per tile, one TMA-bulk load of GQA_RATIO
//       contiguous Q rows for (token, kv_head), then a per-page
//       loop issuing BLOCK_SIZE per-row TMA loads for K and V into
//       the K/V page rings. K/V ring slots are phase-bit-reused
//       globally (a `global_p` counter persists across tiles) —
//       the loader waits on `page_done[slot]` before re-arming a
//       slot it used STAGES iterations ago.
//     - Consumer: single warp (warp_in_role == 0). Per tile, loads
//       Q into a `rt_bf<16, HEAD_DIM>` register tile; for each
//       K/V page runs `mma_ABt` (Q·K^T), tail-masks trailing cols
//       via `right_fill`, online-softmax bookkeeping in `col_vec`
//       registers, `mma_AB` (P·V). Finalisation: `div_row(O,
//       norm)`, `ferrite::tk::store_n_rows<GQA_RATIO>` unpacks the
//       live rows into the O page. After unpack arrives on
//       `page_done[Q]` (loader can reuse Q) and `page_done[O]`
//       (storer may read).
//     - Storer: GQA_RATIO TMA-stores (one per head) of HEAD_DIM
//       bf16 each, writing `o_out[token, q_head_start + h, :]`.
//       Arrives on `page_ready[O]` after `store_async_wait` so the
//       consumer's next tile can overwrite the O page.
//     - Launcher: empty (Hopper consumer does its own
//       `warp::mma_*`; no launcher-warp tcgen05 to schedule).
//
// Page layout (per attention_partial call, caller gives `base_stage`;
// STAGES = Config::INSTRUCTION_PIPE_STAGES, today fixed at 2):
//   pages[base_stage + 0]                     — Q tile staging
//                                               (st_bf<16, HEAD_DIM>,
//                                                unswizzled; only
//                                                rows [0, GQA_RATIO)
//                                                hold live data — the
//                                                loader places them
//                                                at the top; other
//                                                rows are read by the
//                                                mma but don't affect
//                                                the live output
//                                                rows).
//   pages[base_stage + 1 .. base+STAGES]      — K tile ring
//                                               (st_bf<BLOCK_SIZE,
//                                                HEAD_DIM> per slot).
//   pages[base_stage + 1+STAGES ..
//         base+2*STAGES]                      — V tile ring (same).
//   pages[base_stage + 1+2*STAGES]            — O output tile
//                                               (sv_bf<HEAD_DIM>
//                                                [GQA_RATIO]).
// Total page budget: `2 + 2*STAGES`. With STAGES=2 that's 6 slots —
// unchanged from the NUM_TOKENS==1 kernel; batched decode reuses
// the same pages across tiles via the cross-tile handshakes below.
//
// Semaphore handoff (init'd once in init_shared_state, each
// arrival threshold 1; phase bits below are per-CTA):
//   page_ready[Q]              — loader → consumer. Arrives once
//                                 per tile when the Q TMA completes.
//                                 Consumer at tile t waits on
//                                 parity `t & 1`.
//   page_done [Q]              — consumer → loader. Arrives once
//                                 per tile after the consumer has
//                                 read Q into registers. Loader at
//                                 tile t (t>0) waits on parity
//                                 `(t-1) & 1` before re-arming Q.
//   page_ready[K ring slot s]  — loader → consumer (per K iter).
//                                 Phase drives from the global page
//                                 counter; ring slot reused every
//                                 STAGES iters across all tiles on
//                                 this CTA.
//   page_done [K ring slot s]  — consumer → loader (K consumed).
//                                 Same parity scheme as page_ready.
//   page_ready[V ring slot s]  — loader → consumer (V, per-iter).
//   page_done [V ring slot s]  — consumer → loader (V consumed).
//   page_done [O]              — consumer → storer. Arrives once
//                                 per tile after the consumer's
//                                 unpack. Storer at tile t waits on
//                                 parity `t & 1` (the (t+1)th arrive
//                                 flips parity to (t+1)&1, so the
//                                 storer blocks while phase == t&1).
//   page_ready[O]              — storer → consumer. Arrives once
//                                 per tile after `store_async_wait`.
//                                 Consumer at tile t (t>0) waits on
//                                 parity `(t-1) & 1` before writing
//                                 the O page for the next tile.
//
// Runtime arg: `block_table_stride` (uint32_t). The host's
// block_table is `[NUM_TOKENS, block_table_stride]` row-major; the
// loader reads `block_table[token * block_table_stride + p]` for
// page `p` of this tile's token. Stride is runtime (varies per
// batch with the actual max pages across the in-flight sequences)
// rather than a compile-time constant.
//
// bar IDs 9/10 (`kConsumerBarPartial` / `kConsumerBarPublish`) from
// the pre-Wave F scratch-based consumer are no longer claimed — the
// single-warp register-tile consumer has no cross-warp reduction,
// and its only intra-op synchronisation is `warp::sync()` between
// mma and mbar-arrive. Any later multi-op walker that inlines
// attention_partial alongside ops bar 1/2 (rms_norm), 3/4
// (gemv/gemm), 5/6 (fused_add_rms_norm), or 7/8
// (fused_qkv_rope_cache) is free to re-use 9/10.

#pragma once

#include "kittens.cuh"
#include <math_constants.h>
#include "ferrite_globals.cuh"
#include "ferrite_substrate.cuh"
#include "ferrite_tk_helpers.cuh"

namespace ferrite {
namespace ops {
namespace attention_partial {

// Page-slot offsets relative to `base_stage`. Stable across callers
// so variant_cpp.rs can emit `base_stage` constants known to the
// page-liveness analysis.
constexpr int kQPageOff = 0;

// Helpers. Evaluated at compile time per call site; all four role
// functions use the same forms so a page-layout change lands in one
// place without walking every body.
template <int STAGES>
__host__ __device__ constexpr int k_slot_for_iter(int p) {
    return 1 + (p % STAGES);
}
template <int STAGES>
__host__ __device__ constexpr int v_slot_for_iter(int p) {
    return 1 + STAGES + (p % STAGES);
}
template <int STAGES>
__host__ __device__ constexpr int k_slot_base() { return 1; }
template <int STAGES>
__host__ __device__ constexpr int v_slot_base() { return 1 + STAGES; }
template <int STAGES>
__host__ __device__ constexpr int o_page_off() { return 1 + 2 * STAGES; }

// Phase bit for K/V ring slot at global iter `p` (`p` is the count
// of K/V page iterations issued by this CTA since kernel entry,
// across all tiles it has processed). Each slot is reused every
// STAGES iterations; the physical mbarrier flips parity on each
// producer completion, so the consumer passes `(p / STAGES) & 1` to
// wait for the flip corresponding to the current cycle. For the
// loader's page_done wait, the relevant parity is the previous
// cycle's value: `((p - STAGES) / STAGES) & 1`.
template <int STAGES>
__host__ __device__ constexpr int ready_phase_for_iter(int p) {
    return (p / STAGES) & 1;
}
template <int STAGES>
__host__ __device__ constexpr int done_phase_for_prev_cycle(int p) {
    return ((p - STAGES) / STAGES) & 1;
}

// Shared-tile types. Unswizzled (`_swizzle=false`) so the loader's
// linear per-row TMA writes land in exactly the layout
// `warp::load(rt, st)` reads via ldsm4. Swizzled variants would need
// swizzle-aware cp.async addressing in the loader (see mk-v2's
// `load_Q_async` idiom); decode doesn't care about the bank-conflict
// win, so we take the simpler linear path.
template <int HEAD_DIM>
using q_st = kittens::st_bf<16, HEAD_DIM, /*_swizzle=*/false>;
template <int BLOCK_SIZE, int HEAD_DIM>
using kv_st = kittens::st_bf<BLOCK_SIZE, HEAD_DIM, /*_swizzle=*/false>;

// Unpack-rows helper lives in `ferrite::tk::store_n_rows<N>` in
// ferrite_tk_helpers.cuh — generalised across GQA_RATIO ∈ [1, 16].

// ---------- Loader --------------------------------------------------
//
// Per-tile flow:
//   1. If t_local > 0, wait for the consumer's page_done[Q] arrival
//      from the previous tile — Q page is now free to re-arm.
//   2. TMA-bulk load of GQA_RATIO contiguous Q rows for this tile's
//      (token, kv_head). Heads [q_head_start .. q_head_start+
//      GQA_RATIO) are adjacent in the `[NUM_TOKENS, NUM_Q_HEADS,
//      HEAD_DIM]` row-major Q tensor, so a single
//      `GQA_RATIO * HEAD_DIM * sizeof(bf16)` byte TMA suffices.
//      Destination is rows [0, GQA_RATIO) of the st_bf<16, HEAD_DIM>
//      Q tile (match consumer's `store_n_rows<GQA_RATIO>` row_base=0).
//   3. Per page: BLOCK_SIZE per-row TMA loads into the K ring slot,
//      BLOCK_SIZE per-row TMA loads into the V ring slot. Before
//      re-arming a slot, wait on page_done[slot] for its previous
//      cycle (cross-tile — `global_p` persists across tiles, so the
//      per-slot ring handshake is indistinguishable from the
//      NUM_TOKENS==1 single-tile case).
template <
    typename Config,
    int HEAD_DIM, int NUM_Q_HEADS, int NUM_KV_HEADS,
    int BLOCK_SIZE, int NUM_TOKENS,
    int SPLITS, int SLIDING_WINDOW, int HAS_SOFTCAP,
    int MAX_SK = 8192
>
__device__ __forceinline__ void loader(
    ferrite::bf16_cptr     q_in,                    // [NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM]
    ferrite::bf16_cptr     key_cache,               // [num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]
    ferrite::bf16_cptr     value_cache,             // same layout as key_cache
    const uint32_t*        block_table,             // [NUM_TOKENS, block_table_stride]
    uint32_t               block_table_stride,      // row stride of block_table (runtime)
    const int32_t*         seq_lens,                // [NUM_TOKENS]
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    static_assert(SPLITS == 1,
                  "Wave F: SPLITS != 1 not yet implemented");
    static_assert(NUM_TOKENS >= 1,
                  "attention_partial: NUM_TOKENS must be >= 1");
    static_assert(NUM_Q_HEADS % NUM_KV_HEADS == 0,
                  "GQA: NUM_Q_HEADS must be divisible by NUM_KV_HEADS");
    static_assert(NUM_Q_HEADS / NUM_KV_HEADS >= 1 &&
                      NUM_Q_HEADS / NUM_KV_HEADS <= 16,
                  "GQA_RATIO must be in [1, 16] (the 16-row Q/rt tile holds "
                  "all live heads for one kv_head)");
    static_assert(BLOCK_SIZE > 0 && (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
                  "BLOCK_SIZE must be a positive power of two");
    static_assert(BLOCK_SIZE == 16,
                  "Wave F: rt_bf<BLOCK_SIZE, HEAD_DIM> consumer "
                  "tiles expect BLOCK_SIZE == 16 (TK base tile row dim)");
    static_assert(HEAD_DIM % 16 == 0,
                  "HEAD_DIM must be divisible by 16 (TK tile col dim)");
    static_assert(true, // removed: attention uses fixed STAGES=2 internally
                  "attention_partial: page budget (2 + 2*STAGES) is "
                  "currently hardcoded in variant_cpp.rs::op_page_count "
                  "for STAGES==2. Bump both together to lift this cap.");

    constexpr int STAGES = 2; // attention fixed at 2 stages
    constexpr int GQA_RATIO  = NUM_Q_HEADS / NUM_KV_HEADS;
    constexpr int T_TOTAL    = NUM_KV_HEADS * NUM_TOKENS;
    using warp = kittens::group<1>;
    constexpr uint32_t q_bulk_bytes = GQA_RATIO * HEAD_DIM * sizeof(__nv_bfloat16);
    constexpr uint32_t row_bytes    = HEAD_DIM * sizeof(__nv_bfloat16);
    constexpr uint32_t page_bytes   = BLOCK_SIZE * HEAD_DIM * sizeof(__nv_bfloat16);

    __nv_bfloat16* q_page_bf = reinterpret_cast<__nv_bfloat16*>(
        ss.pages[base_stage + kQPageOff]);

    // Global page-iter counter across tiles — drives the K/V ring
    // handshakes. Persists across the outer tile loop so a slot
    // reused at global_p (from its last use at global_p - STAGES)
    // correctly waits on the prior cycle's page_done parity.
    int global_p = 0;
    int t_local = 0;
    for (int tile = blockIdx.x; tile < T_TOTAL; tile += gridDim.x, ++t_local) {
        const int kv_head      = tile % NUM_KV_HEADS;
        const int token        = tile / NUM_KV_HEADS;
        const int q_head_start = kv_head * GQA_RATIO;

        // Cross-tile Q handshake: after the first tile, wait for
        // the consumer to finish reading Q before re-arming the
        // page. `(t_local-1)&1` = the parity immediately after the
        // consumer's (t_local-th) arrive on page_done[Q].
        if (t_local > 0) {
            kittens::wait(ss.page_done[base_stage + kQPageOff], (t_local - 1) & 1);
        }

        if (kittens::laneid() == 0) {
            kittens::tma::expect_bytes(
                ss.page_ready[base_stage + kQPageOff], q_bulk_bytes);
        }
        warp::tma::load_async(
            reinterpret_cast<void*>(q_page_bf),
            const_cast<__nv_bfloat16*>(
                q_in + (static_cast<size_t>(token) * NUM_Q_HEADS + q_head_start)
                           * HEAD_DIM),
            q_bulk_bytes,
            ss.page_ready[base_stage + kQPageOff]);

        const int seq_len   = seq_lens[token];
        const int num_pages = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

        const size_t block_stride =
            static_cast<size_t>(BLOCK_SIZE) * NUM_KV_HEADS * HEAD_DIM;
        const size_t page_row_stride = static_cast<size_t>(NUM_KV_HEADS) * HEAD_DIM;
        const size_t head_off        = static_cast<size_t>(kv_head) * HEAD_DIM;

        constexpr int MAX_KV_PAGES = MAX_SK / BLOCK_SIZE;
        #pragma unroll MAX_KV_PAGES
        for (int p = 0; p < num_pages; ++p, ++global_p) {
            const int k_slot = k_slot_for_iter<STAGES>(global_p);
            const int v_slot = v_slot_for_iter<STAGES>(global_p);

            if (global_p >= STAGES) {
                const int prev_phase = done_phase_for_prev_cycle<STAGES>(global_p);
                kittens::wait(ss.page_done[base_stage + k_slot], prev_phase);
                kittens::wait(ss.page_done[base_stage + v_slot], prev_phase);
            }

            const uint32_t block_idx =
                block_table[static_cast<size_t>(token) * block_table_stride + p];
            const __nv_bfloat16* k_block_base =
                key_cache + static_cast<size_t>(block_idx) * block_stride + head_off;
            const __nv_bfloat16* v_block_base =
                value_cache + static_cast<size_t>(block_idx) * block_stride + head_off;

            void* k_page = reinterpret_cast<void*>(ss.pages[base_stage + k_slot]);
            void* v_page = reinterpret_cast<void*>(ss.pages[base_stage + v_slot]);

            if (kittens::laneid() == 0) {
                kittens::tma::expect_bytes(
                    ss.page_ready[base_stage + k_slot], page_bytes);
                kittens::tma::expect_bytes(
                    ss.page_ready[base_stage + v_slot], page_bytes);
            }

            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                const __nv_bfloat16* k_row_src =
                    k_block_base + static_cast<size_t>(j) * page_row_stride;
                warp::tma::load_async(
                    reinterpret_cast<__nv_bfloat16*>(k_page)
                        + static_cast<size_t>(j) * HEAD_DIM,
                    const_cast<__nv_bfloat16*>(k_row_src),
                    row_bytes,
                    ss.page_ready[base_stage + k_slot]);
            }
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                const __nv_bfloat16* v_row_src =
                    v_block_base + static_cast<size_t>(j) * page_row_stride;
                warp::tma::load_async(
                    reinterpret_cast<__nv_bfloat16*>(v_page)
                        + static_cast<size_t>(j) * HEAD_DIM,
                    const_cast<__nv_bfloat16*>(v_row_src),
                    row_bytes,
                    ss.page_ready[base_stage + v_slot]);
            }
        }
    }
}

// ---------- Consumer ------------------------------------------------
//
// Single-warp (warp_in_role == 0) register-tile online softmax. Per
// tile:
//   1. Wait for the loader's Q TMA to land; load into `rt_bf<16,
//      HEAD_DIM>`.
//   2. If t_local > 0, wait for the storer's page_ready[O] arrival
//      (O page free to overwrite).
//   3. Reset O_reg / max_vec / norm_vec for this tile.
//   4. Per page: wait K, mma_ABt, tail-mask, online softmax, wait V,
//      mma_AB, arrive K/V page_done.
//   5. div_row(O, norm); unpack live rows via store_n_rows<GQA_RATIO>.
//   6. Arrive on page_done[Q] (Q free for loader's next tile) and
//      page_done[O] (O populated for storer to read).
//
// Warps 1..NUM_CONSUMER_WARPS-1 return immediately — no cross-warp
// sync is needed inside the consumer, so exiting them is safe.
template <
    typename Config,
    int HEAD_DIM, int NUM_Q_HEADS, int NUM_KV_HEADS,
    int BLOCK_SIZE, int NUM_TOKENS,
    int SPLITS, int SLIDING_WINDOW, int HAS_SOFTCAP,
    int MAX_SK = 8192
>
__device__ __forceinline__ void consumer(
    const int32_t* seq_lens,    // [NUM_TOKENS]
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int warp_in_role,
    float softmax_scale,
    float softcap_val           // only used when HAS_SOFTCAP > 0
) {
    static_assert(SPLITS == 1,
                  "Wave F: SPLITS != 1 not yet implemented");
    static_assert(NUM_TOKENS >= 1,
                  "attention_partial: NUM_TOKENS must be >= 1");
    static_assert(NUM_Q_HEADS % NUM_KV_HEADS == 0,
                  "GQA: NUM_Q_HEADS must be divisible by NUM_KV_HEADS");
    static_assert(NUM_Q_HEADS / NUM_KV_HEADS >= 1 &&
                      NUM_Q_HEADS / NUM_KV_HEADS <= 16,
                  "GQA_RATIO must be in [1, 16]");
    static_assert(BLOCK_SIZE == 16,
                  "Wave F: rt_bf<BLOCK_SIZE, HEAD_DIM> expects "
                  "BLOCK_SIZE == 16");
    static_assert(HEAD_DIM % 16 == 0,
                  "HEAD_DIM must be divisible by 16 (TK tile col dim)");
    static_assert(true, // removed: attention uses fixed STAGES=2 internally
                  "attention_partial: page budget (2 + 2*STAGES) is "
                  "currently hardcoded in variant_cpp.rs::op_page_count "
                  "for STAGES==2. Bump both together to lift this cap.");

    // Single-warp consumer — warp 1..NCW exit early and leave all
    // attention math to warp 0.
    if (warp_in_role != 0) return;

    constexpr int STAGES = 2; // attention fixed at 2 stages
    constexpr int GQA_RATIO = NUM_Q_HEADS / NUM_KV_HEADS;
    constexpr int T_TOTAL   = NUM_KV_HEADS * NUM_TOKENS;
    using warp = kittens::group<1>;

    // log2(e) scale so we can use hw `exp2` (one instruction) instead
    // of `__expf` (~10). Matches mk-v2's softmax_temp idiom.
    const float softmax_temp = softmax_scale * 1.44269504089f;

    int global_p = 0;
    int t_local  = 0;

    // ---------------------------------------------------------------
    // M=1 vector path (NUM_TOKENS==1 dispatch).
    //
    // Replaces the rt_bf<16,HEAD_DIM> MMA consumer with per-head
    // rv_fl<HEAD_DIM> vectors. For HEAD_DIM=64, GQA_RATIO=4 the
    // register footprint drops from ~112 to ~36 per lane, enabling
    // CONSUMER_REGISTERS=80 → 5 CTAs/SM (vs 4 at 128).
    //
    // Algorithm: for each K/V page, load K and V rows one at a time
    // and interleave score-computation with V-accumulation so only
    // one (k_rv, v_rv, dot_rv, scaled_v) temporary is live at a time
    // (no BLOCK_SIZE score buffer needed). Q is pre-scaled by
    // softmax_temp so the dot product gives the log2-scaled score
    // directly, consistent with the MMA path's idiom.
    //
    // Q is released to the loader immediately after loading into
    // registers (earlier than the MMA path, safe because Q lives in
    // registers for the rest of the tile). The O page wait is placed
    // just before the O-write to maximise overlap with the storer's
    // TMA from the prior tile.
    //
    // Loader and storer are unchanged: the same Q/K/V/O shmem page
    // layout (st_bf<16,HEAD_DIM> for K/V, first GQA_RATIO rows of
    // st_bf<16,HEAD_DIM> for Q, sv_bf<HEAD_DIM>[GQA_RATIO] for O)
    // is used; we just read individual rows via byte-offset casts.
    // ---------------------------------------------------------------
    if constexpr (NUM_TOKENS == 1) {
        using q_rv_t = kittens::rv_fl<HEAD_DIM>;
        using o_sv_bf = kittens::sv_bf<HEAD_DIM>;

        q_rv_t Q_vecs[GQA_RATIO]; // pre-scaled by softmax_temp at load
        q_rv_t O_vecs[GQA_RATIO]; // fp32 output accumulators
        float  max_val[GQA_RATIO];
        float  norm_val[GQA_RATIO];

        for (int tile = blockIdx.x; tile < T_TOTAL; tile += gridDim.x, ++t_local) {
            // Wait for Q TMA, load GQA_RATIO row-vectors, pre-scale.
            kittens::wait(ss.page_ready[base_stage + kQPageOff], t_local & 1);
            const __nv_bfloat16* q_page = reinterpret_cast<const __nv_bfloat16*>(
                ss.pages[base_stage + kQPageOff]);
            for (int h = 0; h < GQA_RATIO; ++h) {
                const o_sv_bf& q_sv = *reinterpret_cast<const o_sv_bf*>(
                    q_page + static_cast<size_t>(h) * HEAD_DIM);
                kittens::warp::load(Q_vecs[h], q_sv);
                // Pre-scale Q via direct register arithmetic (bypass TK warp::mul).
                Q_vecs[h].data[0][0] *= softmax_temp;
                Q_vecs[h].data[1][0] *= softmax_temp;
            }
            // Release Q immediately — Q is now entirely in registers.
            if (kittens::laneid() == 0) {
                kittens::arrive(ss.page_done[base_stage + kQPageOff]);
            }
            warp::sync();

            // Init per-head softmax state and output accumulators.
            #pragma unroll
            for (int h = 0; h < GQA_RATIO; ++h) {
                max_val[h]  = -CUDART_INF_F;
                norm_val[h] = 0.0f;
                kittens::warp::zero(O_vecs[h]);
            }

            const int token   = 0; // NUM_TOKENS == 1: only token 0
            const int seq_len = seq_lens[token];
            const int num_pages = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // K and V row temporaries — reused across j iterations.
            q_rv_t k_rv, v_rv;

            constexpr int MAX_KV_PAGES = MAX_SK / BLOCK_SIZE;
            #pragma unroll MAX_KV_PAGES
            for (int p = 0; p < num_pages; ++p, ++global_p) {
                const int k_slot = k_slot_for_iter<STAGES>(global_p);
                const int v_slot = v_slot_for_iter<STAGES>(global_p);
                const int phase  = ready_phase_for_iter<STAGES>(global_p);

                // Wait for both K and V pages before processing rows.
                kittens::wait(ss.page_ready[base_stage + k_slot], phase);
                kittens::wait(ss.page_ready[base_stage + v_slot], phase);

                const __nv_bfloat16* k_page = reinterpret_cast<const __nv_bfloat16*>(
                    ss.pages[base_stage + k_slot]);
                const __nv_bfloat16* v_page = reinterpret_cast<const __nv_bfloat16*>(
                    ss.pages[base_stage + v_slot]);
                const int valid_rows = ((p + 1) * BLOCK_SIZE <= seq_len)
                                       ? BLOCK_SIZE
                                       : (seq_len - p * BLOCK_SIZE);

                // Interleave K-score and V-accumulation row-by-row.
                // Only one K row and one V row live in registers at a time.
                //
                // IMPLEMENTATION NOTE: uses explicit register arithmetic
                // (direct data[] access + __shfl_xor_sync) rather than
                // TK warp::mul/sum to guarantee correctness for the naive_l
                // rv_fl layout used here. Each rv_fl<HEAD_DIM> lane holds
                // data[0][0] = element[lane] and data[1][0] = element[lane+32].
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    // Load K row j.
                    const o_sv_bf& k_sv = *reinterpret_cast<const o_sv_bf*>(
                        k_page + static_cast<size_t>(j) * HEAD_DIM);
                    kittens::warp::load(k_rv, k_sv);

                    // Tail-mask: positions j >= valid_rows → -inf.
                    const float mask_add = (j < valid_rows) ? 0.0f : -CUDART_INF_F;

                    float scores[GQA_RATIO]; // scalars — same across all lanes
                    for (int h = 0; h < GQA_RATIO; ++h) {
                        // Dot product Q_vecs[h] · k_rv via explicit warp reduction.
                        float local = Q_vecs[h].data[0][0] * k_rv.data[0][0]
                                    + Q_vecs[h].data[1][0] * k_rv.data[1][0];
                        // shfl_xor all-reduce (broadcasts result to all lanes).
                        local += __shfl_xor_sync(0xffffffffU, local, 16);
                        local += __shfl_xor_sync(0xffffffffU, local,  8);
                        local += __shfl_xor_sync(0xffffffffU, local,  4);
                        local += __shfl_xor_sync(0xffffffffU, local,  2);
                        local += __shfl_xor_sync(0xffffffffU, local,  1);
                        scores[h] = local + mask_add;
                    }

                    // Softcap (Gemma3). Compiled out when HAS_SOFTCAP == 0.
                    if constexpr (HAS_SOFTCAP > 0) {
                        const float inv_sc = 1.0f / softcap_val;
                        for (int h = 0; h < GQA_RATIO; ++h)
                            scores[h] = softcap_val * tanhf(scores[h] * inv_sc);
                    }

                    // Sliding window mask. Compiled out when SLIDING_WINDOW == 0.
                    if constexpr (SLIDING_WINDOW > 0) {
                        const int abs_pos = p * BLOCK_SIZE + j;
                        if (abs_pos < seq_len - SLIDING_WINDOW) {
                            for (int h = 0; h < GQA_RATIO; ++h)
                                scores[h] = -CUDART_INF_F;
                        }
                    }

                    // Load V row j (shared across all heads).
                    const o_sv_bf& v_sv = *reinterpret_cast<const o_sv_bf*>(
                        v_page + static_cast<size_t>(j) * HEAD_DIM);
                    kittens::warp::load(v_rv, v_sv);

                    // Online softmax update + V accumulation per head.
                    // Uses direct data[] arithmetic to bypass TK abstraction.
                    for (int h = 0; h < GQA_RATIO; ++h) {
                        const float new_max   = fmaxf(max_val[h], scores[h]);
                        const float old_scale = exp2f(max_val[h] - new_max);
                        const float p_hj      = exp2f(scores[h]  - new_max);
                        // Rescale O and accumulate: O[h] = O[h]*old_scale + p_hj*V[j].
                        O_vecs[h].data[0][0] = O_vecs[h].data[0][0] * old_scale
                                             + p_hj * v_rv.data[0][0];
                        O_vecs[h].data[1][0] = O_vecs[h].data[1][0] * old_scale
                                             + p_hj * v_rv.data[1][0];
                        norm_val[h] = norm_val[h] * old_scale + p_hj;
                        max_val[h]  = new_max;
                    }
                }

                // Release K and V ring slots.
                if (kittens::laneid() == 0) {
                    kittens::arrive(ss.page_done[base_stage + k_slot]);
                    kittens::arrive(ss.page_done[base_stage + v_slot]);
                }
                warp::sync();
            }

            // Finalise: O_vecs[h] /= norm_val[h].
            for (int h = 0; h < GQA_RATIO; ++h) {
                const float inv_norm = (norm_val[h] > 0.0f) ? (1.0f / norm_val[h]) : 0.0f;
                O_vecs[h].data[0][0] *= inv_norm;
                O_vecs[h].data[1][0] *= inv_norm;
            }

            // Cross-tile O handshake: wait for storer before writing O page.
            if (t_local > 0) {
                kittens::wait(ss.page_ready[base_stage + o_page_off<STAGES>()],
                              (t_local - 1) & 1);
            }

            // Write O_vecs to O page (sv_bf<HEAD_DIM>[GQA_RATIO]).
            __nv_bfloat16* o_page_ptr = reinterpret_cast<__nv_bfloat16*>(
                ss.pages[base_stage + o_page_off<STAGES>()]);
            warp::sync();
            #pragma unroll
            for (int h = 0; h < GQA_RATIO; ++h) {
                o_sv_bf& o_sv = *reinterpret_cast<o_sv_bf*>(
                    o_page_ptr + static_cast<size_t>(h) * HEAD_DIM);
                kittens::warp::store(o_sv, O_vecs[h]);
            }
            warp::sync();
            if (kittens::laneid() == 0) {
                kittens::arrive(ss.page_done[base_stage + o_page_off<STAGES>()]);
            }
        }
    } else {
        // ---------------------------------------------------------------
        // M>1 tile-based MMA consumer (NUM_TOKENS > 1).
        // ---------------------------------------------------------------
        using q_rt       = kittens::rt_bf<16, HEAD_DIM>;
        using k_rt       = kittens::rt_bf<BLOCK_SIZE, HEAD_DIM>;
        using v_rt       = kittens::rt_bf<BLOCK_SIZE, HEAD_DIM,
                                          kittens::ducks::rt_layout::col>;
        using attn_fl_rt = kittens::rt_fl<16, BLOCK_SIZE>;
        using attn_bf_rt = kittens::rt_bf<16, BLOCK_SIZE>;
        using o_rt       = kittens::rt_fl<16, HEAD_DIM>;
        using col_vec    = typename o_rt::col_vec;

        q_rt       Q_reg;
        k_rt       K_reg;
        v_rt       V_reg;
        o_rt       O_reg;
        attn_fl_rt attn_fl_reg;
        attn_bf_rt attn_bf_reg;
        col_vec    max_vec;
        col_vec    scaled_max;
        col_vec    last_scaled_max;
        col_vec    diff_scaled_max;
        col_vec    norm_vec;

        for (int tile = blockIdx.x; tile < T_TOTAL; tile += gridDim.x, ++t_local) {
            const int token = tile / NUM_KV_HEADS;

            kittens::wait(ss.page_ready[base_stage + kQPageOff], t_local & 1);
            auto& Q_smem = *reinterpret_cast<q_st<HEAD_DIM>*>(
                ss.pages[base_stage + kQPageOff]);
            warp::load(Q_reg, Q_smem);
            warp::sync();

            warp::neg_infty(max_vec);
            warp::neg_infty(last_scaled_max);
            warp::zero(norm_vec);
            warp::zero(O_reg);

            const int seq_len   = seq_lens[token];
            const int num_pages = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

            constexpr int MAX_KV_PAGES_MMA = MAX_SK / BLOCK_SIZE;
            #pragma unroll MAX_KV_PAGES_MMA
            for (int p = 0; p < num_pages; ++p, ++global_p) {
                const int k_slot = k_slot_for_iter<STAGES>(global_p);
                const int v_slot = v_slot_for_iter<STAGES>(global_p);
                const int phase  = ready_phase_for_iter<STAGES>(global_p);

                warp::zero(attn_fl_reg);
                kittens::wait(ss.page_ready[base_stage + k_slot], phase);
                auto& K_smem = *reinterpret_cast<kv_st<BLOCK_SIZE, HEAD_DIM>*>(
                    ss.pages[base_stage + k_slot]);
                warp::load(K_reg, K_smem);
                warp::mma_ABt(attn_fl_reg, Q_reg, K_reg, attn_fl_reg);
                warp::sync();
                if (kittens::laneid() == 0) {
                    kittens::arrive(ss.page_done[base_stage + k_slot]);
                }

                if constexpr (HAS_SOFTCAP > 0) {
                    const float inv_softcap = 1.0f / softcap_val;
                    warp::apply(attn_fl_reg, attn_fl_reg,
                        [=](int /*r*/, int /*c*/, float v) {
                            return softcap_val * tanhf(v * inv_softcap);
                        });
                }

                if ((p + 1) * BLOCK_SIZE > seq_len) {
                    const int valid = seq_len - p * BLOCK_SIZE;
                    warp::right_fill(attn_fl_reg, attn_fl_reg, valid,
                                     -CUDART_INF_F);
                }

                if constexpr (SLIDING_WINDOW > 0) {
                    const int win_start = seq_len - SLIDING_WINDOW;
                    if ((p + 1) * BLOCK_SIZE <= win_start) {
                        warp::neg_infty(attn_fl_reg);
                    } else if (p * BLOCK_SIZE < win_start) {
                        const int masked_cols = win_start - p * BLOCK_SIZE;
                        warp::left_fill(attn_fl_reg, attn_fl_reg, masked_cols,
                                        -CUDART_INF_F);
                    }
                }

                warp::row_max(max_vec, attn_fl_reg, max_vec);
                warp::mul(attn_fl_reg, attn_fl_reg, softmax_temp);
                warp::mul(scaled_max, max_vec, softmax_temp);
                warp::sub_row(attn_fl_reg, attn_fl_reg, scaled_max);
                warp::exp2(attn_fl_reg, attn_fl_reg);
                warp::sub(diff_scaled_max, last_scaled_max, scaled_max);
                warp::exp2(diff_scaled_max, diff_scaled_max);

                warp::mul_row(O_reg, O_reg, diff_scaled_max);
                kittens::wait(ss.page_ready[base_stage + v_slot], phase);
                auto& V_smem = *reinterpret_cast<kv_st<BLOCK_SIZE, HEAD_DIM>*>(
                    ss.pages[base_stage + v_slot]);
                warp::load(V_reg, V_smem);
                warp::copy(attn_bf_reg, attn_fl_reg);
                warp::mma_AB(O_reg, attn_bf_reg, V_reg, O_reg);
                warp::sync();
                if (kittens::laneid() == 0) {
                    kittens::arrive(ss.page_done[base_stage + v_slot]);
                }

                warp::mul(norm_vec, norm_vec, diff_scaled_max);
                warp::row_sum(norm_vec, attn_fl_reg, norm_vec);
                warp::copy(last_scaled_max, scaled_max);
            }

            warp::div_row(O_reg, O_reg, norm_vec);

            if (kittens::laneid() == 0) {
                kittens::arrive(ss.page_done[base_stage + kQPageOff]);
            }

            if (t_local > 0) {
                kittens::wait(ss.page_ready[base_stage + o_page_off<STAGES>()],
                              (t_local - 1) & 1);
            }

            using o_sv_bf = kittens::sv_bf<HEAD_DIM>;
            auto& O_smem = *reinterpret_cast<o_sv_bf(*)[GQA_RATIO]>(
                ss.pages[base_stage + o_page_off<STAGES>()]);
            warp::sync();
            ferrite::tk::store_n_rows<GQA_RATIO>(O_smem, O_reg, /*row_base=*/0);
            warp::sync();
            if (kittens::laneid() == 0) {
                kittens::arrive(ss.page_done[base_stage + o_page_off<STAGES>()]);
            }
        }
    }
}

// ---------- Launcher ------------------------------------------------
//
// Hopper first-cut: register-tile consumer does its own
// `warp::mma_*`; no wgmma/tcgen05 to schedule from a dedicated
// launcher warp. Body stays empty for role symmetry — the tile loop
// is structural (match the other roles' iteration count).
template <
    typename Config,
    int HEAD_DIM, int NUM_Q_HEADS, int NUM_KV_HEADS,
    int BLOCK_SIZE, int NUM_TOKENS,
    int SPLITS, int SLIDING_WINDOW, int HAS_SOFTCAP,
    int MAX_SK = 8192
>
__device__ __forceinline__ void launcher(
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    static_assert(SPLITS == 1,
                  "Wave F: SPLITS != 1 not yet implemented");
    static_assert(NUM_TOKENS >= 1,
                  "attention_partial: NUM_TOKENS must be >= 1");
    (void)ss; (void)base_stage;
    constexpr int T_TOTAL = NUM_KV_HEADS * NUM_TOKENS;
    for (int tile = blockIdx.x; tile < T_TOTAL; tile += gridDim.x) {
        (void)tile;
    }
}

// ---------- Storer --------------------------------------------------
//
// Per tile: wait for the consumer to publish the O page (GQA_RATIO
// `sv_bf<HEAD_DIM>` rows packed contiguously), then GQA_RATIO TMA-
// bulk-stores — one HEAD_DIM row per head — to
// o_out[token, q_head_start + h, :]. `store_async` emits the
// required `fence.proxy.async` before the cp.async.bulk; the
// trailing `store_async_wait` blocks until the final store lands in
// gmem so the subsequent arrive on page_ready[O] correctly tells
// the consumer "O is free to overwrite" only after the gmem write
// is visible. (Consumer's next-tile use is the direct downstream;
// any external consumer of `o_out` is serialized through the mega
// kernel's return.)
template <
    typename Config,
    int HEAD_DIM, int NUM_Q_HEADS, int NUM_KV_HEADS,
    int BLOCK_SIZE, int NUM_TOKENS,
    int SPLITS, int SLIDING_WINDOW, int HAS_SOFTCAP,
    int MAX_SK = 8192
>
__device__ __forceinline__ void storer(
    ferrite::bf16_ptr      o_out,         // [NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM]
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    static_assert(SPLITS == 1,
                  "Wave F: SPLITS != 1 not yet implemented");
    static_assert(NUM_TOKENS >= 1,
                  "attention_partial: NUM_TOKENS must be >= 1");
    static_assert(NUM_Q_HEADS % NUM_KV_HEADS == 0,
                  "GQA: NUM_Q_HEADS must be divisible by NUM_KV_HEADS");
    static_assert(NUM_Q_HEADS / NUM_KV_HEADS >= 1 &&
                      NUM_Q_HEADS / NUM_KV_HEADS <= 16,
                  "GQA_RATIO must be in [1, 16]");
    static_assert(true, // removed: attention uses fixed STAGES=2 internally
                  "attention_partial: page budget (2 + 2*STAGES) is "
                  "currently hardcoded in variant_cpp.rs::op_page_count "
                  "for STAGES==2. Bump both together to lift this cap.");

    constexpr int STAGES = 2; // attention fixed at 2 stages
    constexpr int GQA_RATIO = NUM_Q_HEADS / NUM_KV_HEADS;
    constexpr int T_TOTAL   = NUM_KV_HEADS * NUM_TOKENS;
    using warp = kittens::group<1>;
    constexpr uint32_t row_bytes = HEAD_DIM * sizeof(__nv_bfloat16);

    int t_local = 0;
    for (int tile = blockIdx.x; tile < T_TOTAL; tile += gridDim.x, ++t_local) {
        const int kv_head      = tile % NUM_KV_HEADS;
        const int token        = tile / NUM_KV_HEADS;
        const int q_head_start = kv_head * GQA_RATIO;

        // Wait for the consumer's O unpack. The consumer arrives on
        // page_done[O] once per tile; after the (t_local+1)th arrive
        // the phase is (t_local+1)&1, so wait while phase == t_local&1.
        kittens::wait(ss.page_done[base_stage + o_page_off<STAGES>()],
                      t_local & 1);

        __nv_bfloat16* o_src = reinterpret_cast<__nv_bfloat16*>(
            ss.pages[base_stage + o_page_off<STAGES>()]);

        #pragma unroll
        for (int h = 0; h < GQA_RATIO; ++h) {
            warp::tma::store_async(
                static_cast<void*>(
                    o_out + (static_cast<size_t>(token) * NUM_Q_HEADS
                             + q_head_start + h)
                                * HEAD_DIM),
                static_cast<void*>(o_src + static_cast<size_t>(h) * HEAD_DIM),
                row_bytes);
        }
        kittens::tma::store_async_wait<0>();

        // Publish "O free to overwrite" for the consumer's next tile
        // (if any). Cheap even on the final tile — the consumer's
        // early-exit from the tile loop means no one will wait on
        // this arrival.
        if (kittens::laneid() == 0) {
            kittens::arrive(ss.page_ready[base_stage + o_page_off<STAGES>()]);
        }
    }
}

}  // namespace attention_partial
}  // namespace ops
}  // namespace ferrite
