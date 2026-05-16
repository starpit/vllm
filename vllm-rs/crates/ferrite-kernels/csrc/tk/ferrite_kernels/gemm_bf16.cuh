// SPDX-License-Identifier: Apache-2.0
// Ferrite-owned TK 2.0 gemm_bf16 op — 4 warp-role functions.
//
// Wave E: compile-time dispatch on NUM_CONSUMER_WARPS.
//
// Path A (NCW == 4, head_dim ≥ 128): warpgroup::mma_ABt + swizzled tiles.
//   Uses WGMMA hardware (sm_90a).  Loader fills swizzled st_bf<64,kKChunk>
//   pages via per-chunk idx() addressing.  Consumer calls
//   warpgroup::mma_ABt(acc, a_smem, b_smem) + mma_async_wait + warpgroup::store.
//   Verified: M=8/64/128/NCW=4 mismatches=0 on H100 sm_90a.
//
// Path B (NCW < 4, head_dim = 64): warp-level mma_ABt + unswizzled tiles.
//   Exact pre-Wave-E implementation (Phase 5 tile loop, kMTile=64 hardcoded).
//   Loader: raw linear loads into st_bf<64,kKChunk,false> pages.
//   Consumer: per-warp warp::load + warp::mma_ABt (N_CHUNKS × K_inner) +
//   warp::store to scratch + linear read to global.
//   Verified: M=8/NCW=2 mismatches=0 on H100; M=8 decode E2E verified.
//
// Math: `out[M, N] = x[M, K] @ W[N, K]^T` (row-major bf16 weights).
// bf16 in/out, fp32 accumulator.
//
// Tile shape:
//   kKChunk = 64  — K-reduction step.
//   kNTile  = 64  — N cols per CTA.
//   kMTile  = 64  — A-tile height (for both paths).
//   kMChunk = 16  — M rows per consumer warp.
//
// Phase 5 tile loop: flat blockIdx.x stride over all N_TILES×M_TILES.
//
// Static constraints:
//   K % kKChunk == 0, N % kNTile == 0 (both paths).
//   Path A: PAGE_SIZE >= kMTile*kKChunk*2; SCRATCH_BYTES >= kMTile*kNTile*2.
//   Path B: PAGE_SIZE >= kMTile*kKChunk*2; SCRATCH_BYTES >= kMTile*kNTile*2.
//   NUM_CONSUMER_WARPS >= 2.
//
// ABI: consumer() takes `out` as its FIRST argument.

#pragma once

#include "kittens.cuh"
#include "ferrite_globals.cuh"
#include "ferrite_substrate.cuh"

namespace ferrite {
namespace ops {
namespace gemm_bf16 {

constexpr int kKChunk = 64;
constexpr int kNTile  = 64;
constexpr int kNChunk = 16;
constexpr int kMTile  = 64;
constexpr int kMChunk = 16;

constexpr int kActPageOff    = 0;
constexpr int kWeightPageOff = 1;

// ── Path B types (unswizzled, warp-level mma_ABt) ──────────────────────────
template <int K_CHUNK>
using a_tile_st = kittens::st_bf<kMTile, K_CHUNK, false>;
template <int K_CHUNK>
using b_tile_st = kittens::st_bf<kNTile, K_CHUNK, false>;
template <int K_CHUNK>
using a_warp_st = kittens::st_bf<kMChunk, K_CHUNK, false>;
template <int K_CHUNK>
using b_warp_st = kittens::st_bf<kNChunk, K_CHUNK, false>;
template <int K_CHUNK>
using a_rt = kittens::rt_bf<kMChunk, K_CHUNK>;
template <int K_CHUNK>
using b_rt = kittens::rt_bf<kNChunk, K_CHUNK>;
using acc_rt    = kittens::rt_fl<kMChunk, kNChunk>;
using out_warp_st = kittens::st_bf<kMChunk, kNTile, false>;
using out_rt    = kittens::rt_bf<kMChunk, kNTile>;

// ── Path A types (swizzled, wgmma) ─────────────────────────────────────────
template <int K_CHUNK>
using wg_a_smem_t = kittens::st_bf<kMTile, K_CHUNK>;          // swizzled
template <int K_CHUNK>
using wg_b_smem_t = kittens::st_bf<kNTile, K_CHUNK>;          // swizzled
using wg_scratch_t = kittens::st_bf<kMTile, kNTile, false>;   // unswizzled output
using wg_acc_rt    = kittens::rt_fl<kMChunk, kNTile>;          // per-warp wgmma acc

// Barrier IDs shared by both paths.
constexpr int kConsumerBarMma = 5;
constexpr int kConsumerBarOut = 6;

// ==========================================================================
// Loader
// ==========================================================================
//
// Path A: loads swizzled 64×64 A and B tiles via per-chunk idx() calls.
// Path B: raw linear element-by-element stores into unswizzled pages.
// Both use kMTile=64 rows for A (M_TILES=ceil(M/64) tiles).

template <typename Config, int K, int N, int M>
__device__ __forceinline__ void loader(
    ferrite::bf16_cptr x,
    ferrite::bf16_cptr W,
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    static_assert(K % kKChunk == 0, "gemm_bf16: K not divisible by kKChunk");
    static_assert(N % kNTile  == 0, "gemm_bf16: N not divisible by kNTile");
    static_assert(Config::PAGE_SIZE >= kMTile * kKChunk * 2,
                  "gemm_bf16: PAGE_SIZE < kMTile*kKChunk*2");

    using bf16 = __nv_bfloat16;
    constexpr int NCW = Config::NUM_CONSUMER_WARPS;
    const int lane = kittens::laneid();

    constexpr int A_elems = kMTile * kKChunk;
    constexpr int B_elems = kNTile * kKChunk;
    constexpr int N_TILES = N / kNTile;
    constexpr int M_TILES = (M + kMTile - 1) / kMTile;

    for (int tile = blockIdx.x; tile < N_TILES * M_TILES; tile += gridDim.x) {
        const int col_tile = tile % N_TILES;
        const int row_tile = tile / N_TILES;
        const int m_base   = row_tile * kMTile;
        const int n_base   = col_tile * kNTile;

        int iter = 0;
        for (int k = 0; k < K; k += kKChunk, ++iter) {
            if (iter >= 1) {
                const int prev = (iter - 1) & 1;
                kittens::wait(ss.page_done[base_stage + kActPageOff],    prev);
                kittens::wait(ss.page_done[base_stage + kWeightPageOff], prev);
            }

            if constexpr (NCW == 4) {
                // Path A: swizzled tiles via idx() per 8-element float4 chunk.
                // Direct pointer arithmetic is WRONG after XOR swizzle;
                // idx(ptr, {r, c}) returns the correct physical address.
                auto& a_smem = *reinterpret_cast<wg_a_smem_t<kKChunk>*>(
                    ss.pages[base_stage + kActPageOff]);
                auto& b_smem = *reinterpret_cast<wg_b_smem_t<kKChunk>*>(
                    ss.pages[base_stage + kWeightPageOff]);

                for (int r = lane; r < kMTile; r += 32) {
                    const int gr = m_base + r;
                    if (gr < M) {
                        const bf16* src = x + (size_t)gr * K + k;
                        for (int c = 0; c < kKChunk; c += 8) {
                            bf16* dst = wg_a_smem_t<kKChunk>::idx(
                                &a_smem.data[0], {r, c});
                            *reinterpret_cast<float4*>(dst) =
                                *reinterpret_cast<const float4*>(src + c);
                        }
                    } else {
                        const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
                        for (int c = 0; c < kKChunk; c += 8) {
                            bf16* dst = wg_a_smem_t<kKChunk>::idx(
                                &a_smem.data[0], {r, c});
                            *reinterpret_cast<float4*>(dst) = z;
                        }
                    }
                }
                for (int r = lane; r < kNTile; r += 32) {
                    const int gr = n_base + r;
                    const bf16* src = W + (size_t)gr * K + k;
                    for (int c = 0; c < kKChunk; c += 8) {
                        bf16* dst = wg_b_smem_t<kKChunk>::idx(
                            &b_smem.data[0], {r, c});
                        *reinterpret_cast<float4*>(dst) =
                            *reinterpret_cast<const float4*>(src + c);
                    }
                }
            } else {
                // Path B: unswizzled tiles, raw linear stores.
                bf16* a_page = reinterpret_cast<bf16*>(
                    ss.pages[base_stage + kActPageOff]);
                bf16* b_page = reinterpret_cast<bf16*>(
                    ss.pages[base_stage + kWeightPageOff]);
                for (int e = lane; e < A_elems; e += 32) {
                    const int lr = e / kKChunk, lc = e % kKChunk;
                    const int gr = m_base + lr;
                    a_page[e] = (gr < M)
                        ? x[(size_t)gr * K + k + lc]
                        : __float2bfloat16_rn(0.f);
                }
                for (int e = lane; e < B_elems; e += 32) {
                    const int lr = e / kKChunk, lc = e % kKChunk;
                    const int gr = n_base + lr;
                    b_page[e] = W[(size_t)gr * K + k + lc];
                }
            }

            __threadfence_block();
            if (kittens::laneid() == 0) {
                kittens::arrive(ss.page_ready[base_stage + kActPageOff]);
                kittens::arrive(ss.page_ready[base_stage + kWeightPageOff]);
            }
        }
    }
}

// ==========================================================================
// Consumer
// ==========================================================================

template <typename Config, int K, int N, int M>
__device__ __forceinline__ void consumer(
    ferrite::bf16_ptr  out,
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int warp_in_role
) {
    static_assert(K % kKChunk == 0, "gemm_bf16: K not divisible by kKChunk");
    static_assert(N % kNTile  == 0, "gemm_bf16: N not divisible by kNTile");
    static_assert(Config::SCRATCH_BYTES >= kMTile * kNTile * 2,
                  "gemm_bf16: SCRATCH_BYTES < kMTile*kNTile*2");
    static_assert(Config::NUM_CONSUMER_WARPS >= 2, "gemm_bf16: NCW must be >= 2");

    using bf16 = __nv_bfloat16;
    constexpr int NCW = Config::NUM_CONSUMER_WARPS;

    constexpr int N_TILES = N / kNTile;
    constexpr int M_TILES = (M + kMTile - 1) / kMTile;

    for (int tile = blockIdx.x; tile < N_TILES * M_TILES; tile += gridDim.x) {
        const int col_tile = tile % N_TILES;
        const int row_tile = tile / N_TILES;
        const int m_base   = row_tile * kMTile;
        const int col_base = col_tile * kNTile;

        int iter = 0;

        if constexpr (NCW == 4) {
            // ── Path A: wgmma (NCW=4, kMTile=64) ─────────────────────────

            wg_acc_rt acc;
            kittens::group<1>::zero(acc);

            for (int k = 0; k < K; k += kKChunk, ++iter) {
                const int phase = iter & 1;
                kittens::wait(ss.page_ready[base_stage + kActPageOff],    phase);
                kittens::wait(ss.page_ready[base_stage + kWeightPageOff], phase);

                const auto& a_smem = *reinterpret_cast<const wg_a_smem_t<kKChunk>*>(
                    ss.pages[base_stage + kActPageOff]);
                const auto& b_smem = *reinterpret_cast<const wg_b_smem_t<kKChunk>*>(
                    ss.pages[base_stage + kWeightPageOff]);

                kittens::warpgroup::mma_ABt(acc, a_smem, b_smem);
                kittens::warpgroup::mma_async_wait();

                kittens::group<NCW>::sync(kConsumerBarMma);
                if (warp_in_role == 0 && kittens::laneid() == 0) {
                    kittens::arrive(ss.page_done[base_stage + kActPageOff]);
                    kittens::arrive(ss.page_done[base_stage + kWeightPageOff]);
                }
            }

            auto& scratch_tile = *reinterpret_cast<wg_scratch_t*>(ss.scratch);
            kittens::warpgroup::store(scratch_tile, acc);
            kittens::group<NCW>::sync(kConsumerBarOut);

            const bf16* sp = reinterpret_cast<const bf16*>(ss.scratch);
            const int lane = kittens::laneid();
            const int warp_row_base = warp_in_role * kMChunk;
            for (int row = 0; row < kMChunk; ++row) {
                const int token = m_base + warp_row_base + row;
                if (token >= M) break;
                const int srow = warp_row_base + row;
                for (int c = lane; c < kNTile; c += 32)
                    out[(size_t)token * N + col_base + c] = sp[srow * kNTile + c];
            }

        } else {
            // ── Path B: warp-level mma_ABt (pre-Wave-E, NCW=2) ───────────
            // Exact implementation from commit 4d6497cc0.

            using warp = kittens::group<1>;
            constexpr int N_CHUNKS = kNTile / kNChunk;

            acc_rt acc[N_CHUNKS];
            #pragma unroll
            for (int nc = 0; nc < N_CHUNKS; ++nc)
                warp::zero(acc[nc]);

            for (int k = 0; k < K; k += kKChunk, ++iter) {
                const int phase = iter & 1;
                kittens::wait(ss.page_ready[base_stage + kActPageOff],    phase);
                kittens::wait(ss.page_ready[base_stage + kWeightPageOff], phase);

                const size_t a_row_off =
                    (size_t)warp_in_role * kMChunk * kKChunk * sizeof(bf16);
                auto& a_smem = *reinterpret_cast<a_warp_st<kKChunk>*>(
                    &ss.pages[base_stage + kActPageOff][a_row_off]);
                a_rt<kKChunk> a_reg;
                warp::load(a_reg, a_smem);

                #pragma unroll
                for (int nc = 0; nc < N_CHUNKS; ++nc) {
                    const size_t b_row_off =
                        (size_t)nc * kNChunk * kKChunk * sizeof(bf16);
                    auto& b_smem_chunk = *reinterpret_cast<b_warp_st<kKChunk>*>(
                        &ss.pages[base_stage + kWeightPageOff][b_row_off]);
                    b_rt<kKChunk> b_reg;
                    warp::load(b_reg, b_smem_chunk);
                    warp::mma_ABt(acc[nc], a_reg, b_reg, acc[nc]);
                }
                kittens::group<NCW>::sync(kConsumerBarMma);
                if (warp_in_role == 0 && kittens::laneid() == 0) {
                    kittens::arrive(ss.page_done[base_stage + kActPageOff]);
                    kittens::arrive(ss.page_done[base_stage + kWeightPageOff]);
                }
            }

            out_rt out_reg;
            #pragma unroll
            for (int nc = 0; nc < N_CHUNKS; ++nc) {
                kittens::rt_bf<kMChunk, kNChunk> tmp;
                warp::copy(tmp, acc[nc]);
                out_reg.tiles[0][nc] = tmp.tiles[0][0];
            }
            const size_t scratch_row_off =
                (size_t)warp_in_role * kMChunk * kNTile * sizeof(bf16);
            auto& out_smem_warp = *reinterpret_cast<out_warp_st*>(
                &ss.scratch[scratch_row_off]);
            warp::store(out_smem_warp, out_reg);
            kittens::group<NCW>::sync(kConsumerBarOut);

            const bf16* scratch_ptr = reinterpret_cast<const bf16*>(ss.scratch);
            const int lane = kittens::laneid();
            for (int row = 0; row < kMChunk; ++row) {
                const int token = m_base + warp_in_role * kMChunk + row;
                if (token >= M) break;
                const int scratch_row_base = (warp_in_role * kMChunk + row) * kNTile;
                for (int c = lane; c < kNTile; c += 32)
                    out[(size_t)token * N + col_base + c] =
                        scratch_ptr[scratch_row_base + c];
            }
        }
    }
}

// ==========================================================================
// Launcher — no-op on Hopper.
// ==========================================================================

template <typename Config, int K, int N, int M>
__device__ __forceinline__ void launcher(
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    constexpr int N_TILES = N / kNTile;
    constexpr int M_TILES = (M + kMTile - 1) / kMTile;
    for (int tile = blockIdx.x; tile < N_TILES * M_TILES; tile += gridDim.x) {
        (void)ss; (void)base_stage;
    }
}

// ==========================================================================
// Storer — no-op (consumer writes global output directly).
// ==========================================================================

template <typename Config, int K, int N, int M>
__device__ __forceinline__ void storer(
    ferrite::bf16_ptr  out,
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    constexpr int N_TILES = N / kNTile;
    constexpr int M_TILES = (M + kMTile - 1) / kMTile;
    for (int tile = blockIdx.x; tile < N_TILES * M_TILES; tile += gridDim.x) {
        (void)out; (void)ss; (void)base_stage;
    }
}

}  // namespace gemm_bf16
}  // namespace ops
}  // namespace ferrite
