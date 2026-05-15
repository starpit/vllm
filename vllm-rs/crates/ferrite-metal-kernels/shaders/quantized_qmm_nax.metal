// SPDX-License-Identifier: Apache-2.0
//
// NAX (Apple9 / M4+) port of MLX's `qmm_t_nax_tgp_impl` /
// `affine_qmm_t_nax` from `mlx/backend/metal/kernels/quantized_nax.h`.
//
// Tile shape: BM=BN=BK=64, WM=WN=2, SK=32, TGP=128 (4 simdgroups).
//   BK=64 matches MLX's production instantiation in
//   `mlx/backend/metal/kernels/quantized_nax.metal`
//   (`instantiate_quantized_aligned_batched(affine_qmm_t_nax, ..., 64, 64, 64, 2, 2, ...)`).
//
// Why BK=64 (not 32):
//   - QuantizedBlockLoader requires `BCOLS_PACKED/n_reads == n_groups`
//     in its gs=32 specialization. With BK=32, gs=32 → BCOLS_PACKED=16,
//     n_reads=8, n_groups=1 → 16/8 != 1; MLX never instantiates this
//     combination. BK=64 with gs=32 → BCOLS_PACKED=32, n_reads=16,
//     n_groups=2 → 32/16 == 2 ✓ (matches MLX's `quantized_nax.metal`).
//   - For NAX we currently skip gs=32 entirely (the dispatcher gates
//     `group_size != 32` for the Nax pick); the general QuantizedBlockLoader
//     path is inlined here for gs=64 and gs=128. gs=32 would need the
//     specialized loader with per-thread `group_id` — straightforward
//     follow-up if production models start shipping gs=32 quants.
//
// MLX reference: `mlx/backend/metal/quantized.cpp:695` dispatches NAX
// when `K % 64 == 0`. Same gate is used Rust-side in
// `pick_qmm_t_kernel`.
//
// Ferrite adaptations relative to MLX's kernel:
//   - K, N, M come from file-scope function constants (indices 0/1/2)
//     rather than `const constant int& [[buffer(5/6/7)]]` — required
//     so the pipeline is recordable into an MTLIndirectComputeCommand.
//   - W-loader is inlined (QuantizedBlockLoader struct not imported);
//     general-loader path only (gs ≥ 64).
//   - batched=0 only; no `adjust_matrix_offsets` (B=1).
//
// Buffer layout matches the existing quantized_qmm.metal:
//   w[0], scales[1], biases[2], x[3], y[4]
//
// Dispatch: grid = (ceil(N/64), ceil(M/64), 1), tpg = (128, 1, 1).
// Kernel selected only on M4+ (is_nax_capable check in pick_qmm_t_kernel).

#include "metal_nax.h"   // vendors NAXTile + tile_matmad_nax + MPP internals
#include <metal_stdlib>

using namespace metal;
using namespace mlx::steel;

#define MLX_MTL_CONST static constant constexpr const

#ifndef MLX_MTL_PRAGMA_NO_UNROLL
#define MLX_MTL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")
#endif

MLX_MTL_CONST int NAX_SIMD_SIZE = 32;

// ─────────────────────────────────────────────────────────────────
// Function constants — same slot indices as quantized_qmm.metal so
// the lowering pass shares the same ConstantValue::int slot map.
// ─────────────────────────────────────────────────────────────────

constant int QMM_K [[function_constant(0)]];
constant int QMM_N [[function_constant(1)]];
constant int QMM_M [[function_constant(2)]];

// ─────────────────────────────────────────────────────────────────
// qmm_t_nax_impl — port of qmm_t_nax_tgp_impl (quantized_nax.h)
//
// Template params:
//   T_act      activation/output type (half or bfloat)
//   T_scale    scale/bias type (half)
//   group_size affine-quant group size (64 or 128 — gs=32 not
//              supported here; the dispatcher falls back to Standard
//              qmm_t for gs=32)
//   bits       = 4
//   aligned_N  N % 64 == 0 → skip N-tail handling
//
// Tile shape (matches MLX BM=BN=BK=64, WM=WN=2):
//   SM = SN = BM/WM = BN/WN = 32   (per-simdgroup M/N extent)
//   SK = 32                          (inner K step; BK/SK = 2 iters)
//   TM = TN = TK = 2                 (NAXTile row/col counts)
//   TGP = 128 threads, 4 simdgroups
//   BK_padded = BK + 16/sizeof(T)  = 72 for half/bfloat (bank-conflict pad)
//
// W-loader (QuantizedBlockLoader equivalent, reduction_dim=1, general):
//   BROWS = BN = 64, BCOLS = BK = 64
//   BCOLS_PACKED = BK/pack_factor = 32
//   n_reads = (BCOLS_PACKED * BN) / TGP = (32*64)/128 = 16
//   bi_w = (n_reads * thread_idx) / BCOLS_PACKED = thread_idx/2 ∈ [0,63]
//   bj_w = (n_reads * thread_idx) % BCOLS_PACKED ∈ {0, 16}
//   scales start at s_block + bi_w * K_g (no bj offset — gs ≥ BK)
//   scale advance: group_steps = group_size/BK = group_size/64
//     gs=64 → advance every BK step (group_steps == 1)
//     gs=128 → advance every 2 BK steps (group_steps == 2)
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits, bool aligned_N>
METAL_FUNC void qmm_t_nax_impl(
    const device uint32_t*  w,
    const device T_scale*   scales,
    const device T_scale*   biases,
    const device T_act*     x,
    device T_act*           y,
    threadgroup T_act*      Ws,
    const int K,
    const int N,
    const int M,
    uint  simd_gid,
    uint  simd_lid,
    uint3 tgid)
{
    static_assert(bits == 4, "qmm_t_nax_impl: only bits=4 instantiated");
    static_assert(group_size == 64 || group_size == 128,
                  "qmm_t_nax_impl: group_size must be 64 or 128 (gs=32 needs "
                  "the specialized QuantizedBlockLoader; dispatcher falls "
                  "back to Standard qmm_t for gs=32)");

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;   // K-reduction tile; matches MLX production
    constexpr int WM = 2;
    constexpr int WN = 2;
    constexpr int TGP = WM * WN * NAX_SIMD_SIZE;  // 128

    // Per-simdgroup tile extents
    constexpr int SM = BM / WM;   // 32
    constexpr int SN = BN / WN;   // 32
    constexpr int SK = 32;
    constexpr int TM = SM / 16;   // 2
    constexpr int TN = SN / 16;   // 2
    constexpr int TK = SK / 16;   // 2

    // BK_padded: pad for bank-conflict avoidance (BK + 16/sizeof(T) = 64+8=72)
    constexpr int BK_padded = BK + 16 / int(sizeof(T_act));

    // For bits=4: pack_factor=2 (2 int4 per byte), bytes_per_pack=1
    constexpr int pack_factor    = 2;
    constexpr int bytes_per_pack = 1;

    // W-loader tile layout
    //   BCOLS_PACKED = BK / pack_factor = 64/2 = 32
    //   N_READS      = (BCOLS_PACKED × BN) / TGP = (32×64)/128 = 16
    constexpr int BCOLS_PACKED = BK / pack_factor;   // 32
    constexpr int N_READS      = (BCOLS_PACKED * BN) / TGP;  // 16

    // group_steps: how many BK steps between scale advances
    //   gs=64  → 1 (advance every step)
    //   gs=128 → 2 (advance every 2 steps)
    constexpr int group_steps = group_size / BK;

    // ── Per-simdgroup M/N offsets within the 64×64 tile ────────────
    //   simd_gid 0 → tm=0,  tn=0
    //   simd_gid 1 → tm=0,  tn=32
    //   simd_gid 2 → tm=32, tn=0
    //   simd_gid 3 → tm=32, tn=32
    const int tm = SM * int(simd_gid / WN);
    const int tn = SN * int(simd_gid % WN);

    // ── Tile origin (grid level) ────────────────────────────────────
    const int y_row = int(tgid.y) * BM;
    const int y_col = int(tgid.x) * BN;
    if (y_row >= M || y_col >= N) return;

    // ── Valid sub-tile extents for M/N tail handling ────────────────
    const short sgp_sm = min(short(SM), short(M - (y_row + tm)));
    const bool is_unaligned_sm = (sgp_sm != short(SM));

    const short tgp_bn = aligned_N ? short(BN) : short(min(BN, N - y_col));
    const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != short(BN));
    const short sgp_sn = aligned_N ? short(SN)
                                   : short(min(int(SN), N - (y_col + tn)));

    // ── Flat thread index for the W-loader layout ───────────────────
    const uint thread_idx = simd_gid * NAX_SIMD_SIZE + simd_lid;

    // W-loader per-thread coords:
    //   bi_w: row in [0, BN) of the BN×BK tile
    //   bj_w: packed-column in [0, BCOLS_PACKED=32) → values {0, 16}
    const uint bi_w = (uint(N_READS) * thread_idx) / uint(BCOLS_PACKED);
    const uint bj_w = (uint(N_READS) * thread_idx) % uint(BCOLS_PACKED);

    // ── Block-level base pointers ───────────────────────────────────
    const int K_w = K * bytes_per_pack / pack_factor;  // K/2 packed bytes per W row
    const int K_g = K / group_size;                    // scale groups per W row

    const device uint8_t* wl_base = (const device uint8_t*)w;
    const device T_act* x_block   = x + int64_t(y_row) * K;
    const device uint8_t* w_block = wl_base + y_col * K_w;
    const device T_scale* s_block = scales  + y_col * K_g;
    const device T_scale* b_block = biases  + y_col * K_g;
    device T_act* y_block         = y + int64_t(y_row) * N + y_col;

    // ── Per-thread W destination in Ws, source in device W ─────────
    threadgroup T_act* Ws_dst = Ws + bi_w * BK_padded + bj_w * pack_factor;
    const device uint8_t* W_src =
        w_block + bi_w * K_w + bj_w * bytes_per_pack;

    // Scale/bias: all threads with the same bi_w start at scale group 0.
    // group_size >= BK=64 always here (gs=32 dispatcher fallback), so
    // one BK step covers ≤ group_size elements and bj_w stays within
    // the same scale group.
    const device T_scale* Sc_row = s_block + bi_w * K_g;
    const device T_scale* Bs_row = b_block + bi_w * K_g;
    int group_step_cnt = 0;

    // Per-simdgroup X pointer: x_block points to top of the M-tile;
    // shift by tm rows for this simdgroup.
    const device T_act* x_ptr = x_block + int64_t(tm) * K;

    // ── Float accumulator (NAX register tiles) ──────────────────────
    NAXTile<float, TM, TN> Dtile;
    Dtile.clear();

    // ── Outer K loop + M/N alignment dispatch ───────────────────────
    dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
        dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {

            for (int k = 0; k < K; k += BK) {
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // ── W loader: collectively dequantize Ws ─────────────
                // 128 threads × 16 packed bytes → 128×32 = 4096 dequant
                // elements = BN×BK = 64×64. One scale per thread (since
                // group_size ≥ BK=64, all 16 reads share the same group).
                if constexpr (kAlignedN.value) {
                    T_act scale = T_act(*Sc_row);
                    T_act bias  = T_act(*Bs_row);
                    T_act s0 = scale;
                    T_act s1 = scale / T_act(16.0f);  // compensates for & 0xf0
                    for (int i = 0; i < N_READS; ++i) {
                        uint8_t b = W_src[i * bytes_per_pack];
                        Ws_dst[i * pack_factor + 0] = s0 * T_act(b & 0x0f) + bias;
                        Ws_dst[i * pack_factor + 1] = s1 * T_act(b & 0xf0) + bias;
                    }
                } else {
                    // N-tail: zero rows past tgp_bn (bi_w is the N-row index)
                    if (bi_w < uint(tgp_bn)) {
                        T_act scale = T_act(*Sc_row);
                        T_act bias  = T_act(*Bs_row);
                        T_act s0 = scale;
                        T_act s1 = scale / T_act(16.0f);
                        for (int i = 0; i < N_READS; ++i) {
                            uint8_t b = W_src[i * bytes_per_pack];
                            Ws_dst[i * pack_factor + 0] = s0 * T_act(b & 0x0f) + bias;
                            Ws_dst[i * pack_factor + 1] = s1 * T_act(b & 0xf0) + bias;
                        }
                    } else {
                        for (int i = 0; i < N_READS * pack_factor; ++i) {
                            Ws_dst[i] = T_act(0);
                        }
                    }
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // ── Inner SK=32 loop (BK=64, SK=32 → 2 iterations) ───
                // Load 32×32 A tile from device and 32×32 B tile from
                // Ws, then issue NAX MMA → 32×32 float C accumulation.
                MLX_MTL_PRAGMA_NO_UNROLL
                for (int kk1 = 0; kk1 < BK; kk1 += SK) {
                    NAXTile<T_act, TM, TK> Atile;
                    NAXTile<T_act, TN, TK> Btile;

                    volatile int compiler_barrier;

                    if constexpr (kAlignedM.value) {
                        Atile.load(x_ptr + kk1, K);
                    } else {
                        Atile.load_safe(x_ptr + kk1, K, short2(SK, sgp_sm));
                    }

                    // B tile: Ws[tn*BK_padded + kk1], stride (BK_padded, 1)
                    Btile.template load<T_act, BK_padded, 1>(
                        Ws + tn * BK_padded + kk1);

                    // MMA: D += A * B^T (transpose_a=false, transpose_b=true)
                    tile_matmad_nax(
                        Dtile,
                        Atile, metal::bool_constant<false>{},
                        Btile, metal::bool_constant<true>{});

                    (void)compiler_barrier;
                }

                // ── Advance pointers for next BK iter ────────────────
                x_ptr += BK;
                W_src += BCOLS_PACKED * bytes_per_pack;  // 32 bytes

                // Scale advance: every group_steps BK iters.
                if constexpr (group_steps > 1) {
                    group_step_cnt++;
                    if (group_step_cnt == group_steps) {
                        group_step_cnt = 0;
                        Sc_row++;
                        Bs_row++;
                    }
                } else {
                    // group_steps == 1 (gs=64=BK): advance every step
                    Sc_row++;
                    Bs_row++;
                }
            }

            // ── Epilogue: store float accumulator to device ──────────
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if constexpr (kAlignedM.value && kAlignedN.value) {
                Dtile.store(y_block + tm * N + tn, N);
            } else if (kAlignedM.value && sgp_sn == short(SN)) {
                Dtile.store(y_block + tm * N + tn, N);
            } else {
                Dtile.store_safe(y_block + tm * N + tn, N, short2(sgp_sn, sgp_sm));
            }

        });
    });
}

// ─────────────────────────────────────────────────────────────────
// affine_qmm_t_nax kernel wrapper — batched=0 only
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits, bool aligned_N>
[[kernel]] void affine_qmm_t_nax_kernel(
    const device uint32_t*  w        [[buffer(0)]],
    const device T_scale*   scales   [[buffer(1)]],
    const device T_scale*   biases   [[buffer(2)]],
    const device T_act*     x        [[buffer(3)]],
    device T_act*           y        [[buffer(4)]],
    // K, N, M ride as file-scope function constants QMM_K/N/M so this
    // kernel is recordable into an MTLIndirectComputeCommand.
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint3 tgid     [[threadgroup_position_in_grid]])
{
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int BK_padded = BK + 16 / int(sizeof(T_act));  // 72
    threadgroup T_act Ws[BN * BK_padded];  // 64 × 72 = 4608 elements
    qmm_t_nax_impl<T_act, T_scale, group_size, bits, aligned_N>(
        w, scales, biases, x, y, Ws,
        QMM_K, QMM_N, QMM_M,
        simd_gid, simd_lid, tgid);
}

// ─────────────────────────────────────────────────────────────────
// Instantiations — one symbol per (dtype, group_size, aligned_N).
// Naming mirrors the non-NAX pattern with "nax" inserted after
// "qmm_t_":
//   affine_qmm_t_nax_<dtype>_s_<scale_dtype>_gs_<gs>_b_4_alN_<bool>_batch_0
//
// gs=32 deliberately not instantiated — the dispatcher routes gs=32
// to Standard qmm_t (BK=64 violates QuantizedBlockLoader's
// `BCOLS<=group_size` for gs=32; the gs=32 specialized loader has
// different scale-indexing semantics and would need a separate path).
// ─────────────────────────────────────────────────────────────────

#define INST_QMM_T_NAX(act_tag, act_type, scale_tag, scale_type, gs, aln_tag, aln_val) \
    template [[host_name(                                                                \
        "affine_qmm_t_nax_" #act_tag "_s_" #scale_tag "_gs_" #gs                       \
        "_b_4_alN_" #aln_tag "_batch_0")]] [[kernel]] void                              \
    affine_qmm_t_nax_kernel<act_type, scale_type, gs, 4, aln_val>(                     \
        const device uint32_t*   w        [[buffer(0)]],                                \
        const device scale_type* scales   [[buffer(1)]],                                \
        const device scale_type* biases   [[buffer(2)]],                                \
        const device act_type*   x        [[buffer(3)]],                                \
        device act_type*         y        [[buffer(4)]],                                \
        uint  simd_gid [[simdgroup_index_in_threadgroup]],                              \
        uint  simd_lid [[thread_index_in_simdgroup]],                                   \
        uint3 tgid     [[threadgroup_position_in_grid]]);

#define INST_QMM_T_NAX_ALL(act_tag, act_type, scale_tag, scale_type, gs) \
    INST_QMM_T_NAX(act_tag, act_type, scale_tag, scale_type, gs, true,  true)  \
    INST_QMM_T_NAX(act_tag, act_type, scale_tag, scale_type, gs, false, false)

INST_QMM_T_NAX_ALL(f16,  half,   f16, half,  64)
INST_QMM_T_NAX_ALL(f16,  half,   f16, half, 128)
INST_QMM_T_NAX_ALL(bf16, bfloat, f16, half,  64)
INST_QMM_T_NAX_ALL(bf16, bfloat, f16, half, 128)
