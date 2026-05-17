// SPDX-License-Identifier: Apache-2.0
//
// Faithful port of MLX `affine_qmm_t` (`mlx/backend/metal/kernels/
// quantized.h:1707`) + `affine_qmm_t_splitk` (`:1780`) + `affine_qmm_n`
// (`:1847`) prefill-matmul kernels. The bodies of `qmm_t_impl`
// (`:1094`) and `qmm_n_impl` (`:1221`) are reproduced via an inlined
// steel-style BlockMMA / BlockLoader / QuantizedBlockLoader trio.
// We keep the kernel surface identical to MLX (same tile sizes
// BM=BN=BK=32, same warp shape WM=WN=2, same per-thread load layout,
// same `aligned_N` template flag for qmm_t, same K loop) so that the
// four `if (m_full)` × `if (aligned_N && n_full)` branches at
// `quantized.h:1158-1202` map line-for-line into the four load paths
// below.
//
// qmm_n (transpose=false) differs from qmm_t in three places:
//   1. W storage is `[K, N]` u32-packed (last axis = N), so the K loop
//      reads BK rows × BCOLS_PACKED packed bytes per BK iter.
//   2. Scales / biases layout is `[K, N/group_size]` — group runs
//      along N, not K — so the per-BK step advances them by
//      `BK * N / group_size` (`group_stride` in MLX's
//      `QuantizedBlockLoader::next()` reduction_dim=0 branch).
//   3. BlockMMA's B-frag is loaded without transpose because Ws is
//      already laid out as `[BK, BN_padded]` (the natural shape MMA
//      expects when `transpose_b == false` per
//      `quantized.h:1124`-`:1252`).
// MLX qmm_n has no `aligned_N` template flag — the dispatcher assumes
// `N % 32 == 0` (the matmul-branch grid at `quantized.cpp:721` ceils
// N, but the store path is unconditional `store_result` for
// `num_els == BM` and silently writes past N when N is unaligned).
// All linear shapes in our model coverage matrix have N % 32 == 0
// (typically a multiple of 64), so we match MLX 1:1 and don't expose
// an aligned_N flag for qmm_n.
//
// Why we inline rather than vendor `mlx/backend/metal/kernels/steel/
// gemm/{mma,loader}.h`: the prior 2026-05-09 session
// (`project_metal_gemm_port_dead_end`) vendored the full steel tree
// and hit a deterministic zero-output wall on scattered threadgroups
// at grid X >= 1024 (N >= ~32K). Llama-1B/3B q4 prefill shapes have
// N ≤ 8192 → grid X ≤ 256, well inside the regime where the existing
// `fused_gate_up_silu_mul_gemm_steel_*_specialized` kernels (same
// BM=BN=32, WM=WN=2 pattern) work; the lm_head N=128256 shape only
// fires at M=1 and routes through `qmv*` (P3), never through `qmm_t`.
// The dispatcher in `MetalAffineQmmT` enforces this routing per the
// `quantized.cpp:1411` `M >= vector_limit` rule.
//
// Per `INT4_PARITY_PLAN.md` §P4, instantiations cover:
//   bits = 4
//   group_size in {32, 64, 128}
//   dtype in {f16, bf16}
//   aligned_N in {true, false}
//   batched = 0 only (qmm_t batched=1 needs `adjust_matrix_offsets`
//     from `quantized.h:1351`, deferred to P13 with the MoE gather
//     variants; dispatcher rejects B>1 here.)
//
// Symbol naming follows the MLX dispatcher's `concatenate` at
// `quantized.cpp:728`:
//   affine_qmm_t_<dtype>_gs_<gs>_b_<bits>_alN_<true|false>_batch_0
//   affine_qmm_t_splitk_<dtype>_gs_<gs>_b_<bits>_alN_<true|false>
//
// QuantizedBlockLoader port:
//   `quantized.h:572-689`. We use the same per-thread layout
//   (n_reads=4 packed bytes per K-iter for bits=4, gs=32+), inline
//   the `dequantize<T, pack_factor, bits>` body (`quantized.h:483`),
//   and step scale/bias every `group_size / BK` BK iterations to
//   match `QuantizedBlockLoader::next()` (`:671`).

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

#ifndef MLX_MTL_PRAGMA_UNROLL
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#endif

MLX_MTL_CONST int SIMD_SIZE = 32;

// ─────────────────────────────────────────────────────────────────
// Function constants — baked at pipeline build time by
// `MetalAffineQmmT::execute` (and the lower_one path once
// `Instruction::AffineQmm` lands). MLX passes K / N / M (and the
// splitk wrapper passes k_partition_size + split_k_partition_stride)
// as setBytes runtime args; ferrite specializes per-shape so each
// (K, N, M[, k_partition_size]) tuple gets its own pipeline. This is
// the same trade `fused_gate_up_silu_mul` already takes — and is
// required for ICB recording, which exposes setKernelBuffer but not
// setKernelBytes.
//
// `split_k_partition_stride` (M * N) is computed inline from QMM_M /
// QMM_N rather than baked separately, so the constant slots stay
// shared across the standard and splitk variants.
//
// Indices match `ConstantValue::uint(N, value)` in the dispatcher;
// keep them stable.
// ─────────────────────────────────────────────────────────────────

constant int QMM_K                [[function_constant(0)]];
constant int QMM_N                [[function_constant(1)]];
constant int QMM_M                [[function_constant(2)]];
constant int QMM_K_PARTITION_SIZE [[function_constant(3)]];

// ─────────────────────────────────────────────────────────────────
// Pack helpers — quantized.h:17-26 (same constants as
// quantized_qmv.metal; duplicated here so each .metal compiles
// stand-alone — `build.rs` produces one .metallib per file).
// ─────────────────────────────────────────────────────────────────

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

// ─────────────────────────────────────────────────────────────────
// qmm_t_impl — quantized.h:1094-1212. Implemented for bits=4 by
// inlining the 4-bit branch of `dequantize` (`:521-527`) directly
// into the W-loader. Other bits land alongside their first model.
//
// Template params match MLX 1:1 (sans the BlockMMA/BlockLoader hooks
// which are spelled out inline):
//   T          activation/output element type (half or bfloat)
//   group_size affine-quant group size (32, 64, or 128 here)
//   bits       = 4 (P4)
//   aligned_N  N % BN == 0 → skip N-tail handling
//
// Hard-coded tile shape (faithful to MLX `qmm_t_impl` defaults at
// `quantized.h:1091-1093`):
//   BM = BN = BK = 32, WM = WN = 2, TM = TN = 2 (4 8×8 frags/SG)
//   BK_padded = BK + 16 / sizeof(T)   (= 40 for half, 40 for bfloat)
//   TGP = WM * WN * SIMD_SIZE = 128 threads / threadgroup
// ─────────────────────────────────────────────────────────────────

// T_compute is the dtype the threadgroup tiles + simdgroup MMA run on.
// Defaults to T_act (current behavior). On Apple7 (M1 family) bf16
// simdgroup_multiply_accumulate is a slow-path emulation (~1.7× slower
// than the f16 path); a `T_act = bfloat, T_compute = half` instantiation
// reads bf16 from device memory, casts to half when populating Xs/Ws,
// runs MMA in half, casts back to bf16 on output store. Output dtype on
// disk stays T_act so the residual stream's bf16 dynamic range is
// preserved (full-f16 streams break Llama-3.x exponent range — see
// `instr.rs:199-202`).
template <typename T_act, typename T_compute, typename T_scale,
          int group_size, int bits, bool aligned_N>
METAL_FUNC void qmm_t_impl_inline(
    const device uint32_t*  w,
    const device T_scale*   scales,
    const device T_scale*   biases,
    const device T_act*     x,
    device T_act*           y,
    threadgroup T_compute*  Xs,
    threadgroup T_compute*  Ws,
    threadgroup float*      out_scratch,
    const int              K,
    const int              N,
    const int              M,
    const int              K_eff,
    uint  simd_group_id,
    uint  simd_lane_id,
    uint3 tgid)
{
  static_assert(bits == 4, "qmm_t_impl_inline only instantiated for bits=4");
  static_assert(group_size == 32 || group_size == 64 || group_size == 128,
                "qmm_t_impl_inline expects group_size in {32, 64, 128}");

  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int TM = BM / (8 * WM);            // 2
  constexpr int TN = BN / (8 * WN);            // 2
  constexpr int KFR = BK / 8;                  // 4 K-frags per BK iter
  // Threadgroup tiles live in T_compute, so BK_padded keys off
  // sizeof(T_compute) (= 40 for both fp16 and bf16, since both are 2 B).
  constexpr int BK_padded = BK + 16 / int(sizeof(T_compute));
  constexpr int TGP = WM * WN * SIMD_SIZE;     // 128

  constexpr int pack_factor    = get_pack_factor<bits, 8>();    // 2
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();    // 1
  constexpr int BCOLS_PACKED   = BK / pack_factor;              // 16
  // Per-thread W reads: n_reads packed bytes → n_reads*pack_factor halves.
  constexpr int N_READS = (BCOLS_PACKED * BN) / TGP;            // 4
  constexpr int group_steps = group_size / BK;                  // 1, 2, or 4

  // Per-thread X read: 8 halves per thread (covers BM*BK = 1024 / 128 = 8
  // halves/thread). Each thread loads 2 vec4's of T (16 bytes per vec
  // = 8 halves) — the same vec4 pattern as the steel BlockLoader (one
  // unrolled iter per thread for BROWS==TROWS==32).
  constexpr int X_N_READS = (BM * BK) / TGP;                    // 8
  constexpr int X_TCOLS   = BK / X_N_READS;                     // 4
  static_assert(X_N_READS == 8, "expected 8 X halves per thread");

  const uint thread_idx = simd_group_id * SIMD_SIZE + simd_lane_id;

  // ── Output-tile origin and per-tile valid extents ──────────────
  const uint c_row = tgid.y * BM;  // along M
  const uint c_col = tgid.x * BN;  // along N
  if (c_row >= uint(M) || c_col >= uint(N)) return;
  const uint m_tile = (c_row + BM <= uint(M)) ? uint(BM) : uint(M) - c_row;
  const uint n_tile = aligned_N
      ? uint(BN)
      : ((c_col + BN <= uint(N)) ? uint(BN) : uint(N) - c_col);
  const bool m_full = (m_tile == BM);
  const bool n_full = aligned_N ? true : (n_tile == BN);

  // ── X loader thread coords (per BlockLoader; transpose dim=1) ──
  const uint bi_x = thread_idx / uint(X_TCOLS);          // [0, BM)
  const uint bj_x = uint(X_N_READS) * (thread_idx % uint(X_TCOLS));  // 0,8,16,24

  // ── W loader thread coords (per QuantizedBlockLoader) ──────────
  const uint bi_w = (uint(N_READS) * thread_idx) / uint(BCOLS_PACKED);     // [0, BN)
  const uint bj_w = (uint(N_READS) * thread_idx) % uint(BCOLS_PACKED);     // 0,4,8,12

  // ── Block base pointers (after y_row / y_col shift per MLX :1145) ──
  const int K_w = K * bytes_per_pack / pack_factor;       // packed bytes per row of W
  const int K_g = K / group_size;                         // groups per row of W
  const device uint8_t* wl_base = (const device uint8_t*)w;
  const device T_act* x_block    = x + int64_t(c_row) * K;
  const device uint8_t* w_block  = wl_base + c_col * K_w;
  const device T_scale* s_block  = scales + c_col * K_g;
  const device T_scale* b_block  = biases + c_col * K_g;
  device T_act* y_block          = y + int64_t(c_row) * N + c_col;

  // ── Per-thread source/dest pointers (BlockLoader constructor :47-58
  //    and QuantizedBlockLoader constructor :605-626) ──────────────
  // Xs_dst/Ws_dst are T_compute (the threadgroup-tile dtype); X_src is
  // T_act (device-memory dtype) and gets cast on load.
  threadgroup T_compute* Xs_dst = Xs + bi_x * BK_padded + bj_x;
  const device T_act* X_src = x_block + bi_x * K + bj_x;

  threadgroup T_compute* Ws_dst = Ws + bi_w * BK_padded + bj_w * pack_factor;
  const device uint8_t* W_src = w_block + bi_w * K_w + bj_w * bytes_per_pack;
  const device T_scale* Sc_row = s_block + bi_w * K_g;
  const device T_scale* Bs_row = b_block + bi_w * K_g;
  int group_step_cnt = 0;

  // Accumulators stored as `vec<float,2>` per frag — the per-thread
  // half of an 8×8 simdgroup_matrix. Matches MLX MMATile<float,TM,TN>
  // ::val_frags storage (mma.h:233). simdgroup_matrix is materialized
  // only at MMA-call time, then result is read back. Apple's compiler
  // is more aggressive with vec<float,2> in registers than with a
  // live simdgroup_matrix across iterations.
  vec<float, 2> acc[TM][TN];
  MLX_MTL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i) {
    MLX_MTL_PRAGMA_UNROLL
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = vec<float, 2>(0.0f);
    }
  }

  // Per-simdgroup tile origin within the threadgroup tile (sgM*16,
  // sgN*16) — same as BlockMMA constructor `tm = kFragSize * (sg / WN);
  // tn = kFragSize * (sg % WN)`.
  const int sgM = int(simd_group_id) / WN;
  const int sgN = int(simd_group_id) % WN;

  // ── Per-lane coordinates in the 8×8 simdgroup_matrix fragment ───
  // (BaseMMAFrag<T,8,8>::get_coord, MLX steel/gemm/mma.h:49-55).
  // Each lane holds 2 contiguous elements: (fm, fn) and (fm, fn+1).
  // We pre-compute the per-lane offsets ONCE so the inner K loop is
  // pure scalar loads — replaces `simdgroup_load(...)` intrinsics
  // which carry per-fragment coordination overhead. MLX's BlockMMA
  // does the same (steel/gemm/mma.h:487-505).
  const ushort qid_lane = simd_lane_id / 4;
  const ushort fm = (qid_lane & 4) + ((simd_lane_id / 2) % 4);
  const ushort fn = (qid_lane & 2) * 2 + (simd_lane_id % 2) * 2;
  // A is stored in Xs as [BM × BK_padded]; A_str_m=BK_padded, A_str_k=1.
  // Frag (i, 0) is at row (sgM*16 + i*8), col (0). Per-lane offset
  // within the frag: (fm * BK_padded + fn).
  // Bs is stored as [BN × BK_padded] (post-dequant) — but with
  // transpose=true. In transposed view: B_str_k=1, B_str_n=BK_padded.
  // Frag (0, j) is at K=0, N=sgN*16 + j*8. Per-lane offset: (fm *
  // 1 + fn * BK_padded) → fm + fn*BK_padded.
  // (Subtle: B's per-lane offset uses fm as the row-of-frag in the
  // transposed view, which maps to K-axis of Ws.)

  // ── K loop: K_eff allows the splitk wrapper to shorten the loop ──
  //
  // Each iter loads a 32×32 X tile + dequant'd 32×32 W tile, runs four
  // 8-wide K-frag MMAs, and advances pointers.
  for (int k = 0; k < K_eff; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ─── X loader (BlockLoader<T, BM, BK, BK_padded, 1, TGP>) ─────
    //   load_unsafe: contiguous vec4 of T × 2 per thread → 8 elements.
    //   load_safe (M-tail): zero rows past m_tile.
    // Loads 8 T_act values from device, casts to T_compute on store
    // into Xs. When T_compute == T_act the compiler emits the same
    // vec4 burst copy. When they differ (T_act=bfloat, T_compute=half)
    // it inserts per-element converting moves; still memory-bound.
    if (m_full || bi_x < m_tile) {
      const device vec<T_act, 4>* X_src_v4 =
          (const device vec<T_act, 4>*)X_src;
      threadgroup vec<T_compute, 4>* Xs_dst_v4 =
          (threadgroup vec<T_compute, 4>*)Xs_dst;
      vec<T_act, 4> a0 = X_src_v4[0];
      vec<T_act, 4> a1 = X_src_v4[1];
      Xs_dst_v4[0] = vec<T_compute, 4>(
          T_compute(a0[0]), T_compute(a0[1]), T_compute(a0[2]), T_compute(a0[3]));
      Xs_dst_v4[1] = vec<T_compute, 4>(
          T_compute(a1[0]), T_compute(a1[1]), T_compute(a1[2]), T_compute(a1[3]));
    } else {
      ((threadgroup vec<T_compute, 4>*)Xs_dst)[0] = vec<T_compute, 4>(0);
      ((threadgroup vec<T_compute, 4>*)Xs_dst)[1] = vec<T_compute, 4>(0);
    }

    // ─── W loader (QuantizedBlockLoader<T, BN, BK, ..., 1, TGP, gs, 4>) ──
    //   load_unsafe: dequantize N_READS=4 packed bytes (= 8 halves).
    //   load_safe (N-tail): zero rows past n_tile.
    //
    // Inlined 4-bit `dequantize` body (`quantized.h:521-527`):
    //   s0 = scale; s1 = scale / 16.
    //   w_local[2i]   = s0 * (b & 0x0f) + bias;
    //   w_local[2i+1] = s1 * (b & 0xf0) + bias;
    if (n_full || bi_w < n_tile) {
      // Dequant math runs in T_compute (the threadgroup-tile dtype).
      // T_scale → T_compute cast keeps the in-register dtype consistent
      // with what the MMA inner loop will read out of Ws.
      T_compute scale = static_cast<T_compute>(*Sc_row);
      T_compute bias  = static_cast<T_compute>(*Bs_row);
      T_compute s0 = scale;
      T_compute s1 = scale / static_cast<T_compute>(16.0f);
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < N_READS; ++i) {
        uint8_t b = W_src[i * bytes_per_pack];
        Ws_dst[i * pack_factor + 0] =
            s0 * static_cast<T_compute>(b & 0x0f) + bias;
        Ws_dst[i * pack_factor + 1] =
            s1 * static_cast<T_compute>(b & 0xf0) + bias;
      }
    } else {
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < N_READS * pack_factor; ++i) {
        Ws_dst[i] = T_compute(0);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ─── BlockMMA::mma — KFR = BK / 8 K-frag MMAs per BK iter ───
    //   Inner loop matches MLX's `BlockMMA::mma`
    //   (steel/gemm/mma.h:507-532): direct per-thread scalar loads
    //   into `simdgroup_matrix`'s `thread_elements()` register pair,
    //   bypassing the `simdgroup_load(...)` intrinsic which carries
    //   per-fragment coordination cost. Saves ~18% per-TG in qmm_t
    //   (matches MLX's ~2.22 µs/TG on M4 vs our prior ~2.7 µs/TG).
    //   `simdgroup_barrier(mem_none)` fences mark the load→MMA
    //   boundaries for the Apple compiler so it can software-
    //   pipeline loads and MMAs across iterations.
    // Match MLX `BlockMMA` frag layout (mma.h:492-505 + 514-530):
    //   tm = kFragSize * (sgid / WN)  (= 8*sgM in our naming)
    //   tn = kFragSize * (sgid % WN)  (= 8*sgN)
    //   simdgroup frag at warp-tile (i, j) sits at
    //     M-pos: tm + (i * kFragSize * WM)   [INTERLEAVED across sgM]
    //     N-pos: tn + (j * kFragSize * WN)   [INTERLEAVED across sgN]
    // Per-thread (fm, fn) per `BaseMMAFrag<...>::get_coord` then
    // shifts the lane into the 8×8 frag. The interleaved layout is
    // what their `WM`/`WN` (warp-tile-count) factors are for, and
    // matches MLX's TM_stride/TN_stride math.
    //
    // Pointer setup: ONE per-thread base, advance by kFragSize per
    // kk-step (tile_stride_a = kFragSize * A_str_k = 8, etc.).
    constexpr int TM_stride = 8 * WM;  // = 16 (M-row stride between frags)
    constexpr int TN_stride = 8 * WN;  // = 16 (N-col stride between frags)
    const int sm = int(fm);
    const int sn = int(fn);
    const int tm = 8 * sgM;
    const int tn = 8 * sgN;
    // Per-thread base offsets — MLX `BlockMMA` constructor (mma.h:500-501):
    //   As_offset = (tm + sm) * A_str_m + sn * A_str_k
    //             = (tm + sm) * BK_padded + sn  (A_str_m=BK_padded, A_str_k=1)
    //   Bs_offset = sm * B_str_k + (tn + sn) * B_str_n
    //             = sm + (tn + sn) * BK_padded  (B_str_k=1, B_str_n=BK_padded
    //                                            since transpose_b=true)
    threadgroup const T_compute* As_iter = Xs + (tm + sm) * BK_padded + sn;
    threadgroup const T_compute* Bs_iter = Ws + sm + (tn + sn) * BK_padded;

    MLX_MTL_PRAGMA_UNROLL
    for (int kf = 0; kf < KFR; ++kf) {
      simdgroup_barrier(mem_flags::mem_none);

      // A frag (TM, 1): per-thread reads dst[0]=src[0], dst[1]=src[1].
      vec<T_compute, 2> a_frag[TM];
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < TM; ++i) {
        threadgroup const T_compute* p = As_iter + i * TM_stride * BK_padded;
        a_frag[i][0] = p[0];
        a_frag[i][1] = p[1];
      }

      simdgroup_barrier(mem_flags::mem_none);

      // B frag (1, TN): per-thread reads dst[0]=src[0], dst[1]=src[BK_padded].
      vec<T_compute, 2> b_frag[TN];
      MLX_MTL_PRAGMA_UNROLL
      for (int j = 0; j < TN; ++j) {
        threadgroup const T_compute* p = Bs_iter + j * TN_stride * BK_padded;
        b_frag[j][0] = p[0];
        b_frag[j][1] = p[BK_padded];
      }

      simdgroup_barrier(mem_flags::mem_none);

      // Serpentine MMA: matches MLX `tile_matmad` (mma.h:411).
      // simdgroup_matrix runs in T_compute — that's the lever: when
      // T_compute=half on Apple7, we hit the fast f16 simdgroup
      // multiply-accumulate path instead of the slow bf16 emulation.
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < TM; ++i) {
        MLX_MTL_PRAGMA_UNROLL
        for (int j = 0; j < TN; ++j) {
          int j_serp = (i & 1) ? (TN - 1 - j) : j;
          simdgroup_matrix<T_compute, 8, 8> A_mat;
          simdgroup_matrix<T_compute, 8, 8> B_mat;
          simdgroup_float8x8 C_mat;
          simdgroup_float8x8 D_mat;
          reinterpret_cast<thread vec<T_compute, 2>&>(A_mat.thread_elements()) = a_frag[i];
          reinterpret_cast<thread vec<T_compute, 2>&>(B_mat.thread_elements()) = b_frag[j_serp];
          reinterpret_cast<thread vec<float, 2>&>(C_mat.thread_elements()) = acc[i][j_serp];
          simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
          acc[i][j_serp] = reinterpret_cast<thread vec<float, 2>&>(D_mat.thread_elements());
        }
      }

      // tile_stride_a/b = kFragSize * A_str_k = 8 (along K for A,
      // along K for transposed B).
      As_iter += 8;
      Bs_iter += 8;
    }

    // ─── Advance pointers per BlockLoader / QuantizedBlockLoader.next() ──
    //   X: tile_stride = BCOLS = BK halves.
    //   W: tile_stride = BCOLS_PACKED * bytes_per_pack = BK/pack_factor bytes.
    //   Scales/biases: advance once per `group_steps` BK iters (reduction_dim=1).
    X_src += BK;
    W_src += BCOLS_PACKED * bytes_per_pack;
    if (group_steps > 1) {
      group_step_cnt += 1;
      if (group_step_cnt == group_steps) {
        group_step_cnt = 0;
        Sc_row += 1;
        Bs_row += 1;
      }
    } else {
      Sc_row += 1;
      Bs_row += 1;
    }
  }

  // ─── Epilogue: direct per-lane register→device write with inline
  //   float→T_act cast. Matches MLX's `BlockMMA::store_result` +
  //   `MMAFrag::store` (steel/gemm/mma.h:534-546, :100-112).
  //
  //   Previous version went through a `threadgroup float` scratch
  //   buffer with a `simdgroup_store` + threadgroup_barrier + per-
  //   thread cast loop. That required an extra 4 KB of threadgroup
  //   memory, an extra barrier per output tile, and 2× the
  //   threadgroup-memory traffic (write scratch + read scratch).
  //   The direct path below uses Apple's `thread_elements()` MSL
  //   extension to access the 2 per-lane floats of each
  //   `simdgroup_float8x8` accumulator fragment in registers, casts
  //   to T_act, and writes straight to device memory.
  //
  //   Per-lane element layout in an 8×8 simdgroup_matrix on Apple7+
  //   (BaseMMAFrag<T, 8, 8>::get_coord):
  //     qid = lane / 4
  //     fm  = (qid & 4) | ((lane / 2) % 4)
  //     fn  = (qid & 2) * 2 + (lane % 2) * 2
  //   Each lane holds two consecutive elements at (fm, fn) and
  //   (fm, fn+1) within the 8×8 frag.
  const ushort qid    = simd_lane_id >> 2;
  const ushort lane_fm = (qid & 4) | ((simd_lane_id >> 1) & 3);
  const ushort lane_fn = (((qid & 2) << 1) | ((simd_lane_id & 1) << 1));
  // Epilogue output coords use the MLX interleaved frag layout
  // (mma.h: sm = tm + simd_coord.y; sn = tn + simd_coord.x; TM_stride =
  // kFragSize*WM = 16; TN_stride = kFragSize*WN = 16).
  //   tm = 8 * sgM (8 = kFragSize)
  //   tn = 8 * sgN
  //   frag(i,j)'s M-row = tm + i*TM_stride + lane_fm
  //   frag(i,j)'s N-col = tn + j*TN_stride + lane_fn
  const int tm_ep = 8 * sgM;
  const int tn_ep = 8 * sgN;
  constexpr int TM_stride_ep = 8 * WM;  // 16
  constexpr int TN_stride_ep = 8 * WN;  // 16
  MLX_MTL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i) {
    MLX_MTL_PRAGMA_UNROLL
    for (int j = 0; j < TN; ++j) {
      // acc is now vec<float, 2>; per-thread elements [0]/[1] map
      // to (lane_fm, lane_fn) and (lane_fm, lane_fn+1) per the
      // BaseMMAFrag<float,8,8> get_coord convention.
      const thread vec<float, 2>& elem = acc[i][j];
      int row = tm_ep + i * TM_stride_ep + int(lane_fm);
      int col = tn_ep + j * TN_stride_ep + int(lane_fn);
      if (m_full && n_full) {
        device T_act* p = y_block + row * N + col;
        p[0] = T_act(elem[0]);
        p[1] = T_act(elem[1]);
      } else {
        if (row < int(m_tile)) {
          if (col + 0 < int(n_tile)) {
            y_block[row * N + col + 0] = T_act(elem[0]);
          }
          if (col + 1 < int(n_tile)) {
            y_block[row * N + col + 1] = T_act(elem[1]);
          }
        }
      }
    }
  }
  (void)out_scratch;
}

// ─────────────────────────────────────────────────────────────────
// affine_qmm_t kernel wrapper — quantized.h:1707-1778
//
// batched=0 only; batched=1 needs `adjust_matrix_offsets`
// (`quantized.h:1351`), deferred to P13 alongside MoE gather.
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_compute, typename T_scale,
          int group_size, int bits, bool aligned_N>
[[kernel]] void affine_qmm_t_kernel(
    const device uint32_t*  w        [[buffer(0)]],
    const device T_scale*   scales   [[buffer(1)]],
    const device T_scale*   biases   [[buffer(2)]],
    const device T_act*     x        [[buffer(3)]],
    device T_act*           y        [[buffer(4)]],
    // buffer(5) / buffer(6) / buffer(7) (K / N / M) replaced by file-
    // scope function constants QMM_K / QMM_N / QMM_M so this kernel is
    // recordable into an MTLIndirectComputeCommand (no setKernelBytes).
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint3 tgid          [[threadgroup_position_in_grid]])
{
  constexpr int BM = 32, BN = 32, BK = 32;
  constexpr int BK_padded = BK + 16 / int(sizeof(T_compute));
  threadgroup T_compute Xs[BM * BK_padded];
  threadgroup T_compute Ws[BN * BK_padded];
  threadgroup float out_scratch[BM * BN];
  qmm_t_impl_inline<T_act, T_compute, T_scale, group_size, bits, aligned_N>(
      w, scales, biases, x, y,
      Xs, Ws, out_scratch,
      QMM_K, QMM_N, QMM_M, /*K_eff=*/QMM_K,
      simd_group_id, simd_lane_id, tgid);
}

// ─────────────────────────────────────────────────────────────────
// affine_qmm_t_splitk kernel wrapper — quantized.h:1780-1837
//
// Shifts the W / scales / biases / x / y pointers by `tid.z *
// k_partition_size` along K and `tid.z * split_k_partition_stride`
// along the M*N output dim, then calls qmm_t_impl with K_eff =
// k_partition_size. Caller (qmm_splitk in quantized.cpp:774) sums
// the resulting [split_k, M, N] intermediate along axis 0 into the
// final out.
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_compute, typename T_scale,
          int group_size, int bits, bool aligned_N>
[[kernel]] void affine_qmm_t_splitk_kernel(
    const device uint32_t*  w                        [[buffer(0)]],
    const device T_scale*   scales                   [[buffer(1)]],
    const device T_scale*   biases                   [[buffer(2)]],
    const device T_act*     x                        [[buffer(3)]],
    device T_act*           y                        [[buffer(4)]],
    // buffer(5)..buffer(9) replaced by file-scope function constants
    // QMM_K / QMM_N / QMM_M / QMM_K_PARTITION_SIZE; the
    // split_k_partition_stride that MLX passes at buffer(9) is
    // computed inline as QMM_M * QMM_N (see `quantized.cpp:808`).
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint3 tgid          [[threadgroup_position_in_grid]])
{
  constexpr int pack_factor    = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  const int k_start = int(tgid.z) * QMM_K_PARTITION_SIZE;
  const int split_k_partition_stride = QMM_M * QMM_N;

  const device T_act*   x_shift = x + k_start;
  const device uint8_t* wl      = (const device uint8_t*)w;
  wl += int64_t(k_start) * bytes_per_pack / pack_factor;
  const device T_scale* scales_shift = scales + k_start / group_size;
  const device T_scale* biases_shift = biases + k_start / group_size;
  device T_act*         y_shift      = y + int64_t(tgid.z) * split_k_partition_stride;

  constexpr int BM = 32, BN = 32, BK = 32;
  constexpr int BK_padded = BK + 16 / int(sizeof(T_compute));
  threadgroup T_compute Xs[BM * BK_padded];
  threadgroup T_compute Ws[BN * BK_padded];
  threadgroup float out_scratch[BM * BN];
  qmm_t_impl_inline<T_act, T_compute, T_scale, group_size, bits, aligned_N>(
      (const device uint32_t*)wl,
      scales_shift,
      biases_shift,
      x_shift,
      y_shift,
      Xs, Ws, out_scratch,
      QMM_K, QMM_N, QMM_M,
      /*K_eff=*/QMM_K_PARTITION_SIZE,
      simd_group_id, simd_lane_id, tgid);
}

// ─────────────────────────────────────────────────────────────────
// qmm_n_impl_inline — quantized.h:1221-1348. Same BlockMMA tile
// (BM=BN=BK=32, WM=WN=2, TM=TN=2, 4 8×8 frags/SG) as qmm_t, but with
// transpose_b=false. The differences vs qmm_t_impl_inline:
//
// 1. Ws layout is `[BK, BN_padded]` (K rows × N cols + pad) instead
//    of `[BN, BK_padded]`. B-frag is loaded without transpose, stride
//    `BN_padded`.
// 2. W src layout is `[K, N/pack_factor]` u32-packed (each K-row has
//    N packed values). Per-thread load: bi indexes K-row in [0, BK),
//    bj indexes N-col-packed in [0, BCOLS_PACKED). Per BK iter,
//    advance W_src by `BK * N * bytes_per_pack / pack_factor` bytes
//    (BlockLoader<reduction_dim=0>::tile_stride at
//    `quantized.h:614-616`).
// 3. Scales / biases layout is `[K, N/group_size]` (group along N).
//    Per-thread scale offset is `bi * (N / group_size)` (each K-row
//    has its own scales row). Per BK iter, advance by
//    `group_stride = BK * N / group_size`
//    (QuantizedBlockLoader<reduction_dim=0>::next() at
//    `quantized.h:686-687`).
// 4. No aligned_N flag — MLX qmm_n asserts caller guarantees N % BN
//    == 0 (see comment above; `quantized.cpp:1847` template signature).
//
// bits=4 only. group_size ∈ {32, 64, 128} (assert at
// `quantized.h:573-578`: BCOLS=BN=32 ≤ group_size). For gs=32, BN=gs
// → exactly one (scale, bias) per K-row per N-tile.
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits>
METAL_FUNC void qmm_n_impl_inline(
    const device uint32_t*  w,
    const device T_scale*   scales,
    const device T_scale*   biases,
    const device T_act*     x,
    device T_act*           y,
    threadgroup T_act*      Xs,
    threadgroup T_act*      Ws,
    threadgroup float*      out_scratch,
    const int              K,
    const int              N,
    const int              M,
    uint  simd_group_id,
    uint  simd_lane_id,
    uint3 tgid)
{
  static_assert(bits == 4, "qmm_n_impl_inline only instantiated for bits=4");
  static_assert(group_size == 32 || group_size == 64 || group_size == 128,
                "qmm_n_impl_inline expects group_size in {32, 64, 128}");

  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int TM = BM / (8 * WM);            // 2
  constexpr int TN = BN / (8 * WN);            // 2
  constexpr int KFR = BK / 8;                  // 4 K-frags per BK iter
  constexpr int BK_padded = BK + 16 / int(sizeof(T_act));
  constexpr int BN_padded = BN + 16 / int(sizeof(T_act));
  constexpr int TGP = WM * WN * SIMD_SIZE;     // 128

  constexpr int pack_factor    = get_pack_factor<bits, 8>();    // 2
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();    // 1
  constexpr int BCOLS_PACKED   = BN / pack_factor;              // 16
  // Per-thread W reads: BROWS=BK, BCOLS=BN → BCOLS_PACKED=16, n_reads
  // = (BCOLS_PACKED * BROWS) / tgp_size = (16 * 32) / 128 = 4
  // (`quantized.h:587-588`).
  constexpr int N_READS = (BCOLS_PACKED * BK) / TGP;            // 4

  // Per-thread X read: 8 halves per thread (BM*BK / TGP = 1024/128).
  constexpr int X_N_READS = (BM * BK) / TGP;                    // 8
  constexpr int X_TCOLS   = BK / X_N_READS;                     // 4
  static_assert(X_N_READS == 8, "expected 8 X halves per thread");

  const uint thread_idx = simd_group_id * SIMD_SIZE + simd_lane_id;

  // ── Output-tile origin and per-tile valid extents ──────────────
  const uint c_row = tgid.y * BM;  // along M
  const uint c_col = tgid.x * BN;  // along N
  if (c_row >= uint(M) || c_col >= uint(N)) return;
  const uint m_tile = (c_row + BM <= uint(M)) ? uint(BM) : uint(M) - c_row;
  const bool m_full = (m_tile == BM);

  // ── X loader thread coords (per BlockLoader; reduction_dim=1) ──
  const uint bi_x = thread_idx / uint(X_TCOLS);          // [0, BM)
  const uint bj_x = uint(X_N_READS) * (thread_idx % uint(X_TCOLS));  // 0,8,16,24

  // ── W loader thread coords (per QuantizedBlockLoader,
  //    reduction_dim=0 — BROWS=BK, BCOLS=BN) ────────────────────
  const uint bi_w = (uint(N_READS) * thread_idx) / uint(BCOLS_PACKED);     // [0, BK)
  const uint bj_w = (uint(N_READS) * thread_idx) % uint(BCOLS_PACKED);     // 0,4,8,12

  // ── Block base pointers (after y_row / y_col shift per
  //    `quantized.h:1267-1273`) ──────────────────────────────────
  const int N_w = N * bytes_per_pack / pack_factor;       // packed bytes per row of W
  const int N_g = N / group_size;                         // groups per row of W
  const device uint8_t* wl_base = (const device uint8_t*)w;
  const device T_act* x_block    = x + int64_t(c_row) * K;
  const device uint8_t* w_block  = wl_base + c_col * bytes_per_pack / pack_factor;
  const device T_scale* s_block  = scales + c_col / group_size;
  const device T_scale* b_block  = biases + c_col / group_size;
  device T_act* y_block          = y + int64_t(c_row) * N + c_col;

  // ── Per-thread source/dest pointers (BlockLoader constructor :47-58
  //    and QuantizedBlockLoader<reduction_dim=0> constructor
  //    :605-626) ─────────────────────────────────────────────────
  threadgroup T_act* Xs_dst = Xs + bi_x * BK_padded + bj_x;
  const device T_act* X_src = x_block + bi_x * K + bj_x;

  threadgroup T_act* Ws_dst = Ws + bi_w * BN_padded + bj_w * pack_factor;
  const device uint8_t* W_src = w_block + bi_w * N_w + bj_w * bytes_per_pack;
  const device T_scale* Sc_row = s_block + bi_w * N_g;
  const device T_scale* Bs_row = b_block + bi_w * N_g;
  // Group stride per BK iter for reduction_dim=0
  // (`quantized.h:618`): BROWS * src_ld / group_size = BK * N / gs.
  const int group_stride = BK * N / group_size;

  // ── Accumulators (BlockMMA Ctile = TM×TN of 8×8 float frags) ───
  simdgroup_float8x8 acc[TM][TN];
  MLX_MTL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i) {
    MLX_MTL_PRAGMA_UNROLL
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = simdgroup_float8x8(0.0f);
    }
  }

  const int sgM = int(simd_group_id) / WN;
  const int sgN = int(simd_group_id) % WN;

  // K loop — same MLX `(K % BK) != 0` tail handling pattern as
  // qmm_t. K_eff allows shortening (no qmm_n splitk in MLX so we
  // always use K_eff = K here, but keep the parameter for symmetry).
  const int k_blocks    = K / BK;
  const int k_tail      = K - k_blocks * BK;

  for (int kb = 0; kb < k_blocks; ++kb) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ─── X loader (load_unsafe for full M, load_safe for M-tail) ──
    if (m_full || bi_x < m_tile) {
      ((threadgroup vec<T_act, 4>*)Xs_dst)[0] =
          ((const device vec<T_act, 4>*)X_src)[0];
      ((threadgroup vec<T_act, 4>*)Xs_dst)[1] =
          ((const device vec<T_act, 4>*)X_src)[1];
    } else {
      ((threadgroup vec<T_act, 4>*)Xs_dst)[0] = vec<T_act, 4>(0);
      ((threadgroup vec<T_act, 4>*)Xs_dst)[1] = vec<T_act, 4>(0);
    }

    // ─── W loader (QuantizedBlockLoader<T, BK, BN, BN_padded, 0,
    //   TGP, gs, 4>::load_unsafe — no N-tail handling per MLX
    //   qmm_n's no-aligned_N assumption). Inline 4-bit dequant. ────
    {
      // In-register T_scale → T_act cast (`INT4_PARITY_PROBES.md` §7):
      // scales/biases ship F16 on disk; dequant math stays in T_act
      // (MLX quantized.h:521-527 keeps it in the kernel's scalar type).
      T_act scale = static_cast<T_act>(*Sc_row);
      T_act bias  = static_cast<T_act>(*Bs_row);
      T_act s0 = scale;
      T_act s1 = scale / static_cast<T_act>(16.0f);
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < N_READS; ++i) {
        uint8_t b = W_src[i * bytes_per_pack];
        Ws_dst[i * pack_factor + 0] =
            s0 * static_cast<T_act>(b & 0x0f) + bias;
        Ws_dst[i * pack_factor + 1] =
            s1 * static_cast<T_act>(b & 0xf0) + bias;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ─── BlockMMA::mma — KFR K-frag MMAs per BK iter; B-frag is
    //   loaded WITHOUT transpose because Ws is already [BK, BN]. ─
    MLX_MTL_PRAGMA_UNROLL
    for (int kf = 0; kf < KFR; ++kf) {
      simdgroup_matrix<T_act, 8, 8> A_frag[TM];
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < TM; ++i) {
        threadgroup const T_act* a_ptr =
            Xs + (sgM * 16 + i * 8) * BK_padded + kf * 8;
        simdgroup_load(A_frag[i], a_ptr, BK_padded);
      }

      simdgroup_matrix<T_act, 8, 8> B_frag[TN];
      MLX_MTL_PRAGMA_UNROLL
      for (int j = 0; j < TN; ++j) {
        int n_off = sgN * 16 + j * 8;
        threadgroup const T_act* b_ptr =
            Ws + (kf * 8) * BN_padded + n_off;
        simdgroup_load(B_frag[j], b_ptr, BN_padded);
      }

      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < TM; ++i) {
        MLX_MTL_PRAGMA_UNROLL
        for (int j = 0; j < TN; ++j) {
          simdgroup_multiply_accumulate(
              acc[i][j], A_frag[i], B_frag[j], acc[i][j]);
        }
      }
    }

    // ─── Advance pointers per BlockLoader / QuantizedBlockLoader.next() ──
    //   X: tile_stride = BCOLS = BK halves (reduction_dim=1).
    //   W: tile_stride = BROWS * src_ld * bytes_per_pack / pack_factor
    //      = BK * N * bytes_per_pack / pack_factor (reduction_dim=0).
    //   Scales/biases: + group_stride = BK * N / group_size per BK iter
    //      (reduction_dim=0 branch at `quantized.h:686-687`).
    X_src  += BK;
    W_src  += BK * N_w;
    Sc_row += group_stride;
    Bs_row += group_stride;
  }

  // ─── K-tail: load_safe with short2(BN, num_k) for both X and W.
  //   X: zero past `num_k` cols; W: zero past `num_k` rows. ─────
  if (k_tail > 0) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // X tail — for each thread, zero positions ≥ num_k in the row.
    {
      vec<T_act, 4> v0 = vec<T_act, 4>(0);
      vec<T_act, 4> v1 = vec<T_act, 4>(0);
      if (m_full || bi_x < m_tile) {
        MLX_MTL_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
          int col = int(bj_x) + i;
          if (col < k_tail) {
            ((thread T_act*)&v0)[i] = X_src[i];
          }
        }
        MLX_MTL_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
          int col = int(bj_x) + 4 + i;
          if (col < k_tail) {
            ((thread T_act*)&v1)[i] = X_src[4 + i];
          }
        }
      }
      ((threadgroup vec<T_act, 4>*)Xs_dst)[0] = v0;
      ((threadgroup vec<T_act, 4>*)Xs_dst)[1] = v1;
    }

    // W tail — zero rows past num_k in the BK direction
    // (load_safe with reduction_dim=0 short2(BCOLS=BN, num_k)
    // at `quantized.h:653-657`).
    if (int(bi_w) < k_tail) {
      // In-register T_scale → T_act cast (`INT4_PARITY_PROBES.md` §7):
      // scales/biases ship F16 on disk; dequant math stays in T_act
      // (MLX quantized.h:521-527 keeps it in the kernel's scalar type).
      T_act scale = static_cast<T_act>(*Sc_row);
      T_act bias  = static_cast<T_act>(*Bs_row);
      T_act s0 = scale;
      T_act s1 = scale / static_cast<T_act>(16.0f);
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < N_READS; ++i) {
        uint8_t b = W_src[i * bytes_per_pack];
        Ws_dst[i * pack_factor + 0] =
            s0 * static_cast<T_act>(b & 0x0f) + bias;
        Ws_dst[i * pack_factor + 1] =
            s1 * static_cast<T_act>(b & 0xf0) + bias;
      }
    } else {
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < N_READS * pack_factor; ++i) {
        Ws_dst[i] = T_act(0);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    MLX_MTL_PRAGMA_UNROLL
    for (int kf = 0; kf < KFR; ++kf) {
      simdgroup_matrix<T_act, 8, 8> A_frag[TM];
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < TM; ++i) {
        threadgroup const T_act* a_ptr =
            Xs + (sgM * 16 + i * 8) * BK_padded + kf * 8;
        simdgroup_load(A_frag[i], a_ptr, BK_padded);
      }

      simdgroup_matrix<T_act, 8, 8> B_frag[TN];
      MLX_MTL_PRAGMA_UNROLL
      for (int j = 0; j < TN; ++j) {
        int n_off = sgN * 16 + j * 8;
        threadgroup const T_act* b_ptr =
            Ws + (kf * 8) * BN_padded + n_off;
        simdgroup_load(B_frag[j], b_ptr, BN_padded);
      }

      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < TM; ++i) {
        MLX_MTL_PRAGMA_UNROLL
        for (int j = 0; j < TN; ++j) {
          simdgroup_multiply_accumulate(
              acc[i][j], A_frag[i], B_frag[j], acc[i][j]);
        }
      }
    }
  }

  // ─── Epilogue (same as qmm_t: simdgroup_store → out_scratch →
  //   per-thread cast to T). MLX qmm_n uses store_result vs
  //   store_result_safe based only on M-tail; we keep the same
  //   short-circuit (m_full ⇒ unconditional N-cols write since N
  //   is assumed aligned to BN by the caller). ──────────────────
  threadgroup_barrier(mem_flags::mem_threadgroup);

  MLX_MTL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i) {
    MLX_MTL_PRAGMA_UNROLL
    for (int j = 0; j < TN; ++j) {
      int row_base = sgM * 16 + i * 8;
      int col_base = sgN * 16 + j * 8;
      simdgroup_store(acc[i][j],
                      out_scratch + row_base * BN + col_base,
                      BN);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (m_full) {
    for (uint t = thread_idx; t < uint(BM * BN); t += uint(TGP)) {
      uint r = t / uint(BN);
      uint c = t % uint(BN);
      y_block[r * N + c] = T_act(out_scratch[t]);
    }
  } else {
    for (uint t = thread_idx; t < uint(BM * BN); t += uint(TGP)) {
      uint r = t / uint(BN);
      uint c = t % uint(BN);
      if (r < m_tile) {
        y_block[r * N + c] = T_act(out_scratch[t]);
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────
// affine_qmm_n kernel wrapper — quantized.h:1847-1897
//
// batched=0 only; batched=1 needs `adjust_matrix_offsets`
// (`quantized.h:1351`), deferred to P13 alongside MoE gather.
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits>
[[kernel]] void affine_qmm_n_kernel(
    const device uint32_t*  w        [[buffer(0)]],
    const device T_scale*   scales   [[buffer(1)]],
    const device T_scale*   biases   [[buffer(2)]],
    const device T_act*     x        [[buffer(3)]],
    device T_act*           y        [[buffer(4)]],
    // K / N / M ride as file-scope function constants
    // QMM_K / QMM_N / QMM_M (same as qmm_t for ICB recording).
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint3 tgid          [[threadgroup_position_in_grid]])
{
  constexpr int BM = 32, BN = 32, BK = 32;
  constexpr int BK_padded = BK + 16 / int(sizeof(T_act));
  constexpr int BN_padded = BN + 16 / int(sizeof(T_act));
  threadgroup T_act Xs[BM * BK_padded];
  threadgroup T_act Ws[BK * BN_padded];
  threadgroup float out_scratch[BM * BN];
  qmm_n_impl_inline<T_act, T_scale, group_size, bits>(
      w, scales, biases, x, y,
      Xs, Ws, out_scratch,
      QMM_K, QMM_N, QMM_M,
      simd_group_id, simd_lane_id, tgid);
}

// ─────────────────────────────────────────────────────────────────
// Instantiations — one symbol per (dtype, group_size, aligned_N)
// for affine_qmm_t (batched=0 only) and affine_qmm_t_splitk, plus
// one symbol per (dtype, group_size) for affine_qmm_n (batched=0).
// Matches the MLX dispatcher's `concatenate(...)` kernel-name pattern
// at `quantized.cpp:728-737` (qmm_t / qmm_n) and `:830-838` (splitk).
// ─────────────────────────────────────────────────────────────────

// INST_QMM_T_C: explicit compute-dtype instantiation. Symbol carries
// `_c_<compute_tag>` after `_<act_tag>` so existing `_s_<scale_tag>_gs_..`
// suffix parsing keeps working. Existing INST_QMM_T (below) is a thin
// wrapper that fixes T_compute = T_act, preserving every legacy symbol.
#define INST_QMM_T_C(act_tag, act_type, ctag, ctype, scale_tag, scale_type, gs, aln_tag, aln_val) \
  template [[host_name(                                                                            \
      "affine_qmm_t_" #act_tag "_c_" #ctag "_s_" #scale_tag "_gs_" #gs                             \
      "_b_4_alN_" #aln_tag "_batch_0")]] [[kernel]] void                                           \
  affine_qmm_t_kernel<act_type, ctype, scale_type, gs, 4, aln_val>(                                \
      const device uint32_t*   w        [[buffer(0)]],                                             \
      const device scale_type* scales   [[buffer(1)]],                                             \
      const device scale_type* biases   [[buffer(2)]],                                             \
      const device act_type*   x        [[buffer(3)]],                                             \
      device act_type*         y        [[buffer(4)]],                                             \
      uint  simd_group_id [[simdgroup_index_in_threadgroup]],                                      \
      uint  simd_lane_id  [[thread_index_in_simdgroup]],                                           \
      uint3 tgid          [[threadgroup_position_in_grid]]);

#define INST_QMM_T_SPLITK_C(act_tag, act_type, ctag, ctype, scale_tag, scale_type, gs, aln_tag, aln_val) \
  template [[host_name(                                                                                   \
      "affine_qmm_t_splitk_" #act_tag "_c_" #ctag "_s_" #scale_tag "_gs_" #gs                            \
      "_b_4_alN_" #aln_tag)]] [[kernel]] void                                                            \
  affine_qmm_t_splitk_kernel<act_type, ctype, scale_type, gs, 4, aln_val>(                               \
      const device uint32_t*   w                        [[buffer(0)]],                                   \
      const device scale_type* scales                   [[buffer(1)]],                                   \
      const device scale_type* biases                   [[buffer(2)]],                                   \
      const device act_type*   x                        [[buffer(3)]],                                   \
      device act_type*         y                        [[buffer(4)]],                                   \
      uint  simd_group_id [[simdgroup_index_in_threadgroup]],                                            \
      uint  simd_lane_id  [[thread_index_in_simdgroup]],                                                 \
      uint3 tgid          [[threadgroup_position_in_grid]]);

// Legacy INST_QMM_T: T_compute = T_act. Emits the old symbol name
// (no `_c_` segment) so existing pipeline-cache lookups keep hitting.
#define INST_QMM_T(act_tag, act_type, scale_tag, scale_type, gs, aln_tag, aln_val)   \
  template [[host_name(                                                               \
      "affine_qmm_t_" #act_tag "_s_" #scale_tag "_gs_" #gs                            \
      "_b_4_alN_" #aln_tag "_batch_0")]] [[kernel]] void                              \
  affine_qmm_t_kernel<act_type, act_type, scale_type, gs, 4, aln_val>(                \
      const device uint32_t*   w        [[buffer(0)]],                                \
      const device scale_type* scales   [[buffer(1)]],                                \
      const device scale_type* biases   [[buffer(2)]],                                \
      const device act_type*   x        [[buffer(3)]],                                \
      device act_type*         y        [[buffer(4)]],                                \
      uint  simd_group_id [[simdgroup_index_in_threadgroup]],                         \
      uint  simd_lane_id  [[thread_index_in_simdgroup]],                              \
      uint3 tgid          [[threadgroup_position_in_grid]]);

#define INST_QMM_T_SPLITK(act_tag, act_type, scale_tag, scale_type, gs, aln_tag, aln_val) \
  template [[host_name(                                                                   \
      "affine_qmm_t_splitk_" #act_tag "_s_" #scale_tag "_gs_" #gs                         \
      "_b_4_alN_" #aln_tag)]] [[kernel]] void                                             \
  affine_qmm_t_splitk_kernel<act_type, act_type, scale_type, gs, 4, aln_val>(             \
      const device uint32_t*   w                        [[buffer(0)]],                    \
      const device scale_type* scales                   [[buffer(1)]],                    \
      const device scale_type* biases                   [[buffer(2)]],                    \
      const device act_type*   x                        [[buffer(3)]],                    \
      device act_type*         y                        [[buffer(4)]],                    \
      uint  simd_group_id [[simdgroup_index_in_threadgroup]],                             \
      uint  simd_lane_id  [[thread_index_in_simdgroup]],                                  \
      uint3 tgid          [[threadgroup_position_in_grid]]);

#define INST_QMM_N(act_tag, act_type, scale_tag, scale_type, gs)               \
  template [[host_name(                                                        \
      "affine_qmm_n_" #act_tag "_s_" #scale_tag "_gs_" #gs                     \
      "_b_4_batch_0")]] [[kernel]] void                                        \
  affine_qmm_n_kernel<act_type, scale_type, gs, 4>(                            \
      const device uint32_t*   w        [[buffer(0)]],                         \
      const device scale_type* scales   [[buffer(1)]],                         \
      const device scale_type* biases   [[buffer(2)]],                         \
      const device act_type*   x        [[buffer(3)]],                         \
      device act_type*         y        [[buffer(4)]],                         \
      uint  simd_group_id [[simdgroup_index_in_threadgroup]],                  \
      uint  simd_lane_id  [[thread_index_in_simdgroup]],                       \
      uint3 tgid          [[threadgroup_position_in_grid]]);

#define INST_QMM_ALL(act_tag, act_type, scale_tag, scale_type, gs)             \
  INST_QMM_T(act_tag, act_type, scale_tag, scale_type, gs, true,  true)        \
  INST_QMM_T(act_tag, act_type, scale_tag, scale_type, gs, false, false)       \
  INST_QMM_T_SPLITK(act_tag, act_type, scale_tag, scale_type, gs, true,  true) \
  INST_QMM_T_SPLITK(act_tag, act_type, scale_tag, scale_type, gs, false, false)\
  INST_QMM_N(act_tag, act_type, scale_tag, scale_type, gs)

// Coverage: see header note in `quantized_qmv.metal` — T_scale=half
// (every mlx-community 4bit ships F16 scales), T_act per `torch_dtype`.
// `bfloat × bfloat` is removed — that was the P1-P6 regression path.
INST_QMM_ALL(f16,  half,   f16, half,  32)
INST_QMM_ALL(f16,  half,   f16, half,  64)
INST_QMM_ALL(f16,  half,   f16, half, 128)
INST_QMM_ALL(bf16, bfloat, f16, half,  32)
INST_QMM_ALL(bf16, bfloat, f16, half,  64)
INST_QMM_ALL(bf16, bfloat, f16, half, 128)

// Apple7 (M1) fast-path: bf16 device dtype, f16 compute. Skips the
// slow bf16 simdgroup_multiply_accumulate emulation and runs the MMA
// in half — ~1.7× faster on M1 Max for prefill matmuls. Output stays
// bf16 so the residual stream's bf16 dynamic range is preserved.
// Only qmm_t (not qmm_n; qmm_n isn't on the prefill hot path).
INST_QMM_T_C(bf16, bfloat, f16, half, f16, half, 64, true,  true)
INST_QMM_T_C(bf16, bfloat, f16, half, f16, half, 64, false, false)
INST_QMM_T_C(bf16, bfloat, f16, half, f16, half, 32, true,  true)
INST_QMM_T_C(bf16, bfloat, f16, half, f16, half, 32, false, false)
INST_QMM_T_C(bf16, bfloat, f16, half, f16, half, 128, true,  true)
INST_QMM_T_C(bf16, bfloat, f16, half, f16, half, 128, false, false)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, f16, half, 64, true,  true)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, f16, half, 64, false, false)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, f16, half, 32, true,  true)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, f16, half, 32, false, false)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, f16, half, 128, true,  true)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, f16, half, 128, false, false)
