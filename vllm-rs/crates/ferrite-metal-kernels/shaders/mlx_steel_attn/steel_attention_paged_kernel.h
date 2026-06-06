// SPDX-License-Identifier: Apache-2.0
//
// Paged-K/V-cache variant of MLX's steel_attention prefill kernel.
// Same FlashAttention-2 algorithm as `steel_attention_kernel.h`;
// only the K/V load path is replaced with `PagedKVBlockLoader` so
// the kernel reads from ferrite-metal's paged KV cache instead of
// a contiguous device buffer.
//
// Bindings (mirrors `attention_prefill_sdpa_v2_paged`):
//   buffer(0) Q             [total_q, num_q_heads, head_dim]
//   buffer(1) k_cache       [num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
//   buffer(2) v_cache       [num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
//   buffer(3) O             [total_q, num_q_heads, head_dim]
//   buffer(4) AttnParamsPaged
//   buffer(5) cu_seqlens_q  [batch+1]
//   buffer(6) seq_used_k    [batch]
//   buffer(7) block_table   [batch, max_blocks_per_seq]
//
// Dispatch grid: `(NQ_blocks_per_seq[seq], num_q_heads, batch)` —
// PER-SEQUENCE Q-block tiling so a BQ tile never straddles a
// sequence boundary. The caller (ferrite-forward lowering) is
// responsible for setting tid.z = seq_idx and tid.x = Q-block
// within that sequence's qL.
//
// (For our prefill, sequences typically run one at a time so
// tid.z=0; the structure generalizes cleanly.)

#include "attn.h"
#include "paged_loader.h"

using namespace mlx::steel;

// (No AttnParamsPaged struct — model params come in as function
// constants 0..5; see ATTN_PAGED_* declarations below.)

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

// Model-level params come in as function constants — pipeline
// specialization makes them compile-time literals inside the
// kernel. Slot numbers mirror `attention_prefill_sdpa_v2_paged`
// (`attention.metal`) so dispatcher code can share them.
//
// Slots 0 (HEAD_DIM) and 4 (BLOCK_SIZE) are accepted from the
// dispatcher for ABI consistency with the prior kernel but are
// unused in this body — both values are already baked as template
// parameters (`BD` and `BLOCK_SIZE_`) at metallib-compile time,
// which is strictly stronger constant folding.
constant uint  ATTN_PAGED_HEAD_DIM           [[function_constant(0)]];  // unused; ==BD
constant uint  ATTN_PAGED_NUM_Q_HEADS        [[function_constant(1)]];
constant uint  ATTN_PAGED_NUM_KV_HEADS       [[function_constant(2)]];
constant float ATTN_PAGED_SCALE              [[function_constant(3)]];
constant uint  ATTN_PAGED_BLOCK_SIZE         [[function_constant(4)]];  // unused; ==BLOCK_SIZE_
constant uint  ATTN_PAGED_MAX_BLOCKS_PER_SEQ [[function_constant(5)]];
// Reactive (chunked) KV pool: k_cache/v_cache (buffers 5/6) are
// per-layer chunk-address TABLES (device uint64 gpuAddresses), not the
// cache buffers. `PagedBlockLoaderT` derefs
// `chunk_table[physical / BLOCKS_PER_CHUNK]` per block. See
// `ferrite_fusion_synth::BLOCKS_PER_CHUNK`.
constant uint  ATTN_PAGED_BLOCKS_PER_CHUNK   [[function_constant(6)]];
// Sliding-window width (Gemma2/3/4 local layers): a query at absolute
// position q attends keys k with 0 <= q - k < window. 0 disables the
// window; the compiler folds every window branch away for
// full-attention pipelines (the dispatcher always sets slot 7 — 0 for
// full attention, W::SLIDING_WINDOW for the sliding prefill arm).
// Same slot/semantics as `ATTN_WINDOW` in attention.metal.
constant int   ATTN_PAGED_WINDOW             [[function_constant(7)]];

// Debug toggle. When `ATTN_PAGED_DEBUG_MODE != 0`, the kernel replaces
// its normal store path with a per-lane marker write so the bench can
// observe which (tid, simdgroup, lane) tuples actually reach the
// store. Modes:
//   1 = write (simd_group_id + 1) * 10 + 1 at O[(tm+sm)*Q_stride_tok + sn]
//       — per-lane single-element write, no Otile involvement.
//   2 = same as 1 but EVERY thread writes regardless of position —
//       proves the kernel reached this point at all.
constant uint  ATTN_PAGED_DEBUG_MODE         [[function_constant(99)]];

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x - y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    int BLOCK_SIZE_,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]]
void attention_paged(
    // Binding indices match the existing
    // `attention_prefill_sdpa_v2_paged_*` so dispatch wiring stays
    // unchanged when we swap kernels.
    device T*          O            [[buffer(0)]],   // [total_q, num_q_heads, head_dim]
    const device T*    Q            [[buffer(1)]],   // [total_q, num_q_heads, head_dim]
    const device uint* cu_seqlens_q [[buffer(2)]],   // [batch+1]
    const device uint* seq_used_k   [[buffer(3)]],   // [batch] total cached K
    const device uint* block_table  [[buffer(4)]],   // [batch, max_blocks_per_seq]
    const device uint64_t* k_cache  [[buffer(5)]],   // per-layer chunk-address table
    const device uint64_t* v_cache  [[buffer(6)]],   // per-layer chunk-address table
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  (void)lid;

  // tid layout: tid.x = Q-block within sequence (0..NQ_blocks-1)
  //             tid.y = q-head index             (0..num_q_heads-1)
  //             tid.z = seq_idx                  (0..batch-1)
  const uint seq_idx     = tid.z;
  const uint q_head_idx  = tid.y;
  const uint gqa_factor  = ATTN_PAGED_NUM_Q_HEADS / ATTN_PAGED_NUM_KV_HEADS;
  const uint kv_head_idx = q_head_idx / gqa_factor;

  const uint seq_start   = cu_seqlens_q[seq_idx];
  const uint seq_end     = cu_seqlens_q[seq_idx + 1];
  const uint new_q_for_seq = seq_end - seq_start;
  const uint kv_len      = seq_used_k[seq_idx];
  const uint prefix_len  = kv_len - new_q_for_seq;
  const uint q_block_base = tid.x * uint(BQ);
  const uint global_q_base = seq_start + q_block_base;

  if (q_block_base >= new_q_for_seq) {
    return;
  }
  const uint q_tile_rows = min(uint(BQ), new_q_for_seq - q_block_base);
  const bool q_tile_full = (q_tile_rows == uint(BQ));

  const int Q_stride_tok = int(ATTN_PAGED_NUM_Q_HEADS) * BD;
  Q += int(global_q_base) * Q_stride_tok + int(q_head_idx) * BD;
  O += int(global_q_base) * Q_stride_tok + int(q_head_idx) * BD;

  const int kv_blk_stride  = int(ATTN_PAGED_NUM_KV_HEADS) * BLOCK_SIZE_ * BD;
  const int kv_head_stride = BLOCK_SIZE_ * BD;
  const int per_token_stride = BD;
  const device uint* row_block_table = block_table + seq_idx * ATTN_PAGED_MAX_BLOCKS_PER_SEQ;

  // Prepare threadgroup memory
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T Q_smem[BQ * (BD + padQ)];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  // Prepare block loaders
  // Q: contiguous (just like MLX). Stride along Q-token axis is
  // num_q_heads * head_dim (we've pre-offset Q to (global_q, head)).
  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDQ_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  // K and V: paged. Use `PagedBlockLoaderT` — inherits MLX's
  // `BlockLoaderT` byte-for-byte and overrides only `next()` to
  // advance via the block table. Same compiler codegen as the contig
  // path's load_unsafe / load_safe.
  using KBlockLoader = PagedBlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ 1,
      /* short kDstStrCol = */ LDK_tgp,
      /* short tgp_size = */ WM * WN * 32,
      /* short BLOCK_SIZE_ = */ BLOCK_SIZE_>;

  using VBlockLoader = PagedBlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDV_tgp,
      /* short kDstStrCol = */ 1,
      /* short tgp_size = */ WM * WN * 32,
      /* short BLOCK_SIZE_ = */ BLOCK_SIZE_>;

  QBlockLoader loader_q(
      Q, Q_stride_tok, Qs, simd_group_id, simd_lane_id);
  // Chunked KV: pass the chunk-address table + the per-block head
  // offset (kv_head_idx * kv_head_stride) + BLOCKS_PER_CHUNK; the
  // loader resolves the chunk base per physical block (it can no
  // longer be folded into a single `cache_base`).
  KBlockLoader loader_k(
      k_cache,
      per_token_stride,
      kv_blk_stride,
      int(kv_head_idx) * kv_head_stride,
      int(ATTN_PAGED_BLOCKS_PER_CHUNK),
      row_block_table,
      Ks,
      simd_group_id,
      simd_lane_id);
  VBlockLoader loader_v(
      v_cache,
      per_token_stride,
      kv_blk_stride,
      int(kv_head_idx) * kv_head_stride,
      int(ATTN_PAGED_BLOCKS_PER_CHUNK),
      row_block_table,
      Vs,
      simd_group_id,
      simd_lane_id);

  const AccumType scale = static_cast<AccumType>(ATTN_PAGED_SCALE) * M_LOG2E_F;

  // Prepare MMA tiles
  constexpr short kFragSize = 8; // MMAFrag size
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  // KV sequence frags (all warps load the same frags)
  constexpr int TK = BK / kFragSize;
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "Check TQ");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q blocks. Bounds-checked load uses `q_tile_rows` (this
  // sequence's remaining new-Q tokens at this Q-block).
  if (q_tile_full) {
    loader_q.load_unsafe();
  } else {
    loader_q.load_safe(short2(BD, int(q_tile_rows)));
  }

  // Init row reduction variables
  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  // KV-block iteration bounds. The Q tile spans rows [q_block_base,
  // q_block_base + q_tile_rows) in *new-tokens* coords; in absolute
  // K-axis coords those rows are at [prefix_len + q_block_base,
  // prefix_len + q_block_base + q_tile_rows). Causal-mask cuts off
  // K positions > this Q's absolute row position; the kb upper
  // bound is the block containing the last masked-IN K position.
  const int abs_q_min = int(prefix_len) + int(q_block_base);
  const int abs_q_max_excl =
      int(prefix_len) + int(q_block_base) + int(q_tile_rows);
  const int kv_blocks_total = int((kv_len + uint(BK) - 1u) / uint(BK));
  int kb_lim = (abs_q_max_excl + BK - 1) / BK;
  if (kb_lim > kv_blocks_total) kb_lim = kv_blocks_total;
  // First kb that needs causal masking. The Q tile's earliest
  // absolute row is `prefix_len + q_block_base`; any K position
  // strictly past that row is masked-out causally.
  const int kb_min_causal = abs_q_min / BK;
  // Sliding window: the EARLIEST key any row of this Q tile attends
  // is `abs_q_min - window + 1` (row q attends k iff 0 <= q - k <
  // window, and abs_q_min is the tile's smallest q). K tiles entirely
  // before that are skipped — this is what makes windowed prefill
  // O(T·window) instead of O(T²). Branch folds away at window == 0.
  int kb_start = 0;
  if (ATTN_PAGED_WINDOW > 0) {
    const int first_k = abs_q_min - ATTN_PAGED_WINDOW + 1;
    if (first_k > 0) {
      kb_start = first_k / BK;
    }
  }
  // Last kb that's a full BK-wide tile (the rest of kv_len fits
  // in this block partially).
  const int kv_aligned_blocks = int(kv_len / uint(BK));
  const uint kv_rem = kv_len % uint(BK);

  // Fast-forward the paged loaders past the skipped tiles (O(1) —
  // one block-table read each).
  if (kb_start > 0) {
    loader_k.seek(kb_start);
    loader_v.seek(kb_start);
  }

  // Loop over KV seq length
  for (int kb = kb_start; kb < kb_lim; kb++) {
    // Load K block from paged cache. The last block may be partial
    // if kv_len is not a multiple of BK; use load_safe to zero the
    // tail.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kb == kv_aligned_blocks && kv_rem != 0u) {
      loader_k.load_safe(short2(BD, int(kv_rem)));
    } else {
      loader_k.load_unsafe();
    }

    // Do S = Q @ K.T
    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    // Apply scale in float32
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    // Mask out partial last K block (positions past kv_len are
    // padding inside the last paged block).
    if (kb == kv_aligned_blocks && kv_rem != 0u) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= int(kv_rem)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Causal mask. Row in our tile corresponds to absolute K-axis
    // row `prefix_len + q_block_base + tm + sm + i*kFragRows`; K
    // position is `kb*BK + sn + j*kFragCols + jj`. Mask out any K
    // position strictly greater than the corresponding Q row.
    if (kb >= kb_min_causal) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = int(prefix_len) + int(q_block_base)
                          + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Sliding-window mask: row q attends k iff q - k < window, so
    // mask out k <= q - window. Only boundary tiles need it — a tile
    // needs masking iff its OLDEST key can fall outside the YOUNGEST
    // row's window (`abs_q_max_excl - 1 - kb*BK >= window`); fully
    // out-of-window tiles were already skipped via `kb_start`. The
    // whole block folds away for full-attention pipelines (window 0).
    if (ATTN_PAGED_WINDOW > 0 &&
        (abs_q_max_excl - 1 - kb * BK) >= ATTN_PAGED_WINDOW) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = int(prefix_len) + int(q_block_base)
                          + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((row_pos - (col_pos + jj)) >= ATTN_PAGED_WINDOW) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load V block from paged cache (same partial-tail handling).
    if (kb == kv_aligned_blocks && kv_rem != 0u) {
      loader_v.load_safe(short2(BD, int(kv_rem)));
    } else {
      loader_v.load_unsafe();
    }

    // Do softmax

    // Temp variables
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
    }

    // Save max for next iteration
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    // Row Sum
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    // Update norm
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    // Load V into registers
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;

          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);

          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }

    // Prepare for next iteration
    loader_k.next();
    loader_v.next();
  }

  // Normalize output (skip in debug mode 4 to preserve the markers).
  if (ATTN_PAGED_DEBUG_MODE != 4u) {
    Otile.template row_bin_op<DivOp>(sum_score);
  }

  // DEBUG_MODE=3: overwrite Otile with constant 1.0 to test whether
  // the Otile.template store<T,1,1>(O, Q_stride_tok) path itself
  // correctly writes 1024 elements per simdgroup (4096 per TG).
  // If yes → bug is in upstream Otile compute. If no → bug is in
  // store path (despite simd-coverage diagnostic saying it's reached).
  if (ATTN_PAGED_DEBUG_MODE == 3u) {
    STEEL_PRAGMA_UNROLL
    for (short f = 0; f < decltype(Otile)::kNumFrags; f++) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < decltype(Otile)::kElemsPerFrag; e++) {
        Otile.val_frags[f][e] = AccumType(1);
      }
    }
  }

  // DEBUG_MODE=4: assign each frag a unique marker `id + 1` (1..16)
  // AFTER the MMA loop, so the store reads from these markers. If
  // output shows correct markers in each frag's col range, the bug
  // is in MMA accumulation upstream. If output is still patterned
  // (frag 14 = 0), the bug is in `row_bin_op<DivOp>` or `store`.
  if (ATTN_PAGED_DEBUG_MODE == 4u) {
    STEEL_PRAGMA_UNROLL
    for (short f = 0; f < decltype(Otile)::kNumFrags; f++) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < decltype(Otile)::kElemsPerFrag; e++) {
        Otile.val_frags[f][e] = AccumType(f + 1);
      }
    }
    // Skip the div_by_sum_score below — we want markers, not normalized.
  }

  threadgroup_barrier(mem_flags::mem_none);

  // Store results. O is already pre-offset to (global_q_base,
  // q_head_idx). O row stride along the Q-token axis is
  // `Q_stride_tok = num_q_heads * BD`.
  O += (tm + sm) * Q_stride_tok + sn;

  if (ATTN_PAGED_DEBUG_MODE == 1u) {
    // Marker write: each lane writes (sg+1)*10 + 1.0 at its O slot.
    // 128 lanes per TG → 128 distinct O writes per TG. If we see this
    // marker, that (tid, simdgroup, lane) reached the store path.
    O[0] = T(AccumType(simd_group_id + 1u) * AccumType(10) + AccumType(1));
    return;
  }
  if (ATTN_PAGED_DEBUG_MODE == 2u) {
    // Write to O[(tid.x, tid.y, simd_group_id, simd_lane_id) hash slot]
    // — a single global location per (kernel-position, lane) tuple.
    // This proves the kernel REACHED this point regardless of where
    // store_safe / store would have routed the lane.
    const uint slot = ((tid.x * 128u) + (tid.y * 4u) + simd_group_id) * 32u + simd_lane_id;
    // Reuse buffer(3) (seq_used_k) is too small — repurpose O global
    // base offset (the original kernel didn't pre-offset for debug).
    // Map slot to O at slot*1 (assuming buffer is big enough; bench
    // sizes O = M * num_q * D >> 768*4*32 = 98k).
    device T* O_base = O - (int)((tm + sm) * Q_stride_tok + sn);
    if (slot < (uint)(int(ATTN_PAGED_NUM_Q_HEADS) * BD * 1024)) {
      O_base[slot] = T(AccumType(simd_group_id + 1u) * AccumType(10) +
                       AccumType(simd_lane_id) * AccumType(0.01));
    }
    return;
  }

  if (!q_tile_full) {
    auto dst_tile_dims =
        short2(BD - sn, int(q_tile_rows) - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Otile.template store_safe<T, 1, 1>(O, Q_stride_tok, dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, Q_stride_tok);
  }
}
