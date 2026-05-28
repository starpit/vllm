// SPDX-License-Identifier: Apache-2.0
// ThunderMittens — paged-cache decode ATTENTION compute atom for the
// PD-wavefront persistent decode megakernel on Apple GPU. A faithful EXTRACTION
// (not a rewrite) of the `attention_via_cache_v2_{f16,bf16}_specialized` body in
// `attention.metal` — itself a paged-cache adaptation of MLX `sdpa_vector`
// (online softmax + per-simdgroup K-axis split). The f16/bf16 kernels are
// byte-identical bar the element type, so one template covers both.
//
// Composed by the thin attention_via_cache_v2_* [[kernel]] wrappers and the
// wavefront megakernel. The threadgroup combine scratch (tg_outputs/tg_max/
// tg_sum), the dims/scale, and the (seq, q_head) grid coordinates ride as
// parameters — the wrapper owns the threadgroup buffers and forwards its ATTN_*
// function constants + tg_pos; a megakernel passes its own slab and the
// schedule's coordinates. Self-contained (no function constants), like the
// other mittens atoms.
//
// Layout: 1 threadgroup per (seq, q_head); 1024 threads = BN(32) simdgroups ×
// BD(32) lanes; HEAD_DIM must be a multiple of 32 (qk_per_thread = HEAD_DIM/32,
// <= 8). The K loop has NO threadgroup_barrier (online softmax in registers);
// barriers appear only in the cross-simdgroup combine. A megakernel reusing the
// scratch across (seq, q_head) MUST barrier between them.
#pragma once
#include <metal_stdlib>

using namespace metal;

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

namespace mittens {

template <typename T>
METAL_FUNC void attention_decode_impl(
    device T* output,
    device const T* q,
    device const uint* seq_used_k,
    device const uint* block_table,
    device const T* k_cache,
    device const T* v_cache,
    threadgroup float* tg_outputs, // [BN * BD]
    threadgroup float* tg_max,     // [BN]
    threadgroup float* tg_sum,     // [BN]
    uint head_dim,
    uint num_q,
    uint num_kv,
    float scale,
    uint block_size,
    uint max_blocks,
    uint seq_idx,
    uint q_head_idx,
    uint simd_gid,
    uint simd_lid,
    uint q_head_base) {
  // NO DEFAULT — Metal `METAL_FUNC` does NOT reliably honour default args;
  // every caller MUST pass q_head_base (0 for whole-op attention).
  constexpr int BN = 32; // simdgroups per threadgroup
  constexpr int BD = 32; // lanes per simdgroup
  typedef float U;

  // qk_per_thread = HEAD_DIM / 32.
  const uint qk_per_thread = head_dim / uint(BD);

  // HEAD-RANGE region contract: `q_head_idx` is LOCAL within this block (it
  // indexes the block-based q/output operands), but the kv-head it reads is a
  // GLOBAL mapping — `q_head_base` is the block's first global q-head, so a
  // head-tiled attn (the partition) reads the right kv-head of the whole cache.
  // `q_head_base == 0` ⇒ whole-op attention (unchanged).
  const uint group_ratio = num_q / num_kv;
  const uint kv_head_idx = (q_head_base + q_head_idx) / group_ratio;
  const uint kv_len = seq_used_k[seq_idx];

  const uint kv_blk_stride = num_kv * block_size * head_dim;
  const uint kv_head_stride = block_size * head_dim;
  const uint kv_tok_stride = head_dim;

  thread U q_reg[8]; // qk_per_thread <= 8 (head_dim <= 256)
  thread U o_reg[8];

  device const T* q_row = q + (seq_idx * num_q + q_head_idx) * head_dim;
  device T* o_row = output + (seq_idx * num_q + q_head_idx) * head_dim;
  device const uint* row_block_table = block_table + seq_idx * max_blocks;

  // Pre-multiply Q by scale (MLX `sdpa_vector`: `q[i] = scale * queries[i]`).
  for (uint i = 0; i < qk_per_thread; ++i) {
    q_reg[i] = U(scale) * U(q_row[simd_lid * qk_per_thread + i]);
    o_reg[i] = 0;
  }

  U max_score = -FLT_MAX;
  U sum_exp_score = 0;

  // Online softmax over the K axis. Each simdgroup `simd_gid` covers tokens at
  // indices simd_gid, simd_gid + BN, ... — no cross-simd reduction in the loop.
  for (uint i = simd_gid; i < kv_len; i += uint(BN)) {
    const uint logical_block = i / block_size;
    const uint physical_block = row_block_table[logical_block];
    const uint token_in_block = i - logical_block * block_size;
    device const T* k_ptr = k_cache + physical_block * kv_blk_stride +
        kv_head_idx * kv_head_stride + token_in_block * kv_tok_stride +
        simd_lid * qk_per_thread;
    device const T* v_ptr = v_cache + physical_block * kv_blk_stride +
        kv_head_idx * kv_head_stride + token_in_block * kv_tok_stride +
        simd_lid * qk_per_thread;

    U score = 0;
    for (uint j = 0; j < qk_per_thread; ++j) {
      score += q_reg[j] * U(k_ptr[j]);
    }
    score = simd_sum(score);

    U new_max = max(max_score, score);
    U factor = metal::fast::exp(max_score - new_max);
    U exp_score = metal::fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    for (uint j = 0; j < qk_per_thread; ++j) {
      o_reg[j] = o_reg[j] * factor + exp_score * U(v_ptr[j]);
    }
  }

  // ── Combine per-simdgroup partials (online-softmax merge) ──────────
  if (simd_lid == 0) {
    tg_max[simd_gid] = max_score;
    tg_sum[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  U other_max = tg_max[simd_lid];
  U global_max = simd_max(other_max);
  U factor = metal::fast::exp(other_max - global_max);
  U global_sum = simd_sum(tg_sum[simd_lid] * factor);

  for (uint j = 0; j < qk_per_thread; ++j) {
    tg_outputs[simd_lid * BD + simd_gid] = o_reg[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    U val = tg_outputs[simd_gid * BD + simd_lid] * factor;
    U combined = simd_sum(val);
    if (global_sum != 0) {
      combined = combined / global_sum;
    }
    o_reg[j] = combined;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    device T* o_ptr = o_row + simd_gid * qk_per_thread;
    for (uint j = 0; j < qk_per_thread; ++j) {
      o_ptr[j] = T(o_reg[j]);
    }
  }
}

} // namespace mittens
