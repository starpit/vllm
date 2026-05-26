// SPDX-License-Identifier: Apache-2.0
// ThunderMittens — RoPE COMPUTE atoms for the PD-wavefront persistent decode
// megakernel on Apple GPU. Faithful EXTRACTIONS (not rewrites) of
// `rope_append_{f16,bf16}_specialized` (`rope.metal`): the NeoX pair-rotation
// (`rope_rotate_pair`) and the paged KV-cache commit (`kv_paged_write`).
//
// The wavefront megakernel composes these as a `rope_append`-style K op
// (rotate K, then write rotated K + un-rotated V into the paged cache) so the
// downstream attention reads the new token from the cache exactly like the
// oracle non-mega path does — that is what makes the megakernel Tier-B
// bit-exact (it runs the oracle's exact computation). The Q-side stays
// rotation-only. cos/sin rows are precomputed (cos_sin[pos*rot_dim ..]); the
// caller slices the (token, head) row and the rotation is per element `d`.
#pragma once
#include <metal_stdlib>

using namespace metal;

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

namespace mittens {

// NeoX (half-split) RoPE: rotate the (d, d + half_dim) pair of `row` in place
// using precomputed cos_row[d] / sin_row[d]. Only d < half_dim does work — it
// owns the pair (d, d + half_dim); threads with d >= half_dim are the upper
// partners and no-op (the d < half_dim thread writes both halves). For
// partial-rope models (rot_dim < head_dim) the tail [rot_dim, head_dim) is left
// unrotated by construction (those `d` are >= half_dim of the rotated span only
// when the caller passes the full head; callers gate by passing the rot span).
template <typename T>
METAL_FUNC void rope_rotate_pair(
    device T* row,
    device const T* cos_row,
    device const T* sin_row,
    uint d,
    uint half_dim) {
  if (d < half_dim) {
    const float c = float(cos_row[d]);
    const float s = float(sin_row[d]);
    const float x0 = float(row[d]);
    const float x1 = float(row[half_dim + d]);
    row[d] = T(x0 * c - x1 * s);
    row[half_dim + d] = T(x1 * c + x0 * s);
  }
}

// Paged KV-cache append — the COMMIT half of `rope_append_*_specialized`
// (rope.metal:317-342), extracted verbatim. Copy one element `d` of one
// kv-head's rotated `k_row` and un-rotated `v_row` into the per-layer paged
// cache at `slot`. Cache layout `[num_blocks, num_kv, block_size, head_dim]`;
// `slot` is the flat token index, split into (block_id, block_offset). The
// `0xFFFFFFFF` sentinel marks a padding lane (pool.rs fills padding
// `slot_mapping` with `u32::MAX`) and skips the write so it can't corrupt slot
// 0. The caller must fence the K rotation (`threadgroup_barrier(mem_device)`)
// before this reads `k_row[d]` — for `d >= half_dim` that element was written
// by another thread during rotation.
template <typename T>
METAL_FUNC void kv_paged_write(
    device T* kv_cache_k,
    device T* kv_cache_v,
    device const T* k_row, // this kv-head's rotated K
    device const T* v_row, // this kv-head's un-rotated V
    uint slot,
    uint kv_head,
    uint d,
    uint num_kv,
    uint block_size,
    uint head_dim) {
  if (slot == 0xFFFFFFFFu) {
    return;
  }
  const uint block_id = slot / block_size;
  const uint block_offset = slot % block_size;
  const uint blk_stride = num_kv * block_size * head_dim;
  const uint head_stride = block_size * head_dim;
  const uint off = block_id * blk_stride + kv_head * head_stride + block_offset * head_dim;
  kv_cache_k[off + d] = k_row[d];
  kv_cache_v[off + d] = v_row[d];
}

} // namespace mittens
