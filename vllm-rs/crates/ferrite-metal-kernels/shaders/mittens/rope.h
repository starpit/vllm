// SPDX-License-Identifier: Apache-2.0
// ThunderMittens — RoPE rotation COMPUTE atom for the PD-wavefront persistent
// decode megakernel on Apple GPU. A faithful EXTRACTION (not a rewrite) of the
// NeoX pair-rotation body shared by `rope_append_{f16,bf16}_specialized` in
// `rope.metal`.
//
// Just the ROTATION — per locked design decision #4 the megakernel keeps the
// new token's rotated K as a dataflow edge into attention (no cache round-trip),
// so the paged KV-cache write stays in the [[kernel]] wrapper as a commit
// side-output, NOT in this atom. The cos/sin rows are precomputed
// (cos_sin[pos*rot_dim ..]); the caller slices the (token, head) row and the
// rotation is per element `d`.
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

} // namespace mittens
