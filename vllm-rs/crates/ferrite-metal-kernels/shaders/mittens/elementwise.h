// SPDX-License-Identifier: Apache-2.0
// ThunderMittens — trivial elementwise COMPUTE atoms for the PD-wavefront
// persistent decode megakernel. The residual add (and any future plain
// binary/unary elementwise) the player composes, kept as named atoms so the
// player stays "compose atoms, never inline math" even for the trivial ops.
// Float accumulation so the bf16/f16 add matches the region IR's f32 eval.
#pragma once
#include <metal_stdlib>

using namespace metal;

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

namespace mittens {

// out[gid] = a[gid] + b[gid] (residual add). `n` guards the trailing partial
// threadgroup.
template <typename T>
METAL_FUNC void add_impl(
    device T* out,
    const device T* a,
    const device T* b,
    uint gid,
    uint n) {
  if (gid >= n) {
    return;
  }
  out[gid] = static_cast<T>(float(a[gid]) + float(b[gid]));
}

// out[gid] = sum_k partials[k][gid] — the split-K all-reduce. `partials` is the
// bindless gpuAddress sub-table (`num_partials` device pointers); each is read
// as T. Accumulated in FLOAT and rounded to T exactly ONCE, matching the region
// IR's f32 `SumReduce` (so the only difference between split-K and a whole
// matvec is this single trailing round — PLAN Tier A′). A consuming worker has
// already ACQUIREd any cross-worker partial into a local copy, so every pointer
// here is locally readable. `n` guards the trailing partial threadgroup.
template <typename T>
METAL_FUNC void sum_reduce_impl(
    device T* out,
    const device ulong* partials,
    uint num_partials,
    uint gid,
    uint n) {
  if (gid >= n) {
    return;
  }
  float acc = 0.0f;
  for (uint k = 0; k < num_partials; k++) {
    const device T* p = (const device T*)(partials[k]);
    acc += float(p[gid]);
  }
  out[gid] = static_cast<T>(acc);
}

} // namespace mittens
