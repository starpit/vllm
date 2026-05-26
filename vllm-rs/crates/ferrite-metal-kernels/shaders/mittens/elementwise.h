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

} // namespace mittens
