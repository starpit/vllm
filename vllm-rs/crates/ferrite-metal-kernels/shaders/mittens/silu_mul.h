// SPDX-License-Identifier: Apache-2.0
// ThunderMittens — SwiGLU silu·mul COMPUTE atom for the PD-wavefront persistent
// decode megakernel on Apple GPU. A faithful EXTRACTION (not a rewrite) of the
// `silu_mul` body in `silu_mul.metal`.
//
// Composed by the thin `silu_mul` [[kernel]] wrapper and the wavefront
// megakernel. The total element count rides as a parameter (the wrapper
// forwards its SILU_MUL_N function constant; a megakernel passes the schedule's
// element count) — self-contained, like the other mittens atoms.
#pragma once
#include <metal_stdlib>

using namespace metal;

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

namespace mittens {

// out[gid] = silu(gate[gid]) * up[gid], silu(x) = x / (1 + exp(-x)). The silu
// is kept in float so the denormalized half/bfloat exp() tail stays
// representable; `n` guards the trailing partial threadgroup.
template <typename T>
METAL_FUNC void silu_mul_impl(
    device T* out,
    const device T* gate,
    const device T* up,
    uint gid,
    uint n) {
  if (gid >= n) {
    return;
  }
  float g = float(gate[gid]);
  float u = float(up[gid]);
  float silu_g = g / (1.0f + exp(-g));
  out[gid] = static_cast<T>(silu_g * u);
}

} // namespace mittens
