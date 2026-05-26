// SPDX-License-Identifier: Apache-2.0
// ThunderMittens — RMSNorm COMPUTE atom for the PD-wavefront persistent decode
// megakernel on Apple GPU. A faithful EXTRACTION (not a rewrite — preserves the
// validated math) of the `rmsnorm_specialized_impl` body in `rmsnorm.metal`
// (itself the in-register-cast port of MLX RMSNorm).
//
// Composed by the thin `rmsnorm_*_specialized` [[kernel]] wrapper and the
// wavefront megakernels. The threadgroup reduction scratch (`shared_sum`) and
// the dims/eps ride as parameters — the wrapper owns the threadgroup buffer and
// forwards its RMSNORM_M/HIDDEN_SIZE/EPS function constants; a megakernel passes
// its own threadgroup slab and the schedule's dims. Self-contained (no function
// constants), exactly like qmv_*_impl take in_vec_size/out_vec_size by value.
//
// BIT-EXACTNESS NOTE: the parallel tree reduction's summation order depends on
// `tg_size`, so a megakernel composing this atom must reduce over the SAME
// thread count as its reference and must NOT reuse `shared_sum` across rows
// without a `threadgroup_barrier` between them.
#pragma once
#include <metal_stdlib>

using namespace metal;

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

namespace mittens {

// y = x * weight / sqrt(mean(x^2) + eps), for the single row `gid`.
// `shared_sum` is caller-provided threadgroup scratch of at least `tg_size`
// floats. All `tg_size` threads of the threadgroup cooperate on one row.
template <typename T_act, typename T_scale>
METAL_FUNC void rmsnorm_impl(
    device T_act* output,
    device const T_act* input,
    device const T_scale* weight,
    threadgroup float* shared_sum,
    uint m,
    uint hidden_size,
    float eps,
    uint gid,
    uint tid,
    uint tg_size) {
  if (gid >= m) return;

  float local_sum = 0.0f;
  for (uint i = tid; i < hidden_size; i += tg_size) {
    float val = float(input[gid * hidden_size + i]);
    local_sum += val * val;
  }
  shared_sum[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float rms = sqrt(shared_sum[0] / float(hidden_size) + eps);

  for (uint i = tid; i < hidden_size; i += tg_size) {
    float val = float(input[gid * hidden_size + i]);
    float w = float(weight[i]);
    output[gid * hidden_size + i] = T_act((val / rms) * w);
  }
}

} // namespace mittens
