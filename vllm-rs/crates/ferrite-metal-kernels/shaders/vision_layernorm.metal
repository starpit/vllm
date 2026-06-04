// SPDX-License-Identifier: Apache-2.0
//
// LayerNorm-with-bias (Qwen3.5-VL / Qwen3-VL ViT norm1/norm2/merger.norm).
//
// The ViT norms are LayerNorm WITH BIAS (each carries .weight AND .bias) —
// NOT RMSNorm. No metal matcher / kernel existed; this is net-new. Faithful to
// `mlx.nn.LayerNorm` (biased variance, eps inside the sqrt, affine):
//     y = (x - mean) * rsqrt(var + eps) * weight + bias
//     mean = E[x],  var = E[x^2] - mean^2   (population / biased, ddof=0)
// Verified == mlx.nn.LayerNorm: max_abs_err 2.4e-7.
//
// One threadgroup per row (token); two threadgroup reductions (sum, sum-of-
// squares). Layout: row-major [M, HIDDEN]. weight/bias are [HIDDEN].
//
// Function constants: LN_M (rows), LN_HIDDEN (D), LN_EPS.

#include <metal_stdlib>

using namespace metal;

constant uint  LN_M      [[function_constant(0)]];
constant uint  LN_HIDDEN [[function_constant(1)]];
constant float LN_EPS    [[function_constant(2)]];

template <typename T>
[[kernel]] void vision_layernorm(
    device       T* output [[buffer(0)]],
    device const T* input  [[buffer(1)]],
    device const T* weight [[buffer(2)]],
    device const T* bias   [[buffer(3)]],
    uint gid     [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
  if (gid >= LN_M) {
    return;
  }
  threadgroup float sh_sum[1024];
  threadgroup float sh_sq[1024];

  float ls = 0.0f, lsq = 0.0f;
  for (uint i = tid; i < LN_HIDDEN; i += tg_size) {
    float v = float(input[gid * LN_HIDDEN + i]);
    ls += v;
    lsq += v * v;
  }
  sh_sum[tid] = ls;
  sh_sq[tid] = lsq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
    if (tid < stride) {
      sh_sum[tid] += sh_sum[tid + stride];
      sh_sq[tid] += sh_sq[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float inv_n = 1.0f / float(LN_HIDDEN);
  float mean = sh_sum[0] * inv_n;
  float var = sh_sq[0] * inv_n - mean * mean;
  var = max(var, 0.0f);                  // guard fp cancellation
  float inv = rsqrt(var + LN_EPS);

  for (uint i = tid; i < LN_HIDDEN; i += tg_size) {
    float v = float(input[gid * LN_HIDDEN + i]);
    float w = float(weight[i]);
    float b = float(bias[i]);
    output[gid * LN_HIDDEN + i] = T((v - mean) * inv * w + b);
  }
}

#define INST_VISION_LAYERNORM(dtype_tag, mtl_type)                            \
  template [[host_name("vision_layernorm_" #dtype_tag)]] [[kernel]] void      \
  vision_layernorm<mtl_type>(                                                 \
      device       mtl_type* output [[buffer(0)]],                           \
      device const mtl_type* input  [[buffer(1)]],                           \
      device const mtl_type* weight [[buffer(2)]],                           \
      device const mtl_type* bias   [[buffer(3)]],                           \
      uint gid     [[threadgroup_position_in_grid]],                         \
      uint tid     [[thread_position_in_threadgroup]],                       \
      uint tg_size [[threads_per_threadgroup]]);

INST_VISION_LAYERNORM(f16,  half)
INST_VISION_LAYERNORM(bf16, bfloat)
INST_VISION_LAYERNORM(f32,  float)
