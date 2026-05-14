// SPDX-License-Identifier: Apache-2.0
//
// In-place L1 row-renormalize for the MoE router top-k scores:
//
//   scores[n, k] /= sum(scores[n, :])
//
// Implements `norm_topk_prob` from qwen3_moe.py:134:
//
//   if self.norm_topk_prob:
//       scores /= mx.sum(scores, axis=-1, keepdims=True)
//
// MLX expresses this as separate `mx.sum` + broadcast-divide ops; we
// fuse them into one tiny kernel since `top_k` is bounded and small
// (≤ 16 in every shipped model). One thread per row; the row loop is
// unrolled by the compiler at `MWS_TOP_K`-specialization time.
//
// Function constants:
//   TKR_TOP_K — top-k row length.
//
// Bindings:
//   buffer(0) = scores  T  [N, top_k] (in-place)
//
// Grid: (N,) threads, one thread per row.

#include <metal_stdlib>

using namespace metal;

constant uint TKR_TOP_K [[function_constant(0)]];

template <typename T>
[[kernel]] void top_k_renormalize(
    device T* scores [[buffer(0)]],
    uint n           [[thread_position_in_grid]])
{
    const size_t row_base = size_t(n) * size_t(TKR_TOP_K);
    float sum = 0.0f;
    for (uint k = 0; k < TKR_TOP_K; ++k) {
        sum += float(scores[row_base + k]);
    }
    if (sum <= 0.0f) {
        return;
    }
    float inv = 1.0f / sum;
    for (uint k = 0; k < TKR_TOP_K; ++k) {
        scores[row_base + k] = T(float(scores[row_base + k]) * inv);
    }
}

#define INST_TKR(dtype_tag, mtl_type)                                  \
    template [[host_name("top_k_renormalize_" #dtype_tag)]] [[kernel]] \
    void top_k_renormalize<mtl_type>(                                  \
        device mtl_type* scores [[buffer(0)]],                         \
        uint n                  [[thread_position_in_grid]]);

INST_TKR(f16,  half)
INST_TKR(bf16, bfloat)
