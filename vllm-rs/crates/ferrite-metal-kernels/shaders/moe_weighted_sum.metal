// SPDX-License-Identifier: Apache-2.0
//
// MoE final reduction: combine top-k expert outputs with their gating
// scores, summing across the top_k axis.
//
//   out[n, d] = sum_k(expert[n, k, d] * scores[n, k])
//
// Faithful port of the MLX expression at qwen3_moe.py:137:
//   y = (y * scores[..., None]).sum(axis=-2)
//
// where `y` is `[N, top_k, hidden]` and `scores` is `[N, top_k]`. We
// fuse the broadcast-multiply and the sum-reduce into one kernel so
// the `[N, top_k, hidden]` intermediate is consumed once instead of
// landing through device memory twice (MLX's two separate ops).
//
// Function constants:
//   MWS_TOP_K — top-k axis extent (e.g. 8 for Qwen3-MoE).
//   MWS_HIDDEN — output last-dim extent (= W::Q_SIZE on the routed path).
//
// Bindings:
//   buffer(0) = out        T  [N, hidden]
//   buffer(1) = expert_out T  [N, top_k, hidden]
//   buffer(2) = scores     T  [N, top_k]
//
// Grid: (hidden_padded, N, 1) threads, one thread per output element.
// hidden is rounded up to a tg-friendly multiple by the caller; threads
// with d >= MWS_HIDDEN early-out. Float accumulator so half/bfloat
// underflow on small-score products doesn't lose precision.

#include <metal_stdlib>

using namespace metal;

constant uint MWS_TOP_K  [[function_constant(0)]];
constant uint MWS_HIDDEN [[function_constant(1)]];

template <typename T>
[[kernel]] void moe_weighted_sum(
    device       T* out         [[buffer(0)]],
    const device T* expert_out  [[buffer(1)]],
    const device T* scores      [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    const uint n = gid.y;
    if (d >= MWS_HIDDEN) {
        return;
    }
    // expert_out[n, k, d] = expert_out[(n * top_k + k) * hidden + d]
    // scores[n, k]        = scores[n * top_k + k]
    const size_t scores_base = size_t(n) * size_t(MWS_TOP_K);
    const size_t expert_row_stride = size_t(MWS_HIDDEN);
    const size_t expert_base =
        scores_base * expert_row_stride + size_t(d);
    float acc = 0.0f;
    for (uint k = 0; k < MWS_TOP_K; ++k) {
        float s = float(scores[scores_base + k]);
        float e = float(expert_out[expert_base + size_t(k) * expert_row_stride]);
        acc = fma(s, e, acc);
    }
    out[size_t(n) * expert_row_stride + size_t(d)] = static_cast<T>(acc);
}

#define INST_MWS(dtype_tag, mtl_type)                                  \
    template [[host_name("moe_weighted_sum_" #dtype_tag)]] [[kernel]]  \
    void moe_weighted_sum<mtl_type>(                                   \
        device       mtl_type* out        [[buffer(0)]],               \
        const device mtl_type* expert_out [[buffer(1)]],               \
        const device mtl_type* scores     [[buffer(2)]],               \
        uint3 gid [[thread_position_in_grid]]);

INST_MWS(f16,  half)
INST_MWS(bf16, bfloat)
