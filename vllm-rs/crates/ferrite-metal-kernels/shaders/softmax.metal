// SPDX-License-Identifier: Apache-2.0
//
// Faithful port of MLX `softmax_single_row` (`mlx/backend/metal/kernels/
// softmax.h:11`) + the `precise` instantiation pattern
// (`softmax.metal:16-18`).
//
// Used by the MoE router (gate.softmax(axis=-1, precise=True)) per
// `~/git/mlx-lm/mlx_lm/models/qwen3_next.py:335`. num_experts on
// Qwen3-Next is 512, on Qwen3-MoE 60-128, on Mixtral 8 — all comfortably
// inside one threadgroup with N_READS=4 (axis_size ≤ 4*max_threads).
//
// `softmax_looped` (the multi-pass variant in mlx softmax.h:101 used
// when axis_size > 4 * max_threads_per_threadgroup) is OMITTED here:
// none of the MoE configs ferrite-metal targets need it. Add when a
// model with num_experts > 4096 lands.
//
// Symbol naming follows the mlx convention plus ferrite's
// `_specialized` suffix where applicable. For router softmax the axis
// size is model-invariant (num_experts) so a specialized variant via
// function_constant could be added; the unspecialized buffer-arg
// variant below matches mlx 1:1 and is the only path for now.
//
// Precise variant uses `AccT=float` for max + normalizer accumulation.
// MoE routing benefits from the wider exponent range — argpartition
// downstream still sees the same ordering, but the score values used
// for the top-k weighted sum (`y * scores[..., None]` at qwen3_next.py:344)
// need the higher precision.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Matches MLX's SOFTMAX_N_READS in `mlx/backend/metal/kernels/utils.h`.
// 4 reads/thread × up to 256 threads/group = 1024 axis_size per group;
// we cap callers to that range and hand off to a looped variant later
// if/when needed.
constant constexpr int N_READS = 4;

// `local_max` and `local_normalizer` are declared in the outer kernel
// (Metal disallows threadgroup vars in inlined helpers) and threaded
// through here.
template <typename T, typename AccT>
static inline void softmax_single_row_impl(
    const device T* in,
    device T* out,
    constant int& axis_size,
    threadgroup AccT* local_max,
    threadgroup AccT* local_normalizer,
    uint gid,
    uint _lid,
    uint simd_lane_id,
    uint simd_group_id) {
    int lid = int(_lid);

    AccT ld[N_READS];

    in += size_t(gid) * size_t(axis_size) + size_t(lid) * N_READS;
    // -INFINITY as the sentinel for absent lanes — mirrors mlx's
    // `Limits<AccT>::min`, which is `-INFINITY` for float / half / bfloat.
    // `fast::exp(-INFINITY) == 0` so the contribution to the normalizer
    // is zero, and the max-reduce treats it as "not-larger-than" anything.
    const AccT NEG_INF = -INFINITY;
    if (lid * N_READS + N_READS <= axis_size) {
        for (int i = 0; i < N_READS; i++) {
            ld[i] = AccT(in[i]);
        }
    } else {
        for (int i = 0; i < N_READS; i++) {
            ld[i] = ((lid * N_READS + i) < axis_size) ? AccT(in[i]) : NEG_INF;
        }
    }
    if (simd_group_id == 0) {
        local_max[simd_lane_id] = NEG_INF;
        local_normalizer[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce max across the row.
    AccT maxval = NEG_INF;
    for (int i = 0; i < N_READS; i++) {
        maxval = (maxval < ld[i]) ? ld[i] : maxval;
    }
    maxval = simd_max(maxval);
    if (simd_lane_id == 0) {
        local_max[simd_group_id] = maxval;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        maxval = simd_max(local_max[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_max[0] = maxval;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxval = local_max[0];

    // Compute exp(x_i - maxval), accumulate normalizer.
    AccT normalizer = 0;
    for (int i = 0; i < N_READS; i++) {
        // fast::exp matches mlx softmax_exp — softmax inputs are in
        // (-INF, 0] post-shift so the lower-precision exp is fine.
        AccT exp_x = metal::fast::exp(ld[i] - maxval);
        ld[i] = exp_x;
        normalizer += exp_x;
    }
    normalizer = simd_sum(normalizer);
    if (simd_lane_id == 0) {
        local_normalizer[simd_group_id] = normalizer;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        normalizer = simd_sum(local_normalizer[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_normalizer[0] = normalizer;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = AccT(1) / local_normalizer[0];

    // Normalize and write.
    out += size_t(gid) * size_t(axis_size) + size_t(lid) * N_READS;
    if (lid * N_READS + N_READS <= axis_size) {
        for (int i = 0; i < N_READS; i++) {
            out[i] = T(ld[i] * normalizer);
        }
    } else {
        for (int i = 0; i < N_READS; i++) {
            if ((lid * N_READS + i) < axis_size) {
                out[i] = T(ld[i] * normalizer);
            }
        }
    }
}

// ── instantiations ──────────────────────────────────────────────────
//
// Non-precise: AccT = T (uses the input dtype's accumulator).
// Precise:     AccT = float (used by `mx.softmax(..., precise=True)`).

// SIMD_SIZE matches Apple-silicon's fixed 32-wide simdgroup; the
// threadgroup-mem reductions need SIMD_SIZE-wide buffers for the
// per-simdgroup partials.
constant constexpr int SOFTMAX_SIMD_SIZE = 32;

#define SOFTMAX_INSTANTIATE(name, T)                                          \
    [[host_name("block_softmax_" #name)]] kernel void                         \
    block_softmax_##name(                                                     \
        const device T* in [[buffer(0)]],                                     \
        device T* out [[buffer(1)]],                                          \
        constant int& axis_size [[buffer(2)]],                                \
        uint gid [[threadgroup_position_in_grid]],                            \
        uint lid [[thread_position_in_threadgroup]],                          \
        uint simd_lane_id [[thread_index_in_simdgroup]],                      \
        uint simd_group_id [[simdgroup_index_in_threadgroup]]) {              \
        threadgroup T local_max[SOFTMAX_SIMD_SIZE];                           \
        threadgroup T local_normalizer[SOFTMAX_SIMD_SIZE];                    \
        softmax_single_row_impl<T, T>(                                        \
            in, out, axis_size, local_max, local_normalizer,                  \
            gid, lid, simd_lane_id, simd_group_id);                           \
    }

#define SOFTMAX_INSTANTIATE_PRECISE(name, T)                                  \
    [[host_name("block_softmax_precise_" #name)]] kernel void                 \
    block_softmax_precise_##name(                                             \
        const device T* in [[buffer(0)]],                                     \
        device T* out [[buffer(1)]],                                          \
        constant int& axis_size [[buffer(2)]],                                \
        uint gid [[threadgroup_position_in_grid]],                            \
        uint lid [[thread_position_in_threadgroup]],                          \
        uint simd_lane_id [[thread_index_in_simdgroup]],                      \
        uint simd_group_id [[simdgroup_index_in_threadgroup]]) {              \
        threadgroup float local_max[SOFTMAX_SIMD_SIZE];                       \
        threadgroup float local_normalizer[SOFTMAX_SIMD_SIZE];                \
        softmax_single_row_impl<T, float>(                                    \
            in, out, axis_size, local_max, local_normalizer,                  \
            gid, lid, simd_lane_id, simd_group_id);                           \
    }

// Non-precise variants exist for f32 and f16 only — Metal's
// `simd_max<bfloat>` / `simd_sum<bfloat>` are not provided on the
// system toolchain (mlx works around this with a custom `bfloat16_t`
// wrapper that ships its own simdgroup overloads). For bf16 we always
// use the precise (float-AccT) path, which matches the MoE router's
// `precise=True` requirement at qwen3_next.py:335.
SOFTMAX_INSTANTIATE(float32, float)
SOFTMAX_INSTANTIATE(float16, half)
SOFTMAX_INSTANTIATE_PRECISE(float16, half)
SOFTMAX_INSTANTIATE_PRECISE(bfloat16, bfloat)
