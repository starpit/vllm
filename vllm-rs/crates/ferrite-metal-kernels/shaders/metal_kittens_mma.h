// SPDX-License-Identifier: Apache-2.0
// Phase 5 prefill tile primitives — simdgroup_matrix 8x8 operations.
// Requires #include <metal_simdgroup_matrix> before this header.
#pragma once
#include <metal_simdgroup_matrix>

// ─────────────────────────────────────────────────────────────────────────────
// Phase 5 prefill primitives — simdgroup_matrix 8x8 tile operations
//
// mk_load_tile   — 8x8 tile load from device or threadgroup memory
// mk_store_tile  — 8x8 tile store to device memory
// mk_mma         — c += a * b
// mk_qload_tile  — int4 dequantize + barrier + simdgroup_load
//   scratch = caller-provided threadgroup T_act[8 * 9] (stride 9 avoids
//   bank conflicts). Barrier is issued inside mk_qload_tile.
// ─────────────────────────────────────────────────────────────────────────────

template <typename T>
inline void mk_load_tile(
    thread simdgroup_matrix<T, 8, 8>& mat,
    const device T* src,
    ulong stride)
{
    simdgroup_load(mat, src, stride);
}

template <typename T>
inline void mk_load_tile_tg(
    thread simdgroup_matrix<T, 8, 8>& mat,
    const threadgroup T* src,
    ulong stride)
{
    simdgroup_load(mat, src, stride);
}

template <typename T>
inline void mk_store_tile(
    device T* dst,
    thread simdgroup_matrix<T, 8, 8>& mat,
    ulong stride)
{
    simdgroup_store(mat, dst, stride);
}

// mk_mma — c += a * b
template <typename T>
inline void mk_mma(
    thread simdgroup_matrix<T, 8, 8>& c,
    const thread simdgroup_matrix<T, 8, 8>& a,
    const thread simdgroup_matrix<T, 8, 8>& b)
{
    simdgroup_multiply_accumulate(c, a, b, c);
}

// mk_qload_tile — dequantize int4 (8 x 8) weight tile into simdgroup_matrix.
template <int group_size, typename T_act, typename T_scale>
inline void mk_qload_tile(
    thread simdgroup_matrix<T_act, 8, 8>& dst,
    threadgroup T_act*    scratch,
    const device uint8_t* w,
    const device T_scale* scales,
    const device T_scale* biases,
    uint row_base,
    uint k_base,
    uint in_features,
    uint tid,
    uint threads_per_tg,
    bool transposed)
{
    const uint bpr = in_features >> 1u;
    const uint gpr = in_features / (uint)group_size;

    for (uint i = tid; i < 64u; i += threads_per_tg) {
        const uint r  = i >> 3u;
        const uint c  = i & 7u;
        const uint gr = row_base + r;
        const uint gc = k_base  + c;
        const uint bi = gr * bpr + (gc >> 1u);
        const uint nb = (gc & 1u) ? (w[bi] >> 4u) : (w[bi] & 0xFu);
        const uint gi = gr * gpr + gc / (uint)group_size;
        scratch[r * 9u + c] = T_act(float(nb) * float(scales[gi]) + float(biases[gi]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_load(dst, scratch, 9u, ulong2(0, 0), transposed);
}

// mk_qload_tile (non-transposed convenience overload)
template <int group_size, typename T_act, typename T_scale>
inline void mk_qload_tile(
    thread simdgroup_matrix<T_act, 8, 8>& dst,
    threadgroup T_act*    scratch,
    const device uint8_t* w,
    const device T_scale* scales,
    const device T_scale* biases,
    uint row_base,
    uint k_base,
    uint in_features,
    uint tid,
    uint threads_per_tg)
{
    mk_qload_tile<group_size>(dst, scratch, w, scales, biases,
                              row_base, k_base, in_features,
                              tid, threads_per_tg, false);
}
