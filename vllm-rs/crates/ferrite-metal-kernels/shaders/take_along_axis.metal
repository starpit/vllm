// SPDX-License-Identifier: Apache-2.0
//
// Faithful port of mlx `gather_axis`
// (`mlx/backend/metal/kernels/indexing/gather_axis.h:6`). Used by the
// MoE router to fetch the top-k expert scores from the per-row softmax
// gates:
//
// ```python
// scores = mx.take_along_axis(gates, inds, axis=-1)
// # qwen3_next.py:342 / qwen3_moe.py:135
// ```
//
// mlx instantiates four contiguity variants (SrcC × IdxC × {true,
// false}^2); we only need the both-contiguous case because the
// upstream `argpartition` writes contiguous `[batch, top_k]` indices
// into a fresh output buffer, and the gates come straight out of a
// contiguous softmax. The fully-general non-contiguous arm is OMITTED
// — add when a non-contiguous router input emerges.
//
// LocT is fixed at `int32_t` (mlx's "small index" path; mlx switches
// to `int64_t` for total sizes > INT32_MAX). MoE router sizes are
// `batch × num_experts ≤ 32 × 512 = 16384`, comfortably inside int32.

#include <metal_stdlib>
using namespace metal;

template <typename T, typename IdxT, typename LocT>
inline void gather_axis_cc_impl(
    const device T* src,
    const device IdxT* indices,
    device T* out,
    const constant int& axis_size,
    const constant int& src_ax_stride,
    const constant int& idx_ax_stride,
    uint3 index,
    uint3 grid_dim) {
    // Direct transcription of mlx `gather_axis<SrcC=true, IdxC=true>`
    // with ndim-1 == 1 (our 2D-axis=-1 reduction): size_pre.size = 1
    // (`z` axis), size_post = 1 (`x` axis collapsed). See
    // `mlx/backend/metal/indexing.cpp:GatherAxis::eval_gpu:489-501`.
    LocT elem_idx = LocT(index.z) * LocT(grid_dim.x);
    LocT out_idx = elem_idx * LocT(grid_dim.y) + LocT(index.x);

    LocT idx_loc = LocT(index.y) * LocT(idx_ax_stride) + out_idx;
    auto idx_val = indices[idx_loc];

    LocT src_idx = LocT(idx_val) * LocT(src_ax_stride)
                 + elem_idx * LocT(axis_size) + LocT(index.x);

    out_idx += LocT(index.y) * LocT(grid_dim.x);
    out[out_idx] = src[src_idx];
}

// ─── kernel: src_contig × idx_contig × dtype × idx_dtype ────────────
//
// IdxT is fixed at `uint` because our `argpartition.metal` emits
// `uint32` indices. mlx's general kernel accepts signed types and
// flips negative values via `axis_size`; the uint path skips that
// branch since router indices are always non-negative.

#define GATHER_AXIS_CC(name, T)                                              \
    [[host_name("gather_axis_cc_" #name "_uint32")]]                         \
    kernel void gather_axis_cc_##name##_uint32(                              \
        const device T* src [[buffer(0)]],                                   \
        const device uint* indices [[buffer(1)]],                            \
        device T* out [[buffer(2)]],                                         \
        constant int& axis_size [[buffer(8)]],                               \
        constant int& src_ax_stride [[buffer(9)]],                           \
        constant int& idx_ax_stride [[buffer(10)]],                          \
        uint3 index [[thread_position_in_grid]],                             \
        uint3 grid_dim [[threads_per_grid]]) {                               \
        gather_axis_cc_impl<T, uint, int>(                                   \
            src, indices, out,                                               \
            axis_size, src_ax_stride, idx_ax_stride,                         \
            index, grid_dim);                                                \
    }

GATHER_AXIS_CC(float32, float)
GATHER_AXIS_CC(float16, half)
GATHER_AXIS_CC(bfloat16, bfloat)
