// SPDX-License-Identifier: Apache-2.0
//
// Specialized 2D row-gather used by the MoE `_gather_sort` /
// `_scatter_unsort` helpers (mlx-lm/mlx_lm/models/switch_layers.py:12).
//
//   out[m, d] = src[idx[m] / divisor, d]
//
// `divisor == 1` is the plain row gather used by:
//   * `indices_sorted = indices_flat[order]` (D = 1)
//   * `_scatter_unsort: out = x[inv_order]`
//
// `divisor == top_k` implements `x_sorted = x.flatten(0,-3)[order // K]`
// (switch_layers.py:17), letting us collapse the two-op expression
// (`mx.divide` + `mx.gather`) into a single kernel.
//
// mlx implements this via the general `gather_impl`
// (`mlx/backend/metal/kernels/indexing/gather.h:7`) which carries full
// ndim shape/stride bookkeeping. The cases ferrite-metal needs are 2D
// contiguous source, 1D contiguous indices, 2D contiguous output, so
// we inline the specialized path here.

#include <metal_stdlib>
using namespace metal;

// ─── kernel ─────────────────────────────────────────────────────────
//
// Bindings:
//   buffer(0) = src          T          [src_rows, D]
//   buffer(1) = idx          uint32     [M]
//   buffer(2) = out          T          [M, D]
//   buffer(3) = D                       int32
//   buffer(4) = idx_divisor             int32  (1 for plain gather,
//                                                top_k for sort-order
//                                                gather)
//
// Grid: (D, M, 1) threads, one thread per output element. D is rounded
// up to a tg-friendly multiple of 32 by the caller; threads with
// d >= D early-out.

template <typename T>
inline void row_gather_impl(
    const device T* src,
    const device uint* idx,
    device T* out,
    int D,
    int idx_divisor,
    uint3 gid) {
    int d = int(gid.x);
    int m = int(gid.y);
    if (d >= D) {
        return;
    }
    int src_row = int(idx[m]) / idx_divisor;
    out[size_t(m) * size_t(D) + size_t(d)] =
        src[size_t(src_row) * size_t(D) + size_t(d)];
}

#define ROW_GATHER_INSTANTIATE(name, T)                                     \
    [[host_name("row_gather_" #name)]]                                      \
    kernel void row_gather_##name(                                          \
        const device T* src [[buffer(0)]],                                  \
        const device uint* idx [[buffer(1)]],                               \
        device T* out [[buffer(2)]],                                        \
        constant int& D [[buffer(3)]],                                      \
        constant int& idx_divisor [[buffer(4)]],                            \
        uint3 gid [[thread_position_in_grid]]) {                            \
        row_gather_impl<T>(src, idx, out, D, idx_divisor, gid);             \
    }

ROW_GATHER_INSTANTIATE(float32, float)
ROW_GATHER_INSTANTIATE(float16, half)
ROW_GATHER_INSTANTIATE(bfloat16, bfloat)
ROW_GATHER_INSTANTIATE(uint32, uint)
