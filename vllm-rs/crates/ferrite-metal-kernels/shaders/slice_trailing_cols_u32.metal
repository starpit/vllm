// SPDX-License-Identifier: Apache-2.0
//
// Slice trailing K columns of a 2D contiguous uint32 buffer:
//
//   out[n, k] = src[n, src_cols - dst_cols + k]
//
// Used by the I::SharedFusedMoe lowering arm to take the trailing top_k
// columns of `argpartition`'s sorted output (full `[N, num_experts]`)
// into a `[N, top_k]` indices buffer that take_along_axis consumes
// (`mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]`, qwen3_moe.py:131).
//
// mlx implements this slice via a strided view (no kernel) at the
// graph level; ferrite-metal materializes it because our scratch
// regions are flat byte ranges, not array views.

#include <metal_stdlib>
using namespace metal;

// ─── kernel ─────────────────────────────────────────────────────────
//
// Bindings:
//   buffer(0) = src         uint32  [N, src_cols]
//   buffer(1) = dst         uint32  [N, dst_cols]
//   buffer(2) = src_cols            int32
//   buffer(3) = dst_cols            int32
//
// Grid: (dst_cols, N, 1). One thread per output element.

[[host_name("slice_trailing_cols_uint32")]]
kernel void slice_trailing_cols_uint32(
    const device uint* src     [[buffer(0)]],
    device uint*       dst     [[buffer(1)]],
    constant int&      src_cols [[buffer(2)]],
    constant int&      dst_cols [[buffer(3)]],
    uint2              gid     [[thread_position_in_grid]]) {
    int k = int(gid.x);
    int n = int(gid.y);
    if (k >= dst_cols) {
        return;
    }
    int src_col = src_cols - dst_cols + k;
    dst[size_t(n) * size_t(dst_cols) + size_t(k)] =
        src[size_t(n) * size_t(src_cols) + size_t(src_col)];
}
