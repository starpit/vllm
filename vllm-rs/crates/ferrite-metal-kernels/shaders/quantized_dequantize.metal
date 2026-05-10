// SPDX-License-Identifier: Apache-2.0
//
// Faithful port of MLX `affine_dequantize` kernel
// (mlx/backend/metal/kernels/quantized.h:2536). Bits = 4 only, scales /
// biases stored in F16 per `INT4_PARITY_PROBES.md` §1, dispatched in
// dtype × group_size combinations the int4 parity mandate exercises.
//
// For each output pair (one byte of packed `w` covers two nibbles, i.e.
// `pack_factor = 8 / bits = 2`), the kernel reads
//   - one byte of packed `w`
//   - one scale + one bias per `group_size` output elements
// and writes
//   out[oindex + 0] = scale * (byte        & 0x0f) + bias
//   out[oindex + 1] = scale * ((byte >> 4) & 0x0f) + bias
//
// The 2D grid matches MLX (`offset = x + grid_dim.x * y`) so callers
// can use the same `get_2d_grid_dims` pattern for large tensors;
// 1D dispatches still work — set `grid_dim.y == 1`, `index.y == 0`.

#include <metal_stdlib>
using namespace metal;

template <typename T, const int group_size>
inline void affine_dequantize_b4_kernel(
    const device uint8_t* w,
    const device T* scales,
    const device T* biases,
    device T* out,
    uint2 index,
    uint2 grid_dim) {
    // pack_factor = 8 / bits = 2 for bits=4.
    constexpr int pack_factor = 2;

    size_t offset = index.x + grid_dim.x * size_t(index.y);
    size_t oindex = offset * pack_factor;
    size_t gindex = oindex / group_size;

    T scale = scales[gindex];
    T bias = biases[gindex];

    uint val = w[offset];
    out[oindex + 0] = scale * T(val & 0x0f)        + bias;
    out[oindex + 1] = scale * T((val >> 4) & 0x0f) + bias;
}

#define DEFINE_AFFINE_DEQUANTIZE_B4(dtype, mtl_type, gs)                       \
    kernel void affine_dequantize_##dtype##_gs_##gs##_b_4(                     \
        const device uint8_t* w        [[buffer(0)]],                          \
        const device mtl_type* scales  [[buffer(1)]],                          \
        const device mtl_type* biases  [[buffer(2)]],                          \
        device mtl_type* out           [[buffer(3)]],                          \
        uint2 index    [[thread_position_in_grid]],                            \
        uint2 grid_dim [[threads_per_grid]]) {                                 \
        affine_dequantize_b4_kernel<mtl_type, gs>(                             \
            w, scales, biases, out, index, grid_dim);                          \
    }

DEFINE_AFFINE_DEQUANTIZE_B4(f16,  half,    32)
DEFINE_AFFINE_DEQUANTIZE_B4(f16,  half,    64)
DEFINE_AFFINE_DEQUANTIZE_B4(f16,  half,   128)
DEFINE_AFFINE_DEQUANTIZE_B4(bf16, bfloat,  32)
DEFINE_AFFINE_DEQUANTIZE_B4(bf16, bfloat,  64)
DEFINE_AFFINE_DEQUANTIZE_B4(bf16, bfloat, 128)
