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

// ============================================================================
// affine_embed_b4_kernel — gather + dequantize in one pass.
//
// Faithful port of MLX's `nn.QuantizedEmbedding.__call__`
// (`python/mlx/nn/layers/quantized.py:144`):
//   x  = self.weight[indices]      # gather packed u32 rows by token id
//   s  = self.scales[indices]      # gather scales rows
//   b  = self.biases[indices]      # gather biases rows
//   y  = mx.dequantize(x, s, b, group_size, bits, "affine")
// MLX dispatches the four ops as four kernels; ferrite-metal fuses them
// here because the gather-row indirection is a constant per output row,
// not a cross-op fusion of independent kernels.
//
// 2D grid:
//   index.x = byte-offset within a token row  ∈ [0, hidden_size / 2)
//   index.y = output token row                ∈ [0, num_tokens)
// Each thread reads one packed byte (= 2 nibbles for bits=4), fetches
// the corresponding scale + bias for that group, and writes 2 output
// elements. Total threads = num_tokens × (hidden_size / 2).
//
// Bindings:
//   buffer(0) w        : [vocab_size, hidden_size / 2] u8 (packed nibbles)
//   buffer(1) scales   : [vocab_size, hidden_size / group_size] T
//   buffer(2) biases   : [vocab_size, hidden_size / group_size] T
//   buffer(3) indices  : [num_tokens] uint32_t
//   buffer(4) out      : [num_tokens, hidden_size] T
//   function_constant(0) AFFINE_EMBED_HIDDEN_SIZE : uint = hidden_size
//
// `hidden_size` rides as a function constant rather than a kernel arg
// so the bucket-specialized pipeline bakes it in (mirroring
// `embed_bf16_specialized` in `embed.metal`).
//
// Note: bits=4 only — the int4 parity mandate doesn't exercise other
// widths from a quantized embedding in any sampled mlx-community
// checkpoint (`INT4_PARITY_PROBES.md` §2). Add other widths if a model
// surfaces.

constant uint AFFINE_EMBED_HIDDEN_SIZE [[function_constant(0)]];

template <typename T, const int group_size>
inline void affine_embed_b4_kernel(
    const device uint8_t* w,
    const device T* scales,
    const device T* biases,
    const device uint* indices,
    device T* out,
    uint hidden_size,
    uint2 index) {
    constexpr int pack_factor = 2;
    // The dispatch rounds threadgroups.x up by threads_per_threadgroup.x;
    // hidden_size/2 may not align (3B's hidden=3072 → K/2=1536 > 1024
    // tpg cap forces 2 threadgroups, last one with partial). Without
    // this check, threads past the row boundary chase past-end bytes
    // of `w` and corrupt out[].
    if (index.x * pack_factor >= hidden_size) return;
    uint vocab_idx = indices[index.y];
    size_t bytes_per_row  = size_t(hidden_size) / pack_factor;
    size_t groups_per_row = size_t(hidden_size) / group_size;

    size_t w_offset    = size_t(vocab_idx) * bytes_per_row  + size_t(index.x);
    size_t out_col     = size_t(index.x) * pack_factor;
    size_t gindex      = size_t(vocab_idx) * groups_per_row + (out_col / group_size);
    size_t out_offset  = size_t(index.y) * size_t(hidden_size) + out_col;

    T scale = scales[gindex];
    T bias  = biases[gindex];
    uint val = w[w_offset];

    out[out_offset + 0] = scale * T(val & 0x0f)        + bias;
    out[out_offset + 1] = scale * T((val >> 4) & 0x0f) + bias;
}

#define DEFINE_AFFINE_EMBED_B4(dtype, mtl_type, gs)                            \
    kernel void affine_embed_##dtype##_gs_##gs##_b_4(                          \
        const device uint8_t* w        [[buffer(0)]],                          \
        const device mtl_type* scales  [[buffer(1)]],                          \
        const device mtl_type* biases  [[buffer(2)]],                          \
        const device uint*    indices  [[buffer(3)]],                          \
        device mtl_type* out           [[buffer(4)]],                          \
        uint2 index    [[thread_position_in_grid]]) {                          \
        affine_embed_b4_kernel<mtl_type, gs>(                                  \
            w, scales, biases, indices, out, AFFINE_EMBED_HIDDEN_SIZE, index); \
    }

DEFINE_AFFINE_EMBED_B4(f16,  half,    32)
DEFINE_AFFINE_EMBED_B4(f16,  half,    64)
DEFINE_AFFINE_EMBED_B4(f16,  half,   128)
DEFINE_AFFINE_EMBED_B4(bf16, bfloat,  32)
DEFINE_AFFINE_EMBED_B4(bf16, bfloat,  64)
DEFINE_AFFINE_EMBED_B4(bf16, bfloat, 128)
