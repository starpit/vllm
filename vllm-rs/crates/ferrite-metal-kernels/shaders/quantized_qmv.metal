// SPDX-License-Identifier: Apache-2.0
//
// Faithful port of MLX `affine_qmv_quad` / `affine_qmv_fast` /
// `affine_qmv` decode-matvec kernels from
// `mlx/backend/metal/kernels/quantized.h` (lines 692-975 for the
// `*_impl` helpers, 1444-1597 for the `[[kernel]]` entry points,
// 28-392 for the qdot / load_vector helpers, 1351-1387 for
// `adjust_matrix_offsets`).
//
// Per `INT4_PARITY_PLAN.md` §P3, instantiations cover:
//   bits = 4
//   group_size in {32, 64, 128}
//   dtype in {f16, bf16}
//   qmv_quad: D in {64, 128} × batched in {0, 1}
//   qmv_fast: batched in {0, 1}
//   qmv:      batched in {0, 1}
//
// Symbol naming follows the existing ferrite-metal precedent
// (`affine_dequantize_<dtype>_gs_<gs>_b_<bits>`):
//   affine_qmv_quad_<dtype>_gs_<gs>_b_<bits>_d_<D>_batch_<batched>
//   affine_qmv_fast_<dtype>_gs_<gs>_b_<bits>_batch_<batched>
//   affine_qmv_<dtype>_gs_<gs>_b_<bits>_batch_<batched>
//
// The helper templates (load_vector / qdot / etc.) keep the full
// `bits in {2,3,4,5,6,8}` switch so the body is byte-identical to
// MLX. Only bits=4 is instantiated; other branches dead-code under
// the constexpr template arg.

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

// ThunderMittens — cross-TG sync primitives (flags + atomic u32-packed
// handoff) the wavefront megakernels below compose.
#include "mittens/sync.h"

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

// ─────────────────────────────────────────────────────────────────
// Function constants — baked at pipeline build time by
// `MetalAffineQmv::execute` (and the lower_one path once
// `Instruction::AffineQmm` lands). These hold the K/N dims that MLX
// passes as setBytes runtime args; ferrite specializes per-shape so
// the values are pipeline-constants the Metal compiler can fold.
//
// Indices match `ConstantValue::uint(0, K)` / `ConstantValue::uint(1, N)`
// in the dispatcher; keep them stable, ICB-recorded commands key on
// the constant bag.
// ─────────────────────────────────────────────────────────────────

constant int IN_VEC_SIZE  [[function_constant(0)]];
constant int OUT_VEC_SIZE [[function_constant(1)]];

// ─────────────────────────────────────────────────────────────────
// Pack helpers — quantized.h:17-26
// ─────────────────────────────────────────────────────────────────

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

// ─────────────────────────────────────────────────────────────────
// load_vector / load_vector_safe — quantized.h:28-189
// ─────────────────────────────────────────────────────────────────

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];

      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }

  return sum;
}

// ─────────────────────────────────────────────────────────────────
// qdot / qdot_safe — quantized.h:191-392
// ─────────────────────────────────────────────────────────────────

template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline U qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

// ─────────────────────────────────────────────────────────────────
// elem_to_loc helpers — utils.h:97-125 + steel/utils.h:7-42
// ─────────────────────────────────────────────────────────────────

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc(
    uint elem,
    constant const int* shape,
    constant const int64_t* strides,
    int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

METAL_FUNC ulong3 elem_to_loc_broadcast(
    uint elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    constant const int64_t* c_strides,
    int ndim) {
  ulong loc_a{0};
  ulong loc_b{0};
  ulong loc_c{0};
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    int pos_in_dim = (elem % shape[i]);
    elem /= shape[i];
    loc_a += pos_in_dim * a_strides[i];
    loc_b += pos_in_dim * b_strides[i];
    loc_c += pos_in_dim * c_strides[i];
  }
  return ulong3(loc_a, loc_b, loc_c);
}

// ─────────────────────────────────────────────────────────────────
// adjust_matrix_offsets — quantized.h:1351-1387 (single-array form)
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale>
METAL_FUNC void adjust_matrix_offsets(
    const device T_act*& x,
    const device uint32_t*& w,
    const device T_scale*& scales,
    const device T_scale*& biases,
    device T_act*& y,
    int output_stride,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  uint32_t x_idx = tid.z;
  uint32_t w_idx = tid.z;
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

// ─────────────────────────────────────────────────────────────────
// qmv_quad_impl — quantized.h:692-747
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits, int D>
METAL_FUNC void qmv_quad_impl(
    const device uint32_t* w,
    const device T_scale* scales,
    const device T_scale* biases,
    const device T_act* x,
    device T_act* y,
    // K / N now baked as function constants on the kernel side
    // (IN_VEC_SIZE / OUT_VEC_SIZE) and forwarded by-value here so
    // the impl body matches the MLX C++ source line-for-line.
    int in_vec_size,
    int out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {
  constexpr int quads_per_simd = SIMD_SIZE / QUAD_SIZE;
  constexpr int pack_factor = 32 / bits;
  constexpr int values_per_thread = D / QUAD_SIZE;
  constexpr int packs_per_thread = values_per_thread / pack_factor;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int results_per_quadgroup = 8;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_quadgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * quads_per_simd * results_per_quadgroup + quad_gid;

  w += out_row * in_vec_size_w + quad_lid * packs_per_thread;
  scales += out_row * in_vec_size_g + quad_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + quad_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + quad_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  U sum = load_vector<T_act, U, values_per_thread, bits>(x, x_thread);

  for (int row = 0; row < results_per_quadgroup; row++) {
    auto wl = (const device uint8_t*)(w + row * in_vec_size_w * quads_per_simd);
    const device T_scale* sl = scales + row * in_vec_size_g * quads_per_simd;
    const device T_scale* bl = biases + row * in_vec_size_g * quads_per_simd;

    // T_scale → U=float expands at load (MLX qmv pattern, quantized.h:692-816).
    U s = sl[0];
    U b = bl[0];
    if (row * quads_per_simd + out_row < out_vec_size) {
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
  }

  for (int row = 0; row < results_per_quadgroup; row++) {
    result[row] = quad_sum(result[row]);
    if (quad_lid == 0 && row * quads_per_simd + out_row < out_vec_size) {
      y[row * quads_per_simd] = static_cast<T_act>(result[row]);
    }
  }
}

// ─────────────────────────────────────────────────────────────────
// qmv_fast_impl — quantized.h:749-814
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits>
METAL_FUNC void qmv_fast_impl(
    const device uint32_t* w,
    const device T_scale* scales,
    const device T_scale* biases,
    const device T_act* x,
    device T_act* y,
    int in_vec_size,
    int out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T_act, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T_scale* sl = scales + row * in_vec_size_g;
      const device T_scale* bl = biases + row * in_vec_size_g;

      U s = sl[0];
      U b = bl[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T_act>(result[row]);
    }
  }
}

// ─────────────────────────────────────────────────────────────────
// qmv_impl — quantized.h:816-975
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits>
METAL_FUNC void qmv_impl(
    const device uint32_t* w,
    const device T_scale* scales,
    const device T_scale* biases,
    const device T_act* x,
    device T_act* y,
    int in_vec_size,
    int out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();

  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  // In this case we need to properly guard all our reads because there isn't
  // even 1 tile in the matrix
  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    ws +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T_act, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T_scale* sl = scales + row * in_vec_size_g;
        const device T_scale* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T_act, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T_scale* sl = scales + row * in_vec_size_g;
        const device T_scale* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }

    for (int row = 0;
         row < results_per_simdgroup && out_row + row < out_vec_size;
         row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T_act>(result[row]);
      }
    }
  }

  // In this case the last tile is moved back to redo some output values
  else {
    ws += used_out_row * in_vec_size_w +
        simd_lid * packs_per_thread * bytes_per_pack;
    scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + used_out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T_act, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T_scale* sl = scales + row * in_vec_size_g;
        const device T_scale* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T_act, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T_scale* sl = scales + row * in_vec_size_g;
        const device T_scale* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T_act>(result[row]);
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────
// affine_qmv_quad — quantized.h:1443-1493
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits, int D, bool batched>
[[kernel]] void affine_qmv_quad(
    const device uint32_t* w [[buffer(0)]],
    const device T_scale* scales [[buffer(1)]],
    const device T_scale* biases [[buffer(2)]],
    const device T_act* x [[buffer(3)]],
    device T_act* y [[buffer(4)]],
    // buffer(5) / buffer(6) (in_vec_size / out_vec_size) replaced by
    // file-scope function constants IN_VEC_SIZE / OUT_VEC_SIZE so this
    // kernel is recordable into an MTLIndirectComputeCommand (which
    // exposes setKernelBuffer but not setKernelBytes).
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T_act, T_scale>(
        x,
        w,
        scales,
        biases,
        y,
        OUT_VEC_SIZE * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmv_quad_impl<T_act, T_scale, group_size, bits, D>(
      w,
      scales,
      biases,
      x,
      y,
      IN_VEC_SIZE,
      OUT_VEC_SIZE,
      tid,
      quad_gid,
      quad_lid);
}

// ─────────────────────────────────────────────────────────────────
// affine_qmv_fast — quantized.h:1495-1545
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits, bool batched>
[[kernel]] void affine_qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device T_scale* scales [[buffer(1)]],
    const device T_scale* biases [[buffer(2)]],
    const device T_act* x [[buffer(3)]],
    device T_act* y [[buffer(4)]],
    // buffer(5) / buffer(6): see note on affine_qmv_quad above —
    // K / N now ride as function constants IN_VEC_SIZE / OUT_VEC_SIZE.
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T_act, T_scale>(
        x,
        w,
        scales,
        biases,
        y,
        OUT_VEC_SIZE * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmv_fast_impl<T_act, T_scale, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      IN_VEC_SIZE,
      OUT_VEC_SIZE,
      tid,
      simd_gid,
      simd_lid);
}

// ─────────────────────────────────────────────────────────────────
// affine_qmv — quantized.h:1547-1597
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, const int group_size, const int bits, bool batched>
[[kernel]] void affine_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T_scale* scales [[buffer(1)]],
    const device T_scale* biases [[buffer(2)]],
    const device T_act* x [[buffer(3)]],
    device T_act* y [[buffer(4)]],
    // buffer(5) / buffer(6): see note on affine_qmv_quad above —
    // K / N now ride as function constants IN_VEC_SIZE / OUT_VEC_SIZE.
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T_act, T_scale>(
        x,
        w,
        scales,
        biases,
        y,
        OUT_VEC_SIZE * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmv_impl<T_act, T_scale, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      IN_VEC_SIZE,
      OUT_VEC_SIZE,
      tid,
      simd_gid,
      simd_lid);
}

// ─────────────────────────────────────────────────────────────────
// wavefront_qmv_mega — PD-wavefront Milestone A1 (persistent multi-TG
// tape-loop proof). P co-resident threadgroups (one per GPU core) each
// replay an N-block qmv as a LOOP: worker `tgpos.x` computes output
// row-groups g = tgpos.x, +grid_tg.x, ... — each group being the 8 rows
// one `qmv_fast_impl` call produces. This is the persistent-megakernel
// structure (one TG running many subtile ops in ONE dispatch, NOT one
// TG per op), proven BIT-EXACT vs the whole `affine_qmv_fast`. No
// cross-TG flags yet (the row-groups are disjoint outputs); A2 adds the
// spinloop Wait/Signal + the folded shape-class switch. K/N ride the
// file-scope IN_VEC_SIZE/OUT_VEC_SIZE function constants exactly as
// `affine_qmv_fast`, so the composed `qmv_fast_impl` call is folded.
// ─────────────────────────────────────────────────────────────────
template <typename T_act, typename T_scale, const int group_size, const int bits>
[[kernel]] void wavefront_qmv_mega(
    const device uint32_t* w [[buffer(0)]],
    const device T_scale* scales [[buffer(1)]],
    const device T_scale* biases [[buffer(2)]],
    const device T_act* x [[buffer(3)]],
    device T_act* y [[buffer(4)]],
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint3 grid_tg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  // `qmv_fast_impl` produces results_per_simdgroup(4) × num_simdgroups(2)
  // = 8 output rows per call, selected by tid.y. Cover all ceil(N/8)
  // row-groups (N % 8 == 0 for the fast variant), striped across the P
  // workers — the worker's "tape" of N-block subtiles.
  const uint num_groups = (uint(OUT_VEC_SIZE) + 7u) / 8u;
  for (uint g = tgpos.x; g < num_groups; g += grid_tg.x) {
    qmv_fast_impl<T_act, T_scale, group_size, bits>(
        w, scales, biases, x, y,
        IN_VEC_SIZE, OUT_VEC_SIZE,
        uint3(0u, g, 0u), simd_gid, simd_lid);
  }
}

// ─────────────────────────────────────────────────────────────────
// wavefront_qmv_mega_2stage — PD-wavefront Milestone A2 (cross-TG
// spinloop sync + the folded shape-class switch).
//
// Proves the producer→consumer point-to-point flag with REAL data: a
// 2-stage chain y1 = qmv(W0, x); y2 = qmv(W1, y1). Stage 1 reads the
// WHOLE y1, so every stage-1 block depends on every stage-0 stripe — an
// all-to-all join of P one-shot device-atomic flags. Each worker:
//   stage 0 → its stripe of y1's row-groups (shape class 0)
//   threadgroup_barrier(mem_device) — the data-before-flag FENCE that
//     flushes its y1 writes to device memory — then atomic_store its own
//     flag (this Metal std exposes only memory_order_relaxed for device
//     atomics, so the barrier, not a release order, provides visibility)
//   spin-wait (relaxed atomic_load) on ALL P flags ── y1 fully assembled
//   threadgroup_barrier(mem_device) — fence before reading y1
//   stage 1 → its stripe of y2's row-groups (shape class 1)
// Bit-exact vs two sequential whole `affine_qmv_fast` dispatches.
//
// The folded shape-class switch (`wf_qmv_shape`): one arm per shape
// class, each calling qmv_fast_impl with that class's operands + LITERAL
// (function-constant) K/N so the matvec is constant-folded per shape.
// The on-GPU interpreter (next) will have one arm per distinct decode
// qmv shape; here there are two. Spin is hard-capped so a non-co-resident
// launch fails the bit-exact check instead of hanging the GPU.
// ─────────────────────────────────────────────────────────────────
constant int WF2_K0 [[function_constant(2)]];
constant int WF2_N0 [[function_constant(3)]];
constant int WF2_K1 [[function_constant(4)]];
constant int WF2_N1 [[function_constant(5)]];
// sync primitives (flags + atomic u32-packed handoff) live in mittens/sync.h

// Folded shape-class dispatch. Class 0 = stage-0 qmv (x → y1); class 1 =
// stage-1 qmv (y1_in → y2). `y1_in` is the consumer's PRIVATE unpacked copy
// of the cross-TG handoff (see the kernel), not the shared y1 scratch.
template <typename T_act, typename T_scale, int group_size, int bits>
METAL_FUNC void wf_qmv_shape(
    uint shape_class,
    const device uint32_t* w0,
    const device T_scale* s0,
    const device T_scale* b0,
    const device T_act* x,
    device T_act* y1,
    const device uint32_t* w1,
    const device T_scale* s1,
    const device T_scale* b1,
    const device T_act* y1_in,
    device T_act* y2,
    uint3 qtid,
    uint simd_gid,
    uint simd_lid) {
  switch (shape_class) {
    case 0u:
      qmv_fast_impl<T_act, T_scale, group_size, bits>(
          w0, s0, b0, x, y1, WF2_K0, WF2_N0, qtid, simd_gid, simd_lid);
      break;
    case 1u:
      qmv_fast_impl<T_act, T_scale, group_size, bits>(
          w1, s1, b1, y1_in, y2, WF2_K1, WF2_N1, qtid, simd_gid, simd_lid);
      break;
  }
}

template <typename T_act, typename T_scale, int group_size, int bits>
[[kernel]] void wavefront_qmv_mega_2stage(
    const device uint32_t* w0 [[buffer(0)]],
    const device T_scale* s0 [[buffer(1)]],
    const device T_scale* b0 [[buffer(2)]],
    const device T_act* x [[buffer(3)]],
    device T_act* y1 [[buffer(4)]],
    const device uint32_t* w1 [[buffer(5)]],
    const device T_scale* s1 [[buffer(6)]],
    const device T_scale* b1 [[buffer(7)]],
    device T_act* y2 [[buffer(8)]],
    device atomic_uint* flags [[buffer(9)]], // [P], zeroed by the host
    device atomic_uint* y1c [[buffer(10)]],  // [N0/2] coherent u32-packed handoff
    device T_act* y1r [[buffer(11)]],         // [P*N0] per-worker unpacked scratch
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint3 grid_tg [[threadgroups_per_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint W = grid_tg.x;
  const uint me = tgpos.x;
  device T_act* my = y1r + me * uint(WF2_N0); // this worker's private y1 copy

  // ── Stage 0 (shape 0): this worker writes its disjoint stripe of y1
  //    (bf16). No cross-TG READ of y1 — only this worker reads it below. ──
  const uint groups0 = (uint(WF2_N0) + 7u) / 8u;
  for (uint g = me; g < groups0; g += W) {
    wf_qmv_shape<T_act, T_scale, group_size, bits>(
        0u, w0, s0, b0, x, y1, w1, s1, b1, my, y2, uint3(0u, g, 0u), simd_gid, simd_lid);
  }
  threadgroup_barrier(mem_flags::mem_device); // this worker's y1 stripe visible to its own lanes

  // ── Publish + signal: each group's 8 bf16 pack to 4 u32, ATOMICALLY
  //    stored to the coherent handoff (the cross-TG-visible path — non-atomic
  //    device writes are NOT reliably cross-TG visible; see mittens/sync.h),
  //    then publish this worker's flag. ──
  if (tid_in_tg == 0u) {
    for (uint g = me; g < groups0; g += W) {
      mittens::wf_publish_pairs<T_act>(y1c, y1, g * 4u, 4u); // group g → y1c[g*4 .. +4]
    }
    mittens::wf_signal(flags, me);
  }

  // ── Wait + acquire: join on every worker's publish, then ATOMICALLY read
  //    the handoff into this worker's PRIVATE copy (intra-TG from here).
  //    Co-residency required; the cap turns a missing producer into a wrong
  //    result (test-caught), never a GPU hang. ──
  if (tid_in_tg == 0u) {
    mittens::wf_wait_all(flags, W);
    mittens::wf_acquire_pairs<T_act>(my, y1c, uint(WF2_N0) / 2u);
  }
  threadgroup_barrier(mem_flags::mem_device); // release lanes + `my` visible to all 64

  // ── Stage 1 (shape 1): qmv reads this worker's private y1 copy. ──
  const uint groups1 = (uint(WF2_N1) + 7u) / 8u;
  for (uint g = me; g < groups1; g += W) {
    wf_qmv_shape<T_act, T_scale, group_size, bits>(
        1u, w0, s0, b0, x, y1, w1, s1, b1, my, y2, uint3(0u, g, 0u), simd_gid, simd_lid);
  }
}

// ─────────────────────────────────────────────────────────────────
// Instantiations — bits=4, gs in {32, 64, 128}, dtype in {f16, bf16}
// ─────────────────────────────────────────────────────────────────

#define INST_QMV_BATCHED(name, act_tag, act_type, scale_tag, scale_type, gs, bits, batched) \
  template [[host_name(                                                                     \
      #name "_" #act_tag "_s_" #scale_tag "_gs_" #gs "_b_" #bits "_batch_" #batched)]]      \
  [[kernel]] decltype(name<act_type, scale_type, gs, bits, batched>)                        \
      name<act_type, scale_type, gs, bits, batched>;

#define INST_QMV_QUAD(name, act_tag, act_type, scale_tag, scale_type, gs, bits, D, batched) \
  template [[host_name(                                                                     \
      #name "_" #act_tag "_s_" #scale_tag "_gs_" #gs "_b_" #bits "_d_" #D                   \
      "_batch_" #batched)]]                                                                 \
  [[kernel]] decltype(name<act_type, scale_type, gs, bits, D, batched>)                     \
      name<act_type, scale_type, gs, bits, D, batched>;

#define INST_QMV_ALL(act_tag, act_type, scale_tag, scale_type, gs)                          \
  INST_QMV_BATCHED(affine_qmv_fast, act_tag, act_type, scale_tag, scale_type, gs, 4, 0)     \
  INST_QMV_BATCHED(affine_qmv_fast, act_tag, act_type, scale_tag, scale_type, gs, 4, 1)     \
  INST_QMV_BATCHED(affine_qmv,      act_tag, act_type, scale_tag, scale_type, gs, 4, 0)     \
  INST_QMV_BATCHED(affine_qmv,      act_tag, act_type, scale_tag, scale_type, gs, 4, 1)     \
  INST_QMV_QUAD(affine_qmv_quad,    act_tag, act_type, scale_tag, scale_type, gs, 4, 64, 0) \
  INST_QMV_QUAD(affine_qmv_quad,    act_tag, act_type, scale_tag, scale_type, gs, 4, 64, 1) \
  INST_QMV_QUAD(affine_qmv_quad,    act_tag, act_type, scale_tag, scale_type, gs, 4, 128,0) \
  INST_QMV_QUAD(affine_qmv_quad,    act_tag, act_type, scale_tag, scale_type, gs, 4, 128,1)

// Coverage: T_scale=half always (every sampled mlx-community 4bit ships
// F16 scales — `INT4_PARITY_PROBES.md:73,287`). T_act per `torch_dtype`.
// The `bfloat × bfloat` family that P1-P6 shipped (loader-cast F16→BF16)
// is removed here — that was the regression site `INT4_PARITY_PROBES.md`
// §7 `Decision: in-register cast` repays.
INST_QMV_ALL(f16,  half,   f16, half,    32)
INST_QMV_ALL(f16,  half,   f16, half,    64)
INST_QMV_ALL(f16,  half,   f16, half,   128)
INST_QMV_ALL(bf16, bfloat, f16, half,    32)
// bf16-scale instantiations — Qwen3-MoE (and any `torch_dtype: bfloat16`
// mlx-community 4bit) ships scales/biases as BF16, not F16. Matches
// MLX's `INSTANTIATE_QUANTIZED_FUNCTIONS(T_scale=bfloat16_t)` surface.
INST_QMV_ALL(bf16, bfloat, bf16, bfloat, 32)
INST_QMV_ALL(bf16, bfloat, bf16, bfloat, 64)
INST_QMV_ALL(bf16, bfloat, bf16, bfloat, 128)
INST_QMV_ALL(f16,  half,   bf16, bfloat, 32)
INST_QMV_ALL(f16,  half,   bf16, bfloat, 64)
INST_QMV_ALL(f16,  half,   bf16, bfloat, 128)

// wavefront_qmv_mega instantiations (Milestone A1). 4 template params
// (no `batched`); symbol `wavefront_qmv_mega_<act>_s_<scale>_gs_<gs>_b_4`.
#define INST_WF_QMV_MEGA(act_tag, act_type, scale_tag, scale_type, gs)                  \
  template [[host_name("wavefront_qmv_mega_" #act_tag "_s_" #scale_tag "_gs_" #gs       \
                       "_b_4")]] [[kernel]]                                             \
  decltype(wavefront_qmv_mega<act_type, scale_type, gs, 4>)                             \
      wavefront_qmv_mega<act_type, scale_type, gs, 4>;
INST_WF_QMV_MEGA(f16, half, f16, half, 64)
INST_WF_QMV_MEGA(bf16, bfloat, f16, half, 64)

#define INST_WF_QMV_MEGA_2STAGE(act_tag, act_type, scale_tag, scale_type, gs)            \
  template [[host_name("wavefront_qmv_mega_2stage_" #act_tag "_s_" #scale_tag "_gs_" #gs \
                       "_b_4")]] [[kernel]]                                              \
  decltype(wavefront_qmv_mega_2stage<act_type, scale_type, gs, 4>)                       \
      wavefront_qmv_mega_2stage<act_type, scale_type, gs, 4>;
INST_WF_QMV_MEGA_2STAGE(bf16, bfloat, f16, half, 64)

// ─────────────────────────────────────────────────────────────────
// affine_gather_qmv_{fast,} — quantized.h:1899-2021 (MoE rhs gather)
//
// SwitchGLU per-expert qmv. Each output row picks an expert via
// `rhs_indices[n * top_k + slot_k]`, offsets w/scales/biases into
// the expert's weight slab, and reuses qmv_fast_impl / qmv_impl
// for the actual matvec compute.
//
// Bindings (MLX gather kernels at quantized.h:1900 use 21-buffer
// layout for full broadcast support; we collapse to the SwitchGLU
// shape where x is [N, hidden] and rhs_indices is [N, top_k]):
//   buffer(0) = w           [num_experts, out_vec, in_vec/8]  uint32
//   buffer(1) = scales      [num_experts, out_vec, in_vec/gs] T_scale
//   buffer(2) = biases      [num_experts, out_vec, in_vec/gs] T_scale
//   buffer(3) = x           [N, in_vec]                       T_act
//   buffer(4) = rhs_indices [N, top_k]                        uint32
//   buffer(5) = y           [N, top_k, out_vec]               T_act
//   buffer(6) = top_k       constant int
//
// IN_VEC_SIZE / OUT_VEC_SIZE ride as function constants 0/1 just
// like the non-gather affine_qmv variants — the lowering arm
// reuses the same SpecializedPipelineCache key shape.
//
// Dispatch: tid.x = 0 (we feed the broadcast token via z-axis),
// tid.y = output-block-row index, tid.z = n * top_k + slot_k. The
// non-gather qmv_*_impl reads `tid.x * in_vec_size` from x and
// `tid.x * out_vec_size` from y, so pinning tid.x=0 and pre-
// offsetting both pointers is identical to a single-batch matvec.
//
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits>
[[kernel]] void affine_gather_qmv_fast(
    const device uint32_t* w           [[buffer(0)]],
    const device T_scale*  scales      [[buffer(1)]],
    const device T_scale*  biases      [[buffer(2)]],
    const device T_act*    x           [[buffer(3)]],
    const device uint32_t* rhs_indices [[buffer(4)]],
    device T_act*          y           [[buffer(5)]],
    const constant int&    top_k       [[buffer(6)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]]) {
  // `tid.z` flattens the (token, top_k_slot) axis. tid.x is fixed
  // to 0 — the M-axis broadcast is folded into z.
  uint nk = tid.z;
  uint token_n = nk / uint(top_k);
  uint expert_idx = rhs_indices[nk];

  // Per-expert weight slab strides: w is packed int4 with
  // `in_vec/8 * out_vec` uint32 per expert; scales/biases hold
  // `in_vec/gs * out_vec` per expert.
  size_t expert_stride_w = size_t(IN_VEC_SIZE / 8) * size_t(OUT_VEC_SIZE);
  size_t expert_stride_sb = size_t(IN_VEC_SIZE / group_size) * size_t(OUT_VEC_SIZE);
  const device uint32_t* w_e = w + expert_idx * expert_stride_w;
  const device T_scale*  s_e = scales + expert_idx * expert_stride_sb;
  const device T_scale*  b_e = biases + expert_idx * expert_stride_sb;
  const device T_act*    x_e = x + size_t(token_n) * size_t(IN_VEC_SIZE);
  device T_act*          y_e = y + size_t(nk) * size_t(OUT_VEC_SIZE);

  uint3 inner_tid = uint3(0, tid.y, 0);
  qmv_fast_impl<T_act, T_scale, group_size, bits>(
      w_e, s_e, b_e, x_e, y_e, IN_VEC_SIZE, OUT_VEC_SIZE,
      inner_tid, simd_gid, simd_lid);
}

template <typename T_act, typename T_scale, int group_size, int bits>
[[kernel]] void affine_gather_qmv(
    const device uint32_t* w           [[buffer(0)]],
    const device T_scale*  scales      [[buffer(1)]],
    const device T_scale*  biases      [[buffer(2)]],
    const device T_act*    x           [[buffer(3)]],
    const device uint32_t* rhs_indices [[buffer(4)]],
    device T_act*          y           [[buffer(5)]],
    const constant int&    top_k       [[buffer(6)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]]) {
  uint nk = tid.z;
  uint token_n = nk / uint(top_k);
  uint expert_idx = rhs_indices[nk];

  size_t expert_stride_w = size_t(IN_VEC_SIZE / 8) * size_t(OUT_VEC_SIZE);
  size_t expert_stride_sb = size_t(IN_VEC_SIZE / group_size) * size_t(OUT_VEC_SIZE);
  const device uint32_t* w_e = w + expert_idx * expert_stride_w;
  const device T_scale*  s_e = scales + expert_idx * expert_stride_sb;
  const device T_scale*  b_e = biases + expert_idx * expert_stride_sb;
  const device T_act*    x_e = x + size_t(token_n) * size_t(IN_VEC_SIZE);
  device T_act*          y_e = y + size_t(nk) * size_t(OUT_VEC_SIZE);

  uint3 inner_tid = uint3(0, tid.y, 0);
  qmv_impl<T_act, T_scale, group_size, bits>(
      w_e, s_e, b_e, x_e, y_e, IN_VEC_SIZE, OUT_VEC_SIZE,
      inner_tid, simd_gid, simd_lid);
}

#define INST_GATHER_QMV(name, act_tag, act_type, scale_tag, scale_type, gs, bits)               \
  template [[host_name(                                                                          \
      #name "_" #act_tag "_s_" #scale_tag "_gs_" #gs "_b_" #bits)]]                              \
  [[kernel]] decltype(name<act_type, scale_type, gs, bits>)                                      \
      name<act_type, scale_type, gs, bits>;

#define INST_GATHER_QMV_ALL(act_tag, act_type, scale_tag, scale_type, gs) \
  INST_GATHER_QMV(affine_gather_qmv_fast, act_tag, act_type, scale_tag, scale_type, gs, 4) \
  INST_GATHER_QMV(affine_gather_qmv,      act_tag, act_type, scale_tag, scale_type, gs, 4)

INST_GATHER_QMV_ALL(f16,  half,   f16, half,    32)
INST_GATHER_QMV_ALL(f16,  half,   f16, half,    64)
INST_GATHER_QMV_ALL(f16,  half,   f16, half,   128)
INST_GATHER_QMV_ALL(bf16, bfloat, f16, half,    32)
INST_GATHER_QMV_ALL(bf16, bfloat, f16, half,    64)
INST_GATHER_QMV_ALL(bf16, bfloat, f16, half,   128)
// bf16-scale variants — see the bf16-scale block under INST_QMV_ALL.
INST_GATHER_QMV_ALL(bf16, bfloat, bf16, bfloat, 32)
INST_GATHER_QMV_ALL(bf16, bfloat, bf16, bfloat, 64)
INST_GATHER_QMV_ALL(bf16, bfloat, bf16, bfloat, 128)
INST_GATHER_QMV_ALL(f16,  half,   bf16, bfloat, 32)
INST_GATHER_QMV_ALL(f16,  half,   bf16, bfloat, 64)
INST_GATHER_QMV_ALL(f16,  half,   bf16, bfloat, 128)
INST_QMV_ALL(bf16, bfloat, f16, half,  64)
INST_QMV_ALL(bf16, bfloat, f16, half, 128)
