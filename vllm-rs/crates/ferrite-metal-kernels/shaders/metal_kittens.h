// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0
//
// MetalKittens — composable MSL primitive library for Apple Silicon.
//
// Core primitive: mk_qmv_fast
//   Cooperative int4 GEMV: 2 simdgroups × 32 threads compute
//   MK_ROWS_PER_SIMDGROUP=4 consecutive output elements each.
//   Matches MLX's qmv_fast_impl (mlx quantized.h:749-814) exactly.
//
// Fused kernel dispatch shape (per-head):
//   Grid: (M, NUM_HEADS_TOTAL, 1)
//   Threads per threadgroup: MK_SIMD_SIZE * (HEAD_DIM / MK_ROWS_PER_SIMDGROUP)
//     = 32 * 32 = 1024 for HEAD_DIM=128
//
// Each threadgroup handles all HEAD_DIM output elements for one (token, head).
// After mk_qmv_fast, lane-0s write 4 results to threadgroup memory; mk_sync()
// makes all HEAD_DIM elements visible for the epilogue (RoPE, cache write).

#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

static constant constexpr int MK_SIMD_SIZE          = 32;
static constant constexpr int MK_ROWS_PER_SIMDGROUP = 4;
static constant constexpr int MK_PACKS_PER_THREAD   = 2;  // bits==4 fast path

// ─────────────────────────────────────────────────────────────────────────────
// Pack helpers — quantized.h:1-26
// ─────────────────────────────────────────────────────────────────────────────

template <int bits, int wsize = 8>
inline constexpr short mk_get_pack_factor() {
    return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short mk_get_bytes_per_pack() {
    constexpr int pow2 = (bits & (bits - 1)) == 0;
    return pow2 ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_load_vector — quantized.h:28-189 (bits==4 path)
// ─────────────────────────────────────────────────────────────────────────────

template <typename T, typename U, int values_per_thread, int bits>
inline U mk_load_vector(const device T* x, thread U* x_thread) {
    U sum = 0;
    if (bits == 4) {
        for (int i = 0; i < values_per_thread; i += 4) {
            sum += x[i] + x[i+1] + x[i+2] + x[i+3];
            x_thread[i]   = x[i];
            x_thread[i+1] = x[i+1] / 16.0f;
            x_thread[i+2] = x[i+2] / 256.0f;
            x_thread[i+3] = x[i+3] / 4096.0f;
        }
    } else if (bits == 8) {
        for (int i = 0; i < values_per_thread; i++) {
            sum += x[i];
            x_thread[i] = x[i];
        }
    }
    return sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_qdot — quantized.h:191-392 (bits==4 path)
// ─────────────────────────────────────────────────────────────────────────────

template <typename U, int values_per_thread, int bits>
inline U mk_qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale, U bias, U sum)
{
    U accum = 0;
    if (bits == 4) {
        const device uint16_t* ws = (const device uint16_t*)w;
        for (int i = 0; i < (values_per_thread / 4); i++) {
            accum += (x_thread[4*i]   * (ws[i] & 0x000f) +
                      x_thread[4*i+1] * (ws[i] & 0x00f0) +
                      x_thread[4*i+2] * (ws[i] & 0x0f00) +
                      x_thread[4*i+3] * (ws[i] & 0xf000));
        }
    } else if (bits == 8) {
        for (int i = 0; i < values_per_thread; i++)
            accum += x_thread[i] * w[i];
    }
    return scale * accum + sum * bias;
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_qmv_fast
//
// Core of qmv_fast_impl (quantized.h:749-814). Computes MK_ROWS_PER_SIMDGROUP
// consecutive output elements cooperatively across MK_SIMD_SIZE threads.
//
// Precondition — caller pre-adjusts pointers:
//   ws     += simdgroup_out_row * in_vec_size_w
//              + simd_lid * MK_PACKS_PER_THREAD * bytes_per_pack
//   scales += simdgroup_out_row * in_vec_size_g
//              + simd_lid / scale_step
//   biases += same as scales
//   x      += token * in_vec_size + simd_lid * values_per_thread
//
// where:
//   in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor
//   in_vec_size_g = in_vec_size / group_size
//   scale_step    = group_size / values_per_thread
//
// On return: result[0..MK_ROWS_PER_SIMDGROUP-1] holds the dot products.
// Valid on ALL simd lanes (simd_sum broadcasts). Only simd_lid==0 should
// write to device or threadgroup memory to avoid races.
// ─────────────────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits>
inline void mk_qmv_fast(
    const device uint8_t* ws,
    const device T_scale* scales,
    const device T_scale* biases,
    const device T_act*   x,
    int                   in_vec_size,
    thread float          result[MK_ROWS_PER_SIMDGROUP],
    uint                  simd_lid)
{
    constexpr int pack_factor       = mk_get_pack_factor<bits, 32>();
    constexpr int bytes_per_pack    = mk_get_bytes_per_pack<bits, 32>();
    constexpr int values_per_thread = pack_factor * MK_PACKS_PER_THREAD;
    constexpr int block_size        = values_per_thread * MK_SIMD_SIZE;

    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / group_size;

    thread float x_thread[values_per_thread];
    for (int r = 0; r < MK_ROWS_PER_SIMDGROUP; r++) result[r] = 0.0f;

    for (int k = 0; k < in_vec_size; k += block_size) {
        float sum = mk_load_vector<T_act, float, values_per_thread, bits>(x, x_thread);

        for (int row = 0; row < MK_ROWS_PER_SIMDGROUP; row++) {
            const device uint8_t* wl = ws + row * in_vec_size_w;
            float s = float(scales[row * in_vec_size_g]);
            float b = float(biases[row * in_vec_size_g]);
            result[row] += mk_qdot<float, values_per_thread, bits>(
                wl, x_thread, s, b, sum);
        }

        ws     += block_size * bytes_per_pack / pack_factor;
        scales += block_size / group_size;
        biases += block_size / group_size;
        x      += block_size;
    }

    for (int row = 0; row < MK_ROWS_PER_SIMDGROUP; row++)
        result[row] = simd_sum(result[row]);
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_sync — threadgroup memory barrier
// ─────────────────────────────────────────────────────────────────────────────

inline void mk_sync() {
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_qmv_fast_to_smem
//
// Calls mk_qmv_fast and writes the 4 results into threadgroup memory at
// smem[smem_base + simd_gid * MK_ROWS_PER_SIMDGROUP + 0..3] (lane-0 only).
//
// After all simdgroups in the threadgroup call this, do mk_sync() so that
// smem[0..HEAD_DIM-1] is fully populated and visible to all threads.
// ─────────────────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits>
inline void mk_qmv_fast_to_smem(
    const device uint8_t* ws,
    const device T_scale* scales,
    const device T_scale* biases,
    const device T_act*   x,
    int                   in_vec_size,
    threadgroup float*    smem,
    uint                  simd_gid,
    uint                  simd_lid)
{
    thread float result[MK_ROWS_PER_SIMDGROUP];
    mk_qmv_fast<T_act, T_scale, group_size, bits>(
        ws, scales, biases, x, in_vec_size, result, simd_lid);

    if (simd_lid == 0) {
        const uint base = simd_gid * MK_ROWS_PER_SIMDGROUP;
        for (int r = 0; r < MK_ROWS_PER_SIMDGROUP; r++)
            smem[base + r] = result[r];
    }
}
