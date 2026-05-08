// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

/// Fused Gate-Up-SiLU-Mul kernel for SwiGLU activation
///
/// Pattern: silu(gate_proj(x)) * up_proj(x)
/// Where: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
///
/// This fusion eliminates memory round-trips by computing the activation
/// in a single pass. Critical for memory-bound MLP layers on Apple Silicon.
///
/// Two variants:
/// 1. Separate GEMMs: Takes gate_out and up_out as inputs
/// 2. Fused GEMM: Takes concatenated gate_up output [B, 2*I] as input
///
/// Grid: (M, 1, 1) where M = batch_size
/// Threadgroup: (min(N, 1024), 1, 1) where N = intermediate_size

/// SiLU activation: x * sigmoid(x)
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

/// Variant 1: Separate gate and up outputs
/// Input: gate_out [M, N], up_out [M, N]
/// Output: silu(gate_out) * up_out [M, N]
kernel void fused_gate_up_silu_mul_f16(
    device const half* gate_out [[buffer(0)]],
    device const half* up_out [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N; i += tg_size) {
        float gate = float(gate_out[gid * N + i]);
        float up = float(up_out[gid * N + i]);
        output[gid * N + i] = half(silu(gate) * up);
    }
}

/// Variant 2: Fused gate_up output (concatenated in column dimension)
/// Input: gate_up [M, 2*N] where first N columns are gate, next N are up
/// Output: silu(gate) * up [M, N]
kernel void fused_gate_up_silu_mul_concat_f16(
    device const half* gate_up [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],  // intermediate_size (not 2*N)
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N; i += tg_size) {
        float gate = float(gate_up[gid * (2 * N) + i]);
        float up = float(gate_up[gid * (2 * N) + N + i]);
        output[gid * N + i] = half(silu(gate) * up);
    }
}

/// BF16 variant - separate outputs
kernel void fused_gate_up_silu_mul_bf16(
    device const float* gate_out [[buffer(0)]],
    device const float* up_out [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N; i += tg_size) {
        float gate = gate_out[gid * N + i];
        float up = up_out[gid * N + i];
        output[gid * N + i] = silu(gate) * up;
    }
}

/// BF16 variant - concatenated input
kernel void fused_gate_up_silu_mul_concat_bf16(
    device const float* gate_up [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N; i += tg_size) {
        float gate = gate_up[gid * (2 * N) + i];
        float up = gate_up[gid * (2 * N) + N + i];
        output[gid * N + i] = silu(gate) * up;
    }
}

/// Vectorized variant (half4) for better memory bandwidth
/// Requires N to be multiple of 4
kernel void fused_gate_up_silu_mul_f16_vec4(
    device const half4* gate_out [[buffer(0)]],
    device const half4* up_out [[buffer(1)]],
    device half4* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N_div4 [[buffer(4)]],  // N / 4
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N_div4; i += tg_size) {
        half4 gate = gate_out[gid * N_div4 + i];
        half4 up = up_out[gid * N_div4 + i];
        
        // Apply SiLU element-wise
        float4 gate_f = float4(gate);
        float4 up_f = float4(up);
        float4 result;
        result.x = silu(gate_f.x) * up_f.x;
        result.y = silu(gate_f.y) * up_f.y;
        result.z = silu(gate_f.z) * up_f.z;
        result.w = silu(gate_f.w) * up_f.w;
        
        output[gid * N_div4 + i] = half4(result);
    }
}

/// Vectorized concatenated variant
kernel void fused_gate_up_silu_mul_concat_f16_vec4(
    device const half4* gate_up [[buffer(0)]],
    device half4* output [[buffer(1)]],
    constant uint& M [[buffer(2)]],
    constant uint& N_div4 [[buffer(3)]],  // N / 4
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N_div4; i += tg_size) {
        half4 gate = gate_up[gid * (2 * N_div4) + i];
        half4 up = gate_up[gid * (2 * N_div4) + N_div4 + i];
        
        float4 gate_f = float4(gate);
        float4 up_f = float4(up);
        float4 result;
        result.x = silu(gate_f.x) * up_f.x;
        result.y = silu(gate_f.y) * up_f.y;
        result.z = silu(gate_f.z) * up_f.z;
        result.w = silu(gate_f.w) * up_f.w;
        
        output[gid * N_div4 + i] = half4(result);
    }
}

/// GELU variant for Gemma2/3 models
/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
inline float gelu_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanh(inner));
}

/// Fused Gate-Up-GELU-Mul for Gemma models
kernel void fused_gate_up_gelu_mul_f16(
    device const half* gate_out [[buffer(0)]],
    device const half* up_out [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N; i += tg_size) {
        float gate = float(gate_out[gid * N + i]);
        float up = float(up_out[gid * N + i]);
        output[gid * N + i] = half(gelu_approx(gate) * up);
    }
}

// Note: Metal does not have erf() function, so exact GELU is not available
// Use approximate GELU instead (gelu_approx above)

// ============================================================================
// fused_gate_up_silu_mul_gemm_f16_specialized
// ============================================================================
//
// Real fused MLP kernel: GEMM (input @ weight^T producing packed gate_up)
// + SwiGLU epilogue, all in one dispatch with intermediates kept in
// registers / threadgroup memory only — never spilled to device memory.
//
// Pre-M4 implementation using `simdgroup_matrix<half, 8, 8>` MMA tiles
// (Metal 2.3+, supported on every Apple Silicon GPU we target). The
// M4+ variant using `MetalPerformancePrimitives::matmul2d` is tracked
// as a follow-up in `FERRITE_METAL_PROGRESS.md` (high-priority once
// the pre-M4 path is correctness-clean).
//
// Bindings (must match `interpreter::metal::lowering::lower_one` for
// `Instruction::FusedGateUpSiluMul`):
//   buffer(0) = output  [M, N]            half — silu(gate) * up
//   buffer(1) = input   [M, K]            half — post-rmsnorm hidden
//   buffer(2) = weight  [2*N, K]          half — packed [gate; up],
//                                                row-major; gate is
//                                                rows [0, N), up is
//                                                rows [N, 2N).
//
// Function constants:
//   0 = M     (uint) — bucket_m
//   1 = N     (uint) — INTERMEDIATE_SIZE (per-side; 2N is gate+up width)
//   2 = K     (uint) — Q_SIZE  (hidden_size)
//
// Tile shape per threadgroup (single simdgroup):
//   M_TILE = 8, N_TILE = 8, K_TILE = 8.
//
// One threadgroup with 32 threads (one simdgroup) computes one 8×8
// output tile by accumulating over the K-axis in 8-wide steps. Two
// `simdgroup_float8x8` accumulators (gate and up) sit alongside; the
// SwiGLU epilogue runs in registers between the K loop and the store.
//
// Threadgroup grid: ((N + 7)/8, (M + 7)/8, 1)
// Threads per group: (32, 1, 1)  — one simdgroup
//
// Tile size note: 8×8 is small but unconditionally correct at every
// (M, N, K) divisible by 8 — and TinyLlama is, with M up to 128 and
// N = 5632, K = 2048. Larger tiles (e.g. 32×32 with 4 simdgroups)
// are a perf upgrade for the same correctness; deferred.
//
// Edge handling: full coverage requires M and N to be multiples of 8.
// The kernel still dispatches `(N+7)/8 × (M+7)/8` tiles and lets the
// trailing partial tile read/write past the live range; per the macro's
// arena layout the row pitch of `output` is exactly N (no slack), so
// any trailing writes corrupt adjacent slots' first columns. We guard
// the trailing partial tile by zero-padding the simdgroup_load source
// rows / using a thread-private scratch buffer for the store. For
// TinyLlama / Llama-1.1B-class shapes (M ∈ {1, 16, 128}, N = 5632,
// K = 2048 — all divisible by 8) the guards never fire on the hot
// path; their only role is to keep the kernel safe at non-multiple-of-8
// shapes that future models might surface.

constant uint FUSED_MLP_M [[function_constant(0)]];
constant uint FUSED_MLP_N [[function_constant(1)]];
constant uint FUSED_MLP_K [[function_constant(2)]];

inline float silu_f(float x) {
    return x / (1.0f + exp(-x));
}

kernel void fused_gate_up_silu_mul_gemm_f16_specialized(
    device       half* output  [[buffer(0)]],   // [M, N]
    device const half* input   [[buffer(1)]],   // [M, K]
    device const half* weight  [[buffer(2)]],   // [2*N, K]
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint3 tid3    [[thread_position_in_threadgroup]],
    uint  sg_lane [[thread_index_in_simdgroup]])
{
    const uint tid = tid3.x;
    constexpr uint TILE = 8u;

    const uint m_base = tgid.y * TILE;   // first row of output tile
    const uint n_base = tgid.x * TILE;   // first col of output tile
    if (m_base >= FUSED_MLP_M || n_base >= FUSED_MLP_N) return;

    const uint M = FUSED_MLP_M;
    const uint N = FUSED_MLP_N;
    const uint K = FUSED_MLP_K;

    const bool m_full = (m_base + TILE <= M);
    const bool n_full = (n_base + TILE <= N);

    // Two accumulators initialized to zero. Layout of A is row-major
    // [M, K], so A's tile is `input + m_base*K` and we load successive
    // 8x8 tiles along the K axis. Layout of weight is row-major [2N, K]
    // = packed [gate; up], so:
    //     gate's k-th tile starts at  weight + n_base*K            + k_base
    //     up's   k-th tile starts at  weight + (N + n_base)*K      + k_base
    // We want C = A @ W^T (gate-side: [TILE_M, K] @ [K, TILE_N]
    // viewed as W slice transposed). simdgroup_multiply_accumulate
    // does C += A * B, so we load B with `simdgroup_load(transpose=true)`
    // to fetch the [TILE_K, TILE_N] B fragment from the [TILE_N, TILE_K]
    // weight slice.
    simdgroup_float8x8 acc_gate = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc_up   = simdgroup_float8x8(0.0f);

    threadgroup half a_pad[TILE * TILE];
    threadgroup half b_pad[TILE * TILE];

    for (uint k_base = 0u; k_base < K; k_base += TILE) {
        const bool k_full = (k_base + TILE <= K);

        simdgroup_half8x8 A;
        simdgroup_half8x8 Bg;
        simdgroup_half8x8 Bu;

        if (m_full && k_full) {
            simdgroup_load(A, input + m_base * K + k_base, K);
        } else {
            // Stage the partial A tile into threadgroup scratch with
            // out-of-range rows/cols zero-padded.
            for (uint t = tid; t < TILE * TILE; t += 32u) {
                uint r = t / TILE;
                uint c = t % TILE;
                uint mr = m_base + r;
                uint kc = k_base + c;
                a_pad[r * TILE + c] = (mr < M && kc < K)
                    ? input[mr * K + kc]
                    : half(0);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(A, a_pad, TILE);
        }

        // gate slice: weight rows [n_base, n_base + TILE)
        if (n_full && k_full) {
            simdgroup_load(Bg, weight + n_base * K + k_base, K, /*matrix_origin*/ ulong2(0, 0), /*transpose*/ true);
        } else {
            for (uint t = tid; t < TILE * TILE; t += 32u) {
                uint r = t / TILE;
                uint c = t % TILE;
                uint nr = n_base + r;
                uint kc = k_base + c;
                b_pad[r * TILE + c] = (nr < N && kc < K)
                    ? weight[nr * K + kc]
                    : half(0);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(Bg, b_pad, TILE, ulong2(0, 0), /*transpose*/ true);
        }
        simdgroup_multiply_accumulate(acc_gate, A, Bg, acc_gate);

        // up slice: weight rows [N + n_base, N + n_base + TILE)
        if (n_full && k_full) {
            simdgroup_load(Bu, weight + (N + n_base) * K + k_base, K, ulong2(0, 0), /*transpose*/ true);
        } else {
            for (uint t = tid; t < TILE * TILE; t += 32u) {
                uint r = t / TILE;
                uint c = t % TILE;
                uint nr = N + n_base + r;
                uint kc = k_base + c;
                b_pad[r * TILE + c] = (nr < 2u * N && kc < K)
                    ? weight[nr * K + kc]
                    : half(0);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(Bu, b_pad, TILE, ulong2(0, 0), /*transpose*/ true);
        }
        simdgroup_multiply_accumulate(acc_up, A, Bu, acc_up);
    }

    // SwiGLU epilogue + store. simdgroup_store writes a contiguous 8x8
    // tile with row-pitch `stride` (in elements). For full tiles we
    // store directly into `output + m_base*N + n_base` with stride N;
    // for partial tiles stage to threadgroup memory and let the threads
    // write out the live cells.
    threadgroup half c_pad[TILE * TILE];

    // Fuse silu(gate) * up via a temp simdgroup_float8x8.
    // simdgroup_matrix supports element-wise ops via thread-private
    // accessors; we use the lane-parallel pattern below.
    //
    // simdgroup_float8x8 doesn't expose a public per-element API across
    // every metal-stdlib version, so we materialize via simdgroup_store
    // to scratch, apply silu(gate)*up in registers per thread, then
    // either simdgroup_store back or write directly to output.
    threadgroup float gate_scratch[TILE * TILE];
    threadgroup float up_scratch  [TILE * TILE];

    simdgroup_store(acc_gate, gate_scratch, TILE);
    simdgroup_store(acc_up,   up_scratch,   TILE);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // 32 threads cooperate to produce 64 outputs (TILE*TILE).
    // 2 outputs per thread, in two passes.
    for (uint t = tid; t < TILE * TILE; t += 32u) {
        float g = gate_scratch[t];
        float u = up_scratch[t];
        float v = silu_f(g) * u;
        c_pad[t] = half(v);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Store live cells. m_full && n_full → contiguous 8x8 store.
    if (m_full && n_full) {
        for (uint t = tid; t < TILE * TILE; t += 32u) {
            uint r = t / TILE;
            uint c = t % TILE;
            output[(m_base + r) * N + (n_base + c)] = c_pad[r * TILE + c];
        }
    } else {
        for (uint t = tid; t < TILE * TILE; t += 32u) {
            uint r = t / TILE;
            uint c = t % TILE;
            uint mr = m_base + r;
            uint nc = n_base + c;
            if (mr < M && nc < N) {
                output[mr * N + nc] = c_pad[r * TILE + c];
            }
        }
    }
}

// ============================================================================
// fused_gate_up_silu_mul_decode_f16_specialized
// ============================================================================
//
// M=1 fast path. Drops simdgroup_matrix entirely — for a single
// query row the MMA tile machinery is overhead, since 7/8 of every
// loaded A fragment is zero-padded waste. Instead each threadgroup
// computes 8 outputs via simd_sum dot products: one simdgroup per
// output, 32 lanes accumulating K/32 partial products each.
//
// Bindings (must match the matrix variant exactly so the lowering
// stays kernel-agnostic):
//   buffer(0) = output [1, N]
//   buffer(1) = input  [1, K]
//   buffer(2) = weight [2*N, K]   packed [gate; up]
//
// Function constants:
//   0 = M (uint) — must be 1 for this kernel; defensive early-out
//                  if M > 1 (lowering should never wire this kernel
//                  for M>1, but the runtime cost is one branch).
//   1 = N (uint) — INTERMEDIATE_SIZE
//   2 = K (uint) — Q_SIZE / hidden_size
//
// Tile shape:
//   - Threadgroup: 256 threads = 8 simdgroups × 32 lanes.
//   - Output tile per threadgroup: 8 elements along N.
//   - One simdgroup owns one output index.
//
// Threadgroup grid: ((N + 7) / 8, 1, 1).
// Threads per group: (256, 1, 1).
//
// No threadgroup-scratch staging, no per-K-iter barriers — every
// device read is coalesced across the 32 lanes. Bandwidth-bound;
// for TinyLlama K=2048 that's K/32 = 64 inner iterations per
// (output, gate|up) pair = 128 inner iterations per output.

constant uint FUSED_MLP_DECODE_M [[function_constant(3)]];
constant uint FUSED_MLP_DECODE_N [[function_constant(4)]];
constant uint FUSED_MLP_DECODE_K [[function_constant(5)]];

kernel void fused_gate_up_silu_mul_decode_f16_specialized(
    device       half* output  [[buffer(0)]],   // [1, N]
    device const half* input   [[buffer(1)]],   // [1, K]
    device const half* weight  [[buffer(2)]],   // [2*N, K]
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint  lane_id [[thread_index_in_simdgroup]])
{
    if (FUSED_MLP_DECODE_M != 1u) return;

    const uint N = FUSED_MLP_DECODE_N;
    const uint K = FUSED_MLP_DECODE_K;

    // Output index this simdgroup owns: tgid.x * 8 + sg_id.
    const uint n = tgid.x * 8u + sg_id;
    if (n >= N) return;

    // Per-row weight pointers (gate row n, up row N+n).
    device const half* w_gate = weight + n * K;
    device const half* w_up   = weight + (N + n) * K;

    // Each lane sums K/32 elements (with K rarely a multiple of 32?
    // for TinyLlama K=2048 it is; guard with `k < K` check anyway).
    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
    for (uint k = lane_id; k < K; k += 32u) {
        const float a = float(input[k]);
        gate_acc += a * float(w_gate[k]);
        up_acc   += a * float(w_up[k]);
    }
    gate_acc = simd_sum(gate_acc);
    up_acc   = simd_sum(up_acc);

    if (lane_id == 0u) {
        const float silu_gate = gate_acc / (1.0f + exp(-gate_acc));
        output[n] = half(silu_gate * up_acc);
    }
}
