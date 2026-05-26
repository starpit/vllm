// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

// ThunderMittens — the RMSNorm compute atom the specialized [[kernel]] below
// and the wavefront megakernel compose.
#include "mittens/rmsnorm.h"

/// RMSNorm kernel: y = x * weight / sqrt(mean(x^2) + eps)
///
/// Grid: (M, 1, 1) where M = batch_size
/// Threadgroup: (min(N, 1024), 1, 1) where N = hidden_size
kernel void rmsnorm_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    // Compute mean of squares using threadgroup reduction
    threadgroup float shared_sum[1024];
    
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float val = float(input[gid * N + i]);
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction in shared memory
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Broadcast RMS to all threads
    float rms = sqrt(shared_sum[0] / float(N) + eps);
    
    // Normalize and scale
    for (uint i = tid; i < N; i += tg_size) {
        float val = float(input[gid * N + i]);
        float w = float(weight[i]);
        output[gid * N + i] = half((val / rms) * w);
    }
}

/// Phase 5.B.3 specialized variant: layer-independent params baked
/// in via `[[function_constant(N)]]`, no runtime constants buffer.
/// Index assignments must match `ferrite-forward::interpreter::metal::pipelines`:
///   0 = M (uint), 1 = N/HIDDEN_SIZE (uint), 2 = EPS (float).
constant uint  RMSNORM_M           [[function_constant(0)]];
constant uint  RMSNORM_HIDDEN_SIZE [[function_constant(1)]];
constant float RMSNORM_EPS         [[function_constant(2)]];

// Template form (`<T_act, T_scale>`): same in-register cast pattern as
// the affine quant kernels (`shaders/quantized_*.metal`). The kernel
// reads activations through a `T_act` device pointer, the gain through
// a separate `T_scale` device pointer, and promotes both to `float`
// for the reduction. Lets the loader keep RMSNorm gains in their
// on-disk dtype (F16 for every sampled mlx-community / Llama-3.x
// checkpoint) instead of F16→BF16 truncating at load on the bf16
// stack — the same regression repair P10b applied to quant scales.
//
// Bindings (must match `interpreter::metal::lowering::lower_one` for
// `Instruction::RmsNorm`):
//   buffer(0) = output (out_slot — written)
//   buffer(1) = input  (in_slot  — read)
//   buffer(2) = weight (read; on-disk dtype)
template <typename T_act, typename T_scale>
[[kernel]] void rmsnorm_specialized_impl(
    device       T_act*   output [[buffer(0)]],
    device const T_act*   input  [[buffer(1)]],
    device const T_scale* weight [[buffer(2)]],
    uint gid     [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Thin wrapper: own the threadgroup reduction scratch and forward the
    // layer-independent function constants to the mittens compute atom.
    threadgroup float shared_sum[1024];
    mittens::rmsnorm_impl<T_act, T_scale>(
        output, input, weight, shared_sum,
        RMSNORM_M, RMSNORM_HIDDEN_SIZE, RMSNORM_EPS,
        gid, tid, tg_size);
}

#define INST_RMSNORM(act_tag, act_type, scale_tag, scale_type)              \
  template [[host_name("rmsnorm_" #act_tag "_s_" #scale_tag "_specialized")]] \
  [[kernel]] decltype(rmsnorm_specialized_impl<act_type, scale_type>)       \
      rmsnorm_specialized_impl<act_type, scale_type>;

// Coverage: T_scale tracks on-disk gain dtype. Llama-3.x / Qwen2.5 /
// SmolLM mlx-community 4bit ship F16 RMSNorm gains; Qwen3 family ships
// BF16. Both are loaded with `take_keep_dtype` (no loader-side cast),
// so the kernel template must cover both. `W::SCALE_DTYPE` picks the
// arm at lowering time.
INST_RMSNORM(f16,  half,   f16, half)
INST_RMSNORM(bf16, bfloat, f16, half)
INST_RMSNORM(bf16, bfloat, bf16, bfloat)
INST_RMSNORM(f16,  half,   bf16, bfloat)

// ─────────────────────────────────────────────────────────────────
// wavefront_rmsnorm_mega — PD-wavefront rmsnorm composition proof.
// P co-resident threadgroups (one per GPU core), each with the SAME thread
// count as the whole rmsnorm, loop the rows they own
// (g = tgpos.x, +grid_tg.x, …) composing mittens::rmsnorm_impl once per row.
// Disjoint output rows ⇒ no cross-TG flags (mirrors A1 wavefront_qmv_mega).
// The per-row reduction reuses the same tg_size as the whole kernel, and the
// trailing barrier serialises shared_sum reuse across a worker's rows, so the
// output is BIT-EXACT vs the whole rmsnorm_specialized for every P. Function
// constants 0/1/2 = M / HIDDEN_SIZE / EPS, identical to the whole kernel.
// ─────────────────────────────────────────────────────────────────
template <typename T_act, typename T_scale>
[[kernel]] void wavefront_rmsnorm_mega(
    device       T_act*   output  [[buffer(0)]],
    device const T_act*   input   [[buffer(1)]],
    device const T_scale* weight  [[buffer(2)]],
    uint tgpos   [[threadgroup_position_in_grid]],
    uint grid_tg [[threadgroups_per_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  threadgroup float shared_sum[1024];
  for (uint g = tgpos; g < RMSNORM_M; g += grid_tg) {
    mittens::rmsnorm_impl<T_act, T_scale>(
        output, input, weight, shared_sum,
        RMSNORM_M, RMSNORM_HIDDEN_SIZE, RMSNORM_EPS,
        g, tid, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup); // serialise shared_sum reuse across rows
  }
}

#define INST_WF_RMSNORM_MEGA(act_tag, act_type, scale_tag, scale_type)        \
  template [[host_name("wavefront_rmsnorm_mega_" #act_tag "_s_" #scale_tag)]]  \
  [[kernel]] decltype(wavefront_rmsnorm_mega<act_type, scale_type>)           \
      wavefront_rmsnorm_mega<act_type, scale_type>;
INST_WF_RMSNORM_MEGA(bf16, bfloat, f16, half)
INST_WF_RMSNORM_MEGA(f16,  half,   f16, half)

/// BF16 variant (uses float16 as Metal doesn't have native bfloat16)
kernel void rmsnorm_bf16(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    threadgroup float shared_sum[1024];
    
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float val = input[gid * N + i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float rms = sqrt(shared_sum[0] / float(N) + eps);
    
    for (uint i = tid; i < N; i += tg_size) {
        float val = input[gid * N + i];
        float w = weight[i];
        output[gid * N + i] = (val / rms) * w;
    }
}
