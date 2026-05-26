// SPDX-License-Identifier: Apache-2.0
//
// Standalone fused `silu(gate) * up` for the decomposed q-MLP path.
//
// Plan P12 branch (i): when both gate_proj and up_proj are MLX-affine
// quantized, the macro emits the SwiGLU MLP as three instructions:
//   AffineQmm(gate_proj)  -> gate scratch [M, I]
//   AffineQmm(up_proj)    -> up   scratch [M, I]
//   SiluMul(gate, up)     -> out          [M, I]
// instead of the single `FusedGateUpSiluMul` (which assumes Dense
// storage and produces a packed [M, 2I] gate_up via cuBLAS GEMM).
// This is one extra device round-trip per MLP layer vs. the dense
// path; eliminating it would require a hand-rolled fused
// affine_qmm + silu + mul kernel which the
// `feedback_no_handcoded_fusion` rule forbids.
//
// Function constants:
//   SILU_MUL_N — total output element count (= M * intermediate_size)
//
// Dispatch: 1 thread per output element. Float accumulator on the
// silu so denormalized half/bfloat exp() doesn't flush to zero on
// the negative tail.

#include <metal_stdlib>

using namespace metal;

// ThunderMittens — the silu·mul compute atom the [[kernel]] below and the
// wavefront megakernel compose.
#include "mittens/silu_mul.h"

constant uint SILU_MUL_N [[function_constant(0)]];

template <typename T>
[[kernel]] void silu_mul(
    device       T* out  [[buffer(0)]],
    const device T* gate [[buffer(1)]],
    const device T* up   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  mittens::silu_mul_impl<T>(out, gate, up, gid, SILU_MUL_N);
}

#define INST_SILU_MUL(dtype_tag, mtl_type)                                \
  template [[host_name("silu_mul_" #dtype_tag)]] [[kernel]] void          \
  silu_mul<mtl_type>(                                                     \
      device       mtl_type* out  [[buffer(0)]],                          \
      const device mtl_type* gate [[buffer(1)]],                          \
      const device mtl_type* up   [[buffer(2)]],                          \
      uint gid [[thread_position_in_grid]]);

INST_SILU_MUL(f16,  half)
INST_SILU_MUL(bf16, bfloat)

// ─────────────────────────────────────────────────────────────────
// wavefront_silu_mul_mega — PD-wavefront silu·mul composition proof.
// P co-resident threadgroups grid-stride over the SILU_MUL_N output elements
// (e = thread_position_in_grid, += threads_per_grid) composing
// mittens::silu_mul_impl. Embarrassingly parallel (one element per thread, no
// cross-thread state) ⇒ BIT-EXACT vs the whole silu_mul for any P.
// ─────────────────────────────────────────────────────────────────
template <typename T>
[[kernel]] void wavefront_silu_mul_mega(
    device       T* out  [[buffer(0)]],
    const device T* gate [[buffer(1)]],
    const device T* up   [[buffer(2)]],
    uint gtid      [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]])
{
  for (uint e = gtid; e < SILU_MUL_N; e += grid_size) {
    mittens::silu_mul_impl<T>(out, gate, up, e, SILU_MUL_N);
  }
}

#define INST_WF_SILU_MUL_MEGA(dtype_tag, mtl_type)                            \
  template [[host_name("wavefront_silu_mul_mega_" #dtype_tag)]] [[kernel]]    \
  decltype(wavefront_silu_mul_mega<mtl_type>)                                 \
      wavefront_silu_mul_mega<mtl_type>;
INST_WF_SILU_MUL_MEGA(f16,  half)
INST_WF_SILU_MUL_MEGA(bf16, bfloat)
