// SPDX-License-Identifier: Apache-2.0
// ThunderMittens — PD-wavefront persistent decode megakernel / trivial tape
// player (the "big metal build", ≈T6). ONE persistent kernel; each of the P=10
// co-resident threadgroups interprets ITS bin-packed tape (from
// `ferrite_wavefront::region_schedule`), composing the validated `mittens`
// atoms. NEVER hand-write kernel math and NEVER make a decision in the player —
// all intelligence (which atom, operands, offsets, K/N, flags) is baked into
// the tape by the compiler; the player only indexes + dispatches.
//
// BINDLESS OPERANDS (proven in `tests/wavefront_addr_probe.rs`): one persistent
// kernel can't `setBuffer`-bind the ~150 operands a decode forward touches
// (Metal's ~31-slot limit). Instead every operand rides a `gpuAddress`
// pointer-table — a plain buffer of u64 GPU virtual addresses — and the kernel
// casts `addrs[BufId] (+ byte offset)` to a `device` pointer. The operand
// buffers are kept resident by the allocator's shared `MetalResidencySet`.
//
// This file grows incrementally with a bit-exact GPU proof at each step:
//   step 2 (here): `wavefront_qmv_bindless` — a single `qmv_fast` atom whose
//     w/scales/biases/x/y are resolved through the address table, proven
//     bit-exact vs the normal whole `affine_qmv_fast`. Isolates "a real atom
//     called through bindless operands" before the tape loop / sync / switch.

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

#include "mittens/attention.h"
#include "mittens/elementwise.h"
#include "mittens/qmv.h"
#include "mittens/rmsnorm.h"
#include "mittens/rope.h"
#include "mittens/silu_mul.h"
#include "mittens/sync.h"

// Shapes ride as function constants (local to the `wavefront_layer` library).
constant int WL_K [[function_constant(0)]]; // qmv in_vec_size (K)
constant int WL_N [[function_constant(1)]]; // qmv out_vec_size (N)

// ── step 2: one qmv_fast atom, operands resolved through the address table ──
// `addrs` holds 5 gpuAddresses in operand-table order [w, scales, biases, x,
// y]. The kernel reconstructs the typed device pointers and forwards them to
// the verbatim `mittens::qmv_fast_impl`. Dispatch shape is identical to the
// whole `affine_qmv_fast`: grid (1, ceil(N/8), 1), TG [32,2,1].
template <typename T_act, typename T_scale, int group_size, int bits>
[[kernel]] void wavefront_qmv_bindless(
    device const ulong* addrs [[buffer(0)]], // [w, scales, biases, x, y]
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const device uint32_t* w = (const device uint32_t*)(addrs[0]);
  const device T_scale* scales = (const device T_scale*)(addrs[1]);
  const device T_scale* biases = (const device T_scale*)(addrs[2]);
  const device T_act* x = (const device T_act*)(addrs[3]);
  device T_act* y = (device T_act*)(addrs[4]);
  mittens::qmv_fast_impl<T_act, T_scale, group_size, bits>(
      w, scales, biases, x, y, WL_K, WL_N, tid, simd_gid, simd_lid,
      /*row_vec_size=*/WL_K);
}

#define INST_WL_QMV_BINDLESS(act_tag, act_type, scale_tag, scale_type, gs)  \
  template [[host_name("wavefront_qmv_bindless_" #act_tag "_s_" #scale_tag   \
                       "_gs_" #gs "_b_4")]] [[kernel]]                       \
  decltype(wavefront_qmv_bindless<act_type, scale_type, gs, 4>)             \
      wavefront_qmv_bindless<act_type, scale_type, gs, 4>;
INST_WL_QMV_BINDLESS(bf16, bfloat, f16, half, 64)
INST_WL_QMV_BINDLESS(f16, half, f16, half, 64)

// ─────────────────────────────────────────────────────────────────
// step 3: the TRIVIAL interpret loop + shape-class switch.
//
// One persistent kernel; each co-resident threadgroup (worker) walks ITS tape
// and dispatches each instruction. The tape, the shape-class table, and the
// operand (gpuAddress) table are DATA produced by the compiler — the player
// makes ZERO decisions: read instruction, index the shape-class table, index
// the operand table, call the atom. Fixed 1024-thread (32-simdgroup) TG; each
// shape-class arm uses the thread subset its atom needs (the deferred per-atom
// TG reconciliation lives HERE, in the arms).
//
// Encoding (hand-built in the test now; emitted by region_schedule serializer
// later):
//   tape[pc]   = uint4(opcode, shape_class, operand_base, flag)
//                opcode: 0=Compute 1=Signal 2=Wait
//   shapes[sc] = uint4(op_kind, K, N, _)   op_kind: 0=QMV
//   operands[] = ulong gpuAddress table; a QMV reads operands[base + 0..5]
//                = (w, scales, biases, x, y), each already + its byte offset.
// ─────────────────────────────────────────────────────────────────

// Each worker (co-resident threadgroup) runs the tape slice
// [tape_offsets[me], tape_offsets[me+1]); `tape_offsets` has P+1 entries.

// QMV shape-class arm, parameterised by the simdgroup sub-range the compiler's
// tranche bin-packer assigned this op (`local_sg` in `[0, sg_count)`, the op's
// simdgroup index; `sg_count` even ≥ 2). The op's `sg_count` simdgroups cover
// `sg_count/2` of `qmv_fast`'s 8-row groups per wave (gi = local_sg>>1 selects
// the group, role = local_sg&1 is qmv_fast's simd_gid in {0,1}), so each
// group's two simdgroups replay the whole kernel's 2-simdgroup compute exactly
// ⇒ bit-exact regardless of how many simdgroups the op got. Solo (`sg_count ==
// 32`) reproduces the old whole-TG behaviour (16 groups/wave); a packed op
// (e.g. `sg_count == 6`) covers 3 groups/wave so several ops run concurrently
// on disjoint simdgroup ranges of the same worker — more outstanding weight
// loads, the latency-hiding the tranche packer is for.
template <typename T_act, typename T_scale, int group_size, int bits>
METAL_FUNC void wl_qmv_arm(
    const device ulong* operands,
    uint operand_base,
    uint k,
    uint n,
    uint local_sg,
    uint sg_count,
    uint simd_lid,
    uint row_vec) {
  // `k` = K-window length (loop + x width); `row_vec` = weight full row K (the
  // row stride). Equal off-window (N-block qmv); `row_vec > k` is a split-K
  // partial whose weight/scales/biases operands are pre-offset to the K-slice.
  const device uint32_t* w = (const device uint32_t*)(operands[operand_base + 0u]);
  const device T_scale* s = (const device T_scale*)(operands[operand_base + 1u]);
  const device T_scale* b = (const device T_scale*)(operands[operand_base + 2u]);
  const device T_act* x = (const device T_act*)(operands[operand_base + 3u]);
  device T_act* y = (device T_act*)(operands[operand_base + 4u]);
  const uint num_groups = (n + 7u) / 8u;
  const uint gi = local_sg >> 1;
  const uint role = local_sg & 1u;
  const uint groups_per_wave = sg_count >> 1; // sg_count even ⇒ exact
  for (uint base = 0u; base < num_groups; base += groups_per_wave) {
    uint g = base + gi;
    if (g < num_groups) {
      mittens::qmv_fast_impl<T_act, T_scale, group_size, bits>(
          w, s, b, x, y, int(k), int(n), uint3(0u, g, 0u), role, simd_lid, int(row_vec));
    }
  }
}

// Coherent-output qmv arm: identical to `wl_qmv_arm` but operand 4 is the
// coherent (atomic u32) handoff buffer, and `qmv_fast_coh_impl` writes each
// simdgroup's rows STRAIGHT into it (sentinel-safe, packed). No compute→publish
// barrier: each simdgroup stores only its own rows, so nothing re-reads another
// simdgroup's output; the store IS the readiness signal (data-IS-the-flag).
template <typename T_act, typename T_scale, int group_size, int bits>
METAL_FUNC void wl_qmv_coh_arm(
    const device ulong* operands,
    uint operand_base,
    uint k,
    uint n,
    uint local_sg,
    uint sg_count,
    uint simd_lid) {
  const device uint32_t* w = (const device uint32_t*)(operands[operand_base + 0u]);
  const device T_scale* s = (const device T_scale*)(operands[operand_base + 1u]);
  const device T_scale* b = (const device T_scale*)(operands[operand_base + 2u]);
  const device T_act* x = (const device T_act*)(operands[operand_base + 3u]);
  device atomic_uint* y_coh = (device atomic_uint*)(operands[operand_base + 4u]);
  const uint num_groups = (n + 7u) / 8u;
  const uint gi = local_sg >> 1;
  const uint role = local_sg & 1u;
  const uint groups_per_wave = sg_count >> 1;
  for (uint base = 0u; base < num_groups; base += groups_per_wave) {
    uint g = base + gi;
    if (g < num_groups) {
      mittens::qmv_fast_coh_impl<T_act, T_scale, group_size, bits>(
          w, s, b, x, y_coh, int(k), int(n), uint3(0u, g, 0u), role, simd_lid);
    }
  }
}

// Tall-skinny qmv arm — the matvec primitive for small CONTIGUOUS K (== D, the
// head_dim) × large N. Used by split-K partials whose K-window is head_dim-wide,
// below qmv_fast's 512-K minimum. One quad-simdgroup (8 quads) covers 64 output
// rows per tile; `quad_gid`/`quad_lid` derive from `simd_lid` so the player needs
// no quadgroup attributes. `row_vec` is the weight's full row K (the row stride),
// so the operands pre-offset to the K-slice read it with the parent stride. `D`
// is the decode head_dim — a compile-time specialization seam (the cost/shape-
// driven choice of variant + chunk width; see project_pd_wavefront_cost_driven).
template <typename T_act, typename T_scale, int group_size, int bits, int D>
METAL_FUNC void wl_qmv_quad_arm(
    const device ulong* operands,
    uint operand_base,
    uint k,
    uint n,
    uint local_sg,
    uint sg_count,
    uint simd_lid,
    uint row_vec) {
  const device uint32_t* w = (const device uint32_t*)(operands[operand_base + 0u]);
  const device T_scale* s = (const device T_scale*)(operands[operand_base + 1u]);
  const device T_scale* b = (const device T_scale*)(operands[operand_base + 2u]);
  const device T_act* x = (const device T_act*)(operands[operand_base + 3u]);
  device T_act* y = (device T_act*)(operands[operand_base + 4u]);
  const uint quad_gid = simd_lid >> 2;  // 0..8 quads within this simdgroup
  const uint quad_lid = simd_lid & 3u;  // 0..4 thread within the quad
  const uint tiles = (n + 63u) / 64u;   // one quad-simdgroup covers 64 output rows
  for (uint t = local_sg; t < tiles; t += sg_count) {
    mittens::qmv_quad_impl<T_act, T_scale, group_size, bits, D>(
        w, s, b, x, y, int(k), int(n), uint3(0u, t, 0u), quad_gid, quad_lid, int(row_vec));
  }
}

// Opcodes (tape[pc].x) and op_kinds (shapes[sc].x).
constant constexpr uint WL_OPC_COMPUTE = 0u;
constant constexpr uint WL_OPC_SIGNAL = 1u;
constant constexpr uint WL_OPC_WAIT = 2u;
constant constexpr uint WL_OPC_BARRIER = 3u; // compiler-placed intra-worker fence
constant constexpr uint WL_OP_QMV = 0u;
constant constexpr uint WL_OP_PUBLISH = 1u; // pack a written region → coherent handoff (atomic)
constant constexpr uint WL_OP_ACQUIRE = 2u; // load+unpack handoff → a private readable copy
constant constexpr uint WL_OP_RMSNORM = 3u; // whole-row rmsnorm, 1024-thread reduction
constant constexpr uint WL_OP_SILU_MUL = 4u; // elementwise silu(gate)*up
constant constexpr uint WL_OP_ROPE = 5u;     // NeoX pair-rotate Q or K in place
constant constexpr uint WL_OP_ATTN = 6u;     // paged decode attention (loops q-heads)
constant constexpr uint WL_OP_ADD = 7u;      // elementwise residual add
constant constexpr uint WL_OP_ROPE_APPEND = 8u; // rotate K in place + write K/V to paged cache
constant constexpr uint WL_OP_QMV_COH = 9u; // qmv whose output is written DIRECTLY into the coherent handoff buffer (PAT-4; no compute→publish barrier)
constant constexpr uint WL_OP_SUM_REDUCE = 10u; // split-K all-reduce: out = sum of num_partials operand regions (f32 accum, one round)
constant constexpr uint WL_OP_QMV_QUAD = 11u; // tall-skinny qmv (small K == head_dim, large N) for split-K partials; K-window via row_vec

// Each shape-class descriptor is a fixed-width record of WL_SHAPE_STRIDE u32s:
// [op_kind, p1, p2, p3, p4, p5, p6, _]. Wide enough for attention's 6 dims;
// the simpler ops use the leading slots. (Kept a flat u32 table rather than a
// struct so the host serializer stays trivially `&[u32]`.)
constant constexpr uint WL_SHAPE_STRIDE = 8u;

template <typename T_act, typename T_scale, int group_size, int bits>
[[kernel]] void wavefront_player(
    device const uint4* tape [[buffer(0)]],         // (opcode, shape_class, operand_base, flag)
    device const uint* shapes [[buffer(1)]],        // flat [num_classes * WL_SHAPE_STRIDE]
    device const ulong* operands [[buffer(2)]],     // gpuAddress table
    device const uint* tape_offsets [[buffer(3)]],  // [P+1]; worker me runs [me, me+1)
    device atomic_uint* flags [[buffer(4)]],        // [num_flags], zeroed by host
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint3 grid_size [[threadgroups_per_grid]],
    uint3 threads_per_tg [[threads_per_threadgroup]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint me = tgpos.x;
  const uint replica = tgpos.y;
  const uint k_replicas = grid_size.y;
  const uint simdgroups_per_tg = threads_per_tg.x >> 5u; // threads/32
  const uint effective_simd_gid = replica * simdgroups_per_tg + simd_gid;
  const uint effective_sg_default = k_replicas * simdgroups_per_tg;
  const uint start = tape_offsets[me];
  const uint end = tape_offsets[me + 1u];
  // Threadgroup scratch the arms reuse across instructions (a barrier between
  // instructions serialises reuse). rmsnorm's 1024-wide reduction uses
  // `shared_sum`; attention's per-simdgroup combine uses tg_outputs/max/sum.
  threadgroup float shared_sum[1024];
  threadgroup float tg_outputs[1024]; // BN(32) * BD(32)
  threadgroup float tg_max[32];       // BN
  threadgroup float tg_sum[32];       // BN
  for (uint pc = start; pc < end; pc++) {
    uint4 ins = tape[pc];
    switch (ins.x) {
      case WL_OPC_COMPUTE: {
        const uint sb = ins.y * WL_SHAPE_STRIDE;
        const uint op = shapes[sb + 0u];
        // Per-op simdgroup range, packed into the flag field (`ins.w`) by the
        // compiler's tranche bin-packer: a tranche's mutually-independent ops
        // get DISJOINT simdgroup ranges so they run CONCURRENTLY on this
        // worker's 32 simdgroups (more outstanding weight loads → better BW).
        // `sg_count == 0` ⇒ the whole 32-simdgroup TG (a solo op, or any tape
        // the packer left untouched). Only the simdgroup-local arms (qmv,
        // silu_mul, add, rope — no threadgroup scratch, no TG-wide barrier)
        // honour the range; the threadgroup-wide arms (rmsnorm/attn/publish/
        // acquire/rope_append reduce or barrier across all 1024 threads) are
        // always emitted solo and use the whole TG via `tid_in_tg`/1024.
        uint sg_start = ins.w & 0xFFu;
        uint sg_count = (ins.w >> 8u) & 0xFFu;
        if (sg_count == 0u) {
          sg_start = 0u;
          sg_count = effective_sg_default;  // K-aware: K*simdgroups_per_TG
        }
        const bool in_range = effective_simd_gid >= sg_start && effective_simd_gid < sg_start + sg_count;
        const uint local_sg = effective_simd_gid - sg_start;
        const uint op_threads = sg_count << 5u;
        const uint local_tid = (local_sg << 5u) + simd_lid;
        switch (op) {
          case WL_OP_QMV:
            if (in_range) {
              // shape (QMV, k, n, row_vec); row_vec slot 0 ⇒ no K-window
              // (row stride = k). row_vec > k ⇒ split-K partial.
              uint row_vec = shapes[sb + 3u];
              if (row_vec == 0u) {
                row_vec = shapes[sb + 1u];
              }
              wl_qmv_arm<T_act, T_scale, group_size, bits>(
                  operands, ins.z, shapes[sb + 1u], shapes[sb + 2u], local_sg, sg_count,
                  simd_lid, row_vec);
            }
            break;
          case WL_OP_QMV_COH:
            if (in_range) {
              wl_qmv_coh_arm<T_act, T_scale, group_size, bits>(
                  operands, ins.z, shapes[sb + 1u], shapes[sb + 2u], local_sg, sg_count,
                  simd_lid);
            }
            break;
          case WL_OP_QMV_QUAD:
            if (in_range) {
              // shape (QMV_QUAD, k==D, n, row_vec); row_vec 0 ⇒ stride == k.
              // D is selected from k at runtime (compile-time specializations
              // for the supported split-K chunk widths). Add more arms as the
              // cost-sweep narrows down the productive granularities.
              uint k_dim = shapes[sb + 1u];
              uint n_out = shapes[sb + 2u];
              uint row_vec = shapes[sb + 3u];
              if (row_vec == 0u) {
                row_vec = k_dim;
              }
              if (k_dim == 64u) {
                wl_qmv_quad_arm<T_act, T_scale, group_size, bits, 64>(
                    operands, ins.z, k_dim, n_out, local_sg, sg_count, simd_lid, row_vec);
              } else if (k_dim == 128u) {
                wl_qmv_quad_arm<T_act, T_scale, group_size, bits, 128>(
                    operands, ins.z, k_dim, n_out, local_sg, sg_count, simd_lid, row_vec);
              } else if (k_dim == 256u) {
                wl_qmv_quad_arm<T_act, T_scale, group_size, bits, 256>(
                    operands, ins.z, k_dim, n_out, local_sg, sg_count, simd_lid, row_vec);
              }
            }
            break;
          case WL_OP_PUBLISH:
            // PAT-4 (data-IS-the-flag): pack operands[base+1] (this worker's
            // written region, bf16) into the coherent operands[base+0] (atomic
            // u32), SENTINEL-SAFE (never stores 0). pair0=p1, n_pairs=p2. The
            // store itself is the readiness signal — no Signal, no fence after.
            mittens::wf_publish_pairs_nz_tg<T_act>(
                (device atomic_uint*)(operands[ins.z + 0u]),
                (const device T_act*)(operands[ins.z + 1u]),
                shapes[sb + 1u], shapes[sb + 2u], tid_in_tg, threads_per_tg.x);
            break;
          case WL_OP_ACQUIRE:
            // PAT-4: SPIN on each coherent slot operands[base+1] (atomic u32)
            // until non-sentinel (written this step), unpacking into the private
            // operands[base+0] (bf16). n_pairs=p1. The per-slot spin IS the wait
            // — no Wait, no flag. (The trailing compiler BARRIER still fences the
            // private copy before the consumer reads it cross-simdgroup.)
            mittens::wf_acquire_pairs_spin_tg<T_act>(
                (device T_act*)(operands[ins.z + 0u]),
                (const device atomic_uint*)(operands[ins.z + 1u]),
                shapes[sb + 1u], tid_in_tg, threads_per_tg.x);
            break;
          case WL_OP_RMSNORM:
            // operands [out, in, weight]; shape (RMSNORM, hidden, eps_bits, ...).
            // The whole 1024-thread TG reduces the single decode row (gid 0);
            // eps rides as its float bit-pattern in the uint shape slot.
            mittens::rmsnorm_impl<T_act, T_scale>(
                (device T_act*)(operands[ins.z + 0u]),
                (const device T_act*)(operands[ins.z + 1u]),
                (const device T_scale*)(operands[ins.z + 2u]),
                shared_sum, 1u, shapes[sb + 1u], as_type<float>(shapes[sb + 2u]),
                0u, tid_in_tg, threads_per_tg.x);
            break;
          case WL_OP_SILU_MUL: {
            // operands [out, gate, up]; shape (SILU_MUL, n, ...). The op's
            // `op_threads` grid-stride the n elements (per-element pure ⇒
            // bit-exact at any thread count).
            if (in_range) {
              uint n = shapes[sb + 1u];
              for (uint e = local_tid; e < n; e += op_threads) {
                mittens::silu_mul_impl<T_act>(
                    (device T_act*)(operands[ins.z + 0u]),
                    (const device T_act*)(operands[ins.z + 1u]),
                    (const device T_act*)(operands[ins.z + 2u]),
                    e, n);
              }
            }
            break;
          }
          case WL_OP_ADD: {
            // operands [out, a, b]; shape (ADD, n, ...). Residual add; out may
            // alias a or b (in-place residual) — each element is independent.
            if (in_range) {
              uint n = shapes[sb + 1u];
              for (uint e = local_tid; e < n; e += op_threads) {
                mittens::add_impl<T_act>(
                    (device T_act*)(operands[ins.z + 0u]),
                    (const device T_act*)(operands[ins.z + 1u]),
                    (const device T_act*)(operands[ins.z + 2u]),
                    e, n);
              }
            }
            break;
          }
          case WL_OP_SUM_REDUCE: {
            // operands [out, p0, p1, …]; shape (SUM_REDUCE, n, num_partials).
            // The split-K all-reduce: out[e] = sum_k partial_k[e]. Each partial
            // is one activation-K-chunk's contribution; a cross-worker partial
            // was ACQUIREd into a local copy, so all operands are readable here.
            // The atom accumulates in f32 and rounds once — so split-K differs
            // from a whole matvec only by that single round (compose, not math).
            if (in_range) {
              uint n = shapes[sb + 1u];
              uint num_partials = shapes[sb + 2u];
              for (uint e = local_tid; e < n; e += op_threads) {
                mittens::sum_reduce_impl<T_act>(
                    (device T_act*)(operands[ins.z + 0u]),
                    operands + ins.z + 1u, num_partials, e, n);
              }
            }
            break;
          }
          case WL_OP_ROPE: {
            // operands [x, cos_sin, positions]; shape (ROPE, head_dim,
            // num_heads, rot_dim, ...). `cos_sin` is the WHOLE rotary table
            // [max_pos, rot_dim] (each row = [cos[half] | sin[half]]); the live
            // decode position is `positions[0]`. The per-position row is a
            // RUNTIME quantity (it changes every decode step), so it is indexed
            // here exactly like the oracle (`cos_sin + pos*rot_dim`) — NEVER a
            // compile-time-baked offset. Each thread owns distinct (head, d<half)
            // pairs and rotates the (d, d+half) pair of its head's row.
            if (in_range) {
              device T_act* x = (device T_act*)(operands[ins.z + 0u]);
              const device T_act* cos_sin = (const device T_act*)(operands[ins.z + 1u]);
              const device uint* positions = (const device uint*)(operands[ins.z + 2u]);
              uint head_dim = shapes[sb + 1u];
              uint num_heads = shapes[sb + 2u];
              uint rot_dim = shapes[sb + 3u];
              uint half_dim = rot_dim / 2u;
              uint pos = positions[0];
              const device T_act* cos_row = cos_sin + pos * rot_dim;
              const device T_act* sin_row = cos_sin + pos * rot_dim + half_dim;
              for (uint idx = local_tid; idx < num_heads * half_dim; idx += op_threads) {
                uint h = idx / half_dim;
                uint d = idx % half_dim;
                mittens::rope_rotate_pair<T_act>(x + h * head_dim, cos_row, sin_row, d, half_dim);
              }
            }
            break;
          }
          case WL_OP_ATTN: {
            // operands [output, q, seq_used_k, block_table, k_cache, v_cache];
            // shape (ATTN, head_dim, num_q, num_kv, scale_bits, block_size,
            // max_blocks, head_range). num_q/num_kv are GLOBAL (the GQA ratio +
            // cache stride). head_range = (qh_start << 16) | qh_count selects a
            // q-head block for a head-tiled (partitioned) attn; q/output are
            // block-based, so q_head_idx loops LOCAL [0, qh_count) and the GLOBAL
            // first head qh_start maps to the right kv-head of the whole cache.
            // head_range == 0 ⇒ whole-op (all num_q heads, base 0).
            device T_act* output = (device T_act*)(operands[ins.z + 0u]);
            const device T_act* q = (const device T_act*)(operands[ins.z + 1u]);
            const device uint* seq_used_k = (const device uint*)(operands[ins.z + 2u]);
            const device uint* block_table = (const device uint*)(operands[ins.z + 3u]);
            const device T_act* k_cache = (const device T_act*)(operands[ins.z + 4u]);
            const device T_act* v_cache = (const device T_act*)(operands[ins.z + 5u]);
            uint head_dim = shapes[sb + 1u];
            uint num_q = shapes[sb + 2u];
            uint num_kv = shapes[sb + 3u];
            float scale = as_type<float>(shapes[sb + 4u]);
            uint block_size = shapes[sb + 5u];
            uint max_blocks = shapes[sb + 6u];
            uint head_range = shapes[sb + 7u];
            uint qh_start = head_range >> 16u;
            uint qh_count = head_range & 0xFFFFu;
            if (head_range == 0u) { // whole-op fallback
              qh_start = 0u;
              qh_count = num_q;
            }
            for (uint local = 0u; local < qh_count; local++) {
              mittens::attention_decode_impl<T_act>(
                  output, q, seq_used_k, block_table, k_cache, v_cache,
                  tg_outputs, tg_max, tg_sum,
                  head_dim, num_q, num_kv, scale, block_size, max_blocks,
                  /*seq_idx=*/0u, /*q_head_idx=*/local, simd_gid, simd_lid,
                  /*q_head_base=*/qh_start);
              threadgroup_barrier(mem_flags::mem_threadgroup); // scratch reuse across heads
            }
            break;
          }
          case WL_OP_ROPE_APPEND: {
            // The oracle's rope_append, K side: rotate K in place, then write
            // rotated K + un-rotated V into the paged cache at slot_mapping[0],
            // so attention reads the new token from the cache (Tier-B exact).
            // operands [k(in/out), cos_sin, positions, v, kv_cache_k,
            // kv_cache_v, slot_mapping]; shape (ROPE_APPEND, head_dim,
            // num_kv_block, rot_dim, block_size, num_kv_global, kvh_start).
            // `cos_sin` is the WHOLE rotary table, position = `positions[0]`
            // (runtime). HEAD-RANGE: this block ropes its `num_kv_block` local
            // heads (k/v operands are block-based) and writes them to the GLOBAL
            // cache heads `kvh_start + h` with `num_kv_global` as the cache
            // stride — so a head-tiled (partitioned) K-rope lands in the right
            // cache slots. num_kv_global == 0 ⇒ whole-op (== num_kv_block, base 0).
            device T_act* k = (device T_act*)(operands[ins.z + 0u]);
            const device T_act* cos_sin = (const device T_act*)(operands[ins.z + 1u]);
            const device uint* positions = (const device uint*)(operands[ins.z + 2u]);
            const device T_act* v = (const device T_act*)(operands[ins.z + 3u]);
            device T_act* kv_cache_k = (device T_act*)(operands[ins.z + 4u]);
            device T_act* kv_cache_v = (device T_act*)(operands[ins.z + 5u]);
            const device uint* slot_mapping = (const device uint*)(operands[ins.z + 6u]);
            uint head_dim = shapes[sb + 1u];
            uint num_kv = shapes[sb + 2u]; // heads THIS block writes
            uint rot_dim = shapes[sb + 3u];
            uint block_size = shapes[sb + 4u];
            uint num_kv_global = shapes[sb + 5u];
            uint kvh_start = shapes[sb + 6u];
            if (num_kv_global == 0u) { // whole-op fallback
              num_kv_global = num_kv;
              kvh_start = 0u;
            }
            uint half_dim = rot_dim / 2u;
            uint pos = positions[0];
            const device T_act* cos_row = cos_sin + pos * rot_dim;
            const device T_act* sin_row = cos_sin + pos * rot_dim + half_dim;
            // Rotate K in place: each thread owns (local kv_head, d<half) pairs.
            for (uint idx = tid_in_tg; idx < num_kv * half_dim; idx += threads_per_tg.x) {
              mittens::rope_rotate_pair<T_act>(
                  k + (idx / half_dim) * head_dim, cos_row, sin_row, idx % half_dim, half_dim);
            }
            // K must be fully rotated before the paged write reads it (a d>=half
            // element was written by the d-half thread during rotation).
            threadgroup_barrier(mem_flags::mem_device);
            uint slot = slot_mapping[0];
            for (uint idx = tid_in_tg; idx < num_kv * head_dim; idx += threads_per_tg.x) {
              uint h = idx / head_dim; // local head in this block
              mittens::kv_paged_write<T_act>(
                  kv_cache_k, kv_cache_v, k + h * head_dim, v + h * head_dim,
                  slot, kvh_start + h, idx % head_dim, num_kv_global, block_size, head_dim);
            }
            break;
          }
        }
        // No blind per-op barrier: intra-worker visibility is fenced by an
        // explicit WL_OPC_BARRIER the compiler placed at the real RAW
        // boundaries (and Signal/Wait carry their own fences). The player is a
        // dumb executor — it never invents sync.
        break;
      }
      case WL_OPC_BARRIER:
        // Compiler-placed intra-worker fence at a real RAW edge between two
        // ops on this same tape (same TG). Data flow here is one TG writing
        // then one TG reading — intra-TG visibility. EXPERIMENTAL: use
        // mem_threadgroup (cheaper than mem_device); if Apple's L1 is coherent
        // across simdgroups in practice this is correct. Coherence check =
        // Tier-B token-stream match. If it fails, the real fix is moving the
        // chain-local arena slots out of MTLBuffer into threadgroup memory
        // (the [[feedback_persistent_decode_objective]] thesis) and keeping
        // mem_threadgroup for those.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        break;
      case WL_OPC_SIGNAL:
        // Data-before-flag fence (relaxed-only MSL device atomics ⇒ the barrier,
        // not a release order, provides the producer's data visibility), then
        // publish this worker's flag from one thread.
        threadgroup_barrier(mem_flags::mem_device);
        if (tid_in_tg == 0u) {
          mittens::wf_signal(flags, ins.w);
        }
        break;
      case WL_OPC_WAIT:
        // Spin on the producer's flag (one thread), then barrier so every lane
        // observes the cross-worker dependency satisfied before proceeding.
        if (tid_in_tg == 0u) {
          mittens::wf_wait(flags, ins.w);
        }
        threadgroup_barrier(mem_flags::mem_device);
        break;
    }
  }
}

#define INST_WL_PLAYER(act_tag, act_type, scale_tag, scale_type, gs)        \
  template [[host_name("wavefront_player_" #act_tag "_s_" #scale_tag         \
                       "_gs_" #gs "_b_4")]] [[kernel]]                       \
  decltype(wavefront_player<act_type, scale_type, gs, 4>)                   \
      wavefront_player<act_type, scale_type, gs, 4>;
INST_WL_PLAYER(bf16, bfloat, f16, half, 64)
INST_WL_PLAYER(f16, half, f16, half, 64)

// ── per-subtile player: many small TGs, each processes ONE subtile ───
//
// One TG per subtile of work. The dispatch grid sizes itself to the level's
// subtile count (the host driver iterates DAG levels and dispatches this
// kernel once per level). Smaller TG (256 threads = 8 simdgroups) lets Apple
// fit ~4 TGs per core → 40 TGs concurrent across 10 cores = 320 simdgroups
// in flight, matching per-op's grid behaviour without any per-worker tape
// machinery. Each TG reads its instruction from `level_tape[level_start +
// tgpos.x]` and dispatches the arm exactly like the persistent player does.
//
// TG-wide arms (rmsnorm/attn/publish/acquire/rope_append) get a 256-thread
// pass via the impl's total_threads parameter; the simdgroup-distributed
// arms (qmv/silu_mul/add/rope) use the TG's 8 simdgroups (sg_count = 8).
template <typename T_act, typename T_scale, int group_size, int bits>
[[kernel]] void wavefront_player_per_subtile(
    device const uint4* level_tape [[buffer(0)]],
    device const uint* shapes [[buffer(1)]],
    device const ulong* operands [[buffer(2)]],
    device atomic_uint* flags [[buffer(3)]],
    constant uint& level_start [[buffer(4)]],
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  // Per-TG threadgroup scratch sized for 256 threads (8 simdgroups).
  threadgroup float shared_sum[256];
  threadgroup float tg_outputs[256];
  threadgroup float tg_max[8];
  threadgroup float tg_sum[8];

  const uint pc = level_start + tgpos.x;
  uint4 ins = level_tape[pc];
  // level_tape contains only COMPUTE entries; ins.x is always WL_OPC_COMPUTE.
  const uint sb = ins.y * WL_SHAPE_STRIDE;
  const uint op = shapes[sb + 0u];
  // All 8 simdgroups participate in every op (no tranche packing in the per-
  // subtile kernel — one subtile per TG, full TG budget).
  const uint sg_count = 8u;
  const uint local_sg = simd_gid;
  const uint op_threads = sg_count << 5u; // 256
  const uint local_tid = (local_sg << 5u) + simd_lid;
  switch (op) {
    case WL_OP_QMV: {
      uint row_vec = shapes[sb + 3u];
      if (row_vec == 0u) row_vec = shapes[sb + 1u];
      wl_qmv_arm<T_act, T_scale, group_size, bits>(
          operands, ins.z, shapes[sb + 1u], shapes[sb + 2u],
          local_sg, sg_count, simd_lid, row_vec);
      break;
    }
    case WL_OP_QMV_COH:
      wl_qmv_coh_arm<T_act, T_scale, group_size, bits>(
          operands, ins.z, shapes[sb + 1u], shapes[sb + 2u],
          local_sg, sg_count, simd_lid);
      break;
    case WL_OP_QMV_QUAD: {
      uint k_dim = shapes[sb + 1u];
      uint n_out = shapes[sb + 2u];
      uint row_vec = shapes[sb + 3u];
      if (row_vec == 0u) row_vec = k_dim;
      if (k_dim == 64u) {
        wl_qmv_quad_arm<T_act, T_scale, group_size, bits, 64>(
            operands, ins.z, k_dim, n_out, local_sg, sg_count, simd_lid, row_vec);
      } else if (k_dim == 128u) {
        wl_qmv_quad_arm<T_act, T_scale, group_size, bits, 128>(
            operands, ins.z, k_dim, n_out, local_sg, sg_count, simd_lid, row_vec);
      } else if (k_dim == 256u) {
        wl_qmv_quad_arm<T_act, T_scale, group_size, bits, 256>(
            operands, ins.z, k_dim, n_out, local_sg, sg_count, simd_lid, row_vec);
      }
      break;
    }
    case WL_OP_RMSNORM:
      mittens::rmsnorm_impl<T_act, T_scale>(
          (device T_act*)(operands[ins.z + 0u]),
          (const device T_act*)(operands[ins.z + 1u]),
          (const device T_scale*)(operands[ins.z + 2u]),
          shared_sum, 1u, shapes[sb + 1u], as_type<float>(shapes[sb + 2u]),
          0u, tid_in_tg, 256u);
      break;
    case WL_OP_SILU_MUL: {
      uint n = shapes[sb + 1u];
      for (uint e = local_tid; e < n; e += op_threads) {
        mittens::silu_mul_impl<T_act>(
            (device T_act*)(operands[ins.z + 0u]),
            (const device T_act*)(operands[ins.z + 1u]),
            (const device T_act*)(operands[ins.z + 2u]),
            e, n);
      }
      break;
    }
    case WL_OP_ADD: {
      uint n = shapes[sb + 1u];
      for (uint e = local_tid; e < n; e += op_threads) {
        mittens::add_impl<T_act>(
            (device T_act*)(operands[ins.z + 0u]),
            (const device T_act*)(operands[ins.z + 1u]),
            (const device T_act*)(operands[ins.z + 2u]),
            e, n);
      }
      break;
    }
    case WL_OP_SUM_REDUCE: {
      uint n = shapes[sb + 1u];
      uint num_partials = shapes[sb + 2u];
      for (uint e = local_tid; e < n; e += op_threads) {
        mittens::sum_reduce_impl<T_act>(
            (device T_act*)(operands[ins.z + 0u]),
            operands + ins.z + 1u, num_partials, e, n);
      }
      break;
    }
    case WL_OP_ROPE: {
      device T_act* x = (device T_act*)(operands[ins.z + 0u]);
      const device T_act* cos_sin = (const device T_act*)(operands[ins.z + 1u]);
      const device uint* positions = (const device uint*)(operands[ins.z + 2u]);
      uint head_dim = shapes[sb + 1u];
      uint num_heads = shapes[sb + 2u];
      uint rot_dim = shapes[sb + 3u];
      uint half_dim = rot_dim / 2u;
      uint pos = positions[0];
      const device T_act* cos_row = cos_sin + pos * rot_dim;
      const device T_act* sin_row = cos_sin + pos * rot_dim + half_dim;
      for (uint idx = local_tid; idx < num_heads * half_dim; idx += op_threads) {
        uint h = idx / half_dim;
        uint d = idx % half_dim;
        mittens::rope_rotate_pair<T_act>(x + h * head_dim, cos_row, sin_row, d, half_dim);
      }
      break;
    }
    // The four big TG-wide ops (PUBLISH/ACQUIRE/ROPE_APPEND/ATTN) NEED 1024-
    // thread cooperation. They cannot run safely at 256 threads. The level
    // groups that contain them must be dispatched via the persistent
    // `wavefront_player` instead — the host scheduler routes accordingly.
    // (Default: subtile is no-op so an accidentally-dispatched non-supported
    // op falls through silently rather than corrupting memory.)
    default:
      break;
  }
  (void)flags;
  (void)tg_outputs;
  (void)tg_max;
  (void)tg_sum;
}

#define INST_WL_PLAYER_PS(act_tag, act_type, scale_tag, scale_type, gs)         \
  template [[host_name("wavefront_player_per_subtile_" #act_tag "_s_"           \
                       #scale_tag "_gs_" #gs "_b_4")]] [[kernel]]               \
  decltype(wavefront_player_per_subtile<act_type, scale_type, gs, 4>)           \
      wavefront_player_per_subtile<act_type, scale_type, gs, 4>;
INST_WL_PLAYER_PS(bf16, bfloat, f16, half, 64)
INST_WL_PLAYER_PS(f16, half, f16, half, 64)
