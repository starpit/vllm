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
      w, scales, biases, x, y, WL_K, WL_N, tid, simd_gid, simd_lid);
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

// QMV shape-class arm in the fixed 1024-thread TG: the 32 simdgroups cover 16
// of `qmv_fast`'s 8-row groups per wave (gi = simd_gid>>1 selects the group,
// role = simd_gid&1 is qmv_fast's simd_gid in {0,1}), so each group's two
// simdgroups replay the whole kernel's 2-simdgroup compute exactly ⇒ bit-exact.
template <typename T_act, typename T_scale, int group_size, int bits>
METAL_FUNC void wl_qmv_arm(
    const device ulong* operands,
    uint operand_base,
    uint k,
    uint n,
    uint simd_gid,
    uint simd_lid) {
  const device uint32_t* w = (const device uint32_t*)(operands[operand_base + 0u]);
  const device T_scale* s = (const device T_scale*)(operands[operand_base + 1u]);
  const device T_scale* b = (const device T_scale*)(operands[operand_base + 2u]);
  const device T_act* x = (const device T_act*)(operands[operand_base + 3u]);
  device T_act* y = (device T_act*)(operands[operand_base + 4u]);
  const uint num_groups = (n + 7u) / 8u;
  const uint gi = simd_gid >> 1;
  const uint role = simd_gid & 1u;
  for (uint base = 0u; base < num_groups; base += 16u) {
    uint g = base + gi;
    if (g < num_groups) {
      mittens::qmv_fast_impl<T_act, T_scale, group_size, bits>(
          w, s, b, x, y, int(k), int(n), uint3(0u, g, 0u), role, simd_lid);
    }
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
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint me = tgpos.x;
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
        switch (op) {
          case WL_OP_QMV:
            wl_qmv_arm<T_act, T_scale, group_size, bits>(
                operands, ins.z, shapes[sb + 1u], shapes[sb + 2u], simd_gid, simd_lid);
            break;
          case WL_OP_PUBLISH:
            // pack operands[base+1] (this worker's written region, bf16) into the
            // coherent operands[base+0] (atomic u32). pair0=p1, n_pairs=p2. All
            // 1024 lanes co-operate (strided); the trailing per-op mem_device
            // barrier fences the stripes before the Signal.
            mittens::wf_publish_pairs_tg<T_act>(
                (device atomic_uint*)(operands[ins.z + 0u]),
                (const device T_act*)(operands[ins.z + 1u]),
                shapes[sb + 1u], shapes[sb + 2u], tid_in_tg, 1024u);
            break;
          case WL_OP_ACQUIRE:
            // load+unpack the coherent operands[base+1] (atomic u32) into the
            // private operands[base+0] (bf16). n_pairs=p1. All 1024 lanes
            // co-operate (strided); the trailing per-op mem_device barrier
            // fences the private copy before the consumer reads it.
            mittens::wf_acquire_pairs_tg<T_act>(
                (device T_act*)(operands[ins.z + 0u]),
                (const device atomic_uint*)(operands[ins.z + 1u]),
                shapes[sb + 1u], tid_in_tg, 1024u);
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
                0u, tid_in_tg, 1024u);
            break;
          case WL_OP_SILU_MUL: {
            // operands [out, gate, up]; shape (SILU_MUL, n, ...). The 1024
            // threads grid-stride the n elements (per-element pure ⇒ bit-exact).
            uint n = shapes[sb + 1u];
            for (uint e = tid_in_tg; e < n; e += 1024u) {
              mittens::silu_mul_impl<T_act>(
                  (device T_act*)(operands[ins.z + 0u]),
                  (const device T_act*)(operands[ins.z + 1u]),
                  (const device T_act*)(operands[ins.z + 2u]),
                  e, n);
            }
            break;
          }
          case WL_OP_ADD: {
            // operands [out, a, b]; shape (ADD, n, ...). Residual add; out may
            // alias a or b (in-place residual) — each element is independent.
            uint n = shapes[sb + 1u];
            for (uint e = tid_in_tg; e < n; e += 1024u) {
              mittens::add_impl<T_act>(
                  (device T_act*)(operands[ins.z + 0u]),
                  (const device T_act*)(operands[ins.z + 1u]),
                  (const device T_act*)(operands[ins.z + 2u]),
                  e, n);
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
            for (uint idx = tid_in_tg; idx < num_heads * half_dim; idx += 1024u) {
              uint h = idx / half_dim;
              uint d = idx % half_dim;
              mittens::rope_rotate_pair<T_act>(x + h * head_dim, cos_row, sin_row, d, half_dim);
            }
            break;
          }
          case WL_OP_ATTN: {
            // operands [output, q, seq_used_k, block_table, k_cache, v_cache];
            // shape (ATTN, head_dim, num_q, num_kv, scale_bits, block_size, max_blocks).
            // One whole-TG attention per (seq 0, q_head); loop the q-heads.
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
            for (uint h = 0u; h < num_q; h++) {
              mittens::attention_decode_impl<T_act>(
                  output, q, seq_used_k, block_table, k_cache, v_cache,
                  tg_outputs, tg_max, tg_sum,
                  head_dim, num_q, num_kv, scale, block_size, max_blocks,
                  /*seq_idx=*/0u, /*q_head_idx=*/h, simd_gid, simd_lid);
              threadgroup_barrier(mem_flags::mem_threadgroup); // scratch reuse across heads
            }
            break;
          }
          case WL_OP_ROPE_APPEND: {
            // The oracle's rope_append, K side: rotate K in place, then write
            // rotated K + un-rotated V into the paged cache at slot_mapping[0],
            // so attention reads the new token from the cache (Tier-B exact).
            // operands [k(in/out), cos_sin, positions, v, kv_cache_k,
            // kv_cache_v, slot_mapping]; shape (ROPE_APPEND, head_dim, num_kv,
            // rot_dim, block_size, ...). `cos_sin` is the WHOLE rotary table and
            // the live decode position is `positions[0]` — indexed at runtime
            // exactly like the oracle (`cos_sin + pos*rot_dim`), never a baked
            // offset (the position changes every decode step).
            device T_act* k = (device T_act*)(operands[ins.z + 0u]);
            const device T_act* cos_sin = (const device T_act*)(operands[ins.z + 1u]);
            const device uint* positions = (const device uint*)(operands[ins.z + 2u]);
            const device T_act* v = (const device T_act*)(operands[ins.z + 3u]);
            device T_act* kv_cache_k = (device T_act*)(operands[ins.z + 4u]);
            device T_act* kv_cache_v = (device T_act*)(operands[ins.z + 5u]);
            const device uint* slot_mapping = (const device uint*)(operands[ins.z + 6u]);
            uint head_dim = shapes[sb + 1u];
            uint num_kv = shapes[sb + 2u];
            uint rot_dim = shapes[sb + 3u];
            uint block_size = shapes[sb + 4u];
            uint half_dim = rot_dim / 2u;
            uint pos = positions[0];
            const device T_act* cos_row = cos_sin + pos * rot_dim;
            const device T_act* sin_row = cos_sin + pos * rot_dim + half_dim;
            // Rotate K in place: each thread owns (kv_head, d<half) pairs.
            for (uint idx = tid_in_tg; idx < num_kv * half_dim; idx += 1024u) {
              mittens::rope_rotate_pair<T_act>(
                  k + (idx / half_dim) * head_dim, cos_row, sin_row, idx % half_dim, half_dim);
            }
            // K must be fully rotated before the paged write reads it (a d>=half
            // element was written by the d-half thread during rotation).
            threadgroup_barrier(mem_flags::mem_device);
            uint slot = slot_mapping[0];
            for (uint idx = tid_in_tg; idx < num_kv * head_dim; idx += 1024u) {
              uint h = idx / head_dim;
              mittens::kv_paged_write<T_act>(
                  kv_cache_k, kv_cache_v, k + h * head_dim, v + h * head_dim,
                  slot, h, idx % head_dim, num_kv, block_size, head_dim);
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
        // Compiler-placed intra-worker device fence (a dependence boundary —
        // the tranche boundary). Uniform across the TG (one tape per worker).
        threadgroup_barrier(mem_flags::mem_device);
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
