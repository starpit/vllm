// SPDX-License-Identifier: Apache-2.0
//! Fused Q/K/V projection + NeoX-style RoPE + paged KV-cache write.
//!
//! Phase 1: dense (BF16 / F16) decode (M small). Replaces the
//! `Q_proj + K_proj + V_proj + RopeAppend` four-dispatch sequence
//! with a single launch. Mirrors the CUDA `FusedQkvRopeCacheImpl`
//! (`ferrite-forward-macro/src/impl_lib.rs:6794`) functionally; the
//! prefill (M >= 2) and affine-int4 variants land separately per
//! `project_metal_fused_qkv_handoff`.
//!
//! Function constants (must match
//! `interpreter::metal::lowering::lower_one` for
//! `Instruction::FusedQkvRopeCache`):
//!   0 = HIDDEN
//!   1 = NUM_Q_HEADS
//!   2 = NUM_KV_HEADS
//!   3 = HEAD_DIM
//!   4 = ROT_DIM     (typically == HEAD_DIM)
//!   5 = BLOCK_SIZE  (paged KV cache page size)
//!   6 = M           (token count for this bucket; 1 for decode)
//!
//! Bindings:
//!   buffer(0) = q_out         [M, NUM_Q_HEADS  * HEAD_DIM]   T_act, written
//!   buffer(1) = input         [M, HIDDEN]                    T_act, read
//!   buffer(2) = weight_packed [(NUM_Q + 2*NUM_KV)*HEAD_DIM, HIDDEN]
//!                             T_act, row-major (concat [Q | K | V])
//!   buffer(3) = cos_sin       [max_pos, ROT_DIM] — row = [cos | sin]
//!                             T_act
//!   buffer(4) = positions     [M] (u32)
//!   buffer(5) = slot_mapping  [M] (u32; sentinel 0xFFFFFFFF skips write)
//!   buffer(6) = kv_cache_k    [num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
//!                             T_act
//!   buffer(7) = kv_cache_v    same shape as kv_cache_k
//!
//! Dispatch:
//!   threadgroups: (M, NUM_Q_HEADS + 2 * NUM_KV_HEADS, 1)
//!   threads_per_threadgroup: (HEAD_DIM, 1, 1)
//!
//! Per (token t, output_head h) threadgroup: each thread `d` in
//! `[0, HEAD_DIM)` computes `acc = dot(W[(h*HEAD_DIM + d), :], input[t, :])`.
//! Then branches on `h`:
//!   - h <  NUM_Q                   → Q head. Stash acc in tg memory,
//!                                    barrier, rotate (d, d+half) pair,
//!                                    write to q_out.
//!   - h <  NUM_Q + NUM_KV          → K head. Same rotation, write to
//!                                    kv_cache_k via paged indexing.
//!   - else                         → V head. Pass-through write to
//!                                    kv_cache_v.
//!
//! Padding lanes (slot_mapping[t] == 0xFFFFFFFF) skip the paged write
//! to match `rope_append_*_specialized`. The Q output is always
//! written — its slot is arena-scoped and downstream attention masks
//! out the padded queries.

#include <metal_stdlib>
using namespace metal;

constant uint FQRC_HIDDEN       [[function_constant(0)]];
constant uint FQRC_NUM_Q_HEADS  [[function_constant(1)]];
constant uint FQRC_NUM_KV_HEADS [[function_constant(2)]];
constant uint FQRC_HEAD_DIM     [[function_constant(3)]];
constant uint FQRC_ROT_DIM      [[function_constant(4)]];
constant uint FQRC_BLOCK_SIZE   [[function_constant(5)]];
constant uint FQRC_M            [[function_constant(6)]];

// Threadgroup scratch ceiling. Llama-3.x: HEAD_DIM <= 128; Qwen2-7B,
// Mistral-7B: 128. 256 leaves headroom without bloating tg memory.
constant constexpr uint FQRC_HEAD_DIM_MAX = 256;

template <typename T>
[[kernel]] void fused_qkv_rope_cache_impl(
    device       T*    q_out         [[buffer(0)]],
    device const T*    input         [[buffer(1)]],
    device const T*    weight_packed [[buffer(2)]],
    device const T*    cos_sin       [[buffer(3)]],
    device const uint* positions     [[buffer(4)]],
    device const uint* slot_mapping  [[buffer(5)]],
    device       T*    kv_cache_k    [[buffer(6)]],
    device       T*    kv_cache_v    [[buffer(7)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid    [[thread_position_in_threadgroup]])
{
    const uint t        = tg_pos.x;
    const uint head     = tg_pos.y;   // 0..NUM_Q + 2*NUM_KV
    const uint d        = tid.x;
    const uint head_dim = FQRC_HEAD_DIM;
    const uint num_q    = FQRC_NUM_Q_HEADS;
    const uint num_kv   = FQRC_NUM_KV_HEADS;
    const uint rot_dim  = FQRC_ROT_DIM;
    const uint half_dim = rot_dim / 2;
    const uint hidden   = FQRC_HIDDEN;
    const uint block_sz = FQRC_BLOCK_SIZE;
    const uint num_heads_total = num_q + 2u * num_kv;

    if (t >= FQRC_M || head >= num_heads_total || d >= head_dim) return;

    // ── Matmul: acc = dot(W[out_row, :], input[t, :]) ─────────────
    const uint out_row = head * head_dim + d;
    device const T* wrow = weight_packed + (size_t)out_row * (size_t)hidden;
    device const T* xrow = input + (size_t)t * (size_t)hidden;
    float acc = 0.0f;
    for (uint k = 0; k < hidden; ++k) {
        acc += float(wrow[k]) * float(xrow[k]);
    }

    // ── Branch on band: Q / K / V ─────────────────────────────────
    const uint kQ_END = num_q;
    const uint kK_END = num_q + num_kv;

    if (head < kQ_END) {
        // Q head: stash in tg memory, barrier, rotate, write to q_out.
        threadgroup float qbuf[FQRC_HEAD_DIM_MAX];
        qbuf[d] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device T* q_row_ptr = q_out
            + (size_t)t    * (size_t)(num_q * head_dim)
            + (size_t)head * (size_t)head_dim;

        if (d < half_dim) {
            const uint pos = positions[t];
            device const T* cos_row = cos_sin + (size_t)pos * (size_t)rot_dim;
            device const T* sin_row = cos_sin + (size_t)pos * (size_t)rot_dim + half_dim;
            const float c  = float(cos_row[d]);
            const float s  = float(sin_row[d]);
            const float x0 = qbuf[d];
            const float x1 = qbuf[half_dim + d];
            q_row_ptr[d]            = T(x0 * c - x1 * s);
            q_row_ptr[half_dim + d] = T(x1 * c + x0 * s);
        } else if (d >= rot_dim) {
            // Partial-rope tail [rot_dim, head_dim): pass-through.
            q_row_ptr[d] = T(qbuf[d]);
        }
        // d in [half_dim, rot_dim): written by the d<half_dim branch.
        return;
    }

    if (head < kK_END) {
        // K head: stash in tg memory, barrier, rotate, paged-write.
        const uint kv_head = head - num_q;
        threadgroup float kbuf[FQRC_HEAD_DIM_MAX];
        kbuf[d] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Sentinel 0xFFFFFFFF (padding lane) → skip cache write.
        // Matches `rope_append_*_specialized` to avoid padding's
        // K_proj(token 0) corrupting slot 0.
        const uint slot = slot_mapping[t];
        if (slot == 0xFFFFFFFFu) return;

        const uint block_id     = slot / block_sz;
        const uint block_offset = slot % block_sz;
        const uint kv_blk_stride  = num_kv * block_sz * head_dim;
        const uint kv_head_stride = block_sz * head_dim;
        const uint kv_tok_stride  = head_dim;
        device T* k_dst = kv_cache_k
            + (size_t)block_id     * (size_t)kv_blk_stride
            + (size_t)kv_head      * (size_t)kv_head_stride
            + (size_t)block_offset * (size_t)kv_tok_stride;

        if (d < half_dim) {
            const uint pos = positions[t];
            device const T* cos_row = cos_sin + (size_t)pos * (size_t)rot_dim;
            device const T* sin_row = cos_sin + (size_t)pos * (size_t)rot_dim + half_dim;
            const float c  = float(cos_row[d]);
            const float s  = float(sin_row[d]);
            const float x0 = kbuf[d];
            const float x1 = kbuf[half_dim + d];
            k_dst[d]            = T(x0 * c - x1 * s);
            k_dst[half_dim + d] = T(x1 * c + x0 * s);
        } else if (d >= rot_dim) {
            k_dst[d] = T(kbuf[d]);
        }
        return;
    }

    // V head: pass-through to kv_cache_v.
    const uint kv_head = head - kK_END;
    const uint slot = slot_mapping[t];
    if (slot == 0xFFFFFFFFu) return;

    const uint block_id     = slot / block_sz;
    const uint block_offset = slot % block_sz;
    const uint kv_blk_stride  = num_kv * block_sz * head_dim;
    const uint kv_head_stride = block_sz * head_dim;
    const uint kv_tok_stride  = head_dim;
    device T* v_dst = kv_cache_v
        + (size_t)block_id     * (size_t)kv_blk_stride
        + (size_t)kv_head      * (size_t)kv_head_stride
        + (size_t)block_offset * (size_t)kv_tok_stride;
    v_dst[d] = T(acc);
}

#define INST_FQRC(act_tag, act_type)                                     \
  template [[host_name("fused_qkv_rope_cache_" #act_tag "_specialized")]] \
  [[kernel]] decltype(fused_qkv_rope_cache_impl<act_type>)               \
      fused_qkv_rope_cache_impl<act_type>;

INST_FQRC(f16,  half)
INST_FQRC(bf16, bfloat)
