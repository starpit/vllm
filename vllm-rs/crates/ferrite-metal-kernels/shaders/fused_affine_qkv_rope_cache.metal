// SPDX-License-Identifier: Apache-2.0
//! Fused affine-int4 Q/K/V projection + NeoX RoPE + paged KV-cache write.
//!
//! Phase 3 affine companion to `fused_qkv_rope_cache.metal`. Replaces
//! the three `AffineQmv` dispatches + `RopeAppend` in the decode loop
//! with a single launch. Decode-only (M small); prefill lands in a
//! follow-up.
//!
//! Layout (post-`load_dense_concat_packed`-style packed-qkv loader):
//!   weight_packed : [N_OUT, HIDDEN / 8] u32      (4-bit nibbles)
//!   scales        : [N_OUT, HIDDEN / GROUP_SIZE] T_scale
//!   biases        : [N_OUT, HIDDEN / GROUP_SIZE] T_scale
//! where `N_OUT = (NUM_Q + 2*NUM_KV) * HEAD_DIM`. Q/K/V weights are
//! concatenated along the output axis.
//!
//! Function constants:
//!   0 = HIDDEN
//!   1 = NUM_Q_HEADS
//!   2 = NUM_KV_HEADS
//!   3 = HEAD_DIM
//!   4 = ROT_DIM
//!   5 = BLOCK_SIZE
//!   6 = M
//!   7 = GROUP_SIZE
//!
//! Bindings:
//!   buffer(0) = q_out         [M, NUM_Q * HEAD_DIM]            T_act
//!   buffer(1) = input         [M, HIDDEN]                       T_act
//!   buffer(2) = weight_packed [N_OUT, HIDDEN/8]                 u32
//!   buffer(3) = scales        [N_OUT, HIDDEN/GROUP_SIZE]        T_scale
//!   buffer(4) = biases        [N_OUT, HIDDEN/GROUP_SIZE]        T_scale
//!   buffer(5) = cos_sin       [max_pos, ROT_DIM]                T_act
//!   buffer(6) = positions     [M] u32
//!   buffer(7) = slot_mapping  [M] u32 (sentinel 0xFFFFFFFF skips write)
//!   buffer(8) = kv_cache_k    [num_blocks, NUM_KV, BLOCK_SIZE, HEAD_DIM]
//!   buffer(9) = kv_cache_v    same
//!
//! Dispatch:
//!   threadgroups: (M, NUM_Q + 2*NUM_KV, 1)
//!   threads_per_threadgroup: (HEAD_DIM, 1, 1)
//!
//! Each thread d in a (token, output_head) threadgroup walks K in
//! GROUP_SIZE-sized blocks, loading one (scale, bias) pair per group
//! and 4-bit nibbles from the packed u32 stream. Same Q/K/V band
//! epilogue as `fused_qkv_rope_cache.metal`.

#include <metal_stdlib>
using namespace metal;

constant uint FAQRC_HIDDEN       [[function_constant(0)]];
constant uint FAQRC_NUM_Q_HEADS  [[function_constant(1)]];
constant uint FAQRC_NUM_KV_HEADS [[function_constant(2)]];
constant uint FAQRC_HEAD_DIM     [[function_constant(3)]];
constant uint FAQRC_ROT_DIM      [[function_constant(4)]];
constant uint FAQRC_BLOCK_SIZE   [[function_constant(5)]];
constant uint FAQRC_M            [[function_constant(6)]];
constant uint FAQRC_GROUP_SIZE   [[function_constant(7)]];

constant constexpr uint FAQRC_HEAD_DIM_MAX = 256;

template <typename T_act, typename T_scale>
[[kernel]] void fused_affine_qkv_rope_cache_impl(
    device       T_act*   q_out         [[buffer(0)]],
    device const T_act*   input         [[buffer(1)]],
    device const uint*    weight_packed [[buffer(2)]],
    device const T_scale* scales        [[buffer(3)]],
    device const T_scale* biases        [[buffer(4)]],
    device const T_act*   cos_sin       [[buffer(5)]],
    device const uint*    positions     [[buffer(6)]],
    device const uint*    slot_mapping  [[buffer(7)]],
    device       T_act*   kv_cache_k    [[buffer(8)]],
    device       T_act*   kv_cache_v    [[buffer(9)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid    [[thread_position_in_threadgroup]])
{
    const uint t        = tg_pos.x;
    const uint head     = tg_pos.y;
    const uint d        = tid.x;
    const uint head_dim = FAQRC_HEAD_DIM;
    const uint num_q    = FAQRC_NUM_Q_HEADS;
    const uint num_kv   = FAQRC_NUM_KV_HEADS;
    const uint rot_dim  = FAQRC_ROT_DIM;
    const uint half_dim = rot_dim / 2;
    const uint hidden   = FAQRC_HIDDEN;
    const uint block_sz = FAQRC_BLOCK_SIZE;
    const uint gs       = FAQRC_GROUP_SIZE;
    const uint num_heads_total = num_q + 2u * num_kv;

    if (t >= FAQRC_M || head >= num_heads_total || d >= head_dim) return;

    // ── Matmul: acc = sum_k dequant(weight, scale, bias)[out_row, k] * input[t, k]
    // Walk K in `gs`-sized groups. 4-bit packing: 8 nibbles per u32.
    const uint out_row   = head * head_dim + d;
    const uint k_packed_stride = hidden / 8u;
    const uint group_stride    = hidden / gs;
    device const uint*    wrow = weight_packed + (size_t)out_row * (size_t)k_packed_stride;
    device const T_scale* srow = scales        + (size_t)out_row * (size_t)group_stride;
    device const T_scale* brow = biases        + (size_t)out_row * (size_t)group_stride;
    device const T_act*   xrow = input + (size_t)t * (size_t)hidden;

    float acc = 0.0f;
    const uint num_groups = hidden / gs;
    for (uint g = 0; g < num_groups; ++g) {
        const float s = float(srow[g]);
        const float b = float(brow[g]);
        // Each group is `gs` consecutive K values: gs/8 u32 words.
        const uint k0 = g * gs;
        const uint words = gs / 8u;
        for (uint w = 0; w < words; ++w) {
            const uint packed = wrow[(k0 / 8u) + w];
            const uint kbase  = k0 + w * 8u;
            // Unpack 8 nibbles (low to high). MLX layout: nibble i lives
            // at bits [4i, 4i+4).
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const float q = float((packed >> (i * 4u)) & 0xFu);
                const float wv = s * q + b;
                acc += wv * float(xrow[kbase + i]);
            }
        }
    }

    // ── Q/K/V band epilogue (mirrors fused_qkv_rope_cache.metal) ───
    const uint kQ_END = num_q;
    const uint kK_END = num_q + num_kv;

    if (head < kQ_END) {
        threadgroup float qbuf[FAQRC_HEAD_DIM_MAX];
        qbuf[d] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device T_act* q_row_ptr = q_out
            + (size_t)t    * (size_t)(num_q * head_dim)
            + (size_t)head * (size_t)head_dim;

        if (d < half_dim) {
            const uint pos = positions[t];
            device const T_act* cos_row = cos_sin + (size_t)pos * (size_t)rot_dim;
            device const T_act* sin_row = cos_sin + (size_t)pos * (size_t)rot_dim + half_dim;
            const float c  = float(cos_row[d]);
            const float s  = float(sin_row[d]);
            const float x0 = qbuf[d];
            const float x1 = qbuf[half_dim + d];
            q_row_ptr[d]            = T_act(x0 * c - x1 * s);
            q_row_ptr[half_dim + d] = T_act(x1 * c + x0 * s);
        } else if (d >= rot_dim) {
            q_row_ptr[d] = T_act(qbuf[d]);
        }
        return;
    }

    if (head < kK_END) {
        const uint kv_head = head - num_q;
        threadgroup float kbuf[FAQRC_HEAD_DIM_MAX];
        kbuf[d] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint slot = slot_mapping[t];
        if (slot == 0xFFFFFFFFu) return;

        const uint block_id     = slot / block_sz;
        const uint block_offset = slot % block_sz;
        const uint kv_blk_stride  = num_kv * block_sz * head_dim;
        const uint kv_head_stride = block_sz * head_dim;
        const uint kv_tok_stride  = head_dim;
        device T_act* k_dst = kv_cache_k
            + (size_t)block_id     * (size_t)kv_blk_stride
            + (size_t)kv_head      * (size_t)kv_head_stride
            + (size_t)block_offset * (size_t)kv_tok_stride;

        if (d < half_dim) {
            const uint pos = positions[t];
            device const T_act* cos_row = cos_sin + (size_t)pos * (size_t)rot_dim;
            device const T_act* sin_row = cos_sin + (size_t)pos * (size_t)rot_dim + half_dim;
            const float c  = float(cos_row[d]);
            const float s  = float(sin_row[d]);
            const float x0 = kbuf[d];
            const float x1 = kbuf[half_dim + d];
            k_dst[d]            = T_act(x0 * c - x1 * s);
            k_dst[half_dim + d] = T_act(x1 * c + x0 * s);
        } else if (d >= rot_dim) {
            k_dst[d] = T_act(kbuf[d]);
        }
        return;
    }

    const uint kv_head = head - kK_END;
    const uint slot = slot_mapping[t];
    if (slot == 0xFFFFFFFFu) return;

    const uint block_id     = slot / block_sz;
    const uint block_offset = slot % block_sz;
    const uint kv_blk_stride  = num_kv * block_sz * head_dim;
    const uint kv_head_stride = block_sz * head_dim;
    const uint kv_tok_stride  = head_dim;
    device T_act* v_dst = kv_cache_v
        + (size_t)block_id     * (size_t)kv_blk_stride
        + (size_t)kv_head      * (size_t)kv_head_stride
        + (size_t)block_offset * (size_t)kv_tok_stride;
    v_dst[d] = T_act(acc);
}

#define INST_FAQRC(act_tag, act_type, scale_tag, scale_type)               \
  template [[host_name(                                                    \
      "fused_affine_qkv_rope_cache_" #act_tag "_s_" #scale_tag             \
      "_b_4_specialized")]]                                                \
  [[kernel]] decltype(fused_affine_qkv_rope_cache_impl<act_type, scale_type>) \
      fused_affine_qkv_rope_cache_impl<act_type, scale_type>;

// Match the qmv coverage matrix: T_scale = half always (mlx-community
// 4bit ships F16 scales). T_act per the model's resolved dtype.
INST_FAQRC(f16,  half,   f16, half)
INST_FAQRC(bf16, bfloat, f16, half)
