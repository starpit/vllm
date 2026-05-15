// SPDX-License-Identifier: Apache-2.0
//
// Paged-K/V-cache variant of MLX's steel_attention prefill kernel.
// Replaces `attention_prefill_sdpa_v2_paged_*` (which was a port of
// the wrong MLX source — sdpa_vector is decode-only).
//
// Algorithm is byte-for-byte MLX's `steel_attention` (FA-2 tiled);
// only the K/V load path is rewritten to use `PagedKVBlockLoader`
// over the ferrite-metal paged cache. See
// `mlx_steel_attn/steel_attention_paged_kernel.h`.

#include "mlx_steel_attn/steel_attention_paged_kernel.h"

// Llama-3.2-{1B,3B,8B} shape: BQ=32, BK=16, BD=128, WM=4, WN=1.
// BLOCK_SIZE=16 (ferrite-metal paged-cache block size matches BK
// so one MLX kb iter = one paged block).
template [[host_name("attention_steel_paged_f16_bq32_bk16_bd128_wm4_wn1_bs16")]]
[[kernel]] decltype(attention_paged<half, 32, 16, 128, 4, 1, 16, float>)
    attention_paged<half, 32, 16, 128, 4, 1, 16, float>;

template [[host_name("attention_steel_paged_bf16_bq32_bk16_bd128_wm4_wn1_bs16")]]
[[kernel]] decltype(attention_paged<bfloat, 32, 16, 128, 4, 1, 16, float>)
    attention_paged<bfloat, 32, 16, 128, 4, 1, 16, float>;
