// Multi-segment SDPA for MLX — attention across multiple contiguous K/V segments.
// Used for span support: each segment is a contiguous K/V region (cached span or
// current request's cache). The kernel iterates segments with online softmax.

#ifndef VLLM_MULTI_SEGMENT_SDPA_H
#define VLLM_MULTI_SEGMENT_SDPA_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { void* ctx; } vllm_mlx_array;
typedef struct { void* ctx; } vllm_mlx_stream;

// Multi-segment SDPA.
//
// q:            [1, num_heads, 1, head_dim] — single query (decode)
// k_segments:   N arrays, each [1, num_kv_heads, seg_len_i, head_dim]
// v_segments:   N arrays, each [1, num_kv_heads, seg_len_i, head_dim]
// seg_position_offsets: [N] i32 — RoPE position offset per segment
// cos_sin_cache: [max_pos, rotary_dim] — precomputed cos/sin for RoPE
// scale:        1/sqrt(head_dim)
// num_segments: N
//
// Returns [1, num_heads, 1, head_dim]
//
// Returns 0 on success.
int vllm_multi_segment_sdpa(
    vllm_mlx_array* result,
    vllm_mlx_array query,
    const vllm_mlx_array* k_segments,
    const vllm_mlx_array* v_segments,
    int num_segments,
    vllm_mlx_array seg_position_offsets,
    const int* seg_needs_rope,
    vllm_mlx_array cos_sin_cache,
    float scale,
    int rotary_dim,
    vllm_mlx_stream stream);

#ifdef __cplusplus
}
#endif

#endif // VLLM_MULTI_SEGMENT_SDPA_H
