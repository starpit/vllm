// Thin FFI shim for FlashAttention-2.
//
// Sets up Flash_fwd_params identically to vllm-flash-attn/kernels/flash_api.cu
// (run_mha and run_mha_paged), but takes raw GPU pointers instead of at::Tensor.
//
// IMPORTANT: We do NOT include flash_fwd_launch_template.h here. That header
// contains full template definitions with <<<kernel launch>>> syntax, which
// would cause nvcc to instantiate ALL kernel variants in this TU (~28 min).
// Instead we include only flash.h (which has the template DECLARATIONS) and
// static_switch.h (dispatch macros). The template bodies are compiled in the
// per-headdim .cu files; we resolve them at link time.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "namespace_config.h"
#include "flash.h"
#include "static_switch.h"

#include <cutlass/numeric_types.h>

using namespace FLASH_NAMESPACE;

// ---------------------------------------------------------------------------
// Dispatch — matches vllm-flash-attn/csrc/flash_attn/flash_api.cpp
// ---------------------------------------------------------------------------

static void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel = false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.num_splits <= 1 && !force_split_kernel) {
                    run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }
            });
        });
    });
}

static inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

// ---------------------------------------------------------------------------
// mha_varlen_fwd — unified entry point for both contiguous and paged KV.
//
// Parameter setup matches vllm-flash-attn/kernels/flash_api.cu exactly.
// For paged KV: block_table != nullptr, pass seqused_k for actual K lengths.
// For contiguous: block_table == nullptr.
// ---------------------------------------------------------------------------

extern "C" void mha_varlen_fwd(
    // Tensor pointers (device)
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *out_ptr,
    void *softmax_lse_ptr,

    // Sequence metadata (device)
    int32_t *cu_seqlens_q,
    int32_t *cu_seqlens_k,
    int32_t *seqused_k,       // [batch_size] actual K lengths (paged), or nullptr

    // Paged KV cache (device), or nullptr for non-paged
    int32_t *block_table,
    int32_t block_table_batch_stride,

    // Dimensions
    int32_t batch_size,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    int32_t num_heads,
    int32_t num_heads_k,
    int32_t head_size,
    int32_t page_block_size,

    // Q strides (elements, not bytes)
    int64_t q_row_stride,
    int64_t q_head_stride,
    // K/V strides
    int64_t k_batch_stride,
    int64_t k_row_stride,
    int64_t k_head_stride,
    // O strides
    int64_t o_row_stride,
    int64_t o_head_stride,

    // Config
    float softmax_scale,
    int32_t is_causal,
    int32_t window_size_left,
    int32_t window_size_right,
    float softcap,
    int32_t is_bf16,
    int32_t num_splits,

    // Split-K accumulators (unused — kept for API compat)
    void *softmax_lse_accum_ptr,
    void *out_accum_ptr,

    cudaStream_t stream
) {
    // --- Parameter setup matches vllm-flash-attn flash_api.cu exactly ---

    const int head_size_rounded = round_up(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_up(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_up(max_seqlen_k, 128);

    Flash_fwd_params params = {};

    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = out_ptr;

    params.softmax_lse_ptr = softmax_lse_ptr;
    params.alibi_slopes_ptr = nullptr;

    params.q_batch_stride = 0;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = k_batch_stride;
    params.o_batch_stride = 0;
    params.alibi_slopes_batch_stride = 0;

    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = k_row_stride;
    params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = k_head_stride;
    params.o_head_stride = o_head_stride;

    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = head_size;
    params.d_rounded = head_size_rounded;

    if (softcap > 0.0f) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * (float)M_LOG2E;
    } else {
        params.softcap = 0.0f;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * (float)M_LOG2E;
    }

    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = uint8_t(std::floor(1.0f * 255.0));
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.is_bf16 = is_bf16;

    params.cu_seqlens_q = cu_seqlens_q;
    params.cu_seqlens_k = cu_seqlens_k;
    params.seqused_k = seqused_k;
    params.p_ptr = nullptr;

    // Causal / window — set directly, same as upstream flash_api.cu
    params.is_causal = is_causal;
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.is_seqlens_k_cumulative = true;
    params.unpadded_lse = 1;
    params.num_splits = num_splits;

    // Paged KV
    const bool paged = (block_table != nullptr);
    if (paged) {
        params.block_table = block_table;
        params.block_table_batch_stride = block_table_batch_stride;
    }
    params.page_block_size = page_block_size;

    // Split-K accumulators (for num_splits > 1)
    params.softmax_lseaccum_ptr = softmax_lse_accum_ptr;
    params.oaccum_ptr = out_accum_ptr;

    // Launch — paged KV MUST use the splitkv kernel (the standard kernel
    // compute_attn_1rowblock has no block_table support).
    if (max_seqlen_k > 0) {
        run_mha_fwd(params, stream, /*force_split_kernel=*/paged);
    }
}
