// SPDX-License-Identifier: Apache-2.0
//
// gather_last_token / scatter_first_to_last_row — paired sampling-
// position slice kernels around the lm_head GEMM.
//
//   gather:  hidden[0, :]  <-  hidden[num_tokens-1, :]   (pre-GEMM)
//   scatter: logits[num_tokens-1, :]  <-  logits[0, :]   (post-GEMM)
//
// The lm_head GEMM in between dispatches a single m-tile (BM rows) so
// the work shrinks from `M=num_tokens * N=vocab * K=hidden` down to
// `M=BM * N=vocab * K=hidden` — a 32× reduction at num_tokens=1024.
// Pre/post wrappers keep the worker's existing
// `embedding_gather(logits, last_token_indices=[num_tokens-1])` path
// correct without it needing to know about the slice.
//
// `num_tokens` arrives via a 4-byte runtime buffer the worker
// re-writes each forward() — see `RuntimeBindingKind::NumTokensU32`.
// `HIDDEN` / `VOCAB` are baked as function constant 0 (the row
// stride of the tensor the kernel slices).

#include <metal_stdlib>
using namespace metal;

constant uint GATHER_ROW_STRIDE [[function_constant(0)]];

// Copy row `num_tokens-1` to row 0 (in-place). For `num_tokens <= 1`
// the kernel is a no-op — row 0 is already the (only) row.
#define GATHER_KERNEL(NAME, T_act) \
kernel void NAME( \
    device T_act* hidden                 [[buffer(0)]], \
    device const uint* num_tokens_buf    [[buffer(1)]], \
    uint tid [[thread_position_in_grid]]) \
{ \
    if (tid >= GATHER_ROW_STRIDE) return; \
    const uint num_tokens = num_tokens_buf[0]; \
    if (num_tokens <= 1) return; \
    const uint src_row = num_tokens - 1; \
    hidden[tid] = hidden[(ulong)src_row * (ulong)GATHER_ROW_STRIDE + (ulong)tid]; \
}

// Reverse direction: copy row 0 to row `num_tokens-1` (in-place).
// Same shape, same dtype, same row-stride constant. The lm_head GEMM
// wrote real logits to row 0 (per the shrunk-to-1-m-tile dispatch);
// the worker's downstream `embedding_gather` indexes at
// `last_token_indices[0] = num_tokens-1`, so we re-write that row to
// match.
#define SCATTER_KERNEL(NAME, T_act) \
kernel void NAME( \
    device T_act* logits                 [[buffer(0)]], \
    device const uint* num_tokens_buf    [[buffer(1)]], \
    uint tid [[thread_position_in_grid]]) \
{ \
    if (tid >= GATHER_ROW_STRIDE) return; \
    const uint num_tokens = num_tokens_buf[0]; \
    if (num_tokens <= 1) return; \
    const uint dst_row = num_tokens - 1; \
    logits[(ulong)dst_row * (ulong)GATHER_ROW_STRIDE + (ulong)tid] = logits[tid]; \
}

GATHER_KERNEL(gather_last_token_f16_specialized,  half)
GATHER_KERNEL(gather_last_token_bf16_specialized, bfloat)
SCATTER_KERNEL(scatter_first_to_last_row_f16_specialized,  half)
SCATTER_KERNEL(scatter_first_to_last_row_bf16_specialized, bfloat)
