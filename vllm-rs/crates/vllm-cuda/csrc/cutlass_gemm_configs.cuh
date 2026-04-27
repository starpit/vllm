// SPDX-License-Identifier: Apache-2.0
// Shared CUTLASS 2.x GEMM tile-config list.
//
// Single source of truth for the device-callable (DC) configs the
// primitive megakernel dispatches. Used today by `prim_mega.cu`'s
// inner switch on `CutlassConfig`; the host launchers in
// `cutlass_standalone_gemm.cu` will migrate onto this list in a
// follow-up commit so the two stay in lockstep.
//
// Usage (X-macro): callers `#define CUTLASS_DC_GEMM_LIST_ENTRY(...)`
// then `#include "cutlass_gemm_configs.inc"` (or invoke the
// `CUTLASS_DC_GEMM_LIST(X)` macro below). The expander gets seven
// arguments per row: TB_M, TB_N, TB_K, STAGES, WARP_M, WARP_N,
// WARP_K. All configs share:
//
//   - bf16 A / B / C, RowMajor A, ColumnMajor B, RowMajor C
//   - float accumulator
//   - sm80 OpClassTensorOp (works on sm89)
//   - GemmShape<16, 8, 16> instruction shape
//   - LinearCombination<bf16, 8, float, float> epilogue
//   - GemmIdentityThreadblockSwizzle<>
//
// Add a row only after verifying the corresponding host launcher
// in cutlass_standalone_gemm.cu compiles for the target arch
// (some tile × stage combos exceed sm89's ~100KB SMEM budget and
// fail `can_implement` silently — the host launcher's instantiation
// site is the canary).

#pragma once

// The list. Each row corresponds to a `using Gemm_*` typedef in
// cutlass_standalone_gemm.cu lines 89–103, 343–348. Subset for
// Phase 1 — the workhorse tiles cuBLAS picks at typical Llama
// dims. Deep-pipeline (stages 5+), 32×*, K64, W8, swizzle, silu,
// splitK, sm90, GEMV variants land in follow-up commits.
#define CUTLASS_DC_GEMM_LIST(X)                                  \
    /* TB_M  TB_N  TB_K  STAGES  WARP_M  WARP_N  WARP_K */       \
    X(  64,    64,    32,    4,     32,     32,     32)          \
    X(  64,    64,    32,    3,     32,     32,     32)          \
    X(  64,    64,    32,    2,     32,     32,     32)          \
    X(  64,   128,    32,    4,     32,     64,     32)          \
    X(  64,   128,    32,    3,     32,     64,     32)          \
    X(  64,   128,    32,    2,     32,     64,     32)          \
    X( 128,    64,    32,    4,     64,     32,     32)          \
    X( 128,    64,    32,    3,     64,     32,     32)          \
    X( 128,    64,    32,    2,     64,     32,     32)          \
    X( 128,   128,    32,    4,     64,     32,     32)          \
    X( 128,   128,    32,    3,     64,     32,     32)          \
    X( 128,   128,    32,    2,     64,     32,     32)          \
    X( 128,   256,    32,    3,     64,     64,     32)          \
    X( 128,   256,    32,    2,     64,     64,     32)          \
    X( 256,    64,    32,    4,     64,     32,     32)          \
    X( 256,    64,    32,    3,     64,     32,     32)          \
    X( 256,    64,    32,    2,     64,     32,     32)          \
    X( 256,   128,    32,    2,     64,     32,     32)
