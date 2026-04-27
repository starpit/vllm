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
// **Wholesale invariant:** every (TB_M, TB_N, STAGES) tuple in the
// host's `CUTLASS_TILE_ZOO` (Rust-side, drives `CutlassGemmImpl`
// registration + CSV cost lookup) appears in this list, and vice
// versa. Pinned by `dc_zoo_equals_host_zoo` in impl_lib.rs::tests.
// Drift on either side breaks the invariant at test time.
//
// To add a tile here you must (1) add the matching CSV calibration
// row to `cutlass_<TBM>x<TBN>_s<S>` in the per-target CSV, (2) add
// the tuple to `CUTLASS_TILE_ZOO` in impl_lib.rs, and (3) add the
// row below. To remove a tile, do all three in reverse. The host
// launcher in cutlass_standalone_gemm.cu is the canary for SMEM-
// fits — `can_implement` failures there mean the tile must be
// dropped from the zoo entirely (host AND DC), not silently
// excluded from one side.
#define CUTLASS_DC_GEMM_LIST(X)                                  \
    /* TB_M  TB_N  TB_K  STAGES  WARP_M  WARP_N  WARP_K */       \
    X(  32,    64,    32,    3,     32,     32,     32)          \
    X(  32,    64,    32,    4,     32,     32,     32)          \
    X(  32,   128,    32,    3,     32,     64,     32)          \
    X(  32,   128,    32,    4,     32,     64,     32)          \
    X(  32,   256,    32,    3,     32,     64,     32)          \
    X(  64,    64,    32,    3,     32,     32,     32)          \
    X(  64,    64,    32,    4,     32,     32,     32)          \
    X(  64,   128,    32,    3,     32,     64,     32)          \
    X(  64,   128,    32,    4,     32,     64,     32)          \
    X( 128,    64,    32,    3,     64,     32,     32)          \
    X( 128,    64,    32,    4,     64,     32,     32)          \
    X( 128,   128,    32,    3,     64,     32,     32)          \
    X( 128,   128,    32,    4,     64,     32,     32)          \
    X( 128,   256,    32,    3,     64,     64,     32)          \
    X( 256,    64,    32,    3,     64,     32,     32)          \
    X( 256,    64,    32,    4,     64,     32,     32)

// SplitK config list. Mirrors `CUTLASS_SPLITK_ZOO` in
// `crates/ferrite-forward-macro/src/impl_lib.rs`. Drift between the
// Rust list and this C++ X-macro is caught by
// `dc_splitk_zoo_equals_host_zoo` in impl_lib.rs::tests. Each row
// expands to a `(TB_M, TB_N, TB_K, STAGES, WARP_M, WARP_N, WARP_K,
// SPLIT_K)` 8-tuple — the warp shape mirrors the host's standalone
// instantiation in `cutlass_standalone_gemm.cu`.
#define CUTLASS_DC_SPLITK_LIST(X)                                  \
    /* TB_M  TB_N  TB_K  STAGES  WARP_M  WARP_N  WARP_K  SPLIT_K */ \
    X(  64,    64,    32,    4,     32,     32,     32,    2)      \
    X(  64,    64,    32,    4,     32,     32,     32,    4)      \
    X(  64,    64,    32,    4,     32,     32,     32,    8)      \
    X(  64,   128,    32,    4,     32,     64,     32,    2)      \
    X(  64,   128,    32,    4,     32,     64,     32,    4)      \
    X(  64,   128,    32,    4,     32,     64,     32,    8)      \
    X( 128,    64,    32,    4,     64,     32,     32,    2)      \
    X( 128,    64,    32,    4,     64,     32,     32,    4)      \
    X( 128,    64,    32,    4,     64,     32,     32,    8)      \
    X( 128,   128,    32,    4,     64,     32,     32,    2)      \
    X( 128,   128,    32,    4,     64,     32,     32,    4)      \
    X( 128,   128,    32,    4,     64,     32,     32,    8)
