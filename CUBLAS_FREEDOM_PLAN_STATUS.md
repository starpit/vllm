# cuBLAS-Freedom Plan — Status

## Resume point
Next step: **1** (FusedNormGemm)

## History
_(append-only log of completed steps, one line each)_

- 2026-04-28 22:30 · Plan authored at tip `525a234fa` · `vllm ferrite info -c` baseline:
  - 3461 distinct cuBLAS picks
  - 1571 in `g_gt_5.00`, 1403 in `no_csv_data`, 487 close-margin
  - fusion-gap absorbable: 1702 (49.2%) — 1095 lm_head + 563 norm→gemm + 32 gemm→scalarmul + 12 gemm→add
  - Stream-K-recoverable: 540 (with 20 fusion overlap)

## Open blockers
(none)

## Final results
_(filled by Step 6)_
