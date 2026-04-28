# cuBLAS-Freedom Plan — Status

## Resume point
Next step: **3** (Stream-K kernel family)

## History
_(append-only log of completed steps, one line each)_

- 2026-04-28 22:30 · Plan authored at tip `525a234fa` · `vllm ferrite info -c` baseline:
  - 3461 distinct cuBLAS picks
  - 1571 in `g_gt_5.00`, 1403 in `no_csv_data`, 487 close-margin
  - fusion-gap absorbable: 1702 (49.2%) — 1095 lm_head + 563 norm→gemm + 32 gemm→scalarmul + 12 gemm→add
  - Stream-K-recoverable: 540 (with 20 fusion overlap)
- 2026-04-28 23:50 · Step 1 ✓ (partial) — landed `FusedRmsNormGemm`,
  `FusedLayerNormGemm`, `FusedAddRmsNormGemm` Impls (3 × 20 tile zoo
  = 60 registrations). cuBLAS surface 3461 → 3325 (-136 logical
  picks; -464 raw rows on FusedAddRmsNormGemm). lm_head: 1095 → 962
  (-133); body Norm→Gemm: 563 → 560 (-3). commandr-1l correctness
  passed. Gaps documented:
  - FusedRmsNormGemm 2-tile (RmsNorm,Gemm) caught 0 picks: body
    norms feed multi-consumer fusions (QKV/gate-up); single-consumer
    candidates are M=1 patterns where CutlassGemv beats both.
  - FusedLayerNormGemm 2-tile caught 0 picks despite cohere lm_head
    matching the structural pattern. Root cause not pinned in this
    session (matches() should fire on `LayerNorm [s0,s0] → Cublas
    [s0,s7]` at the lm_head; left as a follow-up — would unlock the
    cohere/commandr lm_head capture which is part of the residual
    962 lm_head class).
  - FusedAddRmsNormWithOffset → Gemm (gemma2/3 lm_head) and
    ScalarOffsetRmsNorm → Gemm (gemma standalone) — additional
    Norm→Gemm shapes the analyzer flags but my Impls don't claim.
    Each would need a parallel 3-tile claim Impl following the same
    pattern as FusedAddRmsNormGemm.

- 2026-04-29 00:00 · Step 2 ✓ — landed `FusedCublasGemmAddImpl`
  (cuBLAS-side peer to `CutlassGemmAddImpl`). Gated on the same
  `FERRITE_DISABLE_CUBLAS_GEMM` env var as `GemmRefImpl` so Step 5
  drops every cuBLAS path symmetrically. cuBLAS surface 3325 →
  3473 (+148 distinct picks; 71 of those are new `Gemm→Add`
  pairs the analyzer flags). FusedCublasGemmAdd captured 123 raw
  picks. CutlassGemmAdd raw picks shifted (242 post). The +148
  regression is unexpected vs the plan's +0 expectation —
  appears to be a DP cost-tiebreaker interaction at M=2..64
  cohere-style parallel attn+mlp where my Impl claims one
  (Gemm,Add) pair and the other shatters into singletons rather
  than going to CutlassGemmAdd. Not a correctness issue
  (commandr-1l passes, 1 passed). Inspecting at Step 5 / Step 6
  may reveal whether this matters at the ship state — with
  cuBLAS disabled, my Impl is dropped too, and the affected
  pairs revert to CutlassGemmAdd alone (which is exactly what
  Step 5 needs).

## Open blockers
(none)

## Final results
_(filled by Step 6)_
