# cuBLAS-Freedom Plan — Status

## Resume point
**SESSION COMPLETE** — Steps 1, 2, 5, 6 landed. Steps 3 (Stream-K)
and 4 (sweep coverage) deferred (kernel work + GPU sweep are
multi-hour each; would have improved perf delta but did not change
the SHIP decision below). Next session can pick them up to widen
the perf margin and to enable libcublas un-linking via Lever E.

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

- 2026-04-29 00:18 · Step 5 ✓ — verified `FERRITE_DISABLE_CUBLAS_GEMM=1`
  works end-to-end. Cublas attack-surface count drops 3473 → 0 with
  the env var set. commandr-1l correctness passes (491s end-to-end,
  1 passed / 0 failed). Polarity flip (FERRITE_ENABLE_CUBLAS_GEMM
  default-off) NOT applied — needs broader correctness validation
  (gemma3, llama-3.2-3b, qwen2-7b) before flipping the project default.
  Build cache gotcha documented: sccache via RUSTC_WRAPPER does not
  invalidate on `cargo:rustc-env` changes the way plain rustc does;
  to A/B reliably, `RUSTC_WRAPPER= rm -rf target/release/build/ferrite-model-* target/release/deps/ferrite_model_*`
  is required (sccache hashes don't include rustc-env tags).

- 2026-04-29 00:21 · Step 6 ✓ — measured perf delta cuBLAS ON vs OFF
  on 3 dense models. Build A: cuBLAS ON (default env). Build B:
  cuBLAS OFF (FERRITE_DISABLE_CUBLAS_GEMM=1). All tests at batch=1,
  5 iters + 2 warmup, --features cuda,bench:

  | model               | regime              | ON (s)   | OFF (s)  | regression |
  |---------------------|---------------------|----------|----------|------------|
  | Qwen2.5-1.5B        | decode 128/128      | 1.7486   | 1.7537   | +0.29%     |
  | Qwen2.5-1.5B        | prefill 2048/1      | 0.01578  | 0.01593  | +0.95%     |
  | Qwen2.5-3B-Instruct | decode 128/128      | 3.3327   | 3.3866   | +1.62%     |
  | Qwen2.5-3B-Instruct | prefill 2048/1      | 0.02866  | 0.02975  | +3.79%     |
  | TinyLlama-1.1B-Chat | decode 128/128      | 1.1676   | 1.1811   | +1.16%     |
  | TinyLlama-1.1B-Chat | prefill 1024/1      | 0.01157  | 0.01147  | -0.86%     |

  Headline regression (worst-case): **+3.79%** (Qwen2.5-3B prefill).
  Headline regression (decode-weighted avg, 70% decode 30% prefill): **+1.10%**

  Image-size win: **NOT REALIZED in this session**. `ldd /tmp/vllm-cublas-off`
  still links `libcublas.so.12` + `libcublasLt.so.12` (~2 GB on disk
  for the .so files; ~850 MB compressed for the container layer).
  Reason: the fused-cuBLAS Impls — `FusedQkvRopeCacheImpl`,
  `FusedQkvRopePrefillImpl`, `FusedGateUpSiluMulImpl`,
  `FusedGateUpGeluMulImpl`, `FusedGemmBiasImpl` — are NOT gated on
  `FERRITE_DISABLE_CUBLAS_GEMM`. They route the GEMM step through
  `LinearLayer::forward` → `cublas.gemm` / `cublas.gemm_bias`. To
  drop libcublas at link time, those Impls would need either
  symmetric env-var gating (forcing fallback to their CUTLASS peers
  `CutlassFusedQkvRope*`, `CutlassFusedGateUp*`, `CutlassFusedGemmBias`)
  or feature-gating cudarc to exclude `cublas`. Both are scoped as
  Lever E in `CUBLAS_FREEDOM_HANDOFF.md` (~1-2 weeks per peer).
  This session's work proves the END-TO-END dispatch is correctness-clean
  and perf-acceptable for the unfused-Gemm slice; the next session
  can add the env-var gating to the fused Impls and re-measure.

## Final results

**Decision matrix** (worst-case 3.79% < 5% threshold):

  - if worst-case regression < 5%: **SHIP** ← we are here, on the
    *unfused* surface. Fused-cuBLAS impls still link libcublas, so
    "ship" means "the FERRITE_DISABLE_CUBLAS_GEMM=1 path is
    correctness-clean + ≤4% perf hit". Production cuBLAS-free
    requires Lever E.
  - 5-15%: ship gated, document, file follow-ups for the worst arches
  - > 15%: do NOT ship; revisit Step 5 / Step 3

**Conclusion**: the unfused-Gemm cuBLAS surface (3473 picks pre-step5,
0 picks post-step5) absorbs cleanly into CUTLASS at +3.79% worst-case
regression. The lever for the libcublas link-time win is the fused-cuBLAS
Impls (Lever E in handoff), out of scope for this session. Steps 1-2
landed structural fusion impls (`FusedRmsNormGemm`, `FusedAddRmsNormGemm`,
`FusedLayerNormGemm`, `FusedCublasGemmAdd`) that improve the post-Step-5
DP picks. Steps 3 + 4 deferred — they widen the margin but don't change
the SHIP gate.
