# cuBLAS-Freedom Plan — Status

## Resume point
**Step 5d / Step 6 (Lever A2 prerequisite)** — Steps 1b(a), 5b
(partial), and 5c are landed across commits `637ef2fba` →
`3eff714f4`. Steps 1b(b/c/d), 2b, 3, and 4 are deferred (see
"Outstanding work" below). Step 5d's polarity flip and Step 6's
libcublas un-link are blocked on Lever A2 (qwen2 biased-QKV
bias-zoo CSV sweep) — without it three cuBLAS-using Impls
(`FusedGemmBiasImpl`, `FusedQkvRopeCacheImpl`,
`FusedQkvRopePrefillImpl`) stay registered, and `ldd` still
shows `libcublas.so` linked.

## Verified state at tip `3eff714f4`

`vllm ferrite info -c` — default env (cuBLAS ON):
```
0 distinct cuBLAS picks       (Cublas-singleton attack surface)
```
The "0" reflects **only the singleton `Cublas` Instruction kind**
(per the analyzer's `step.kind == "Gemm"` filter); `FusedQkvRope*`
and `FusedGemmBias` Instructions are still picked at qwen2 biased
shapes and still call `cublas.gemm*` via `LinearLayer::forward`.

`vllm ferrite info -c` — `FERRITE_DISABLE_CUBLAS_GEMM=1` (cuBLAS
OFF):
```
0 distinct cuBLAS picks       (singleton)
```
Same caveat — fused-cuBLAS Impls are still routed at qwen2.

Pre-step-1b(a) headline (tip `1a756be34`) was **3423 distinct
cuBLAS picks · lm_head 958 · body Norm→Gemm 561 · Gemm→Add 41**.

Correctness:
- `test_cuda_correctness_command_r_1l` passes ON (28s) and OFF
  (102s) at `3eff714f4`.
- Step 5c broader pass (FERRITE_DISABLE_CUBLAS_GEMM=1):
  - commandr-1l       : pass
  - qwen2-0.5b        : pass
  - smollm-135m       : pass
  - gemma2-2b         : pass
  - granite-3.3-2b    : pass (after alignment fix landed in `3eff714f4`)
  - qwen3-0.6b        : FAIL — pre-existing per-head qk-norm
    divergence (memory `project_perhead_qknorm_golden_divergence`),
    fails on cuBLAS-ON too. Out of scope.

Perf delta (single-batch, 5 iters + 2 warmup, 50p latency):

| model               | regime           | ON (s)   | OFF (s)  | regression |
|---------------------|------------------|----------|----------|------------|
| Qwen2.5-1.5B        | decode 128/128   | 1.7476   | 1.7505   | +0.16%     |
| Qwen2.5-1.5B        | prefill 2048/1   | 0.01568  | 0.01563  | -0.32%     |
| Qwen2.5-3B-Instruct | decode 128/128   | 3.3315   | 3.3385   | +0.21%     |
| Qwen2.5-3B-Instruct | prefill 2048/1   | 0.02898  | 0.02900  | +0.07%     |
| TinyLlama-1.1B-Chat | decode 128/128   | 1.1829   | 1.1833   | +0.04%     |
| TinyLlama-1.1B-Chat | prefill 1024/1   | 0.01165  | 0.01167  | +0.17%     |

**Worst-case: +0.21% (Qwen2.5-3B decode). Decode-weighted avg:
+0.14%.** Down from prior STATUS's +3.79% / +1.10% (the win is
Step 1b(a)'s DP fix; cuBLAS-OFF now routes through CUTLASS
fused-norm-gemm at lm_head where it previously fell to
unfused chains).

`ldd /tmp/vllm-cublas-off | grep libcublas` still shows
`libcublas.so` linked. The image-size win is NOT realized; see
§"Outstanding work" Step 6 below.

## History

- 2026-04-28 22:30 · Plan authored at tip `525a234fa` · baseline:
  - 3461 distinct cuBLAS picks
  - 1571 in `g_gt_5.00`, 1403 in `no_csv_data`, 487 close-margin
  - fusion-gap absorbable: 1702 (49.2%) — 1095 lm_head + 563 norm→gemm + 32 gemm→scalarmul + 12 gemm→add
- 2026-04-28 23:50 · Step 1 partial (commit `aaf367ea9`) — landed
  `CutlassFusedRmsNormGemm`, `CutlassFusedLayerNormGemm`,
  `CutlassFusedAddRmsNormGemm` (60 tile-zoo Impls). cuBLAS surface
  3461 → 3325 (−136). lm_head: 1095 → 962 (−133). Body Norm→Gemm:
  563 → 560 (−3). commandr-1l ON passes. **Below the plan's
  expected ~1095 capture — see prior STATUS Outstanding Step 1.**
- 2026-04-29 00:00 · Step 2 (commit `495e24ee7`) — landed
  `FusedCublasGemmAddImpl`, gated on FERRITE_DISABLE_CUBLAS_GEMM.
  FusedCublasGemmAdd captured 123 raw picks. cuBLAS surface
  3325 → 3473 (+148, regression).
- 2026-04-29 00:18 · Step 5a verified (cuBLAS-off picks = 0,
  commandr-1l OFF passes). **Polarity flip NOT applied.**
- 2026-04-29 00:21 · Step 6a measured (perf table above).
  **Image-size win NOT realized.**
- 2026-04-29 09:30 · Rename (commit `3bd9dbcee`) — `FusedNormGemm`
  family → `CutlassFusedNormGemm` to match the file-wide convention.
- 2026-04-29 11:00 · Step 1b(a) landed (commit `637ef2fba`) —
  fixed two interacting bugs:
  - hand-rolled norm-bytes formula in fused `cost_us` overcounted
    vs `RmsNormRefImpl`'s `elementwise_cost` (extra `+ hidden`
    weight read), making fused unconditionally lose at uncalibrated
    shapes;
  - DP `update` closure used strict `>` and tied on HashMap
    iteration order, so equal-cost paths picked non-deterministically.
    Added `multi_tile_picks` field to SparseCell.
- 2026-04-29 11:10 · Step 1b(a) follow-up (commit `9092ea341`) —
  refined DP tiebreak to `picks_count` (lower wins). Caught the
  3-tile-vs-(2-tile + singleton) tie that the multi_tile_picks
  metric missed. Headline drop: cuBLAS surface 3423 → 2726 (-697),
  lm_head absorbable 958 → 311 (-647).
- 2026-04-29 11:30 · Step 5b (commit `840287cd1`) — gated five
  fused-cuBLAS Impls on `FERRITE_DISABLE_CUBLAS_GEMM`. Triggered
  granite-3.3-2B alignment crash (vocab=49159 not divisible by 8;
  CUTLASS tile zoo requires alignment 8) and qwen2-0.5B biased-QKV
  layout mismatch.
- 2026-04-29 12:00 · Step 5b/5c (commit `3eff714f4`) — added
  `cutlass_tile_supports_shape` alignment gate, reverted gates on
  `FusedGemmBiasImpl`, `FusedQkvRopeCacheImpl`, `FusedQkvRopePrefillImpl`
  (Lever A2 prerequisite for full drop). 5/6 Step 5c arches green;
  perf table above.

## Outstanding work — to complete the plan

Each item below is what someone resuming should pick up. Items are
ordered by dependency, not by impact. Time estimates carry over from
the original plan.

### Lever A2 — bias-zoo CSV sweep + drop bias gate (~1-2 days)

**Blocks Step 5b's full closure, Step 5d, and Step 6's binary
unlink.** Per CUBLAS_FREEDOM_HANDOFF.md §A2:

1. Add `cutlass_gemm_bias_<TM>x<TN>_s<S>` rows to `gemm_sweep.rs`
   across the qwen2 packed-QKV shapes — qwen2-0.5b/1.5b/7b QKV
   widths × hidden × M-grid. The `cutlass_gemm_bias_*` kernel
   already exists in `csrc/cutlass_gemm_bias.cu` (built `72304b029`).
2. Re-sweep on L4, install `cost_l4_sm89.csv`.
3. In `CutlassFusedQkvRope{Cache,Prefill}Impl::matches`, drop the
   `if fused_qkv_claim_is_biased(fuf, &info) { return None; }` gate.
4. Add a `biased: bool` field to the opcode shape; runtime branches
   to `cutlass::cutlass_gemm_bias` when biased and `cutlass::cutlass_gemm`
   otherwise.
5. `cost_us` selects between `cutlass_<tile>` and
   `cutlass_gemm_bias_<tile>` CSV row by inspecting the claim's
   bias state.

After A2, re-attempt the gates on `FusedGemmBiasImpl`,
`FusedQkvRopeCacheImpl`, `FusedQkvRopePrefillImpl` and re-run
Step 5c to verify qwen2 still passes at cuBLAS-OFF.

### Step 1b(b) — `CutlassFusedAddRmsNormWithOffsetGemm` (~3-4h)

4-tile claim: `(residual-Add, scalar-offset-Add, RmsNorm, Gemm)`.
Mirrors `CutlassFusedAddRmsNormGemm` 3-tile structure with the
upstream scalar-Add. Current lm_head residual:
- 311 picks remaining in `lm_head: Norm→Gemm[→ScalarMul]`
- bulk live in gemma2/3/qwen3 lm_head shapes the 3-tile claim
  doesn't match because the upstream `+ 1.0` makes the FUF a
  4-tile chain.

Defer until Lever A2 is in motion (the main shipping blocker is
Lever A2, not residual lm_head capture).

### Step 1b(c) — `CutlassFusedScalarOffsetRmsNormGemm` (~2-3h)

3-tile claim: `(scalar-offset-Add, RmsNorm, Gemm)` for gemma
standalone (no upstream residual). Smaller capture than Step 1b(b)
but closes a structural gap.

### Step 1b(d) — `CutlassFusedNormGemmScalarMul` family (~3-4h)

One variant per existing Norm→Gemm Impl that has a `ScalarMul`
consumer of the Gemm output (gemma logit-scaling). ~32 picks per
analyzer (down from 32 today, mostly already absorbed by Step 1b(a)).

### Step 2b — fix `FusedCublasGemmAdd` regression (~1-2h)

Cohere parallel attn+mlp at M=2..64 over-claims `FusedCublasGemmAdd`.
Add `WorkloadConstraint::NumTokensRange { min: 64, max: u32::MAX }`
to `FusedCublasGemmAddImpl` so the cuBLAS path only competes where
cuBLAS-Gemm legitimately wins. Doesn't affect cuBLAS-OFF (the
Impl is gated off there); only matters for the cuBLAS-ON
benchmark baseline cleanliness.

### Step 3 — Stream-K kernel family (~5-7h)

Plan §"Step 3" is unchanged. Expected capture: 520 picks in the
`g_gt_5.00 small/mid-M long-K` regime. Closes most of the residual
1567 g_gt_5.00 cuBLAS picks. Pre-requisite: kernel rebuild via
`rm -f ~/.cache/cudaforge/vllm-cuda/libcutlass_standalone_gemm.a`
+ touch the .cu (per `feedback_cudaforge_cache.md`).

### Step 4 — sweep coverage for residual no_csv_data (~3-4h)

Plan §"Step 4" is unchanged. Run after Step 3. Stop criterion:
`no_csv_data` count below 50.

### Step 5d — flip the project default (~30 min)

Per plan §"Step 5.5": invert env var polarity so
`FERRITE_ENABLE_CUBLAS_GEMM=1` re-enables cuBLAS and the default
is OFF. **Blocked on Lever A2** — flipping today would make qwen2
the default-broken arch.

### Step 6 — re-measure with libcublas un-linked (~2-3h)

After Lever A2 + Step 5b's full closure, rebuild State B with
cuBLAS truly disabled and re-run the bench matrix. Critical
checks:

(a) `ldd /tmp/vllm-cublas-off | grep libcublas` produces no output.
    If it still does, `cargo tree -p vllm-cli --features cuda |
    grep -i cublas` to find the residual dep — likely `cudarc`'s
    `cublas` feature; gate via Cargo feature flag in
    `crates/{ferrite-kernels, ferrite-cuda-core, vllm-cuda}/Cargo.toml`.
    All three crates currently have `cudarc = { features = [
    "cublas", "cublaslt", ... ] }` regardless of env var.

(b) Worst-case perf regression — currently +0.21% with cuBLAS-OFF
    Impls partially gated. Once A2 lands and the remaining gates
    flip, expect this to widen. If it grows above 5%, Step 3
    (Stream-K) and Step 4 (sweep coverage) are no longer optional.

(c) Image-size delta: `ls -la /usr/local/cuda-12.9/.../libcublas*` →
    ~850 MB on disk → ~850 MB compressed container layer.

### Step 6 decision matrix (deferred)

Defer SHIP/GATED/BLOCKED until Step 6 (post-Lever-A2) re-measures.

## Procedural notes — what was NOT done per the plan

The plan-author's projection in PLAN_STATUS at tip `1a756be34`
listed Step 1b(a) as "matchers structurally correct but somehow not
firing" with a hypothesis about missing roofline fallback. The
root cause turned out to be different: a hand-rolled norm-bytes
formula in `cost_us` (overcounted vs the singleton's
`elementwise_cost` by `+ hidden` bytes) PLUS a non-deterministic
DP tiebreak on equal cost (HashMap iteration order). Both fixes
were needed; the roofline fallback was already in place on the
2-tile.

Steps 1b(b/c/d), 2b, 3, 4 were not delivered in this session.
Lever A2 is the blocker for the SHIP gate; follow-on work on
1b(b/c/d) and 3/4 will incrementally close the cuBLAS-OFF residual
but doesn't change the architectural picture.

## Open blockers

- **Lever A2**: bias-zoo CSV sweep on L4 + drop the matcher's
  bias gate. Required for `FusedGemmBiasImpl` / `FusedQkvRope*Impl`
  to be safely gated off. Without it, qwen2 fleet correctness
  breaks at cuBLAS-OFF.

- **Cargo feature gating** (Step 6 prerequisite): even with all
  five cuBLAS-using Impls gated, `cudarc` is built with the
  `cublas` + `cublaslt` features in three crates, so libcublas
  stays linked. Need feature-flag refactor across `ferrite-kernels`,
  `ferrite-cuda-core`, `vllm-cuda`.
