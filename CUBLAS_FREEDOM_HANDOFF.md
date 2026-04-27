# cuBLAS Freedom Handoff

**Goal**: drop the cuBLAS dependency from the inference container. CUDA 12's `libcublas.so` + `libcublasLt.so` are ~850 MB combined; on a 3-5 GB inference image that's 17–30% off — enough to materially change Kubernetes cold-start and cluster image cache pressure.

**Branch**: `ff-interpreter` (this worktree). Tip: pending commit on top of `5cbd6f2d0`.

**Target arch**: sm89 (L4 / Ada). sm90+ will be ThunderKittens-based in a separate workstream — none of this code path applies there.

## Status snapshot

```
standalone Cublas picks:          3250 → 2758   (-492, -15%)
fused-gate-up cuBLAS-using picks: 1498 →  302   (-1196, -80%)
all-cuBLAS-using surface:         (~10148 baseline) → ~9020 across 271 variants
CutlassFusedGateUpSiluMul picks:    0 →  124   (broke the zero-pick floor)
```

| Commit | Lever | Δ (cuBLAS-using) |
|---|---|---|
| `9f89f2d3e` | Delete `gemm_is_fusion_partner` heuristic — DP solves naturally | -469 |
| `1d570158d` | tile_m=16 CUTLASS variants + recalibrated L4 CSV | -125 |
| `a5ad45609` | 2-param linear regression predictor (correctness, not pick-shifter) | +33 |
| _pending_ | `FusedGateUpSiluMul::cost_us` measured cuBLAS, not roofline-peak | **-1127** |
| Plus infra: `33cc08517` shape annotation, `28d545f88` M-range in info | | |

The -1127 fix replaced an 8×-optimistic roofline (peak FP16 FLOPS) with the calibrated cuBLAS row at the packed `(M, 2I, K)`. The cuBLAS path was being preferred over CutlassFused-* and over honest singleton chains because its cost looked free; once it's honest, 124 picks find the CUTLASS-fused sibling and ~1000 picks route to non-cuBLAS singleton chains where the cuBLAS-fused path was simply the wrong choice.

The +33 from the linreg predictor is the predictor being honest: at memory-bound shapes cuBLAS legitimately has a slightly better fitted DRAM saturation slope. The eff-scale + clamp predictor was hiding that with a tiebreak that papered over physics. The new fit doesn't flip picks for free, but it correctly distinguishes kernels at extrapolation — necessary for any future kernel work to actually win in the solver.

## What's left: 2689 picks by pattern

```
tall-skinny / small prefill (M=2..)         454   11 archs   lm_head, vocab-N
long-K / small prefill (M=2..)              364   10 archs   down_proj, K=4×N
square-ish / large prefill (M=512..)        337    4 archs   o_proj at large M
square-ish / huge prefill (M=4096..)        255    5 archs   o_proj at huge M
square-ish / small prefill (M=2..)          241    7 archs   o_proj at small M
moderate-K / large prefill (M=512..)        178    2 archs   gate-up
long-K / huge prefill (M=4096..)            180    8 archs   down_proj huge
moderate-K / med prefill (M=64..)           138    3 archs
long-K / large prefill (M=512..)            136    5 archs
long-K / med prefill (M=64..)               126    6 archs
... + ~250 long-tail picks
```

**Two real gaps**, measured directly from the L4 CSV (not predictor inference):

1. **Small-M tall-skinny** (lm_head prefill, M=2..32, vocab N): cuBLAS ~5-7% faster than best CUTLASS at calibrated similar shapes. cuBLASLt has small-M tall-skinny dispatch heuristics the static tile zoo lacks.

2. **Long-K medium-M** (down_proj at M=64..512, K=4×N): cuBLAS 7-22% faster (worst at M=128). Static SplitK={2,4,8} grid doesn't match cuBLAS's adaptive K-split.

The "square-ish" buckets (~830 picks) are mostly noise-band (cuBLAS ≤1-3% faster) — they'd flip with a feature gate at minimal perf cost.

## The actual surface — much bigger than "Cublas" picks suggest

`grep device.cublas vllm-rs/crates/ferrite-{kernels,forward}/src/` returns **19 references**. Most aren't `GemmRefImpl` — they're fused impls calling `LinearLayer::forward` which calls cuBLAS internally. Pick-count by cuBLAS-using kind across the 271-variant dense fleet:

| Fusion / kind | Picks | Has CUTLASS sibling? | DP picks the CUTLASS sibling? |
|---|---|---|---|
| `Cublas` (standalone Gemm) | 2689 | Yes (CutlassGemm zoo) | 1042 — sometimes |
| `FusedGateUpGeluMul` | 2950 | **No (missing)** | n/a |
| `FusedQkvRopePrefill` | 1905 | **No (missing)** | n/a |
| `FusedGateUpSiluMul` | 1498 | Yes (`CutlassFusedGateUpSiluMul`) | **0 picks** |
| `FusedQkvRopeCache` | 1106 | **No (missing)** | n/a |
| `FusedGemmBias` | (in dump) | Yes (`CutlassFusedGemmBias`) | **0 picks** |

**Total cuBLAS-using picks: ~10,148** (dense fleet). The 2689 standalone-Gemm picks I'd been tracking are ~26% of the actual surface.

Worse: where CUTLASS fused siblings exist (`CutlassFusedGemmBias`, `CutlassFusedGateUpSiluMul`), **they're at zero picks across the fleet**. The DP isn't picking them. Either their `cost_us` is higher than the cuBLAS path on every shape (a calibration or model issue), their `matches` gate excludes most cases, or some other mechanism is silencing them. **This is a cheap-to-investigate, potentially-huge-lever item.**

**Investigated 2026-04-27.** The zero-pick story for `CutlassFusedGateUpSiluMul` was a pure cost-model bug: `FusedGateUpSiluMulImpl::cost_us` (the cuBLAS peer) used roofline peak-FP16 FLOPS instead of the measured `cublas` CSV row, while the CUTLASS sibling used measured cutlass rows. At llama-7b prefill the roofline gave ~95 µs vs measured ~825 µs — 8× optimistic. After fixing the cost call to `cost_us_for("cublas", M, 2I, K)`: CutlassFusedGateUpSiluMul jumps from 0 → 124 picks, FusedGateUpSiluMul drops from 1498 → 302, and ~1000 picks correctly migrate to non-cuBLAS singleton chains. **CutlassFusedGateUpSiluMul itself is a competitive kernel — the design (2 GEMMs at I + EVT silu_mul vs cuBLAS's 1 packed GEMM at 2I) costs ~18% more than cuBLAS-packed at compute-bound shapes; that's not a kernel-quality issue, just a regime where cuBLAS happens to win narrowly.**

`CutlassFusedGemmBias` is a different story and remains at zero picks. The cost model was already apples-to-apples (both peers use measured CSV rows). The kernel itself is genuinely ~3× slower than cuBLAS+bias_add at compute-bound shapes (e.g. M=128 N=4096 K=4096: cublas=70 µs, cutlass_fused_gemm_bias=219 µs, plain cutlass_128x128_s3=66.9 µs). The kernel template uses `cutlass_2x_gemm` from `scaled_mm_c2x.cuh` — the same scaffolding as FP8 quant scaled GEMMs — and likely carries broadcast-overhead that doesn't apply to the bf16 bias-add case. **Re-implementing on the same template family as the plain CUTLASS singleton (no `scaled_mm_c2x.cuh`) is the right next step here.**

To actually drop `libcublas`:

1. **Diagnose the zero-pick CUTLASS-fused situation.** First and cheapest. Plausibly a calibration miss (their CSV rows are missing or under-fit) or a solver gate (`gemm_is_fusion_partner`-class issue we haven't found yet). Could be 2700+ picks freed at near-zero cost.
2. **Write the missing CUTLASS-fused impls** for GateUpGelu, QkvRopeCache, QkvRopePrefill. ~6000 picks. Real epilogue-kernel work but the pattern is established by the existing CUTLASS-fused pair.
3. **Standalone-Gemm kernel work** (Levers B/C below) — closes the residual ~700 picks where cuBLAS legitimately wins.

`LinearLayer::forward` would still link `libcublas` until step 1+2 actually have all-CUTLASS picks fleet-wide AND we drop the cuBLAS branch from `LinearLayer::forward` itself. The cargo feature gate is the LAST step, after every dense path has a CUTLASS realization.

## Bigger levers, ranked by effort × impact

### Lever A — Just feature-gate cuBLAS off and accept 5-22% perf hit (3-5 days)
The audit shows ~830 picks are noise-band, ~1100 picks have predictor extrapolation behind them, ~700 picks are real gaps. If we accept whatever the DP picks when cuBLAS is unavailable, perf drops on the long-tail by 5-22%. Decode is unaffected (already CutlassGemv). Prefill latency goes up modestly.

**Win**: ~850 MB image-size, immediate. Image-size goal achieved.
**Cost**: degraded prefill perf on the patterns above; per-arch verification needed; some fused impls might need to be rewired (the surface-area issue above).
**Effort**: ~1 week, mostly de-link work + correctness verify on the fleet.

### Lever B — Stream-K Implementation (3-5 days)
CUTLASS 2.10+ ships Stream-K; we have it in `cutlass_standalone_gemm.cu` (search for `CUTLASS_SPLITK`). Currently registered as fixed `split_k ∈ {2,4,8}` variants. Stream-K is *adaptive* — splits K to fill SMs based on workload. Targets the **long-K medium-M** regime (down_proj prefill).

**Effort breakdown**: register `cutlass_streamk_*` Impls in `impl_lib.rs`, add CSV calibration sweep rows, verify pick on commandr.
**Estimated picks closed**: ~400 (down_proj across M-buckets).

### Lever C — Small-M batched-GEMV SIMT kernel (1-2 weeks)
The lm_head prefill regime needs a fundamentally different algorithm: stream B once, accumulate M output rows in shared memory. Hand-CUDA, sm89-specific. Mirrors the existing `cutlass_gemv` (M=1) but with M ∈ [2, 32] mapped to thread-block outer-dim.

**Reference**: there may be precedent in `worktree-ferrite-mega` per memory `feedback_check_older_mega_branch`. Check first.
**Estimated picks closed**: 454 lm_head + a chunk of small-M square-ish.

### Lever D — Drop ALL cuBLAS-using fused impls (1-2 weeks)
Replace `LinearLayer::forward`'s cuBLAS path with CUTLASS, propagate to the 9 fused-impl downstreams. Some have CUTLASS-only siblings (`CutlassFusedGemmBiasImpl`, `CutlassFusedGateUpSiluMulImpl`); others don't. Implies extending the CUTLASS impl set.

**Win**: this is the actual prerequisite for Lever A's de-link. Without it, even feature-gating `GemmRefImpl` off doesn't shrink the image.
**Estimated picks closed**: lots of "Cublas" labels actually represent FusedGemmBias-via-cublas, so this could be the biggest single lever.

### Lever E — Triton or hand-PTX (out of scope, noted for completeness)
~Same as Lever C but in a higher-level DSL. Triton's runtime is hundreds of MB, defeats image-size goal. Hand-PTX is sm-version-coupled. Stay in CUDA C++ / CUTLASS.

### Recommended sequence

Given the new finding that CUTLASS-fused siblings exist with zero picks AND that fused impls are 75%+ of the cuBLAS-using surface, the order should be:

1. **Diagnose zero-pick CutlassFused-\* (1-2 days).** Why aren't they winning? If it's a fixable solver/calibration issue, this is the single biggest lever — could free 2700+ picks for free. Look at: cost ordering at the workload points, registration order, any `matches` gate, CSV row presence for their kernel names.
2. **Lever D variant — write CUTLASS-fused for the three missing patterns (1-2 weeks each).** GateUpGeluMul, QkvRopeCache, QkvRopePrefill. Patterns established by existing siblings.
3. **Lever B (Stream-K, 3-5 days).** Closes long-K medium-M down_proj gap (~700 picks).
4. **Lever C (small-M tall-skinny SIMT kernel, 1-2 weeks).** Closes lm_head prefill (~454 picks).
5. **Lever A (feature gate + de-link, 3-5 days).** After dense paths have all-CUTLASS picks, gate cuBLAS off and verify image-size win.

Total realistic budget: **5-8 weeks** of focused work. Step 1 alone could change the budget significantly — if the zero-pick CutlassFused-\* turn out to be a one-line solver fix, we'd jump from 26% solved to 75%+ solved overnight.

## Tools we built this session (still useful)

- **`vllm ferrite info`** now emits `(M=…,N=…,K=…)` per GEMM line. Bucket header keeps precise per-bucket labels; line-level annotation is the cluster's M-union. Pipe into `/tmp/classify_cublas.py` for pattern analysis.
- **`/tmp/classify_cublas.py`** — buckets cuBLAS picks by shape aspect (tall-skinny / long-K / square / moderate) × M-range. Run against a fresh dump after any change.
- **`/tmp/borderline_cublas.py`** — for each cuBLAS pick, compares against best non-cuBLAS in CSV; classifies as `clear_win` / `borderline` / `non_cublas_faster` / `no_csv_data`. Threshold-sweepable.
- **Linear regression predictor** — extrapolates honestly. Held-out test pins it within 3×.
- **Runtime shape assertions** — every Gemm-class Instruction's eval body asserts the runtime weight shape matches the codegen-time `(N, K)`. Catches loader/codegen drift loud.

## Verification setup

- Cost-sweep on L4: `CUDA_PATH=/usr/local/cuda-12.9 cargo run -p ferrite-cost-sweep --features cuda --release --bin gpu_cost_sweep > /tmp/cost_l4_new.csv`. ~3-5 min on this hardware. Replace `vllm-rs/crates/ferrite-cuda-targets/profiles/cost_l4_sm89.csv` to install. **Touch `vllm-rs/crates/ferrite-cuda-targets/src/lib.rs` after to force ferrite-cuda-targets rebuild** — cargo doesn't always notice CSV-only changes.
- After kernel changes: clear `/home/moosevan/.cache/cudaforge/vllm-cuda/libcutlass_standalone_gemm.{a,manifest}` + the `.o` files to force CUDA recompile. Stale-cache memory: `feedback_cudaforge_cache.md`.
- `vllm chat` correctness on commandr (per `feedback_smallest_model_for_verify.md`).
- Audit run: `vllm-rs/target/release/vllm ferrite info --color never > /tmp/dump.txt && grep -c 'Cublas ' /tmp/dump.txt`.

## Open decisions for next session

1. **Lever ordering** — A first (ship image-size, accept perf hit) or C first (kernel work, then ship)? Strategic call.
2. **Perf-regression tolerance** — fail CI on >X% prefill latency vs current? Need a number to design Lever A around.
3. **Whether to write Lever D (cuBLAS-free fused impls) as one PR or per-impl** — it's many lines but each impl is small. Per-impl is incrementally landable.

## Memory pointers

- `feedback_dp_solver_no_heuristics` — never gate impl matches on downstream consumers, DP solves naturally
- `feedback_smallest_model_for_verify` — verify on commandr, not llama
- `feedback_cudaforge_cache` — clear stale `.a` files after kernel edits
- `feedback_check_older_mega_branch` — prior small-M kernel work may exist in `worktree-ferrite-mega`
- `feedback_no_run_chat` — `vllm chat` is the correctness oracle, with timeout
