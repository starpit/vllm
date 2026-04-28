# cuBLAS Freedom Handoff

**Goal**: drop the cuBLAS dependency from the inference container. CUDA 12's `libcublas.so` + `libcublasLt.so` are ~850 MB combined; on a 3-5 GB inference image that's 17–30% off — enough to materially change Kubernetes cold-start and cluster image cache pressure.

**Branch**: `ff-interpreter`. Tip: `72304b029`.

**Target arch**: sm89 (L4 / Ada). sm90+ will be ThunderKittens-based in a separate workstream — none of this code path applies there.

> ⚠️ **Read this first**: The pick counts below are derived from `vllm ferrite info --color never | grep -c …` on commit `72304b029`'s installed CSV. **Do not** repeat what you see here from memory or from older git revisions. Re-run the audit yourself before picking a target — earlier sessions burned hours by trusting stale or speculative numbers ("(in dump)" was once used as a placeholder for an unmeasured count and was later read as if it implied non-zero picks). The verification recipe is in **Audit recipe** below.

## Status snapshot

cuBLAS-using surface across the 271-variant dense fleet, by Instruction kind:

```
Cublas (standalone)           3574
FusedQkvRopePrefill            699   ← no CUTLASS sibling
FusedGateUpGeluMul             520   ← partial CUTLASS sibling (752 already migrated)
FusedGateUpSiluMul             295   ← CUTLASS sibling at 144 picks (BW-bound regime only)
FusedQkvRopeCache              194   ← no CUTLASS sibling
FusedGemmBias                    0   ← (no claim sites; all bias_add absorbed by FusedQkvRope*)
                              ----
Total bf16 cuBLAS-using       5282
```

Non-cuBLAS picks (already migrated):

```
CutlassGemm (standalone zoo)             1293
CutlassFusedGateUpGeluMul                 752   ← landed 72304b029
CutlassGemmSplitK + CutlassGemmAdd + Gemv (~1500 combined; not the lever)
CutlassFusedGateUpSiluMul                 144
CutlassFusedGemmBias                        0   ← 0 claim sites in fleet today
```

FP8 and Marlin paths are large but separate workstreams; not cuBLAS-using.

| Commit | Lever | Notes |
|---|---|---|
| `9f89f2d3e` | Delete `gemm_is_fusion_partner` heuristic — DP solves naturally | -469 |
| `1d570158d` | tile_m=16 CUTLASS variants + recalibrated L4 CSV | -125 |
| `a5ad45609` | 2-param linear regression predictor (correctness, not pick-shifter) | +33 |
| `4a67304ec` | `FusedGateUpSiluMul::cost_us` measured cuBLAS, not roofline | -1127 |
| `72304b029` | **`FusedGateUpGeluMul::cost_us` measured cuBLAS, not roofline** | **-752** |
| `72304b029` | Bias zoo on plain `cutlass::gemm::device::Gemm` (3.21× faster, 0 picks) | 0 |
| `72304b029` | Megakernels build disabled (cross-worktree cache pollution workaround) | 0 |

## The cost-eval bug pattern that's already paid out twice

`FusedGateUp{Silu,Gelu,…}MulImpl::cost_us` and `FusedQkvRope{Cache,Prefill}Impl::cost_us` (the cuBLAS peers) compute the GEMM cost as `cublas(M, packed_N, K) + bw_bound_elementwise`. The original code used **roofline FLOPS** (`peak_tflops_fp16`) for the GEMM step — ~8× optimistic vs measured cuBLAS. The CUTLASS peers were always reading measured CSV rows, so the DP saw `roofline-cublas << measured-cutlass` and never picked the CUTLASS path.

**The fix is identical in every case**: replace the roofline call with `ctx.profile.cost_us_for("cublas", num_tokens, packed_n, hidden)` and keep the roofline as a fall-back when the predictor has no signal.

Already applied:
- `FusedGateUpSiluMulImpl` in `4a67304ec` (-1127 picks)
- `FusedGateUpGeluMulImpl` in `72304b029` (-752 picks; doubles as CUTLASS peer landing)

**Strongly suspected, not yet verified**: `FusedQkvRopeCacheImpl` and `FusedQkvRopePrefillImpl` cost_us. If they have the same roofline anti-pattern *and* a CUTLASS peer existed, that's another ~893 picks at the cost of a one-line fix per Impl + a CUTLASS peer per Impl. (No CUTLASS peer exists yet for either; building one is the next major lever.)

**Check first** before doing kernel work: `grep -A20 'fn cost_us' impl_lib.rs` on the relevant Impl, look for `peak_tflops_fp16` or a `flops / peak * 1e6` pattern. If present, that's the same bug.

## What's been built that you can reuse

### CutlassFusedGateUpGeluMul (72304b029)

Mirrors the cuBLAS-peer structure exactly: ONE packed CUTLASS GEMM at `(M, 2I, K)` writing `[M, 2I]`, then `gelu_and_mul_fused_bf16` BW-bound elementwise. **No new .cu file** — reuses the standalone CUTLASS zoo + the existing activation kernel. Tile-zoo-pickable like `CutlassGemmImpl`; one Impl per `CUTLASS_TILE_ZOO` entry.

This is the **right architectural pattern** for any "packed-GEMM-then-elementwise" cuBLAS-using fusion. Use it as the template for `CutlassFusedQkvRope*` and any other peer you build. Earlier in the same session a `cutlass_2x_gemm`-based 2-GEMM EVT design was tried (analogous to `CutlassFusedGateUpSiluMul`) and reverted — it captures **zero picks for gemma** because gemma's MLP shapes are compute-bound, where the 2-GEMM-EVT scaffolding loses 18% to packed cuBLAS. The packed-CUTLASS approach matches cuBLAS's structure and wins iff `cutlass_<tile>(M, 2I, K) < cublas(M, 2I, K)`.

### Bias zoo on plain `cutlass::gemm::device::Gemm` (72304b029)

`cutlass_gemm_bias.cu` was rewritten off `scaled_mm_c2x.cuh` (`DefaultGemmWithVisitor` + `ThreadblockSwizzleStreamK` + EVT visitor tree + `AlignmentCD=4`) onto plain `cutlass::gemm::device::Gemm` + `LinearCombination` + `GemmIdentityThreadblockSwizzle` + alignment 8. **The bias is passed as the C operand at `ldc=0`** so the standard GEMM API gives row-broadcast for free — no custom epilogue, no visitor tree.

Single-shape benchmark at M=128, N=K=4096: 219.1 µs → **68.3 µs** (3.21×, beats cuBLAS at 73.5 µs).

**Captures zero picks today** because no model in the fleet emits a `(Gemm, BiasAdd)` lone pair — qwen2 is the only arch using `bias_add` and its three Q/K/V `bias_add` tiles are all claimed by `FusedQkvRopeCache/Prefill`. Retained because:
1. Proof-of-concept that `ldc=0` bias-broadcast works on the standalone-Gemm template family — directly reusable when building the missing CUTLASS-fused QKV-rope kernels (where qwen2's biases live).
2. If a model with a bias_add outside QKV rope ever joins the fleet, this is ready to capture it.

### Sweep: packed-N rows for gemma

`gemm_sweep.rs` now emits cublas + cutlass-tile rows at `(M, 2*intermediate_size, hidden_size)` for every gemma2/gemma3 fleet variant (13824–73728 packed N). The predictor's linreg fits these directly, eliminating the roofline fall-back at gemma MLP shapes. If you add a new packed-cuBLAS path for any other arch family, mirror this pattern (add `(packed_n, hidden)` rows for that arch's MLP).

## Known regression to investigate

Standalone `Cublas` picks went **up** 2696 → 3574 (+878) and `FusedQkvRopePrefill` went up 454 → 699 (+245) between the pre-`72304b029` audit (`/tmp/dump_after.txt`) and the post-`72304b029` audit (`/tmp/dump_solver_fix.txt`). Two suspect causes:
- **Predictor-refit artifact**: the new packed-N CSV rows changed the linreg fit globally; some standalone-GEMM buckets now extrapolate cublas as cheaper than they previously did (acceptable if real, but worth confirming).
- **Re-routing from the GELU fix**: the freed gate-up GELU tiles may be claimed by other cuBLAS-using fusions before reaching CUTLASS-peer alternatives.

**Not yet investigated.** The 752 pick GELU win is unambiguous; the +878 standalone-Cublas number is concerning enough to verify before adding more leverage on top. Diagnostic recipe in **Audit recipe** below.

## Levers ranked by current leverage

### A — `FusedQkvRope{Cache,Prefill}` cost-eval bugfix + CUTLASS peer (~1-3 days)
Apply the same playbook as `4a67304ec` / `72304b029`:
1. Check `FusedQkvRopeCacheImpl::cost_us` + `FusedQkvRopePrefillImpl::cost_us` for the roofline-FLOPS anti-pattern. If present, replace with measured `cublas` lookup at the packed QKV shape.
2. Add `CutlassFusedQkvRope{Cache,Prefill}Impl` peers: packed CUTLASS GEMM at `(M, q_size+2*kv_size, hidden)` + `qk_norm_rope_kernels` (or whichever existing rope kernel today's cuBLAS peer's Instruction body calls). Mirror `CutlassFusedGateUpGeluMulImpl`'s tile-zoo-pickable shape.
3. Add packed-QKV-N rows to `gemm_sweep.rs` per arch (qwen2, qwen3, llama, granite, etc).

**Estimated**: 893 picks of headroom. Realized capture depends on `cutlass_<tile>(M, qkv_packed, K) vs cublas(M, qkv_packed, K)` at fleet shapes — same dynamic as the GELU win.

### B — Stream-K standalone CUTLASS (handoff-original Lever B, 3-5 days)
CUTLASS 2.10+ ships Stream-K; we have the splitK template family in `cutlass_standalone_gemm.cu`. Currently registered as fixed `split_k ∈ {2,4,8}` variants. Stream-K is *adaptive* — splits K to fill SMs based on workload. Targets the **long-K medium-M** regime (down_proj prefill), part of the 3574 standalone Cublas picks.

### C — Small-M batched-GEMV SIMT kernel (1-2 weeks)
The lm_head prefill regime needs a fundamentally different algorithm at M ∈ [2, 32] tall-skinny. Hand-CUDA, sm89-specific. Mirrors the existing `cutlass_gemv` (M=1) but with M mapped to thread-block outer-dim. **Reference**: prior small-M work may exist in `worktree-ferrite-mega` per memory `feedback_check_older_mega_branch` — check before writing from scratch.

### D — Investigate +878 Cublas / +245 FusedQkvRopePrefill regression (~30 min)
Diagnostic-only. Diff the dumps at the bucket level; identify which buckets changed pick. If predictor-refit, accept and move on. If real bug, fix before A.

### E — `LinearLayer::forward` de-cublas (Lever D from prior handoff, ~1-2 weeks per peer)
`LinearLayer::forward` is the entry point that 5 fused cuBLAS-peer Impls (Gate-up, Qkv-rope, etc.) call. Replacing its cuBLAS branch with CUTLASS would eliminate the cuBLAS dep across all of them at once — but it's load-bearing for shapes where cuBLAS legitimately wins. Best done after CUTLASS-peer Impls exist for every fused pattern.

### F — Feature-gate cuBLAS off and accept perf hit (3-5 days)
Last step of the project. Only viable after every dense path has a CUTLASS realization (i.e. after A + B + C + E land). Premature today.

## Recommended sequence from `72304b029`

1. **D** (~30 min): confirm the +878 Cublas regression is predictor-refit not real.
2. **A** (~1-3 days): cost-eval bugfix + CUTLASS peer for `FusedQkvRope{Cache,Prefill}`. Same playbook as the GELU fix.
3. **B** (~3-5 days): Stream-K for standalone Cublas long-K medium-M regime.
4. **C** (~1-2 weeks): batched-GEMV for lm_head prefill.
5. **E** (~2-4 weeks): de-cublas `LinearLayer::forward`. Last code work before F.
6. **F** (~3-5 days): feature-gate cuBLAS off, verify image-size win, ship.

## Audit recipe

**Always run this on every change** — don't trust narrative pick counts.

```bash
# 1. Re-sweep on L4
CUDA_PATH=/usr/local/cuda-12.9 cargo run -p ferrite-cost-sweep --features cuda --release --bin gpu_cost_sweep > /tmp/cost.csv

# 2. Install CSV + force ferrite-cuda-targets rebuild (cargo doesn't always notice CSV-only changes)
cp /tmp/cost.csv vllm-rs/crates/ferrite-cuda-targets/profiles/cost_l4_sm89.csv
touch vllm-rs/crates/ferrite-cuda-targets/src/lib.rs

# 3. Build vllm-cli, dump audit
cargo build -p vllm-cli --features cuda --release
./target/release/vllm ferrite info --color never > /tmp/dump.txt

# 4. Per-kind pick count (the only number that matters)
for k in Cublas FusedGateUpGeluMul FusedGateUpSiluMul FusedQkvRopeCache FusedQkvRopePrefill \
         CutlassGemm CutlassFusedGateUpGeluMul CutlassFusedGateUpSiluMul CutlassFusedGemmBias FusedGemmBias; do
  printf "%-32s %d\n" "$k" "$(grep -c "\b${k}\b" /tmp/dump.txt)"
done
```

If you're tempted to act on a number from this doc without re-running the audit: don't.

## Tools

- **`vllm ferrite info`** emits `(M=…,N=…,K=…)` per Cublas/standalone-GEMM line. Pipe into `/tmp/classify_cublas.py` for shape-bucket analysis. Note: fused-pattern Instructions (`FusedGateUpGeluMul`, `FusedQkvRopePrefill`, etc.) do **not** carry shape annotations — derive from `<arch>` + `bounds`.
- **`/tmp/classify_cublas.py`** — buckets cuBLAS picks by shape aspect (tall-skinny / long-K / square / moderate) × M-range.
- **`/tmp/borderline_cublas.py`** — for each cuBLAS pick, compares against best non-cuBLAS in CSV; classifies as `clear_win` / `borderline` / `non_cublas_faster` / `no_csv_data`.
- **Linear regression predictor** in `target.rs` — extrapolates honestly, held-out test pins it within 3×.
- **Runtime shape assertions** — every Gemm-class Instruction's eval body asserts the runtime weight shape matches the codegen-time `(N, K)`. Catches loader/codegen drift loud.

## Verification quirks

- **Cudaforge cache stomp**: `~/.cache/cudaforge/vllm-cuda/` is shared across worktrees. If another claude in another worktree builds `cutlass_gemm_bias` from a different .cu, your `.a` gets overwritten with their symbol set. Per `feedback_cudaforge_cache.md`: when seeing `undefined reference: cutlass_gemm_bias_<TILE>_launch`, `rm libcutlass_gemm_bias.*` + touch the .cu to force re-build.
- **Megakernels build is disabled on this branch** (`build_megakernels` short-circuits). Required because the same shared cache dir is populated by another worktree's session with `.cu` files that `#include "llama.cuh"`, which doesn't exist on our include path. Re-enable only if this branch needs to ship megakernels — until then leave it off.
- **`vllm chat` correctness on commandr** is the oracle for any correctness-affecting change (per `feedback_smallest_model_for_verify.md`, `feedback_no_run_chat.md`). Wrap in `timeout 30s`.

## Memory pointers

- `feedback_dp_solver_no_heuristics` — never gate impl matches on downstream consumers, DP solves naturally
- `feedback_smallest_model_for_verify` — verify on commandr, not llama
- `feedback_cudaforge_cache` — clear stale `.a` files after kernel edits
- `feedback_check_older_mega_branch` — prior small-M kernel work may exist in `worktree-ferrite-mega`
- `feedback_no_run_chat` — `vllm chat` is the correctness oracle, with timeout
- `feedback_read_data_before_claiming_bottleneck` — measure first, pattern-match second
- `feedback_show_dont_tell` — structural claims need real-input tests, not synthetic ones
