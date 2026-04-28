# cuBLAS-Freedom Execution Plan

**Goal**: drop `libcublas.so` from the inference container. ~850 MB image-size win on a 3-5 GB image (17-30%). Final ship state: `FERRITE_DISABLE_CUBLAS_GEMM=1` is the default, every singleton GEMM lands on CUTLASS, fused-cuBLAS Impls have CUTLASS peers, perf regression numerically bounded and documented.

**Branch**: `ff-interpreter`. **Tip when this plan was written**: `525a234fa`.

**The diagnostic that frames every step**: `vllm ferrite info -c` (subcommand added in `5a50a23f8`/`49b5c092e`/`387af55ca`/`525a234fa`). Reads the bundled cost CSV + every arch's `Vec<BucketDump>`; classifies every `Cublas` Gemm pick by margin (cuBLAS-vs-best-CUTLASS gap) and fusion-gap reason (which fusion claim, if any, would have absorbed it). Run after every step to track progress against the projection.

**Baseline (tip `525a234fa`)**:
```
3461 distinct cuBLAS picks · margin: 1571 >5% / 1403 no_csv_data / 487 close-margin
fusion-gap absorbable: 1702 (49.2%)
   1095  lm_head: Norm→Gemm[→ScalarMul]
    563  Norm→Gemm
     32  Gemm→ScalarMul
     12  Gemm→Add  (CutlassGemmAdd peer; no cuBLAS peer)
>5% picks by regime:  540 Stream-K-recoverable / 589 large-M compute-bound /
                      434 mid-M short-K / 8 misc
```

---

## Operating principles for the executing agent

These are LOAD-BEARING. Honor them or stop and escalate.

1. **Honor existing memory rules**. `~/.claude/projects/-home-moosevan-vllm/memory/` is the source of truth. In particular:
   - `feedback_never_delete_tests`, `feedback_no_special_case_macros`, `feedback_dp_solver_no_heuristics`, `feedback_no_use_based_skipping`, `feedback_no_refusal_chasing`.
   - `feedback_serial_for_goldens` — `--test-threads=1` for any e_correctness run.
   - `feedback_smallest_model_for_verify` — verify on commandr-1-layer first.
   - `feedback_no_run_chat` — wrap `vllm chat` in `timeout`.
   - `feedback_build_flags` — `cargo build -p vllm-cli --features cuda --release`.
   - `feedback_vllm_bench_feature` — bench commands need `-F cuda,bench`.
   - `feedback_no_double_build` — don't pre-build before tests.
   - `feedback_concurrent_builds` — harness notifies on background completion; do NOT poll.
   - `feedback_never_pkill_cargo` — never kill cargo by pattern.
   - `feedback_cudaforge_cache` — `rm libcutlass*.a` to force kernel rebuild on cache pollution.
   - **`feedback_no_shortcuts`** — no GMEM-disguised fusion, no dim-gated fallbacks, every step on the critical path.

2. **Never destructive without permission**. No `git checkout --`, `git reset --hard`, `git clean -f`, force push, branch deletion, `pkill cargo`. If you need to throw away changes, **stop and write to the status file**.

3. **Commit incrementally**. Every step ends with a green commit on `ff-interpreter`. Commit messages follow recent style: `ff-interpreter: <imperative summary>`. Trailer `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.

4. **Status file is the resume point**. `CUBLAS_FREEDOM_PLAN_STATUS.md` (sibling of this file). Each step writes a section noting completion / blocker / metric. Future-Claude reads STATUS first, runs `vllm ferrite info -c`, decides which step to resume from.

5. **One step at a time. Verify, then move on**. After each step:
   - `cargo fmt -p <touched crates>`
   - `cargo clippy -p <touched crates> --release --features cuda -- -D warnings`
   - `vllm ferrite info -c --color never > /tmp/post_step.txt` and diff against the previous snapshot
   - Commit

6. **Bail criteria. Stop and write to STATUS** when:
   - Any commandr correctness test fails (not the bf16-noise warnings — actual `failed` count > 0).
   - A step's diff exceeds 1500 lines (probably wrong scope).
   - A clippy warning needs `#[allow(...)]` to suppress (usually a sign of structural problem).
   - You'd need to introduce a heuristic (`if biased { ... }`, hardcoded `if M < 8 { ... }`) — `feedback_dp_solver_no_heuristics`.
   - You'd need to delete or shrink an existing test — `feedback_never_delete_tests`.
   - You'd need to rebase, merge, or pull — `feedback_git_rebase`.
   - **CRITICAL: any kernel-quality regression on commandr.** The bf16-noise warnings at position ≥ 10 are EXPECTED. A new failure (e.g. position 0/1/2 mismatch, or `0 passed; 1 failed`) is a stop signal.

7. **Time budget**. Each step is annotated with an expected duration. If you're at 2× and not done, write to STATUS and stop.

---

## Pre-flight (run on every plan resume — ~5 min)

```bash
cd /home/moosevan/vllm/.claude/worktrees/ff-interpreter
git status                                          # must be clean (or only STATUS file modified)
git log --oneline -3                                # confirm tip
cd vllm-rs
CUDA_PATH=/usr/local/cuda-12.9 cargo build -p vllm-cli --features cuda --release
./target/release/vllm ferrite info -c --color never > /tmp/preflight.txt
head -20 /tmp/preflight.txt                         # snapshot — check distinct pick count
```

If `git status` is dirty with non-STATUS changes: read STATUS, decide whether to commit / discard. **Do not discard without explicit confirmation in STATUS that says "discardable: yes"**.

If pre-flight produces a build error not seen in STATUS: write to STATUS as `STEP X — BLOCKED: <error>`, stop.

---

## Step 1 — FusedNormGemm  (estimated 4-6h, captures up to **1658 picks**)

**Why first**: 48% of the entire cuBLAS surface (lm_head 1095 + body Norm→Gemm 563). Single Impl can absorb both shapes since lm_head IS just `Norm → Gemm` with no body-tail consumer. The follow-up `FusedGemmScalarMul` (Step 1.5) handles the gemma logit-scaling tail.

**1.1 Read the existing fusion templates first**
- `vllm-rs/crates/ferrite-forward-macro/src/impl_lib.rs`:
  - `FusedAddRmsNormImpl` (residual + rmsnorm) — closest precedent for "norm is followed by a single consumer".
  - `CutlassFusedQkvRope{Cache,Prefill}Impl` (commit `812452cff`) — most recent fusion peer, mirror its structure: tile-zoo-pickable, runtime calls cutlass kernel, opcode carries (tile_m, tile_n, stages, packed_n, k).
- `vllm-rs/crates/ferrite-forward/src/instr.rs`: `Instruction::CutlassFusedQkvRopeCache` runtime arm — same shape as the new arms you'll add.
- `vllm-rs/crates/ferrite-forward/src/info.rs`: corresponding pretty-printer entries.

**1.2 Author `FusedNormGemmImpl`** (one Impl, parameterised by tile, runtime calls CUTLASS only)

Claim shape: `(Norm, Gemm)` where:
- `Norm` is `OpKind::RmsNorm` or `OpKind::LayerNorm` or `OpKind::FusedAddRmsNorm` (residual+rms) — **only when its output has exactly ONE consumer** (the Gemm). If multiple consumers, do NOT claim — that pattern is already handled by `FusedQkvRope*` / `FusedGateUp*`.
- `Gemm` is `OpKind::Gemm` whose only input is the Norm's output.
- Weight storage on the Gemm is `Dense` (Marlin/Bnb4/Fp8 paths have their own peers).

Cost: `cost_norm + cost_cutlass_<tile>(M, N, K)`. The norm cost lives elsewhere — read `RmsNormRefImpl::cost_us` for the pattern. **Do NOT include a cuBLAS path** — this Impl is CUTLASS-only by construction (that's the point: forces CUTLASS at picks where the unfused (Norm singleton + cuBLAS Gemm) currently wins).

Runtime: call existing `kernels::rms_norm_bf16` (or LayerNorm equivalent) + `cutlass::cutlass_gemm`. Two kernel launches, sequenced. The fusion is at the CLAIM level, not the kernel level — no new .cu work needed.

Register one Impl per `CUTLASS_TILE_ZOO` entry (mirror `CutlassFusedGateUpGeluMulImpl` registration loop).

**1.3 Add `FusedNormGemmScalarMul` variant for gemma logit-scaling**

Claim shape: `(Norm, Gemm, ScalarMul)`. Reuses FusedNormGemm's structure with a trailing `kernels::scalar_mul_inplace_bf16` after the Gemm. **Bail criterion**: if the gemma ScalarMul kernel binding is missing, drop this scope to "Step 1 ships only FusedNormGemm; FusedNormGemmScalarMul becomes Step 1.5 follow-up". Do not invent a kernel.

**1.4 Verify before commit**
- `cargo fmt -p ferrite-forward-macro -p ferrite-forward`
- `cargo clippy -p ferrite-forward-macro -p ferrite-forward --release --features cuda -- -D warnings`
- `cargo build -p vllm-cli --features cuda --release`
- `vllm ferrite info -c` — record new totals. **Expected**: ~1095 lm_head picks gone if FusedNormGemmScalarMul lands; ~563 body Norm→Gemm picks gone otherwise. Allow ±50 noise from bucket-multiplication.
- `cargo test -p vllm-e2e --release --features e2e,cuda --test e_correctness test_cuda_correctness_command_r_1l -- --ignored --test-threads=1 --nocapture` — must pass with bf16-noise warnings only.

**1.5 Commit**
Subject: `ff-interpreter: FusedNormGemm[+ScalarMul] — absorb lm_head + body norm→gemm`
Body: include the before/after pick counts, fusion-gap rollup delta, regime-residual delta.

**Status checkpoint**: STATUS file gains `Step 1 ✓ — captured X picks · commit <sha>`.

---

## Step 2 — FusedGemmAdd cuBLAS peer  (estimated 1-2h, captures **12 picks**)

**Why bother for 12 picks**: it's a 30-line code change and it closes a specific structural anomaly the analyzer surfaced — `CutlassGemmAdd` exists as a fusion peer but no cuBLAS-side variant, so when CUTLASS loses the bucket, the fusion shatters into `(Cublas, Add)` standalone. Tiny perf-impact but it's structural completion, mostly a sanity fix.

**2.1 Read precedents**
- `CutlassGemmAddImpl` in `impl_lib.rs` — claims `(Gemm, Add)`.
- `FusedQkvRopeCacheImpl` — uses `cuBLAS LinearLayer::forward`. Similar idea here.

**2.2 Author `FusedCublasGemmAddImpl`**
Claim shape: same `(Gemm, Add)` as `CutlassGemmAdd`. Cost: `cublas(M,N,K) + bw_bound_add`. Runtime: cuBLAS GEMM + `add_inplace` kernel.

This Impl is part of the cuBLAS-using surface — REMOVED automatically by `FERRITE_DISABLE_CUBLAS_GEMM=1` since its cost reads `"cublas"`. Don't gate it on the env var (let the matcher always claim; the DP picks based on cost; the env var only drops `GemmRefImpl` from the library, not other cuBLAS users).

Wait — re-read the existing env-var hook. It only drops `GemmRefImpl`. To make `FusedCublasGemmAddImpl` follow the same A/B, gate its registration on the same env var:

```rust
if std::env::var_os("FERRITE_DISABLE_CUBLAS_GEMM").is_none() {
    lib.push(Box::new(FusedCublasGemmAddImpl));
}
```

**2.3 Verify** as Step 1.

**2.4 Commit**
Subject: `ff-interpreter: FusedCublasGemmAddImpl — close the (Gemm,Add) cuBLAS-side gap`

---

## Step 3 — Stream-K kernel family  (estimated 5-7h, captures **520 picks**)

**Why now**: 540 picks in `g_gt_5.00` are small/mid-M long-K shapes — Stream-K's exact sweet spot. Adds a real CUTLASS kernel family rather than just rearranging fusions.

**3.1 CUTLASS Stream-K templates already exist; we use the existing splitK family extended to adaptive split.**

Read `vllm-rs/crates/vllm-cuda/csrc/cutlass_standalone_gemm.cu`:
- `CUTLASS_SPLITK_*` macros — fixed-split-K templates (split_k ∈ {2, 4, 8}).
- The tile_m=16 splitK additions from `63d13d499` are the most recent precedent.

**Stream-K** is `cutlass::gemm::device::Gemm` with `ThreadblockSwizzle = StreamKThreadblockSwizzle` (CUTLASS 2.10+). The split is workload-driven at runtime — single launch, no separate reduction kernel. Add 6-12 kernel instantiations covering tile_m ∈ {16, 32, 64, 128} × tile_n ∈ {64, 128} at stages=4. New macro `CUTLASS_STREAMK(...)` mirroring `CUTLASS_SPLITK_CONFIG` + `CUTLASS_SPLITK_LAUNCH`.

**3.2 FFI + dispatch**
- `vllm-rs/crates/ferrite-kernels/src/cutlass.rs` — extern decls + a new `CutlassStreamKTile { tile_m, tile_n, stages }` + `launch_fn_for_streamk`.
- `vllm-rs/crates/ferrite-cost-sweep/src/gemm_sweep.rs` — sweep imports + bench_cutlass_streamk! macro.
- `vllm-rs/crates/ferrite-forward-macro/src/impl_lib.rs` — `CutlassGemmStreamKImpl` (mirror `CutlassGemmSplitKImpl`) + `CUTLASS_STREAMK_ZOO` registration.

**3.3 Force kernel rebuild**
```bash
rm -f ~/.cache/cudaforge/vllm-cuda/libcutlass_standalone_gemm.a
touch vllm-rs/crates/vllm-cuda/csrc/cutlass_standalone_gemm.cu
rm -f ~/.cache/cudaforge/megakernels/*.cu  # per build_megakernels short-circuit
cargo build -p vllm-cli --features cuda --release  # 5-10 min for kernel compile
```
Verify symbols landed: `nm -gC ~/.cache/cudaforge/vllm-cuda/libcutlass_standalone_gemm.a | grep streamk`.

**3.4 Re-sweep CSV**
```bash
CUDA_PATH=/usr/local/cuda-12.9 cargo run -p ferrite-cost-sweep --features cuda --release --bin gpu_cost_sweep > /tmp/cost.csv 2>/tmp/cost_sweep.err
cp /tmp/cost.csv vllm-rs/crates/ferrite-cuda-targets/profiles/cost_l4_sm89.csv
touch vllm-rs/crates/ferrite-cuda-targets/src/lib.rs
cargo build -p vllm-cli --features cuda --release
```
The FlashInfer head_dim=256 errors in stderr are pre-existing — not a stop signal.

**3.5 Verify + commit**
- `vllm ferrite info -c` — expect `>5% small-M long-K` and `>5% mid-M long-K` to drop. Allow some shifts: Stream-K doesn't have to win EVERY long-K shape; some picks may stay on the fixed-split splitK that's already there.
- commandr correctness must pass.

Subject: `ff-interpreter: Stream-K kernel family for small/mid-M long-K`

---

## Step 4 — Sweep coverage for residual no_csv_data  (estimated 3-4h, unblocks Step 5)

**Why now**: after Steps 1-3, the residual `no_csv_data` class will mostly be lm_head-shape rows that FusedNormGemm has now absorbed. But there will still be picks at unswept (M, N, K) shapes that Step 5 (feature-gate cuBLAS off) can't safely route — they'd hit `UNCALIBRATED_COST_US` and pick whichever CUTLASS Impl has the noisiest extrapolation. This step closes the calibration debt.

**4.1 Identify the residual gap**
After Step 3's commit:
```bash
./vllm-rs/target/release/vllm ferrite info -c --color never > /tmp/post_step3.txt
```
Read the `no_csv_data` count + sample picks. Group sample shapes by arch.

**4.2 Add the missing shapes to `gemm_sweep.rs::NK_SHAPES`**
Mirror the gemma2 expansion in commit `f44eb134a`. One commit per arch family if there's significant per-arch work, else a single batched commit. Don't sweep vocab-N rows (Step 1's FusedNormGemm should have eaten lm_head Gemms — if any vocab-N picks remain post-Step-1, that's a Step 1 bug, NOT a sweep gap; STOP and revisit Step 1).

**4.3 Re-sweep + install**
```bash
CUDA_PATH=/usr/local/cuda-12.9 cargo run -p ferrite-cost-sweep --features cuda --release --bin gpu_cost_sweep > /tmp/cost.csv 2>/tmp/cost_sweep.err
cp /tmp/cost.csv vllm-rs/crates/ferrite-cuda-targets/profiles/cost_l4_sm89.csv
touch vllm-rs/crates/ferrite-cuda-targets/src/lib.rs
cargo build -p vllm-cli --features cuda --release
```

**4.4 Verify**
- `vllm ferrite info -c` — `no_csv_data` count should drop substantially. Stop criterion: `no_csv_data` count is below 50, OR all remaining `no_csv_data` picks are at vocab-N shapes (those are FusedNormGemm-absorbable, not sweep-fillable).
- commandr correctness.

Subject: `ff-interpreter: sweep coverage for residual no_csv_data shapes`

---

## Step 5 — Feature-gate cuBLAS off, fix what breaks  (estimated 2-4h)

**5.1 Set the env var, force a clean recompile**
The env-var hook is fixed in `525a234fa` (`cargo:rustc-env=` propagation). Each model crate re-emits the env value as a rustc-env directive, so changing the env var actually invalidates compilation. But the proc-macro is in a different crate; force rebuild explicitly:

```bash
cd /home/moosevan/vllm/.claude/worktrees/ff-interpreter/vllm-rs
for c in ferrite-model-llama ferrite-model-gemma2 ferrite-model-gemma3 ferrite-model-qwen2 \
         ferrite-model-qwen3 ferrite-model-granite ferrite-model-mistral ferrite-model-phi3 \
         ferrite-model-deepseek-v2 ferrite-model-deepseek-v3 ferrite-model-commandr ferrite-models; do
  cargo clean -p $c
done
FERRITE_DISABLE_CUBLAS_GEMM=1 CUDA_PATH=/usr/local/cuda-12.9 cargo build -p vllm-cli --features cuda --release
```

If the build still skips per-arch crates (Compiling line count < 15), the env-var-hook fix isn't taking effect. **Stop and write to STATUS**: build-cache layer issue, needs investigation.

**5.2 Audit with cuBLAS disabled**
```bash
FERRITE_DISABLE_CUBLAS_GEMM=1 ./vllm-rs/target/release/vllm ferrite info -c --color never > /tmp/cublas_off.txt
head -20 /tmp/cublas_off.txt
```

Expected: `Cublas` distinct pick count drops to **0** (or very close). If it doesn't, the env var isn't being read at proc-macro time despite the rustc-env propagation — STOP, write STATUS.

If `no_csv_data` jumped: those picks are now picking CUTLASS variants by linreg extrapolation. Some may be runtime-fragile.

**5.3 Verify correctness with cuBLAS off**
```bash
FERRITE_DISABLE_CUBLAS_GEMM=1 timeout 600s cargo test -p vllm-e2e --release --features e2e,cuda \
  --test e_correctness test_cuda_correctness_command_r_1l -- --ignored --test-threads=1 --nocapture
```
Must pass. If it fails:
- The first thing to check: is it a kernel that simply doesn't support the runtime shape? Many CUTLASS tiles have alignment requirements (N, K must be multiples of 8 or 16). The DP picks based on cost but doesn't validate alignment at codegen time.
- Identify the failing shape via the test output. Add it to the sweep / add a tile that supports it / accept the fallback.

If multiple correctness tests need adjustment, do them one per commit:
- Commit message: `ff-interpreter: cublas-off — fix alignment for <arch> <shape>`

**5.4 Run a broader correctness pass** (if time permits, ~30 min)
Pick 3 more arches that aren't commandr — llama-3.2-3b, qwen2.5-0.5b, gemma3-1b. Run their correctness goldens with FERRITE_DISABLE_CUBLAS_GEMM=1.

**5.5 Commit the FERRITE_DISABLE_CUBLAS_GEMM=1 default**
Once correctness is green across the sample arches, change the default:

```rust
// in impl_lib.rs starter_library()
// CUBLAS-FREEDOM: default off. Re-enable with FERRITE_ENABLE_CUBLAS_GEMM=1
// (note inverted polarity — the new env var ENABLES instead of DISABLES).
if std::env::var_os("FERRITE_ENABLE_CUBLAS_GEMM").is_some() {
    lib.push(Box::new(GemmRefImpl));
}
```

Update the model build.rs files: `forward_env("FERRITE_ENABLE_CUBLAS_GEMM");` (delete the `_DISABLE_` line — A/B polarity is inverted now).

Subject: `ff-interpreter: cuBLAS off by default; FERRITE_ENABLE_CUBLAS_GEMM=1 to re-enable`

---

## Step 6 — Measure the actual perf hit (estimated 2-3h, the deciding number)

**This is the answer to "we'll see"**. The 5-15% I quoted was the per-pick gap on the residual cuBLAS surface. The system-level question is: of total wall-time spent in inference, what fraction is in those residual shapes? If that fraction is small, the system-level regression is small.

**6.1 Choose the workload sample**
Three representative arches the L4 fleet exercises:
- `llama-3.2-3b` (mainstream, GQA, hidden=3072) — body-dominant
- `gemma3-12b-it` (large MLP, head_dim=256, hidden=3840) — gemma family is 53% of cuBLAS surface
- `qwen2-7b` (biased QKV — ALSO covers the qwen2 bias-zoo path Lever A2 hasn't unlocked yet)

For each, measure:
- Decode throughput at batch=1, prompt=128, output=128
- Prefill throughput at batch=1, prompt=2048, output=1
- (Optional) batch=4 decode if the GPU has room

**6.2 Build both states**

State A (cuBLAS enabled, baseline):
```bash
cd vllm-rs && cargo clean -p ferrite-models  # forces re-expansion
FERRITE_ENABLE_CUBLAS_GEMM=1 CUDA_PATH=/usr/local/cuda-12.9 cargo build -p vllm-cli --features cuda,bench --release
cp target/release/vllm /tmp/vllm-cublas-on
```

State B (cuBLAS disabled, the new default):
```bash
cd vllm-rs && cargo clean -p ferrite-models
CUDA_PATH=/usr/local/cuda-12.9 cargo build -p vllm-cli --features cuda,bench --release
cp target/release/vllm /tmp/vllm-cublas-off
```

**6.3 Bench (per `feedback_vllm_bench_feature`)**
```bash
for binary in /tmp/vllm-cublas-on /tmp/vllm-cublas-off; do
  for model in meta-llama/Llama-3.2-3B google/gemma-3-12b-it Qwen/Qwen2-7B; do
    echo "=== $binary · $model · decode ==="
    $binary bench latency --model $model --input-len 128 --output-len 128 --num-iters 5
    echo "=== $binary · $model · prefill ==="
    $binary bench latency --model $model --input-len 2048 --output-len 1 --num-iters 5
  done
done | tee /tmp/cublas_freedom_bench.txt
```

Some of those arches may not be locally cached. If a model fails to load: skip + note in STATUS.

**6.4 Compute the regression**
For each (model, workload), the regression is `(B_us - A_us) / A_us`. Tabulate. The **headline number** is the worst-case regression across the matrix and the AVERAGE regression weighted by typical workload mix (decode 70% / prefill 30% if unsure).

**6.5 Compute the image-size win**
```bash
# Inspect the linker output of state B
ldd /tmp/vllm-cublas-off | grep -i cublas       # should be empty
ls -la /usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublas* | awk '{sum+=$5} END {print sum/1024/1024 " MB"}'
```

If `ldd` still shows `libcublas` linked: some other crate is pulling it in. Investigate: `cargo tree -p vllm-cli --features cuda | grep -i cublas`. cudarc's cublas module is a likely culprit and may need a feature flag.

**6.6 Decision matrix**

Write to `CUBLAS_FREEDOM_PLAN_STATUS.md`:
```
Step 6 — RESULTS
================

Per-model regression (cuBLAS off vs on):
  llama-3.2-3b decode:  +X.X%
  llama-3.2-3b prefill: +Y.Y%
  gemma3-12b decode:    +Z.Z%
  ...

Headline regression (worst-case): +N.N%
Headline regression (avg, decode-heavy weighting): +M.M%

Image-size win:
  libcublas.so + libcublasLt.so total: ~850 MB
  vllm binary linked w/ cublas: K MB
  vllm binary linked w/o cublas: J MB
  Container image saving: 850 MB

Decision:
  if worst-case regression < 5%: SHIP
  if 5-15%: ship gated, document, file follow-ups for the worst arches
  if > 15%: do NOT ship; revisit Step 5 / Step 3 — there's a missing kernel family
```

If "do not ship", the analyzer's `>5% by regime` table tells you which kernel family is missing. Most likely candidates after Steps 1-5: hand-CUDA SIMT for small-M short-K, or a sm89-tuned compute-bound large-M kernel.

**6.7 Final commit**
- Update `CUBLAS_FREEDOM_HANDOFF.md` with results.
- Subject: `ff-interpreter: cuBLAS-freedom shipped — N% worst-case regression, 850 MB saved`
- OR if not shipping: `ff-interpreter: cuBLAS-freedom blocked — <kernel family> needed`

---

## Failure handling

If at any step you'd need to:
- Revert another commit's work
- Disable a registered Impl unconditionally
- Add `#[allow(...)]` to suppress clippy
- Skip a correctness test
- Pull / rebase / merge anything

**Stop. Write to STATUS the section header `ESCALATE — <step>` with the diagnostic. The user will pick it up on next session.**

If a build fails due to cudaforge cache pollution (per `feedback_cudaforge_cache`):
```bash
rm -f ~/.cache/cudaforge/vllm-cuda/libcutlass*.a
rm -f ~/.cache/cudaforge/megakernels/*.cu
touch vllm-rs/crates/vllm-cuda/csrc/cutlass_standalone_gemm.cu
touch vllm-rs/crates/vllm-cuda/build.rs
```
Then retry the build once. If it fails again with the same error, escalate.

---

## STATUS file convention

`CUBLAS_FREEDOM_PLAN_STATUS.md` lives next to this file. Schema:

```markdown
# cuBLAS-Freedom Plan — Status

## Resume point
Next step: <N>

## History
- 2026-MM-DD HH:MM · Step 1 ✓ — captured 1657 picks · commit abc1234
- 2026-MM-DD HH:MM · Step 2 ✓ — captured 12 picks · commit def5678
- ...

## Open blockers
(none) — or — Step X: <description of why it stopped>

## Final results (filled by Step 6)
... (decision table from 6.6)
```

The agent reads this as the FIRST thing on resume, after pre-flight. The agent NEVER deletes prior history entries — only appends.
