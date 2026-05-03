# Real-Fusion Handoff

**Branch:** `no-cublas` (post-Phase-2, uncommitted)
**Off:** `ff-interpreter` @ `5e1ce868b`
**Goal:** Close the ~2.6 ms/iter gap on qwen2.5-3b prefill 2048/1 vs cuBLAS-on baseline by making the IR-level "fused" Cutlass instructions actually fused at the GPU level.

## STATE 2026-05-03 (Phase 2 landed)

**Phase 2 (CutlassFusedQkvRopeCacheGemv, BF16/F16) done.** New kernel `fused_gemv_qkv_rope_cache_neox` + `_interleaved` (in `pos_encoding_kernels.cu`) folds the GEMV + post-GEMV split + rope + paged kv-cache scatter into ONE launch. One block per QKV head (head_size contiguous outputs / block); 4-warp × 32-lane shfl-reduction over K with x[K] cached in smem; rope/cache write happens after the per-block GEMV completes (no inter-block dependency since rope pairs live within a single head). FP8 cache + (N,K) misalignment + `head_dim%4!=0` fall through to the legacy 2-launch (gemv + rope_cache_write) path.

`instr.rs:CutlassFusedQkvRopeCacheGemv` now runs **1 kernel/decode-layer** on the BF16/F16 happy path (was 2: cutlass_gemv → fused_qkv_rope_cache).

**Bench (L4, qwen2.5-3b prefill 2048/1, batch=1, 30 iter, 10 warmup):**
- baseline (`037e63496`): 31.77 ms p50
- Phase 1 (`c29a9965c`): 30.27 ms p50 (−1.50 ms)
- **Phase 2 (this commit): 30.07 ms p50** (−0.20 ms / total −1.70 ms)
- cuBLAS-on: 29.16 ms p50

Gap to cuBLAS-on now **+0.91 ms** (was +1.11 ms post-Phase-1, +2.61 ms originally). Cumulative fusions have closed **65%** of the gap. The Phase 2 200 µs win is right at the 36-layer × ~5 µs launch-overhead estimate (slight overshoot from also eliminating the qkv intermediate's L2 round-trip).

**Smoke:** `vllm chat Qwen/Qwen2.5-3B-Instruct` emits coherent haiku.

## STATE 2026-05-03 (Phase 1 landed)

**Phase 1 (CutlassFusedQkvRopePrefill, BF16/F16) done.** New kernel `fused_qkv_rope_cache_prefill` (NeoX + interleaved) folds rope + paged kv-cache scatter into one launch; `instr.rs:CutlassFusedQkvRopePrefill` now runs gemm + 1 fused-kernel = 2 launches (was 3) on the non-FP8 path. FP8 cache falls through to the old 3-launch path.

**Bench (L4):** no-cublas 30.27 ms p50 (down from 31.77 ms) vs cuBLAS-on 29.16 ms — gap +1.11 ms (was +2.61 ms). Phase 1 closed **57%** of the gap.

**Discovery:** every `Instruction::CutlassFused*` is a multi-kernel sequence at runtime. The "fusion" only collapses IR nodes — the GPU still pays per-op launch overhead + memory round-trips. Confirmed in `crates/ferrite-forward/src/instr.rs` (see line refs below).

## Fake-fusion inventory

| instruction | instr.rs L | launches today | what's already fused | target |
|---|---|---|---|---|
| CutlassFusedQkvRopePrefill | 2058 | ~~3~~ → **2** (gemm + fused rope+cache) | rope+cache (NEW, BF16/F16) | done; Phase 1b: gemm fold |
| CutlassFusedQkvRopeCacheGemv | 1968 | ~~2~~ → **1** (single fused kernel) | gemv+rope+cache (NEW, BF16/F16) | done; FP8 fold deferred |
| CutlassFusedGateUpSiluMul | 1752 | 2 (up_gemm→gate_silu_mul) | gate+silu+mul EVT (real) | 1 kernel |
| CutlassFusedGateUpSiluMulPacked | 1824 | 2 (packed_gemm→silu_mul) | none | 1 kernel |
| CutlassFusedRmsNormGemm | 671 | 2 (rms→gemm) | none | 1 kernel |
| CutlassFusedAddRmsNormGemm | 749 | 2 (add+rms→gemm) | add+rms (real) | 1 kernel |
| CutlassGemmAdd | 1672 | 1 (β=1 epilogue) | already real | n/a |

**Net per qwen2.5-3b iter** (1 prefill + 1 decode, 36 layers):
- Prefill: 36 × (QkvRopePrefill saves 2) + 36 × (GateUpSiluMul saves 1) + 73 × (RmsNorm pre-gemm saves 1) ≈ 180 launches
- Decode: 36 × (QkvRopeCacheGemv saves 1) + 36 × (GateUpSiluMul saves 1) + 73 × rms ≈ 145 launches
- ~325 launches/iter eliminable + HBM round-trips on prefill activations

## Order of attack

Per-iter savings descending. Pick the next phase based on prior measurement.

### Phase 1 — CutlassFusedQkvRopePrefill — DONE
Folded rope + paged kv-cache scatter into one kernel (`fused_qkv_rope_cache_prefill_*` in `pos_encoding_kernels.cu`). Gemm still separate. 2 launches per prefill layer instead of 3. Measured −1.50 ms / iter on qwen2.5-3b prefill 2048/1 (more than the rough ~180 µs estimate — HBM round-trip elimination on the contiguous K/V intermediate read also helps). Phase 1b (fold gemm in too via custom epilogue) deferred — measure other phases first.

### Phase 2 — CutlassFusedQkvRopeCacheGemv (decode equivalent) — DONE
Mirror of Phase 1 for decode M=1. Was 2 launches (gemv + rope_cache_write); now 1. New `fused_gemv_qkv_rope_cache_neox/_interleaved` kernel: 1 block per QKV head, 4-warp × 32-lane shfl-reduction over K with x[K] in smem, in-block rope/cache after gemv (no inter-block dep — rope pairs are within a head). FP8 + (N, K)%8 != 0 + head_dim%4 != 0 still fall through to the legacy 2-launch path. Measured −0.20 ms / iter on qwen2.5-3b prefill 2048/1 — at the 36-layer × ~5 µs launch-overhead estimate.

### Phase 3 — CutlassFusedRmsNormGemm + AddRmsNormGemm
Add an "input scaling prologue" to the basic Cutlass GEMM macro so the row-wise rms norm + per-element norm-weight scale can be applied during the activation read. The norm reduction itself happens in a tiny pre-kernel (scale[N] f32 output) since CUTLASS mainloops can't do cross-CTA row reductions cleanly. Saves the rms_norm kernel launch + the activation HBM write/read round-trip. ~150 µs/iter.

### Phase 4 — CutlassFusedGateUpSiluMul / Packed
Up gemm + gate-with-silu-mul-evt today. Combine to one packed gemm (N = 2*intermediate_size) with silu*mul in the EVT epilogue (the Packed variant is already structurally close). At qwen2.5-3b: ~70 µs/iter.

### Phase 5 — measure remaining gap, decide next
Some cuBLAS-quality kernel hand-tuning may still be needed for the per-launch time gap that's NOT launch overhead.

## Implementation strategy

**Per kernel:**
1. Write a single CUDA `.cu` source extending the existing CUTLASS basic-Gemm macro with the additional ops (rope/cache/silu/etc) folded into prologue (input side) or epilogue (output side).
2. Add `extern "C"` launch wrapper following the existing `cutlass_gemm_<TM>x<TN>_s<S>_launch` naming pattern.
3. Wire into `ferrite-cuda-builder/build.rs` source list + `ferrite-kernels/src/cutlass.rs` extern decl + dispatch.
4. Update the corresponding `Instruction::*` runtime case in `crates/ferrite-forward/src/instr.rs` to call the single kernel instead of multiple.
5. Add per-tile Impl entries in `impl_lib.rs` (already exist for some — they just call separate kernels in their runtime case).
6. Sweep cost rows into `cost_l4_sm89.csv`.
7. A/B bench vs the prior commit.

**Correctness:** for each new fused kernel, write a small standalone test that runs the kernel + the reference (separate-kernels) path, compares output within 1e-3 bf16. Run before any DP/Impl wiring.

## Key files

- `crates/ferrite-forward/src/instr.rs` — runtime impl per Instruction (the "fake fusion" sites)
- `crates/ferrite-forward-macro/src/impl_lib.rs` — Impl entries + matchers
- `crates/vllm-cuda/csrc/cutlass_standalone_gemm.cu` — basic Cutlass GEMM macro family
- `crates/vllm-cuda/csrc/cutlass_gemm_silu_mul.cu` — existing EVT silu+mul template
- `crates/vllm-cuda/csrc/cutlass_gemm_bias.cu` — existing EVT bias template
- `crates/vllm-cuda/csrc/pos_encoding_kernels.cu` — rope kernels
- `crates/vllm-cuda/csrc/cache_kernels.cu` — kv-cache write
- `crates/ferrite-cuda-builder/build.rs` — kernel build orchestration
- `crates/ferrite-kernels/src/cutlass.rs` — extern decls + Rust dispatch

## Open questions

- **CUTLASS EVT for rope:** can rope (cos/sin tables, position-dependent) be expressed as an EVT epilogue? If not, need hand-rolled. Likely hand-rolled — rope is position-dependent and EVT visitors don't naturally take position arrays.
- **EVT for kv-cache write:** writes go to a paged kv-cache layout that's NOT the gemm output layout. Probably requires a custom epilogue iterator OR a separate write inside the same kernel after a __syncthreads.
- **Rms-into-Gemm prologue:** CUTLASS doesn't have a stock input-prologue scale hook. Two options: (a) custom mainloop variant that scales operand A during smem load, or (b) extend the input iterator with a per-row scale tensor. (b) is cleaner.
- **Sweep CSV explosion:** each new fused kernel needs its own CSV rows. Targeted-mode env `FERRITE_SWEEP_NK_FILTER` (added in `675e37ad2`) makes this manageable.

## Risks

- Hand-rolled fused kernels lose CUTLASS's tile auto-tuning. Need to either use CUTLASS templates with custom epilogue/prologue (preferred) or sweep multiple tile shapes manually.
- Per-launch overhead estimate (~5 µs) assumes eager mode. If CUDA Graphs eventually lands, fusion savings shrink (graph eliminates dispatch cost). Fusions still help via HBM round-trip elimination on prefill.
- Multimodal worktree's parallel cudaforge cache races (per memory: shared `.a` files clobber). Mitigation: `XDG_CACHE_HOME=/tmp/no-cublas-cache` for this worktree's builds (used in `675e37ad2`'s session). 

## Test plan per phase

1. Standalone correctness test: kernel output vs separate-kernels reference, atol=1e-3.
2. `vllm chat` smoke on qwen2.5-3b — coherent output prerequisite for any DP wiring.
3. nsys per-iter time-isolated profile (within `bench_iters` NVTX range) — verify launch count drops as expected.
4. `vllm bench latency` p50 — record before/after, target 50%+ of estimated savings as actual.
5. Sweep affected (M, N, K) shapes via `FERRITE_SWEEP_NK_FILTER` so DP picks correctly.

## Reference: gap data (commit 675e37ad2 baseline)

Per-iter compute time (within bench_iters range, qwen2.5-3b prefill 2048/1):
- nc 28.0 ms vs cb 26.4 ms (+1.6 ms)

Distributed across kernels; no single dominant contributor.
