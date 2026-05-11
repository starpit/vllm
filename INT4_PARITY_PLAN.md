# int4 parity with MLX — full ferrite-metal port plan

> **Read [`INT4_PARITY_PROBES.md`](./INT4_PARITY_PROBES.md) first.** It captures the P0 verified facts that supersede early guesses in this plan, including:
> - Scales/biases dtype is **F16**, not BF16
> - Workspace path is `vllm-rs/crates/...`, not `crates/...`
> - 24 ferrite-model crates exist, not 10
> - NAX detect is `macOS 26.2+ runtime + arch_gen ≥ 17` (`≥ 18` for `'p'` arch), not `Apple9 + arch_gen ≥ 13`
> - All sampled mlx-community 4bit checkpoints use uniform `gs=64, bits=4`
> - Embedding is quantized in every checkpoint sampled (with one exception: Gemma-3-MM language_model embedding stored fp)

## Status (2026-05-11)

> **Metal-runtime perf landing**, orthogonal to int4 but unblocks
> the P16 perf comparison: batched ICB exec is now the production
> default for `MetalWorkerPool::run_forward_with_inputs_inner`,
> with one compute encoder per forward (commits `efb2983b7`,
> `c0bec10b7`, `717664b58`). Llama-3.2-1B-Instruct-4bit decode at
> single-prompt greedy goes ~37ms/forward (~27 tok/s) → ~9ms/forward
> (~110 tok/s); Llama-3.2-3B-Instruct-4bit ~23ms/forward (~43 tok/s).
> Root cause of the prior batched-ICB "wrong output for decode
> buckets" was `RuntimeBindings` metadata buffers missing from
> `MetalResidencySet` — fixed in `efb2983b7`. The follow-on goal of
> a single `executeCommandsInBuffer` per forward isn't reachable
> with Apple's compute-ICB API (only `ConcurrentDispatch` available;
> intra-range commands race on RAW), so the production shape is one
> encoder + N `executeCommandsInBuffer` per forward — cross-exec
> ordering on a Serial-default encoder is the RAW-safety guarantee
> we rely on. P16 now has a real baseline to A/B vs `mlx_lm.generate`
> against.



| Phase | State | Commit | Notes |
|---|---|---|---|
| P0 — verification + probes | ✅ done | `a798db0d3` | `INT4_PARITY_PROBES.md` |
| P1 — plumbing (`LinearLayer::AffineQuant` + macro emit) | ✅ done | `4bf6f7ba8` | Plumbed but unreachable until P2 |
| P2 — kernel + cpu_golden parity | ✅ done | `42fececd8` | `quantized_dequantize.metal` + `cpu_golden::affine_dequantize_b4_*`; bit-exact ≤ 1 ULP |
| P2 — E2E (Llama-3.2-1B-4bit coherent) | ✅ done | `c2ba7c459` | Load-time CPU dequant in lieu of forward-time `AffineDequantizeThenGemm`; see deviation note below. **Superseded by P6.** |
| P3 — decode GEMV (`qmv_quad` / `qmv_fast` / `qmv`) | ✅ kernels | `93eb4a846` | Three faithful ports + cpu-parity tests; forward-time swap deferred to P3-P4 integration |
| P4 — prefill GEMM transpose=true | ✅ kernels | `49a51e485` | `qmm_t` + `qmm_t_splitk` ports + dispatcher (`pick_qmm_t_kernel` + split_k heuristic) + cpu-parity tests for aligned / unaligned / splitk |
| P3-P4 C1 — function-constant refactor (ICB readiness) | ✅ done | `cd6eb49ba` | qmv/qmm_t K/N/M moved to `[[function_constant(N)]]` + `ShaderCache::get_pipeline_specialized` + `ConstantValue::Int` variant; standalone parity tests preserved |
| P3-P4 C2 — `Instruction::AffineQmm` + `lower_one` | ✅ done | `405a36931` | Variant + cuda unreachable arm + lowering arm (qmv quad/fast/generic + qmm_t Standard) + 3 lowering-shape unit tests; SplitK still C3 |
| P3-P4 C3 — splitk reduce kernel | ✅ kernel | `1e4426d26` | `splitk_reduce_sum_<dtype>` shader + `MetalSplitKReduce` dispatcher + bf16/f16 parity tests; lowering integration + scratch slot allocation deferred to C4 |
| P3-P4 C4a — `Instruction::SiluMul` + kernel + lowering arm | ✅ done | `6f1592b1a` | New `silu_mul_<dtype>` shader + `Instruction::SiluMul` + `KernelId::SiluMul` + lowering arm + 1 lowering-shape unit test |
| P3-P4 C4b parts 1-4 — Llama-1B-4bit coherent E2E | ✅ done | `9c62889cc`, `02aff6bea`, `de9241fbb`, `e304f8a07` | Forward-time qmv/qmm_t wiring (part 1) + SplitK plumbing + per-layer affine load + tied lm_head (part 2) + per-tile dp slot allocator (part 3) + u32-unaligned mmap-alias fix (part 4 — Apple's M-series driver returns garbage on unaligned u32 binding offsets; mlx-community Llama-3.2-1B-4bit safetensors header is 41161 bytes so data section starts at file pos %4=1). Llama-3.2-1B-4bit + Llama-3.2-3B-4bit both produce coherent end-to-end output. |
| P5 — transpose=false (`qmm_n` / `qvm` / `qvm_split_k`) | ✅ kernels | `907953ff1` | `qmm_n` (extend `quantized_qmm.metal`) + new `quantized_qvm.metal` with `qvm` + `qvm_split_k` + `qouter` helper. `MetalAffineQmmN` + `MetalAffineQvm` dispatchers (pick_qvm_kernel mirrors `quantized.cpp:1444-1453`: K<1024→Standard, K∈[1024,8192]→SplitK(8), K>8192→SplitK(32)). Parity tests: 4 qmm_n cases + 5 qvm cases, all pass. Forward-time wiring (`Instruction::AffineQmm` transpose=false arm in `lower_one`) deferred to first transpose=false model integration (DeepSeek MLA). |
| P6 C1 — `affine_embed` kernel + dispatcher + cpu parity | ✅ done | `5349cfbc6` | Fused gather + dequant kernel co-located in `quantized_dequantize.metal`; 6 instantiations `(f16, bf16) × gs ∈ {32, 64, 128}`. `MetalAffineEmbed` dispatcher + 3 parity tests (incl. partial trailing threadgroup case). |
| P6 C2 — forward-path plumbing | ✅ done | `b473b3a4c` | `AffineQuantEmbedding` struct + `Instruction::AffineEmbed` + `KernelId::AffineEmbed` + `WeightBundleKind::AffineQuantEmbedding` + lowering arm (2D dispatch baking `hidden_size` into function_constant(0)) + worker resolver arm + lowering-shape unit test. CUDA eval is `unreachable!` (metal-only variant). |
| P6 C3 — macro emits AffineEmbed | ✅ done | `9c144a6aa` | `MetalAffineEmbedImpl` (matches OpKind::Embed ∧ Affine storage) + `MetalEmbedImpl::matches` rejects Affine + `is_affine_quant_embedding` type-string check in `field_load_for_accessor` + `FieldLoad::EmbeddingAffine` codegen swap (Embedding::load_affine_dequant → AffineQuantEmbedding::load) + `LinearTiedToEmbedding { affine: Option<(gs, bits)> }` extension + tied-affine codegen arm. |
| P6 C4 — tied lm_head storage gate fix + E2E | ✅ done | `74545207c` | `storage_format_for_weight(lm_head)` returned Dense unconditionally for tied embeds (legacy P2 workaround); now returns Affine when `QuantMethod::Affine`, so the solver picks `MetalAffineQmmImpl` over `MetalGemmImpl` and the tied lm_head emits `LinearLayer::AffineQuant(...)` sharing the embed's packed buffers. Llama-3.2-1B-4bit + Llama-3.2-3B-4bit produce coherent output on the canonical smoke prompts under forward-time embedding lift. |
| P7 — NAX (M4+) | pending | — | |
| P8 — pipeline cache key extension | pending | — | |
| P9 — cpu_golden q4 reference + per-model 4bit goldens | partial | — | C1 lifts cpu_reference matmul refs; C2 captures Llama-3.2-1B-4bit MLX golden. 3B + the remaining mlx-community 4bit families pending. |
| P9 C1 — `cpu_reference` module + lift duplicated test helpers | ✅ done | `accc3d3e4` | New `ferrite-metal-kernels::cpu_reference` module hosts `affine_dequantize_b4_*` + `affine_qmm_t_b4_*` + `affine_qmm_n_b4_*` + `qmv`/`qvm` aliases (HalfF trait dedupes f16/bf16 bodies). Lifted out of `quantized_q{mv,mm,mm_n,vm}_test.rs` (−161/+14 LoC). Lives in metal-kernels because tests can't reach up to `ferrite_forward::cpu_golden`; can re-export later when a non-test consumer needs it. 6 new unit tests + 25 existing kernel parity tests all pass. |
| P9 C2 + P10 — MLX golden + Metal arm in vllm-e2e | partial | `bdd4b2b41` | `scripts/generate_mlx_goldens.py` captures `mlx_lm.generate` greedy decode at temp=0, max_tokens=64 over the canonical 8-prompt sweep. `testdata/golden/llama_3_2_1b_mlx_4bit.json` lands (logprobs included for future use even though `vllm-mlx::worker.rs:1359` doesn't yet populate them). `run_metal_correctness_test_prefix_match` + `test_metal_correctness_llama_3_2_1b_mlx_4bit` in `e_correctness.rs` gate on strict char-prefix match (leading-whitespace-normalized) for `prefix_token_match=5` decoded segments. Baseline: 3 of 8 prompts match the full 64 tokens, 4 of 8 match ≥ 18 tokens before drift, prompt 7 drifts at token 9 ("has been" vs "is often"); 5-token strict bar leaves safety margin. P10 strict ≥ 64-token bar gated on aligning bf16-vs-fp16 activation-dtype path. |
| P10 — Llama-3.2-1B/3B 4bit E2E | partial | `c2ba7c459`, `bdd4b2b41` | 1B coherent (P6 E2E + C2 gate). 3B + token-stream A/B vs `mlx_lm.generate` to ≥ 64 strict tokens pending. |
| P10b — in-register `<T_act, T_scale>` cast (regression repair) | ✅ done (tight scope) | `213313a2a` | Repays the deferral recorded at `INT4_PARITY_PROBES.md:259`. Kernels in `quantized_qmv.metal` / `quantized_qmm.metal` / `quantized_qvm.metal` / `quantized_dequantize.metal` go `<T>` → `<T_act, T_scale>`; symbol naming `affine_<op>_<T_act>_s_<T_scale>_gs_…`; `bfloat × bfloat` instantiations removed (those were the regression path). `MetalAffine*::execute` + lowering call sites take a new `ScaleDtype` axis (single variant `F16` today; forward-compat for P11). New `GpuWeights::take_keep_dtype` skips the loader-side `maybe_cast_cpu`; `AffineQuantLinear::load` + `AffineQuantEmbedding::load` use it for `*.scales` / `*.biases`. `cpu_reference` parametrized on `<TAct, TScale>` and tests synthesize F16 scales for the bf16 path. Validation: 38/38 affine-quant parity tests + `test_metal_correctness_llama_3_2_1b_mlx_4bit` still pass; prompt 4 divergence shifts ≥90→170 chars; `RUST_LOG=ferrite_cuda_core::weights=debug` confirms no F16→BF16 cast on `*.scales` / `*.biases` (33×2048 RMSNorm-gain casts remain — out of P10b's tight scope, deferred to P10c). Strict ≥64-token P10 bar still gated on bf16-vs-fp16 activation-dtype alignment (prompt 7 unchanged: "has been" vs "is often" drift). |
| P10c — RMSNorm F16→BF16 cast cleanup | ✅ done | `4c530bff4` | Out-of-scope follow-up to P10b, same in-register cast pattern. Kernels `rmsnorm_*_specialized` + `fused_add_rmsnorm_*_specialized` (`shaders/rmsnorm.metal`, `shaders/fused_add_rmsnorm.metal`) collapse from two raw `kernel void` definitions into a `<T_act, T_scale>` template body + `INST_RMSNORM` / `INST_FUSED_ARN` macros; symbol naming follows P10b (`rmsnorm_<T_act>_s_<T_scale>_specialized`); `bfloat × bfloat` instantiations removed. `RmsNorm::load` now calls `GpuWeights::take_keep_dtype` instead of `take` on the metal target (cfg-gated; CUDA's `rms_norm_{f16,bf16}` still needs the loader-side dtype-matched cast). Lowering picker plumbed via new `rmsnorm_kernel_static_name<W>` / `fused_add_rmsnorm_kernel_static_name<W>` helpers paralleling `affine_embed_kernel_static_name`. Validation: 71/71 `ferrite-forward --lib` (5 pre-existing P10b stale-test-string assertions also repaired here as adjacent cleanup); `test_metal_correctness_llama_3_2_1b_mlx_4bit` still passes — late-divergence chars unchanged (prompt 4 stays 170, prompt 7 shifts to char 34); `RUST_LOG=trace … chat -p Hi` confirms zero `Cast weight: F16 → BF16` lines for the Llama-3.2 4bit family. Strict ≥ 64-token P10 bar still gated on bf16-vs-fp16 activation-dtype alignment. |
| P11 — mixed-quant loader | pending | — | |
| P12 — q-MLP composition | branch (i) shipped; (ii) pending | — | Branch (i) (decomposed `AffineQmm + AffineQmm + SiluMul`) shipped via P3-P4 C4a. Branch (ii) (`MetalFusedAffineGateUpSiluMulImpl` — packed gate+up q-GEMM + SiluMul epilogue in one dispatch, mirroring the dense `MetalFusedGateUpSiluMulImpl`) deferred until `pipelines.rs` Phase 2 lands (`project_metal_pipelines_rs_must_die`). Visible perf cost: the affine-quant `vllm ferrite info llama 3.2 3b mlx` decode loop is `(AffineQmm, AffineQmm, SiluMul)` — 3 dispatches per MLP — vs the cuda L40s prefill loop's single `FusedGateUpSiluMul`. |
| P13 — MoE | pending | — | |
| P14 — NAX MoE | pending | — | |
| P15 — long-prompt / long-decode under q4 | pending | — | Intersects `project_metal_long_decode_panic` |
| P16 — A/B vs MLX + perf gate | partial baseline | `efb2983b7`, `c0bec10b7`, `717664b58`, `8bf6388f8`, `2444c844c`, `a829749bc`, `487e11a01` | Metal-runtime perf overhaul lands: residency-set fix unblocks batched ICB, batched ICB becomes default, then one encoder per forward replaces the per-step end+reopen pattern. Llama-3.2-1B-4bit decode 27 tok/s → 110 tok/s; Llama-3.2-3B-4bit ~43 tok/s. **Open gaps observed against `vllm ferrite info` cuda L40s baseline**: (a) metal lacks any `Fused Qkv*RopeCache*Impl` (decode + prefill) — both BF16 dense and affine paths show 3 separate Q/K/V matmuls + `RopeAppend` instead of one fused dispatch; (b) affine q-MLP runs 3 dispatches vs cuda's 1 `FusedGateUpSiluMul` (gated on P12 branch (ii)); (c) no metal `FusedGemmAdd` for the output-proj + residual chain. Concrete loop-body op counts on Llama-3.2-3B-4bit: cuda L40s TP=8 prefill loop = 9 ops; metal affine prefill loop = 13 ops. **Startup-time on `mlx-community/Llama-3.2-3B-Instruct-4bit`**: pre-fix cold `try_load` ≈ 1.34 s / warm ≈ 460 ms. After `2444c844c` candidate (1) (dtype-relaxed gate): 411–990 ms warm. After `a829749bc` candidate (2)+(3) (register-time MTLBlit DMA bulk copy with safetensors-aware shift): **`try_load` 21 ms consistent**, `GpuWeights::from_dir` ~302 ms (where the bulk blit now lives), total weight-loading ~323 ms. Routing: `zero-copy 648 / 1723.7 MiB | memcpy 1 / 32 MiB (outside-mmap, tied lm_head shadow)`. The shift trick puts every canonical-layout safetensors tensor at `mod 16 == 0` post-blit so the strict gate passes — the dtype-relaxed counter from candidate (1) now reads 0. RAM: `weights+overhead` 3.4 GiB → 1.9 GiB (duplicate mmap-noCopy + arena buffers collapse into one aligned_buffer/shard); `kv_budget` 11.2 GiB → 12.7 GiB. Llama-3.2-1B-4bit `try_load` 13 ms. **A/B vs `mlx_lm.generate` captured** (M4, 128 max-tokens greedy): Llama-3.2-1B-4bit ferrite 108 tok/s vs MLX 132.7 tok/s (ferrite at 81% of MLX); Llama-3.2-3B-4bit ferrite 45.7 tok/s vs MLX 52.5 tok/s (87%). P16 gate is ≥95%; remaining gap is ~1.8ms/token on 1B, ~2.9ms/token on 3B (per-layer ~100µs), CPU encode already ~70-80µs/forward so the bulk is GPU compute. **Per-shape kernel benchmarks landed** via `487e11a01`: `ferrite-metal-cost-sweep` migrated to objc2-metal, new `affine_qmv` + `affine_qmm_t` sweeps emit 2184 rows covering every mlx-community 4bit Linear shape (Llama-1B/3B, Qwen2-7B, Qwen3-{1.7,4,7}B, Mistral-7B, Gemma-2-2B, Gemma-3-{1,4}B) at M ∈ {1, 8, 16, ..., 2048}. `m4_with_costs()` loads `profiles/cost_m4.csv`; `MetalAffineQmmImpl::cost_us` reconstructs the dispatcher pick and reads the matching row. Solver-pick currently unchanged (no competing impl yet); becomes load-bearing under P7. Tracking: `project_metal_fused_qkv_handoff` for QKV-fusion (next major perf bite). |
| P17 — FP-quant mode (mxfp4 / mxfp8 / nvfp4) | pending | — | Parallel parity track |

**P2 deviation worth carrying forward.** The plan called for forward-time
`Instruction::AffineDequantizeThenGemm` with a per-Linear scratch arena
slot. Load-time CPU dequant ships in `c2ba7c459` instead: the kernel
landing (`42fececd8`) proves the affine math against `cpu_golden::
affine_dequantize_b4_*` (≤ 1 ULP) in isolation, and the load path
materializes the same math into a BF16 Dense `LinearLayer` at load
time — every fuser in the metal solver matches as if the model were
plain BF16. Same correctness gate, simpler infra, ~2 GB extra arena
on Llama-3.2-1B (fits with headroom on 24 GiB; 3B not yet validated
under this scheme). The forward-time swap waits on P4 because every
Llama Linear sees prefill (M >= vector_limit, routes to qmm_t) AND
decode (M=1, routes to qmv) — flipping the macro before P4 lands
would break prefill. P3 ships the qmv kernels + standalone parity
tests; the macro flip + FUF revert + Instruction::AffineQmm wiring
land in the P3-P4 integration commit once `qmm_t` arrives.

**P3 kernel landing.** `quantized_qmv.metal` ports MLX's `affine_qmv_quad`
(`quantized.h:1444`), `affine_qmv_fast` (`:1496`), and `affine_qmv`
(`:1548`) verbatim, including the `load_vector` / `qdot` / `qdot_safe`
helpers (`:28-392`) and `adjust_matrix_offsets` (`:1351-1387`).
Instantiation grid: bits=4 × gs∈{32,64,128} × dtype∈{f16, bf16} ×
batched∈{0,1} (qmv_fast / qmv) plus D∈{64, 128} (qmv_quad) — 48
exported symbols. `MetalAffineQmv` dispatcher in
`ferrite-metal-kernels::quantized` mirrors `dispatch_qmv`
(`quantized.cpp:1365`) + the inner `qmv_fast`/`qmv` pick (`:259`).
`KernelId::AffineQmv{Quad,Fast,}` enum variants land for
diagnostics. `tests/quantized_qmv_test.rs` exercises all three
kernels against a CPU reference (dequantize-then-matmul); the noise
floor is bounded by `sqrt(K) × bf16_eps × max_per_elem_magnitude`
with a 4× safety factor.

**P4 kernel landing.** `quantized_qmm.metal` ports MLX's `affine_qmm_t`
(`quantized.h:1707`) and `affine_qmm_t_splitk` (`:1780`) with the
`qmm_t_impl` body (`:1094`) reproduced inline. The steel BlockMMA /
BlockLoader / QuantizedBlockLoader trio is inlined (not vendored)
to stay in the validated grid-X regime that
`fused_gate_up_silu_mul_gemm_steel_*_specialized` already operates
in — the prior vendored-steel attempt hit a deterministic wall at
grid_x ≥ 1024 (per `project_metal_gemm_port_dead_end`); Llama-1B/3B
q4 prefill shapes have N ≤ 8192 → grid_x ≤ 256, well inside the
safe regime. Tile constants match MLX (BM=BN=BK=32, WM=WN=2; BK=32
keeps every BK iter aligned to a single quant group for gs ∈ {32,
64, 128}). Instantiation grid: bits=4 × gs∈{32,64,128} ×
dtype∈{f16, bf16} × aligned_N∈{true, false} — 24 exported symbols
across qmm_t (batched=0) + qmm_t_splitk. `MetalAffineQmmT`
dispatcher mirrors the `quantized.cpp:1411-1424` matmul-branch rule
plus the `qmm_splitk` heuristic at `:788-805` (target ~512
threadgroups, capped by K/group_size, K-divisibility-guarded
fallback to `Standard`). `KernelId::AffineQmmT{,SplitK}` variants
land for diagnostics. `tests/quantized_qmm_test.rs` covers aligned
(N % 32 == 0) / unaligned / splitk paths; splitk test reads the
`[split_k, M, N]` intermediate and sum-reduces in CPU (mirroring
`strided_reduce_general_dispatch`). Forward-time wiring (macro flip
from load-time dequant to `Instruction::AffineQmm` + `lower_one`
dispatch tree covering both qmv and qmm_t branches + downstream
reduce for splitk) lands in the next integration commit, gated on
both P3 and P4 kernels existing.

**Mandate.** 100% parity with MLX's int4 (`affine` mode) quantization across every kernel, every model class (dense + MoE), every backend variant (standard + NAX/M4+). No omitted kernels, no skipped models. Sequencing prioritizes; nothing is dropped. The only out-of-scope item is `fp_quantized.metal` (NVFP4 / MXFP8 / MXFP4) — this is *production* in MLX (`jit_kernels.cpp:861`), not experimental, but it's a different mode with different weight layouts, captured as Phase 17 (parallel parity track).

**Hard rules carried in** (memories):
- `feedback_ferrite_metal_mlx_only` — every kernel is a faithful port; no inventions.
- `feedback_no_shortcuts_kernels` — copy MLX kernels verbatim; minimal changes only.
- `feedback_no_invented_files` — files map 1:1 to MLX (`quantized.h` → `quantized.metal` shader; `quantized.cpp` → `quantized.rs` dispatcher).
- `feedback_match_python_exactly` — same tile sizes, dispatch heuristics, instantiation grid.
- `feedback_no_into_gpu_tensor` — every weight bundle stays an `OwnedTensor`-equivalent; no CUDA-tensor leakage.
- `feedback_no_handcoded_fusion` — see Phase 12. Intrinsic dequant-inside-matmul is not "handcoded fusion"; cross-op fusion (q4 GEMM + SwiGLU into one kernel) is.
- `feedback_no_reinvent_testing` — extend `cpu_golden` + `vllm-e2e` goldens.
- `feedback_never_modify_forward_paths` does **not** apply (Metal forward is macro-generated, not a hand-rolled forward; changes are additive in `lower_one`).

---

## Dependency graph

```
                 ┌──────────────────────────┐
                 │ P0 Verification + Probes │
                 └────────────┬─────────────┘
                              │
                              ▼
            ┌────────────────────────────────────┐
            │ P1 Plumbing: LinearLayer + Loader  │
            │  AffineQuant arm, WeightTensor +,  │
            │  WeightBundleKind +, macro emit    │
            └────────┬───────────────────┬───────┘
                     │                   │
                     ▼                   ▼
       ┌──────────────────┐    ┌────────────────────┐
       │ P2 affine_dequant│    │ P11 mixed-quant    │
       │   (slow ref) +   │    │     loader         │
       │   AffineDeqGemm  │    │ (RMSNorm bf16,     │
       │   instr lowering │    │  tied lm_head etc.)│
       └────────┬─────────┘    └────────┬───────────┘
                │                       │
                ▼                       │
   ┌────────────────────┐               │
   │ P3 Decode GEMV     │               │
   │  qmv_quad / qmv /  │               │
   │  qmv_fast          │               │
   └────────┬───────────┘               │
            │                           │
            ▼                           │
   ┌────────────────────┐               │
   │ P4 Prefill GEMM-T  │               │
   │  qmm_t aligned/un  │               │
   │  qmm_t_splitk      │               │
   └────────┬───────────┘               │
            │                           │
            ▼                           │
   ┌────────────────────┐               │
   │ P5 Transpose=false │               │
   │  qmm_n, qvm,       │               │
   │  qvm_split_k       │               │
   └────────┬───────────┘               │
            │                           │
            ▼                           │
   ┌────────────────────┐               │
   │ P6 Quant embedding │◀──────────────┘
   │  affine_dequant +  │
   │  index gather      │
   └────────┬───────────┘
            │
            ▼
   ┌────────────────────────────────────┐
   │ P7 NAX (M4+) — qmm_t_nax/qmm_n_nax │
   │   + is_nax_available detect        │
   │   + dual-codegen pipeline cache    │
   └────────┬───────────────────────────┘
            │
            ▼
   ┌────────────────────┐
   │ P8 Pipeline cache  │
   │   key extension    │  (parallel; can land alongside P3)
   └────────┬───────────┘
            │
            ▼
   ┌────────────────────────────┐
   │ P9 cpu_golden q4 reference │
   │ + per-model 4bit goldens   │
   └────────┬───────────────────┘
            │
   ┌────────▼─────────────────────┐
   │ P10 Llama-3.2-1B/3B 4bit E2E │
   └────────┬─────────────────────┘
            │
            ▼
   ┌────────────────────────────────────┐
   │ P12 Quantized MLP — composition    │
   │  branch (i) compose AffineQmm +    │
   │  silu_mul OR (ii) ported fused MLP │
   └────────┬───────────────────────────┘
            │
            ▼
   ┌────────────────────────────────────────┐
   │ P13 MoE: gather_qmm/qmv/qvm + rhs path │
   └────────┬───────────────────────────────┘
            │
            ▼
   ┌────────────────────────┐
   │ P14 NAX MoE variants   │
   │  gather_qmm_*_nax      │
   └────────┬───────────────┘
            │
            ▼
   ┌────────────────────────────────────┐
   │ P15 Long-prompt / long-decode      │
   │   under q4 + decode-panic interact │
   └────────┬───────────────────────────┘
            │
            ▼
   ┌────────────────────────────┐
   │ P16 A/B vs MLX + perf gate │
   └────────────┬───────────────┘
                │
                ▼
   ┌──────────────────────────────────┐
   │ P17 FP-quant mode (mxfp4/mxfp8/  │
   │      nvfp4) — parallel track     │
   └──────────────────────────────────┘
```

**Critical path** (Llama-1B q4 first token): **P0 → P1 → P2 → P3 → P6 → P9 → P10**. Seven phases.
**Critical path** (Llama-1B q4 decode + prefill, all matmul shapes): adds **P4 → P5 → P12**.
**Critical path** (Qwen3-MoE q4): adds **P11 → P13**.
**Critical path** (M4+ perf parity): adds **P7 → P14**.
**Critical path** (declared parity, all shapes/all archs/all models): full chain through **P16**.

---

## Complete kernel inventory (MLX side)

Source files: `~/git/mlx/mlx/backend/metal/kernels/quantized.{h,metal}` + `quantized_nax.{h,metal}` + `quantized_utils.h` + dispatcher `mlx/backend/metal/quantized.cpp`.

### `affine` mode kernels — primary parity target

| Kernel | Source line (`quantized.cpp`) | Purpose | Tile / dispatch | Templates |
|---|---|---|---|---|
| `affine_quantize` | `:1657` | Host: pack fp→u32 | `nthreads = w.size() / per_thread`; `per_thread = group_size / 32` | `(type, group_size, bits)` |
| `affine_dequantize` | `:1657` | Host: u32→fp | `nthreads = out.size() / packs_per_int` | same |
| `affine_quantize_dequantize` | `:1564` | Round-trip "fake-quant" of activations (used by `QQMatmul` for x) | as quantize | same |
| `affine_qmv_quad` | `:177` | Decode matvec, K∈{64,128}, pow2 bits | `(simd, 1, 1)` group; grid `(M, ⌈N/64⌉, B)` | `(type, gs, bits, D, batched)` |
| `affine_qmv_fast` | `:235` (path inside `qmv()`) | Decode matvec, `N%8==0 && K%512==0` | `(32, 2, 1)` group; grid `(M, ⌈N/8⌉, B)` | `(type, gs, bits, batched)` |
| `affine_qmv` | `:235` | Decode matvec generic | same dispatch shape as `qmv_fast` | same |
| `affine_qmm_t` | `:680` | Prefill matmul, transpose=true | `(32, 2, 2)` group; grid `(⌈N/32⌉, ⌈M/32⌉, B)`; bm=bn=32 | `(type, gs, bits, aligned, batched)` |
| `affine_qmm_n` | `:680` | Prefill matmul, transpose=false | same | `(type, gs, bits, batched)` |
| `affine_qmm_t_splitk` | `:774` | B=1 small-M splitk | `(32, 2, 2)`; grid `(n_tiles, m_tiles, split_k)`; split_k targets ~512 tgs | `(type, gs, bits, aligned)` |
| `affine_qvm` | `:419` | Vector × matrix, transpose=false, K<1024 | `(32, 2, 1)`; bn = `min(gs, 32) * 2` | `(type, gs, bits, batched)` |
| `affine_qvm_split_k` | `:298` | Same, K≥1024; `split_k = 8` (K≤8192) or `32` | grid `(M, N/bn, B*split_k)` | `(type, gs, bits, split_k)` |
| `affine_gather_qmm_t` | `:869` | MoE prefill matmul transpose=true | bm=bn=32, wm=wn=2 | `(type, gs, bits, aligned)` |
| `affine_gather_qmm_n` | `:869` | MoE prefill matmul transpose=false | same | `(type, gs, bits)` |
| `affine_gather_qmv_fast` | `:960` | MoE decode matvec | as `qmv_fast` | `(type, gs, bits)` |
| `affine_gather_qmv` | `:960` | MoE decode matvec generic | same | `(type, gs, bits)` |
| `affine_gather_qvm` | `:1026` | MoE transpose=false matvec | as `qvm` | `(type, gs, bits)` |
| `affine_gather_qmm_rhs_nt` | `:1215` | Sorted-MoE rhs-gather, transpose=true (hot path for `M==1, B≥16, right_sorted`) | bm=16, bn=32, bk=32, wm=1, wn=2 | `(type, gs, bits, bm, bn, bk, wm, wn, transpose=true)` |
| `affine_gather_qmm_rhs_nn` | `:1215` | Same, transpose=false | same tile | same with `transpose=false` |

### `affine` NAX (M4+) kernels

Header: `quantized_nax.h` (1680 lines). Instantiation: `quantized_nax.metal`. All NAX kernels use bm=bn=bk=64, wm=wn=2 — different MMA path (`steel/gemm/nax.h`).

| Kernel | Source line | Purpose |
|---|---|---|
| `affine_qmm_t_nax` | `:473` (`qmm_nax`) transpose=true | NAX prefill matmul transpose=true, gated `K%64==0 && (tf32 || dtype!=f32)` |
| `affine_qmm_n_nax` | `:473` transpose=false | NAX prefill matmul transpose=false |
| `affine_gather_qmm_t_nax` | `:576` (`gather_qmm_nax`) | NAX MoE matmul transpose=true |
| `affine_gather_qmm_n_nax` | `:576` | NAX MoE matmul transpose=false |
| `affine_gather_qmm_rhs_nax_nt` | `:1084` (`gather_qmm_rhs_nax`) | NAX sorted-MoE rhs-gather transpose=true |
| `affine_gather_qmm_rhs_nax_nn` | `:1084` | Same, transpose=false |

NAX has **no** decode (qmv) or `qvm` variants — small-M paths route to non-NAX.

### Dispatcher heuristics (`QuantizedMatmul::eval_gpu` `:1387`, `GatherQMM::eval_gpu` `:1456`)

Routing rules to mirror byte-for-byte (per `feedback_match_python_exactly`):

1. **`get_qmv_batch_limit(D, O, d)`** at `:84` — tabular over `(arch_gen, arch_size, D≤2048, D≤4096, else)`. Returns vector_limit ∈ {6, 10, 12, 14, 18, 32}.
2. **Top-level rule (`QuantizedMatmul`)**: `vector_limit = transpose ? get_qmv_batch_limit(K,N,d) : 4`. If `M ≥ vector_limit`: matmul branch (`qmm_splitk` if transpose+B==1 else `qmm`). Else: matvec branch.
3. **Matvec transpose=true** (`dispatch_qmv` `:1365`): `K∈{64,128} && is_pow2(bits)` → `qmv_quad`, else → `qmv` (which internally chooses `qmv_fast` if `N%bn==0 && K%512==0` for bn=8).
4. **Matvec transpose=false**: `K<1024` → `qvm`, else → `qvm_split_k`.
5. **NAX gate inside `qmm()`** (`:695`): `is_nax_available() && transpose && K%64==0 && (tf32 || dtype!=f32)` → `qmm_nax`, else fall through to standard `qmm`.
6. **GatherQMM rule**: `M==1 && B≥16 && right_sorted && B/E≥4` → `gather_qmm_rhs` (NAX-aware). Else `M ≥ vector_limit` → `gather_qmm`. Else if transpose → `gather_qmv`. Else → `gather_qvm`.

### Group sizes / bits / dtypes instantiated (`quantized.metal:140-156`)

```
group_size ∈ {32, 64, 128}
bits       ∈ {2, 3, 4, 5, 6, 8}
dtype      ∈ {float, float16_t, bfloat16_t}
```

NAX has the same dimensions (`quantized_nax.metal:88-104`).

### `fp_quantized` mode — Phase 17

`fp_quantized.metal:147-150`: instantiated for `(nvfp4, gs=16, b=4)`, `(mxfp8, gs=32, b=8)`, `(mxfp4, gs=32, b=4)`. Same kernel surface as affine (qmv_quad / qmv_fast / qmv / qmm_t / qmm_n / qmm_t_splitk / qvm / qvm_split_k / gather_qmm{,_t,_n,_rhs} / gather_qmv{,_fast} / gather_qvm), instantiated only over fp16/bf16/f32. Different in-kernel dequant math (FP minifloat unpack, no per-group bias). Wired through the same `QuantizedMatmul::eval_gpu` dispatcher via `mode` discriminator. **In scope as a separate parity track** (Phase 17), not folded into int4.

---

## Complete model coverage (ferrite-metal side)

### Models in tree

`vllm-rs/crates/ferrite-model-{llama, qwen2, qwen2-vl, qwen3, mistral, gemma2, gemma3, gemma3-mm, deepseek-v3, deepseek-v3-flat}` — 10 model crates. Of these, only **llama** has a `metal_pool` wired today (`ferrite-model-llama/src/lib.rs:40,70`); other model crates have no metal-gated code (verified: grep for `metal_pool|cfg.*metal` in qwen2, qwen3 returned nothing).

Coverage matrix (what each model exercises once int4 is wired):

| Model class | Linear shape | Embedding | Kernels exercised | Canonical 4bit ckpt | Golden file (new) |
|---|---|---|---|---|---|
| Llama-3.2-1B | dense, transpose=true | quantized | qmv_fast, qmm_t, qmm_t_splitk | `mlx-community/Llama-3.2-1B-Instruct-4bit` (gs=64) | `llama_3_2_1b_mlx_4bit.json` |
| Llama-3.2-3B | dense | quantized | same + larger shapes | `mlx-community/Llama-3.2-3B-Instruct-4bit` | `llama_3_2_3b_mlx_4bit.json` |
| Qwen2-7B | dense | needs verification (P0) | same | `mlx-community/Qwen2-7B-Instruct-4bit` | `qwen2_7b_mlx_4bit.json` |
| Qwen3-1.7B/4B/8B dense | dense | TBD | same | `mlx-community/Qwen3-{1.7B,4B,8B}-Instruct-4bit` | `qwen3_dense_mlx_4bit.json` |
| Qwen3-MoE-30B-A3B | gather (MoE) | TBD | gather_qmm_rhs (sorted decode hot path), gather_qmm, gather_qmv | `mlx-community/Qwen3-30B-A3B-Instruct-4bit` | `qwen3_moe_mlx_4bit.json` |
| Mistral-7B-v0.3 | dense | TBD | same as Llama | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | `mistral_7b_mlx_4bit.json` |
| Mixtral-8x7B | gather (MoE) | TBD | full gather suite | `mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit` | `mixtral_8x7b_mlx_4bit.json` |
| Gemma-2-2B/9B | dense | tied | same + tied lm_head case | `mlx-community/gemma-2-{2b,9b}-it-4bit` | `gemma2_mlx_4bit.json` |
| Gemma-3-1B/4B | dense | tied | same | `mlx-community/gemma-3-{1b,4b}-it-4bit` | `gemma3_mlx_4bit.json` |
| Gemma-3-MM (vision) | dense | tied + non-quant vision | same + vision encoder dense | `mlx-community/gemma-3-4b-it-4bit` (multimodal repo) | `gemma3_mm_mlx_4bit.json` |
| DeepSeek-V3 (MLA) | dense + gather (MoE) | quantized | full (incl. transpose=false on MLA absorb) | `mlx-community/DeepSeek-V3-...-4bit` | `deepseek_v3_mlx_4bit.json` |

Per-model verification list lives in P0.

### Existing CUDA-side goldens (already present in `vllm-rs/crates/vllm-e2e/testdata/golden/`)

`llama_3_2_1b_awq.json`, `llama_3_2_1b_bnb_4bit.json`, `gemma2_2b_awq.json`, `gemma2_2b_gptq.json`, `gemma2_2b_w4a16_ct.json`, `granite_3_2b_bnb_4bit.json`, `granite_3_1_2b_gptq.json` — these validate **CUDA** quantization stacks (Marlin / GPTQ / AWQ / BNB), not MLX-affine. None are reusable as Metal goldens because they capture different output token streams (different rounding from different quant schemes). New `*_mlx_4bit.json` goldens are required.

---

## Plumbing surface (ferrite-metal side)

Citations are to current `worktree-ferrite-metal` HEAD `4efc5c17f`.

### `LinearLayer` enum (`ferrite-kernels/src/layers.rs:738`)

Current arms: `Dense(Linear)`, `Marlin(Box<MarlinLinear>)`, `Ggml(Box<GgmlLinear>)`, `GgmlConcat(Vec<GgmlLinear>)`, `Bnb4bit(Box<Bnb4bitLinear>)`, `Fp8(Box<Fp8Linear>)`, `Fp8Block(Box<Fp8BlockLinear>)`. All quant arms hold **CUDA** types (`ferrite_cuda_core::tensor::GpuTensor`); `dense_weight()` at `:858` panics on every non-Dense variant — comment at `:855` is the structural gate: *"Quant variants are unreachable under metal (the macro only emits Dense LinearLayer's on that path)"*.

**Required change** (P1): add `AffineQuant(Box<AffineQuantLinear>)` arm. The struct holds `OwnedTensor`-equivalent Metal buffers (raw `Buffer = Retained<ProtocolObject<dyn MTLBuffer>>` per `weights.rs:17`), not CUDA tensors. The accessor extends from `dense_weight()` to four:
- `affine_weight()` → packed u32 buffer
- `affine_scales()` → fp16/bf16 buffer
- `affine_biases()` → fp16/bf16 buffer (the *per-group offset*; not the linear-layer bias)
- `linear_bias()` → optional fp16/bf16 buffer (the actual layer bias)

Naming: `affine_biases()` vs `linear_bias()` keeps the MLX-source distinction visible in code. The `feedback_no_into_gpu_tensor` constraint is satisfied trivially because Metal never had `into_gpu_tensor()` — these are all `Buffer`s.

### `WeightTensor` enum (`ferrite-forward/src/interpreter/metal/lowered.rs:214`)

Current: `Weight`, `Bias` (2 arms). Extend (P1) to: `Weight`, `Bias`, **`AffineScales`**, **`AffineBiases`**, **`AffineLinearBias`** — the last reuses `Bias` semantics for the optional layer bias on a quantized linear (avoids name collision with `AffineBiases` per the prior fork's "biases" gotcha).

### `WeightBundleKind<W>` (`lowered.rs:202`)

Current: `Embedding`, `RmsNorm`, `LinearLayer`, `CosSin`. No change required for matmul (the `LinearLayer` arm will route to `AffineQuant` polymorphically via the worker's resolver). For embedding (P6) we need a new variant that resolves into the embedding's quantized weight + scales + biases — call it `QuantEmbedding`. (The default `Embedding` arm stays for unquantized cases — Llama could ship as either.)

### Worker resolver (`worker.rs:1407 weight_for_bundle`)

Current `:1414` `LinearLayer` arm always calls `l.dense_weight()`. Extend (P1):

```
WeightBundleKind::LinearLayer(wtfn) => {
    let l = (wtfn)(weights, layer);
    match (which, l) {
        (WeightTensor::Weight,             LinearLayer::Dense(d))            => d.weight,
        (WeightTensor::Weight,             LinearLayer::AffineQuant(q))     => q.weight,
        (WeightTensor::AffineScales,       LinearLayer::AffineQuant(q))     => q.scales,
        (WeightTensor::AffineBiases,       LinearLayer::AffineQuant(q))     => q.affine_biases,
        (WeightTensor::AffineLinearBias,   LinearLayer::AffineQuant(q))     => q.linear_bias.expect(...),
        (WeightTensor::Bias,               LinearLayer::Dense(d))            => d.bias.expect(...),
        // any other combo => MissingBias / type mismatch panic
    }
}
```

### `Instruction<W>` (`ferrite-forward/src/instr.rs:339`)

Current dense: `Gemm(in, out, layer, wtfn, n, k)`. Add (P2-P5):

- `AffineDequantizeThenGemm(in, out, layer, wtfn, n, k, gs, bits)` — slow ref (P2)
- `AffineQmm(in, out, layer, wtfn, n, k, gs, bits, transpose)` — fast (P3-P5); the lowering pass picks the right kernel variant (qmv_quad/qmv/qmm_t/qmm_t_splitk/qvm/qvm_split_k/qmm_n) at lowering time per the dispatcher heuristic
- `AffineGatherQmm(in, out, layer, wtfn, lhs_idx_slot, rhs_idx_slot, m, n, k, gs, bits, transpose, sorted)` — MoE (P13)
- `AffineEmbed(out, wtfn)` — quant embedding lookup (P6)

Rationale for one `AffineQmm` covering 7 underlying kernels: the dispatcher decisions (qmv_quad vs qmv_fast vs qmm_t vs ...) depend on `(M, N, K, transpose, bits, group_size, batched, NAX-availability)` — exactly the data lower_one has. Putting the dispatcher in `lower_one` (the analog to MLX's `QuantizedMatmul::eval_gpu`) keeps the macro's scheduler oblivious to per-shape kernel splits and matches MLX's split-of-concerns. Per `project_metal_pipelines_rs_must_die.md`, this *should* eventually move into the macro/solver, but in the interim adding it to `lower_one` matches every other multi-variant pick already present (`FusedGateUpSiluMul` decode-vs-steel pick at `lowering.rs:356`).

### `KernelId` (`lowered.rs`, scan for the enum)

Add (P3-P5): `AffineQmvQuad`, `AffineQmvFast`, `AffineQmv`, `AffineQmmT`, `AffineQmmTSplitK`, `AffineQmmN`, `AffineQvm`, `AffineQvmSplitK`, `AffineDequantize`. P7 adds NAX variants (`AffineQmmTNax`, `AffineQmmNNax`). P13 adds `AffineGather*` (8 IDs). Total new `KernelId` variants: ~17.

### Pipeline cache key (`ferrite-metal-kernels/src/specialized_pipeline_cache.rs:143`)

Current `with_standard_shaders()` registers libraries by name only; per-pipeline keys carry `(KernelId, dtype, function_constants)`. For affine kernels the key must additionally include `(group_size, bits, batched, aligned, transpose)`. NAX adds an axis (`backend ∈ {standard, nax}`). Phase 8 extends `PipelineKey` to carry `group_size: u8, bits: u8, transpose: bool, aligned: bool, batched: bool, backend: PipelineBackend`. Also: register two new metallibs:

- `affine_quantized` — single shader with all `qmv*/qvm*/qmm*/dequantize` kernels
- `affine_quantized_nax` — NAX variants
- `affine_quantized_gather` — gather variants
- `affine_quantized_gather_nax` — NAX gather variants

Each maps to an `embedded_metallib!()` invocation in `with_standard_shaders()` (`specialized_pipeline_cache.rs:147`), an entry in `build.rs` to compile the `.metal` to `.metallib`, and a directory of source `.metal` files under `shaders/`.

### Macro-side emission

The macro decides per-layer which `LinearLayer` variant to materialize. `ferrite-forward-macro/src/impl_lib.rs` is the loader factory. Currently it emits `Dense(Linear { weight, bias: opt })` from safetensors keys `<prefix>.weight` + `<prefix>.bias` for every linear. Extend (P1) to:

- Detect `quantization_config` in `config.json` (MLX format: `{"group_size": 64, "bits": 4}`).
- If quantization is per-layer scoped (which it usually is), inspect each linear's safetensors keys: if `<prefix>.weight` exists with dtype `"U32"` AND `<prefix>.scales` AND `<prefix>.biases` (the per-group affine offset), emit `AffineQuant`. Else emit `Dense`.
- Per-layer `AffineQuant` emission means a "mixed" model is supported automatically — RMSNorm gains stay bf16, and a config that quantizes only some layers (some 4bit configs leave router/gate weights unquantized) Just Works.

The existing AWQ macro stub at `ferrite-forward-macro/src/metal/awq.rs:114` (`return None`) is dead — remove in P1 cleanup. Replace with a faithful Affine matcher (or, more likely, fold AffineQuant directly into the canonical Linear loader path; the impl-trait MatchInfo machinery is overkill here since solver doesn't get to choose between dense and affine — that's determined at load by safetensors layout).

### Safetensors loader (`ferrite-metal-kernels/src/weights.rs`)

Already content-blind: it reads `dtype` as a `String` (`weights.rs:31,103`), allocates a `Buffer` by byte count (`:146`), and stores in a `HashMap<String, Buffer>`. **No changes required** — adding new dtype strings (`"U32"` for packed weights) is automatic. The byte-size calc (`end - start` at `:128`) is dtype-blind too.

P0 verifies the safetensors `dtype` string MLX uses for packed u32 — almost certainly `"U32"`, but should be confirmed against an actual checkpoint.

### Forward-emission paths

The metal forward is generated by the macro from the canonical config. New `Instruction::AffineQmm` / `AffineEmbed` variants are emitted at the `Linear::forward` / `Embedding::forward` lowering points respectively. **No model crate changes** — this is purely a macro-emission change for the Metal target. The 10 model crates compile against the same `Instruction<W>` enum and the macro handles the rest.

---

## Phases

### P0 — Verification + probes (S)

**Deliverables**

1. **Confirm MLX safetensors layout for affine 4bit.** Pick `mlx-community/Llama-3.2-1B-Instruct-4bit`. List safetensors keys, dtypes, shapes for one decoder layer. Specifically resolve:
   - dtype string for packed weights — `"U32"` vs `"I32"` vs `"UINT32"`? (loader matches by string equality at `weights.rs:103`).
   - Shape of packed weight: `[N, K/8]` u32 (per `quantized.h:1094` qmm_t_impl indexing) — confirm.
   - dtype for scales + biases: `"BF16"` or `"F16"`?
   - Per-group `biases` shape: `[N, K/group_size]` — confirm.
   - Optional linear-layer bias: where stored (`<prefix>.bias`?), what dtype?
2. **Confirm embedding quantization.** Same checkpoint: is `model.embed_tokens.weight` quantized (dtype=U32 + scales + biases)? Or fp16/bf16? Same for `lm_head.weight`. If tied: only one is stored.
3. **Group sizes in the wild.** `mlx-community/Llama-3.2-1B-Instruct-4bit` advertises gs=64. Survey gs across mlx-community 4bit checkpoints for the model coverage matrix above; record per-model gs in an internal table to drive instantiation set in P3+. Probably some Qwen3 variants use gs=128.
4. **NAX detect.** Read `~/git/mlx/mlx/backend/metal/device.cpp:828-845` (`is_nax_available`). Reproduce the check: arch_gen ≥ Apple9 + tf32 toggle. Map to `objc2_metal::MTLGPUFamily::Apple9` (already reachable per `project_ferrite_metal_status.md`).
5. **Per-model verification.** For each model in the coverage matrix, confirm or refute the "Embedding" + "Linear shape" columns from the mlx-community checkpoint. Probably faster as a single python+mlx script that loads each repo and inspects keys/dtypes.
6. **`fp_quantized` deferral confirmation.** Confirm fp_quantized.metal is wired in production (`jit_kernels.cpp:861` shows it is). Document `nvfp4 / mxfp8 / mxfp4` as Phase 17. No further action in P0.

**Validation**

A single document (`INT4_PARITY_PROBES.md`) with verified facts, refutations, and per-model gs/bits/dtype/embedding tables. No code yet.

**Why first**: P1 onward bakes assumptions about safetensors layout and embedding quantization. Cheaper to verify than to rework.

---

### P1 — Plumbing: LinearLayer, WeightTensor, macro emit (M)

**Dependencies**: P0.

**Deliverables**

1. **`LinearLayer::AffineQuant(Box<AffineQuantLinear>)` arm** in `ferrite-kernels/src/layers.rs:738`.
   - `AffineQuantLinear` struct holds `weight: Buffer, scales: Buffer, affine_biases: Buffer, linear_bias: Option<Buffer>, in_features: u32, out_features: u32, group_size: u32, bits: u8`.
   - Buffer type is `ferrite_metal_kernels::weights::Buffer = Retained<ProtocolObject<dyn MTLBuffer>>`. CUDA-side does **not** see this type — it's gated to `cfg(feature = "metal")` arms in `LinearLayer`. (Sketch: keep `Self::AffineQuant` behind `#[cfg(feature = "metal")]` so the CUDA build doesn't try to construct it.)
   - Accessors: `pub fn affine_weight() -> Buffer`, `pub fn affine_scales() -> Buffer`, `pub fn affine_biases() -> Buffer`, `pub fn linear_bias() -> Option<Buffer>`, `pub fn in_features()` / `out_features()` / `group_size()` / `bits()`.
2. **`WeightTensor` extension** in `ferrite-forward/src/interpreter/metal/lowered.rs:214`. Add: `AffineScales`, `AffineBiases`, `AffineLinearBias`. The `Bias` arm stays as the dense-bias path; AffineLinearBias is for quantized linears that carry a fp bias.
3. **Worker resolver extension** in `ferrite-forward/src/interpreter/metal/worker.rs:1414`. Match new `WeightTensor` arms against `LinearLayer::AffineQuant`. Other `WeightTensor`-on-`AffineQuant` combos panic with `LoweringError::Mismatched`.
4. **Macro safetensors detection** in `ferrite-forward-macro/src/impl_lib.rs`. New per-linear arm: if the safetensors header advertises `<prefix>.weight` as `"U32"` + `<prefix>.scales` + `<prefix>.biases`, emit `LinearLayer::AffineQuant { ... }`; else emit `Dense`. Read `quantization_config` from `config.json` for default `(group_size, bits)`.
5. **Remove dead AWQ macro stub** at `ferrite-forward-macro/src/metal/awq.rs:114` (whole file, since it's `None` everywhere). Per `feedback_no_invented_files`, the AWQ shader at `ferrite-metal-kernels/shaders/awq_dequantize.metal` and the partial Rust wrapper at `ferrite-metal-kernels/src/awq.rs` should also be deleted in this phase — they're dead code that masquerades as a quantization plumbing site we don't use. Keep `ferrite-metal-kernels/tests/awq_test.rs` only if it tests dequantize math we're about to port; otherwise delete with the rest. Per `feedback_fix_warnings_properly`, removing dead code is the right shape; per `feedback_gate_feature_specific_additions`, leaving it as a feature-flagged dead path is not.

**Validation**

- `cargo check --all-targets -F metal` clean.
- `cargo test --lib -F metal` green (no functional change yet — `AffineQuant` arms are unreachable until P2 lowering).
- New unit test at `ferrite-metal-kernels/src/weights.rs` that loads a synthesized safetensors file with one `"U32"` tensor + one `"BF16"` scales tensor + one `"BF16"` biases tensor and confirms the `MetalWeights` map populates correctly.
- New unit test in `ferrite-kernels/src/layers.rs` that constructs `LinearLayer::AffineQuant` from raw buffers and exercises the accessors.

---

### P2 — Slow correctness reference: `affine_dequantize` + `AffineDequantizeThenGemm` (S)

**Dependencies**: P1.

**Deliverables**

1. **Port `affine_dequantize` kernel.** Source: `quantized.h:1961` (the `affine_dequantize` template). Target: `ferrite-metal-kernels/shaders/quantized_dequantize.metal` containing the kernel under all `(dtype × group_size × bits)` instantiations needed (gs ∈ {32,64,128}, bits=4 only for now, dtype ∈ {f16, bf16} — we drop bits ∈ {2,3,5,6,8} for the slow ref since they're not exercised by the int4 mandate; revisit when a model uses them).
2. **Rust dispatcher** at `ferrite-metal-kernels/src/quantized.rs` (new file). Mirrors `quantized.cpp:1657 fast::Quantize::eval_gpu`'s dequantize path: pick kernel symbol from `(dtype, group_size, bits)`, set bindings, dispatch `nthreads = out.size() / packs_per_int` threads with `packs_per_int = 8 / bits = 2`.
3. **`Instruction::AffineDequantizeThenGemm(in_slot, out_slot, layer, wtfn, n, k, gs, bits)`** in `instr.rs`.
4. **Lowering** in `interpreter/metal/lowering.rs`: emit two `LoweredCommand`s — first dequant from packed buffer + scales + biases into a scratch arena slot, then existing bf16 GEMM `[m, k] @ [n, k]^T` against the dequantized scratch. Use the existing `KernelId::Gemm` path for the matmul half.
5. **Pipeline cache registration** for the new `affine_quantized` metallib in `with_standard_shaders()` (`specialized_pipeline_cache.rs:147`). Add to `build.rs` to compile the new `.metal` shader.

**Validation**

- New cpu_golden ref `cpu_golden::affine_dequantize` in `ferrite-kernels-cpu-golden` that takes `(packed_u32, scales_bf16, biases_bf16, group_size, bits)` and produces dequantized bf16. Test in isolation: random q4 → dequant → match within 1ulp of golden math.
- E2E: load `mlx-community/Llama-3.2-1B-Instruct-4bit`, run model with `Instruction::AffineDequantizeThenGemm` substituted at every Linear site, generate one token. Compare against MLX `mlx_lm.generate` on the same prompt; greedy decode token-stream must match through at least the first 32 tokens. **This is the layout-correctness gate** — passes iff the safetensors layout, group/bits/dtype assumptions, and dequant math all agree with MLX.

**Why second**: a bug in the dequant math or layout is invisible behind a fast `qmv` kernel because both do dequant-then-multiply atomically. Validating the dequant in isolation, outside any matmul, is the only way to bisect cleanly.

---

### P3 — Decode GEMV (M)

**Dependencies**: P2 (for A/B against dequant-then-GEMM during bring-up), P8 if landed first.

**Deliverables**

1. **Port `qmv_quad` + `qmv_fast` + `qmv`** from `quantized.h:692` (`qmv_quad`), `:750` (`qmv_fast_impl`), `:817` (`qmv_impl`). Target: `ferrite-metal-kernels/shaders/quantized_qmv.metal` (single shader file, three kernels; mirrors MLX's grouping in `quantized.h`).
2. **Instantiation grid** matching MLX (`quantized.metal:140-156`): `dtype ∈ {f16, bf16}` (drop `float` — vLLM is fp16/bf16 only), `gs ∈ {32, 64, 128}`, `bits = 4` (P15 + future expand to 2/3/5/6/8 if a model uses them — track in P0 survey). For `qmv_quad`: `D ∈ {64, 128}` × `batched ∈ {0, 1}` per `quantized.metal:104-108`. For `qmv_fast`/`qmv`: `batched ∈ {0, 1}` per `:82-86`.
3. **`KernelId::AffineQmvQuad / AffineQmvFast / AffineQmv`**.
4. **`Instruction::AffineQmm(...)`** dispatcher in `lower_one`. Mirrors `dispatch_qmv` (`quantized.cpp:1365`) + `qmv` shape pick (`:259`):
   ```
   if M >= vector_limit: matmul branch (P4)
   else if transpose:
     if (K==64 || K==128) && is_pow2(bits): emit AffineQmvQuad
     else if N%8==0 && K%512==0:           emit AffineQmvFast
     else:                                  emit AffineQmv
   else:                                    P5 path
   ```
   `vector_limit = get_qmv_batch_limit(K, N, arch_gen, arch_size)` ported from `quantized.cpp:84`. The `arch_gen` / `arch_size` come from `MTLDevice` — extend `ferrite-metal-kernels::device` with a `architecture_gen()` helper if not present.
5. **`add_strides_and_shapes` analog** for batched dispatch (`quantized.cpp:128`). Sets x_batch_ndims, x.shape, x.strides, w.shape, w.strides, scales.strides, optional biases.strides as separate `setBytes` calls. ferrite-metal-kernels needs a small helper.
6. **Pipeline cache key extension** (P8 prerequisite or alongside): include `group_size, bits, batched, transpose` in `PipelineKey`. Without this, qmv vs qmv_fast vs qmv_quad share a key and the wrong one gets bound.

**Validation**

- Per-shape unit test in `ferrite-metal-kernels/tests/quantized_qmv_test.rs`: random q4 weights + bf16 activations; run `AffineQmm` (forced through each of qmv_quad/qmv_fast/qmv); compare vs `cpu_golden::affine_dequantize_then_matmul`. Three tests, one per kernel.
- E2E: same Llama-1B prompt as P2 with `AffineQmm` swapped in for decode steps only (prefill still on P2 path). Token stream must equal P2 result and MLX result.

---

### P4 — Prefill GEMM transpose=true (M)

**Dependencies**: P3 (for dispatcher routing) + P8 (key extension).

**Deliverables**

1. **Port `qmm_t_impl` + `qmm_t_splitk_impl`** from `quantized.h:1094` (`qmm_t`) and `:1651` (`qmm_t_splitk`). Target: `ferrite-metal-kernels/shaders/quantized_qmm.metal`.
2. **Instantiation**: `dtype ∈ {f16, bf16}`, `gs ∈ {32, 64, 128}`, `bits=4`, `aligned ∈ {true, false}`, `batched ∈ {0, 1}` per `quantized.metal:96-102`.
3. **`KernelId::AffineQmmT / AffineQmmTSplitK`**.
4. **Dispatcher rule** in `lower_one` for the matmul branch (mirrors `quantized.cpp:1411`):
   ```
   if M >= vector_limit:
     B = out.size() / M / N
     if transpose && B == 1: emit AffineQmmTSplitK
     else if transpose:       emit AffineQmmT
     else:                    P5 path
   ```
   For `qmm_t_splitk`, also emit a downstream `KernelId::Reduce` (sum-reduce across the split_k dim into final out) — match the pattern at `quantized.cpp:861` (`strided_reduce_general_dispatch`). Reuse existing reduce infrastructure if present, else add it. The `aligned` template constant comes from `N % 32 == 0`.

**Validation**

- Per-shape unit tests (qmm_t aligned + unaligned, splitk small-M).
- E2E: prefill 64-token prompt on Llama-1B-4bit through `AffineQmmT/SplitK` path; decode through P3; full output must match MLX.

---

### P5 — Transpose=false: `qmm_n` + `qvm` + `qvm_split_k` (M)

**Dependencies**: P4.

**Deliverables**

1. **Port `qmm_n_impl`** from `quantized.h:1221`. Target: extend `quantized_qmm.metal`.
2. **Port `qvm_impl` + `qvm_split_k_impl`** from `quantized.h:978` (`qvm`) and `:1419` (`qvm_split_k`). Target: extend `quantized_qmv.metal` or new `quantized_qvm.metal` (1:1 with MLX file structure — quantized.h has them all in one, so one shader file is fine).
3. **`KernelId::AffineQmmN / AffineQvm / AffineQvmSplitK`**.
4. **Instantiation**: gs ∈ {32, 64, 128}, bits=4, dtype ∈ {f16, bf16}. `qvm_split_k`: `split_k ∈ {8, 32}` per `quantized.metal:111-112`. `qmm_n`: `batched ∈ {0, 1}`.
5. **Dispatcher rule** in `lower_one`: complete the qmv / qvm branches per `quantized.cpp:1438-1453`.

**Validation**

- Per-shape tests for qmm_n (batched + non-batched), qvm (small K), qvm_split_k (K=2048, K=8192). Some models — or attention paths in some models — will hit transpose=false: e.g. DeepSeek MLA's K^T projection step. P0 verifies which models in the coverage matrix actually exercise transpose=false.
- E2E re-run on Llama-1B (which won't hit qmm_n/qvm at all — pure transpose=true). The unit tests carry the validation; an E2E for a model that actually exercises transpose=false comes in P10/P11.

---

### P6 — Quantized embedding lookup (M)

**Dependencies**: P2 (need affine_dequantize port).

MLX's `nn.QuantizedEmbedding.__call__` is implemented in Python — embedding lookup on a quantized weight is a *gather followed by per-row dequant*. The relevant code path under MLX:

- Forward: `x = w[indices, :]` then `x = mx.dequantize(x, scales, biases, group_size, bits)`. (Search MLX python `nn/layers/quantized.py`.) Hardware-wise: the gather happens on packed u32 + scales + biases in parallel, and dequant runs the standard `affine_dequantize` kernel on the gathered slice.
- Backward / lm_head: the lm_head matmul in a tied-embedding model reuses the same packed weight via `qmm_t` — handled by P3/P4.

**Deliverables**

1. **`Instruction::AffineEmbed(out_slot, wtfn)`** in `instr.rs`.
2. **`KernelId::AffineEmbed`** in `lowered.rs`.
3. **Lowering** that emits two `LoweredCommand`s: (a) a gather (`embed`-shaped) on the packed u32 weight + scales + biases into a packed-row scratch slot, (b) an `AffineDequantize` from packed-row scratch into the activation slot. Or — equivalently — a single fused `affine_embed.metal` kernel that gathers + dequants in one pass. Start with two-kernel decomposition (rule-clean per `feedback_no_handcoded_fusion`); revisit fused option if profiling shows it matters.
4. **`WeightBundleKind::QuantEmbedding`** new arm in `lowered.rs:202` that resolves to a quantized embedding (`Embedding` doesn't carry scales/biases today).
5. **Macro detection**: at the embed_tokens layer, detect dtype=`"U32"` + scales + biases the same way as Linear, emit `AffineEmbed` instead of `Embed`.

**Validation**

- Unit test against `cpu_golden::affine_embed` (gather + dequant in numpy).
- E2E gates: Llama-1B-4bit first token correctness now stands without P2 fallback. (Until P6 lands, P2/P3/P4 E2Es synthetically dequant the embedding once at load time as a workaround — mark this clearly in the test harness and remove in P6.)

**Tied lm_head** (P0 verifies which models tie): when `lm_head.weight === embed_tokens.weight`, the safetensors only stores the embedding (with `embed_tokens` keys). The macro must:
- Emit one quantized weight bundle for embedding.
- The lm_head Linear's `wtfn` resolves to the same bundle's `(weight, scales, biases)`.
- The lm_head emits as `AffineQmm` (transpose=true), reusing the same buffers.

This is a macro-side wiring problem (recognizing `tie_word_embeddings: true` in config and emitting a single resolver), not a kernel problem — kernels see two separate `AffineQmm` / `AffineEmbed` dispatches against shared buffers.

---

### P7 — NAX (M4+) integration (M)

**Dependencies**: P3 + P4 (so we have a non-NAX baseline to A/B against).

**Deliverables**

1. **`is_nax_available()` port.** New function in `ferrite-metal-kernels/src/device.rs`. Mirror `~/git/mlx/mlx/backend/metal/device.cpp:828`: check `device.supportsFamily(MTLGPUFamily::Apple9)` (already reachable per `project_ferrite_metal_status.md`) and arch_gen ≥ 13. tf32 toggle (`enable_tf32()` in MLX) — for Metal, fp32 isn't relevant in vLLM; gate on `dtype != f32` (always true).
2. **Port `qmm_t_nax` + `qmm_n_nax`** from `quantized_nax.h`. Target: `ferrite-metal-kernels/shaders/quantized_nax.metal`. Tile: bm=bn=bk=64, wm=wn=2 per `quantized.cpp:490-494`. NAX MMA path uses `steel/gemm/nax.h` from MLX; port the relevant header to `ferrite-metal-kernels/shaders/include/nax.metal` (1:1 file mapping; `feedback_no_invented_files`).
3. **`KernelId::AffineQmmTNax / AffineQmmNNax`**.
4. **Pipeline cache backend axis** (`PipelineKey` extension, dovetails with P8): add `backend: PipelineBackend ∈ {Standard, Nax}` to the key. When Standard and NAX both have a kernel, the lower picks based on `is_nax_available()`.
5. **Dispatcher gate** in `lower_one`: at the matmul branch (`M >= vector_limit && transpose`), check `is_nax_available() && K % 64 == 0 && dtype != f32`; if all hold, emit NAX kernel ID instead. This mirrors `qmm()` at `quantized.cpp:695`.

**Validation**

- Unit test on M4-only machine (M3 dev box can build but skip-execute via `if !is_nax_available() { return Ok(()) }`). Compare NAX qmm_t output to standard qmm_t output for the same input — must match within fp16 ulp.
- E2E: rerun Llama-1B-4bit with FERRITE_METAL_FORCE_NAX=1 / =0; both should produce identical token streams.

**Two-codegen-paths cache infra**: the existing `specialized_pipeline_cache` keys pipelines by `KernelId + dtype + constants`; the NAX axis is an additional constant. The infra supports it cleanly if `KernelId::AffineQmmTNax` is a separate variant (which we add), so no infra rework needed beyond the key extension. **Confirmed clean** — no infra blocker.

---

### P8 — Pipeline cache key extension (S, parallel with P3)

**Dependencies**: none structural; lands in parallel with first kernel phase.

**Deliverables**

Extend `PipelineKey` in `ferrite-metal-kernels/src/specialized_pipeline_cache.rs` to carry:
- `kernel: KernelId`
- `dtype: MetalDtype`
- `group_size: u8` (0 for non-quant kernels)
- `bits: u8` (0 for non-quant)
- `batched: bool`
- `aligned: bool` (for qmm_t / qmm_t_splitk; ignored for others)
- `transpose: bool` (defaults to true; used to distinguish qmm_t vs qmm_n)
- `backend: PipelineBackend ∈ {Standard, Nax}` (default Standard)
- `constants_hash: u64` (existing; covers function constants)

Cache lookups for non-quant kernels are unaffected (gs/bits=0 + Standard backend path is identical to today).

**Validation**: `cargo test --lib` green.

---

### P9 — cpu_golden q4 reference + per-model 4bit goldens (M)

**Dependencies**: P2 (cpu_golden::affine_dequantize) + P10 (E2E gates).

**Deliverables**

1. Extend `cpu_golden` (in `vllm-rs/crates/...` — find via grep — likely under `vllm-e2e` or `ferrite-cpu-golden`) with:
   - `affine_dequantize(packed: &[u32], scales: &[f16], biases: &[f16], group_size: u32, bits: u8) -> Vec<f16>`
   - `affine_qmv(x: &[f16], packed_w: &[u32], scales, biases, M, N, K, gs, bits) -> Vec<f16>` (slow ref = dequant + matmul)
   - `affine_qmm_t(x, w, ...)` (same with M >= vector_limit shape)
   - `affine_qmm_n(...)`, `affine_qvm(...)`, `affine_gather_qmm(...)` per kernel coverage.
2. **Per-model golden** in `vllm-rs/crates/vllm-e2e/testdata/golden/`:
   - `llama_3_2_1b_mlx_4bit.json` (P10 P-model, gs=64, bits=4)
   - `llama_3_2_3b_mlx_4bit.json`
   - `qwen2_7b_mlx_4bit.json`
   - `qwen3_dense_*_mlx_4bit.json`
   - `qwen3_moe_30b_a3b_mlx_4bit.json`
   - `mistral_7b_v03_mlx_4bit.json`
   - `mixtral_8x7b_mlx_4bit.json`
   - `gemma2_*_mlx_4bit.json` × 2
   - `gemma3_*_mlx_4bit.json` × 3 (including MM)
   - `deepseek_v3_mlx_4bit.json`
   Each captures `(prompt, n_tokens, expected_token_ids)` from `mlx_lm.generate` greedy decode at temp=0.

**Validation**: every golden matches when its model runs through the relevant kernel path.

---

### P10 — Llama-3.2-1B/3B 4bit E2E (S)

**Dependencies**: P3 + P4 + P6 + P9.

**Deliverables**

1. Add Llama-3.2-1B-Instruct-4bit + Llama-3.2-3B-Instruct-4bit to the model registry (`vllm-rs/crates/vllm-config/src/models.rs`).
2. Run `vllm chat --device metal --model mlx-community/Llama-3.2-1B-Instruct-4bit` and verify coherent output on the canonical smoke prompts (per `project_ferrite_metal_status.md`'s "What is the capital of France?" / "Why is the sky blue?" / "The grass is green and the sky is blue."*500 long-prefill).
3. Token-stream A/B vs MLX `mlx_lm.generate` — must match through ≥ 64 tokens at temp=0.

**Validation**: covered by goldens + manual smoke. Honor `feedback_one_vllm_chat_at_a_time` (one process at a time) and `feedback_machine_24gi_limit` (3B model fits, 7B does not on the dev machine — Qwen2-7B et al. tested via CI / borrowed hardware).

---

### P11 — Mixed-quant model loader (S, parallel)

**Dependencies**: P1 (loader hooks).

**Deliverables**

1. Macro must handle a model where some Linear layers are `Dense` (bf16) and others `AffineQuant` — automatic per-layer detection from safetensors keys (already in P1).
2. RMSNorm gains, layer biases on dense layers, and unquantized embedding (when not tied or when config opts out) must all load as `Dense`/`Embedding`/`RmsNorm`.
3. Loader test: synthesize a "mixed" safetensors (one quant Linear + one dense Linear + one RMSNorm) and confirm the macro emits the right `LinearLayer::*` per layer.

**Validation**: integration test on `mlx-community/gemma-2-2b-it-4bit` if available — Gemma typically leaves embedding tied + bf16 even when the rest is q4. Confirms mixed-quant model wiring.

---

### P12 — Quantized MLP composition decision (M-L)

**Dependencies**: P3 + P4.

**The decision**: `feedback_no_handcoded_fusion` says fusion comes from DAG atom composition, not from special-case fused emitters. Two readings:

- **Strict**: any kernel that does q4 matmul + SwiGLU + multiply in one Metal kernel is "handcoded fusion." Forbidden.
- **Permissive**: dequant-inside-matmul is intrinsic to a quantized GEMM (every MLX qmm kernel does this). It's not "fusion" — it's the kernel's defining behavior. Cross-op fusion (e.g. fold `silu(gate @ w_gate^T) * (up @ w_up^T)` into a single shader) is what the rule forbids.

This plan adopts the **permissive reading** for *intrinsic* dequant-inside-matmul (P3/P4/P5/P6 stand) and the **strict reading** for cross-op fusion (P12 chooses below).

**Branches**

- **(i) Composed `AffineQmm + silu_mul` (rule-clean, recommended).** Emit q-MLP as three `Instruction`s: `AffineQmm(gate)`, `AffineQmm(up)`, `Activation::SiluMul(gate, up)`. Two device round-trips per MLP layer (write dequant'd activations to scratch, read for SwiGLU). Slower than MLX's fused MLP but rule-clean and simple. ~10-20% perf gap on MLP, depending on shape.
- **(ii) Hand-rolled fused `affine_qmm_t_silu_mul.metal` (rule-violating).** Mirror MLX's pattern: a single shader that does `dequant(W_gate) @ x`, `dequant(W_up) @ x`, `silu(g) * u` in one threadgroup with no device round-trip. Faster (matches MLX), but introduces the kind of hand-rolled fused emitter the rule forbids.

**Recommendation**: ship branch (i) first — it's correct, rule-clean, and the perf gap is tractable. Branch (ii) is unblocked once `pipelines.rs` Phase 2 lands (`project_metal_pipelines_rs_must_die.md`) and the DAG / atom architecture is in place — at that point a `q4_silu_mul_atom` composition can fuse via DAG edge-removal exactly as `feedback_no_handcoded_fusion` requires. Phase 12.b becomes "compose q-MLP via DAG atoms once pipelines.rs Phase 2 lands."

**Deliverables (branch (i))**

1. Lowering pass folds `Linear(gate_proj) + Linear(up_proj) + SiluMul` into:
   - `AffineQmm(... → gate_scratch)`
   - `AffineQmm(... → up_scratch)`
   - `Activation::SiluMul(gate_scratch, up_scratch, → out)` (re-use existing path)
2. Validate against MLX golden — output equality first, perf delta documented.

**Phase 12.b (deferred)**: After pipelines.rs Phase 2, introduce `q4_silu_mul_atom` composed via DAG. Fused output produces a single Metal kernel through edge-removal.

---

### P13 — MoE: gather_qmm/qmv/qvm + sorted-rhs path (L)

**Dependencies**: P3 + P4 + P5 (so non-MoE q4 is solid before we add MoE).

**Deliverables**

1. **Port** all 8 `affine_gather_*` kernels from `quantized.h:2085-2536`:
   - `affine_gather_qmm_t / _n` (matmul) → `shaders/quantized_gather_qmm.metal`
   - `affine_gather_qmv_fast / _ ` (matvec transpose=true) → `shaders/quantized_gather_qmv.metal`
   - `affine_gather_qvm` (transpose=false) → same
   - `affine_gather_qmm_rhs_nt / _nn` (sorted-decode hot path) → `shaders/quantized_gather_qmm_rhs.metal`
2. **`Instruction::AffineGatherQmm(in, out, layer, wtfn, lhs_idx_slot, rhs_idx_slot, m, n, k, gs, bits, transpose, sorted)`**.
3. **Dispatcher rule** mirrors `GatherQMM::eval_gpu` (`quantized.cpp:1456`): sorted-rhs path → `gather_qmm_rhs_nax/std`; matmul branch → `gather_qmm`; matvec branch → `gather_qmv` (transpose=true) or `gather_qvm` (transpose=false).
4. **`KernelId::AffineGather*`** (8 IDs).
5. **`add_gather_strides_and_shapes`** analog ports the index strides/shape parameters per `quantized.cpp:158`.

**Validation**: Qwen3-MoE-30B-A3B (P0 verifies it ships in q4) end-to-end on borrowed hardware (machine spec is 24Gi-capped; 30B q4 weights ≈ 17GB so fits). Token stream matches MLX. Per `project_qwen3_moe_graph_capture.md`, the MoE graph capture has known issues — verify those are CUDA-specific and don't apply on Metal.

---

### P14 — NAX MoE variants (M)

**Dependencies**: P7 + P13.

**Deliverables**

1. Port `affine_gather_qmm_t_nax / _n_nax` (`quantized.cpp:576`).
2. Port `affine_gather_qmm_rhs_nax_nt / _nn` (`:1084`).
3. NAX has no gather_qmv / gather_qvm (verified: `quantized_nax.metal` only instantiates qmm + gather_qmm + gather_qmm_rhs). Decode MoE on M4+ uses standard `gather_qmv`. Document this asymmetry.
4. Dispatcher gate inside `gather_qmm` mirrors `quantized.cpp:886` (`is_nax_available && transpose && K%64==0`).

**Validation**: Qwen3-MoE-4bit on M4 hardware. Token stream matches.

---

### P15 — Long-prompt + long-decode under q4 (S)

**Dependencies**: P10 (1B/3B baseline) + interaction with `project_metal_long_decode_panic`.

**Deliverables**

1. Run `Llama-3.2-3B-Instruct-4bit` on the long-prompt smoke prompt (`"The grass is green and the sky is blue."*500`).
2. Run long-decode test (max_tokens 4096, then 8192, then 16k). Hits the same `per-step commit failed` panic at ~38k+ that the bf16 path hits per `project_metal_long_decode_panic.md`. Q4 weights free up enough memory pressure that we may exercise this earlier than bf16 did — long-context perf testing requires the panic fix.
3. Drop diagnostic `eprintln` (`a892b2fc0`) only after fix.

**Validation**: long-decode at 16k+ tokens runs to completion without panic on 3B-4bit.

---

### P16 — A/B vs MLX + perf parity gate (L)

**Dependencies**: P10 (Llama 1B/3B) + P13 (MoE) + P14 (NAX-MoE) for full parity.

**Deliverables**

1. **Benchmark harness** in `vllm-rs/crates/vllm-bench/`. Existing `bench` crate; add q4-specific runs:
   - Decode tokens/sec at batch 1, prompt 32, output 256 — for each model.
   - Prefill tokens/sec at prompt 1024, 4096, 8192 — for each model.
   - MoE-decode tokens/sec at expert sparsity 8/64 (Qwen3-MoE) — when applicable.
2. **MLX baseline harness**: invoke `mlx_lm.generate --temp 0 --max-tokens N` from a test driver, parse output rate, capture as a JSON line. Per `feedback_use_uv_venv`, run from `~/.venv` with `mlx_lm` installed.
3. **Parity definitions** (M3 + M4 separately):
   - Decode tokens/sec ≥ 95% of MLX on Llama-1B-4bit
   - Decode tokens/sec ≥ 95% of MLX on Llama-3B-4bit
   - Prefill tokens/sec ≥ 90% of MLX on Llama-1B-4bit prompt=1024
   - MoE decode tokens/sec ≥ 90% of MLX on Qwen3-MoE-4bit
   - NAX-active runs: Decode ≥ 95% of MLX on M4 Llama-3B-4bit
   The 5–10% asymmetry budget covers harness overhead + the few cross-op fusions in MLX we don't yet replicate (P12 branch (ii) deferred). Tighter parity expected once P12.b lands.
4. **Per-shape kernel benchmarks** (`vllm-bench/benches/quantized_kernels.rs`, new): isolate qmv_fast / qmv_quad / qmm_t / qmm_t_splitk / gather_qmm vs MLX's same kernel via the MLX C++ benchmark harness. Pure kernel A/B, decoupled from end-to-end overhead.

**Validation**: parity numbers checked into a perf board (markdown table or JSON dashboard); regressions block merges.

---

### P17 — FP-quant mode (mxfp4 / mxfp8 / nvfp4) (XL — separate parity track)

**Dependencies**: P3 + P4 + P5 + P13 (full int4 surface, since FP-quant kernels have the same dispatcher shape).

**Deliverables**

1. Port `fp_quantized.h` + `fp_quantized.metal` (1905 + 155 lines) — the FP minifloat dequant + the same matrix kernel shapes as `affine`. MLX instantiates only over fp16/bf16/f32 × {nvfp4 gs=16 b=4, mxfp8 gs=32 b=8, mxfp4 gs=32 b=4} (`fp_quantized.metal:147-150`).
2. Port `fp_quantized_nax.h/metal` (1020 + 79 lines) for M4+ NAX FP variants.
3. **`LinearLayer::FpQuant`** new arm; mode discriminator threaded through `Instruction::FpQmm(...)` separate from `AffineQmm`.
4. Reuse the dispatcher heuristics from int4 unchanged — same `dispatch_qmv` / `qmm` / split-K thresholds.
5. **Models exercised**: DeepSeek-V3 ships FP variants (MXFP4 in some configs); some MoE configs in nvfp4. Survey + add to model coverage matrix in P0.b (P0 follow-up after FP scope lands).

**Validation**: per-mode goldens for each FP-quant model. Same A/B vs MLX gates as P16, on FP-quant models.

**Why last**: int4 is the user-stated mandate; FP-quant is a parallel track with different math. Folding it earlier risks dispatching the wrong kernel mode and serves no Llama/Qwen/Gemma int4 user.

---

## Phase 0 next-actions (directly actionable next session)

1. **Probe `mlx-community/Llama-3.2-1B-Instruct-4bit` safetensors header.** Single command:
   ```
   uv run --with mlx --with safetensors python -c "
   from huggingface_hub import snapshot_download
   from safetensors import safe_open
   p = snapshot_download('mlx-community/Llama-3.2-1B-Instruct-4bit')
   import os, json
   for f in sorted(os.listdir(p)):
     if f.endswith('.safetensors'):
       print(f)
       with safe_open(os.path.join(p, f), framework='numpy') as st:
         for k in sorted(st.keys()):
           t = st.get_slice(k); print(' ', k, t.get_shape(), t.get_dtype())
   print(json.load(open(os.path.join(p, 'config.json'))).get('quantization'))
   "
   ```
   Expected output: dtype strings, shapes for `model.embed_tokens.{weight,scales,biases}` and one decoder layer's `q_proj.{weight,scales,biases,bias?}`. Captures every P0 ambiguity (dtype string for U32, embedding quant status, group_size, biases shape).
2. **Create `INT4_PARITY_PROBES.md`** at the worktree root with per-question answers from #1 + a per-model table (Llama-{1B,3B}, Qwen2-7B, Qwen3-{1.7,4,8}B, Qwen3-MoE-30B-A3B, Mistral-7B, Mixtral-8x7B, Gemma-{2,3}-{2,9}B, Gemma3-MM, DeepSeek-V3) of `(quantization scheme, group_size, bits, embedding_quantized?, lm_head_tied?)`.
3. **Confirm `is_nax_available()` arch_gen check** at `~/git/mlx/mlx/backend/metal/device.cpp:828-845`. Plan the `architecture_gen()` accessor in `ferrite-metal-kernels/src/device.rs`. Read MTLDevice.h via objc2-metal docs to confirm `MTLGPUFamily::Apple9` cleanly maps to gen-13.
4. **Locate `cpu_golden` infrastructure.** `grep -rn 'cpu_golden::affine_dequantize\|fn affine_dequant\|mod cpu_golden' vllm-rs/crates/`. Confirm extension path is one crate (likely `vllm-e2e/src/cpu_golden/`).
5. **Confirm `quantization_config` shape** by reading the actual MLX 4bit config.json (output of #1). Should be `{"group_size": 64, "bits": 4}` at config root. If MLX ships a per-layer quantization scheme (`{"...": null}` for unquantized layers), document it in INT4_PARITY_PROBES.md so P11 macro detection handles it.

---

## Risks + design call summary

| Risk | Phase | Severity | Mitigation |
|---|---|---|---|
| MLX safetensors `"U32"` vs `"I32"` ambiguity | P0/P1 | low | Verify in P0; extend dtype-match list. |
| `mlx-community/...-4bit` quantizes embedding | P0/P6 | medium | Verify in P0; P6 handles both cases (embedding either Dense or AffineQuant). |
| Tied lm_head + quantized embedding sharing | P6 | medium | Macro emits one buffer-bundle, two consumers. |
| `feedback_no_handcoded_fusion` collision in P12 | P12 | high — design call | Branch (i) (composed) recommended; defer (ii) until pipelines.rs Phase 2. |
| `dispatch_qmv` heuristic drift from MLX | P3 | medium | Port C++ heuristic byte-for-byte; instrument to log decision path; A/B vs MLX during bring-up. |
| `qmv_quad`'s "is_pow2(bits)" check excludes bits=3,5,6 — these route to `qmv` | P3 | low | Document; default models use bits=4 anyway. |
| Sentinel-row hazard from bf16 fix (`project_metal_bf16_fixed`) | P3/P4 | medium | Verify q4 prefill kernels honor `u32::MAX` sentinel pad-row exclusion. Add long-prompt smoke test. |
| Pipeline cache thrash with 1000s of `(KernelId, gs, bits, batched, ...)` keys | P8 | low | Cache size unchanged in shape; just more hash variants. |
| NAX MMA path uses `steel/gemm/nax.h` — porting that header may pull in non-quant deps | P7 | medium | Header is self-contained; verify in P7. |
| Long-decode panic (`project_metal_long_decode_panic`) blocks q4 long-context perf | P15 | medium | P15 handles. |
| 24Gi machine can't run 7B+ q4 + dev tools + MLX baseline simultaneously | P16 | low — operational | Per `feedback_one_vllm_chat_at_a_time`, run sequentially; borrow CI hardware for >7B. |
| MoE graph capture issues from CUDA (`project_qwen3_moe_graph_capture`) leak into Metal | P13 | low | CUDA-specific (cublasLtAlgoHeuristic etc.) — confirm in P13. |
| FP-quant mode introduces parallel `Instruction::FpQmm` that drifts from `AffineQmm` | P17 | low | Shared dispatcher; mode discriminator is a string at the kernel name, not a separate code path. |

**Single open design call for the user**: Phase 12 branch (i) vs (ii). Recommendation is (i). Reversible later via 12.b.

---

## Out-of-scope (deliberately, with justification)

| Out | Reason |
|---|---|
| `bits ∈ {2, 3, 5, 6, 8}` instantiation in initial P3-P5 | No model in coverage matrix uses them. Add when a model surfaces. |
| `float32` dtype instantiation | vLLM is fp16/bf16 — the f32 instantiation in MLX is for gradient training paths we don't run. |
| `QQMatmul` kernel (`quantized.cpp:1611`) | MLX-internal "quantized × quantized" with a single decode-only path; no model in the wild does on-line activation quantization. Add only if a future model lands with live `quantize_dequantize` in its forward. |
| `affine_quantize` (forward direction) | We ingest pre-quantized weights from MLX checkpoints. Never quantize on-device. |
| GGUF / GPTQ / AWQ ports on Metal | `feedback_ferrite_metal_mlx_only` — Metal port is MLX-only. CUDA stack handles GGUF/GPTQ/AWQ. |

Nothing else is omitted. Every MLX quant kernel that's in the production dispatcher (`QuantizedMatmul::eval_gpu`, `GatherQMM::eval_gpu`, `QQMatmul::eval_gpu`, `fast::Quantize::eval_gpu`) maps to a phase or to the "out-of-scope with justification" table above.
