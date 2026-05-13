# ferrite-metal — current state and remaining work

Single-file pickup doc for any machine. Last updated 2026-05-13 at HEAD `dbab34b10` with uncommitted NAX-infrastructure changes (see "Then do this next"). If you're a fresh agent, read this and the cited source files.

**Pickup quickstart:** there are uncommitted changes in the worktree wiring NAX end-to-end. Run `git status` and `git diff --stat` first to see them; nothing is half-wired (default `vllm chat` is coherent), but the NAX kernel itself produces wrong output and is gated behind `FERRITE_ENABLE_NAX=1`. The next action is fixing that one bug — see "Then do this next" below.

---

## Orient first — run these before reading further

```bash
git -C $WORKTREE log --oneline -10
git -C $WORKTREE status
cd $WORKTREE/vllm-rs
FERRITE_MODELS=llama-3.2-3b-mlx-affine-b4-g64 \
  cargo build --release -Fmetal --bin vllm

# M1 Max: MTL3 crashes; MTL4 is the only working path. Use this:
FERRITE_METAL_MTL4=1 ./target/release/vllm chat --device metal \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --bench --max-tokens 200 -p "Write a 200-word poem about the ocean."

# M4 (MTL3 still works as default; MTL4 also works under the env flag):
./target/release/vllm chat --device metal \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --bench --max-tokens 200 -p "Write a 200-word poem about the ocean."

./target/release/vllm ferrite info llama 3.2 3b mlx \
  | grep -E "SynthPreAttn|SynthMlpPreDown|FusedAddRmsNorm"
```

If chat produces a coherent poem and bench reports ~46 tok/s on M4 (or whatever your chip's number is — capture it), the synth-solver migration is healthy and you're picking up from a green baseline. If anything panics or produces gibberish, that's the work — drop everything else.

## Then do this next

The NAX track is **closed on M4** as of 2026-05-13 — MLX itself doesn't use NAX on M4 (`mlx/backend/metal/device.cpp:828` gates on `arch_gen >= 17` and M4 is gen 16). MPP `matmul2d` on M4 emulates via the standard simdgroup matmul with a cooperative-tensor per-thread layout that doesn't match MLX `BaseNAXFrag`. Infra stays in the tree dormant behind `is_nax_capable(_) == false` for all currently modelled gens — see `memory/project_metal_nax_layout_bug.md`. So: **pick a different track from §"Remaining work" below**. Track 2 (MLP synth redesign) and track 3 (MTL4 finish) are the next two perf levers on M4 and M1 Max respectively.

---

## What works today

`vllm chat --device metal --model <model>` and `vllm bench latency --device metal --model <model>` run end-to-end on both **M4** and **M1 Max**, with coherent output. Models confirmed running:

- mlx-community/Llama-3.2-1B-Instruct-4bit
- mlx-community/Llama-3.2-3B-Instruct-4bit
- TinyLlama-1.1B
- Llama / Mistral / Qwen3 / Granite text decoders compile + run cleanly under `--features metal`

The full pipeline is in place: weight loading → solver-driven kernel selection (including synth-fusion) → batched ICB on a Serial encoder → paged KV cache → greedy sampling (`argmax_f16`) → ModelRunnerOutput. No `Backend` trait; cfg-mutex everywhere. `vllm-serve` instantiates `FerriteWorker` under metal for supported arches, falls through to `MlxWorker` for the rest (Mixtral, Qwen-MoE, Qwen3-MoE, DeepSeek-V2/V3, CommandR, multimodal).

### Bench (Apple M4 base, Llama-3.2-3B-Instruct-4bit, 200-tok decode, 5-run mean)

| build | tok/s |
|---|---|
| ferrite-metal `5d04524e8` | **46.18** (σ≈0.95) |
| reference: pre-ferrite vllm-mlx (feat/rust) | 48.32 (σ≈0.34) |
| reference: `mlx_lm.generate` | 52.5 |

Ferrite is at ≈ 97 % of pre-ferrite vllm-mlx. The ~3 % gap on M4 and the ~16 % gap to `mlx_lm.generate` are mostly NAX MMA (M4+ only, not yet wired).

---

## Build / run

```bash
cd vllm-rs
FERRITE_MODELS=llama-3.2-3b-mlx-affine-b4-g64 \
  cargo build --release -Fmetal --bin vllm

./target/release/vllm chat --device metal \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --bench --max-tokens 200 -p "Write a 200-word poem about the ocean."

./target/release/vllm ferrite info llama 3.2 3b mlx \
  | grep -E "SynthPreAttn|SynthMlpPreDown|FusedAddRmsNorm"
```

Reference A/B (CUDA-era worker, uses vllm-mlx on `--device metal`):
`~/git/vllm/vllm-rs/target/release/vllm` on the `feat/rust` branch.

### Cost sweep regen (per chip)

```bash
# Full regen — overwrites the chip's CSV
cargo run -p ferrite-metal-cost-sweep --release --bin metal_cost_sweep \
  > crates/ferrite-metal-targets/profiles/cost_<chip>.csv

# Single family — appends; only safe if non-synth rows already exist
FERRITE_SWEEP=synth_pre_attn,synth_mlp_pre_down \
  cargo run -p ferrite-metal-cost-sweep --release --bin metal_cost_sweep \
  | grep '^synth_' >> crates/ferrite-metal-targets/profiles/cost_<chip>.csv
```

Filterable families: `rmsnorm`, `affine_qmv`, `affine_qmm`, `synth_pre_attn`, `synth_mlp_pre_down`.

---

## Architecture at a glance

- **Macro is the compiler.** `ferrite-forward-macro` walks the FUF, runs the solver DP, picks one `Implementation` per claim, and emits a per-canonical `forward` body. Structural analyses (hazards, scheduling, liveness, barriers) all run at macro expansion. Never at runtime.
- **Solver picks Impls by `cost_us`.** Each `Implementation` reports a cost; the DP picks the global min. Costs come from `cost_<chip>.csv` (per-kernel measured); when a row is missing the Impl falls back to an analytical roofline.
- **Synth fusion is solver-driven.** `MetalSynthPreAttnImpl` and `MetalSynthMlpPreDownImpl` claim the pre-attn and post-attn chains and emit single synth-kernel dispatches when they win on cost. No post-pass fixup; the old `apply_synth_replacement{,_init,_mlp}` machinery is fully retired.
- **Runtime is just an executor.** `MetalWorkerPool` holds per-canonical pipelines + the residency set; `FerriteWorker(metal)::execute_model` plumbs `ForwardCtx` into the macro-emitted `forward`. Batched ICB on a single Serial compute encoder per forward (`717664b58`); per-step fallback available under `FERRITE_METAL_PER_STEP_CMDBUF=1`.

### Key source locations

| what | path |
|---|---|
| Solver Impls (metal) | `vllm-rs/crates/ferrite-forward-macro/src/metal/*.rs` |
| Synth Impls | `metal/synth_pre_attn.rs`, `metal/synth_mlp_pre_down.rs` |
| Synth source emission | `vllm-rs/crates/ferrite-fusion-synth/src/fuse_pass.rs` |
| AOT compile (xcrun metal/metallib) | `ferrite-fusion-synth::aot::aot_compile_metallib` |
| IR + lowering | `vllm-rs/crates/ferrite-forward/src/interpreter/metal/lowering.rs` |
| Worker pool + ICB | `vllm-rs/crates/ferrite-forward/src/interpreter/metal/pool.rs` |
| `FerriteWorker(metal)` | `vllm-rs/crates/vllm-executor/src/ferrite_worker.rs` (`:9415` load_model, `:9591` initialize_cache, `:9745` execute_model) |
| Cost CSVs | `vllm-rs/crates/ferrite-metal-targets/profiles/cost_<chip>.csv` |
| Cost sweep | `vllm-rs/crates/ferrite-metal-cost-sweep/src/` |
| Hand-written MSL kernels | `vllm-rs/crates/ferrite-metal-kernels/shaders/*.metal` |

### CSV status

- `cost_m4.csv` — full sweep including `synth_pre_attn_*` (30 rows) and `synth_mlp_pre_down_*` (15 rows).
- `cost_m1_max.csv` — synth sweep landed in `1e03bb73d`. If the file is missing other families on M1 Max, re-run the full sweep on that box.

---

## Remaining work

Tracks 1–3 are the high-leverage perf moves. 4–9 are correctness / coverage / cleanup. Pick by chip + interest:

- **On M4:** start with track 2 (MLP synth redesign). The ~3 % gap to vllm-mlx and ~16 % gap to `mlx_lm.generate` on M4 are NOT NAX-attributable (MLX itself doesn't use NAX on M4 either — see track 1) — they live in dispatch / kernel-selection / synth-shape tuning.
- **On M1 Max / M1–M3:** track 3. MTL3 crashes on M1 Max so you must run with `FERRITE_METAL_MTL4=1`; first move is to flip the default and stop relying on the env var, then land Phase B + remaining C.
- **Correctness or stuck on perf?** Tracks 4 (long-decode panic), 5 (safetensors zero-copy), or 8 (Phi-3 LongRoPE).

### 1. NAX / MPP `matmul2d` — DORMANT, M5+/A19+ only

Status: infrastructure landed (shader, kernel cache, dispatcher, lowering, instantiations, reproducer test, layout-dump probe) but gated **off** for every Apple Silicon generation we currently model. `is_nax_capable(_)` returns `false` unconditionally — re-enable per-generation only after running the probe on that chip and confirming MPP returns the `BaseNAXFrag` layout.

Why off: MLX's `is_nax_available()` in `mlx/backend/metal/device.cpp:828` requires `arch_gen >= 17` for the `'g'` arch class. M4 reports `applegpu_g16g` → arch_gen = 16, so **MLX never uses NAX on M4 either**. MPP `matmul2d` is callable on M4 but emulates via the standard simdgroup matmul, with a cooperative-tensor per-thread layout that doesn't match `BaseNAXFrag`'s 2-row × 4-col contiguous assumption — see the `nax_probe_dump_layout` test dump in `crates/ferrite-metal-kernels/tests/quantized_qmm_test.rs` and the bug memory `project_metal_nax_layout_bug.md`. The earlier framing of NAX as "the M4 perf lever" was wrong from the start.

When M5+/A19+ work begins:
1. Add a new variant to `AppleSiliconGen` (e.g., `M5`).
2. Run the probe on the new chip:
   ```bash
   cargo test -p ferrite-metal-kernels --test quantized_qmm_test \
     nax_probe_dump_layout -- --ignored --nocapture
   ```
   Confirm lane 0 covers rows ∈ {0, 8} × cols ∈ {0, 1, 2, 3, 8, 9, 10, 11} for ct_c — the `BaseNAXFrag` layout. Other lanes follow `get_coord`.
3. If it matches, flip `is_nax_capable` true for that gen, remove `#[ignore]` from `affine_qmm_t_nax_b4_bf16_matches_cpu_reference`, and re-sweep `cost_<chip>.csv`.
4. If it doesn't match, do NOT enable. Either MPP emulates on that chip too, or the layout has shifted. Rewrite `BaseNAXFrag::mma` in `shaders/metal_nax.h` to bridge via `get_multidimensional_index` rather than the per-index assumption.

What's wired (dormant, ready for M5+):
- `shaders/quantized_qmm_nax.metal` — BK=64 port of MLX's `qmm_t_nax_tgp_impl`, 12 instantiations across (f16/bf16) × (gs64/gs128) × (alN=true/false). Inlined `QuantizedBlockLoader` general-loader path.
- `shaders/metal_nax.h` — vendored `BaseNAXFrag` / `NAXTile` / `tile_matmad_nax` + MPP internals.
- `shaders/nax_probe.metal` — layout-dump diagnostic.
- `pick_qmm_t_kernel(..., is_nax)` — gates on `K % 64 == 0 && gs != 32 && is_nax`.
- `ShaderCache`/`SpecializedPipelineCache` route `affine_qmm_t_nax_*` → `quantized_qmm_nax` metallib.
- `MetalAffineQmmT::execute_with_kernel(.., QmmTKernel::Nax, ..)` — test hook to force NAX (used by the reproducer test).

### 2. MLP synth kernel redesign

`MetalSynthMlpPreDownImpl` is correct (decode coherent E2E) but the qmv-shape dispatch redoes per-TG full-HIDDEN norm reads. At intermediate=8192/14336 that's 64-112 TGs per token doing redundant work. The M4 sweep measured 671 µs at M=1 (3B) vs the unfused chain's ~310 µs, so the solver currently picks unfused for the MLP side on M4. Same outcome as the prior session's `FERRITE_NO_MLP_SYNTH=1` direct probe.

Fix: matmul-tile dispatch + shared-mem reduction so the norm runs once per token, not once per (token, intermediate-tile). Re-running the sweep auto-picks the new variant if it scores lower than unfused. Pairs naturally with track 1 (NAX MMA).

### 3. MTL4 migration — finish it; MTL4 is the only working path on M1 Max

**MTL3 crashes on M1 Max. MTL4 works.** The env-var gate (`FERRITE_METAL_MTL4=1`) is not a "side-by-side experiment" — it's the only way to get a non-crashing forward on M1 Max today. Treat MTL4 as the production target and rip MTL3 out once the remaining phases land.

Wired and running today (`interpreter/metal/mtl4.rs`, pool at `pool.rs:271,608+`; pick-up via `device.newMTL4CommandQueue()` at warmup):
- A.1 probe (`ad7a4e5e1`)
- A.2 + A.3 bake-time + runtime path (`4d790077f`)
- Partial Phase C compile-time DAG barrier analysis (`51c4ab812`, `59ac9b7ff`)

Open:
- **Flip the default.** Drop the `FERRITE_METAL_MTL4` env-var gate, make MTL4 the default whenever `device.newMTL4CommandQueue()` returns Some (macOS 15+ on Apple Family 7+). MTL3 stays only as a fallback for older OS. Also capture an A/B bench on M4 (m=1..2 decode) for the record, but the M1 Max evidence already makes the call — MTL3 isn't an option there.
- **B — MTL4Compiler + serialized pipeline cache.** Today MTL4 path rebuilds compute pipelines per warmup; B caches them.
- **C (rest) — explicit barriers, kill conservative serialization.** Compile-time barrier analysis already lands per `59ac9b7ff`; rest of C wires the runtime to emit only the necessary barriers instead of full Serial.
- **D — stitched compute pipelines.** Research phase; follow-up.

Why MTL4 wins anyway (independent of M1 crash):
- `MTL4ArgumentTable` collapses ~18 `setBuffer` calls/dispatch into one bind for the synth kernels.
- MTL4 has compute sequencing as a first-class concept — replaces the "executeCommandsInBuffer-on-Serial-encoder" hack we use because compute ICBs only ship `ConcurrentDispatch`.
- OS-gated (macOS 15+), not hardware-gated.

### 4. Long-decode panic

`per-step commit failed` panic on Llama-3.2-3B at decode beyond ~38k tokens. Bisected ≤35k clean; threshold is the KV cache cap. Diagnostic eprintln still in a droppable commit; root cause not yet identified.

### 5. Safetensors load: zero memcpy

`Llama-3.2-3B-4bit` load is alignment-bound — 0 of 649 tensors hit the zero-copy path, every tensor is at `offset mod 16 == 2`, total 1.76 GiB memcpy on load. Loader can't `newBufferWithBytesNoCopy` because Apple wants page-aligned starts; need a strategy (re-emit aligned safetensors, or align-and-mmap with a pad buffer).

### 6. `arch_gen` hardcoded M4 in `affine_qmm_vector_limit`

Prior status flagged this; still hardcoded. Should consult the target profile rather than baking M4.

### 7. `pipelines.rs` Phase 2

`FERRITE_METAL_*` doc on the macro-time lowering / per-kernel modules refactor. Phase 1 collapsed match tables (`4efc5c17f`); Phase 2 (macro-time lowering, per-kernel modules) open.

### 8. Phi-3 LongRoPE under metal

`ferrite-model-phi3` has metal commented out — needs `build_longrope` + `new_partial_longrope_from_stream` ported to `_from_gpuweights` counterparts.

### 9. MoE / MLA / vision under metal

Mixtral, Qwen-MoE, Qwen3-MoE (`OpKind::Moe`), DeepSeek-V2/V3 (`MlaSplit`), CommandR (`Mean`), and any multimodal arch — missing metal Impls. These don't load under `--features metal` today; users see the `MlxWorker` fallback. Not blocking the core text path; tracked here so it doesn't get re-discovered.

---

## Hard rules (carried forward; don't relearn)

- **No `Backend` trait. No `<B>` parameter. No per-backend forks** of `OwnedTensor` / `KvCachePool` / `Worker`. Cfg branches go inside one type, not across two types. Cuda-only field → `#[cfg(feature = "cuda")] foo: Foo` on the shared struct.
- **Kernel work lives in `ferrite-metal-kernels`** as its own commit. Never entangled with worker / scheduler / unification surface.
- **No hand-coded fusion impls.** Fusion comes from atoms composed at the synthesis layer.
- **No reaching for Philip Turner's MFA.** Old / private Metal API; broken on Metal 4.
- **`simdgroup_event` is NOT public.** No async tile-copy mechanism on current Metal. Don't design around it.
- **Slot order: `Add(delta, residual)` — `inputs[0]=delta`, `inputs[1]=residual`.** Both synth Impls' `fan_out` MUST emit `(residual_slot, delta_slot, ...)` in that order. Reverse swaps the in-place residual write and produces silent decode gibberish (commit `6ffd780d4` ate this trap).
- **`OpKind::Silu` + `OpKind::Mul` are separate FUF tiles.** They get fused into one `SiluMul` OpInstance by a downstream pass. Solver Impls match on the FUF (separate); retired post-passes operated on lowered OpInstances (already fused).
- **Synth kernel symbol naming must match `fuse_pass::synthesize_*_chunk`'s emitted `symbol`** exactly. Mismatch → silent nil from `newFunctionWithName` at runtime.
- **`ChunkConstants::num_q_heads` is overloaded as INTERMEDIATE** inside `synthesize_mlp_pre_down_chunk`. Pass intermediate there when sweeping the MLP variant.
- **Don't `git commit -m` via bash heredoc** — backticks/braces get expanded. Use `git commit --no-gpg-sign -F /tmp/msg.txt`. Always `--no-gpg-sign` (Claude's terminal can't prompt for GPG).
- **Don't `--no-verify` to skip hooks. Don't `git reset --hard`** without explicit user permission. Don't rewrite history on `feat/rust` or `main`.
- **One `vllm chat` / model-loading process at a time** — concurrent multi-GB weight loads OOM the dev machine (24 GiB Apple cap).
- **MLX models only under metal.** No invented files, no scaffolding, no shortcuts. Port the MLX reference faithfully.
- **Don't touch `worktree-metal-ferrite`** — user said it is bullshit.

---

## How synth fusion picks today

Two solver Impls. Each `cost_us` does CSV lookup first; on miss it returns a component-sum analytical estimate × 0.95 (fused bias).

| Impl | Claims | M4 pick at decode (M=1) | M4 pick at prefill (M≥2) |
|---|---|---|---|
| `MetalSynthPreAttnImpl` (init=false) | `(Add, RmsNorm, 3×Gemm, RopeAppend)` | **Fused** | Unfused |
| `MetalSynthPreAttnImpl` (init=true) | `(RmsNorm, 3×Gemm, RopeAppend)` (layer-0 prelude) | **Fused** | Unfused |
| `MetalSynthMlpPreDownImpl` | `(Add, RmsNorm, 2×Gemm, Silu, Mul)` | Unfused (kernel correct, qmv shape is a wash; see track 2) | Unfused |

`vllm ferrite info <arch>` prints the picked tree per bucket — use it to verify whatever you're about to change.
