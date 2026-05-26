# PD Wavefront — Persistent Decode Megakernel Plan

Resume-from-cold reference for the subtile-wavefront decode megakernel.
Companion to memory `project_pd_clean_wavefront_design.md` (`[[pd-clean-wavefront-design]]`).
Last updated 2026-05-25.

## Objective & ceiling

A persistent multi-threadgroup decode megakernel for Apple Silicon (M4+), structured as a
**subtile-wavefront tape-player schedule**. It recovers the ~1.5 ms of inter-dispatch overhead
in the existing per-op split-K decode path (8.2 ms baseline on M4 base, Llama-3.2-1B-Instruct-4bit)
and frees the CPU from ~80 dispatches/token.

**Honest ceiling:** decode at M=1 is bandwidth-bound. Floor ~6.7 ms vs 8.2 ms ⇒ **~15–18% on M4
base**; **~30–50% projected on M4 Max** (UNMEASURED). Real latency wins need M>1
(spec-decode/batch → compute-bound).

## Worktree / setup

- Branch `worktree-pd-wavefront` off `worktree-ferrite-metal` @ `24997d429`.
- Path `/Users/moosevan/git/vllm/.claude/worktrees/pd-wavefront`; Rust workspace in `vllm-rs/`.
- `pd-clean` worktree is **ABANDONED** (left untouched, rebase conflicts unresolved). Do NOT
  resurrect its `SynthForward`/`SynthLayer`/`_mt` code.
- Carried over only: `vllm-rs/crates/ferrite-metal-cost-sweep/src/flag_sync_sweep.rs` (+2 lines in
  that crate's `main.rs`) — the cross-TG sync go/no-go microbench. Re-run per chip:
  `FERRITE_SWEEP=flag_sync ./target/release/metal_cost_sweep`.

## Validated facts (measured, M4 base)

- Decode is **bandwidth-bound**; per-op split-K already saturates ~85% of BW (8.2 ms / 695 MB ≈
  85 GB/s). A megakernel can't beat it on BW — the only headroom is the ~1.5 ms dispatch gap to the
  ~6.7 ms BW floor.
- **Single-TG = 1 core = 1/10 BW = 55 ms = 7× SLOWER.** Single-TG is falsified.
- **Cross-TG point-to-point flag sync = 0.18 µs/hop, FLAT at 2/4/10 TGs** (O(1) in core count).
  Bare global barrier = 0.41 µs and grows with TG count. ⇒ p2p flags are the mechanism; **global
  barriers are the trap** (they, plus imbalance idle, killed the old `_mt` kernel — NOT the barrier
  primitive itself; bin-packing + p2p waits fix both).

## Locked design decisions

1. **Oracle = ferrite-metal non-mega** (same repo/tokenizer/weights/sampler; IS the 8.2 ms
   baseline). `cpu_golden` is the host **calculator** (deterministic f32 substrate), NOT the oracle.
2. **Lower from the SOLVED FUF** (`Fuf` + solver `Assignment`): cost is Impl-keyed
   (`TargetProfile::cost_us` via `Impl::cost_us` — there is NO per-raw-op cost); the `Assignment`
   gives fusion groups (`SubgraphId`); split-K is an Impl artifact.
3. **New crate `ferrite-wavefront`** (normal lib, NOT the proc-macro crate) holds IR + lowering +
   scheduler + host player + ALL tests. Why: proc-macro crates can't export types, AND the macro
   crate's tests are cuda-coupled (8 files, 42 sites in `impl_lib`) so it won't `cargo test` on Mac.
   Mirrors the `ferrite-fusion-synth` pattern; the macro crate calls into it. Deps `ferrite-forward`
   (for `cpu_golden`). Mac-testable, ~0.2 s incremental build.
4. **KV-cache append modeled by FUSION** (user's call): rope_append + attention are one fused unit;
   the new token's K/V is an **internal dataflow `Sub` edge** (rope→attention), NEVER a cache
   round-trip. The cache appears only as a read-only **prefix `Source` segment** + a write-only
   **commit side-output** (no intra-step reader). No mutable cache buffer in the IR ⇒ no
   shared-mutable-state hazard (the trap that killed `_mt`).
5. **Two-tier (really three) validation:**
   - **Tier A** (self-consistency): tape replay == direct topo eval (`eval_dag`), same arithmetic.
     Exact, no horizon. Validates the schedule + `Wait`/`Signal` edges.
   - **Tier A′** (decomposition equivalence): `eval_dag` == `cpu_golden` whole-op. Bit-exact at
     `k_chunks = 1`; split-K within f32 tol (reassociation expected).
   - **Tier B** (faithfulness): temp=0 token stream == ferrite-metal non-mega. EXACT f16-vs-f16 on
     GPU (mega vs non-mega); on host (f32) only a first-N-token sanity check (host f32 vs GPU f16
     diverges past a horizon).
6. `schedule.rs` (existing, macro crate) is the **BSP/global-barrier wave binner** = exactly what we
   supersede. The wavefront scheduler is a NEW sibling. No II/ResMII/modulo exists anywhere (net-new).
7. `SubOp` is non-`Eq` (carries f32 fields `eps`/`scale`).
8. The scheduler takes an **injected** cost fn `Fn(&SubtileNode) -> f64` (µs) — macro supplies real
   `cost_us`, tests supply synthetic — so `ferrite-wavefront` stays decoupled from `TargetProfile`.

Lesson learned this session: seek the fuse/synthesis option before presenting an A/B binary or an
AskUserQuestion (`[[feedback-seek-fusion-over-binary]]`).

## Architecture / pipeline

```
LoweringInput → lower() → SubtileGraph (DAG)
                              ├── eval_dag()  ............... Tier A′ reference
                              └── schedule_wavefront()/partition → Schedule (per-worker tapes,
                                       Wait/Signal) → play() ... Tier A
```

`ferrite-wavefront/src/`:

- **`subtile.rs`** — the IR + host eval.
  - `SubtileGraph { nodes: Vec<SubtileNode>, sources: Vec<SourceShape>, result_rows, result_cols,
    outputs: Vec<OutputSlot> }`. Topo-ordered, dense `SubtileId(u32)`.
  - `SubtileNode { id, op: SubOp, inputs: Vec<Operand>, out_rows, out_cols }`.
  - `SubOp`: `MatmulTile`, `SumReduce`, `Elementwise(EwKind{Silu,Mul,Add})`, `RmsNorm{eps}`,
    `RopeRotate{head_dim}`, `AttnDecode{num_q_heads,num_kv_heads,head_dim,scale}`.
  - `Operand`: `Source{id:SourceId, region:Region}` (slice of an external/leaf buffer) | `Sub(SubtileId)`
    (whole producer output — no slicing yet; see Deferred).
  - `Region{rows,cols}`, `Range{start,len}`, `SourceShape{rows,cols}`, `OutputSlot{node, dest:Region}`,
    `TilingPolicy{nb, k_chunks}`.
  - `eval_dag(graph, sources:&[&[f32]]) -> Vec<Vec<f32>>`; `eval_node` (shared with the player);
    `assemble_result`.
  - Standalone lowerings (dev/test): `lower_gemm_standalone`, `lower_gate_up_silu_mul_standalone`,
    `lower_residual_add_standalone`, `lower_rmsnorm_standalone`, `lower_decode_attention_standalone`.
- **`tape.rs`** — scheduled/replayable form + host player.
  - `TapeInstr`: `Compute(SubtileId) | Signal(u32) | Wait(u32)`. `Worker{tape}`. `Schedule{workers, num_flags}`.
  - `schedule_from_assignment(graph, worker_of:&[u32], num_workers) -> Schedule`. **Deadlock-free for
    ANY assignment** (ascending-id emission per worker + id-ordered DAG ⇒ a producer's `Signal`
    precedes any consumer's `Wait`). One one-shot flag per producer with a cross-worker consumer.
  - `partition_roundrobin` (test fixture).
  - `play(graph, schedule, sources) -> Vec<Vec<f32>>`. Simulates P workers honoring `Wait`/`Signal`;
    asserts no deadlock, every node computed once, inputs-ready.
- **`lower.rs`** — `LoweringInput` → `SubtileGraph` (coarse: one subtile/op).
  - `InputRef::Op(usize) | Ext(usize)`. `LoweredOp`: `Gemm{n,k}`, `RmsNorm{eps}`, `Silu`, `Mul`,
    `Add`, `RopeRotate{head_dim}`, `AttnDecode{...}`. `OpDesc{op, m, inputs}`.
    `LoweringInput{sources, ops, result}`. `lower(&LoweringInput) -> SubtileGraph`.
- **`scheduler.rs`** — the wavefront scheduler.
  - `ScheduleParams{num_workers, wait_cost_us}`. `ScheduleMetrics{max_stack_us (P1), total_us, edge_cut (P2)}`.
  - `schedule_wavefront(graph, cost:Fn(&SubtileNode)->f64, params) -> Schedule` — greedy list
    scheduler over topo order; each node → worker `argmin(new_load + wait_cost·preds_elsewhere)`;
    feeds `schedule_from_assignment`. `measure(graph, schedule, cost) -> ScheduleMetrics`.

## Current state — DONE & GREEN (23 tests, `cargo test -p ferrite-wavefront`, ~1 s, 0 warnings)

> 2026-05-25 session added 2 tests + the T2b bridge: `full_forward_bit_exact` (embed-as-source →
> 2 layers → final norm → lm_head, the structure the bridge targets) and `validate` + its test.
> The macro→wavefront bridge `to_wavefront.rs` is built, build-clean on Mac (`-Fmetal`), and runs
> on the real Llama-3.2-1B FUF (see T2b DONE below). Next: T5 real-weight executor (see roadmap).

- **T1** (IR: DAG + Tape) ✓.
- **T3** (host player + Tier A/A′) ✓.
- **Complete decode op vocabulary**, all Tier A′ bit-exact vs `cpu_golden` + Tier A bit-exact under
  the player: gemm/split-K, Elementwise/SwiGLU, rmsnorm, residual add, fused rope+attention (GQA).
- **T2 ENGINE** (`lower()`) ✓ — proven on: rmsnorm→gemm chain; full SwiGLU MLP block; **full
  Llama-style decode layer** (bit-exact vs `cpu_golden` AND replayed via the tape player AND via the
  real wavefront scheduler, P∈{2,4,8}).
- **T4.1** (cost-aware greedy wavefront scheduler) ✓ — strictly beats round-robin max-stack on
  imbalance (40→22); `wait_cost` pulls dependent chains onto one worker (0 cut); correctness preserved.
- ⇒ **The design's stated "first deliverable, no GPU" (scheduler + host tape player + bit-exact
  validation) is COMPLETE** — on synthetic/hand-built `LoweringInput`s, not yet a real model's
  solved FUF.
- **T2b DONE (2026-05-25, structural).** `ferrite-forward-macro::to_wavefront` translates a solved
  decode FUF + `Assignment` → `ferrite_wavefront::lower::LoweringInput` (+ a `SourceBinding`
  manifest) and `lower()`s it. Verified on the REAL Llama-3.2-1B FUF (all 3 variants) at
  macro-expansion time (the macro RUNS on Mac during `-Fmetal` builds; only its `#[test]`s are
  cuda-coupled): **243 FUF tiles → 258 valid subtile nodes**, every count reconciles exactly
  (embed→source; each `rope_append`→2 `RopeRotate`+v-alias; 113 Gemm = 7/layer·16 + lm_head;
  RmsNorm 33; AttnDecode/Silu/Mul 16; RopeRotate/Add 32), `validate()` passes, and `lower()`'s
  per-Gemm `act_cols==k` assert holding on all 113 matmuls proves the dataflow is shape-consistent
  end-to-end. 181 sources = 146 weights + 32 prefix-kv + 3 singletons (embed/cos/sin). The
  wavefront-side target structure is locked Mac-testable by `full_forward_bit_exact` (23 green).

## Remaining roadmap

- **PIVOT (2026-05-26): "host tape player" = host ORCHESTRATION + GPU kernels** (CUDA-interpreter
  sense), NOT a host-f32 simulator. The host-f32 real-weight executor that an earlier draft of this
  T5 described was a misread and is ABANDONED (`ferrite-wavefront-exec` deleted). See design memory
  `[[pd-clean-wavefront-design]]` UPDATE 8/9 + `[[feedback-wavefront-host-player-is-gpu-kernels]]`.
  The pure-host-f32 `eval_dag`/`play()`/unit tests stay only as IR+schedule unit tests.
- **T5-region — tensor-region IR DONE + VERIFIED (Mac, 27 tests).** `crates/ferrite-wavefront/src/
  region.rs`: every op output is a logical buffer (tensor); a subtile WRITES a region + READS regions;
  deps = region overlap. `lower_region(LoweringInput, nb)` N-block-tiles every GEMM (whole norm/rope/
  attn). Bit-exact vs cpu_golden incl. a full Llama decode layer at nb∈{4,8,1000}. Built alongside the
  old `subtile.rs` (Operand::Sub) IR, which stays intact (T2b macro bridge still uses it).
- **T5-gpu — GPU subtile tape player (NEXT; the big metal build, ≈T6).** Host orchestrator (real GPU
  f16 kernels) walks the region graph in TOPO ORDER and dispatches: each matmul N-block via the `qmv`
  atom with OFFSET bindings + `OUT_VEC_SIZE=nb` pipeline (linchpin confirmed: `qmv_fast_impl` indexes
  relative to base pointers, `quantized_qmv.metal:620-630`, no kernel changes); whole RMSNorm/RoPE/
  SiLU·Mul/attention via existing atoms; into a Metal command buffer with barriers on `predecessors`
  edges; wire as an ALT decode path; greedy-decode; compare temp=0 tokens BIT-EXACT vs ferrite-metal
  non-mega (both GPU f16 → exact, not first-N). Per-subtile dispatch ⇒ SLOWER than 8.2 ms (correctness
  scaffold; perf = the on-device megakernel after). Multi-worker wavefront scheduling is NOT needed on
  host (host dispatch can't overlap). Build/run `--bin vllm -Fmetal FERRITE_MODELS=llama-3.2-1b`; ONE
  vllm chat at a time; 24 GiB cap. Then migrate scheduler.rs/tape.rs to region-overlap deps + retire
  old subtile.rs/lower.rs once the region path is proven end-to-end.
- **T4.2 — modulo scheduling (software pipelining).** Schedule ONE layer body, loop it NL× (like
  `apply_loop_compression` for the per-op path); II=ResMII (layer bytes / aggregate BW) overlaps the
  next layer's weight loads with this layer's serial drain; RecMII (residual recurrence) is tiny at
  M=1. Needs repeating-layer detection (reuse the `detect_repeating_run` fingerprint technique).
  Mainly a GPU-makespan optimization — host correctness already holds.
- **T6 — GPU tape players.** ONLY after host green. Emit P MSL sub-tapes composing VALIDATED
  atoms/primitives (`mk_rope_pair`, `mk_tg_rmsnorm_scale`, `mk_qmv_fast`, split-K qmm, attention
  partials) — NEVER hand-write kernel math. Stamp p2p `Wait`/`Signal` with correct data-before-flag
  ordering (THE one correctness-critical generated thing). Persistent launch sized to occupancy, M4+.
  Measure ≥5 runs vs 8.2 ms (min/median/p99); distributions must not overlap.

## Deferred decisions

- **Fine inter-op tiling.** Split-K on a producer-fed activation (e.g. gemm2 slicing gemm1's hidden,
  or rmsnorm reading a producer emitted as N-blocks) needs either `Operand::Sub{region}` slicing or a
  **Tensor-region IR model** (every op output = a logical tensor; edges derived from region overlap).
  Coarse `lower()` (1 subtile/op, consumers read whole producer outputs via `Sub`) is CORRECT
  meanwhile and needs no IR change. Decide at T4.2/GPU where fine tiling is actually motivated.

## Don't-do list

- NO single-TG (7× slower, falsified). NO global-barrier multi-TG (the `_mt` trap). NO hand-rolled
  fused MSL (compose atoms). NO mutable KV-cache buffer in the IR (fusion makes the new token a
  dataflow edge). Don't resurrect pd-clean code. Don't add Metal-specific consts to `CanonicalParams`.
  Don't `cargo test` the macro crate on Mac (cuda-coupled — won't compile).

## Build / test

- `cargo test --manifest-path /Users/moosevan/git/vllm/.claude/worktrees/pd-wavefront/vllm-rs/Cargo.toml -p ferrite-wavefront`
  (fast; Mac; no backend feature needed — `cpu_golden` is backend-agnostic).
- The `ferrite-forward-macro` test suite needs `--features cuda` (a CUDA box); it does NOT build on
  Mac. Keep wavefront logic in `ferrite-wavefront` so it stays Mac-testable. BUT the macro crate's
  LIB compiles + RUNS on Mac during `-Fmetal` builds — that's how T2b is verified without its tests.
- **Exercise the T2b bridge on the real FUF:**
  `FERRITE_WAVEFRONT=1 FERRITE_MODELS=llama-3.2-1b cargo build -p ferrite-model-llama -F metal`
  — the `#[forward]` macro runs the bridge at expansion time and prints a `[wavefront] …` dump
  line per model variant (tile/subgraph/source/op counts, `valid (N nodes)`, op histogram). Errors
  log `not lowered — <reason>` and never gate the build (env-gated + fully fallible).
