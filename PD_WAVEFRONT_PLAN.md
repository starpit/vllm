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

## LATEST STATUS — 2026-05-26 (COURSE CORRECTION: host scaffold abandoned → on-GPU 10-tape megakernel)

**The host-orchestrated single-tape scaffold (UPDATEs 12–14) was a DRIFT off the plan and is abandoned.**
The user re-affirmed the plan in plain terms: **(1) 10 tapes (M4 = 10 cores), (2) tape players that run
IN the GPU (one persistent kernel, 10 co-resident threadgroups, each running its own tape), (3) producer→
consumer SPINLOOP barriers on the tapes (fine-grained p2p flags, NOT global barriers, NOT command-encoder
barriers).** The scaffold (`subtile_compile`→flat `SubtileIr`→host `MetalExecutor` per-subtile dispatch with
NO-OP Wait/Signal, A/B `wavefront_ab_compare`) is none of those: single flat tape, no bins, no flags emitted
(`num_flags=0`), Metal-3 serial encoder. Its `B != A` is the wrong thing to chase. STOP debugging it.

**What the scaffold got right and is kept:** the typed `SubtileIr` + `validate` (compile-time shape/dataflow
safety) and the N-block qmv linchpin (proven bit-exact on-device). The bin-packer (`scheduler.rs`/`tape.rs`)
and p2p-flag stamping were proven on host but bound to the COARSE `subtile.rs` graph (whole-producer
`Operand::Sub`) — which can't N-block a matmul across workers (the bandwidth point). The drift was: the GPU
path bypassed the SSA dataflow IR + bin-packer entirely and went `Instruction`→colored-arena flat tape.

**Landed this session (corrective, host-proven, `cargo test -p ferrite-wavefront` = 46 green):**
`crates/ferrite-wavefront/src/region_schedule.rs` bin-packs the **SSA `region.rs` graph** (N-block-accurate,
`predecessors()` = region-overlap RAW edges) into **P=10 balanced worker tapes** with cross-tape `Wait`/
`Signal` flags. `region_schedule_replays_bit_exact` (P∈{1,2,4,10}, nb∈{4,8,coarse}) replays bit-exact vs
`eval_dag` ⇒ bin-packing + flag stamping deadlock-free + order-correct (Tier A). `wide_matmul_spreads_across_
workers`: 30 N-blocks over 10 workers, even, 0 global barriers. `flag_invariants`. **This is the megakernel's
SCHEDULE — the exact input the on-GPU players consume.**

**Atom composability for the GPU player (CHECKED):** qmv is already factored into `METAL_FUNC` device fns
(`qmv_fast_impl` etc., `quantized_qmv.metal:593`; `out_row = tid.y*8 + simd_gid*4`, indexes off base ptrs) —
composable NOW. rmsnorm / silu·mul / attention / rope are `[[kernel]]` entry points whose bodies need a
mechanical "extract → `METAL_FUNC` device fn + thin wrapper" (extraction, NOT rewrite — preserves validated math).

**BACKEND-NEUTRALITY RULE — LOCKED (user, 2026-05-26): the megakernel's tape/table ENCODING is neutral data in
`ferrite-wavefront`.** The on-GPU interpreter's input — the per-TG instruction stream (opcode, shape-class index,
operand-table indices, flag ids), the shape-class table ((op,K,N,gs,bits) descriptors), and the operand table —
are plain serializable types in `ferrite-wavefront`, NOT baked into Metal/objc2 code. The MSL interpreter and a
future CUDA `.cu` interpreter are PARALLEL CONSUMERS of the identical encoding. Per-target ONLY: the kernel
source, the atom device fns (`qmv_fast_impl` ↔ CUDA qmv), operand-index→pointer resolution (Metal `gpuAddress` ↔
CUDA `CUdeviceptr`), and the flag PRIMITIVE (Metal device-atomic spinloop ↔ CUDA cooperative-groups grid sync —
SAME `Signal`/`Wait` semantics in the tape; do NOT force one primitive on both). NB the host `Executor` seam in
`subtile_ir.rs` is for the per-dispatch scaffold; the *megakernel's* neutral seam is this DATA encoding, not a
host trait. (Milestone A1/A2 are raw proofs dispatched directly; the encoding lands when the interpreter is built.)

**GPU PLAYER SHAPE — LOCKED (user, 2026-05-26): data-driven interpreter, NOT unrolled codegen.** One persistent
kernel; each co-resident TG runs an interpret loop over its per-TG tape (instructions = data). Rationale: a
persistent megakernel is ONE compiled binary (function constants are baked per-launch, uniform), so the only
real perf edge unrolled has is per-op constant-folding of `K`/`N` — and that's the CHEAP kind (the inner
`load_vector`/`qdot` work is templated on `bits`/`group_size`, uniform across decode ⇒ folded either way; only
the short outer K-reduction loop bound, 4–16 iters on a BW-bound matvec, loses static unroll = <1%). Interpreter
wins on: debuggability, small reused icache footprint, it IS the exact GPU analogue of the proven host player
(`region_schedule::play` — free bit-exact oracle), trivial-player law, and no MSL-emitter. **BUILT IN FROM THE
START (user): a `switch(shape_class)` whose arms call the atom with LITERAL `K`/`N`** (decode has ~8 distinct
qmv shapes: q/k/v/o/gate/up/down/lm_head) ⇒ per-shape folding recovered, one kernel, ~8 atom copies (not
one-per-op), schedule still pure data. The compiler enumerates shape-classes + emits the tape; the kernel's only
generated part is the ~8-arm switch table.

**NEXT = on-GPU megakernel, built incrementally (this is where `_mt` died — compose atoms, never hand-write):**
- **Milestone A1 — DONE + GPU-VERIFIED.** `wavefront_qmv_mega` (appended to `quantized_qmv.metal` so it calls
  `qmv_fast_impl`; instantiated `wavefront_qmv_mega_<act>_s_<scale>_gs_<gs>_b_4`): a persistent kernel, P
  co-resident TGs, each looping `g = tgpos.x; g < ceil(N/8); g += grid_tg.x` and composing `qmv_fast_impl`
  (synthetic `tid.y = g`, 64 threads = 2 simdgroups). Test `tests/wavefront_mega_gpu.rs` =
  **bit-exact vs whole `affine_qmv_fast` for P∈{1,2,4,10}** (`cargo test -p ferrite-forward -F metal --test
  wavefront_mega_gpu`). Proves persistent multi-TG tape-loop + on-device atom composition + co-residency. No
  flags/switch yet (disjoint outputs). Uncommitted.
- **Milestone A2 — BUILT + RUNS; root-caused; `#[ignore]` pending the fix.** `wavefront_qmv_mega_2stage`
  (folded shape-class `switch` calling `qmv_fast_impl` at LITERAL K/N per stage; producer TGs compute y1 blocks,
  `threadgroup_barrier(mem_device)`, `atomic_store` flag; consumer TGs spin-`Wait`, read whole y1, compute y2)
  is bit-exact at P=1 but RACES at P>=2. `tests/wavefront_mega_gpu.rs` (ignored). 

- **CRITICAL FINDING — cross-TG data handoff must be ATOMIC (microbench `tests/wavefront_sync_probe.rs`).**
  On Apple GPU with relaxed-only MSL atomics, NON-atomic device writes are NOT reliably visible across
  threadgroups even with `threadgroup_barrier(mem_device)` (it's intra-TG per Apple docs). ATOMIC device
  writes/reads ARE device-coherent. Probe (200 retries each, P∈{1,2,4,8,10}): PAT 0/1/2 (non-atomic data) RACE;
  **PAT 3 (atomic data write+read + flag + barrier) and PAT 4 (data-IS-the-flag sentinel spin) both CORRECT.**
  `flag_sync_sweep` only ever proved TOKEN handoff (data == the atomic); this is the missing data-behind-flag
  result. **Megakernel consequence:** the cross-TG activation handoff (the N-block join — a consumer reading
  producers' outputs on other workers) must use atomic accesses. Activations are bf16/f16, Metal atomics are
  32-bit ⇒ u32-pack pairs (or an f32 staging slot) on cross-WORKER edges only. Intra-worker edges stay
  non-atomic (program order + barrier; the scheduler's edge-cut objective already pulls dependent chains onto
  one worker, minimizing cross-worker edges).

- **A2 — DONE + GPU-VERIFIED (atomic u32-packed handoff).** `wavefront_qmv_mega_2stage` now: stage-0 producers
  write their disjoint y1 stripe (bf16), `threadgroup_barrier(mem_device)`, then PACK each group (8 bf16 → 4 u32)
  and `atomic_store` to a coherent `y1c` handoff + signal; consumers join on all flags, `atomic_load` + unpack
  `y1c` into a PRIVATE per-worker bf16 copy (`y1r[me*N0..]`), then stage-1 qmv reads that. The only cross-TG
  buffer is the atomic `y1c`; everything else is intra-worker. **Bit-exact vs two sequential whole-qmv dispatches,
  50 retries × P∈{1,2,4,10}** (`tests/wavefront_mega_gpu.rs`, no longer ignored). Proves cross-TG spinloop sync +
  folded shape-class switch + the atomic handoff together. ⇒ **the full on-GPU megakernel execution model is
  proven** (persistent multi-TG tape-loop + atom composition + co-residency + p2p spinloop + atomic cross-TG
  handoff), alongside the host-proven 10-tape schedule.

- **ThunderMittens SEEDED (sync primitives done + proven).** `shaders/mittens/sync.h` (`#pragma once`,
  `namespace mittens`): the cross-TG sync PRIMITIVES — `wf_signal`/`wf_wait`/`wf_wait_all` (p2p flags) +
  `wf_pack2`/`wf_unpack_lo|hi`/`wf_publish_pairs`/`wf_acquire_pairs` (atomic u32-packed bulk handoff) + `WF_SPIN_CAP`.
  `quantized_qmv.metal` `#include "mittens/sync.h"` (relative-resolved by `xcrun metal`; the `.h` is a header, not a
  metallib target); A2 refactored to compose them and STILL bit-exact (50× × P∈{1,2,4,10}). The library is the
  per-target primitive home (IR/encoding stays neutral in `ferrite-wavefront`; ≈ ThunderKittens for CUDA).
- **NEXT — extract the COMPUTE atoms → ThunderMittens, then scale up.** `qmv_fast_impl` is already a `METAL_FUNC`
  (move to `mittens/qmv.h`); then extract rmsnorm/rope/silu·mul/attention bodies from their `[[kernel]]` entries
  into `mittens/` `METAL_FUNC` headers + thin kernel wrappers (each: extract → compose in the megakernel →
  bit-exact). Then full decode layer → 16 layers + lm_head, driven by `region_schedule`; wire as the alt decode
  path; bit-exact vs non-mega; then persistent-launch occupancy sizing + ≥5-run perf vs 8.2 ms.
- Then: extract rmsnorm→silu·mul→rope→attention device fns one at a time (each: extract + compose + bit-exact
  test); scale to full layer → 16 layers + lm_head; drive from `region_schedule` (10 tapes + flags); persistent
  launch sized to occupancy (M4+); measure ≥5 runs vs 8.2 ms. Sync primitive reference = `flag_sync_sweep.rs`.

Landed in one commit off `0b00278a5` (fmt + clippy clean; region_schedule + A1/A2 megakernels + ThunderMittens
`mittens/sync.h` + the sync probe + this plan). Not yet wired into the live decode path. The OLD status below is
superseded.

## LATEST STATUS — 2026-05-26 (ThunderMittens COMPUTE-ATOM EXTRACTION COMPLETE — all 5 atoms)

**All five decode compute atoms are extracted into ThunderMittens (`shaders/mittens/`), each composed in a
persistent multi-TG megakernel and bit-exact-verified. UNCOMMITTED (user gates commits).** This finishes the
"extract the compute atoms into ThunderMittens" half of the prior NEXT; the full-layer megakernel is next.

Atoms (all `namespace mittens`, VERBATIM bodies — extraction not rewrite; dims/eps/scale + threadgroup scratch
ride as params so each is self-contained, exactly like `qmv_*_impl` take K/N by value):
- `mittens/qmv.h` — `qmv_fast_impl` / `qmv_impl` / `qmv_quad_impl` + `load_vector(_safe)` / `qdot(_safe)` / pack
  helpers + `SIMD_SIZE`/`QUAD_SIZE`, moved out of `quantized_qmv.metal` (its sole includer; other quantized
  shaders keep their own copies and compile independently). All `affine_qmv*` + gather + A1/A2 megakernels call
  `mittens::qmv_*`.
- `mittens/rmsnorm.h` — `rmsnorm_impl` (the `rmsnorm_*_specialized` body; `shared_sum` + M/HIDDEN/EPS as params).
- `mittens/rope.h` — `rope_rotate_pair` (the NeoX pair-rotation shared by Q + K in `rope_append_*_specialized`;
  ROTATION ONLY — the paged-cache write stays in the wrapper per design #4).
- `mittens/silu_mul.h` — `silu_mul_impl` (the `silu_mul` body; element count as param).
- `mittens/attention.h` — `attention_decode_impl` (the `attention_via_cache_v2_*_specialized` body: paged-cache
  `sdpa_vector`, online softmax, 32-simdgroup K-split + combine; tg scratch + dims/scale + (seq,q_head) as params;
  one template covers f16+bf16 — they were byte-identical bar the type).

Each production `[[kernel]]` is now a THIN WRAPPER (declares its threadgroup scratch, forwards function constants,
calls the atom) — production decode path signatures/bindings/constants/dispatch unchanged.

Per-atom composition proof (in each atom's shader): `wavefront_{qmv,rmsnorm,rope,silu_mul,attention}_mega` — P
co-resident TGs loop the work-items they own (g = tgpos, += grid_tg) composing the atom. A1-style: disjoint
outputs, NO cross-TG flags (qmv also keeps A2's cross-TG atomic-handoff proof). Bit-exactness holds because each
loop iteration uses the atom's natural per-item TG layout identical to the reference (rmsnorm/attention reduce
over the same thread count; rope/silu·mul are per-element). Megas that reuse threadgroup scratch across items add
a trailing `threadgroup_barrier` between iterations. New per-mega function constants for the flattened bound:
`ROPE_NUM_TOKENS` (fc5), `ATTN_BATCH` (fc6).

Verification (`cargo test -p ferrite-forward -F metal --test wavefront_mega_gpu` = 6 green: 5 mega + A2; also
`cargo test -p ferrite-metal-kernels --test quantized_qmv_test` = 12 green): each `*_mega` is BIT-EXACT vs its
whole [[kernel]] for P∈{1,2,4,10}; correctness vs an independent reference (qmv: the 12-test parity suite;
rmsnorm/rope/silu·mul: CPU ref within noise floor — Metal may FMA-contract / Metal-exp ≠ Rust-exp; attention: a
CONSTANT-V cache ⇒ output == that kv-head's V regardless of scores, a reference-free softmax/weighted-V/paging
check). `cargo build -p ferrite-metal-kernels` compiles all MSL. fmt clean (Rust test).

Files: NEW `shaders/mittens/{qmv,rmsnorm,rope,silu_mul,attention}.h`; MODIFIED
`shaders/{quantized_qmv,rmsnorm,rope,silu_mul,attention}.metal` (include + thin wrappers + `*_mega`) and
`tests/wavefront_mega_gpu.rs` (+4 mega tests). `/tmp/{quantized_qmv,attention}.metal.bak` are pre-edit backups.

LSP shows false-positive errors on `mittens/*.h` (`metal_stdlib` not found, `device`/`constant` unknown) — that's
the editor's C++ clang, NOT the Metal compiler; `xcrun metal` (via build.rs) is the real gate and passes.

**NEXT = the full-layer megakernel (the "big metal build", ≈T6) — compose the atoms, never hand-write:**
1. Build ONE persistent decode-layer megakernel composing the mittens atoms in the SSA dataflow order
   (rmsnorm → qkv qmv → rope → attention → o_proj → rmsnorm → gate/up qmv → silu·mul → down qmv → residual),
   driven by `region_schedule`'s 10 tapes with A2's atomic u32-packed cross-WORKER handoff for the N-block
   activation joins + p2p `wf_signal`/`wf_wait` flags; intra-worker edges stay non-atomic.
2. KEY DESIGN PROBLEM (deferred here intentionally): reconcile per-atom TG shapes in one persistent TG — rmsnorm
   wants a big-TG reduction (tg_size = min(N,1024)), qmv wants 2 simdgroups (64 threads) per 8-row group,
   attention wants 32 simdgroups (1024 threads). Pick a fixed TG (e.g. 1024 = 32 simdgroups) and have each atom
   use the subset it needs. THIS is where TG-shape decisions belong — that is why the per-atom proofs each used
   the atom's own natural TG rather than forcing a shared one.
3. Bit-exact vs sequential (Tier A) → then full 16 layers + lm_head via `region_schedule` → wire as the alt decode
   path (`FERRITE_WAVEFRONT_GPU`-style env gate) → temp=0 bit-exact vs ferrite-metal non-mega (Tier B, e2e —
   the FIRST whole-system check the thin-wrapper extractions get) → persistent-launch occupancy sizing → ≥5-run
   perf vs 8.2 ms (min/median/p99, distributions must not overlap).

Constraints unchanged: build/run `--bin vllm -Fmetal FERRITE_MODELS=llama-3.2-1b`; ONE vllm chat at a time; 24 GiB
cap; `cargo build -p ferrite-metal-kernels` (no feature) compiles MSL; `xcrun metal` resolves `#include
"mittens/..."` relative to the .metal; MSL device atomics are relaxed-only; NEVER hand-write kernel math (compose
atoms — this is what killed `_mt`); user gates commits.

## (superseded) LATEST STATUS — 2026-05-26 (typed-OpDataflow IR + end-to-end metal decode; B != A open)

Two commits past `24997d429` on `worktree-pd-wavefront`. Detail in memory `[[pd-clean-wavefront-design]]`
UPDATEs 10–14; the short version:

**The GPU subtile decode path is wired end-to-end and runs on real Llama-3.2-1B** (env `FERRITE_WAVEFRONT_GPU=1`).
Pipeline: `subtile_compile::compile_decode` (runtime decode `Instruction` tape → `SubtileIr`) → `validate` →
`subtile_player::{resolve_pipelines, resolve_buffers}` → `play` (trivial `MetalExecutor`). Wired as a
non-destructive A/B check in `MetalWorkerPool::wavefront_ab_compare` (runs the wavefront, compares logits vs
the normal forward, restores the trusted result). N-block qmv linchpin proven bit-exact on-device.

**The SubtileIr is now compile-time-shape-safe + memory-error-catching** (the session's main thrust, per the
user): `OpDataflow` is a typed fixed-arity enum (NOT `Vec`), `RegionRef` is a 2-D `Region` (row/col, not byte
intervals), and `validate` does region-overlap use-before-def + partial-coverage on arena slots. 43 host tests;
`validate_rejects_{read_before_write,partial_coverage}` prove the checks fire. The player stays trivial.

**OPEN — the only thing left for a bit-exact decode: root-cause `B != A`.** On the live decode, `validate`
PASSES (429 subtile instrs) yet the wavefront logits differ from the baseline (first diff @ byte 0). Per the
compile-time-or-residue discipline: validate passing ⇒ it's NOT a dataflow bug. Next, in order:
1. **Bound the A/B compare to the logit region** (`vocab*2 ≈ 256 KB`), not the whole 1.05 GB terminal buffer
   (sized for the 4096-tok prefill bucket) — the mismatch *count* is polluted by stale tail + coloring reuse;
   `first diff @ byte 0` is the real signal.
2. **Lift the `binding⟺region` invariant** into `validate` (a binding's byte offset == its region's
   col-offset × elem_bytes; the write binding's buffer == the dataflow write buffer). For the qmv these come
   from the same `n0` (consistent by construction), which already argues the bug is elsewhere.
3. **lm_head slice-vs-raw** (PRIME residue suspect): the baseline runs `lower_pair`'s single-seq
   gather→qmv→scatter slice; the wavefront runs the raw lm_head `AffineQmm`. To localize, dump wavefront vs
   normal arena slots op-by-op and find the first divergence.

Synth-fused decode tape note: q/k/v/gate/up are inside `SynthPreAttn`/`SynthMlpPreDown` megakernels (dispatched
whole); only o_proj/down_proj/lm_head are standalone qmvs that get N-blocked. Un-fusing to N-block the rest is
a later step. `lower_pair`'s lm_head slice is NOT in the source tape (synthesized at lowering) — so matching the
source dataflow wouldn't flag the slice difference.
