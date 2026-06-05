# Wavefront IR redesign — Fuf → SubtileIR → SubtileTape → TkTape

Status: APPROVED. This supersedes the v1 plan (moved to
`.attic/SUBTILE_IR_REDESIGN_v1.md`).

## 0. The architecture

```
Fuf  (ferrite-forward macro internal)
 │
 ▼
SubtileIR     (DAG, target-agnostic; "the math at subtile granularity")
 │
 ▼            [DAG → linear: a topological linearization, sequential semantics]
SubtileTape   (linear, target-agnostic; just Compute + OpenLoop + CloseLoop)
 │
 ▼            [target-agnostic → target-specific; work-distribution + sync chosen here]
TkTape  /  MetalTape  /  …
 │
 ▼            [trivial executor, no interpretation]
GPU
```

**SubtileTape carries no parallelism, no sync, no memory tier.** Workers
(CTAs / threadgroups / warp roles), barriers, fences, signals, waits,
shmem-vs-gmem routing — all of that is target-specific and lives at
the per-target lowering. The tape is a single-thread topological order
of the SubtileIR's Computes, with explicit `OpenLoop` / `CloseLoop`
brackets for runtime-bounded loops. The DAG parallelism is recoverable
by the lowering from the SubtileIR's region-overlap predecessors.

Two **syntax-directed** lowerings (`lower_dag_to_tape`,
`lower_tape_to_tk`); a pipeline of **`TkTape → TkTape` optimizer
passes** (shmem promotion, fence narrowing, page coalescing, …); two
validators (one per tape; the TkTape validator runs after lowering and
after every pass); one trivial player per target. **Compile-time safety
at every layer** — all wirings are proof-carrying typed witnesses.

**The load-bearing invariant: TkTape is always executable.** The output
of `lower_tape_to_tk` is a complete, validator-green tape that runs
correctly (conservative all-gmem routing). Every optimizer pass is a
strict performance rewrite — disabling any pass yields a kernel that
is correct, only slower. Whether an edge can live in shmem is a
target-specific question (smem capacity, mbar slot count,
NUM_CONSUMER_WARPS, page-lifetime windows), so the analysis lives at
TkTape — not at the target-agnostic SubtileTape, and not at lowering
time. We are a compiler.

## 1. Naming

- `subtile.rs::SubtileGraph` (current) → `subtile_ir.rs::SubtileIR`
  (the DAG; reclaim the name, it IS the SubtileIR).
- `subtile_ir.rs::SubtileIr` (current, Metal-flavored) →
  `metal_tape.rs::MetalTape` (drop the "neutral" pretense — the
  `PipelineSpec` etc. were always Metal-shaped).
- NEW `subtile_tape.rs::SubtileTape` — the missing linear,
  target-agnostic layer where every DAG edge becomes an explicit
  Wait/Signal/Fence/MemoryRoute instruction.
- `tk_tape.rs::TkTape` stays.
- The DAG layer absorbs `region.rs` (region-granularity SSA wins;
  v1 whole-output `Operand::Sub` dies).

## 2. Compile-time safety contract

Every step lands typed witnesses. The pattern is the existing
SoftmaxState<Phase> / RopeForm / KvCacheLayout / KvCacheProducer
shape, applied uniformly.

Sealed newtypes throughout: `SubtileId`, `SourceId`, `BufId`, `FlagId`,
`BarrierId`, `LoopVarId`, `KernelArgRef`, `SoftmaxStateId`,
`KvLayoutId`, `PageId`. Construction only via crate-private builders.

Witness placement (per layer):

| Witness | DAG | SubtileTape | TkTape |
|---|---|---|---|
| `KvCacheLayout` | field on `SubOp::RopeAppend`, `SubOp::AttnDecode` (single instance per K-cache BufId; both reach for same value) | n/a (SubtileTape carries no per-Compute fields beyond `SubtileId`; lowering looks up the witness on the SubtileIR node) | propagates; consumer reads via single-source method |
| `KvCacheProducer` | field on `SubOp::AttnDecode` | n/a (same reason) | exhaustive match in lowering, no `_ =>` arm |
| `RopeForm` | const generic on `SubOp::Rope*` | n/a (carried by the SubtileIR `<F>`) | const generic on `ComputeBody::RopeRotate`'s template |
| `SoftmaxState<Phase>` | id on `SubOp::AttnDecode` | n/a | typestate threaded through `lower_tape_to_tk`'s emit of init/qkt/sv/finalise; template field-substitution |
| `LoopVarId` | n/a | sealed; `OpenLoop(v)` and matching `CloseLoop(v)` share the same id by construction | sealed; carries through to `Instr::ForLoop` |
| `Phase` (parity) | n/a | n/a | const-generic split `Instr` variants on TkTape; runtime→const dispatch at the optimizer pass's `match` site, never as a `u8` field |

## 3. The validator stretch

Two layers, both sealed.

### 3.1 `TapeBuilder<S>` — typestate, compile-time

Every `push_*` is a state transition. State `S` tracks live buffers,
signaled barriers, in-flight `ForLoop`s. Misuse = no method on that
state.

Examples:
- `Wait(B)` on a `B` never `Signal`'d in this worker's prefix → no
  matching impl, compile error.
- Closing a `ForLoop` that never opened → no matching impl.
- Reading a buffer no producer wrote → no matching impl.

### 3.2 `validate(&Tape) -> Result<(), Vec<ValidationError>>` — abstract interpreter, runtime

For invariants the typestate can't see at compile time (e.g. dynamic
barrier-id matching, cross-worker deadlock cycles).

Two validators, mirroring the two tapes:

- **`validate_subtile_tape(&SubtileTape, &SubtileIR)`** — target-
  agnostic, IR-shape only. No worker, no barrier, no fence, no memory
  class — none of those exist at this layer.
  - **Compute well-formedness**: every `Compute` names an in-range
    `SubtileId`; every SubtileIR node is `Compute`'d exactly once;
    adjacent Computes appear in strictly ascending `SubtileId` order
    (the tape is a topological linearization of the DAG).
  - **Loop balance**: every `OpenLoop` has a matching `CloseLoop`
    with the same `LoopVarId`; no nesting; no unclosed loops.
- **`validate_tk_tape(&TkTape)`** — TK-specific. Runs **after lowering
  and again after every optimizer pass**. Each pass declares the
  postcondition it must preserve; the validator is the conjunction.
  - **Fence-before-arrive** (the missing-fence check, but at the layer
    where fence is a real concept): for every Gmem-routed cross-worker
    edge, between the producer's last `StoreAsync` to the edge's
    buffer and the matching `PageBarrierArrive`, there is a
    `FenceDevice` (or stricter scope as required by the consumer's
    reach).
  - **LoadAsync ↔ PageBarrierWait{Ready}** matched downstream.
  - **Page-cycle closure**: every page slot's
    `Consumed → Ready → Done → Consumed` cycle closes — no orphan
    transitions.
  - **Phase parity**: `start_parity` propagated correctly across loop
    iterations; phantom-round arrives present where required.
  - **Edge closure**: every shmem-promoted edge has exactly one
    matching arrive/wait on the same page at the same const-generic
    `Phase`.
  - **Pass postconditions**: each optimizer pass adds its own
    invariant to the assertion set (e.g. shmem-promotion guarantees
    `consumer_count == 1` and "no other edge aliases this `(buf,
    offset)` on the producer worker"; fence-elimination guarantees
    every dropped fence had no unprotected `StoreAsync` reaching its
    arrive).

Both validators run as `assert!` at the exit of their producing fn —
invalid tape never escapes; consumer fns can assume validity.

**Kill criterion (push invariants up to types):** if either validator
finds an invariant the typestate-builder *should* have caught at
compile time, push the invariant up into the type system; don't keep
it as runtime check.

**Kill criterion (no lowering-time analysis):** if an optimizer pass
needs a fact `lower_tape_to_tk` doesn't trivially expose, the fact
moves onto the IR (typed field on the relevant Instr or edge record)
or into a prior pass's postcondition — never into the lowering. The
lowering stays O(n) syntax-directed.

## 4. Staged plan (12 commits, each buildable, each green)

Every commit: cargo build clean, cargo test green, no `unwrap_or_else`
fallbacks added, no `_ =>` match arms added.

1. **Rename `subtile_ir.rs` → `metal_tape.rs`, type `SubtileIr` →
   `MetalTape`.** Pure `git mv` + s///. No semantics. Frees the name
   "SubtileIR" for what it is.

2. **Fold `region.rs` + `subtile.rs` into `subtile_ir.rs::SubtileIR`.**
   Region-granularity SSA wins. v1 whole-output `Operand::Sub` dies.
   `tape.rs` (per-worker scheduling artifact) stays alive temporarily.

3. **Add `subtile_tape.rs`, additive, no consumers.** Full module:
   instrs, sealed handles, `validate_subtile_tape`, `play` skeleton.
   Sealed `BarrierId` (Wait/Signal share by construction).
   Compile-fail tests for orphan handles, mismatched barrier ids.
   **Constraint inventory + policy defaults (keep-in-smem,
   chain-local placement, coarse-loops-only) live in
   [`SUBTILE_TAPE_CONSTRAINTS.md`](SUBTILE_TAPE_CONSTRAINTS.md) —
   the source of truth for what every DAG edge must surface as.
   Anything the walker (commit 5) needs that isn't there yet
   gets added there first, never as a one-off in lowering code.**

4. **Move typed witnesses onto SubtileIR DAG nodes.** `KvCacheLayout`,
   `KvCacheProducer`, `RopeForm` (const-generic phantom),
   `SoftmaxStateId` become fields on the relevant `SubOp` variants.
   `tk_lower.rs` thinned. Compile-fail doctests for: layout drift
   between producer and consumer; mixed `NeoX`/`Interleaved` in one
   forward; non-exhaustive `KvCacheProducer` match.

3.b **Scrub target-leaky concepts from SubtileTape.** Delete
   `Instr::Fence`, `Instr::Route`, `MemoryClass`, and any `Route`-class
   field on `Signal`/`Wait`. SubtileTape carries DAG facts only
   (Compute, Signal, Wait, OpenLoop, CloseLoop). Tests for those
   variants delete or move down to TkTape goldens. Companion edits in
   `SUBTILE_TAPE_CONSTRAINTS.md`: strike rows 3 (memory routing) + 4
   (memory hazard / fence) from the constraint inventory; strike the
   "keep activations in shared memory" policy default (it's a TkTape
   pass policy, not a SubtileTape rule). The `subtile_tape.rs` module
   header documents: "no smem, no gmem, no fence, no page, no parity".

5. **Implement `lower_dag_to_tape(&SubtileIR<F>) -> SubtileTape`.**
   Single deterministic fn: validate the IR, walk
   `graph.nodes` in ascending `SubtileId` order, emit one `Compute`
   per node — wrapping `SubOp::AttnDecode` in an
   `OpenLoop`/`CloseLoop` pair over a runtime-bounded count (the
   KV-sweep). **No worker assignment, no Signal/Wait emission, no
   fence, no route** — every parallelism / sync / memory-tier
   decision lives at the per-target lowering. Returns a validated
   `SubtileTape`. Unit tests: chain, attn-decode loop wrap, multiple
   AttnDecodes get distinct LoopVarIds.

5b. **`validate_subtile_tape`** — full impl + tests. Compute
    well-formedness (every node Compute'd exactly once; ascending-id
    adjacency); loop balance (matching `OpenLoop`/`CloseLoop`, no
    nesting, no unclosed loops). Runs at `lower_dag_to_tape`'s exit.

6. **Implement `lower_tape_to_tk(&SubtileTape) -> TkTape`.** Trivial
   syntax-directed translation. **Always-executable invariant: the
   output is a complete, validator-green tape that runs correctly.**
   Conservative all-gmem routing for every cross-worker edge: TMA
   store + `FenceDevice` + `LoadAsync` + `PageBarrierWait{Ready}`.
   Page/buffer allocation lives here (full-tape liveness available).
   `SoftmaxState<Phase>` typestate enforced (Sv-before-Qkt = no impl).
   `KvCacheProducer` consumed via exhaustive match. `RopeForm`
   const-generic threaded through. **No analysis, no lookahead, no
   shmem decisions** — those are the optimizer's job.

6b. **`validate_tk_tape`** — full impl + tests. Fence-before-arrive
    on every Gmem-routed cross-worker edge, `LoadAsync` ↔
    `PageBarrierWait{Ready}` matching, page-cycle closure, phase
    parity, edge closure. Runs after lowering AND after every
    optimizer pass.

6.5. **`TkTape → TkTape` optimizer passes.** Ordered pipeline; each
    pass is a strict performance rewrite that preserves all post-pass
    invariants. Lands as one or more commits; the floor is shmem
    promotion (the headline optimization the redesign exists for).
    Each pass declares: input precondition, output postcondition,
    target-knowledge consumed (smem capacity, mbar slot count,
    NUM_CONSUMER_WARPS, page-lifetime windows, parity allocator
    state), and idempotency. Headline passes:
    - **`promote_shmem_carry_forward`** — rewrite a single-consumer
      Gmem edge to a cross-IType mbar handshake on a shared smem
      page (skip TMA store + skip TMA load). Phase parity flows
      through const-generic `CarryForwardWaitP{0,1}` Instr variants;
      runtime→const dispatch happens once at the pass's `match` site.
    - **`narrow_fence_scope`** — `FenceDevice` → `FenceBlock` when no
      cross-CTA reader exists (single-CTA persistent kernels: always).
    - **`eliminate_dead_fences`** — drop a fence whose every reachable
      Gmem `StoreAsync` already has another fence between it and the
      next arrive.
    - **`coalesce_pages`**, **`barrier_init_hoist`**, parity-aware
      reorderings — added as the working perf knobs require them.
    Disabling any single pass yields a kernel that is correct, only
    slower (per kill criterion K6).

7. **Wire `to_wavefront.rs` to build `SubtileIR` directly.** Delete
   `lower.rs::LoweringInput` + `LoweredOp` (~1100 lines). The
   ferrite-forward macro now produces `SubtileIR` SSA in one pass.
   `partition.rs` re-targeted to consume `SubtileIR`.

8. **`tk_player::play(&TkTape) -> String` becomes the single trivial
   match.** Strip every `format!` from `tk_tape.rs`'s emit helpers.
   Player is one `match` over `Instr`, every arm ≤5 lines, no `_`
   arm. Pre-existing CUDA text byte-identical (golden test).

9. **End-to-end on H100.** Fuf → SubtileIR → SubtileTape → TkTape →
   nvcc → run on Llama-3.2-1B. **The Paris-decode bug must surface
   as a compile error** via the typed `KvCacheLayout` witness, not
   as a runtime divergence. PR-block on coherent decode output.

10. **Delete carcasses.** `tape.rs`, v1 `scheduler.rs`, empty
    `region.rs`, fold `region_schedule.rs` into `lower_dag_to_tape`,
    dead Metal-only `MetalTape` arms. `cargo deadcode` clean.

11. **Memory bridge.** `to_wavefront.rs`'s `Fuf → SubtileIR`
    construction is host-pure / Mac-testable. Add a property test:
    every Fuf the macro can produce builds a tape that
    `validate_subtile_tape` accepts.

12. **Re-audit.** Run the same invention-audit workflow that produced
    the original 137 findings on the new substrate. Goal: ≤10
    verified inventions, all rated low bug-likelihood.

## 5. Kill criteria

Revert the entire stack to commit `5206d10d49` if **any** of:

- **K1 (trivial player).** Step 8's `play` requires any arm > 5 lines
  OR any conditional beyond `match` on instr kind, OR any arm
  containing more than one TK 2.0 call. The optimization that motivated
  the cleverness should have been a TkTape pass.
- **K2 (target-agnostic SubtileTape).** `subtile_tape.rs` source
  contains any of `smem`, `gmem`, `Route`, `Fence`, `Page`, `Phase`,
  `parity`, `Worker`, `Signal`, `Wait`, `Barrier`. Mechanically
  grep-checkable. SubtileTape leaks parallelism or memory model →
  re-design, not patch.
- **K3 (always-executable TkTape).** `lower_tape_to_tk`'s output is
  not a runnable, validator-green tape. Optimizer passes are *strict
  performance* rewrites; if a pass is required for correctness, the
  conservative lowering was wrong.
- **K4 (no lowering-time analysis).** `lower_tape_to_tk` does any
  cross-Instr analysis (page-liveness window, consumer-count walk,
  parity allocator state). Lowering is O(n) syntax-directed; analysis
  belongs in passes.
- **K5 (compile-time witnesses).** Either validator (5b, 6b) finds a
  class of error the typestate builder *should* have caught at
  compile time. Per `feedback_compile_time_or_garbage`: push the
  invariant up to a typed witness / sealed proof / const-generic
  with `where`-clause; don't ship as runtime check.
- **K6 (every pass is strict optimization).** Disabling any single
  optimizer pass post-lowering does not yield a kernel that is
  correct (only slower). The pass is doing correctness work that
  belongs at lowering or in a typed witness.
- **K7 (Paris compile error).** Step 9's Paris-decode bug does not
  surface as a *compile error* in the new typed-witness pipeline.
  Witnesses placed wrong.
- **K8 (net-surface).** Step 10's surface area has not net-shrunk by
  ≥1500 LOC vs `5206d10d49`. This was a rename party, not a redesign.

## 6. What this plan does NOT cover

- Per-arch differences (Mistral, Qwen, DeepSeek). Llama-3.2-1B first;
  other arches follow once the substrate is proven.
- Quantized weights (GGUF, AWQ, GPTQ, FP8, BNB4). Dense bf16 only.
- Performance tuning (page count, NUM_CONSUMER_WARPS, buckets). The
  redesign is correctness-first; perf knobs become tape fields and
  are tunable post-cutover.
- TP > 1 (NCCL all-reduce). Single-rank decode emit only.
- The Metal path's runtime executor stays on `MetalTape`; the
  `lower_tape_to_metal` lowering is out of scope (tracked separately).

## 7. Net surface delta (target)

| Layer | Before | After |
|---|---|---|
| `subtile.rs` + `region.rs` | ~2300 LOC | folded into `subtile_ir.rs` ~1400 LOC |
| `subtile_ir.rs` (Metal-flavored) | 1471 LOC | renamed `metal_tape.rs`, dead arms deleted (~900 LOC) |
| `subtile_tape.rs` | 0 LOC | new ~500 LOC (smaller after 3.b scrub: no Route/Fence/MemoryClass) |
| `lower.rs` (`LoweringInput`/`LoweredOp`) | 1105 LOC | DELETED |
| `tape.rs` | 391 LOC | DELETED |
| `tk_tape.rs` | 727 LOC | grows ~+200 LOC (split fence Instrs, edge records, post-promote variants) |
| `tk_player.rs` | 248 LOC | grows to ~600 LOC (full emit, ≤5 lines/arm) |
| `tk_lower.rs` | 114 LOC | DELETED (witnesses fold into SubtileIR) |
| `lower_tape_to_tk` (commit 6, conservative) | 0 LOC | new ~400 LOC |
| Optimizer passes (commit 6.5.*) | 0 LOC | new ~700 LOC across the headline passes |
| Validators (5b, 6b) | 0 LOC | new ~300 LOC each |
| **Total** | **~6500 LOC** | **~4900 LOC** (net -1600; meets K8 ≥1500) |
