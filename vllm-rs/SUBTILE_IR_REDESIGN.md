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
  **slot-lifecycle instruction** (`AllocSlot` / `Compute { writes,
  reads }` / `FreeSlot`) plus loop brackets (`OpenLoop` / `CloseLoop`).
  The slot lifecycle is the hazard model: `SlotHandle` (move-only)
  → `SlotWritten` (move-only, multi-borrow for readers) → freed.
  Per-target sync primitives (Wait / Signal / Fence / MemoryRoute) live
  ONLY at TkTape — slot-physical-realization is target-specific.
- `tk_tape.rs::TkTape` — flat `Instr` enum, one variant per
  TK 2.0 / CUDA primitive (no nested `ComputeBody`); source identifiers
  are `subtile_ir::TensorId` (no `BufId`).
- The DAG layer absorbs `region.rs` (region-granularity SSA wins;
  v1 whole-output `Operand::Sub` dies).

## 2. Compile-time safety contract

Every step lands typed witnesses. The pattern is the existing
SoftmaxState<Phase> / RopeForm / KvCacheLayout / KvCacheProducer
shape, applied uniformly.

Sealed newtypes throughout: `SubtileId`, `TensorId`, `SlotId`,
`SlotHandle` (move-only), `SlotWritten` (move-only), `LoopVarId`,
`RuntimeBoundId`, `KernelArgRef`, `SoftmaxStateId`, `KvLayoutId`,
`PageId`. Construction only via crate-private builders. (`BufId`
from the v1 metal_tape namespace is dead at the TkTape layer; it
survives only in the §10-deletion carcasses pending their own
removal.)

Witness placement (per layer):

| Witness | DAG | SubtileTape | TkTape |
|---|---|---|---|
| `KvCacheLayout` | field on `SubOp::RopeAppend`, `SubOp::AttnDecode` (single instance per K-cache `TensorId`; both reach for same value) | n/a (SubtileTape carries no per-Compute fields beyond `SubtileId`; lowering looks up the witness on the SubtileIR node) | propagates; consumer reads via single-source method |
| `KvCacheProducer` | field on `SubOp::AttnDecode` | n/a (same reason) | exhaustive match in lowering, no `_ =>` arm |
| `RopeForm` | const generic on `SubOp::Rope*` | n/a (carried by the SubtileIR `<F>`) | const generic on `Instr::RopeRotate`'s template (flat, not nested) |
| `SoftmaxState<Phase>` | id on `SubOp::AttnDecode` | n/a (one `Compute` for AttnDecode; the 4-phase split is the lowering's job) | typestate threaded through `lower_tape_to_tk`'s emit of `Instr::AttnDecodeInit` → loop[`Qkt`/`Sv`] → `Instr::AttnDecodeFinalise`; flat Instr variants, no `ComputeBody` indirection |
| `SlotHandle` / `SlotWritten` | n/a | move-only typestate proofs of slot-write-once and read-after-write; consumed by `compute_to` / `free_slot` | n/a (TkTape uses `PageId`-on-mbarrier handshake for slot realization) |
| `LoopVarId` | n/a | sealed; `OpenLoop(v)` and matching `CloseLoop(v)` share the same id by construction | sealed; carries through to `Instr::ForLoop` |
| `Phase` (parity) | n/a | n/a | const-generic split `Instr` variants on TkTape; runtime→const dispatch at the optimizer pass's `match` site, never as a `u8` field |

## 3. The validator stretch

Two layers, both sealed.

### 3.1 `TapeBuilder<S>` — typestate, compile-time

Every transition is a state change. State `S ∈ { Outside, InsideLoop }`
tracks the loop bracket; the slot lifecycle is enforced by move-only
`SlotHandle` / `SlotWritten` tokens (independent of `S`). Misuse = no
matching impl OR no consumable token, compile error.

Examples:
- Reading a slot before a writer wrote it → no `&SlotWritten` token
  exists, borrow type error.
- Writing the same slot twice → second `compute_to` has no
  `SlotHandle` to consume, move error.
- Reading a slot after free → `free_slot` consumed the `SlotWritten`,
  borrow error.
- `alloc_slot` / `free_slot` inside a loop body → no matching impl on
  `TapeBuilder<state::InsideLoop>` (slot lifetime spans iterations).
- Closing a loop never opened → no matching impl on
  `TapeBuilder<state::Outside>`.
- Calling `finish` with a loop still open → no matching impl on
  `TapeBuilder<state::InsideLoop>`.

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
  - **Slot lifecycle**: each slot transits Allocated → Written →
    Freed exactly once. Errors: `WriteUnallocatedSlot`,
    `ReadBeforeWrite`, `DoubleWrite`, `UseAfterFree`,
    `FreeUnallocatedSlot`, `DoubleFree`, `SlotNeverFreed`. The
    typestate prevents these at compile time when the builder is used;
    the validator backs against hand-built tapes that bypass it.
  - **Slot id range**: every slot id is in `0..num_slots`.
  - **Edge coverage** (load-bearing — this is what "every DAG edge
    is an explicit instruction" reduces to): for every
    `Compute { node, reads }`, the set-of-writers of `reads` equals
    the SubtileIR predecessor set of `node`. Mismatch =
    `EdgeMismatch`.
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
   Compile-fail tests for orphan handles. **Constraint inventory +
   policy defaults live in
   [`SUBTILE_TAPE_CONSTRAINTS.md`](SUBTILE_TAPE_CONSTRAINTS.md) — the
   source of truth for what every DAG edge must surface as.** (As
   landed, this commit's instr set was Compute + Signal/Wait/Fence/
   Route + OpenLoop/CloseLoop. Commits 3.b + 5b + the slot-lifecycle
   rebuild progressively replaced Signal/Wait/Fence/Route/MemoryClass
   with the slot-lifecycle model — see commits below.)

4. **Move typed witnesses onto SubtileIR DAG nodes.** `KvCacheLayout`,
   `KvCacheProducer`, `RopeForm` (const-generic phantom),
   `SoftmaxStateId` become fields on the relevant `SubOp` variants.
   `tk_lower.rs` thinned. Compile-fail doctests for: layout drift
   between producer and consumer; mixed `NeoX`/`Interleaved` in one
   forward; non-exhaustive `KvCacheProducer` match.

3.b **Scrub target-leaky concepts from SubtileTape.** Delete
   `Instr::Fence`, `Instr::Route`, `MemoryClass`, and any `Route`-class
   field on `Signal`/`Wait`. Tests for those variants delete or move
   down to TkTape goldens. Companion edits in
   `SUBTILE_TAPE_CONSTRAINTS.md`: strike rows 3 (memory routing) + 4
   (memory hazard / fence) from the constraint inventory; strike the
   "keep activations in shared memory" policy default (it's a TkTape
   pass policy, not a SubtileTape rule). The `subtile_tape.rs` module
   header documents: "no smem, no gmem, no fence, no page, no parity".

3.c **Drop Signal/Wait/Worker.** Remove `Instr::Signal`, `Instr::Wait`,
   `WorkerId`, and any cross-worker concept from SubtileTape — these
   were over-pruning candidates from 3.b that survived too long.
   (Landed: `ba492c767a`.)

3.d **Slot-lifecycle rebuild — restore hazards-explicit.** 3.c
   over-pruned: it removed Signal/Wait *along with* the load-bearing
   hazard model. Replace with the slot lifecycle:
   `Instr::{ AllocSlot, Compute { node, writes, reads }, FreeSlot,
   OpenLoop, CloseLoop }`. Sealed `SlotId`, move-only `SlotHandle` /
   `SlotWritten` enforce single-writer, write-before-read,
   no-use-after-free at compile time. Validator gains slot-lifecycle
   + edge-coverage checks. Edge coverage is the load-bearing backstop:
   `Compute.reads`'s set-of-writers must equal the SubtileIR
   predecessor set. (Landed: `f8a359d71f`.)

5. **Implement `lower_dag_to_tape(&SubtileIR<F>) -> SubtileTape`.**
   Single deterministic fn: validate the IR, walk `graph.nodes` in
   ascending `SubtileId` order. Per node: `alloc_slot()` mints a
   `SlotHandle`; `compute_to(node, handle, &[<predecessor
   &SlotWritten tokens>])` writes the slot and returns `SlotWritten`;
   `free_slot(written)` retires after the last consumer
   (consumer-count countdown over `predecessors()`). `SubOp::AttnDecode`
   wraps in `OpenLoop` / `CloseLoop`; the slot lifecycle (alloc / free)
   lives outside the bracket. **No worker assignment, no fence, no
   memory class** — every parallelism / sync / memory-tier decision
   lives at the per-target lowering. Returns a validated `SubtileTape`.
   Unit tests: chain, attn-decode loop wrap, diamond multi-reader,
   compile-fail typestate doctests. (Landed across `e46825435a` and
   the slot-lifecycle rebuild `f8a359d71f`.)

5b. **`validate_subtile_tape`** — full impl + tests. Compute
    well-formedness (every node Compute'd exactly once; ascending-id
    adjacency); loop balance (matching `OpenLoop`/`CloseLoop`, no
    nesting, no unclosed loops); slot lifecycle (Alloc → Write → Free
    once); slot id range; **edge coverage** (load-bearing — see §3.2).
    Runs at `lower_dag_to_tape`'s exit. (Landed: `d57a5aa6b4` for the
    structural checks, `f8a359d71f` for slot + edge coverage.)

5c. **Nuke `ComputeBody` + `BufId` from TkTape.** `ComputeBody` was an
    unjustified nesting (`Instr::Compute { body: ComputeBody, role }`
    where every other Instr variant was flat). Flattened into
    `Instr::{RmsNorm, GemmM1, SiluMul, ResidualAdd, RopeRotate,
    AttnDecodeInit/Qkt/Sv/Finalise, DebugOpBeginMarker}`. `BufId` (the
    v1 metal_tape buffer namespace) replaced everywhere on the TkTape
    side with `subtile_ir::TensorId`; `tk_lower.rs` deleted (its
    BufId-keyed witnesses superseded by the canonical TensorId-keyed
    versions in `subtile_ir.rs`). Aligns TkTape with §0's "we are a
    compiler" stance. (Landed: `56413bdf6f`.)

6. **Implement `lower_tape_to_tk(&SubtileTape, &SubtileIR<F>) ->
   TkTape`.** Trivial syntax-directed translation. **Always-executable
   invariant: the output is a complete, validator-green tape that
   runs correctly.** Slot mapping: each `SlotId` becomes a `PageId`
   on a conservative all-gmem path — `Compute` write surfaces as
   `StoreAsync` + `FenceDevice` + `PageBarrierArrive{Done}`; consumer
   reads surface as `PageBarrierWait{Ready}` + `LoadAsync`; `FreeSlot`
   recycles the page id. `SubOp::AttnDecode` lowers to four flat
   `Instr` variants — `Instr::AttnDecodeInit` (outside loop) → loop[
   `Instr::AttnDecodeQkt`, `Instr::AttnDecodeSv` ] →
   `Instr::AttnDecodeFinalise` — with the SubtileTape OpenLoop/
   CloseLoop pair becoming the inner `Instr::ForLoop`. The
   `SoftmaxState<Phase>` typestate threads through these four emits
   (Sv-before-Qkt = no impl). `KvCacheProducer` consumed via
   exhaustive match. `RopeForm` const-generic threaded through.
   `KvCacheLayout` resolved by TensorId on the SubtileIR node.
   **No analysis, no lookahead, no shmem decisions** — those are the
   optimizer's job.

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

Snapshot as of commit `56413bdf6f` (post-nuke):

| Layer | Before | Now (snapshot) | After (target) |
|---|---|---|---|
| `subtile.rs` + `region.rs` | ~2300 LOC | folded into `subtile_ir.rs` (1536 LOC) | folded ~1400 LOC |
| `subtile_ir.rs` (Metal-flavored, v1) | 1471 LOC | renamed `metal_tape.rs` (1471 LOC; dead arms not yet deleted) | dead arms deleted (~900 LOC) |
| `subtile_tape.rs` | 0 LOC | 1662 LOC (slot-lifecycle landed; tests dominate) | ~600 LOC after the §10 cleanup |
| `lower.rs` (`LoweringInput`/`LoweredOp`) | 1105 LOC | 206 LOC (much already cut) | DELETED |
| `tape.rs` | 391 LOC | DELETED ✓ | DELETED |
| `tk_tape.rs` | 727 LOC | 725 LOC (BufId nuked; ComputeBody flat) | grows ~+200 LOC for fence-Instr split + edge records (commit 6+) |
| `tk_player.rs` | 248 LOC | 257 LOC (flat compute arms stubbed) | grows to ~600 LOC (full emit, ≤5 lines/arm) |
| `tk_lower.rs` | 114 LOC | DELETED ✓ | DELETED |
| `lower_tape_to_tk` (commit 6, conservative) | 0 LOC | 0 LOC (next) | ~400 LOC |
| Optimizer passes (commit 6.5.*) | 0 LOC | 0 LOC | ~700 LOC across headline passes |
| Validators (5b, 6b) | 0 LOC | 5b ✓ inline in subtile_tape.rs; 6b pending | ~300 LOC each |
| **Net status** | | -113 LOC (commit 56413bdf6f) on top of slot-lifecycle commits; carcasses pending §10 | net ≥ -1500 LOC (K8) |
