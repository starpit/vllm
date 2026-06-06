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
SubtileTape   (linear, target-agnostic; AllocSlot + Compute + FreeSlot + OpenLoop + CloseLoop — slot lifecycle wraps each Compute)
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
  scrubbed: the `MetalTape` type itself was deleted in commit 10
  (the v1 Metal-runtime ISA had no consumers in the new pipeline);
  `metal_tape.rs` survives as a 153-LOC carcass holding leaf types
  (`BufId`, `PipeId`, `FlagId`, `WeightRole`, `WeightBundle`,
  `WeightLoc`, `InputKind`, `BufferRef`) still referenced by
  callers outside the wavefront pipeline.
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
SoftmaxStateId (runtime id; phase ordering procedural via `pending_post_loop` in `lower_tape_to_tk`. The `SoftmaxState<Phase>` typestate is deferred future work — see §6 deferral list) / RopeForm / KvCacheLayout / KvCacheProducer
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
| `RopeForm` | type-level (sealed `RopeForm` trait, threaded as `SubOp<F: RopeForm>` / `SubtileNode<F>` / `SubtileIR<F>`; `SubOp::Rope*` carries `PhantomData<F>`) | n/a (carried by the SubtileIR `<F>`) | flat variant-identity split: `Instr::RopeRotateNeoX` / `Instr::RopeRotateInterleaved` — never a runtime `RopeFormTag` field; the `Instr::rope_rotate::<F>` constructor matches once on `F::TAG` to pick the variant. A Q/K-side mismatch between rope nodes is a wrong-variant Rust type error |
| `SoftmaxStateId` | id on `SubOp::AttnDecode` | n/a (one `Compute` for AttnDecode; the 4-phase split is the lowering's job) | sealed runtime id field on `Instr::AttnDecodeInit/Qkt/Sv/Finalise`; ordering is enforced procedurally by `emit_attn_decode`'s syntax-directed walk (`pending_post_loop` queue), NOT by typestate. The `SoftmaxState<Phase>` typestate (Sv-before-Qkt = no impl) was originally specced here but is deferred to a future commit that lifts the ordering proof to the type system. Today `feedback_compile_time_or_garbage` is unmet for this witness only. |
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
    *Lands with §6.5 shmem-promotion pass*: the commit-6 conservative
    all-gmem path pairs `Wait{Ready}` with producer `Arrive{Done}`
    rather than with `LoadAsync`, so the closure check only becomes
    meaningful once pipelined Load→Wait edges land.
  - **Page-cycle closure**: every page slot's
    `Consumed → Ready → Done → Consumed` cycle closes — no orphan
    transitions. *§6.5 pass postcondition*.
  - **Phase parity**: `start_parity` propagated correctly across loop
    iterations; phantom-round arrives present where required.
    *§6.5 pass postcondition*.
  - **Edge closure**: every shmem-promoted edge has exactly one
    matching arrive/wait on the same page at the same const-generic
    `Phase`. *§6.5 pass postcondition*.
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

6. **Implement `lower_tape_to_tk(&SubtileTape, &SubtileIR<F, K>) ->
   TkTape`.** Trivial syntax-directed translation. **Always-executable
   invariant: the output is a complete, validator-green tape that
   runs correctly.** Slot mapping: each `SlotId` becomes a `PageId`
   on a conservative all-gmem path. The conservative-path edge
   protocol is split:
   - **Producer write**: `StoreAsync` + `CommitGroupBulk` +
     `ThreadfenceDevice` + `PageBarrierArrive{Done}`. The
     `CommitGroupBulk` lands the in-flight TMA stores before the
     fence publishes them.
   - **Consumer read of a slot edge**: `PageBarrierWait{Ready}` only.
     The producer's `StoreAsync` to the page IS the data path on
     this all-gmem lowering; a redundant `LoadAsync` would be
     busywork. (The shmem-promotion pass at §6.5 will rewrite slot
     edges to `Wait{Ready}` + `LoadAsync` once it runs.)
   - **External (source-tensor) read**: `LoadAsync` only (no
     preceding `Wait` — the dst page is freshly allocated).
   - **`FreeSlot`**: recycles the page id.

   `SubOp::AttnDecode` lowers to four flat `Instr` variants —
   `Instr::AttnDecodeInit` (outside loop) → loop[
   `Instr::AttnDecodeQkt`, `Instr::AttnDecodeSv` ] →
   `Instr::AttnDecodeFinalise` — with the SubtileTape OpenLoop/
   CloseLoop pair becoming the inner `Instr::ForLoop`. Phase
   ordering (Init→Qkt→Sv→Finalise) is enforced procedurally by
   `emit_attn_decode`'s syntax-directed walk; the
   `SoftmaxState<Phase>` typestate originally specced here is
   deferred (see §2 row note). `KvCacheProducer` consumed via
   exhaustive match. `RopeForm` is threaded as `<F: RopeForm>` and
   produces a flat variant-identity split at the `Instr` boundary
   (`RopeRotateNeoX` / `RopeRotateInterleaved`). `KvCacheLayout<K>`
   propagates via the tape's interned `kv_layouts` table; consumer
   Instrs (RopeRotate*, AttnDecode*) carry only `kv_layout:
   KvLayoutId` and read `head_dim` / `num_kv_heads` via the
   single-source method `tape.kv_layout(id)`.
   **No analysis, no lookahead, no shmem decisions** — those are the
   optimizer's job.

6b. **`validate_tk_tape`** — staged. Commit 6b lands the
    fence-before-arrive check on every Gmem-routed cross-worker
    edge (FenceDevice or stricter required; ThreadfenceBlock is
    CTA-scope and rejected). The remaining checks —
    `LoadAsync` ↔ `PageBarrierWait{Ready}` matching, page-cycle
    closure, phase parity, edge closure — land in commit 6.5
    alongside the optimizer passes that introduce the
    pipelined/shmem-promoted edges they target. Runs after
    lowering AND after every optimizer pass.

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

8. **`tk_player::emit_kernel(name: &str, &TkTape) -> String` becomes
   the single trivial match.** Public entry-point takes the kernel
   name as a separate parameter (the FFI symbol the launcher binds
   to). Player body is one `match` over `Instr`, every arm ≤5
   lines, no `_` arm. The K1 kill criterion (trivial player) is
   enforced by these structural rules.

   Two carve-outs from the original "no `format!` outside player"
   wording: (a) `tk_tape::ByteOffset::{from_const, linear_loop}`
   bake CUDA byte-offset expressions at *tape-build* time (not
   emit time) so the player emits literally — these are pre-baked
   fields, not emit helpers. (b) The "byte-identical to pre-existing
   CUDA text (golden test)" verification mechanism is deferred:
   today the player has substring-equality and per-arm tests; a
   checked-in `.cu.golden` snapshot will land alongside the §6.5
   optimizer passes (which are the only thing that could change
   the emitted text outside the IR shape).

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

Snapshot updated for HEAD post-audit-`w95ad4bpn` (Parity sealed enum + GemmK witness):

| Layer | Before | Now (snapshot) | After (target) |
|---|---|---|---|
| `subtile.rs` + `region.rs` | ~2300 LOC | folded into `subtile_ir.rs` (1696 LOC) | folded ~1400 LOC |
| `subtile_ir.rs` (Metal-flavored, v1) | 1471 LOC | scrubbed to `metal_tape.rs` (153 LOC carcass; MetalTape type deleted in commit 10, leaf types kept for callers) | further-cut to ~50 LOC once leaf-type callers migrate |
| `subtile_tape.rs` | 0 LOC | 1734 LOC (slot-lifecycle + validate_subtile_tape landed; tests dominate) | ~600 LOC after the §10 cleanup |
| `lower.rs` (`LoweringInput`/`LoweredOp`) | 1105 LOC | 213 LOC (much already cut) | DELETED |
| `tape.rs` | 391 LOC | DELETED ✓ | DELETED |
| `tk_tape.rs` | 727 LOC | 1128 LOC (parity-split Wait variants, KvLayoutId table, KvCacheShape const-generic K, GemmK typed witness, validate_tk_tape) | grows for §6.5 pass postcondition checks (closure / parity / edge-pairing) |
| `tk_player.rs` | 248 LOC | 758 LOC (one ≤5-line tk20:: arm per Instr; emit_kernel host wrapper; tape-resolved KvLayoutEntry consumer) | ~700 LOC once scaffolding `kittens::*` strings migrate to `tk20::*` helpers |
| `tk_lower.rs` | 114 LOC | DELETED ✓ | DELETED |
| `lower_tape_to_tk` (commit 6, conservative) | 0 LOC | 1192 LOC ✓ (4-phase AttnDecode split, KvLayout interning, K-shape threading, GemmK::derive call, conservative all-gmem routing) | trims as §6.5 pass postconditions land |
| Optimizer passes (commit 6.5.*) | 0 LOC | 0 LOC (NOT-YET-IMPLEMENTED — sequenced after 6b) | ~700 LOC across headline passes |
| Validators (5b, 6b) | 0 LOC | 5b ✓ inline in subtile_tape.rs (~270 LOC); 6b ✓ inline in tk_tape.rs (~150 LOC, conservative checks; closure / parity / edge-pairing land alongside §6.5 passes) | ~300 LOC each at §6.5 land |
| **Net status** | | K8-target file set is currently +2.4k LOC vs Before (subtile_tape.rs/tk_tape.rs/tk_player.rs/lower_tape_to_tk.rs grew alongside the new substrate; carcasses mega.rs ~2949, launcher.rs ~1150, partition.rs ~1096, region_schedule.rs ~590 pending §10 deletions which would close the gap) | net ≥ -1500 LOC (K8) |
