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
 ▼            [DAG → linear: every edge becomes an explicit instruction]
SubtileTape   (linear, target-agnostic; sync/barrier/memory hazards explicit)
 │
 ▼            [target-agnostic → target-specific, table-driven]
TkTape  /  MetalTape  /  …
 │
 ▼            [trivial executor, no interpretation]
GPU
```

Two lowerings (`lower_dag_to_tape`, `lower_tape_to_tk`); two validators
(one per tape); one trivial player per target. **Compile-time safety
at every layer** — all wirings are proof-carrying typed witnesses.

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
| `KvCacheLayout` | field on `SubOp::RopeAppend`, `SubOp::AttnDecode` (single instance per K-cache BufId; both reach for same value) | propagates as opaque id | propagates; consumer reads via single-source method |
| `KvCacheProducer` | field on `SubOp::AttnDecode` | propagates | exhaustive match in lowering, no `_ =>` arm |
| `RopeForm` | const generic on `SubOp::Rope*` | const generic on tape's rope op | const generic on `ComputeBody::RopeRotate`'s template |
| `SoftmaxState<Phase>` | id on `SubOp::AttnDecode` | typestate threaded through `lower_tape_to_tk`'s emit of init/qkt/sv/finalise | template field-substitution |
| `BarrierId` | n/a (DAG has no barriers) | sealed; `Wait(B)` and `Signal(B)` share the same id by construction | sealed; PageBarrier kind layered on top |
| `Phase` (parity) | n/a | const-generic typestate carried by `PageHandle<P>` | propagates; `complete_round_with_parity_correction` returns `PageHandleAfterRuntimeLoop` |

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

- **`validate_subtile_tape(&SubtileTape)`** — target-agnostic checks:
  - **Deadlock cycle**: build worker × barrier digraph; cycle =
    every worker waiting on someone else's signal that never fires.
  - **Data race**: per-buffer `(reader_workers, writer_workers,
    fence_position)`. Reader without fence after another worker's
    latest write = race.
  - **Missing fence**: gmem-crossing edge with no `Fence` between
    producer's `Signal` and consumer's `Wait` = error.
  - **Orphan signal/wait**: every `BarrierId` allocated must have ≥1
    `Signal` and ≥1 `Wait`.
- **`validate_tk_tape(&TkTape)`** — TK-specific checks:
  - Every `LoadAsync` has a matching `PageBarrierWait { Ready }`
    downstream.
  - Every page slot's `Consumed → Ready → Done → Consumed` cycle
    closes (no orphan transitions).
  - Phase parity matches across loop iterations
    (start_parity carried correctly; phantom-round arrives present).

Both validators run as `assert!` at the exit of their building fn —
invalid tape never escapes; consumer fns can assume validity.

**Kill criterion**: if either validator finds an invariant the
typestate-builder *should* have caught at compile time, push the
invariant up into the type system; don't keep it as runtime check.

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

5. **Implement `lower_dag_to_tape(&SubtileIR) -> SubtileTape`.**
   Single deterministic fn: validate → assign workers → topo → emit.
   Cross-worker `Operand::Sub` → `Signal`/`Wait`; gmem-crossing →
   `Fence`; runtime-bounded loop only for `AttnDecode`'s KV sweep.
   Returns a `TapeBuilder<EndState>` that gates the call to `validate`.
   Unit tests: chain, fork, join, attn, runtime loop.

5b. **`validate_subtile_tape`** — full impl + tests for the four
    error classes above. Runs at `lower_dag_to_tape`'s exit.

6. **Implement `lower_tape_to_tk(&SubtileTape) -> TkTape`.** Fixed
   table-driven translation. `SoftmaxState<Phase>` typestate enforced
   (Sv-before-Qkt = no impl). `KvCacheProducer` consumed via
   exhaustive match. `RopeForm` const-generic threaded through.

6b. **`validate_tk_tape`** — full impl + tests. Page-cycle closure,
    phase parity, `LoadAsync`↔`PageBarrierWait{Ready}` matching.

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

- Step 6's `lower_tape_to_tk` for `AttnDecode` cannot be expressed as
  a fixed instruction sequence and needs a backend-aware analysis
  pass (e.g. peeking `WarpRole` to choose barriers). SubtileTape is
  missing a primitive — re-design, not patch.
- Step 8's `play` requires any arm > 5 lines OR any conditional
  beyond `match` on instr kind. Tape was wrong.
- Step 9's Paris-decode bug does not surface as a *compile error*
  in the new typed-witness pipeline. Witnesses placed wrong.
- Step 10's surface area has not net-shrunk by ≥1500 LOC vs
  `5206d10d49`. This was a rename party, not a redesign.
- Either validator (5b, 6b) finds a class of error the typestate
  builder *should* have caught at compile time. Push the invariant
  up; don't ship it as runtime check.

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
| `subtile_tape.rs` | 0 LOC | new ~600 LOC |
| `lower.rs` (`LoweringInput`/`LoweredOp`) | 1105 LOC | DELETED |
| `tape.rs` | 391 LOC | DELETED |
| `tk_tape.rs` | 727 LOC | unchanged |
| `tk_player.rs` | 248 LOC | grows to ~600 LOC (full emit) |
| `tk_lower.rs` | 114 LOC | DELETED (witnesses fold into SubtileIR) |
| Validators (5b, 6b) | 0 LOC | new ~300 LOC each |
| **Total** | **~6500 LOC** | **~4400 LOC** (net -2100) |
