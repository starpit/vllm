# SubtileTape — what becomes a tape instruction

Status: APPROVED. Side-file for [`SUBTILE_IR_REDESIGN.md`](SUBTILE_IR_REDESIGN.md).
Source of truth for the (very small) instruction set of `subtile_tape::Instr`.

`SubtileTape` is a **topological linearization of the SubtileIR DAG**
under sequential semantics — **with every dataflow edge explicit as a
slot-lifecycle instruction**. Five instruction kinds:

| Kind | Constraint encoded | Source in SubtileIR |
|---|---|---|
| `AllocSlot { slot }` | "Mint a fresh logical arena cell for one producer's output." | One per `SubtileNode` (each op-output is a distinct slot). |
| `Compute { node, writes, reads }` | "Compute this node, writing into `writes` and reading from `reads`." Single-writer (one Compute per slot); multi-reader (any node whose predecessor list mentions this slot's writer). | One per `SubtileNode`, in ascending `SubtileId` order. `writes` is the slot allocated for `node`'s output; `reads` is the slot list of `node`'s region-overlap predecessors. |
| `FreeSlot { slot }` | "Last consumer is done; slot returns to the pool." | Emitted exactly once per slot, after its last reader's `Compute`. |
| `OpenLoop { var, bound }` / `CloseLoop { var }` | "The bracketed Computes execute `bound` times in sequence." | One pair per `SubOp::AttnDecode` (the KV-sweep over `seq_len` pages — runtime-bounded). No nested loops. |

Everything else is **explicitly out of scope** at this layer.

## Slot lifecycle (the hazard model)

```text
SlotHandle (allocated, unwritten) ──compute_to──▶ SlotWritten (written, readers OK)
                                                    │
                                              free_slot consumes
                                                    ▼
                                                  freed (slot id back in pool)
```

- `SlotHandle` is move-only (non-`Copy`, non-`Clone`). The Compute that
  writes to a slot **consumes** the `SlotHandle`. **Cannot write the
  same slot twice** at compile time.
- `SlotWritten` is move-only. Readers borrow it (`&SlotWritten` —
  multi-read OK). `free_slot` consumes the `SlotWritten`. **Cannot read
  after free** at compile time.
- A read takes `&SlotWritten` — **cannot read a slot before it was
  written** at compile time.
- `SlotId` / `SlotHandle` / `SlotWritten` sealed — no struct-literal
  construction; only path is via the builder.

**Why slots, not Signal/Wait/HazardId?** Slot count, slot lifetime, and
slot-write proof are target-agnostic — a function of the IR DAG's
liveness analysis, not of the target. Only **slot-physical-realization**
(smem capacity per slot, gmem fallback, the sync primitive that
discharges the hazard, page lifecycle, parity) is target-specific. So
slots belong at SubtileTape; their realization at TkTape.

## Source identifiers

SubtileTape `Instr` references node identity by `subtile_ir::SubtileId`
and slot identity by tape-local `SlotId` only. There is no second
buffer namespace (no `BufId`); the `metal_tape::BufId` namespace is a
v1 carcass surviving only in the §10-deletion files. The TkTape
lowering (commit 6) similarly preserves `subtile_ir::TensorId` for
source identifiers — one identifier per layer, no aliasing tables.

## What does NOT live here

- **Workers / CTAs / threadgroups / warp roles.** SubtileTape carries
  no parallelism abstraction. The DAG parallelism is in the SubtileIR
  (region-overlap predecessors); how to spread it across the target's
  execution units is a per-target lowering decision.
- **Memory class** (smem / gmem / smem-carry-forward). A slot is a
  *logical* arena cell; whether it lives in smem or gmem is a TkTape
  optimizer-pass decision.
- **Fences.** Visibility primitives are target primitives. The slot
  ordering (`AllocSlot` ≺ `Compute` ≺ `FreeSlot`) is the abstract
  hazard; the per-target lowering picks the realization (mbar arrive /
  wait, threadfence, NCCL, etc.).
- **Pages / page lifecycle / parity.** TkTape concepts.
- **Witnesses on instrs** (`KvCacheLayout`, `KvCacheProducer`,
  `RopeForm`, `SoftmaxState<Phase>`). They live on the SubtileIR
  `SubOp` variants; the lowering looks them up by `SubtileId`. The
  tape just names node identity.

## Sealed handles

- `SlotId`, `LoopVarId`, `RuntimeBoundId` — sealed value types,
  constructable only via the builder. So `Compute { writes, reads }`
  and `FreeSlot { slot }` cannot point at fabricated ids.
- `SlotHandle`, `SlotWritten` — sealed *move-only* token types. The
  builder produces them; the typestate consumes them; external code
  cannot fabricate one.

## TapeBuilder<S> typestate (compile-time)

State `S ∈ { Outside, InsideLoop }`. Misuse = no matching impl,
compile-fail.

| Method | `Outside` | `InsideLoop` |
|---|:-:|:-:|
| `alloc_slot` | ✓ → returns `SlotHandle` | ✗ (hazard primitive — slot lifetime is loop-iteration-invariant) |
| `compute_to` | ✓ → consumes `SlotHandle`, returns `SlotWritten` | ✓ (AttnDecode body lives inside the KV-sweep loop) |
| `free_slot` | ✓ → consumes `SlotWritten` | ✗ (hazard primitive; pairs with `alloc_slot` outside the loop) |
| `alloc_runtime_bound` | ✓ | ✗ |
| `open_loop` | ✓ → returns `TapeBuilder<InsideLoop>` + fresh `LoopVarId` | ✗ (no nesting) |
| `close_loop` | ✗ | ✓ → returns `TapeBuilder<Outside>` |
| `finish` | ✓ → returns `SubtileTape` | ✗ (must close the loop first) |

**Compile-fail tests:**

- Constructing `LoopVarId` / `SlotId` outside `TapeBuilder` (sealed types, no public constructor).
- Calling `close_loop` on `TapeBuilder<Outside>` (no impl).
- Calling `finish` on `TapeBuilder<InsideLoop>` (no impl).
- Calling `alloc_slot` on `TapeBuilder<InsideLoop>` (no impl).
- Reading a slot before its writer (`&SlotWritten` doesn't exist yet — borrow type error).
- Writing the same slot twice (the second `compute_to` has no `SlotHandle` to consume).
- Reading after free (the `free_slot` consumed the `SlotWritten`).

## Runtime validator

`validate_subtile_tape(&tape, &graph) -> Result<(), Vec<ValidationError>>`
runs three relational (tape, SubtileIR) checks. Per audit BLOCKER fix
`wewpteccb` (K5 + `feedback_compile_time_or_garbage`): slot lifecycle,
loop balance, and slot id range were removed from the runtime
validator because they are sealed at compile time by `TapeBuilder<S>`
typestate plus the now-private `SubtileTape::instrs` field — there is
no construction path that bypasses the typestate, so the runtime
checks would be unreachable.

- **Compute well-formedness** — every SubtileIR node is `Compute`'d
  exactly once (`MissingCompute` / `DuplicateCompute`); no `Compute`
  references an out-of-range node id (`UnknownNode`); adjacent
  Computes appear in strictly ascending `SubtileId` order
  (`TopoOrderViolation`).
- **Edge coverage** — for every `Compute { node, reads }`, the
  set-of-writers of `reads` equals the SubtileIR predecessor set of
  `node` (`EdgeMismatch`). This is the load-bearing check that "every
  DAG edge is an explicit instruction."

The `ValidationError` enum carries exactly five variants:
`UnknownNode`, `MissingCompute`, `DuplicateCompute`,
`TopoOrderViolation`, `EdgeMismatch`.

## Compile-time guarantees (TapeBuilder typestate)

The following invariants are sealed at the type level — there is no
runtime check, and there cannot be one, because constructing a
violation is a Rust type error:

- **Slot lifecycle** — `SlotHandle` (Allocated, unwritten) is
  move-only and consumed by `compute_to`; `SlotWritten` (written,
  multi-reader) is move-only and consumed by `free_slot`. Single-
  writer / read-before-write / double-write / use-after-free /
  double-free / write-unallocated-slot become use-after-move
  compile errors. See compile-fail doctests on `TapeBuilder`.
- **Loop balance** — `TapeBuilder<state::Outside>` / `<state::InsideLoop>`
  typestate makes nesting unrepresentable, `close_loop` callable
  only inside a loop, `finish` callable only outside, and the
  `LoopVarId` carried by `state::InsideLoop::Loop` makes
  `MismatchedLoopVar` impossible.
- **Slot id range** — `TapeBuilder::alloc_slot` mints sequential
  ids and `num_slots = next_slot` at `finish`, so every minted
  `SlotId` is by construction in `0..num_slots`. `SlotId` is sealed
  (only constructable via `alloc_slot`).

## Design principles

- **The tape is a tape.** No side-tables, no parallel arrays, no
  per-edge metadata, no consumer-count caches. The slot lifecycle IS
  the per-edge metadata, encoded as instructions in the tape itself.
- **Hazards explicit.** Every SubtileIR DAG dataflow edge surfaces in
  the tape as an instruction (the `Compute`'s `reads` list, the
  `FreeSlot` after the last reader). Validator catches misses.
- **Per-target decisions live at per-target lowerings.** Whether a
  slot lives in smem or gmem, whether the cross-CTA coordination uses
  a gmem flag handshake, an inter-CTA mbarrier, a grid sync, or static
  work-slicing-per-CTA-with-no-coordination — that's a TkTape (or
  MetalTape) lowering decision, not a SubtileTape concept.
- **Coarse loops, not many small ones.** `OpenLoop`/`CloseLoop`
  exist for AttnDecode's KV-sweep. Static N-block fan-out is N
  separate `Compute` instructions, not a runtime loop. Reaching for
  a loop where parallelism could exist is a policy bug.
- **Slot lifecycle outside the loop bracket.** AttnDecode's slot
  `AllocSlot` lives Outside the OpenLoop, and `FreeSlot` lives Outside
  the CloseLoop; only the `Compute` itself lives inside the bracket.
  This is the hazard model — slot lifetime spans iterations, the
  workload spans one iteration.

## Open questions

- **`LoopBound` shape.** Decode `seq_len` is a runtime quantity;
  TkTape models this with `LoopCount::KernelArg`. SubtileTape stays
  target-agnostic with an opaque `LoopBound::Runtime(RuntimeBoundId)`
  vs `LoopBound::Const(u32)`; the binding to a real kernel-arg slot
  happens at the per-target lowering.

## Where this sits in the redesign

- Plan: [`SUBTILE_IR_REDESIGN.md`](SUBTILE_IR_REDESIGN.md) §3
  (validator stretch) and §4 commit 5 (`lower_dag_to_tape`).
- Code: `vllm-rs/crates/ferrite-wavefront/src/subtile_tape.rs`.
- Memory pointer: `memory/project_subtile_tape_constraints.md`.

If the per-target lowering needs a constraint not surfaced by the
slot-lifecycle + loop-bracket instruction set, **add it as a TkTape
(or MetalTape) primitive at that lowering**, not as a new SubtileTape
Instr. SubtileTape is the target-agnostic floor — adding parallelism /
sync / memory primitives here re-introduces the v1 mistake.
