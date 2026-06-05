# SubtileTape — what becomes a tape instruction

Status: APPROVED. Side-file for [`SUBTILE_IR_REDESIGN.md`](SUBTILE_IR_REDESIGN.md).
Source of truth for the (very small) instruction set of `subtile_tape::Instr`.

`SubtileTape` is a **topological linearization of the SubtileIR DAG**
under sequential semantics. Two instruction kinds, plus loop brackets:

| Kind | Constraint encoded | Source in SubtileIR |
|---|---|---|
| `Compute { node }` | "Compute this node next." | One per `SubtileNode`, in ascending `SubtileId` order. |
| `OpenLoop { var, bound }` / `CloseLoop { var }` | "The bracketed Computes execute `bound` times in sequence." | One pair per `SubOp::AttnDecode` (the KV-sweep over `seq_len` pages — runtime-bounded). No nested loops. |

Everything else is **explicitly out of scope** at this layer.

## What does NOT live here

- **Workers / CTAs / threadgroups / warp roles.** SubtileTape carries
  no parallelism abstraction. The DAG parallelism is in the SubtileIR
  (region-overlap predecessors); how to spread it across the target's
  execution units is a per-target lowering decision.
- **Sync between parallel units** (Signal / Wait / Barrier / Flag).
  Not present. Whatever cross-unit coordination a target needs gets
  emitted at that target's lowering.
- **Memory hazards / fences.** Not present. Visibility primitives are
  target primitives.
- **Memory class** (shmem / gmem / smem-carry-forward). Not present.
- **Pages / page lifecycle / parity.** TkTape concepts.
- **Witnesses on instrs** (`KvCacheLayout`, `KvCacheProducer`,
  `RopeForm`, `SoftmaxState<Phase>`). They live on the SubtileIR
  `SubOp` variants; the lowering looks them up by `SubtileId`. The
  tape just names node identity.

## Sealed handles

- `LoopVarId`, `RuntimeBoundId` — sealed value types, constructable
  only via `TapeBuilder::open_loop` / `alloc_runtime_bound`. So
  `OpenLoop(v)` and `CloseLoop(v)` cannot point at fabricated ids;
  the only `LoopVarId`s in existence are well-formed.

## TapeBuilder<S> typestate (compile-time)

State `S ∈ { Outside, InsideLoop }`. Misuse = no matching impl,
compile-fail.

| Method | `Outside` | `InsideLoop` |
|---|:-:|:-:|
| `compute` | ✓ | ✓ |
| `alloc_runtime_bound` | ✓ | ✗ |
| `open_loop` | ✓ → returns `TapeBuilder<InsideLoop>` + fresh `LoopVarId` | ✗ (no nesting) |
| `close_loop` | ✗ | ✓ → returns `TapeBuilder<Outside>` |
| `finish` | ✓ → returns `SubtileTape` | ✗ (must close the loop first) |

**Compile-fail tests:**

- Constructing `LoopVarId` outside `TapeBuilder` (sealed type, no public constructor).
- Calling `close_loop` on `TapeBuilder<Outside>` (no impl).
- Calling `finish` on `TapeBuilder<InsideLoop>` (no impl).

## Runtime validator

`validate_subtile_tape(&tape, &graph) -> Result<(), Vec<ValidationError>>`
runs three checks. Each is derivable from the instruction stream + the
SubtileIR alone:

- **Compute well-formedness** — every SubtileIR node is `Compute`'d
  exactly once (`MissingCompute` / `DuplicateCompute`); no `Compute`
  references an out-of-range node id (`UnknownNode`); adjacent
  Computes appear in strictly ascending `SubtileId` order
  (`TopoOrderViolation`). The SubtileIR is itself ascending-id-topo,
  so a valid linearization stays ascending.
- **Loop balance** — every `OpenLoop` matches a `CloseLoop` with the
  same `LoopVarId`; no nesting (`NestedLoop`); no unclosed loops at
  end-of-tape (`UnclosedLoop`); no `CloseLoop` without a matching
  `OpenLoop` (`UnmatchedCloseLoop`); `OpenLoop` and `CloseLoop` agree
  on `var` (`MismatchedLoopVar`).

The TapeBuilder typestate prevents most of these at compile time; the
runtime validator defends against hand-built tapes that mutate
`SubtileTape::instrs` directly.

## Design principles

- **The tape is a tape.** No side-tables, no parallel arrays, no
  per-edge metadata, no consumer-count caches. If a downstream pass
  needs a fact, it derives it from the SubtileIR + tape on the fly,
  and caches in pass-local state — never on the IR.
- **Per-target decisions live at per-target lowerings.** Whether
  cross-CTA coordination uses a gmem flag handshake, an inter-CTA
  mbarrier, a grid sync, or static work-slicing-per-CTA-with-no-
  coordination — that's a TkTape (or MetalTape) lowering decision,
  not a SubtileTape concept.
- **Coarse loops, not many small ones.** `OpenLoop`/`CloseLoop`
  exist for AttnDecode's KV-sweep. Static N-block fan-out is N
  separate `Compute` instructions, not a runtime loop. Reaching for
  a loop where parallelism could exist is a policy bug.

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

If the per-target lowering needs a constraint not surfaced by this
two-instruction set, **add it as a TkTape (or MetalTape) primitive at
that lowering**, not as a new SubtileTape Instr. SubtileTape is the
target-agnostic floor — adding parallelism / sync / memory primitives
here re-introduces the v1 mistake.
