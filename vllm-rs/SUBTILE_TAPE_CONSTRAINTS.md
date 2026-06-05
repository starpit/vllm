# SubtileTape — DAG constraints that must become instructions

Status: APPROVED. Side-file for `SUBTILE_IR_REDESIGN.md` §3 + §4
commit 3. Source of truth for what `subtile_tape::Instr` must encode.

The job of `SubtileTape` is to **manifest every SubtileIR DAG
constraint as an explicit, target-agnostic instruction** so the
TkTape / MetalTape lowerings have a flat list to walk and the
runtime validator (`validate_subtile_tape`) has something to check.
Anything left implicit at this layer is a class of bug the validator
cannot see.

## 1. Constraint inventory

Six classes of constraint live in (or follow from) the SubtileIR.
Each row names how it surfaces as a tape instruction.

| # | Constraint | Source in SubtileIR | Tape instruction |
|---|---|---|---|
| 1 | **Compute slot per node** — which worker computes which node | `SubtileNode::id` + scheduler's `worker_of[id]` | `Compute { worker, node }` |
| 2 | **Producer→consumer dataflow edge** — region overlap on op-output tensors | `predecessors(graph)` (region overlap) | Same-worker → implicit by per-worker program order, NO instruction. Cross-worker → `Signal { worker_p, barrier }` after producer + `Wait { worker_c, barrier }` before consumer. |
| 3 | **Memory routing (coloring) per producer output** — Internal (shmem carry-forward) vs External (gmem drain) | `routing::classify_outputs` (already computes this from the LoweringInput DAG) | `Route { node, class: Shmem \| Gmem }` |
| 4 | **Memory hazard at a gmem-crossing edge** — write must be globally visible before remote read | derived: cross-worker edge whose producer `Route.class == Gmem` | `Fence { worker }` between producer's `Signal` and consumer's `Wait`, on producer's worker |
| 5 | **Runtime-bounded loop** — AttnDecode KV-sweep over `seq_len` pages (a runtime quantity at decode) | implicit in `SubOp::AttnDecode` semantics | `OpenLoop { worker, var, bound }` / `CloseLoop { worker, var }` bracketing the body Computes + Fences. No nested loops; no cross-worker sync inside (would reorder vs the iteration count) |
| 6 | **Per-worker linear order** — each worker's prefix is a topological order of its assigned nodes | scheduler invariant + `id` ordering | implicit: the tape is one linear stream, workers tagged on each instr; the per-worker subset is in program order |

## 2. Sealed handles — Wait/Signal/Loop matched by construction

- `WorkerId`, `BarrierId`, `LoopVarId` are sealed value types,
  constructable **only** via `TapeBuilder` allocators
  (`alloc_barrier`, `open_loop`).
- Therefore `Wait(b)` and `Signal(b)` cannot point at fabricated
  ids; the only `BarrierId`s in existence are well-formed.
- "Mismatched" at the value level (signal on B1, wait on B2 by
  mistake) is a runtime check (`validate::orphan_signal_wait`).

## 3. TapeBuilder<S> typestate (compile-time)

State `S ∈ { Outside, InsideLoop }`. Misuse = no matching impl,
compile-fail.

| Method | `Outside` | `InsideLoop` |
|---|:-:|:-:|
| `alloc_barrier` | ✓ | ✗ (cross-worker sync inside a loop reorders vs iteration count) |
| `signal` / `wait` | ✓ | ✗ (same reason) |
| `compute` | ✓ | ✓ |
| `fence` | ✓ | ✓ |
| `route` | ✓ | ✓ |
| `open_loop` | ✓ → returns `TapeBuilder<InsideLoop>` + fresh `LoopVarId` | ✗ (no nesting) |
| `close_loop` | ✗ | ✓ → returns `TapeBuilder<Outside>` |
| `finish` | ✓ → returns `SubtileTape` | ✗ (must close the loop first) |

**Compile-fail tests (commit 3 ships):**
- Constructing `BarrierId` outside `TapeBuilder` (sealed type, no public constructor).
- Calling `close_loop` on `TapeBuilder<Outside>` (no impl).
- Calling `finish` on `TapeBuilder<InsideLoop>` (no impl).
- Calling `signal` / `wait` on `TapeBuilder<InsideLoop>` (no impl).

## 4. What the runtime validator infers from this set

`validate_subtile_tape(&tape, &graph) -> Result<(), Vec<ValidationError>>`
runs four checks (plan §3.2). Each one is derivable from the
instructions above:

- **Orphan signal/wait** — count `Signal` / `Wait` per `BarrierId`. Every barrier needs ≥1 of each.
- **Deadlock cycle** — build a worker × barrier dependency digraph (worker W waits on barrier B → producer of B is on worker W'; W blocks on W'). Cycle ⇒ deadlock.
- **Missing fence** — for every cross-worker `(Signal, Wait)` pair whose producer's `Route.class == Gmem`, there must be a `Fence` between them on the producer's worker.
- **Data race** — for each `Compute(node)`, look up `node`'s `inputs`/`output` regions in the SubtileIR. Build per-tensor `(reader_workers, writer_workers, latest_fence_index)`. Reader without a fence after another worker's latest write of an overlapping region ⇒ race.

## 5. What does NOT become a tape instruction

These live elsewhere by design — putting them in tape instructions
would either duplicate IR-level facts or pollute the target-agnostic
layer with backend choices.

- **Physical arena slot / `BufId`** — TK / Metal lowering decides; the tape names tensors transitively via `Compute(node)` → SubtileIR's `TensorRegion`s.
- **Op-specific tile shapes / fields** (head_dim, eps, scale, num_q_heads) — live on `SubOp` variants in SubtileIR; the tape just references node identity.
- **Typed witnesses** (`KvCacheLayout`, `KvCacheProducer`, `RopeForm`, `SoftmaxState<Phase>`, `Phase` parity) — fields on `SubtileIR::SubtileNode` (per redesign §2 table) and TkTape's `ComputeBody` template, propagated through the lowering, never invented at the tape layer.
- **Per-op kernel-arg bindings** (cos/sin row, seq_len, block_table) — TkTape's `KernelArgRef` machinery; the abstract `bound: LoopBound` on SubtileTape's `OpenLoop` references one but doesn't resolve it.

## 6. Design principles (policy defaults)

These are the choices the lowering walker (commit 5) **must** make
unless a specific constraint forces otherwise. Drift from these
silently re-introduces the perf regressions this redesign avoids.

- **Keep activations in shared memory whenever possible** — the
  whole reason `Route` exists is to express the Internal/External
  choice from `routing.rs`. The walker's default for every producer
  output is `Route.class = Shmem` (carry-forward via mbar handshake);
  only emit `Route.class = Gmem` when the routing analysis proves
  the producer output is observable post-kernel (the canonical's
  `result`), or when keeping it in shmem would violate a lifetime
  / multi-consumer constraint that `routing::classify_outputs`
  flags as `External`. Gmem drain is the **fallback**, not the
  default. Phase 12 `OutputRouting::Internal` is the live signal;
  do not ignore it. Every gmem drain on a path that could have been
  shmem-carried is a bandwidth round-trip the megakernel pays.
- **Prefer same-worker (chain-local) placement** — slice-index
  scheduling (`region_schedule::assign_owners_slice_index`) already
  co-locates `q→rope→attn→o-partial` and `gate/up→silu→down-partial`
  chains on one worker. The lowering walker emits `Wait`/`Signal`
  only for surviving cross-worker edges (genuine joins:
  split-K all-reduces, the cheap shared K/V); any new chain it
  introduces should aim for chain-locality first.
- **Coarse loops, not many small ones** — `OpenLoop`/`CloseLoop`
  exist for AttnDecode's KV-sweep, not for general "loop over
  N-blocks". Static N-block fan-out becomes N parallel `Compute`
  instructions on different workers, NOT a runtime loop on one
  worker. Reaching for a loop where parallelism could exist is a
  policy bug.

## 7. Open questions

These are flagged here so they don't get answered silently inside
implementation:

- **(a) `Route` granularity.** Per-producer-output (one `Route` per
  Compute) vs per-edge (one `Route` per consumer). Decision:
  **per-producer-output** — the producer's output buffer is a
  single piece of memory; all consumers see the same class.
  Multi-class fan-out (one consumer wants shmem, another wants
  gmem) doesn't arise in this codebase and would need a copy node
  in the SubtileIR if it ever did.
- **(b) Intra-worker RAW barrier through gmem.** `mega.rs` currently
  emits opcode `BARRIER = 3` for an intra-worker `threadgroup_barrier`
  at a same-worker RAW where data went through gmem. Plan §3.2 only
  calls out the **cross-worker** gmem fence. **Decision for
  commit 3:** do not add an intra-worker `Sync`/`Barrier` instr
  yet; surface it later if the lowering walker actually emits one.
  Until then, intra-worker RAW relies on per-worker program order +
  the fact that same-worker writes-then-reads through registers /
  shmem don't need a fence under the abstract model.
- **(c) `LoopBound` shape.** Decode `seq_len` is a runtime quantity;
  TkTape models this with `LoopCount::KernelArg`. SubtileTape stays
  target-agnostic with an opaque `LoopBound::Runtime(RuntimeBoundId)`
  vs `LoopBound::Const(u32)`; the binding to a real kernel-arg slot
  happens at the TkTape lowering. Commit 3 ships the enum even if
  only `Const` is used until commit 5 wires AttnDecode.

## 8. Where this sits in the redesign

- Plan: `vllm-rs/SUBTILE_IR_REDESIGN.md` §3 (validator stretch) and
  §4 commit 3 (add `subtile_tape.rs`, additive, no consumers).
- Code: `vllm-rs/crates/ferrite-wavefront/src/subtile_tape.rs` (commit 3
  onward).
- Memory pointer: `memory/project_subtile_tape_constraints.md`.

If the lowering walker (commit 5) needs a constraint not in §1's
table, **add it here first**, not as a one-off in `lower_dag_to_tape`.
A constraint that doesn't surface as a tape instruction is one the
validator cannot see — exactly the class of silent-runtime-divergence
bug this redesign exists to prevent.
