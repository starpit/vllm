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

Four classes of constraint live in (or follow from) the SubtileIR and
surface at SubtileTape. **Memory routing (shmem vs gmem) and memory
hazards (fences) do NOT appear here** — they're target-specific and
live at TkTape, where a pipeline of optimizer passes decides them
(see [`SUBTILE_IR_REDESIGN.md`](SUBTILE_IR_REDESIGN.md) §4 commit 6.5).
Each row names how the constraint surfaces as a tape instruction.

| # | Constraint | Source in SubtileIR | Tape instruction |
|---|---|---|---|
| 1 | **Compute slot per node** — which worker computes which node | `SubtileNode::id` + scheduler's `worker_of[id]` | `Compute { worker, node }` |
| 2 | **Producer→consumer dataflow edge** — region overlap on op-output tensors | `predecessors(graph)` (region overlap) | Same-worker → implicit by per-worker program order, NO instruction. Cross-worker → `Signal { worker_p, barrier }` after producer + `Wait { worker_c, barrier }` before consumer. |
| 3 | **Runtime-bounded loop** — AttnDecode KV-sweep over `seq_len` pages (a runtime quantity at decode) | implicit in `SubOp::AttnDecode` semantics | `OpenLoop { worker, var, bound }` / `CloseLoop { worker, var }` bracketing body Computes. No nested loops; no cross-worker sync inside (would reorder vs the iteration count) |
| 4 | **Per-worker linear order** — each worker's prefix is a topological order of its assigned nodes | scheduler invariant + `id` ordering | implicit: the tape is one linear stream, workers tagged on each instr; the per-worker subset is in program order |

**Hazards at this layer.** SubtileTape DOES express memory hazards
— it just doesn't name the visibility primitive (fence vs mbar handshake
vs threadfence-block — those are TK 2.0 primitives that live at TkTape).
Three kinds, three places:

- **Cross-worker RAW / WAR / WAW** — surfaced as `Signal`/`Wait` pairs
  (row 2). The hazard *kind* is recoverable from SubtileIR region
  overlap (producer's `output` region vs consumer's `inputs` regions);
  the Signal/Wait pair encodes "Wp's writes happen-before Wc's reads"
  abstractly, without committing to a primitive.
- **Intra-worker** — implicit in per-worker program order. SubtileIR
  guarantees acyclic ascending-id; the per-worker subset is a
  topological order; abstract happens-before holds without an
  instruction.
- **Pure ordering with no data dependency** — does not arise today
  (Signal/Wait is emitted only where SubtileIR has a region-overlap
  edge). If a use case appears (resource lifetime, side-effect
  ordering) it gets a new SubtileTape Instr — never a fence.

`validate_subtile_tape`'s **cross-worker data race** check (§4) is
exactly: every region-overlap cross-worker write→read in the SubtileIR
has a matching `Signal`/`Wait` pair in the tape. That IS the hazard
check at this layer.

**What surfaces only at TkTape (target-specific):**

- **Memory class per producer output** (shmem carry-forward vs gmem
  drain). Decided by the `promote_shmem_carry_forward` optimizer pass
  with target knowledge SubtileTape doesn't have: smem capacity, mbar
  slot count, NUM_CONSUMER_WARPS, page-lifetime windows, parity
  allocator state.
- **Visibility primitive selection** (`__threadfence_block` /
  `_device` / `_system` vs cross-IType mbar handshake). TK 2.0 has all
  of these; lowering picks the conservative one (`FenceDevice` on
  every Gmem-routed cross-worker edge), and the `narrow_fence_scope` /
  `eliminate_dead_fences` passes prune. **Every surviving Gmem edge
  always has a fence guarding it** — `validate_tk_tape`'s
  fence-before-arrive check enforces this on the materialized tape.
- **Intra-worker RAW through gmem**. Materializes only when the
  TkTape rewrites route same-worker data through a gmem buffer (page
  coalescing, register pressure spills). A post-rewrite invariant
  check on TkTape, not a SubtileTape primitive.

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
| `open_loop` | ✓ → returns `TapeBuilder<InsideLoop>` + fresh `LoopVarId` | ✗ (no nesting) |
| `close_loop` | ✗ | ✓ → returns `TapeBuilder<Outside>` |
| `finish` | ✓ → returns `SubtileTape` | ✗ (must close the loop first) |

(`fence` and `route` builders are gone — SubtileTape has no fence /
route concept. See commit 3.b in `SUBTILE_IR_REDESIGN.md` §4.)

**Compile-fail tests (commit 3 ships):**
- Constructing `BarrierId` outside `TapeBuilder` (sealed type, no public constructor).
- Calling `close_loop` on `TapeBuilder<Outside>` (no impl).
- Calling `finish` on `TapeBuilder<InsideLoop>` (no impl).
- Calling `signal` / `wait` on `TapeBuilder<InsideLoop>` (no impl).

## 4. What the runtime validator infers from this set

`validate_subtile_tape(&tape, &graph) -> Result<(), Vec<ValidationError>>`
runs target-agnostic checks only. Fence-correctness is a TkTape
concern (`validate_tk_tape`'s **fence-before-arrive** check); see
[`SUBTILE_IR_REDESIGN.md`](SUBTILE_IR_REDESIGN.md) §3.2.

- **Orphan signal/wait** — count `Signal` / `Wait` per `BarrierId`. Every barrier needs ≥1 of each. One-shot signals fire at most once.
- **Deadlock cycle** — build a worker × barrier dependency digraph (worker W waits on barrier B → producer of B is on worker W'; W blocks on W'). Cycle ⇒ deadlock.
- **Cross-worker data race** — for each `Compute(node)`, look up `node`'s `inputs`/`output` regions in the SubtileIR. A read on worker `Wc` of a region written by `Wp ≠ Wc` must be preceded by a `Wait` whose matching `Signal` is after the write. (No fence required at this layer; the visibility primitive is whatever the TkTape lowering picks.)
- **Compute well-formedness** — every `Compute` names an in-range `SubtileId`; the per-worker subset is a topological order of that worker's assigned nodes.
- **Loop balance** — every `OpenLoop` has a matching `CloseLoop` on the same worker with the same `LoopVarId`; no cross-worker sync inside.

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

The "keep activations in shared memory" policy is **not a SubtileTape
rule** — it's a TkTape pass policy (`promote_shmem_carry_forward` in
`SUBTILE_IR_REDESIGN.md` §4 commit 6.5). SubtileTape neither names
shmem nor decides routing; the rules below are the placement /
loop-shape decisions the SubtileTape walker still owns.

- **Prefer same-worker (chain-local) placement** — slice-index
  scheduling (`region_schedule::assign_owners_slice_index`) already
  co-locates `q→rope→attn→o-partial` and `gate/up→silu→down-partial`
  chains on one worker. The lowering walker emits `Wait`/`Signal`
  only for surviving cross-worker edges (genuine joins:
  split-K all-reduces, the cheap shared K/V); any new chain it
  introduces should aim for chain-locality first. (This also
  enables the downstream shmem-carry-forward pass — cross-worker
  edges are cross-SM in a persistent megakernel, so the optimizer
  can only promote chain-local edges to shmem.)
- **Coarse loops, not many small ones** — `OpenLoop`/`CloseLoop`
  exist for AttnDecode's KV-sweep, not for general "loop over
  N-blocks". Static N-block fan-out becomes N parallel `Compute`
  instructions on different workers, NOT a runtime loop on one
  worker. Reaching for a loop where parallelism could exist is a
  policy bug.

## 7. Open questions

These are flagged here so they don't get answered silently inside
implementation:

- **`LoopBound` shape.** Decode `seq_len` is a runtime quantity;
  TkTape models this with `LoopCount::KernelArg`. SubtileTape stays
  target-agnostic with an opaque `LoopBound::Runtime(RuntimeBoundId)`
  vs `LoopBound::Const(u32)`; the binding to a real kernel-arg slot
  happens at the TkTape lowering. Commit 3 ships the enum even if
  only `Const` is used until commit 5 wires AttnDecode.

(Routing-class granularity and intra-worker RAW barriers were once
open questions here; both moved down to TkTape with the
conservative-then-optimize redesign — see
[`SUBTILE_IR_REDESIGN.md`](SUBTILE_IR_REDESIGN.md) §4 commits 3.b
+ 6.5.)

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
