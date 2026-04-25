# ff-interpreter — handoff

> Read this first. This branch (`ff-interpreter`) is a **pivot off
> `ferrite-forward`** at commit `961188f5c`. ff4 (the stencil
> rewrite) is dead. The existing `HANDOFF.md` next to this file
> is the inherited ferrite-forward handoff; keep it for context,
> but the work happening *now* is what's described below.

## The actual work — one paragraph, no fucking around

ferrite-forward already codegens the entire forward pass for every
(variant × workload-point): solver picks Impls, emit_workload walks
them, each Impl's `emit_call` produces a `let tile_X = kernel_call(…)`
Rust statement, and the macro stitches them into a fully-unrolled
forward fn. **Replace that codegen wholesale.** Same solver. Same
Impl set. Same kernel calls. The macro emits, instead of an
unrolled Rust body:

1. **One instruction list per (variant × workload-point)** — a
   `&'static [Instruction]` const where each row is one
   kernel-call-shaped opcode + its packed arguments. Same
   granularity as today's per-workload forward fn. (Open
   question, see below: could be elevated to one list per
   variant, with workload-point selecting a subrange at
   runtime — but that's a future refactor; start with what
   matches today's granularity.)
2. **One interpreter per arch** — a single `match instr.opcode()
   { … }` block emitted by each `#[forward] fn <arch>()` macro
   invocation. Each arm decodes fields off the Instruction and
   calls the same `ferrite_kernels` function the inlined path
   would have called.

Solver, Impl library, library invariants, fingerprinting, weights
loading, dispatcher — all unchanged. Only the *backend* of the
macro changes: it stops emitting Rust let-bindings and starts
emitting `Instruction` rows + a `match` arm per opcode. The same
IR will later drive an on-GPU megakernel backend; the host
interpreter is the proof.

The opcode vocabulary is taken from KVM tp_throughput
(`~/Megakernels/megakernels/demos/tp_throughput/`) — that demo
runs both prefill and decode through one persistent grid using
this exact opcode set.

## Failure pattern this handoff explicitly rejects

> "Migrate one Impl as proof-of-concept. Then the next Impl. Then
> the next."

That is **the wrong shape**. There is no PoC layer to prove. The
existing codegen already proves every kernel call works. The work
is one wholesale change to the macro: every Impl that
participates in a real arch's forward gets its emission shape
changed at the same time, in the same commit's worth of work.
Half-migrated trait defaults that compile_error! at codegen are
*temporary* — they exist so the new methods can land before the
refactor is done, not so we ship a half-migrated tree.

## Locked decisions

1. **Wire format**: `repr(transparent) Instruction([i32; 32])`.
   Slot 0 is opcode (low 16 bits); 31 i32 payload slots,
   opcode-defined layout. Matches KVM tp_throughput.
2. **Opcodes**: first vocabulary is the llama-family set from
   tp_throughput — `ATTN_NORM=1, QKV_ROPE_APPEND=2,
   ATTENTION_PREFILL=3, ATTENTION_DECODE=4, O_PROJ_RESIDUAL=5,
   MLP_NORM=6, GATE_SILU=7, UP_MATMUL=8, DOWN_PROJ_RESIDUAL=9,
   LM_HEAD_NORM=10, LM_HEAD=11, INC_BARRIER=12, DIE=13,
   ALL_DEVICE_BARRIER=14, FREE=0xFF`. A non-llama arch (Mamba,
   MoE, MLA) adds opcodes its kernels need; opcodes are per-arch.
3. **One interpreter per arch.** Each `#[forward] fn <arch>()`
   emits its own `match`. Three reasons, not one: (a) the match
   only contains arms for opcodes this arch actually uses
   (smaller, branch-predictor-friendlier); (b) arm bodies need
   per-arch `Weights` field access — a global interpreter would
   require generics + a vtable to reach them, which is exactly
   the indirection bloat we're avoiding; (c) the same opcode
   often means different kernels per arch (Gemma's
   ATTN_NORM folds `+1.0`; Gemma3's QKV_ROPE picks
   `rotary_local` for some layers) — encoding arch into the
   opcode would explode the opcode space. Cross-arch sharing is
   a non-goal.
4. **One instruction list per (variant × workload-point)** —
   matches today's granularity. Elevating to one list per variant
   with the workload-point selecting a subrange (or
   parameterizing fields at runtime) is a real possibility but
   is **not in scope here**; raising it would tangle the
   workload-point dispatch into the IR and that is the kind of
   detour ff4 died on.
5. **Tile output table** (`TileEntry::{Owned, View}`) replaces
   today's per-tile `let` bindings. Slot index = tile id.
6. **No alloc opcode.** Tile output allocation is implicit in the
   compute opcode (the kernel writes back; the interpreter arm
   stores the resulting `OwnedTensor` into `tiles[dst_slot]`).
7. **No view opcode.** When a claim aliases an upstream tile
   (today's `as_view()` borrow), the macro emits
   `tiles[dst_slot] = Some(TileEntry::View { ref_slot: src });`
   inline at codegen time — alongside the const array, not as a
   runtime instruction. Megakernel doesn't need a runtime view
   op either; aliasing on GPU is just pointer reuse.
8. **`FREE` is the only memory-management opcode.** Macro's
   existing drop-pass already computes last-reader per tile;
   emit one `FREE { slot }` row at that point.

## What's landed in this worktree (uncommitted)

```
vllm-rs/crates/ferrite-forward/src/lib.rs                (modified — re-exports)
vllm-rs/crates/ferrite-forward/src/instruction.rs        (NEW, 331 lines + 6 tests)
vllm-rs/crates/ferrite-forward/src/tile_table.rs         (NEW, 140 lines, cuda-only)
vllm-rs/crates/ferrite-forward-macro/src/impl_lib.rs     (modified — trait + InstrEmit)
```

`cd vllm-rs && cargo test -p ferrite-forward -p ferrite-forward-macro`
passes against the in-flight state. Nothing committed yet.

## What's next — concretely

This is a single refactor, not a migration loop:

1. **Pick the new emission seam in `emit_workload.rs`.** Today
   it walks Impls and concatenates their `emit_call` outputs
   into a Rust body. The seam is: where today it pushes a Rust
   `let` statement, push instead (a) an `InstrEmit` for the
   const array, (b) a `match` arm for the per-arch interpreter
   (deduped by opcode).
2. **Rewrite every llama-family Impl's emission.** Each Impl
   that participates in a real arch's forward (residual-add,
   rms-norm, fused-add-rms-norm, qkv-rope, attention prefill,
   attention decode, o_proj+residual, gate+silu+mul, up matmul,
   down_proj+residual, lm_head_norm, lm_head) gets a `fan_out`
   that returns its `InstrEmit`s and an `interpreter_arm` that
   decodes fields and calls the same kernel its old `emit_call`
   called.
3. **Emit the per-arch interpreter `match`.** One per
   `#[forward] fn <arch>()`. Arms come from each Impl's
   `interpreter_arm`. Macro deduplicates by opcode (an opcode
   has exactly one arm).
4. **Wire the new path through the dispatcher.** The generated
   `forward` fn becomes: build/select the static instruction
   list for this (variant, workload-point), build the tile
   table, run the interpreter loop. No fallback to inlined
   path — this is the path.
5. **Run llama golden** (`vllm-e2e --features e2e,cuda
   --release --test e_correctness -- --ignored
   --test-threads=1`, llama subset). Match must be exact.

## Transitional flag — to delete on completion

A boolean `interpreter` arg on `#[forward(...)]` (default false)
selects the new instruction-list emission path. Arches migrate
one at a time by setting `interpreter = true`; old path stays
the default until every arch flips. **Final commit of this
refactor removes the flag and the old emit_call path.** This is
the "minor architectural decision easy to fix later" the user
authorized in the kickoff message.

While the flag exists, `emit_call` and `fan_out` coexist on the
trait. After flag removal, `emit_call` is deleted.

## Decisions locked in this session (2026-04-25)

- **Granularity**: One instruction list per (variant × workload-point).
  Matches today's `forward_m_<N>[_sk_<SK>]` granularity. Elevate to
  per-variant later when the megakernel backend lands.
- **Slot allocation**: Codegen builds a `SlotMap` per (variant ×
  workload-point) FUF. Slot index = `slot_map[(tile_id, output_slot)]`.
  Densely packed; total slot count is the size of the runtime
  `tiles: Vec<Option<TileEntry>>` table. fan_out gets `&SlotMap`
  to resolve tile references inside its `InstrEmit` field i32s.
- **InstrEmit field type**: stays `[i32; 31]`. fan_out resolves
  slot ids itself via the SlotMap. (Considered a symbolic field
  enum; rejected — the slot resolution is monotone, the i32-only
  shape keeps the wire format obvious, and codegen passes the
  SlotMap anyway.)
- **View prelude**: All `tiles[dst] = Some(View { ref_slot: src })`
  setups emitted *before* the interpreter loop, in the forward
  fn body. Reads through `tile_ref` panic if `ref_slot` is None,
  so the schedule's owner-before-reader invariant suffices —
  no mid-loop View setup needed.
- **Drop pass → FREE opcodes**: Every `drop(local)` the existing
  drop-pass would have emitted becomes a `FREE { slot }` row in
  the instruction list at the same scheduling point.
- **Per-arch interpreter match**: One match per `#[forward] fn
  <arch>()`. Built by collecting `(opcode, interpreter_arm)`
  across every (variant × workload-point) of the arch and
  deduplicating by opcode. Conflict (same opcode, different arm
  bodies in same arch) panics at macro expansion.
- **Opcode vocabulary follows ferrite-kernels' actual fusion
  shape, not KVM's split**. Where ferrite has a single fused
  kernel that KVM splits across two opcodes (e.g.
  `FUSED_GATE_UP_SILU_MUL` vs KVM's `GATE_SILU` + `UP_MATMUL`),
  ferrite uses one new opcode. Reconcile when the megakernel
  backend lands and needs the KVM split. Opcodes added in this
  refactor: `FUSED_ADD_RMSNORM`, `FUSED_GATE_UP_SILU_MUL`,
  `EMBED`, `RESHAPE`, `ADD`, `RESIDUAL_RMSNORM`, etc. — see
  `instruction.rs::opcode`.
- **Unmigrated Impl behavior**: `fan_out` defaults to `None`. If
  the solver picks an Impl that returns `None` from `fan_out`
  for any compiled (variant × workload-point), codegen emits a
  `compile_error!` naming the Impl. Forces wholesale.
- **`FREE` × `View`**: Drop pass does not FREE an Owned slot
  while any View slot still aliases it. Same invariant the
  existing drop pass already guarantees (owner outlives every
  view). Verified: the migration only changes the emission
  shape, not the schedule.

## Reference points

- `~/Megakernels/megakernels/demos/tp_throughput/` — KVM
  tp_throughput demo. Source of opcodes + prefill+decode-in-one
  pattern.
- `vllm-rs/crates/ferrite-forward-macro/src/emit_workload.rs` —
  today's per-workload-point inlined codegen. **This is the
  file that gets rewritten.**
- `vllm-rs/crates/ferrite-forward-macro/src/impl_lib.rs` — the
  Implementation trait. New methods (`opcode`, `fan_out`,
  `interpreter_arm`) at line ~574+; existing `emit_call` stays
  for now but has no new callers after the refactor.
- `vllm-rs/crates/ferrite-forward/src/instruction.rs`,
  `tile_table.rs` — runtime types the generated interpreter
  uses.

## Things that must not happen

- **Do not migrate one Impl at a time.** This was called out
  explicitly. Wholesale or not at all.
- **Do not invent new opcode semantics.** If KVM tp_throughput
  has an opcode for a kernel call, use that opcode and that
  field layout.
- **Do not add per-Impl bespoke runtime types.** All Impls
  share `Instruction` + `TileEntry`. Per-Impl behavior lives
  in `interpreter_arm`'s emitted Rust.
- **Do not touch solver, library invariants, fingerprint, or
  weights loading.** This is a *backend* change. If the diff
  reaches `form_regions.rs`, `library_invariants.rs`,
  `solver/`, or any loader, stop — that change does not
  belong in this refactor.
- **Do not start the megakernel backend before the host
  interpreter lands one full llama forward.** The IR isn't
  proven until the host backend matches the golden.

## Pre-existing baseline failures (not introduced by this work)

These were red on `ferrite-forward` at the fork point and remain
red here. Not blockers:

- `vllm-e2e` ignored quant variants (per inherited `HANDOFF.md`
  "Real gap inventory" section).
- `cargo check --workspace` in the top-level vllm dir trips on
  `mlx-sys` BLAS (per memory `feedback_build_flags.md`); use the
  documented `-p` builds instead.
