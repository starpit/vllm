# Ferrite Mega IR — Substrate-aware typed lowering plan

This document is the contract. Re-read every time. The previous
revision had soft language that let drift compound into "ship a
commit that compiles and has tests passing" instead of "make the
substrate bug classes unrepresentable." This revision is hard.

---

## 1. What the MegaIR IS

A typed proof-carrying value `MegaTape<S: Substrate>` such that:

> If a `MegaTape<S>` value exists, the megakernel that lowers from
> it is provably free of these substrate-layout bug classes:
>
> 1. **Out-of-bounds page slot index** — every `Page<ID, S, _>` field
>    is constructible only when `ID < S::NUM_PAGES` (sealed
>    `IsValidSlot<ID, S>` witness).
> 2. **Page-slot lifecycle violation** — every load/store/wait/arrive
>    in a variant carries a `Page<ID, S, State>` typestate field.
>    Reading a page requires `State = Filled`. Writing requires
>    `State = Empty`. Storing requires `State = Produced`. The
>    typestate transitions consume the token (`!Clone`, `!Send`,
>    `!Sync`); a token can't be reused after a transition.
> 3. **Mbarrier phase mismatch across iterations** — every `wait`
>    site carries `MbarrierPhase<P>`. The lowering threads cumulative
>    arrive counts and only constructs `MbarrierPhase<P>` where the
>    sealed `ArriveCountToPhase<N, P>` witness proves `P == N & 1`.
> 4. **Scratch-byte region overlap within a scope** — every variant
>    that uses scratch carries `ScratchRegion<OFFSET, BYTES, Scope>`.
>    Multiple regions in the same `Scope` require sealed `Disjoint<A,
>    B>` witnesses, which the lowering constructs by const-fn-checking
>    `[A.OFFSET, A.OFFSET+A.BYTES) ∩ [B.OFFSET, B.OFFSET+B.BYTES) =
>    ∅`.
> 5. **Scratch budget overrun** — every `ScratchRegion<OFFSET, BYTES,
>    S>` requires the sealed `WithinBudget<OFFSET, BYTES, S>` witness
>    proving `OFFSET + BYTES ≤ S::SCRATCH_BYTES`.
> 6. **Warp-role / handshake mis-pairing** — every
>    producer/consumer pair (loader-arrives ↔ consumer-waits;
>    consumer-arrives ↔ storer-waits) carries
>    `WarpRoleTag<R_PROD>` and `WarpRoleTag<R_CONS>` where the role
>    constants are pinned to the substrate's role assignment.
>    Mis-pairing fails to typecheck.

If any of these fails to hold for a particular `Vec<Instruction<W>>`
input, `lower(...)` panics at proc-macro construction time and the
panic surfaces as a compile error on the user's `#[forward]`. There
is no runtime check, no `unwrap`, no `_ =>`, no `unreachable!()`,
and no "TODO assert later".

## 2. What the MegaIR IS NOT

It is **not** a typed version of `Instruction<W>`. `Instruction<W>`
is the semantic op (RmsNorm in_slot → out_slot using weight). The
MegaIR is the **substrate-aware lowering** of that op into physical
kernel layout (page IDs, lifecycle states, mbarrier phases, scratch
offsets, warp roles). They live at different abstraction layers.

It is **not** "typed runtime newtypes for op fields." Wrapping a
`u32` slot id in `ActivationSlotId(u32)` with a bounds-check
constructor catches "slot id out of range" — which is **not on the
bug-class list above**. Field-validity newtypes are NOT the IR's
deliverable. They were the trap that made the previous revision of
this plan look done when it wasn't.

## 3. Forbidden field types on `MegaTape<S>` variants

The following are **forbidden** as the load-bearing field type on
any variant of the typed lowered form:

- `ActivationSlotId(u32)` — semantic slot id, not a substrate page.
- `LayerIndex(u32)` — semantic layer, no substrate proof attached.
- `WeightRef(String)` — fine to keep alongside, but it is NOT a
  substrate proof.
- `NormEps(f32)`, `FiniteF32(f32)`, `MatmulShape { ... }`,
  `CosSinRef(String)`, `RopeFlags { ... }`, `VocabSize(u32)`,
  `SlidingWindowSize(u32)`, `BarrierEdge(u32)`,
  `BarrierTotalExpr(String)`, `BaseStage(u32)` — every one of these
  is a field-validity newtype. They prove a value is well-formed in
  isolation. They do NOT prove substrate-layout invariants.

These types may EXIST as helper carriers (e.g. the lowering function
needs to pass a weight path through to the emit step), but they
cannot be the **fields** the type system uses to discharge the bug
classes. A variant whose only typed fields are field-validity
newtypes is **not a MegaIR variant**; it's just a renamed
`Instruction<W>`.

## 4. Required field types on `MegaTape<S>` variants

Each variant must carry, as fields, instances of the substrate
typestate vocabulary (in `ferrite-mega-ir/src/lowered.rs`):

- `Page<const ID: u32, S: Substrate, State: IsLifecycleState>` —
  one per page slot the variant touches, with the lifecycle state
  the variant requires (`Empty`/`Filled`/`Produced`).
- `MbarrierPhase<const P: u32>` — one per `wait` site, with the
  parity proven correct via `ArriveCountToPhase<N, P>`.
- `ScratchRegion<const OFFSET: u32, const BYTES: u32, Scope>` —
  one per scratch claim. Multiple regions per Scope come paired with
  `Disjoint<A, B>` witnesses.
- `WarpRoleTag<const R: u8>` — one per warp-role-bound action, with
  the role constant matching the substrate's role assignment.
- `TokScope<const NT: u32, const IPT: u32, Body>` — wraps any body
  that lives inside a per-token loop, threading the global arrive
  count.

A variant may carry one or two non-substrate fields **alongside**
these (a `WeightRef`, a `LayerIndex`) for runtime weight-pointer
resolution, but the substrate fields are the load-bearing ones.

**Newtypes are still mandatory everywhere.** No raw `u32` floats
through the IR — not in helper fields, not in lowered-form
constructors, not in the lowering function's locals. The rule §3
forbids field-validity newtypes from being the **load-bearing**
field on a variant; it does NOT permit raw `u32` to take their
place. Every numeric value travels in a typed wrapper. The
distinction:

- **Substrate proof fields** (load-bearing): `Page<>`,
  `MbarrierPhase<>`, `ScratchRegion<>`, `WarpRoleTag<>`. These are
  what discharges the bug classes.
- **Helper newtypes** (alongside): `LayerIndex(u32)` with a
  `LayerIndex::new(idx, num_layers)` bounds check, `WeightRef`
  with a non-empty path check, `BaseStage(u32)` with a substrate
  budget check, etc. These prevent raw-`u32` slip-throughs in the
  helper data the variant also needs (weight pointer resolution,
  emit-time formatting). They are not enough on their own to make
  a variant a real MegaIR variant; substrate proofs are required
  too.

## 5. Pipeline

```
Rust DSL (#[forward])
   ↓
CFG + FUF
   ↓ shape::infer + solver
SFUF
   ↓ Impl::fan_out → Vec<Instruction<W>>
Ferrite Tape  ← typed semantic Tape, ALREADY EXISTS in
                ferrite-forward::instr::Instruction
   ↓ ferrite_mega_ir::lower
MegaTape<S>   ← typed substrate-aware lowered form, this plan
   ↓ purely syntactic emit
emitted .cu shim
   ↓ cudaforge
.o → libmegakernels.a
```

Notes on the pipeline:

- **The Tape input is `Vec<Instruction<W>>`** — the existing typed
  enum in `vllm-rs/crates/ferrite-forward/src/instr.rs`. There is
  no `MegaOp` semantic enum. There is no `TkInstruction` enum.
  There is no `OpInstance::field_values: Vec<TokenStream>` to
  parse. Anything that says "parsing OpInstance" is a smell — go
  back to typed Instructions.
- **`fan_out` SHOULD return `Vec<Instruction<W>>` directly.** The
  current `fan_out → Vec<OpInstance> → quote!{} → static slice →
  Instruction<W>` round-trip is gratuitous type erasure. Killing
  that round-trip is part of the work this plan covers.
- **The lowering function is where the work happens.** Substrate
  resources (page IDs, scratch offsets, phases) are allocated here.
  Each typestate token is constructed only when its sealed witness
  is satisfiable. The lowering panics if no satisfying assignment
  exists. The lowering runs at proc-macro time.
- **The emit step is purely syntactic.** It pattern-matches a
  `MegaTape<S>`, extracts the const-generic IDs and witnesses, and
  splices them into CUDA strings. No decisions, no bug-class
  checks. Every check has already been discharged by the time emit
  sees the tape.

## 6. Files

```
vllm-rs/crates/ferrite-mega-ir/         — typed IR crate
├── Cargo.toml                          — zero external deps
└── src/
    ├── lib.rs                          — re-exports + crate docs
    ├── substrate.rs                    — Substrate trait + sealed
    │                                     witness traits +
    │                                     typestate primitives
    │                                     (Page<>, ScratchRegion<>,
    │                                     MbarrierPhase<>,
    │                                     WarpRoleTag<>, IsValidSlot,
    │                                     Disjoint, WithinBudget,
    │                                     ArriveCountToPhase, etc.)
    ├── nodes.rs                        — typed lowered variants
    │                                     (one per Instruction<W>
    │                                     kind, fields are substrate
    │                                     proofs)
    ├── tape.rs                         — MegaTape<S> + private
    │                                     constructor + read API
    └── lower.rs                        — lower(&[Instruction<W>]) →
                                          MegaTape<S>; the work

vllm-rs/crates/ferrite-forward/src/instr.rs   — Instruction<W> already
                                                exists; semantic Tape

vllm-rs/crates/ferrite-forward-macro/src/...  — proc-macro consumers
```

The previous plan revision said `lowered.rs` and `lowering.rs`. The
new layout splits those into `substrate.rs` (vocabulary) +
`nodes.rs` (variants) + `tape.rs` (container) + `lower.rs`
(function) so each concern lives in one file. The split is a soft
guideline; what's hard is the type vocabulary and the variant
field types.

## 7. Definition of Done — per sprint

A sprint is **done** only when:

1. **Variant fields are substrate proofs**, not field-validity
   newtypes. Verifiable by inspection: every load-bearing field on
   each migrated variant is `Page<>`, `MbarrierPhase<>`,
   `ScratchRegion<>`, or `WarpRoleTag<>` (or a sealed witness). Not
   `ActivationSlotId(u32)`.
2. **The lowering function panics at construction time** for the
   bug class the sprint targets. Verifiable by a unit test that
   constructs a buggy `Vec<Instruction<W>>` (e.g. a phase that
   doesn't match the cumulative arrive count) and confirms the
   `lower` call panics with the substrate-proof error.
3. **E2E COHERENT OUTPUT.** Capital test on pod (`vllm serve
   unsloth/Llama-3.2-1B-Instruct --enforce-eager`, curl the prompt
   "The capital of France is", confirm "Paris" + a coherent
   continuation, NOT "lib lib lib lib"). cudaforge cache hit means
   nothing changed; cudaforge cache miss + coherent output means
   the sprint shipped.

A sprint is **not** done when only conditions 1+2 hold and 3 is
deferred. The previous revision let "byte-identical CUDA + unit
tests pass" stand in for E2E coherence. That is the failure mode
this plan is structured to prevent.

## 8. INVIOLABLE INVARIANTS

### 8.1. IF IT COMPILES, IT RUNS COHERENTLY.

Substrate bug classes (the six listed in §1) are unrepresentable.
Every miswire is a Rust compile error or a proc-macro construction
panic, not a runtime crash, not silent garbage output, not "first
token is correct then it degenerates." Coherent E2E output is the
ground truth.

### 8.2. NO TOKEN ERASURE BETWEEN TYPED VALUES.

If both ends of a value's flight are typed (`fan_out` has a typed
Instruction; codegen consumes a typed Instruction), the value
travels typed. No `quote!{ #x }` → `Vec<TokenStream>` →
`parse(...)` round-trips. TokenStream is allowed only at the
serialization-to-source boundary (interpreter-codegen's emit step
that produces user-visible Rust source), not inside the
proc-macro's own data flow.

### 8.3. NO PARALLEL TYPED ENUMS.

If `Instruction<W>` already covers an op kind, a new enum that
duplicates the variant set with type-erased fields is forbidden.
The MegaIR variants are NOT a duplicate of `Instruction<W>`'s
variant set — they have **different fields** (substrate proofs).
Same variant name, different abstraction layer.

### 8.4. NO WALKER NAMING.

The previous codegen ("WalkerLines", "EmitCtx", per-op
`emit_*_walker` functions) is dead. New naming for the syntactic
emit step does not use "walker." It is a `lower_to_cuda(tape:
&MegaTape<S>) -> CuVariant` function that pattern-matches the typed
tape — no per-op dispatcher, no role-string accumulator, no
4-string `WalkerLines` struct.

### 8.5. NO COMMIT WITHOUT EXPLICIT USER PERMISSION.

Per session rule. Land work in the worktree, run cargo check / pod
build / E2E, present results, wait for go-ahead. No autonomous
commits.

## 9. Concrete next steps

(Phase B of the un-fuck-it-up.)

1. **Delete** the wrong scaffolding in `ferrite-mega-ir`:
   - `MegaOp` enum (semantic-typed variants).
   - `MegaNode` wrapper (was a wrapper around `MegaOp`).
   - `MegaTapeBuilder` with per-op `push_*` taking
     `ActivationSlotId`/`LayerIndex`/`WeightRef`/`NormEps`/etc.
   - The runtime newtypes that are semantic-flavored:
     `ActivationSlotId`, `LayerIndex`, `NormEps`, `WeightRef`,
     `CosSinRef`, `MatmulShape`, `RopeFlags`, `FiniteF32`,
     `VocabSize`, `SlidingWindowSize`, `BarrierEdge`,
     `BarrierTotalExpr`, `BaseStage`. They live in `lowered.rs`
     today; that whole file is the wrong layer.

2. **Keep** the sealed substrate vocabulary that's correct (move it
   into a new `substrate.rs`):
   - `Substrate` trait.
   - `Page<ID, S, State>`, `IsLifecycleState`, `Empty` / `Filled` /
     `Produced`.
   - `ScratchRegion<OFFSET, BYTES, Scope>`, `IsScratchScope`.
   - `MbarrierPhase<P>`, `IsValidPhase`.
   - `WarpRoleTag<R>`, `IsValidWarpRole`, `ROLE_LOADER` /
     `ROLE_LAUNCHER` / `ROLE_CONSUMER` / `ROLE_STORER`.
   - `IsValidSlot`, `Disjoint`, `WithinBudget`,
     `ArriveCountToPhase`, `NumTokens`, `NumConsumerWarps`,
     `TokScope`.

3. **Skeleton** `nodes.rs` with one variant per `Instruction<W>`
   kind, fields TODO substrate-typestate. Each variant compile-error
   stub (uninhabited or empty) until the lowering for that variant
   is built. Sprint A picks the first variant.

4. **Kill** `OpInstance::field_values: Vec<TokenStream>`. Change
   `Implementation::fan_out` to return `Vec<Instruction<W>>`
   directly. Move the source-fragment serialization (`quote!{ #var
   ( #(#exprs),* ) }`) into the host interpreter codegen step that
   actually needs Rust source.

5. **Implement `lower()`** for the first migrated variant, with a
   unit test for a buggy input that panics.

6. **Implement syntactic `emit`** for the first migrated variant.

7. **E2E coherence proof on pod** before claiming sprint done.

## 10. Sprint sequencing (after un-fuck-it-up)

| Sprint | Variant | Bug class targeted | Done = |
|---|---|---|---|
| A | `RmsNorm` | #1 (slot bounds), #2 (lifecycle), #5 (scratch budget), #6 (warp roles) | E2E coherent |
| B | `FusedQkvRopeCache` | #2, #3 (mbarrier phase math), #4 (scratch overlap inside per-tok loop) | E2E coherent |
| C | `FusedAddRmsNorm`, `SiluUpgate`, `GeluUpgate`, `DownProjResidual` | all six bug classes | E2E coherent |
| D | `Embed`, `AttentionViaCache`, `Gemm`, lm_head fusions, barrier ops, sliding attention, scalar/offset/softcap ops | all six | E2E coherent |
| E | Delete every per-op `.cuh` in `ferrite-kernels/csrc/tk/ferrite_kernels/`. Substrate `.cuh`s stay. | Substrate is the only `__device__` C++; everything else generated | E2E coherent on every model arch the project supports |

Each sprint produces a substrate-proof-bearing lowered form for its
ops, the lowering function, and the syntactic emit. No sprint
"completes" with byte-identical CUDA. Every sprint completes with
coherent E2E output on `unsloth/Llama-3.2-1B-Instruct`.
