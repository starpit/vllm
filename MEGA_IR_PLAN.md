# Ferrite Mega IR — Substrate-aware typed lowering plan

This document is the contract. Re-read every time. The previous
revision had soft language that let drift compound into "ship a
commit that compiles and has tests passing" instead of "make the
substrate bug classes unrepresentable." This revision is hard.

---

## 0. THE HEADLINE — MegaIR IS THE AST FOR THE EMITTED `.cu`

**Both halves of the contract are inviolable.**

1. **The IR carries every substrate proof** that discharges the six
   bug classes in §1. (This was the previous revision's emphasis.)

2. **The IR carries every field the emitted `.cu` needs.** Every
   kernel template parameter (`HIDDEN_DIM`, `HEAD_DIM`, `NUM_TOKENS`,
   `NUM_Q_HEADS`, `NUM_KV_HEADS`, `INTERMEDIATE_DIM`, `M`, `K`, `N`,
   `BIASED`, `INTERLEAVED`, …), every kernel runtime arg (`eps`,
   `base_stage`, `tok`, `act_ptrs[in_slot]`, `weight_ptrs[acc *
   NUM_LAYERS + layer]`, `output_ptrs[out_slot]`, …), every constant
   the emit step splices into the `.cu` source, **must be a typed
   field on the corresponding `MegaNode` variant**. The emit step
   reads typed getters and `format!()`s them. Nothing else.

If a `.cu` value is not derivable from a typed field on the
`MegaNode`, **the IR is incomplete and emit code does not exist for
that variant yet**. No "scaffold," no "placeholder body," no "TODO
sprint X." Extend the IR first; emit second. Always.

> **MegaIR IS THE AST.** Pure literal transcription. If
> `emit_<variant>` does anything beyond `format!()`-ing IR getters
> into a template that calls a pre-existing `.cuh`, the emit code is
> wrong. Revert it. Extend the IR. Try again.

This is enforced at code-review time by the principle: **show me the
IR field that produces this `.cu` value.** If the answer is "I
inferred it from the canonical name / substrate budget / context
wrapper," the field belongs on the node.

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

## 4a. Required AST fields per variant (the §0 contract, made concrete)

This section enumerates what every `MegaNode` variant MUST carry,
as typed fields, for the emit step to be a pure literal
transcription against the kernel ABI in
`crates/ferrite-kernels/csrc/tk/ferrite_kernels/`. Reference:
`crates/ferrite-kernels/csrc/smoke/ferrite_pool_abi_smoke.cu` is the
canonical "well-formed `.cu` emit" — every variant's emit must
produce text of that form.

Common to every variant (alongside the substrate proofs of §4):

- **Per-op host-slot indices**: `in_act_slot`, `out_act_slot`
  (indices into `act_ptrs[]`); `weight_accessor_idx` (index into
  `weight_ptrs[acc * NUM_LAYERS + layer]`). These are NOT the same
  as substrate page ids — the substrate page is a per-op scratch
  page id; the host slot is the gmem ptr table index.
- **Per-op `base_stage`**: the substrate page slot the kernel uses
  as offset zero. The kernel hardcodes `base_stage + kInputPageOff
  (0)`, `base_stage + kWeightPageOff (1)`, etc. — so substrate
  proofs MUST enforce that the per-op pages are contiguous starting
  at `base_stage`. (Today the proofs only enforce
  `IN_ID != WEIGHT_ID`; that is too weak — promote to
  `WEIGHT_ID == IN_ID + 1`, or replace dual page-ids with a single
  `base_stage` field whose proof spans the variant's full page
  count.)

Per-variant kernel-shape fields (every one a typed field on the
`MegaNode` variant; populated by `dispatch_instruction_to_push` from
the canonical context):

| Variant | `.cuh` template params | `.cuh` runtime args | Required IR fields |
|---|---|---|---|
| `RmsNorm` | `<Config, HIDDEN_DIM, NUM_TOKENS>` | `eps` | `hidden_dim`, `num_tokens`, `eps` |
| `FusedAddRmsNorm` | `<Config, HIDDEN_DIM, NUM_TOKENS>` | `eps` | same |
| `ScalarOffsetRmsNorm` (`rms_norm_offset`) | `<Config, HIDDEN_DIM, NUM_TOKENS>` | `eps`, `offset` | `hidden_dim`, `num_tokens`, `eps`, `offset` |
| `Embed` | `<Config, HIDDEN_DIM, NUM_TOKENS>` | (input_ids ptr) | `hidden_dim`, `num_tokens` |
| `FusedQkvRopeCache` | `<Config, HIDDEN_DIM, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS, BIASED, INTERLEAVED>` | `tok` | `hidden_dim`, `head_dim`, `num_q_heads`, `num_kv_heads`, `biased`, `interleaved` (last two ✓ already on IR) |
| `FusedGateUpActivateMul` (silu/gelu_upgate) | `<Config, HIDDEN_DIM, INTERMEDIATE_DIM, NUM_TOKENS>` | — | `hidden_dim`, `intermediate_dim`, `num_tokens`, `activation` (✓) |
| `Gemm` (gemm_bf16) | `<Config, K, N, M>` | — | `m` (IR has `n`, `k` ✓; `m` missing) |
| `FusedCublasGemmAdd` (down_proj_residual) | `<Config, K, N, NUM_TOKENS, K_OFFSET, K_FULL>` | — | `num_tokens`, `k_offset`, `k_full` |
| `CutlassFusedNormGemm` (lm_head) | `<Config, K, N, NUM_TOKENS>` | `eps` (and `offset` for the offset-rms variant) | `num_tokens`, `eps` (`offset` already on IR via `Option<FiniteF32>` ✓) |
| `AttentionViaCache` (attention_partial + attention_reduction) | many: `HEAD_DIM`, `NUM_Q_HEADS`, `NUM_KV_HEADS`, `BLOCK_SIZE`, `NUM_PAGES`, `MAX_SPLITS`, sliding-window flags | `block_table`, `seq_lens`, `block_table_stride` | head_dim, num_q_heads, num_kv_heads, block_size, max_splits, sliding_window flag (`AttentionKind::Sliding(w)` ✓), interleaved (✓) |
| `BarrierSignal` / `BarrierWait` | n/a — emit produces explicit `__syncthreads()` / `arrive` / `wait` lines | edge id, expected count | `edge` (✓), `expected` (✓) |
| `Add` | (no per-op kernel — emit is a per-page residual fold) | — | `hidden_dim`, `num_tokens` (for the load/add/store loop) |
| `ScalarMul` | n/a | `scale` | `hidden_dim`, `num_tokens`, `scale` (`scale` ✓) |
| `TanhSoftCap` | n/a | `cap` | `hidden_dim`, `num_tokens`, `cap` |
| `SpliceMmEmbeds` | (host-side splice op) | `slot_id` | `slot_id` (✓), shape fields |

**This table is the §0 contract made auditable.** When a sprint
opens, the first commit extends the IR and `dispatch_instruction_to_push`
to populate every field for that variant. The second commit (and
only the second) writes the per-variant emit body. Per §0 there is
no scaffold — extend, then transcribe.

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

### 8.0a. TK 2.0 PRIMITIVES ONLY. NO TK 1.0 BULLSHIT.

The emit step calls **TK 2.0 primitives only**. The TK 2.0 primitive
surface lives in:

```
third_party/thunderkittens/include/         ← TK 2.0 (canonical)
  types/global/gl.cuh                       ← gl<T, b, d, r, c, TMA_Types...>
  types/global/tma.cuh                      ← tensor-map descriptor dict
  ops/group/memory/tile/tma.cuh             ← tma::load_async / store_async
  ops/group/util/sync.cuh                   ← arrive / wait
  ops/group/group.cuh                       ← group<N>::sync(int id)
  ops/group/mma/warp.cuh                    ← mma_AB / mma_ABt / ...
  ops/group/register/{tile,vec}/maps.cuh    ← warp::add / mul / sum / ...
  types/register/{rt,rv}.cuh                ← register tile / vector types
```

The TK 1.0 / VM-style reference lives in:

```
third_party/thunderkittens/tests/vm/llama_official/  ← TK 1.0-style
  matvec_pipeline.cuh                                ← raw-ptr scheduling
  rms_matvec_rope_append.cu                          ← old globals + PTM
  utils.cuh                                          ← helpers (some valid)
```

**Reading the TK 1.0 / VM reference for emit pattern recognition is a
documented failure mode.** It looks like working CUDA, the function
names overlap, but the calling conventions diverge in ways that don't
fail any narrow per-variant unit test — they only fail at link/compile
time against actual TK 2.0. The previous cuda_emit revision shipped 9
sprints of code that pattern-matched on `tests/vm/llama_official/`
references (raw `bf16**` Globals tables, `g.act_ptrs[3]` index math,
init-list `{0}` coordinates) and was nuked without ever compiling once
against TK 2.0.

Concrete TK 2.0 contracts every emit must respect:

- **Globals are `gl<...>` descriptors**, not raw `T**` tables.
  `gl<T, b, d, r, c, TMA_Types...>` carries a `T* raw_ptr` plus
  compile-time-or-runtime dims plus a `tma_descs` dict over the
  shared-tile types it'll be TMA'd against. Constructed host-side
  via `gl(T*, batch, depth, rows, cols)`.
- **TMA calls take a `gl` by reference and a `coord<>` index**:
  `kittens::tma::load_async(ST &dst, const GL &src, const COORD &idx,
  semaphore& bar)`. NOT raw pointers. NOT init-lists.
- **Cross-warp sync is `kittens::group<N>::sync(int id)`** for named
  PTX `bar.sync` 1..=15 (bar 0 = `__syncthreads`). The IR's `BarRef`
  fields feed this.
- **Mbarrier handoff is `kittens::wait(sem, phase)` /
  `kittens::arrive(sem)`**. The IR's `MbarrierPhase` typestate +
  page-handoff semaphores in `ferrite_substrate.cuh::SharedState`
  feed this.
- **Matmul is `kittens::warp::mma_AB(D, A, B, C)` and friends**
  on register tiles, NOT a hand-rolled "matvec_pipeline" port.
  Layout shapes (rt_fl/rt_bf, NxK/NxM/MxK) come from the IR's
  matmul-shape primitives.

**Workflow before emitting any new variant:**

1. **Read the actual TK 2.0 primitive header for every call you'd
   emit.** If a header doesn't exist or you can't find the signature,
   the primitive doesn't exist — find another way or add an IR field.
2. **Check the host-side ABI** in `crates/ferrite-forward/src/
   interpreter/mega/mod.rs` (`LaunchArgs*`, `launch*` fns). The
   kernel signature you emit must match what the host already
   constructs and passes.
3. **If the emit needs a value not on the IR, STOP — extend the IR.**
   This rule is §8.0; the TK 2.0 constraint is §8.0a; both apply.

### 8.0. MEGAIR IS THE AST FOR THE EMITTED `.cu`.

Every kernel template parameter and every kernel runtime arg the
emitted `.cu` needs is a typed field on the corresponding `MegaNode`
variant (see §0). Emit is pure literal transcription: read typed
getters off the IR, splice them into TK 2.0 primitive calls (see
§8.0a). No inference, no helper computation that shapes the emitted
source, no "wiring up" anything not on the node. **If the IR lacks
a field the `.cu` needs, STOP — extend the IR. Do NOT invent on the
emit side.**

The shape of every variant's emit body is determined by:

- The IR's typed-field surface (HIDDEN_DIM, NUM_TOKENS, page IDs,
  scratch offsets, mbarrier phases, BAR IDs, weight-accessor
  indices, ...) — these are the *values* spliced in.
- The TK 2.0 primitive surface (§8.0a) — these are the *function
  calls* spliced in.
- Ferrite's substrate (`ferrite_substrate.cuh::SharedState<Config>`,
  `ferrite_warp_roles.cuh`, `ferrite_barrier.cuh`) — these define
  the per-CTA scaffolding the emit wraps the primitives in.

There is no fourth source. There are no per-op `.cuh` wrappers
(those were nuked); there is no hand-rolled scheduling logic; there
is no "matvec_pipeline port." If you find yourself reaching for any
of those, stop and re-read §8.0a.

A scaffold that ships an emit pipeline whose data model can't reach
the kernel ABI is not progress; it is structural debt that must be
reverted. The shape of every variant's typed-field surface is
determined by the kernel ABI it lowers to, full stop.

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

> **STATUS (2026-05-16): items 1-6 below are DONE.** Phase B of the
> un-fuck-it-up shipped in `84df2b3f4` (substrate vocabulary +
> nodes.rs + lower.rs + dispatch). The IR-as-AST extension on top
> (every emitted variant carries every kernel template/runtime
> field per §0/§4a/§8.0) shipped across 5 iteration commits
> (`4dc3b7210`→`18ee08f95`) on top of the revert of the wrong
> scaffold (`288324633`).
>
> Pod audit: 607 emitted, 54 skipped (MoE/MLA/vision — substrate
> work, not AST work), 0 errors.
>
> The history of items 1-6 is preserved here as the un-fuck-it-up
> reference. Item 7 (E2E coherence) remains the per-sprint bar in
> §10 — the per-variant CUDA emit step is the next sprint, and
> "done" still means E2E coherent on `unsloth/Llama-3.2-1B-Instruct`.

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

| Sprint | Variant | Bug class targeted | IR substrate | IR AST | Emit | Done = |
|---|---|---|---|---|---|---|
| A | `RmsNorm` | #1 (slot bounds), #2 (lifecycle), #5 (scratch budget), #6 (warp roles) | ✅ | ✅ | ⏳ | E2E coherent |
| B | `FusedQkvRopeCache` | #2, #3 (mbarrier phase math), #4 (scratch overlap inside per-tok loop) | ✅ | ✅ | ⏳ | E2E coherent |
| C | `FusedAddRmsNorm`, `SiluUpgate`, `GeluUpgate`, `DownProjResidual` | all six bug classes | ✅ | ✅ | ⏳ | E2E coherent |
| D | `Embed`, `AttentionViaCache`, `Gemm`, lm_head fusions, barrier ops, sliding attention, scalar/offset/softcap ops | all six | ✅ | ✅ | ⏳ | E2E coherent |
| E | Delete every per-op `.cuh` in `ferrite-kernels/csrc/tk/ferrite_kernels/`. Substrate `.cuh`s stay. | Substrate is the only `__device__` C++; everything else generated | ⏳ | ⏳ | ⏳ | E2E coherent on every model arch the project supports |

Column legend:
- **IR substrate** — substrate-proof primitives (page ids, scratch,
  phases, layer) on the typed `MegaNode` variant.
- **IR AST** — every kernel template parameter and runtime arg the
  emit step needs is a typed field on the `MegaNode` (per §0/§4a).
- **Emit** — pure literal `format!()` `.cu` source emit per §8.0;
  the next-sprint deliverable.

✅ = shipped (commits in worktree). ⏳ = next.

Each sprint produces a substrate-proof-bearing lowered form for its
ops, the lowering function, and the syntactic emit. No sprint
"completes" with byte-identical CUDA. Every sprint completes with
coherent E2E output on `unsloth/Llama-3.2-1B-Instruct`.

## 11. Current state — pod audit

> Last verified 2026-05-16 on H100 nick (full workspace,
> `cargo clean -p ferrite-forward-macro` then `FERRITE_MEGA=1
> cargo build -p ferrite-models --features cuda --release`):
>
> - **607 canonicals emitted** (every used `MegaNode` variant carries
>   every kernel-ABI field per §0/§4a/§8.0).
> - **54 skipped** — MlaSplit/MlaAttention (28) + MoE (16) + vision
>   tower (10). Each needs a new `MegaNode` variant AND a new
>   `.cuh` kernel — substrate work, not AST work.
> - **0 errors.**
>
> ## TK 1.0 pollution incident — 2026-05-17
>
> A first attempt at `cuda_emit` shipped 9 sprints (foundation +
> RmsNorm + Add + ScalarMul + TanhSoftCap + FusedAddRmsNorm +
> ScalarOffsetRmsNorm + Embed + BarrierSignal/Wait) that
> pattern-matched on `tests/vm/llama_official/` (TK 1.0-style)
> reference code and would NOT compile against TK 2.0's actual
> primitive surface (raw `bf16**` Globals tables instead of
> `kittens::gl<...>`; init-list `{0}` coordinates instead of
> `coord<>`; no `gl::tma_descs` for the TMA descriptor dict). The
> per-variant unit tests passed because they only checked emitted
> text patterns, never compiled the output against TK 2.0.
>
> **Recovery:** the `crates/ferrite-mega-ir/src/cuda_emit/`
> directory has been nuked. The IR-side additions made during
> those sprints are kept because they're substrate / model-metadata,
> ABI-neutral:
>
> - `TapeBudget.num_layers` (model param, used for weight indexing)
> - `MegaTapeBuilder::finish(num_layers)` (companion API change)
> - FusedAddRmsNorm `consumer_bar_reduce` / `consumer_bar_publish`
>   / `bar_pair_proof` (sealed-witness BAR IDs in 1..=15 — feed
>   into TK 2.0's `kittens::group<N>::sync(int id)` legitimately)
> - ScalarOffsetRmsNorm: same three fields
> - Proc-macro side emits the BAR literals + `b.finish(NUM_LAYERS)`
>
> Re-emit happens against TK 2.0 from a clean audit. See §8.0a for
> the inviolable TK 2.0-only constraint and the workflow.
>
> Next: TK 2.0 surface audit (gl, coord, tma::*, warp::*, mma_*,
> group<N>::sync), THEN cuda_emit rebuild starting from RmsNorm,
> THEN E2E coherence on `unsloth/Llama-3.2-1B-Instruct` per §7.
