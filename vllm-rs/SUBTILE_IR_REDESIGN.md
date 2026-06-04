# SubtileIR + dumb-tape-player redesign

Status: PROPOSAL. Read end-to-end and correct before any code lands.
This is the plan that closes the "codegen invents shit" bug class so
megakernel decode either works or fails as a Rust compile error.

## 0. Contract

**Every operation the megakernel performs at runtime is an `Instr` in
the SubtileIR tape.** That includes: TMA loads, TMA stores, barrier
inits/waits/arrives, every fence (`commit_group`, `wait_group`,
`threadfence`, `__syncthreads`), persistent-state declarations,
loops, kernel-arg declarations, kernel entry, kernel exit, every
compute body (RmsNorm, GEMM, SiLU, Mul, Add, RoPE, AttnDecode init/
qkt/sv/finalise, lm_head, embed). If the kernel runs it, an `Instr`
encodes it.

**Tape generation is enumeration.** The lowering walker
(`tk_lower.rs`) reads SubtileIR nodes (`LoweredOp`) and emits a flat
`Vec<Instr>`. The walker cannot decide "I will insert a fence here"
— a `Fence { ... }` is already a node in the input IR. The walker
cannot pick a parity — parity is a typed field on each
`BarrierWait`. The walker reads fields and pushes `Instr`s.

**The tape player is dumb transcription.** The codegen file (`emit.rs`,
new — replaces today's tk_codegen.rs body-string functions) is a
single `match` over `Instr` kinds. Each arm is **≤5 lines** of
`format!()` whose `{...}` placeholders are ALL filled from fields on
the instruction. No conditionals beyond the kind dispatch. No control
flow beyond IR-encoded `ForLoop`. No constants pasted from ambient
state. No lookups into "the program" or "the previous Instr."

**Hard line budget on the player file: ≤800 lines total.** If we
exceed it, the IR is wrong: split an `Instr` kind into more granular
kinds. The current `tk_codegen.rs` is 3230 lines — the redesign
deletes ~80% of it.

**Compile-time witnesses replace runtime guarantees.** Every paired
producer/consumer (RopeAppend → AttnDecode for K and V; QKt → SV for
softmax state; LoadAsync → BarrierWait for page readiness) shares one
typed handle. Mismatched arity, missing producer, wrong scope, wrong
parity, wrong rope form are all Rust compile errors.

---

## 1. Current state — what the SubtileIR is missing

Citations from the recent invention audit (137 verified inventions
across 12330 lines).

### A. Fence/drain has no IR fields

- `crates/ferrite-wavefront/src/tk_codegen.rs:142`
  (`cross_op_gmem_fence_body`) — emits a fixed 5-line string. Fields
  it should carry but doesn't: scope (block/device/system), wait_n
  (drain-all vs drain-N), producer warp role, consumer warp role,
  whether the bracketing `__syncthreads()` are emitted.
- `crates/ferrite-wavefront/src/tk_codegen.rs:67`
  (`KernelEndDrain::emit`) — same 5-line shape, separately
  hardcoded. Duplicates the cross-op fence with no shared
  abstraction.
- `crates/ferrite-wavefront/src/tk_codegen.rs:226`
  (`tma_store_async`) — bakes commit_group + wait_group<0> into the
  store call's tail. Wait-group N is hardcoded.

### B. K/V layout reconstructed independently per-op

- `crates/ferrite-wavefront/src/tk_lower.rs:562`
  (`AttnDecodeOp::kv_layout`) and
  `crates/ferrite-wavefront/src/tk_lower.rs:1812`
  (`RopeAppendOp::kv_layout`) — each rebuilds `KvCacheLayout` from
  its own `(num_kv_heads, head_dim, act_elem)` fields. Numerically
  agree only because Llama-1B happens to fix the triple to (8,64,2).
- `crates/ferrite-wavefront/src/tk_lower.rs:1486`
  (RopeRotate Q-side) — hand-rolls `head_dim * act_elem` inline,
  bypassing `KvCacheLayout::cos_sin_row_bytes` used by RopeAppend.

### C. Producer-consumer dataflow via side-table

- `crates/ferrite-wavefront/src/tk_orchestrate.rs:547-552` —
  `pending_k_unfenced.remove(&k_cache).unwrap_or_else(|| GmemHandle::new_initial(k_cache))`.
  The `unwrap_or_else` silently fabricates a fresh handle if the
  BTreeMap entry is missing. No IR edge linking layer-N RopeAppend
  to layer-N AttnDecode for either K or V.

### D. Online softmax state via stringly-typed identifiers

- `crates/ferrite-wavefront/src/tk_codegen.rs:644`
  (`TkProgram::prelude: String`) — per-warp accumulators (`__m_max`,
  `__l_sum`, `__o_accum`) are raw CUDA declarations in a freeform
  string with positional placement contract.
- `crates/ferrite-wavefront/src/tk_codegen.rs:2962`
  (`AttnDecodeQktSoftmaxStepBody`) — produces `__p_a{u}` keyed off
  `unique_id`; consumed by `AttnDecodeSvAccumStepBody` purely by C++
  identifier sharing. No typed handle linking the two phases.

### E. RopeForm not in IR

- `crates/ferrite-wavefront/src/tk_codegen.rs:1054`
  (`rope_consumer_body`) — hardcodes NeoX `(i, i+half)` pair pattern.
  No `RopeForm` enum on `RopeRotateOp` / `RopeAppendOp`. Q-side and
  K-side could silently pick different forms.

These five clusters are 130+ of the 137 invention sites. Closing them
removes the bug class.

---

## 2. New IR shape

### 2.1 The flat tape

```rust
pub struct TkTape {
    pub kernel_args: Vec<KernelArg>,        // ordered, ABI-fixed
    pub prelude: Vec<PreludeDecl>,          // one decl per Instr; was a String
    pub instrs: Vec<Instr>,                 // the body
    pub end_drain: FenceSpec,               // tail Instr, separated for type clarity
}
```

The walker produces a `TkTape` from a `LoweringInput`. The player
runs `match` over `instrs` (and `prelude`, and `kernel_args`,
trivially). Nothing else.

### 2.2 `Instr` kinds — exhaustive

```rust
pub enum Instr {
    // ── synchronization primitives (each emits exactly one CUDA call)
    Syncthreads { scope: SyncScope },
    Threadfence { scope: FenceScope },
    CommitGroup { kind: CommitKind },          // BulkStore | NonBulk
    WaitGroup { kind: CommitKind, n: u32 },

    // ── cross-op + kernel-end fence (consolidated, fully fielded)
    Fence(FenceSpec),

    // ── named barriers (page-ready / page-done / page-consumed etc.)
    BarrierInit { id: BarrierId, count: u32 },
    BarrierWait { id: BarrierId, parity: ParityExpr, role: WarpRole },
    BarrierArrive { id: BarrierId, role: WarpRole },

    // ── memory ops
    LoadAsync(LoadSpec),
    StoreAsync(StoreSpec),

    // ── compute body — body_id keys a sealed template
    Compute { body_id: ComputeBodyId, role: WarpRole },

    // ── control flow — only here, not implicit
    ForLoop { var: LoopVarId, count: KernelArgRef, body: Vec<Instr> },

    // ── role-gated raw asm escape (sealed per-arch — see §6 open Q)
    Asm { role: WarpRole, lines: AsmLines },
}

pub struct FenceSpec {
    pub scope: FenceScope,
    pub wait: WaitMode,
    pub producer_role: WarpRoleSet,
    pub consumer_role: WarpRoleSet,
    pub bracket_pre_sync: bool,
    pub bracket_post_sync: bool,
}

pub enum FenceScope { Block, Device, System }
pub enum WaitMode { DrainAll, DrainN(u32) }
pub enum SyncScope { Cta, GroupOf(u32) }
pub enum CommitKind { BulkStore, NonBulk }

pub struct LoadSpec {
    pub dst_page: PageId,
    pub src_buf: BufId,
    pub src_byte_off: ByteOffsetExpr,    // either ConstU64 or LoopMul(LoopVarId, u64)
    pub bytes: u32,
    pub role: WarpRole,
    pub barrier: BarrierId,              // arms expect_bytes on this barrier
}

pub struct StoreSpec {
    pub src_page: PageId,
    pub dst_buf: BufId,
    pub dst_byte_off: ByteOffsetExpr,
    pub bytes: u32,
    pub role: WarpRole,
    pub commit_strategy: StoreCommitStrategy,
}

pub enum StoreCommitStrategy {
    InlineCommitWait,                    // store + commit_group + wait_group<0>
    DeferredToFence(FenceId),            // store; commit/wait emitted by a later Fence Instr
}

pub enum ByteOffsetExpr {
    Const(u64),
    LinearLoop { var: LoopVarId, stride: u64, base: u64 },  // base + var*stride
}

pub enum ParityExpr {
    Static(u8),                          // compile-time 0 or 1
    LoopParity(LoopVarId),               // (var & 1) at runtime
}

pub struct WarpRoleSet(u32);             // bitmask over WarpRole
```

### 2.3 ComputeBody — sealed templates, fully fielded

Compute bodies are the one place where a single `Instr` legitimately
expands to many CUDA lines — but only because the underlying TK 2.0
primitive sequence is itself the operation (e.g. `RmsNorm` is an mma
+ reduce + rsqrt + multiply over a tile). To keep the player dumb
**the body's CUDA text is a single `&'static str` template per
`ComputeBodyId`**, with `{field_name}` placeholders bound from the
Instr's body-specific field struct.

```rust
pub enum ComputeBodyId {
    RmsNormFp32 { src_page: PageId, dst_page: PageId, gain_buf: BufId,
                  cols: u32, rows: u32, eps_bits: u32 },
    GemmM1 { lhs_page: PageId, rhs_buf: BufId, rhs_byte_off: u64,
             out_page: PageId, m: u32, n: u32, k: u32, accum: AccumKind },
    SiluMul { gate_page: PageId, up_page: PageId, out_page: PageId, cols: u32 },
    RopeNeoxQ { src_page: PageId, dst_page: PageId,
                cos_sin_buf: BufId, position: KernelArgRef,
                cos_sin_layout: KvLayoutWitness, head_dim: u32, num_heads: u32 },
    RopeNeoxK { src_page: PageId, dst_page: PageId,
                cos_sin_buf: BufId, position: KernelArgRef,
                kv_layout: KvLayoutWitness, head_dim: u32, num_kv_heads: u32 },
    AttnDecodeInitSoftmax { state: SoftmaxStateId, num_q_heads: u32, num_kv_heads: u32 },
    AttnDecodeQktStep { state: SoftmaxStateId, q_page: PageId, k_page: PageId,
                        scale: f32, num_q_heads: u32, num_kv_heads: u32, head_dim: u32 },
    AttnDecodeSvStep { state: SoftmaxStateId, v_page: PageId,
                       num_q_heads: u32, num_kv_heads: u32, head_dim: u32 },
    AttnDecodeFinalise { state: SoftmaxStateId, out_page: PageId,
                         num_q_heads: u32, head_dim: u32 },
    EmbedHidden { /* … */ },
    LmHeadArgmax { /* … */ },
    // … one per architectural primitive, all fielded
}
```

The player matches on `ComputeBodyId`, looks up a `&'static str`
template via a sealed accessor, and `format!()`s the fields in. **No
arm exceeds 5 lines.** The CUDA template itself is reviewed as a
fixed asset; field substitution is mechanical.

`SoftmaxStateId` and `KvLayoutWitness` are sealed types described
below.

### 2.4 PreludeDecl — typed declarations

```rust
pub enum PreludeDecl {
    PerWarpFloatArray { name: PreludeName, len: u32, owner: ComputeBodyId },
    PerWarpFloatMatrix { name: PreludeName, rows: u32, cols: u32, owner: ComputeBodyId },
    SmemTilePtr { name: PreludeName, page: PageId },
    KernelArgAlias { name: PreludeName, arg: KernelArgRef },
}
```

Replaces `TkProgram::prelude: String`. Each variant emits one CUDA
declaration line. Walker pushes one `PreludeDecl` per ComputeBody
that needs persistent state; player emits in deterministic order at
the top of the kernel.

### 2.5 KernelArg

```rust
pub struct KernelArg { pub name: KernelArgName, pub ty: KernelArgTy }
pub enum KernelArgTy { U32 { source: U32Source }, BufPtr(BufId), CosSinPtr }
pub enum U32Source { NumKvPages, DecodePosition, DecodeSlot }
pub struct KernelArgRef(pub u16);  // index into TkTape::kernel_args
```

The player emits the kernel signature from `kernel_args` directly —
again, no ambient lookup.

---

## 3. The dumb tape player

### 3.1 File and budget

- New file: `crates/ferrite-wavefront/src/tk_player.rs`.
- Hard budget: **≤800 lines total**, **≤5 lines per match arm**.
- Single public fn: `pub fn emit_kernel(tape: &TkTape) -> String`.
- Internal: `fn emit_instr(out: &mut String, instr: &Instr)` is the
  match dispatch.

### 3.2 Worked example arms

```rust
// Sync primitives — one CUDA call each.
Instr::Syncthreads { scope: SyncScope::Cta } =>
    out.push_str("__syncthreads();\n"),

Instr::Threadfence { scope: FenceScope::Device } =>
    out.push_str("__threadfence();\n"),

Instr::CommitGroup { kind: CommitKind::BulkStore } =>
    out.push_str("asm volatile(\"cp.async.bulk.commit_group;\");\n"),

Instr::WaitGroup { kind: CommitKind::BulkStore, n } =>
    write!(out, "asm volatile(\"cp.async.bulk.wait_group {n};\");\n").unwrap(),

// Fence — fully fielded, no branches besides scope.
Instr::Fence(FenceSpec { scope, wait, bracket_pre_sync, bracket_post_sync, .. }) => {
    if *bracket_pre_sync { out.push_str("__syncthreads();\n"); }
    write!(out, "asm volatile(\"cp.async.bulk.commit_group;\");\n").unwrap();
    write!(out, "asm volatile(\"cp.async.bulk.wait_group {};\");\n", wait_n_for(*wait)).unwrap();
    write!(out, "{}\n", threadfence_call(*scope)).unwrap();
    if *bracket_post_sync { out.push_str("__syncthreads();\n"); }
}

// LoadAsync — single TK 2.0 call (`expect_bytes` arms barrier; load fires).
Instr::LoadAsync(LoadSpec { dst_page, src_buf, src_byte_off, bytes, role, barrier }) =>
    write!(out, "if (__role == {role}) {{ kittens::group<1>::tma::expect_bytes({bar}, {bytes}); kittens::group<1>::tma::load_async(reinterpret_cast<void*>(page_buf[{dst}]), reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(buf{src}) + {off}), {bytes}, {bar}); }}\n",
        role = role.id(), bar = barrier.var(), bytes = bytes,
        dst = dst_page.0, src = src_buf.0,
        off = byte_off_expr(src_byte_off)).unwrap(),

// ForLoop — recurses; body lives in the tape, no implicit anything.
Instr::ForLoop { var, count, body } => {
    write!(out, "for (uint {var} = 0; {var} < {count}; ++{var}) {{\n",
        var = var.name(), count = count.name()).unwrap();
    for inner in body { emit_instr(out, inner); }
    out.push_str("}\n");
}

// Compute — body_id keys a sealed template; fields format in.
Instr::Compute { body_id, role } =>
    write!(out, "if (__role == {}) {{ {} }}\n", role.id(), compute_body_template(body_id)).unwrap(),
```

`wait_n_for`, `threadfence_call`, `byte_off_expr`,
`compute_body_template` are all const-fn or pure projections from
fields → string fragments. Each is ≤10 lines and lives in
`tk_player.rs`.

### 3.3 Forbidden in the player

- No reading from program-wide state ("the previous Instr was X so I
  emit Y").
- No formula computation (no `head_dim * act_elem`, no
  `slot * row_bytes`, no `1u << k` — those are fields).
- No conditional emission (no "if barrier already initialized, skip")
  — every Instr emits its full body.
- No fallthrough between match arms.
- No `Vec` building inside an arm except for the trivial body
  recursion in ForLoop.

If a CUDA primitive needs more than 5 lines to emit, **split the
Instr** so each component is its own kind.

---

## 4. Compile-time witnesses

### 4.1 KvLayoutWitness — shared per-K-cache-BufId

```rust
pub struct KvLayoutWitness {
    id: KvLayoutId,                     // private; constructed only via builder
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub act_elem: u32,
}
impl KvLayoutWitness {
    pub fn row_bytes(&self) -> u32 { self.num_kv_heads * self.head_dim * self.act_elem }
    pub fn cos_sin_row_bytes(&self) -> u32 { self.head_dim * self.act_elem }
}
```

The orchestrator builds ONE `KvLayoutWitness` per K-cache `BufId` at
tape-build time, stores them in a `Vec<KvLayoutWitness>` on the
tape. RopeAppend's emit and AttnDecode's emit both reference the
witness by `KvLayoutId`. `RopeRotateOp` for Q rotation also receives
the same `KvLayoutWitness` (cos_sin_row_bytes is shared). Mismatch
between producer and consumer becomes structurally impossible: there
is one instance, dereferenced by both.

Optional stronger form (open question §7): const-generic
`KvCacheLayout<const NUM_KV_HEADS: u32, const HEAD_DIM: u32, const ACT_ELEM: u32>`
on the relevant ComputeBody arms. Per-arch monomorphization keeps
this practical for single-arch builds; cross-arch dispatch reverts
to id-based.

### 4.2 KvCacheProducer — typed dataflow edge

```rust
pub enum KvCacheProducer {
    SameForwardRopeAppend { rope_op_idx: usize },
    KernelEntryInitial,                 // K cache populated by per-op prefill, read-only here
}

pub struct AttnDecodeOp {
    pub k_producer: KvCacheProducer,
    pub v_producer: KvCacheProducer,
    // …
}
```

The `unwrap_or_else(|| GmemHandle::new_initial(...))` in
`tk_orchestrate.rs:547` becomes a compile error: the orchestrator
must construct an `AttnDecodeOp` with explicit producers. If
`SameForwardRopeAppend(j)` is chosen, a tape-build assertion fires
that op `j` is a RopeAppend writing the same BufId — a panic at
build time, not a silent runtime fall-through.

### 4.3 SoftmaxState<U, Phase> — typed recurrence

```rust
pub struct Empty; pub struct Initialised; pub struct FreshP;
pub struct Accumulated; pub struct Finalised;

pub struct SoftmaxState<U: Unit, Phase> {
    id: SoftmaxStateId,
    _marker: PhantomData<(U, Phase)>,
}

impl<U: Unit> SoftmaxState<U, Empty> {
    pub fn init(...) -> SoftmaxState<U, Initialised> { ... }
}
impl<U: Unit> SoftmaxState<U, Initialised> {
    pub fn qkt_step(self, ...) -> SoftmaxState<U, FreshP> { ... }
}
impl<U: Unit> SoftmaxState<U, FreshP> {
    pub fn sv_step(self, ...) -> SoftmaxState<U, Accumulated> { ... }
}
impl<U: Unit> SoftmaxState<U, Accumulated> {
    pub fn next_iter(self) -> SoftmaxState<U, FreshP> { ... }
    pub fn finalise(self, ...) -> SoftmaxState<U, Finalised> { ... }
}
```

Calling `sv_step` before `qkt_step` is a compile error. Hoisting
`init` inside the loop is a compile error (the loop entry consumes
`Initialised`; later iterations consume `Accumulated::next_iter`).

The walker emits AttnDecode bodies via this typestate API, threading
one `SoftmaxState<U, Phase>` per consumer warp through init → loop
(qkt → sv) → finalise. The C++ identifiers `__m_max_a{u}`,
`__l_sum_a{u}`, `__o_accum_a{u}`, `__p_a{u}` become an
implementation detail of `SoftmaxStateId::name()`; freeform string
sharing is gone.

### 4.4 RopeForm — const-generic shared between Q and K

```rust
pub trait RopeForm { fn pair_lo(i: u32, half: u32) -> u32; fn pair_hi(i: u32, half: u32) -> u32; }
pub struct NeoX; pub struct Interleaved;
impl RopeForm for NeoX { /* (i, i+half) */ }
impl RopeForm for Interleaved { /* (2i, 2i+1) */ }

pub struct RopeRotateOp<F: RopeForm> { _form: PhantomData<F>, /* … */ }
pub struct RopeAppendOp<F: RopeForm> { _form: PhantomData<F>, /* … */ }
```

A forward-pass that mixes NeoX and Interleaved becomes a Rust type
error. The hardcoded `(i, i+half)` in `rope_consumer_body` is
replaced by `F::pair_lo(i, half)` / `F::pair_hi(i, half)` — both
sides of Q/K rotation share the same `F`.

### 4.5 PageId / BarrierId / KvLayoutId — sealed newtypes

All `*Id` types are `pub struct Foo(u32)` with a private inner field
and a sealed builder. The walker can construct ids only via the
tape-builder API; user code can only consume ids by reading them off
an Instr.

---

## 5. Migration phases

Each phase lands as one or more commits on `worktree-ff-subtile`. Each
phase has a **stop condition** — if the per-phase pod test
regresses, stop and bisect before moving on.

### Phase 0 — scaffold

- Add new module `crates/ferrite-wavefront/src/tk_tape.rs` with
  empty `TkTape`, `Instr`, `FenceSpec`, etc. Compiles, no callers.
- Add new module `crates/ferrite-wavefront/src/tk_player.rs` with
  `emit_kernel(tape: &TkTape) -> String` returning empty kernel.
- Stop condition: workspace builds, tests pass, megakernel still
  runs via the old path.

### Phase 1 — sync primitives

- Migrate `Syncthreads`, `Threadfence`, `CommitGroup`, `WaitGroup`
  to new Instrs.
- New tape-builder helpers; old codegen calls now route through new
  Instrs.
- Stop condition: pod build + decode produces same Paris!!! output
  as today (no regression). Confirms the migration is byte-identical.

### Phase 2 — Fence consolidation

- Replace `TkInstr::CrossOpGmemFence` and `KernelEndDrain` with
  `Instr::Fence(FenceSpec)`. All five fields populated by the
  orchestrator/walker.
- Stop condition: pod decode unchanged (Paris!!! still). The
  consolidation alone shouldn't fix anything; if it FIXES the bug,
  we have learned something — the fence shape differed by accident
  before and we just fixed it (welcome surprise; document and move
  on).

### Phase 3 — KvLayoutWitness

- Build one `KvLayoutWitness` per K-cache BufId at orchestrate time.
- Thread the witness id onto RopeAppend, AttnDecode, RopeRotate (Q).
- Replace the three independent `kv_layout()` reconstructions with a
  single read off the shared witness.
- Stop condition: pod decode produces SAME bytes for layer-0 K[8]
  and V[8] as before (they're already byte-identical to per-op).

### Phase 4 — KvCacheProducer

- Add `k_producer` / `v_producer` fields on `LoweredOp::AttnDecode`.
- Orchestrator constructs them explicitly. Delete the
  `unwrap_or_else(GmemHandle::new_initial)` fallback from
  `tk_orchestrate.rs`.
- Build-time assert: `SameForwardRopeAppend(j)` references a real
  RopeAppend with matching BufId.
- Stop condition: if Phase 4 panics at build time on Llama-1B, we
  have FOUND a real bug — orchestrator was constructing AttnDecode
  without a producer in some path. Fix and continue.

### Phase 5 — SoftmaxState typestate

- Introduce `SoftmaxState<U, Phase>` typed handles.
- Refactor AttnDecode init/qkt/sv/finalise emit to take and return
  typed states.
- Replace `TkProgram::prelude: String` with `Vec<PreludeDecl>`.
  Persistent-state declarations (`__m_max`, `__l_sum`, `__o_accum`,
  `__p_a{u}`) become `PreludeDecl::PerWarpFloatArray` /
  `PerWarpFloatMatrix` keyed on `SoftmaxStateId`.
- Stop condition: pod decode preserves whatever decode quality
  Phase 4 had. If broken, bisect inside Phase 5.

### Phase 6 — RopeForm

- Add `pub trait RopeForm` and `NeoX` / `Interleaved` zero-sized
  impls.
- Make `RopeRotateOp` / `RopeAppendOp` generic over `F: RopeForm`.
- Codegen body picks the right pair indices via `F::pair_lo` /
  `F::pair_hi`.
- For Llama-1B both Q and K are NeoX — a single `F = NeoX`
  monomorphization.
- Stop condition: byte-identical to Phase 5 on Llama-1B.

### Phase 7 — full tape player cutover

- Walker now emits a `TkTape` instead of a `String`.
- Old `tk_codegen.rs` body-string functions deleted.
- New `tk_player.rs` is the sole emit path.
- Audit the player file: ≤800 lines total, ≤5 lines per arm. If
  over budget, split Instrs.
- Stop condition: pod decode WORKS (Paris coherent) OR we have a new
  failure mode that points concretely at one Instr kind.

### Phase 8 — re-audit

- Re-run the invention audit (the same workflow that produced the
  137 findings).
- Goal: ≤10 verified inventions remain, all rated low bug-likelihood.
- Any high-likelihood invention left = an Instr kind needs more
  fields. Fix and repeat.

---

## 6. Compile-time invariants enforced

After all phases land, the following bug classes are Rust compile
errors instead of runtime divergence:

- Producer/consumer K-cache layout mismatch (§4.1)
- Missing producer for AttnDecode K or V (§4.2)
- Out-of-order softmax recurrence (§4.3)
- Mixed RopeForm in one forward (§4.4)
- Fence emitted with wrong scope/wait_n (§4.0 Fence struct fields)
- Persistent prelude decl emitted inside a loop (§4.3 + PreludeDecl)
- Kernel-end drain forgotten (existing KernelEndDrain witness, kept)
- Slot-mapping value passed to kernel without d2h sync (existing
  Pending<u32> witness, kept)
- Compute body emitted without its required prelude (link
  PreludeDecl.owner to ComputeBodyId at type level — see open Q)

---

## 7. Open questions for the user

These are decisions the plan deliberately leaves open. Please answer
before code lands.

**Q1. KvLayout: id-based vs const-generic.**
Option A (proposed): one `KvLayoutWitness` per BufId, dereferenced
by id at emit time.
Option B: const-generic `KvCacheLayout<NUM_KV_HEADS, HEAD_DIM, ACT_ELEM>`,
monomorphized per arch. Stronger (compile error on mismatch) but
ties the ComputeBody arms to monomorphization.
Recommendation: A for now (less invasive); switch to B if/when we
support multiple K-cache shapes in one kernel.

**Q2. ComputeBody granularity.**
Option A (proposed): one `ComputeBodyId` per architectural primitive
(RmsNormFp32, GemmM1, RopeNeoxK, AttnDecodeQktStep, …) with a
sealed `&'static str` template per id.
Option B: fully expand each body into Instr-level mma/reduce/rsqrt
nodes. Closer to the contract ("every operation an Instr") but
30-100x more Instrs and the templates exist in TK 2.0 already.
Recommendation: A. The TK 2.0 primitives ARE the level of
abstraction; one Instr per primitive call is the right granularity.

**Q3. ForLoop body — inline vs InstrId references.**
Option A (proposed): inline `body: Vec<Instr>`. Simple, no
indirection.
Option B: `body: Vec<InstrId>` referencing a flat instr arena.
Enables sharing (e.g. multiple loops with the same body shape).
Recommendation: A. We don't share bodies today; if we do later,
refactor.

**Q4. WarpRole on every Instr vs role-specific Instr variants.**
Option A (proposed): every Instr that is role-gated carries a
`role: WarpRole` field; the player wraps the emit in
`if (__role == X) { ... }`.
Option B: separate `LoaderLoadAsync`, `ConsumerCompute`,
`StorerStoreAsync` variants per role. Reduces the conditional in the
player but doubles/triples Instr kinds.
Recommendation: A. The role gate is one line in the player — fine.

**Q5. Asm escape — keep or forbid.**
The current `Instr::Asm { role, lines }` escape is a back door for
PTX intrinsics we haven't lifted into typed Instrs. Should we permit
it as a "to-be-lifted" marker, or forbid it (any new PTX MUST become
a typed Instr)?
Recommendation: forbid — every `Asm` we ship is a place we're
admitting the IR is incomplete.

**Q6. Where does the per-arch model lowering produce TkTape?**
The `LoweredOp` → `Vec<Instr>` walker lives where? Today
`tk_lower.rs` produces `TkProgram` (string-shaped). Proposal: the
per-op `lower_*` functions return tape-builder fragments that the
orchestrator concatenates. Walker code shrinks dramatically.

**Q7. Migration cadence — incremental or atomic?**
Phases 0-7 are incremental, each a green commit. Phase 7 is the
final cutover that deletes old code paths. Land incrementally
(proposed), or land atomic on a feature branch and merge in one
shot?
Recommendation: incremental. Each phase is independently reviewable
and bisects cleanly.

**Q8. What does the tape player do for variants we haven't ported
yet (e.g. non-Llama models)?**
During the migration, both old (`tk_codegen.rs`) and new
(`tk_player.rs`) paths exist. The orchestrator routes per-arch.
Llama goes through the new path first; Mistral, Qwen, etc. follow.
Acceptable?

**Q9. Const-generic forms vs runtime fields.**
Several proposed witnesses (`KvLayoutWitness`, `RopeForm`,
`SoftmaxState<U, Phase>`) could be either runtime values or
compile-time const generics / phantoms. Const generics give stronger
invariants but harder ergonomics (every fn signature carries the
generic). Runtime values are looser but easier to refactor.
Per-witness call: which gets compile-time, which gets runtime?

---

## 8. What this plan does NOT cover

- Per-arch differences (Mistral, Qwen, DeepSeek, etc.). Llama-1B
  first; other arches follow once Phase 7 lands and the surface is
  proven.
- Performance tuning (page count, NUM_CONSUMER_WARPS, buckets). The
  redesign is correctness-first; perf knobs become Instr fields and
  are tunable post-cutover.
- TP > 1 (NCCL all-reduce). The current AllReduce instruction
  pattern carries over; the redesign affects only single-rank
  decode emit.
- Quantized weights (GGUF, AWQ, GPTQ, FP8, BNB4). The dense Llama-1B
  path is the redesign target; quant variants stay on the old path
  until Phase 8 audit.

---

## 9. How to act on this doc

Please go through §7 question by question. When the open questions
are resolved, the plan converges to a single reading. Then I land
Phase 0 and we proceed phase-by-phase, with the pod stop-condition
at every phase as the gate.

If any §1-§6 section is wrong (e.g. you want a different `Instr`
kind, a different witness shape, a different player budget), call it
out and I'll revise before Phase 0.
