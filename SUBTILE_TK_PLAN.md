# Subtile-IR for TK 2.0 Megakernel Codegen

Status: design proposal (worktree `worktree-ff-subtile`, branched off
`origin/worktree-pd-wavefront`). Implementation has not started.

## The diagnosis (recap from ff-mega-codegen)

The ff-mega-codegen `MegaIr` is a **whole-op IR** (`I::TkGemm`,
`I::TkAttentionViaCache`, `I::TkFusedQkvRopeCache`, …). The CUDA emitter
*synthesises* everything inside one persistent CTA at emit time:

- which warp role (loader / consumer / storer) does what,
- which one of 13 mbarrier pages to use for each handoff,
- the phase parity for every wait / arrive,
- TMA load offsets, scratch byte offsets, page lifecycle.

That synthesis is the bug class behind the m=1 megakernel deadlock:
codegen has to *predict* every page's lifecycle phase from a coarse-op
walk (`MegaDispatchState::page_rounds`), and one mis-bumped page →
phase parity drifts → mbarrier wait blocks forever.

The user's framing: *"the MegaIr is not a subtile IR. you are filling
in the blanks at codegen time, rather than having the codegen being a
simple walk of the subtile IR."*

## The pd-wavefront SubtileIr — what it already is

`ferrite-wavefront::subtile_ir` is a **per-worker tape IR** with:

- `BufId` / `PipeId` / `FlagId` typed handles (no string lookups,
  no `validate()` cross-reference miss possible at construction).
- `Dispatch`: one resolved kernel call (`pipeline`, `bindings`, `grid`,
  `OpDataflow`). Per-op constructors (`Dispatch::qmv_block`) are
  fixed-arity, so a malformed op is a Rust *type* error.
- `OpDataflow`: typed read/write region set per op kind. The
  `validate()` pass checks every arena read is covered by prior writes
  — use-before-def is caught at compile time.
- `SubtileInstr::{Run, Wait, Signal}`: one entry per op + p2p flags.
  The player (`play()`) is a `match`-walk: never makes a decision.

This is the right level for **inter-CTA / cross-worker** sync (it is
what Metal's `wavefront_player` consumes — one tape per threadgroup,
flags between threadgroups). What it does NOT model is **intra-CTA
warp-specialised** sync: TK 2.0's loader/consumer/storer warps,
mbarrier phase ping-pong, TMA async stages.

## What TK adds — the three things SubtileIr must grow

For one persistent TK CTA the codegen has to know:

1. **Warp role.** Each instruction is owned by exactly one role
   (loader / consumer-N / storer / all). Roles must agree on which
   page they're talking about.
2. **Phase parity.** Every mbarrier handoff is on a typed page slot
   with a `u32` phase that flips `0 ↔ 1` on each round. Loader,
   consumer and storer must read the *same* phase per round.
3. **Page lifecycle.** A page goes `EMPTY → FILLING → FILLED →
   READING → DRAINING → EMPTY`. The IR must encode whose turn it is at
   each transition — `arrive` flips ownership.

The `SubtileInstr` tape is too coarse for this — by the time a `Run`
instruction lands, the warp/page/phase decisions are already implicit
in the kernel body the pipeline points at. Those are exactly the
decisions that today get re-derived (and mis-derived) in codegen.

## The proposal — `tk_warp_ir` as a sibling layer

Add a new module `ferrite-wavefront::tk_warp_ir` (sibling of
`subtile_ir`). It is the IR for the **inside of one persistent CTA**.
SubtileIr is unchanged: it stays the per-worker tape (Metal's wavefront
player still consumes it 1:1).

### Typed primitives

```rust
/// Warp role — every TkInstr carries one. The CUDA emit routes the
/// instruction to a `if (warp_role == X) { ... }` arm; instructions
/// with role `All` are emitted unguarded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarpRole {
    Loader,
    Consumer(u8),     // 0..NUM_CONSUMER_WARPS
    Storer,
    All,              // grid-wide: page init, final sync, etc.
}

/// One of the persistent kernel's mbarrier pages, carrying its
/// CURRENT phase as a const generic. `arrive()` returns
/// `PageHandle<{(P + 1) & 1}>` — phase consistency falls out of the
/// type system (no MegaDispatchState::page_rounds, no asserts, no
/// runtime drift).
#[derive(Clone, Copy, Debug)]
pub struct PageHandle<const PHASE: u32> {
    pub id: u8,        // 0..NUM_PAGES
}

impl<const PHASE: u32> PageHandle<PHASE> {
    pub fn wait(self) -> Self { self }                 // emit: tk20::group<8>::wait(page_done[id], PHASE)
    pub fn arrive(self) -> PageHandle<{ (PHASE + 1) & 1 }> {
        PageHandle { id: self.id }                     // emit: tk20::group<8>::arrive(page_done[id])
    }
}

/// One scratch-bytes region inside the persistent CTA. Carries its
/// length + alignment as const generics. `slice_at::<OFF, LEN>()`
/// returns a sub-handle whose const-generic offsets prove no
/// overlap at compile time (replacing today's runtime byte-offset
/// arithmetic).
#[derive(Clone, Copy, Debug)]
pub struct ScratchSlice<const OFF: u32, const LEN: u32, const ALIGN: u32>;

/// A sub-CTA tile: `[ROWS x COLS]` in a `Page` or `Scratch`, used as
/// the source/dest of TMA + MMA. Const-generic, so a Mma whose
/// operand tiles disagree on K is a Rust compile error.
#[derive(Clone, Copy, Debug)]
pub struct Tile<const ROWS: u32, const COLS: u32, const ELEM_BYTES: u32, Loc> {
    pub loc: Loc,
}
```

### Instruction set

```rust
pub enum TkInstr {
    /// Wait on a page with const-generic phase. Codegen → one
    /// `kittens::group<N>::wait(page_done[id], P)` call.
    Wait { role: WarpRole, page: PageHandle<P> },

    /// Arrive on a page → the page's typed phase advances.
    Arrive { role: WarpRole, page: PageHandle<P> },

    /// TMA load → page (loader role).
    LoadAsync {
        src: BufId,                        // SubtileIr buffer
        src_region: RegionRef,             // already-resolved
        dst: PageHandle<P>,                // page must be FILLING
        dst_tile: Tile<R, C, B, PageLoc>,
    },

    /// MMA `D += A @ B^T` over const-generic shapes. A and B live in
    /// pages or scratch; D in registers.
    MmaAB {
        role: WarpRole,
        a: Tile<M, K, B, _>,
        b: Tile<N, K, B, _>,
        d_reg: RegId,                      // register-tile handle
    },

    /// Store register-tile → scratch / page, with a typed handoff
    /// flag the consumer waits on next round.
    StoreShared { role: WarpRole, src_reg: RegId, dst: ScratchSlice<...> },

    /// TMA store → device (storer role).
    StoreAsync { role: WarpRole, src: PageHandle<P>, dst: BufId, dst_region: RegionRef },

    /// Inline compute fragment in a consumer warp's K-loop. Only used
    /// for ops that don't have a TK 2.0 primitive (residual add, RMS
    /// reduction). The fragment text is *not* synthesised — it is a
    /// const-generic-resolved string fragment from the per-op atom.
    Compute { role: WarpRole, body: ComputeBodyId },

    /// Group-wide sync; one tk20::group<8>::sync() call.
    Sync { role: WarpRole },
}
```

The whole IR has **no `Vec<u32> page_rounds`, no `phase_for_page`,
no `bump_pages`.** A page's phase is `PageHandle<P>::PHASE`, threaded
through `arrive()`. Wait/arrive on the wrong phase is a Rust compile
error.

### Per-op lowering — the atom split

Each high-level op (RmsNorm, RopeRotate, AttnDecode, …) ports as a
function:

```rust
pub fn lower_rmsnorm_to_tk(
    dispatch: &Dispatch,
    pages: &mut PageAllocator,        // the only stateful piece —
    scratch: &mut ScratchAllocator,   //   page-id / scratch-byte assignment
) -> Vec<TkInstr> { ... }
```

The TK persistent megakernel = `concat(lower(op_i) for op_i in tape)`,
where between adjacent ops the page allocator can REUSE pages whose
last `arrive()` advanced their phase (no double-increment bug — phase
moves only when the type system says so).

### Codegen — the literal walk

```rust
fn emit(prog: &TkWarpProgram) -> String {
    for instr in &prog.tape {
        match instr {
            TkInstr::Wait { role, page } => emit_wait(role, page),
            TkInstr::Arrive { role, page } => emit_arrive(role, page),
            TkInstr::LoadAsync { .. } => emit_tma_load(...),
            TkInstr::MmaAB { .. } => emit_mma(...),
            TkInstr::StoreAsync { .. } => emit_tma_store(...),
            TkInstr::Compute { .. } => emit_compute_body(...),
            TkInstr::Sync { role } => emit_sync(role),
        }
    }
}
```

That's it. Codegen makes no decisions. No `MegaDispatchState`. No
`phase_for_page`. No skip-guards (those exist today because synthesis
might disagree with itself between renders). No deadlock class.

## Metal downgrade

Metal has no warp specialisation. The downgrade is a *projection*:

- `WarpRole::Loader` / `Storer` become inline reads/writes in the same
  threadgroup body the consumer runs.
- `Wait` / `Arrive` collapse to one `mk_sync()` per pair.
- Pages collapse to threadgroup-memory regions.
- Phase parity becomes irrelevant (no producer/consumer split).

Metal continues to consume **SubtileIr** unchanged — its
`wavefront_player` tape is per-op-granularity and that's the right
level for it. The new `tk_warp_ir` is *only* used by the CUDA TK
emitter; Metal never sees it.

So the rule "anything you add for TK should be easy to downgrade for
Metal" reduces to: **don't change SubtileIr. Add TK warp-tier as a
sibling.** Which is what this proposal does.

## Phasing

1. **Survey** — done before this doc was written.
2. **Skeleton** — add `tk_warp_ir.rs` with empty types + a one-arm
   match codegen stub.
3. **RmsNorm vertical slice** — port `lower_rmsnorm_to_tk` end-to-end.
   Compare emitted .cu against ff-mega-codegen's RmsNorm canonical
   byte-for-byte.
4. **AttnDecode vertical slice** — the m=1 deadlock canonical. If TK
   warp-tier IR can express it without runtime asserts, the deadlock
   is structurally gone.
5. Remaining ops (Gemm, FusedQkvRopeCache, FusedGateUpSiluMul, Add,
   ResidualWriteback) one at a time.
6. Wire `ferrite-forward-macro`'s mega path to call this codegen
   instead of `ferrite-megakernel::cuda_emit`.
7. Pod E2E on Llama-3.2-1B per `feedback_pod_before_commit`.

## Non-goals

- **No** changes to `subtile_ir.rs`. It is the right shape.
- **No** changes to `mega.rs` (Metal `wavefront_player` encoding).
- **No** changes to the wavefront scheduler / region IR.
- **No** Metal-side warp specialisation (Apple silicon doesn't have it).
- **No** runtime asserts on phase / page / role consistency. Const
  generics or it doesn't ship.
