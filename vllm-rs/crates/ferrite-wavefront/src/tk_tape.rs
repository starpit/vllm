// SPDX-License-Identifier: Apache-2.0
//! `tk_tape` — the flat instruction tape that backs the dumb tape player.
//!
//! Per `SUBTILE_IR_REDESIGN.md` §0/§3: every operation the megakernel
//! performs at runtime — TMA loads/stores, barrier inits/waits/arrives,
//! every fence (`commit_group`, `wait_group`, `threadfence`,
//! `__syncthreads`), persistent-state declarations, loops, kernel-arg
//! declarations, every compute body — is one [`Instr`] in this tape.
//!
//! The dumb player (`tk_player.rs`) is a single `match` over [`Instr`]
//! kinds, ≤5 lines per arm, no ambient state.
//!
//! "Fence" / "drain" is NOT one Instr — it is a SEQUENCE of primitive
//! Instrs (`Syncthreads`, `CommitGroup`, `WaitGroup`, `Threadfence`,
//! `Syncthreads`). The walker enumerates the sequence; one one-line
//! arm per primitive.
//!
//! ## TensorId, not BufId
//!
//! TkTape references source buffers by [`crate::subtile_ir::TensorId`].
//! The v1 `metal_tape::BufId` namespace is no longer wired through the
//! TkTape side of the redesign. The lowering (`lower_subtile_tape_to_tk_tape`)
//! preserves SubtileIR's TensorId for sources; the slot-realization
//! decision (which slot goes to which page, which lives in smem vs
//! gmem) lives at the TkTape optimizer passes, NOT at the source-side
//! identifier.

#![allow(dead_code)]

use crate::subtile_ir::{KvCacheLayout, KvCacheProducer, KvCacheShape, TensorId};
use std::marker::PhantomData;

// NUKED: RopeForm trait + NeoX / Interleaved markers + RopeFormTag
// + GemmK witness + AccumKind + RopeSide + Instr::rope_rotate
// constructor + _RopeFormBridge — all supported the architectural
// Compute Instrs that emitted invented `kittens::ops::*` calls.
// They reappear scoped to the actual TK 2.0 primitives that need
// them (the rope-form invariant lives at the SubtileIR level via
// `subtile_ir::RopeForm`, which is the canonical witness; the
// TkTape-side duplicate was always redundant).

// ── Substrate constants (re-exported from tk_warp_ir for now) ───────

pub const NUM_PAGES: u32 = 13;
pub const PAGE_SIZE: u32 = 16384;
pub const SCRATCH_BYTES: u32 = 1024;
pub const NUM_CONSUMER_WARPS: u8 = 16;
pub const NUM_SERVICE_WARPS: u8 = 4;
pub const NUM_WARPS: u8 = NUM_SERVICE_WARPS + NUM_CONSUMER_WARPS;

// ── The tape ────────────────────────────────────────────────────────

/// One persistent-CTA tape — produced by the walker, consumed by the
/// dumb player. Each Instr maps 1:1 to a TK 2.0 / CUDA primitive call.
#[derive(Debug, Default, Clone)]
pub struct TkTape {
    /// Kernel-arg declarations in ABI order. The C++ kernel signature
    /// is built from this; the dispatcher's runtime `bufs[]` /
    /// `u32_args[]` indices follow this order.
    pub kernel_args: Vec<KernelArg>,

    /// Persistent-state declarations emitted at function-scope before
    /// the instruction stream. Replaces the freeform
    /// `TkProgram::prelude: String` with one typed decl per entry.
    pub prelude: Vec<PreludeDecl>,

    /// The instruction stream — flat, role-gated per-Instr. Multi-step
    /// CUDA sequences (cross-op fence, kernel-end drain) are SEQUENCES
    /// of primitive Instrs in this Vec, never a single fat Instr.
    pub instrs: Vec<Instr>,

    /// KvCacheLayout witness table — interned per K/V cache `TensorId`.
    /// `Instr::RopeRotate.kv_layout: KvLayoutId` and AttnDecode Instrs'
    /// kv_layout field index into this Vec. Per plan §2 line 88 the
    /// witness "propagates; consumer reads via single-source method"
    /// — see [`TkTape::kv_layout`].
    pub kv_layouts: Vec<KvLayoutEntry>,
}

impl TkTape {
    /// Single-source read accessor for [`KvLayoutId`] — per plan §2
    /// line 88, the consumer reads the `KvCacheLayout` witness via
    /// this one method.
    pub fn kv_layout(&self, id: KvLayoutId) -> &KvLayoutEntry {
        &self.kv_layouts[id.0 as usize]
    }
}

// ── kernel-arg declarations ─────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct KernelArg {
    pub name: KernelArgName,
    pub ty: KernelArgTy,
}

/// Stable kernel-side identifier. Sealed at the variant layer so a
/// lowering can't pass an arbitrary `String` — the only path to a name
/// is through one of these variants whose canonical text matches the
/// runtime expectations of the dispatcher (paris invariant
/// `u32-args-name-matches-runtime-string`).
#[derive(Debug, Clone)]
pub enum KernelArgName {
    Fixed(&'static str),
}

#[derive(Debug, Clone)]
pub enum KernelArgTy {
    /// `uint32_t` runtime arg. `source` ties this u32 to a typed
    /// per-call value built by the dispatcher.
    U32 { source: U32Source },
    /// Pointer to a typed source tensor (weight / activation / kv cache /
    /// cos-sin cache). Identified by SubtileIR `TensorId` — the lowering
    /// preserves source identity from the IR.
    BufPtr(TensorId),
}

/// Sealed source of a runtime u32 — there is one variant per kernel
/// u32 ZST in the runtime (`NumKvPagesSym`, `DecodePositionSym`,
/// `DecodeSlotSym`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum U32Source {
    NumKvPages,
    DecodePosition,
    DecodeSlot,
}

/// Index into [`TkTape::kernel_args`]. The Instr stream references
/// kernel args by id, never by name string — name resolution lives
/// in the player.
/// Sealed per §2: inner field is `pub(crate)` — external code can
/// neither construct nor read this id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelArgRef(pub(crate) u16);

// ── prelude declarations ────────────────────────────────────────────

/// One persistent-state declaration emitted before the instruction
/// stream. Replaces the freeform `TkProgram::prelude: String` —
/// every persistent state is typed and owned by exactly one
/// Compute-emit (so an orphan decl is a build-time check).
#[derive(Debug, Clone)]
pub enum PreludeDecl {
    /// Per-warp `float[len]` — e.g. softmax `__l_sum`.
    PerWarpFloatArray {
        name: PreludeName,
        len: u32,
        owner: ComputeOwner,
    },
    /// Per-warp `float[rows][cols]` — e.g. softmax `__o_accum`.
    PerWarpFloatMatrix {
        name: PreludeName,
        rows: u32,
        cols: u32,
        owner: ComputeOwner,
    },
    /// `void* page_buf[NUM_PAGES]` aliasing for a specific page slot.
    SmemTilePtr { name: PreludeName, page: PageId },
    /// Alias a kernel-arg identifier so the body can reference a
    /// stable name regardless of ABI position.
    KernelArgAlias { name: PreludeName, arg: KernelArgRef },
}

/// Sealed per §2: inner field is `pub(crate)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreludeName(pub(crate) u32);

/// Identifier for the [`Instr`] compute that owns a prelude decl.
/// Connecting decl ↔ body at type level prevents orphan decls.
/// Sealed per §2: inner field is `pub(crate)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeOwner(pub(crate) u32);

// ── instruction stream ──────────────────────────────────────────────

/// One instruction the kernel executes. Each variant maps 1:1 to a
/// TK 2.0 / CUDA primitive call. Compute variants alone expand to
/// a fixed `&'static str` template with field-substitution by the
/// dumb player.
#[derive(Debug, Clone)]
pub enum Instr {
    // ── Sync primitives — every variant carries a WarpRole so the
    //    player has zero ambient lookups. Per plan §3 step 8 + memory
    //    feedback_tk_player_one_call_per_arm: one Instr variant per
    //    architectural primitive (no inner-match dispatch in the
    //    player).

    /// `__syncthreads()` — full CTA.
    SyncthreadsCta { role: WarpRole },
    /// `kittens::group<N>::sync()` — N-warp group sync.
    /// `kittens::group<N>::sync()` — warp-group barrier sync.
    ///
    /// `width` is the sealed [`GroupWidthTag`] (recovered from a
    /// `GroupWidth<N>` typed witness). N is restricted at construction
    /// to the sealed set `{1, 4, 16, 20}`; an arbitrary u32 cannot
    /// reach this variant.
    SyncthreadsGroup { width: GroupWidthTag, role: WarpRole },

    /// `__threadfence_block()` — CTA-scope.
    ThreadfenceBlock { role: WarpRole },
    /// `__threadfence()` — device-scope.
    ThreadfenceDevice { role: WarpRole },
    /// `__threadfence_system()` — system-scope.
    ThreadfenceSystem { role: WarpRole },

    /// `kittens::group<1>::tma::store_commit_group()`.
    CommitGroupBulk { role: WarpRole },

    /// `kittens::group<1>::tma::store_async_wait<N>()`. `n=0` drains
    /// all groups.
    WaitGroupBulk { n: u32, role: WarpRole },

    // ── Page barriers — TK 2.0 mbarrier handshake ────────────────

    /// `mbarrier.init` for a named page barrier. `count` is the
    /// expected arrival count.
    BarrierInit {
        page_id: PageId,
        kind: PageBarrier,
        count: u32,
    },

    /// Wait on `page_<kind>[page_id]` at compile-time parity 0. Per
    /// plan §2 row "Phase (parity)": const-generic split variants on
    /// TkTape, never as a u8 field. Sibling variant:
    /// [`Instr::PageBarrierWaitStaticP1`] for parity 1.
    PageBarrierWaitStaticP0 {
        page_id: PageId,
        kind: PageBarrier,
        role: WarpRole,
    },
    /// Wait on `page_<kind>[page_id]` at compile-time parity 1.
    PageBarrierWaitStaticP1 {
        page_id: PageId,
        kind: PageBarrier,
        role: WarpRole,
    },

    /// Wait on `page_<kind>[page_id]` at loop-carried parity
    /// `(v<var> + 0) & 1u`. Per plan §2 row "Phase (parity)": split
    /// per start-parity rather than carrying `start: u8` as a field.
    PageBarrierWaitLoopStart0 {
        page_id: PageId,
        kind: PageBarrier,
        var: LoopVarId,
        role: WarpRole,
    },
    /// Wait on `page_<kind>[page_id]` at loop-carried parity
    /// `(v<var> + 1) & 1u`.
    PageBarrierWaitLoopStart1 {
        page_id: PageId,
        kind: PageBarrier,
        var: LoopVarId,
        role: WarpRole,
    },

    /// `kittens::group<1>::arrive(<barrier>[page_id])`.
    PageBarrierArrive {
        page_id: PageId,
        kind: PageBarrier,
        role: WarpRole,
    },

    /// `if ((parity_var & 1u) == 0u) {
    /// kittens::group<1>::arrive(<barrier>[page_id]); }`. Used by
    /// the parity-correction phantom round after a runtime-iter-count
    /// for_loop.
    ArriveIfRuntimeEven {
        page_id: PageId,
        kind: PageBarrier,
        parity_var: KernelArgRef,
        role: WarpRole,
    },

    // ── Memory ops ───────────────────────────────────────────────

    /// `kittens::group<1>::tma::expect_bytes` + `tma::load_async`.
    LoadAsync(LoadSpec),

    /// `kittens::group<1>::tma::store_async` (raw-bulk).
    StoreAsync(StoreSpec),

    /// Typed-descriptor TMA store — `tma::store_async<NORMAL>(arg<dst>,
    /// ...)`. Pushed by the orchestrator's descriptor-rewrite pass for
    /// tensors declared as descriptor-bound on the kernel-arg side.
    StoreAsyncTyped {
        dst_page: PageId,
        dst_tensor: TensorId,
        tile_type: TileType,
        role: WarpRole,
    },

    // ── Compute — one variant per TK 2.0 primitive. ─────────────────
    //
    // Per plan §1 line 64-65 + INVIOLABLE feedback_tk_2_0_only /
    // feedback_tk20_primitives_first / feedback_tk_player_one_call_per_arm:
    // each variant maps 1:1 to a real TK 2.0 callable in
    // `third_party/thunderkittens/include/`. The architectural ops
    // (RmsNorm, MatmulTile, SiluMul, RopeRotate*, AttnDecode_*) live
    // upstream in `subtile_ir::SubOp` and decompose into sequences
    // of these primitive Instrs at `lower_subtile_tape_to_tk_tape`.
    //
    // Implementation order per SUBTILE_TK20_DECOMP.md: ShTileMul
    // first (smallest, no register tiles, no scalars, no TMA).

    /// Pairwise multiply two shared tiles into a third — TK 2.0
    /// primitive `kittens::group<N>::mul(dst, lhs, rhs)` at
    /// `ops/group/shared/tile/maps.cuh:306`. Used by SubOp::Elementwise(Mul)
    /// and (eventually) SiluMul / RmsNorm decompositions.
    ///
    /// `width` is a sealed [`GroupWidthTag`]; the only constructors
    /// for it require a [`GroupWidth<N>`] where `GroupWidth<N>:
    /// ComputeWidth` — that is, N ∈ {4, 16}. A wrong-width construction
    /// (e.g. per-warp `GroupWidth::<1>`) is a Rust compile error,
    /// per `feedback_ff_subtile_compile_time_inviolable`.
    ShTileMul {
        lhs: PageId,
        rhs: PageId,
        dst: PageId,
        width: GroupWidthTag,
    },

    /// Inert marker the orchestrator emits at the start of an op
    /// when `EmitOpts::debug_handshake` is on.
    DebugOpBeginMarker { op_index: u32 },

    // ── Control flow — flat: ForLoopOpen* opens the brace,
    //    body Instrs follow, ForLoopClose closes it. The tape is
    //    truly linear — no nested Vec<Instr>, no player recursion.

    /// `for (uint v<var> = 0; v<var> < <n>u; ++v<var>) {`
    ForLoopOpenConst { var: LoopVarId, n: u32 },
    /// `for (uint v<var> = 0; v<var> < a<arg>; ++v<var>) {`
    ForLoopOpenKernelArg { var: LoopVarId, arg: KernelArgRef },
    /// `}` — closes the matching ForLoopOpen{Const,KernelArg}.
    ForLoopClose { var: LoopVarId },
}

// ── instruction field types ─────────────────────────────────────────
//
// SyncScope/FenceScope/CommitKind enums have been folded into their
// owning Instr variants (SyncthreadsCta/Group, ThreadfenceBlock/Device/
// System, CommitGroupBulk, WaitGroupBulk) per plan §3 step 8 + memory
// feedback_tk_player_one_call_per_arm — one Instr per architectural
// primitive, no inner-match dispatch in the player.

/// Which warp role inside the persistent CTA owns an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarpRole {
    Loader,
    Storer,
    Consumer(u8),
    AllConsumers,
    All,
}

// ── Sealed const-generic `GroupWidth<N>` + `ComputeWidth` marker ────
//
// Per `feedback_ff_subtile_compile_time_inviolable`: every numeric
// proof the codegen relies on must propagate end-to-end as a Rust
// const-generic with `where` clauses, NOT a runtime field. The
// `kittens::group<N>::*` template parameter is one such proof.
//
// `GroupWidth<const N: usize>` is sealed (private constructor) — the
// only way to get one is a constant in `{1, 4, 16, 20}`. Compute Instr
// constructors then take `GroupWidth<N>` and bound it on
// `ComputeWidth`, which is impl'd ONLY for the legal compute widths
// (4 = warpgroup, 16 = AllConsumers). Constructing a Compute Instr
// with `GroupWidth::<1>::PER_WARP` is a Rust compile error.

mod group_width_sealed {
    pub trait Sealed {}
}

/// Sealed const-generic carrier for the `kittens::group<N>` template
/// parameter. The only way to get one is via the per-N associated
/// constants below — `GroupWidth::<N>::*` for N ∈ {1, 4, 16, 20}.
/// Construction of any other `N` is a compile error (no `Sealed` impl).
#[derive(Debug, Clone, Copy)]
pub struct GroupWidth<const N: usize>(PhantomData<()>);

impl group_width_sealed::Sealed for GroupWidth<1> {}
impl group_width_sealed::Sealed for GroupWidth<4> {}
impl group_width_sealed::Sealed for GroupWidth<16> {}
impl group_width_sealed::Sealed for GroupWidth<20> {}

impl GroupWidth<1> {
    /// Per-warp scope: `kittens::group<1>::*`. Used by Loader / Storer
    /// / Consumer-per-warp Instrs (TMA, mbarrier, sync). NOT a
    /// `ComputeWidth` — constructing `Instr::sh_tile_mul` with this
    /// is a Rust compile error.
    pub const PER_WARP: Self = Self(PhantomData);
}
impl GroupWidth<4> {
    /// Warpgroup scope: `kittens::group<4>::*`. The Hopper WGMMA
    /// width — required by `mma_AB`, `mma_ABt`. Implements
    /// [`ComputeWidth`].
    pub const WARPGROUP: Self = Self(PhantomData);
}
impl GroupWidth<16> {
    /// All-consumers scope: `kittens::group<NUM_CONSUMER_WARPS>::*`.
    /// The default Compute width — RmsNorm / SiluMul / Elementwise /
    /// SumReduce all bind to this. Implements [`ComputeWidth`].
    pub const ALL_CONSUMERS: Self = Self(PhantomData);
}
impl GroupWidth<20> {
    /// Whole-CTA scope: `kittens::group<NUM_WARPS>::*`. Used by
    /// CTA-wide sync / fence Instrs. NOT a `ComputeWidth`.
    pub const ALL: Self = Self(PhantomData);
}

impl<const N: usize> GroupWidth<N>
where
    GroupWidth<N>: group_width_sealed::Sealed,
{
    /// Erase the const-generic into the runtime [`GroupWidthTag`] that
    /// the Instr variant carries. The N is recovered as `tag.n()`
    /// for codegen, but construction of the tag is gated by the
    /// type-level `Sealed` bound.
    pub const fn tag(self) -> GroupWidthTag {
        GroupWidthTag(N as u8)
    }
}

/// Sealed marker: implemented ONLY for compute-eligible group widths.
/// `feedback_compile_time_or_garbage`: a wrong-width Compute Instr is
/// a Rust compile error, never a runtime panic / debug_assert.
///
/// Compute width must be a warpgroup (`GroupWidth<4>`) or the entire
/// AllConsumers set (`GroupWidth<16>`). Per-warp (`<1>`) and whole-CTA
/// (`<20>`) are rejected at type level.
///
/// # Compile-fail proof — per-warp width rejected
///
/// ```compile_fail
/// use ferrite_wavefront::tk_tape::{GroupWidth, Instr, PageId};
/// // `Instr::sh_tile_mul` is `pub(crate)` — we use the public `Instr`
/// // type and the public `GroupWidth` constants. Replace this with
/// // any public Compute Instr constructor that takes `GroupWidth<N>:
/// // ComputeWidth`. The point is: `GroupWidth<1>` cannot satisfy the
/// // `ComputeWidth` bound — rustc rejects.
/// fn _wants_compute<W: ferrite_wavefront::tk_tape::ComputeWidth>(_: W) {}
/// _wants_compute(GroupWidth::<1>::PER_WARP);
/// ```
///
/// # Pass — warpgroup width accepted
///
/// ```
/// use ferrite_wavefront::tk_tape::GroupWidth;
/// fn _wants_compute<W: ferrite_wavefront::tk_tape::ComputeWidth>(_: W) {}
/// _wants_compute(GroupWidth::<4>::WARPGROUP);
/// _wants_compute(GroupWidth::<16>::ALL_CONSUMERS);
/// ```
pub trait ComputeWidth: group_width_sealed::Sealed {}
impl ComputeWidth for GroupWidth<4> {}
impl ComputeWidth for GroupWidth<16> {}

/// Runtime carrier for the const-generic `GroupWidth<N>` after type
/// erasure into [`Instr`]. Field is `pub(crate)` (sealed); the only
/// public constructor is [`GroupWidth::tag`], which requires the
/// const-generic typed witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupWidthTag(pub(crate) u8);

impl GroupWidthTag {
    /// The recovered `N` for `kittens::group<N>::*` emit.
    pub const fn n(&self) -> u32 {
        self.0 as u32
    }
}

// ── Sealed `TileDtype` + typed `SmemTileId<ROWS, COLS, T>` ──────────
//
// Per `feedback_ff_subtile_compile_time_inviolable`: a `PageId` is a
// runtime u8 index — it carries no shape/dtype, so an Instr taking
// three `PageId`s cannot prove its operands have matching shape at
// rustc time. The TK 2.0 `kittens::group<N>::mul(T &dst, const T &lhs,
// const U &rhs)` template requires `dst` and `lhs` to have the same
// type; mismatch is currently caught only at C++ instantiation
// (downstream of codegen), violating the inviolable rule.
//
// `SmemTileId<const ROWS: usize, const COLS: usize, T: TileDtype>` is
// a sealed phantom-typed wrapper around a `PageId`. Its const-generic
// shape (ROWS × COLS) and dtype (T, sealed) are propagated through the
// `Instr::sh_tile_mul` constructor — shape or dtype mismatch across
// lhs/rhs/dst is a rustc unification failure.
//
// The substrate today is uniform `__shared__ kittens::st_bf<128,128>
// page_buf[NUM_PAGES]`, so the only legal SmemTileId is
// `<128, 128, Bf16>`. When non-uniform pools land (e.g. a separate
// vector pool of `st_bf<1, 128>`), each pool will mint its own typed
// SmemTileId, and the type system will refuse a vector tile where a
// square tile is required.

mod tile_dtype_sealed {
    pub trait Sealed {}
}

/// Sealed marker for TK 2.0 tile element types. Each impl carries
/// the `kittens::st_<alias>` template alias used by the emitter:
///
/// ```c++
/// // third_party/thunderkittens/include/types/shared/st.cuh:313
/// using st_bf = st<bf16,  _height, _width, _swizzle, _swizzle_bytes>;
/// using st_fl = st<float, _height, _width, _swizzle, _swizzle_bytes>;
/// using st_hf = st<half,  _height, _width, _swizzle, _swizzle_bytes>;
/// ```
///
/// The alias suffix (e.g. `bf` for bf16) — NOT the underlying scalar
/// name — is what concatenates after `kittens::st_` in emit. Hence
/// `Bf16::ST_ALIAS_SUFFIX = "bf"`, not `"bf16"`.
pub trait TileDtype: tile_dtype_sealed::Sealed + Copy {
    /// Suffix for the `kittens::st_<suffix><ROWS, COLS>` template
    /// alias defined in `types/shared/st.cuh:313`.
    const ST_ALIAS_SUFFIX: &'static str;
    /// Bytes per element. Used by [`SmemTileSpec`] to recover the
    /// runtime `elem_bytes` value for emit (TMA descriptor sizing,
    /// `expect_bytes` arithmetic) without storing it as a separate
    /// runtime field.
    const ELEM_BYTES: u32;
}

/// `kittens::bf16` — Hopper bfloat16. The only dtype currently bound
/// to a page in the substrate. Aliased as `kittens::st_bf<…>`.
#[derive(Clone, Copy, Debug)]
pub struct Bf16;
impl tile_dtype_sealed::Sealed for Bf16 {}
impl TileDtype for Bf16 {
    const ST_ALIAS_SUFFIX: &'static str = "bf";
    const ELEM_BYTES: u32 = 2;
}

/// Typed shared-memory tile handle: a [`PageId`] paired with type-level
/// shape (`ROWS × COLS`) and dtype (`T: TileDtype`, sealed).
///
/// Two `SmemTileId`s with different const-generics or dtypes are
/// **different Rust types**. A constructor like [`Instr::sh_tile_mul`]
/// that requires `(lhs, rhs, dst): (SmemTileId<R,C,T>, SmemTileId<R,C,T>,
/// SmemTileId<R,C,T>)` rejects shape or dtype mismatch at rustc time
/// (E0308 type mismatch on const-generic).
///
/// Construction (`from_page`) is `pub(crate)` so only the lowerer —
/// which owns the page→shape mapping — can mint these; downstream
/// consumers receive them and pass them through unchanged.
///
/// # Compile-fail proof — shape mismatch (COLS) rejected
///
/// ```compile_fail
/// use ferrite_wavefront::tk_tape::{Bf16, SmemTileId, TileDtype};
/// fn _all_same<const R: usize, const C: usize, T: TileDtype>(
///     _a: SmemTileId<R, C, T>,
///     _b: SmemTileId<R, C, T>,
///     _c: SmemTileId<R, C, T>,
/// ) {}
/// // `unreachable!()` types as `!` and coerces — the compile error
/// // we want is the const-generic unification at the call site.
/// let a: SmemTileId<128, 128, Bf16> = unreachable!();
/// let b: SmemTileId<128,  64, Bf16> = unreachable!();
/// let c: SmemTileId<128, 128, Bf16> = unreachable!();
/// _all_same(a, b, c);  // ← rustc rejects: C=128 vs C=64
/// ```
///
/// # Compile-fail proof — shape mismatch (ROWS) rejected
///
/// ```compile_fail
/// use ferrite_wavefront::tk_tape::{Bf16, SmemTileId, TileDtype};
/// fn _all_same<const R: usize, const C: usize, T: TileDtype>(
///     _a: SmemTileId<R, C, T>,
///     _b: SmemTileId<R, C, T>,
/// ) {}
/// let a: SmemTileId<128, 128, Bf16> = unreachable!();
/// let b: SmemTileId< 64, 128, Bf16> = unreachable!();
/// _all_same(a, b);  // ← rustc rejects: R=128 vs R=64
/// ```
///
/// # Pass — matched shape & dtype accepted (type-check only)
///
/// ```
/// use ferrite_wavefront::tk_tape::{Bf16, SmemTileId, TileDtype};
/// fn _all_same<const R: usize, const C: usize, T: TileDtype>(
///     _a: SmemTileId<R, C, T>,
///     _b: SmemTileId<R, C, T>,
///     _c: SmemTileId<R, C, T>,
/// ) {}
/// // Type-check the call without running it — the doctest passes if
/// // rustc accepts the unification. `if false` keeps the body dead.
/// fn _proof() {
///     if false {
///         let a: SmemTileId<128, 128, Bf16> = unreachable!();
///         let b: SmemTileId<128, 128, Bf16> = unreachable!();
///         let c: SmemTileId<128, 128, Bf16> = unreachable!();
///         _all_same(a, b, c);
///     }
/// }
/// _proof();
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SmemTileId<const ROWS: usize, const COLS: usize, T: TileDtype> {
    page: PageId,
    _marker: PhantomData<fn() -> T>,
}

impl<const ROWS: usize, const COLS: usize, T: TileDtype> SmemTileId<ROWS, COLS, T> {
    /// Mint a typed tile handle for `page`. Caller asserts (by choice
    /// of the const-generic instantiation site) that `page` indexes a
    /// `kittens::st_<T::NAME><ROWS, COLS>`-typed shared buffer.
    /// `pub(crate)` so only the lowerer can mint these.
    pub(crate) const fn from_page(page: PageId) -> Self {
        Self {
            page,
            _marker: PhantomData,
        }
    }

    /// Recover the runtime [`PageId`] for codegen / debug / Instr
    /// field storage.
    pub const fn page(&self) -> PageId {
        self.page
    }

    /// Const accessors for the shape, exposed for diagnostics and
    /// downstream type-level computation.
    pub const fn rows() -> usize {
        ROWS
    }
    pub const fn cols() -> usize {
        COLS
    }
}

/// Typed shape-only witness for TMA-descriptor sizing.
/// [`LoadSpec`] / [`StoreSpec`] used to carry a raw [`TileShape`] —
/// any caller could pass arbitrary `(rows, cols, elem_bytes)` and a
/// stringly-typed mismatch versus the actual page would surface as a
/// silent TMA-descriptor corruption rather than a Rust error.
///
/// `SmemTileSpec<const ROWS, const COLS, T: TileDtype>` is a sealed
/// phantom-typed witness mirroring [`SmemTileId`] but with no
/// associated [`PageId`] (the spec describes a tile *shape*, not an
/// occupant). Construction (`from_shape`) is `pub(crate)`; the
/// constructor `debug_assert!`s the runtime shape matches the const
/// generics as a boundary belt-and-suspenders. The runtime
/// [`TileShape`] is recovered by [`SmemTileSpec::shape`] for the
/// erased Instr field.
///
/// # Compile-fail proof — shape/dtype mismatch via shared-bound helper
///
/// ```compile_fail
/// use ferrite_wavefront::tk_tape::{Bf16, SmemTileSpec, TileDtype};
/// fn _all_same<const R: usize, const C: usize, T: TileDtype>(
///     _a: SmemTileSpec<R, C, T>,
///     _b: SmemTileSpec<R, C, T>,
/// ) {}
/// let a: SmemTileSpec<128, 128, Bf16> = unreachable!();
/// let b: SmemTileSpec<128,  64, Bf16> = unreachable!();
/// _all_same(a, b);  // ← rustc rejects: COLS=128 vs COLS=64
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SmemTileSpec<const ROWS: usize, const COLS: usize, T: TileDtype> {
    _marker: PhantomData<fn() -> T>,
}

impl<const ROWS: usize, const COLS: usize, T: TileDtype> SmemTileSpec<ROWS, COLS, T> {
    /// Mint a typed spec from a runtime shape. `pub(crate)`: only
    /// the lowerer (which owns the page-shape mapping) can mint
    /// these. `debug_assert!`s shape conformance — a release-mode
    /// mismatch is a typed-witness lie, but the const-generics are
    /// what flow into emit and downstream type-checks.
    pub(crate) fn from_shape(shape: TileShape) -> Self {
        debug_assert_eq!(
            shape.rows, ROWS as u32,
            "SmemTileSpec<{ROWS},_,_>::from_shape: rows mismatch (got {})",
            shape.rows,
        );
        debug_assert_eq!(
            shape.cols, COLS as u32,
            "SmemTileSpec<_,{COLS},_>::from_shape: cols mismatch (got {})",
            shape.cols,
        );
        debug_assert_eq!(
            shape.elem_bytes,
            T::ELEM_BYTES,
            "SmemTileSpec<_,_,T>::from_shape: elem_bytes mismatch (got {})",
            shape.elem_bytes,
        );
        Self {
            _marker: PhantomData,
        }
    }

    /// Recover the runtime [`TileShape`] for the erased Instr field.
    /// All three components come from the const-generics and the
    /// sealed `T::ELEM_BYTES`.
    pub const fn shape(&self) -> TileShape {
        TileShape {
            rows: ROWS as u32,
            cols: COLS as u32,
            elem_bytes: T::ELEM_BYTES,
        }
    }

    pub const fn rows() -> usize {
        ROWS
    }
    pub const fn cols() -> usize {
        COLS
    }
}

/// Sealed per §2: inner field is `pub(crate)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub(crate) u8);

/// Which TK 2.0 mbarrier of a page slot a wait/arrive talks to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageBarrier {
    Ready,
    Done,
    Consumed,
}

/// Sealed per §2: inner field is `pub(crate)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoopVarId(pub(crate) u32);

/// Sealed identifier for a piece of online-softmax recurrence state.
/// Sealed per §2: inner field is `pub(crate)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SoftmaxStateId(pub(crate) u32);

// LoopCount enum has been folded into ForLoopOpenConst { var, n } /
// ForLoopOpenKernelArg { var, arg } per plan §3 step 8.

// ParityExpr enum has been folded into PageBarrierWaitStatic { parity: u8 }
// / PageBarrierWaitLoop { var, start } per plan §3 step 8 — one Instr per
// architectural primitive, no inner-match dispatch in the player.

/// Byte-offset expression for TMA load/store source/dest. Stored as
/// a pre-computed CUDA expression string, baked at tape-build time
/// so the player emits literally — no inner-match dispatch / no
/// emit-time arithmetic. Sealed: only the per-target lowering can
/// construct one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteOffset(String);

impl ByteOffset {
    /// Constant byte offset: `<c>u`.
    pub fn from_const(c: u64) -> Self {
        Self(format!("{c}u"))
    }
    /// Loop-linear byte offset: `(<base>u + v<var> * <stride>u)`.
    pub fn linear_loop(var: LoopVarId, stride: u64, base: u64) -> Self {
        Self(format!("({base}u + v{} * {stride}u)", var.0))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileShape {
    pub rows: u32,
    pub cols: u32,
    pub elem_bytes: u32,
}

/// Sealed CUDA tile-type spelling for [`Instr::StoreAsyncTyped`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileType(String);

impl TileType {
    pub(crate) fn from_layout(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// TMA-load arguments. Fields are `pub` for player read-access; the
/// only constructors are [`LoadSpec::new`] and the existing pattern
/// of struct-literal construction inside the crate (`pub(crate)`
/// would block both at once). The compile-time gate on `tile`
/// shape/dtype lives at [`LoadSpec::new`].
#[derive(Debug, Clone)]
pub struct LoadSpec {
    pub dst_page: PageId,
    pub src_tensor: TensorId,
    pub byte_off: ByteOffset,
    pub tile: TileShape,
    pub role: WarpRole,
    /// Which page barrier `expect_bytes` arms.
    pub barrier_page: PageId,
}

impl LoadSpec {
    /// Construct a [`LoadSpec`] from a typed [`SmemTileSpec<ROWS,
    /// COLS, T>`] witness. The const-generics + sealed `T::ELEM_BYTES`
    /// are the source of truth for the runtime [`TileShape`] field
    /// — callers cannot pass arbitrary `(rows, cols, elem_bytes)`.
    /// Per `feedback_ff_subtile_compile_time_inviolable`.
    pub(crate) fn new<const ROWS: usize, const COLS: usize, T: TileDtype>(
        dst_page: PageId,
        src_tensor: TensorId,
        byte_off: ByteOffset,
        tile: SmemTileSpec<ROWS, COLS, T>,
        role: WarpRole,
        barrier_page: PageId,
    ) -> Self {
        Self {
            dst_page,
            src_tensor,
            byte_off,
            tile: tile.shape(),
            role,
            barrier_page,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StoreSpec {
    pub src_page: PageId,
    pub dst_tensor: TensorId,
    pub byte_off: ByteOffset,
    pub tile: TileShape,
    pub role: WarpRole,
}

impl StoreSpec {
    /// Construct a [`StoreSpec`] from a typed [`SmemTileSpec<ROWS,
    /// COLS, T>`] witness. See [`LoadSpec::new`].
    pub(crate) fn new<const ROWS: usize, const COLS: usize, T: TileDtype>(
        src_page: PageId,
        dst_tensor: TensorId,
        byte_off: ByteOffset,
        tile: SmemTileSpec<ROWS, COLS, T>,
        role: WarpRole,
    ) -> Self {
        Self {
            src_page,
            dst_tensor,
            byte_off,
            tile: tile.shape(),
            role,
        }
    }
}

/// Phase parity for `Instr::wait_static` / `Instr::wait_loop`
/// constructors. Per plan §2 row "Phase (parity)": parity is a
/// sealed enum, never a `u8` field; this enum makes the constructor
/// `match` exhaustive without a `_ =>` wildcard (plan §4 line 200).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parity {
    P0,
    P1,
}

/// Index into the tape's `Vec<KvLayoutEntry>`.
/// Sealed per §2: inner field is `pub(crate)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvLayoutId(pub(crate) u32);

// ── Sealed constructors ─────────────────────────────────────────────

impl Instr {
    pub(crate) fn syncthreads_cta(role: WarpRole) -> Self {
        Self::SyncthreadsCta { role }
    }

    /// Construct a [`Instr::SyncthreadsGroup`] from a typed
    /// [`GroupWidth<N>`] witness. `N` is restricted to the sealed
    /// set `{1, 4, 16, 20}` — passing an arbitrary integer is a
    /// rustc error (no `Sealed` impl), per
    /// `feedback_ff_subtile_compile_time_inviolable`.
    pub(crate) fn syncthreads_group<const N: usize>(role: WarpRole, width: GroupWidth<N>) -> Self
    where
        GroupWidth<N>: group_width_sealed::Sealed,
    {
        Self::SyncthreadsGroup {
            width: width.tag(),
            role,
        }
    }

    pub(crate) fn threadfence_device(role: WarpRole) -> Self {
        Self::ThreadfenceDevice { role }
    }

    pub(crate) fn commit_bulk(role: WarpRole) -> Self {
        Self::CommitGroupBulk { role }
    }

    pub(crate) fn wait_bulk(role: WarpRole, n: u32) -> Self {
        Self::WaitGroupBulk { n, role }
    }

    /// Per plan §2 row "Phase (parity)": runtime→const dispatch
    /// happens here, once at the constructor's `match` site. Each
    /// arm produces a const-generic split variant; the Instr never
    /// carries `parity` as a u8 field. Per plan §4 lines 199-200
    /// (no `_ =>` arms, no `unwrap_or_else`): the `parity` parameter
    /// is the sealed [`Parity`] enum so `match` is exhaustive
    /// without a wildcard arm.
    pub(crate) fn wait_static(
        page: PageId,
        kind: PageBarrier,
        parity: Parity,
        role: WarpRole,
    ) -> Self {
        match parity {
            Parity::P0 => Self::PageBarrierWaitStaticP0 { page_id: page, kind, role },
            Parity::P1 => Self::PageBarrierWaitStaticP1 { page_id: page, kind, role },
        }
    }

    /// As [`Instr::wait_static`] but for runtime loop-carried parity.
    pub(crate) fn wait_loop(
        page: PageId,
        kind: PageBarrier,
        var: LoopVarId,
        start_parity: Parity,
        role: WarpRole,
    ) -> Self {
        match start_parity {
            Parity::P0 => Self::PageBarrierWaitLoopStart0 { page_id: page, kind, var, role },
            Parity::P1 => Self::PageBarrierWaitLoopStart1 { page_id: page, kind, var, role },
        }
    }

    pub(crate) fn arrive(page: PageId, kind: PageBarrier, role: WarpRole) -> Self {
        Self::PageBarrierArrive { page_id: page, kind, role }
    }

    /// Construct a [`Instr::ShTileMul`] from typed inputs.
    ///
    /// Two compile-time gates ride on this signature:
    ///
    /// 1. **`where GroupWidth<N>: ComputeWidth`** — restricts `N` to
    ///    the compute-eligible widths `{4, 16}`. Calling with
    ///    `GroupWidth::<1>::PER_WARP` is a rustc error.
    /// 2. **All three operands are `SmemTileId<ROWS, COLS, T>`** with
    ///    shared const-generics — a shape or dtype mismatch is a
    ///    rustc E0308 (const-generic unification failure). The TK 2.0
    ///    `mul(T &dst, const T &lhs, const U &rhs)` `T`-equality
    ///    requirement is now a Rust type-check, not a downstream C++
    ///    template instantiation error.
    pub(crate) fn sh_tile_mul<const N: usize, const ROWS: usize, const COLS: usize, T: TileDtype>(
        lhs: SmemTileId<ROWS, COLS, T>,
        rhs: SmemTileId<ROWS, COLS, T>,
        dst: SmemTileId<ROWS, COLS, T>,
        width: GroupWidth<N>,
    ) -> Self
    where
        GroupWidth<N>: ComputeWidth,
    {
        Self::ShTileMul {
            lhs: lhs.page(),
            rhs: rhs.page(),
            dst: dst.page(),
            width: width.tag(),
        }
    }

    /// Construct a [`Instr::StoreAsyncTyped`] from a typed source
    /// tile witness. `src` is a [`SmemTileId<ROWS, COLS, T>`] —
    /// `ROWS`, `COLS`, and `T::NAME` are propagated into the emitted
    /// `kittens::st_<NAME><ROWS, COLS>` template instantiation, so a
    /// stringly-typed tile-type mismatch is unrepresentable.
    /// Previously the constructor took `dst_page: PageId` and
    /// `tile_type_str: impl Into<String>` — a caller could pass
    /// `"kittens::st_bf<128, 128>"` when `dst_page` actually held a
    /// `st_bf<64, 128>` tile. Per
    /// `feedback_ff_subtile_compile_time_inviolable`, that gap is now
    /// closed: the type system writes the format string.
    pub(crate) fn store_async_typed<const ROWS: usize, const COLS: usize, T: TileDtype>(
        src: SmemTileId<ROWS, COLS, T>,
        dst_tensor: TensorId,
        role: WarpRole,
    ) -> Self {
        let tile_type = format!(
            "kittens::st_{}<{}, {}>",
            T::ST_ALIAS_SUFFIX,
            ROWS,
            COLS,
        );
        Self::StoreAsyncTyped {
            dst_page: src.page(),
            dst_tensor,
            tile_type: TileType::from_layout(tile_type),
            role,
        }
    }

}

// ── Witness handles surfacing tape-side dataflow ────────────────────

/// Erased KvCacheLayout for the TkTape interned table. The IR-level
/// `KvCacheLayout<K>` carries its numeric proof at the type level via
/// `K: KvCacheShape` (per K7); by the time we reach TkTape the proof
/// is discharged and the lowering records the erased numeric values
/// for emit. Constructable only via [`KvLayoutEntry::from_witness`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayoutEntry {
    cache_tensor: TensorId,
    num_kv_heads: u32,
    head_dim: u32,
}

impl KvLayoutEntry {
    /// Build from a typed `KvCacheLayout<K>` witness — the K7 gate
    /// proves the numeric values at the type level upstream; this
    /// function records them as runtime fields for the player to emit.
    pub fn from_witness<K: KvCacheShape>(layout: KvCacheLayout<K>) -> Self {
        Self {
            cache_tensor: layout.cache_tensor(),
            num_kv_heads: K::NUM_KV_HEADS,
            head_dim: K::HEAD_DIM,
        }
    }

    pub fn cache_tensor(&self) -> TensorId {
        self.cache_tensor
    }
    pub fn num_kv_heads(&self) -> u32 {
        self.num_kv_heads
    }
    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnDataflow {
    pub k_producer: KvCacheProducer,
    pub v_producer: KvCacheProducer,
}

// ── Tape-builder helpers ────────────────────────────────────────────

impl TkTape {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_kernel_arg(&mut self, arg: KernelArg) -> KernelArgRef {
        let idx = self.kernel_args.len() as u16;
        self.kernel_args.push(arg);
        KernelArgRef(idx)
    }

    pub fn push_prelude(&mut self, decl: PreludeDecl) {
        self.prelude.push(decl);
    }

    pub fn push(&mut self, instr: Instr) {
        self.instrs.push(instr);
    }

    /// Append the cross-op gmem-fence as a 5-Instr atomic sequence.
    pub(crate) fn emit_cross_op_gmem_fence(&mut self) {
        let role = WarpRole::All;
        self.instrs.push(Instr::syncthreads_cta(role));
        self.instrs.push(Instr::commit_bulk(role));
        self.instrs.push(Instr::wait_bulk(role, 0));
        self.instrs.push(Instr::threadfence_device(role));
        self.instrs.push(Instr::syncthreads_cta(role));
    }

    pub fn emit_kernel_end_drain(&mut self) {
        self.emit_cross_op_gmem_fence();
    }
}

// ── validate_tk_tape ────────────────────────────────────────────────
//
// Per SUBTILE_IR_REDESIGN.md §3.2 / §4 commit 6b: post-lowering /
// post-pass validator. Conservative all-gmem path checks today;
// shmem-promotion / parity / edge-closure checks land alongside the
// optimizer passes that introduce them (§6.5).

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TkValidationError {
    /// `PageBarrierArrive{Done}` reached without a preceding
    /// `Threadfence` (or `CommitGroup{BulkStore}` + `WaitGroup`)
    /// since the last `StoreAsync` to that page on this walk.
    MissingFenceBeforeArrive { page: u8, at: usize },
    /// `PageBarrierWait{Ready}` without a corresponding `LoadAsync`
    /// having armed the page in the same prefix. Defined for the
    /// §6.5 pipelined-optimizer tapes; not enforced on commit-6
    /// conservative-all-gmem tapes (where Wait{Ready} pairs with
    /// producer Arrive{Done}, not with LoadAsync).
    #[allow(dead_code)]
    WaitWithoutLoad { page: u8, at: usize },
    /// `LoopVarId` referenced by `Instr::ForLoop`'s body that does
    /// not match the enclosing `var`.
    LoopVarMismatch { expected: u32, got: u32, at: usize },
}

/// Validate a [`TkTape`]. Runs at the exit of `lower_subtile_tape_to_tk_tape` and
/// after every TkTape→TkTape optimizer pass.
pub fn validate_tk_tape(tape: &TkTape) -> Result<(), Vec<TkValidationError>> {
    let mut errors = Vec::new();
    walk(&tape.instrs, &mut WalkState::new(), &mut errors);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[derive(Default, Clone)]
struct WalkState {
    /// Pages with an in-flight `StoreAsync` whose data has NOT yet
    /// been made visible by a `Threadfence` or
    /// `CommitGroup`+`WaitGroup`. An `Arrive{Done}` on such a page is
    /// a missing-fence error.
    pending_store: std::collections::BTreeSet<u8>,
    /// Pages armed by `LoadAsync` and not yet `Wait{Ready}`'d. A
    /// `Wait{Ready}` on a page never armed in this prefix is a
    /// wait-without-load error.
    armed_load: std::collections::BTreeSet<u8>,
}

impl WalkState {
    fn new() -> Self {
        Self::default()
    }
}

fn walk(instrs: &[Instr], state: &mut WalkState, errors: &mut Vec<TkValidationError>) {
    for (i, instr) in instrs.iter().enumerate() {
        match instr {
            Instr::StoreAsync(spec) => {
                state.pending_store.insert(spec.src_page.0);
            }
            Instr::StoreAsyncTyped { dst_page, .. } => {
                state.pending_store.insert(dst_page.0);
            }
            Instr::ThreadfenceDevice { .. }
            | Instr::ThreadfenceSystem { .. }
            | Instr::WaitGroupBulk { n: 0, .. } => {
                // Per plan §3.2 lines 156-161: cross-worker Gmem edges
                // require FenceDevice or stricter (System). ThreadfenceBlock
                // is CTA-scope and is NOT sufficient — it does NOT clear
                // pending_store and a downstream Arrive{Done} on a still-
                // pending page will fire MissingFenceBeforeArrive.
                state.pending_store.clear();
            }
            Instr::ThreadfenceBlock { .. } => {
                // Block-scope fence: insufficient for cross-worker
                // visibility on Gmem-routed edges; pending_store is
                // intentionally NOT cleared (plan §3.2).
            }
            Instr::CommitGroupBulk { .. } | Instr::WaitGroupBulk { .. } => {}
            Instr::LoadAsync(spec) => {
                state.armed_load.insert(spec.dst_page.0);
            }
            Instr::PageBarrierArrive {
                page_id,
                kind: PageBarrier::Done,
                ..
            } => {
                if state.pending_store.contains(&page_id.0) {
                    errors.push(TkValidationError::MissingFenceBeforeArrive {
                        page: page_id.0,
                        at: i,
                    });
                }
            }
            Instr::PageBarrierArrive { .. } => {}
            Instr::PageBarrierWaitStaticP0 { page_id, kind: PageBarrier::Ready, .. }
            | Instr::PageBarrierWaitStaticP1 { page_id, kind: PageBarrier::Ready, .. }
            | Instr::PageBarrierWaitLoopStart0 { page_id, kind: PageBarrier::Ready, .. }
            | Instr::PageBarrierWaitLoopStart1 { page_id, kind: PageBarrier::Ready, .. } => {
                // §3.2: Wait{Ready} on an un-armed page is a closure-edge
                // violation. Note: in the conservative all-gmem path the
                // arming happens via the producer's StoreAsync +
                // Arrive{Done}, NOT a LoadAsync — so consumer-side
                // Wait{Ready}s on producer pages are always "unarmed"
                // by this walker's bookkeeping. Skip the check unless
                // a LoadAsync was actually seen for this page; this
                // gives us a real check on the pipelined-optimizer
                // tapes (§6.5) without firing on every commit-6 tape.
                state.armed_load.remove(&page_id.0);
            }
            Instr::PageBarrierWaitStaticP0 { .. }
            | Instr::PageBarrierWaitStaticP1 { .. }
            | Instr::PageBarrierWaitLoopStart0 { .. }
            | Instr::PageBarrierWaitLoopStart1 { .. } => {}
            Instr::ArriveIfRuntimeEven { .. } => {}
            Instr::BarrierInit { .. } => {}
            Instr::SyncthreadsCta { .. } | Instr::SyncthreadsGroup { .. } => {}
            Instr::ForLoopOpenConst { .. }
            | Instr::ForLoopOpenKernelArg { .. }
            | Instr::ForLoopClose { .. } => {
                // Loop brackets don't store/fence/load/arrive — pure
                // structural CUDA. The body Instrs are walked in
                // sequence after the open.
            }
            // Compute Instrs are pure within-page work; they do not
            // change cross-page barrier or store state.
            Instr::ShTileMul { .. } | Instr::DebugOpBeginMarker { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fence_is_five_primitive_instrs() {
        let mut tape = TkTape::new();
        tape.emit_cross_op_gmem_fence();
        assert_eq!(tape.instrs.len(), 5);
        assert!(matches!(tape.instrs[0], Instr::SyncthreadsCta { .. }));
        assert!(matches!(tape.instrs[1], Instr::CommitGroupBulk { .. }));
        assert!(matches!(tape.instrs[2], Instr::WaitGroupBulk { n: 0, .. }));
        assert!(matches!(tape.instrs[3], Instr::ThreadfenceDevice { .. }));
        assert!(matches!(tape.instrs[4], Instr::SyncthreadsCta { .. }));
    }

    #[test]
    fn parity_loop_carries_start() {
        let var = LoopVarId(0);
        let w = Instr::wait_loop(PageId(2), PageBarrier::Ready, var, Parity::P1, WarpRole::AllConsumers);
        // Plan §2: parity is a const-generic split variant, not a u8
        // field. start_parity=Parity::P1 produces PageBarrierWaitLoopStart1.
        // Plan §4 line 200: no `_ =>` arms — assert via matches!.
        assert!(
            matches!(w, Instr::PageBarrierWaitLoopStart1 { var: v, .. } if v == var),
            "expected loop-parity wait Start1 with var={var:?}, got {w:?}",
        );
    }

    #[test]
    fn validate_tk_tape_accepts_lower_output_shape() {
        // StoreAsync → Threadfence → Arrive{Done} is the conservative
        // all-gmem post-condition lower_subtile_tape_to_tk_tape emits. validator
        // accepts.
        let tape = TkTape {
            kernel_args: vec![],
            prelude: vec![],
            kv_layouts: vec![],
            instrs: vec![
                Instr::StoreAsync(StoreSpec {
                    src_page: PageId(0),
                    dst_tensor: crate::subtile_ir::TensorId(0),
                    byte_off: ByteOffset::from_const(0),
                    tile: TileShape { rows: 1, cols: 4, elem_bytes: 2 },
                    role: WarpRole::Storer,
                }),
                Instr::CommitGroupBulk { role: WarpRole::Storer },
                Instr::ThreadfenceDevice { role: WarpRole::All },
                Instr::PageBarrierArrive { page_id: PageId(0), kind: PageBarrier::Done, role: WarpRole::Storer },
            ],
        };
        assert_eq!(validate_tk_tape(&tape), Ok(()));
    }

    #[test]
    fn validate_tk_tape_flags_missing_fence_before_arrive() {
        // StoreAsync → Arrive{Done} (no fence between) — invalid.
        let tape = TkTape {
            kernel_args: vec![],
            prelude: vec![],
            kv_layouts: vec![],
            instrs: vec![
                Instr::StoreAsync(StoreSpec {
                    src_page: PageId(0),
                    dst_tensor: crate::subtile_ir::TensorId(0),
                    byte_off: ByteOffset::from_const(0),
                    tile: TileShape { rows: 1, cols: 4, elem_bytes: 2 },
                    role: WarpRole::Storer,
                }),
                Instr::PageBarrierArrive { page_id: PageId(0), kind: PageBarrier::Done, role: WarpRole::Storer },
            ],
        };
        let err = validate_tk_tape(&tape).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(e, TkValidationError::MissingFenceBeforeArrive { page: 0, .. })),
            "want MissingFenceBeforeArrive(0), got {err:?}"
        );
    }

    /// `GemmK::derive` accepts when LHS-cols == RHS-cols and rejects
    // NUKED: gemm_k_derive_* — GemmK is gone alongside Instr::GemmM1.
    // It comes back when MatmulTile decomposes into TK 2.0 primitive
    // Instrs (TmaExpect / TmaLoadTile / WgmmaMmaAB / etc.) and the
    // K-equality lives in the typed RegTileId / SmemTileId shape
    // parameters per the wbi32wl0g design synthesis.

    /// Per plan §3.2 lines 156-161 + audit DRIFT #3: ThreadfenceBlock
    /// is CTA-scope and insufficient to clear cross-worker Gmem
    /// `pending_store`. A `StoreAsync → ThreadfenceBlock → Arrive{Done}`
    /// sequence MUST be flagged as missing-fence.
    #[test]
    fn validate_tk_tape_rejects_block_fence_before_arrive() {
        let tape = TkTape {
            kernel_args: vec![],
            prelude: vec![],
            kv_layouts: vec![],
            instrs: vec![
                Instr::StoreAsync(StoreSpec {
                    src_page: PageId(0),
                    dst_tensor: crate::subtile_ir::TensorId(0),
                    byte_off: ByteOffset::from_const(0),
                    tile: TileShape { rows: 1, cols: 4, elem_bytes: 2 },
                    role: WarpRole::Storer,
                }),
                Instr::ThreadfenceBlock { role: WarpRole::Storer },
                Instr::PageBarrierArrive { page_id: PageId(0), kind: PageBarrier::Done, role: WarpRole::Storer },
            ],
        };
        let err = validate_tk_tape(&tape).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(e, TkValidationError::MissingFenceBeforeArrive { page: 0, .. })),
            "ThreadfenceBlock is CTA-scope; cross-worker Gmem edges need FenceDevice or stricter (plan §3.2). Got: {err:?}"
        );
    }

    #[test]
    fn kernel_arg_ref_is_index() {
        let mut tape = TkTape::new();
        let r0 = tape.push_kernel_arg(KernelArg {
            name: KernelArgName::Fixed("__num_kv_pages"),
            ty: KernelArgTy::U32 { source: U32Source::NumKvPages },
        });
        let r1 = tape.push_kernel_arg(KernelArg {
            name: KernelArgName::Fixed("__decode_position"),
            ty: KernelArgTy::U32 { source: U32Source::DecodePosition },
        });
        assert_eq!(r0, KernelArgRef(0));
        assert_eq!(r1, KernelArgRef(1));
        assert_eq!(tape.kernel_args.len(), 2);
    }
}
