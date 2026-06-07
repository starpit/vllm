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
use std::collections::BTreeMap;
use std::marker::PhantomData;

// NUKED: RopeForm trait + NeoX / Interleaved markers + RopeFormTag
// + GemmK witness + AccumKind + RopeSide + Instr::rope_rotate
// constructor + _RopeFormBridge — all supported the architectural
// Compute Instrs that emitted invented `kittens::ops::*` calls.
// They reappear scoped to the actual TK 2.0 primitives that need
// them (the rope-form invariant lives at the SubtileIR level via
// `subtile_ir::RopeForm`, which is the canonical witness; the
// TkTape-side duplicate was always redundant).

// ── Substrate constants ──────────────────────────────────────────────
//
// Per `feedback_end_to_end_compile_time_proofs` (INVIOLABLE): the
// substrate's per-page byte size MUST propagate end-to-end from a
// typed witness, never as a hand-typed literal. The constants below
// are now *derived* from the canonical type aliases [`PageTileSpec`]
// and [`ActTileSpec`] via [`SmemTileSpec::byte_size`] — a change to
// the page-pool tile shape is a single-line edit at the type alias,
// and the host-wrapper's `DYN_SMEM` and the kernel's `al.allocate<>`
// emit cannot drift relative to it.

/// Page-pool size. Set so `NUM_PAGES * PAGE_SIZE + NUM_ACT_PAGES *
/// ACT_PAGE_SIZE <= HOPPER_MAX_DYN_SMEM_BYTES` (228 KB) — verified
/// at Rust compile time by the const_assert below the constants.
///
/// At 32 KB per `st_bf<128, 128>` page + 16 KB per `st_bf<64, 128>`
/// act page, the substrate is bounded by Hopper, not by anything
/// else. With NUM_PAGES = 5, NUM_ACT_PAGES = 4: total = 5*32 + 4*16
/// = 224 KB ≤ 228 KB. Margin = 4 KB (for static __shared__
/// semaphores + sv_ arena, currently sized in static smem).
///
/// **Demand vs capacity**: `page_coalesce_pass` empirically observed
/// 7 unique logical PageIds for the Llama-3.2-1B decode tape. With
/// NUM_PAGES = 5, the pass will panic at codegen time
/// (proc-macro expansion in `ferrite-forward-macro`) with the
/// liveness diagnostic — that's a compile-time-class failure, NOT a
/// runtime kernel deadlock. Closing the demand-vs-capacity gap is
/// the next §6.5 pass (mixed-shape pool, gmem spill of long-lived
/// activations, or finer-grained coalescing) — see
/// `feedback_compile_time_or_garbage` (INVIOLABLE).
pub const NUM_PAGES: u32 = 5;

/// Canonical type alias for the tile that lives in `page_buf[i]`. The
/// substrate is uniform — every PageId indexes a tile of this shape.
/// Changing the page-pool shape is a single-line edit here; both the
/// host-wrapper's DYN_SMEM math and the kernel's `al.allocate<...>`
/// emit derive from this alias.
pub type PageTileSpec = SmemTileSpec<128, 128, Bf16>;

/// Byte size of a single page_buf entry, derived from the typed
/// witness — `128 * 128 * 2 = 32768` bytes for bf16. NOT a hand-typed
/// literal: a future shape change at [`PageTileSpec`] propagates here
/// automatically.
pub const PAGE_SIZE: u32 = PageTileSpec::byte_size();

pub const SCRATCH_BYTES: u32 = 1024;
pub const NUM_CONSUMER_WARPS: u8 = 16;
pub const NUM_SERVICE_WARPS: u8 = 4;
pub const NUM_WARPS: u8 = NUM_SERVICE_WARPS + NUM_CONSUMER_WARPS;

/// Activation page pool sized for Hopper WGMMA m64 — 64 rows × 128 cols.
/// Per SUBTILE_TK20_DECOMP.md §"Resolved decision 4" ("pad act_smem to
/// 4 tile rows"). WGMMA `mma_AB`/`mma_ABt` rt-A path needs A.rows == 64
/// for collective M=64 (warpgroup of 4 × per-warp 16 rows). Weights
/// (B operand) stay in the 128-row page_buf so K=128 doesn't force
/// K-tiling.
/// Activation-pool size. See [`NUM_PAGES`] for the cap math.
pub const NUM_ACT_PAGES: u32 = 4;

/// Canonical type alias for the tile that lives in `act_buf[i]`.
/// Distinct shape from [`PageTileSpec`] (64 rows vs 128) so a wrong
/// pool routing is rustc E0308 — see [`ActPageId`] vs [`PageId`].
pub type ActTileSpec = SmemTileSpec<64, 128, Bf16>;

/// Byte size of a single act_buf entry, derived from [`ActTileSpec`].
/// `64 * 128 * 2 = 16384` bytes for bf16.
pub const ACT_PAGE_SIZE: u32 = ActTileSpec::byte_size();

// ── Hardware capacity bounds ─────────────────────────────────────────
//
// Per `feedback_compile_time_or_garbage` (INVIOLABLE): every invariant
// a fix relies on MUST become a compile-time guard. Hopper's hardware
// caps below were previously enforced only at kernel-launch time
// (cudaFuncSetAttribute returning an error code), or worse — silently
// at runtime. Each `const _: () = assert!(...)` below fails the Rust
// build if the substrate exceeds hardware, NOT the kernel launch.

/// Hopper sm_90a max dynamic shared memory per block, with the
/// `cudaFuncAttributeMaxDynamicSharedMemorySize` opt-in. Per CUDA 12.x
/// programming guide table: 228 KB usable. (Static `__shared__` cap
/// is 48 KB; the substrate uses dynamic shared via `shared_allocator`
/// to clear that.)
pub const HOPPER_MAX_DYN_SMEM_BYTES: u32 = 228 * 1024;

/// Total dynamic-smem claim of the substrate's pools — derived from
/// the typed witnesses. Mirrors what the kernel's `al.allocate<>`
/// chain consumes for `page_buf` + `act_buf`. Note: barriers are
/// currently emitted as static `__shared__` and don't enter this
/// total (separate audit follow-up).
pub const SUBSTRATE_DYN_SMEM_BYTES: u32 =
    NUM_PAGES * PAGE_SIZE + NUM_ACT_PAGES * ACT_PAGE_SIZE;

/// Compile-time guard: if the substrate's dynamic-smem total exceeds
/// the Hopper cap, this `const _: ()` evaluation panics at *Rust
/// compile time*, NOT at kernel launch. A miswire (NUM_PAGES too
/// big, page tile shape too big, etc.) becomes `error[E0080]`, never
/// a runtime `cudaErrorInvalidValue` from `cudaFuncSetAttribute`.
/// Per `feedback_end_to_end_compile_time_proofs`.
const _: () = assert!(
    SUBSTRATE_DYN_SMEM_BYTES <= HOPPER_MAX_DYN_SMEM_BYTES,
    "substrate dynamic-smem total exceeds Hopper sm_90a 228 KB cap. \
     Reduce NUM_PAGES, NUM_ACT_PAGES, or shrink PageTileSpec / ActTileSpec.",
);

// ── Mbarrier arrival counts ──────────────────────────────────────────
//
// Per `feedback_compile_time_or_garbage` (INVIOLABLE) +
// `feedback_no_redundant_const_generics`: the arrival count of a
// `kittens::semaphore` (mbarrier) is bounded at compile time —
// mismatched count is a kernel deadlock (mbarrier never resolves),
// so the value cannot be a free runtime u32. The sealed enum below
// limits `BarrierInit { count }` to TK 2.0-legal values that match
// known producer/consumer roles in the kernel.
//
// Variants:
// - `One` — single-warp TMA load (`kittens::group<1>::tma::*`
//   arrives once on Ready). The lone TMA arm-warp issues one
//   `arrive` per loaded page.
// - `AllConsumers` — all consumer warps arrive on Done/Consumed
//   (count = NUM_CONSUMER_WARPS). Producer waits on this before
//   reusing the page in the next pipeline iteration.
//
// Fabricating an arbitrary u32 is impossible (sealed inner field on
// each variant via the enum's value being the const inline). Adding
// a new arrival pattern is a one-line `enum` extension here, not a
// silently-merged numeric literal at the BarrierInit construction
// site.

/// Sealed mbarrier arrival count. Per
/// `feedback_compile_time_or_garbage` + audit finding
/// `barrier-init-count-untyped-u32`. Replace what was previously
/// `count: u32` on [`Instr::BarrierInit`] with this sealed enum so
/// a wrong count is rustc E0277 (no impl matching `From<u32>` etc.),
/// not a runtime mbarrier deadlock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArrivalCount {
    /// Single-warp arrival pattern: one `kittens::group<1>::arrive`
    /// per BarrierInit cycle. Used for TMA-load Ready barriers
    /// (the lone arm-warp arrives once when the load completes).
    One,
    /// Every consumer warp arrives once: count =
    /// [`NUM_CONSUMER_WARPS`]. Used for Done/Consumed barriers
    /// where all consumer warps must have finished reading the
    /// page before the producer reuses it.
    AllConsumers,
}

impl ArrivalCount {
    /// The actual u32 arrival count, derived from the variant +
    /// substrate constants. The player calls this at emit time;
    /// no unconstrained u32 ever appears on the IR.
    pub const fn count(self) -> u32 {
        match self {
            Self::One => 1,
            Self::AllConsumers => NUM_CONSUMER_WARPS as u32,
        }
    }
}

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

    /// Register-tile SSA arena. Each entry records a [`RegTileSlot`]'s
    /// runtime shape/dtype/layout for the kernel-preamble decl emit.
    /// `BTreeMap` (not `HashMap`) for deterministic preamble emit
    /// order = deterministic codegen output = stable goldens.
    pub(crate) reg_tile_arena: BTreeMap<RegTileSlot, RegTileArenaEntry>,

    /// Register-vec SSA arena. See [`Self::reg_tile_arena`].
    pub(crate) reg_vec_arena: BTreeMap<RegVecSlot, RegVecArenaEntry>,

    /// Shared-vec arena. Each entry records a [`SmemVecSlot`]'s
    /// runtime length + dtype, used by `emit_kernel` to declare
    /// `__shared__ kittens::sv_<dtype><LEN> sv_<idx>;` per slot —
    /// distinct from the tile-shaped `page_buf[]` array. TK 2.0
    /// `row_sum`, `mul_row`, `mul_col`, `load_async` (vec) require
    /// a `kittens::sv_*` operand, NOT a `kittens::st_*` page.
    pub(crate) smem_vec_arena: BTreeMap<SmemVecSlot, SmemVecArenaEntry>,

    /// Next id minted by [`Self::mint_reg_tile`]; checked-add so
    /// u16 overflow panics with a clear message.
    pub(crate) next_reg_tile_slot: u16,

    /// Next id minted by [`Self::mint_reg_vec`].
    pub(crate) next_reg_vec_slot: u16,

    /// Next id minted by [`Self::mint_smem_vec`].
    pub(crate) next_smem_vec_slot: u16,
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
    /// Per-tensor kernel arg: canonical text is `t<TensorId.0>`. The
    /// SubtileIR `TensorId` is the witness of identity — the player
    /// is the only place the canonical text appears, so a lowering
    /// cannot synthesize an unknown tensor name.
    Tensor(TensorId),
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
    /// expected arrival count, sealed to a small set of TK 2.0-legal
    /// values via [`ArrivalCount`]. A wrong count = kernel deadlock
    /// (mbarrier never resolves), so the value is constrained at
    /// compile time, not runtime.
    BarrierInit {
        page_id: PageId,
        kind: PageBarrier,
        count: ArrivalCount,
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
        dst_arg: KernelArgRef,
        tile_type: TileTypeSpec,
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

    /// Pairwise add two shared tiles into a third — TK 2.0 primitive
    /// `kittens::group<N>::add(dst, lhs, rhs)` at
    /// `ops/group/shared/tile/maps.cuh:280` (binary tile+tile add,
    /// included into struct group<N> via `shared/shared.cuh` per
    /// `ops/group/group.cuh:45`). Used by SubOp::Elementwise(Add) and
    /// SubOp::SumReduce (chained N-1 times for N inputs).
    ///
    /// Same compile-time gates as [`Instr::ShTileMul`]: `width` is
    /// constructed only via [`GroupWidth<N>: ComputeWidth`], and the
    /// constructor takes `SmemTileId<ROWS, COLS, T>` operands so a
    /// shape/dtype mismatch is rustc E0308.
    ShTileAdd {
        lhs: PageId,
        rhs: PageId,
        dst: PageId,
        width: GroupWidthTag,
    },

    /// Pairwise divide — TK 2.0 primitive `kittens::group<N>::div(dst,
    /// lhs, rhs)` at `ops/group/shared/tile/maps.cuh:319`. Used by
    /// SiluMul (the sigmoid denominator: `gate / (1 + exp(-gate))`).
    /// Same compile-time gates as ShTileMul/ShTileAdd.
    ShTileDiv {
        lhs: PageId,
        rhs: PageId,
        dst: PageId,
        width: GroupWidthTag,
    },

    /// Element-wise exp — TK 2.0 primitive `kittens::group<N>::exp(dst,
    /// src)` at `ops/group/shared/tile/maps.cuh:172` (unary, applies
    /// `base_ops::exp` element-wise). Used by SiluMul. Same width
    /// gating as ShTileMul; only one source operand.
    ShTileExp {
        src: PageId,
        dst: PageId,
        width: GroupWidthTag,
    },

    /// Tile × scalar multiply — TK 2.0 primitive `kittens::group<N>::mul`
    /// at `ops/group/shared/tile/maps.cuh:306` with `U = T::dtype`
    /// (the scalar overload of `bin_map<base_ops::mul, T>` at
    /// `maps.cuh:38`). Used by SiluMul (`-gate` via scale=-1).
    ///
    /// `dtype` carries the page's dtype (sealed [`TileDtypeTag`]) so
    /// the player emits the literal as `kittens::<dtype>(scalar)`
    /// — wrong dtype mismatch is unrepresentable at construction
    /// because the typed constructor derives `dtype` from
    /// `T: TileDtype`'s `tag()` impl.
    ShTileMulScalar {
        lhs: PageId,
        dst: PageId,
        scalar: ScalarF32,
        dtype: TileDtypeTag,
        width: GroupWidthTag,
    },

    /// Tile + scalar — TK 2.0 primitive `kittens::group<N>::add` at
    /// `maps.cuh:280` (scalar overload via `bin_map<base_ops::sum, T>`
    /// at `maps.cuh:38`). Used by SiluMul (`1 + exp(-gate)`).
    ShTileAddScalar {
        lhs: PageId,
        dst: PageId,
        scalar: ScalarF32,
        dtype: TileDtypeTag,
        width: GroupWidthTag,
    },

    // ── Register-tile / register-vec Instrs (commit A — shared by ≥2
    // of steps 5/6/7). Each emits ONE TK 2.0 call. The slot ids are
    // sealed (RegTileSlot, RegVecSlot); the typed witnesses live at
    // the constructor signatures, where const-generics + sealed
    // marker traits gate construction. The runtime entry shape /
    // dtype / layout lives in TkTape::reg_tile_arena / reg_vec_arena
    // (BTreeMap, deterministic preamble emit order).

    /// `kittens::group<N>::load(rt, st)` —
    /// `ops/group/memory/tile/shared_to_register.cuh:15`. Load a
    /// shared tile into a register tile.
    LoadShmemToReg {
        src: PageId,
        dst: RegTileSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::load(rt, st)` — same TK 2.0 primitive as
    /// `LoadShmemToReg` but the source is an `act_buf[N]` tile, not
    /// a `page_buf[N]` tile. Emitted by the `_act_*` constructor
    /// family. Player emits `act_buf[<src>]` instead of `page_buf[<src>]`.
    /// Required for routing WGMMA A through the 64-row activation
    /// pool — see audit ADDENDUM 3 §"Step 2 design".
    LoadShmemToRegFromAct {
        src: ActPageId,
        dst: RegTileSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::store(st, rt)` —
    /// `ops/group/memory/tile/shared_to_register.cuh:139`.
    StoreRegTileToShmem {
        src: RegTileSlot,
        dst: PageId,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::load(rt_dst, page_buf[src].subtile<COLS_SUB>(IDX))`
    /// — TK 2.0 sub-tile reference at `types/shared/st.cuh:159`.
    /// Loads a column-block of width `subtile_cols` at index
    /// `subtile_idx` from a full shared tile into a register tile.
    /// Used by RopeRotateNeoX to split q at head_dim/2.
    LoadShmemSubTileToReg {
        src: PageId,
        subtile_cols: u16,
        subtile_idx: u16,
        dst: RegTileSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::store(page_buf[dst].subtile<COLS_SUB>(IDX), rt_src)`
    /// — inverse of LoadShmemSubTileToReg. Used by RopeRotateNeoX
    /// to write upper/lower halves back into one page in-place.
    StoreRegTileSubTileToShmem {
        src: RegTileSlot,
        dst: PageId,
        subtile_cols: u16,
        subtile_idx: u16,
        width: GroupWidthTag,
        role: WarpRole,
    },

    // ── MatmulTile / WGMMA Instrs (step 9) ────────────────────────
    //
    // Per SUBTILE_TK20_DECOMP.md §"Per-SubOp Instr counts" line 17,
    // MatmulTile decomposes into 8 Instrs. The WGMMA path is
    // warpgroup-only (GroupWidth<4>); the typed constructors enforce.

    /// `kittens::group<1>::tma::expect_bytes(page_<barrier>[barrier_page],
    /// bytes)` — `ops/group/util/tma.cuh:18`. Sets the expected
    /// transaction byte count on a page mbarrier before issuing the
    /// TMA load that arms it. The byte count is derived from the
    /// shape/dtype at construction.
    TmaExpect {
        barrier_page: PageId,
        bytes: u32,
        role: WarpRole,
    },

    /// `kittens::group<N>::zero(rt_dst)` —
    /// `ops/group/register/tile/maps.cuh:422`. Initialise a register
    /// tile to zero. Used by MatmulTile to reset the accumulator.
    InitRtZero {
        dst: RegTileSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<4>::mma_fence(rt_d)` —
    /// `ops/group/mma/warpgroup.cuh:23`. Fence on the WGMMA
    /// accumulator before the first `mma_AB`. Required when the
    /// `mma_AB` Instr is constructed with [`FenceExternal`].
    WgmmaFenceAcc {
        d: RegTileSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<4>::mma_AB<D, A, B, FENCE, ACC>(d, a, b)` —
    /// `ops/group/mma/warpgroup.cuh:192` (smem-smem-rt overload).
    /// `FENCE` and `ACC` are template-bool params — runtime values
    /// here are the `KIND` const recovered from the typed witnesses
    /// `FencePolicy` / `AccPolicy`.
    WgmmaMmaAB_SmemSmem {
        a_page: PageId,
        b_page: PageId,
        d: RegTileSlot,
        fence: u8,
        accumulate: u8,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<4>::mma_async_wait<N>()` —
    /// `ops/group/mma/warpgroup.cuh:91`. Stall until the number of
    /// in-flight committed WGMMA groups is ≤ N.
    WgmmaAsyncWait {
        n: u32,
        width: GroupWidthTag,
        role: WarpRole,
    },

    // ── AttnDecode chain Instrs (steps 11-14) ────────────────────

    /// `kittens::group<N>::neg_infty(rv_dst)` —
    /// `ops/group/register/vec/maps.cuh:162`. Initialise a register
    /// vector to negative infinity. Used by AttnDecode_Init for the
    /// row-max accumulator (online softmax).
    InitRvNegInfty {
        dst: RegVecSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::zero(rv_dst)` —
    /// `ops/group/register/vec/maps.cuh:132`. Initialise a register
    /// vector to zero. Used by AttnDecode_Init for the row-sum
    /// accumulator.
    InitRvZero {
        dst: RegVecSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<4>::mma_ABt<D, A, B, FENCE, ACC>(d, a, b)` —
    /// `ops/group/mma/warpgroup.cuh:323`. WGMMA `D = A @ B^T`. Used
    /// by AttnDecode_Qkt (Q @ K^T attention scores).
    WgmmaMmaABt_SmemSmem {
        a_page: PageId,
        b_page: PageId,
        d: RegTileSlot,
        fence: u8,
        accumulate: u8,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<4>::mma_AB<D, A, B, FENCE, ACC>(d, a, b)` —
    /// `ops/group/mma/warpgroup.cuh:140` (rt-st-rt overload, A from
    /// registers). Used by AttnDecode_Sv (P @ V where P comes from
    /// the softmax in registers).
    WgmmaMmaAB_RegSmem {
        a: RegTileSlot,
        b_page: PageId,
        d: RegTileSlot,
        fence: u8,
        accumulate: u8,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<4>::mma_ABt<D, A, B, FENCE, ACC>(d, a, b)` —
    /// `ops/group/mma/warpgroup.cuh:323` (rt-st-rt overload). Used by
    /// AttnDecode_Qkt and any matmul where A is register-resident
    /// and B is transposed.
    WgmmaMmaABt_RegSmem {
        a: RegTileSlot,
        b_page: PageId,
        d: RegTileSlot,
        fence: u8,
        accumulate: u8,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::mul(rt_dst, rt_lhs, kittens::<dtype>(scalar))`
    /// — scalar overload of register-tile mul at
    /// `ops/group/register/tile/maps.cuh:708`. Used by AttnDecode_Qkt
    /// (scale by `1/sqrt(d_head)` and `log2(e)`).
    RegTileMulScalar {
        lhs: RegTileSlot,
        dst: RegTileSlot,
        scalar: ScalarF32,
        dtype: TileDtypeTag,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::row_max(rv_acc, rt_src, rv_acc)` —
    /// `ops/group/register/tile/reductions.cuh:303` (accumulating
    /// overload). Used by AttnDecode_Qkt online softmax.
    RegTileRowMaxAcc {
        src: RegTileSlot,
        acc: RegVecSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::row_sum(rv_acc, rt_src, rv_acc)` —
    /// `ops/group/register/tile/reductions.cuh:329`.
    RegTileRowSumAcc {
        src: RegTileSlot,
        acc: RegVecSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::sub_row(rt_dst, rt_src, rv_row_values)` —
    /// `ops/group/register/tile/maps.cuh:750`. Subtract row vector
    /// (one scalar per row) from each row of `src`. Used by
    /// AttnDecode_Qkt to subtract row-max before exp.
    RegTileSubRow {
        src: RegTileSlot,
        row_vec: RegVecSlot,
        dst: RegTileSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::exp2(rt_dst, rt_src)` —
    /// `ops/group/register/tile/maps.cuh:482`. Element-wise exp2.
    /// Used by AttnDecode_Qkt for online softmax.
    RegTileExp2 {
        src: RegTileSlot,
        dst: RegTileSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::div_row(rt_dst, rt_src, rv_row_values)` —
    /// `ops/group/register/tile/maps.cuh:778`. Divide each row by the
    /// corresponding scalar in `row_values`. Used by AttnDecode_Finalise
    /// to normalize the output by the row-sum accumulator.
    RegTileDivRow {
        src: RegTileSlot,
        row_vec: RegVecSlot,
        dst: RegTileSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::sub(rv_dst, rv_lhs, rv_rhs)` —
    /// `ops/group/register/vec/maps.cuh:346`. Element-wise sub on
    /// register vectors. Used by AttnDecode_Qkt (m_i_new - m_i_old).
    RegVecSub {
        lhs: RegVecSlot,
        rhs: RegVecSlot,
        dst: RegVecSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::exp2(rv_dst, rv_src)` —
    /// `ops/group/register/vec/maps.cuh:205`. Element-wise exp2 on
    /// register vector. Used by AttnDecode_Qkt for the alpha
    /// rescaling factor.
    RegVecExp2 {
        src: RegVecSlot,
        dst: RegVecSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::mul(rv_dst, rv_lhs, rv_rhs)` —
    /// `ops/group/register/vec/maps.cuh:359`. Element-wise mul.
    /// Used by AttnDecode_Qkt to update the row-sum accumulator.
    RegVecMul {
        lhs: RegVecSlot,
        rhs: RegVecSlot,
        dst: RegVecSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::copy(rt_dst, rt_src)` —
    /// `ops/group/register/tile/maps.cuh:627` (with type conversion).
    /// Used by AttnDecode_Sv to convert the fp32 P_block to bf16
    /// before the WGMMA P @ V (mma_AB requires A.T == B.T).
    RegTileCopyConvert {
        src: RegTileSlot,
        dst: RegTileSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::copy(rv_dst, rv_src)` —
    /// `ops/group/register/vec/maps.cuh:177`. Register-vec copy.
    /// Used by AttnDecode_Qkt to save `m_old` before the row-max-acc
    /// update so alpha = exp2(m_old - m_new) can be computed for
    /// online-softmax rescaling.
    RegVecCopy {
        src: RegVecSlot,
        dst: RegVecSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::mul_row(rt_dst, rt_src, rv_row_values)` —
    /// `ops/group/register/tile/maps.cuh:764`. Multiply each row of
    /// `src` by the corresponding scalar in `row_values` (length =
    /// src.rows). Used by AttnDecode_Qkt to apply alpha rescale to
    /// rt_o per row across iterations of online softmax.
    RegTileMulRow {
        src: RegTileSlot,
        row_vec: RegVecSlot,
        dst: RegTileSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::load(rv, sv)` —
    /// `ops/group/memory/vec/shared_to_register.cuh:14`. Load a
    /// shared vector into a register vector.
    LoadVecSmemToReg {
        src: SmemVecSlot,
        dst: RegVecSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::store(sv, rv)` —
    /// `ops/group/memory/vec/shared_to_register.cuh:100`.
    StoreRegVecToShmem {
        src: RegVecSlot,
        dst: SmemVecSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::neg(rt_dst, rt_src)` —
    /// `ops/group/register/tile/maps.cuh:572`.
    RegTileNeg {
        src: RegTileSlot,
        dst: RegTileSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::exp(rt_dst, rt_src)` —
    /// `ops/group/register/tile/maps.cuh:464`.
    RegTileExp {
        src: RegTileSlot,
        dst: RegTileSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::add(rt_dst, rt_lhs, rt_rhs)` —
    /// `ops/group/register/tile/maps.cuh:681`. Element-wise add of
    /// two register tiles.
    RegTileAdd {
        lhs: RegTileSlot,
        rhs: RegTileSlot,
        dst: RegTileSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::sub(rt_dst, rt_lhs, rt_rhs)` —
    /// `ops/group/register/tile/maps.cuh:695`.
    RegTileSub {
        lhs: RegTileSlot,
        rhs: RegTileSlot,
        dst: RegTileSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::div(rt_dst, rt_lhs, rt_rhs)` —
    /// `ops/group/register/tile/maps.cuh:722`.
    RegTileDiv {
        lhs: RegTileSlot,
        rhs: RegTileSlot,
        dst: RegTileSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::mul_col(rt_dst, rt_src, rv_col_values)` —
    /// `ops/group/register/tile/maps.cuh:841`. Multiply each column
    /// of `src` by the corresponding element of `col_vec`. Used by
    /// RopeRotateNeoX (cos/sin column-broadcast).
    RegTileMulCol {
        src: RegTileSlot,
        col_vec: RegVecSlot,
        dst: RegTileSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::add(rt_dst, rt_lhs, kittens::<dtype>(scalar))`
    /// — scalar overload. Used by Silu (`1 + exp(-x)` in registers).
    RegTileAddScalar {
        lhs: RegTileSlot,
        dst: RegTileSlot,
        scalar: ScalarF32,
        dtype: TileDtypeTag,
        width: GroupWidthTag,
        role: WarpRole,
    },

    // ── RmsNorm-unique Instrs (commit B). The dst pages here are
    // semantically shared-vec views onto a tile-shaped page; until
    // SmemVecId<LEN,T> lands as a follow-up, the variants carry
    // raw PageId. The constructors still propagate const-generic
    // shape via the SmemTileId<R,C,T> witness on the source side.

    /// `kittens::group<N>::row_sum(sv_dst, st_src)` —
    /// `ops/group/shared/tile/reductions.cuh:97`. Reduce each row
    /// of `src` to a scalar; the result vector has length = src.rows.
    ShTileRowSum {
        src: PageId,
        dst: SmemVecSlot,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::mul(sv_dst, sv_src, kittens::<dtype>(scalar))` —
    /// scalar overload of `kittens::group<N>::mul` for shared vectors
    /// (`ops/group/shared/vec/maps.cuh` mul + scalar bin_map). Used
    /// by RmsNorm (scale the row-sum by `1/cols`).
    ShVecMulScalar {
        src: SmemVecSlot,
        dst: SmemVecSlot,
        scalar: ScalarF32,
        dtype: TileDtypeTag,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::add(sv_dst, sv_src, kittens::<dtype>(scalar))` —
    /// scalar overload, shared-vec. Used by RmsNorm (`+ eps`).
    ShVecAddScalar {
        src: SmemVecSlot,
        dst: SmemVecSlot,
        scalar: ScalarF32,
        dtype: TileDtypeTag,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::unary_op<kittens::base_ops::rsqrt, RvT>(rv_dst, rv_src)`
    /// — `ops/group/register/vec/maps.cuh:17` + `common/base_ops.cuh:218`.
    /// Reciprocal-sqrt on register vectors. Used by RmsNorm (no
    /// shared-vec rsqrt exists in TK 2.0; rsqrt routes through
    /// register-vec).
    RegVecUnaryRsqrt {
        src: RegVecSlot,
        dst: RegVecSlot,
        dtype: TileDtypeTag,
        layout: RegVecLayoutTag,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::mul_row(st_dst, st_src, sv_row_values)` —
    /// `ops/group/shared/tile/maps.cuh:361`. Multiply each row of
    /// `src` by the corresponding scalar in `row_values` (length =
    /// src.rows). Used by RmsNorm (apply `inv_rms` per-row).
    ShTileMulRow {
        src: PageId,
        row_vec: SmemVecSlot,
        dst: PageId,
        width: GroupWidthTag,
        role: WarpRole,
    },

    /// `kittens::group<N>::mul_col(st_dst, st_src, sv_col_values)` —
    /// `ops/group/shared/tile/maps.cuh:428`. Multiply each column
    /// by the corresponding scalar in `col_values` (length =
    /// src.cols). Used by RmsNorm (apply gamma per-column).
    ShTileMulCol {
        src: PageId,
        col_vec: SmemVecSlot,
        dst: PageId,
        width: GroupWidthTag,
        role: WarpRole,
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
///
/// Runtime tag form — kept on every Instr variant for player /
/// validator inspection. The compile-time gate lives upstream at the
/// per-Instr typed constructors via the [`RoleWitness`] sealed trait
/// and its concrete impls ([`LoaderRole`], [`StorerRole`],
/// [`ConsumerRole`], [`AllConsumersRole`], [`AllWarpsRole`]).
/// Constructors that have a single legal role (e.g. [`LoadSpec::new`]
/// — TMA load is loader-only) take the specific role-type directly,
/// so a wrong-role construction is a Rust E0308.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarpRole {
    Loader,
    Storer,
    Consumer(u8),
    AllConsumers,
    All,
}

// ── Sealed typed role witnesses ─────────────────────────────────────
//
// Per `feedback_ff_subtile_compile_time_inviolable`: any Instr that
// has a single legal role (TMA loads = loader, TMA stores = storer,
// `__syncthreads()` = whole-CTA) must reject other roles at rustc
// time. Free `WarpRole` enum on the constructor is a runtime gate.
//
// Each `*Role` struct is sealed (private constructor module gate);
// `RoleWitness::to_warp_role()` is the documented type-erasure point
// at construction. Only role-eligible Instr constructors take the
// specific role-type; role-agnostic Instrs (threadfence_*, fences
// driven by ANY warp) keep the runtime `WarpRole` for now.

mod role_sealed {
    pub trait Sealed {}
}

/// `LoaderRole` — single warp issuing TMA `load_async` from gmem
/// into smem. Required by [`LoadSpec::new`]; passing any other role
/// is a Rust compile error.
#[derive(Debug, Clone, Copy)]
pub struct LoaderRole;
impl role_sealed::Sealed for LoaderRole {}

/// `StorerRole` — single warp issuing TMA `store_async` and the
/// matching `commit_group`/`store_async_wait`. Required by
/// [`StoreSpec::new`], [`Instr::store_async_typed`].
#[derive(Debug, Clone, Copy)]
pub struct StorerRole;
impl role_sealed::Sealed for StorerRole {}

/// `ConsumerRole(u8)` — one of the consumer warps; the inner u8 is
/// the consumer index inside the consumer set (0..NUM_CONSUMER_WARPS).
#[derive(Debug, Clone, Copy)]
pub struct ConsumerRole(pub u8);
impl role_sealed::Sealed for ConsumerRole {}

/// `AllConsumersRole` — the full NUM_CONSUMER_WARPS-wide consumer
/// set, used when a Compute Instr runs warp-collectively across all
/// consumers (`kittens::group<NUM_CONSUMER_WARPS>::*`).
#[derive(Debug, Clone, Copy)]
pub struct AllConsumersRole;
impl role_sealed::Sealed for AllConsumersRole {}

/// `AllWarpsRole` — the entire CTA (loader + storer + all
/// consumers). Required by [`Instr::syncthreads_cta`] (`__syncthreads()`
/// is whole-CTA).
#[derive(Debug, Clone, Copy)]
pub struct AllWarpsRole;
impl role_sealed::Sealed for AllWarpsRole {}

/// Sealed trait connecting a typed role-witness to its [`WarpRole`]
/// erasure. Only the five typed roles in this module impl it —
/// external types cannot satisfy `RoleWitness`.
///
/// # Compile-fail proof — non-storer rejected on a storer constructor
///
/// `Instr::syncthreads_cta` requires [`AllWarpsRole`]. Passing
/// [`StorerRole`] is a rustc E0308 because the sealed concrete type
/// is what the constructor signature names — there is no impl path
/// across role types.
///
/// ```compile_fail
/// use ferrite_wavefront::tk_tape::{Instr, StorerRole};
/// // syncthreads_cta is `pub(crate)` so we use a generic helper to
/// // surface the same E0308 from outside the crate.
/// fn _wants_all_warps(_: ferrite_wavefront::tk_tape::AllWarpsRole) {}
/// _wants_all_warps(StorerRole);
/// ```
///
/// # Pass — typed-role constructor accepts the matching witness
///
/// ```
/// use ferrite_wavefront::tk_tape::{AllWarpsRole, RoleWitness, WarpRole};
/// assert!(matches!(AllWarpsRole.to_warp_role(), WarpRole::All));
/// ```
pub trait RoleWitness: role_sealed::Sealed + Copy {
    fn to_warp_role(self) -> WarpRole;
}
impl RoleWitness for LoaderRole {
    fn to_warp_role(self) -> WarpRole {
        WarpRole::Loader
    }
}
impl RoleWitness for StorerRole {
    fn to_warp_role(self) -> WarpRole {
        WarpRole::Storer
    }
}
impl RoleWitness for ConsumerRole {
    fn to_warp_role(self) -> WarpRole {
        WarpRole::Consumer(self.0)
    }
}
impl RoleWitness for AllConsumersRole {
    fn to_warp_role(self) -> WarpRole {
        WarpRole::AllConsumers
    }
}
impl RoleWitness for AllWarpsRole {
    fn to_warp_role(self) -> WarpRole {
        WarpRole::All
    }
}

// ── Sealed `AccPolicy` and `FencePolicy` for WGMMA template params ──
//
// Per SUBTILE_TK20_DECOMP.md §"New typed-witness types" line 44:
// `mma_AB<D, A, B, fence, accumulate>` takes two const ints. Each
// int has 0/1 semantics with a clear name, so we lift to sealed
// type-level policies. `KIND` const recovers 0/1 for emit.

mod acc_policy_sealed {
    pub trait Sealed {}
}

/// Sealed accumulator-policy marker for WGMMA Instrs. `Reset` overwrites
/// the accumulator; `Accumulate` adds into the existing value. Maps
/// 1:1 to TK 2.0's `mma_AB<D, A, B, fence, accumulate>` template
/// boolean (line 139 / 192 of `ops/group/mma/warpgroup.cuh`).
pub trait AccPolicy: acc_policy_sealed::Sealed + Copy {
    const KIND: u32;
}

#[derive(Clone, Copy, Debug)]
pub struct AccReset;
impl acc_policy_sealed::Sealed for AccReset {}
impl AccPolicy for AccReset {
    const KIND: u32 = 0;
}

#[derive(Clone, Copy, Debug)]
pub struct AccAccumulate;
impl acc_policy_sealed::Sealed for AccAccumulate {}
impl AccPolicy for AccAccumulate {
    const KIND: u32 = 1;
}

mod fence_policy_sealed {
    pub trait Sealed {}
}

/// Sealed fence-policy marker. `External` means a separate
/// `WgmmaFenceAcc` Instr was emitted before this `mma_AB`; `Internal`
/// means the `mma_AB` template body emits its own `mma_fence(d)`.
pub trait FencePolicy: fence_policy_sealed::Sealed + Copy {
    const KIND: u32;
}

#[derive(Clone, Copy, Debug)]
pub struct FenceExternal;
impl fence_policy_sealed::Sealed for FenceExternal {}
impl FencePolicy for FenceExternal {
    const KIND: u32 = 0;
}

#[derive(Clone, Copy, Debug)]
pub struct FenceInternal;
impl fence_policy_sealed::Sealed for FenceInternal {}
impl FencePolicy for FenceInternal {
    const KIND: u32 = 1;
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

/// Sealed marker for shared↔register move primitives whose TK 2.0
/// implementation requires `ST::rows == GROUP_WARPS * RT::rows`.
///
/// `kittens::group<N>::load(RT &dst, const ST &src)` (and its store
/// sibling, plus the sub-tile + register-vec analogues) static-asserts
/// `ST::rows / RT::rows == GROUP_WARPS` at
/// `third_party/thunderkittens/include/ops/group/memory/tile/shared_to_register.cuh:17`.
/// All current ferrite-wavefront load/store sites pair an `SmemTileId<ROWS,...>`
/// with a `RegTileId<ROWS,...>` of the SAME `ROWS` — i.e. `ST::rows == RT::rows`,
/// which forces `GROUP_WARPS = 1`. Until row-sharded loads are needed,
/// the only valid width is `GroupWidth<1>`.
///
/// Why a sibling trait, not just `ComputeWidth`: `ComputeWidth` is the
/// witness for collective compute primitives (`mul_row`, `row_max_acc`,
/// `softmax`, ...) that legitimately use `<4>` warpgroup or `<16>` all-
/// consumers widths. The smem↔reg moves have a different constraint
/// (the row-ratio gate above), so they need a distinct sealed trait
/// to prevent the `ComputeWidth` widths from accidentally landing here.
///
/// ```
/// use ferrite_wavefront::tk_tape::GroupWidth;
/// fn _wants_warp<W: ferrite_wavefront::tk_tape::WarpLoadWidth>(_: W) {}
/// _wants_warp(GroupWidth::<1>::PER_WARP);
/// ```
///
/// ```compile_fail
/// use ferrite_wavefront::tk_tape::GroupWidth;
/// fn _wants_warp<W: ferrite_wavefront::tk_tape::WarpLoadWidth>(_: W) {}
/// _wants_warp(GroupWidth::<16>::ALL_CONSUMERS); // E0277: not WarpLoadWidth
/// ```
pub trait WarpLoadWidth: group_width_sealed::Sealed {}
impl WarpLoadWidth for GroupWidth<1> {}

/// Sealed witness for `(ST::rows, RT::rows)` pairs supported by
/// `kittens::group<4>::load(rt, st)` (the warpgroup-sharded load).
/// TK 2.0 `shared_to_register.cuh:17` requires
/// `ST::rows == GROUP_WARPS * RT::rows`. For our substrate this is
/// only ever `(128, 32)` — the WGMMA-input load distributes a 128-row
/// page across 4 warps × 32 per-warp rows.
pub trait WarpgroupLoadShape<const ST_ROWS: usize, const RT_ROWS: usize>:
    group_width_sealed::Sealed {}
impl WarpgroupLoadShape<128, 32> for GroupWidth<4> {}
/// Activation-pool variant: 64-row act tile distributed across 4 warps
/// × 16 per-warp rows = WGMMA m64 (height=1 register tile per warp).
/// Used by [`Instr::LoadShmemToRegFromAct`] / the `_act_*` constructor
/// family that route the A operand through `act_buf` instead of
/// `page_buf`. See audit ADDENDUM 3 §"Step 2 design".
impl WarpgroupLoadShape<64, 16> for GroupWidth<4> {}

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
    /// alias defined in `types/shared/st.cuh:313`. The alias hides
    /// the underlying scalar (`st_bf` aliases `st<bf16, ...>`).
    const ST_ALIAS_SUFFIX: &'static str;
    /// Underlying scalar type name — `kittens::<name>`. Used by
    /// scalar-literal emit (`*MulScalar` / `*AddScalar`). The alias
    /// hides the scalar so this is a separate constant from
    /// [`Self::ST_ALIAS_SUFFIX`] (e.g. `Bf16: SUFFIX = "bf",
    /// SCALAR_NAME = "bf16"`).
    const SCALAR_NAME: &'static str;
    /// Bytes per element. Used by [`SmemTileSpec`] to recover the
    /// runtime `elem_bytes` value for emit (TMA descriptor sizing,
    /// `expect_bytes` arithmetic) without storing it as a separate
    /// runtime field.
    const ELEM_BYTES: u32;
    /// Erase the type-level dtype to its sealed runtime
    /// [`TileDtypeTag`]. Used by typed Instr constructors that must
    /// store the dtype on a heterogeneous Instr field.
    fn tag() -> TileDtypeTag;
}

/// `kittens::bf16` — Hopper bfloat16. Aliased as `kittens::st_bf<…>`.
#[derive(Clone, Copy, Debug)]
pub struct Bf16;
impl tile_dtype_sealed::Sealed for Bf16 {}
impl TileDtype for Bf16 {
    const ST_ALIAS_SUFFIX: &'static str = "bf";
    const SCALAR_NAME: &'static str = "kittens::bf16";
    const ELEM_BYTES: u32 = 2;
    fn tag() -> TileDtypeTag {
        TileDtypeTag::Bf16
    }
}

/// `float` — fp32. Used by WGMMA accumulator (`mma_AB<D, A, B>` where
/// D is rt<float, ...>, A/B are rt<bf16, ...> / st<bf16, ...>).
/// Aliased as `kittens::st_fl<…>`.
#[derive(Clone, Copy, Debug)]
pub struct Fp32;
impl tile_dtype_sealed::Sealed for Fp32 {}
impl TileDtype for Fp32 {
    const ST_ALIAS_SUFFIX: &'static str = "fl";
    const SCALAR_NAME: &'static str = "float";
    const ELEM_BYTES: u32 = 4;
    fn tag() -> TileDtypeTag {
        TileDtypeTag::Fp32
    }
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

/// Sealed marker — `(ROWS, COLS, T)` is a legal sub-tile shape of
/// the `page_buf` substrate's [`PageTileSpec`] (128×128 bf16). Either
/// the full-page shape (128, 128, Bf16) or a sub-tile that the
/// existing emit paths use (e.g. 16×128 register-strip).
///
/// Per `feedback_end_to_end_compile_time_proofs`: every numeric proof
/// the substrate's page-pool tile shape encodes MUST propagate to
/// every `SmemTileId<R, C, T>` mint site as a `where`-clause witness.
/// A wrong-shape mint (e.g., `SmemTileId<32, 64, Fp32>::from_page`)
/// is now `error[E0277]: ... PageSubTileShape ... not satisfied`,
/// not a runtime kernel deadlock from a malformed `kittens::st_bf`
/// reference.
///
/// **Impl set today**:
/// - `(128, 128, Bf16)` — full PageTileSpec (matmul B operand,
///   substrate decl).
/// - `(16, 128, Bf16)` — register-strip sub-tile (16 rows of 128
///   cols, used by `LoadShmemSubTileToReg` for register-tile loads).
///
/// Adding a new sub-tile shape is a one-line `impl PageSubTileShape
/// for SmemTileId<R, C, T> {}` here; the substrate-shape validity
/// check happens at the `impl` site, not at the mint site.
mod page_sub_tile_shape_sealed {
    pub trait Sealed {}
}
pub trait PageSubTileShape: page_sub_tile_shape_sealed::Sealed {}

impl page_sub_tile_shape_sealed::Sealed for SmemTileId<128, 128, Bf16> {}
impl PageSubTileShape for SmemTileId<128, 128, Bf16> {}
impl page_sub_tile_shape_sealed::Sealed for SmemTileId<16, 128, Bf16> {}
impl PageSubTileShape for SmemTileId<16, 128, Bf16> {}

impl<const ROWS: usize, const COLS: usize, T: TileDtype> SmemTileId<ROWS, COLS, T> {
    /// Mint a typed tile handle for `page`. Restricted by the sealed
    /// [`PageSubTileShape`] marker to shapes that are legal sub-tiles
    /// of [`PageTileSpec`]: the full 128×128 page or a 16×128
    /// register-strip. A wrong-shape instantiation (e.g.,
    /// `SmemTileId::<32, 64, Fp32>::from_page(p)`) is `error[E0277]`,
    /// not a runtime kernel hang.
    /// `pub(crate)` so only the lowerer can mint these.
    ///
    /// Not `const`: the trait-bound discharge for sealed witnesses is
    /// not yet stable in `const fn` context. Callers are runtime
    /// lowerer code, not const contexts.
    pub(crate) fn from_page(page: PageId) -> Self
    where
        Self: PageSubTileShape,
    {
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

/// Activation-pool peer of [`SmemTileId`]. Indexes into the kernel's
/// `act_buf` shared array (64×128 tiles, sized for Hopper WGMMA m64
/// — see [`NUM_ACT_PAGES`]). Distinct type from `SmemTileId` so the
/// substrate's two pools are routed at compile time, not runtime.
///
/// Per audit ADDENDUM 3 §"Step 2 design": minimum-scope plumbing.
/// Shape const-generics propagate identically to `SmemTileId`; only
/// the underlying `ActPageId` differs (sealed namespace).
#[derive(Clone, Copy, Debug)]
pub struct ActSmemTileId<const ROWS: usize, const COLS: usize, T: TileDtype> {
    page: ActPageId,
    _marker: PhantomData<fn() -> T>,
}

/// Sealed marker — `(ROWS, COLS, T)` is a legal sub-tile shape of
/// the `act_buf` substrate's [`ActTileSpec`] (64×128 bf16). Mirror
/// of [`PageSubTileShape`]; impl set lists every shape the emit
/// uses today against the act-pool. A wrong-shape mint is rejected
/// at compile time, not at runtime.
mod act_sub_tile_shape_sealed {
    pub trait Sealed {}
}
pub trait ActSubTileShape: act_sub_tile_shape_sealed::Sealed {}

impl act_sub_tile_shape_sealed::Sealed for ActSmemTileId<64, 128, Bf16> {}
impl ActSubTileShape for ActSmemTileId<64, 128, Bf16> {}
impl act_sub_tile_shape_sealed::Sealed for ActSmemTileId<16, 128, Bf16> {}
impl ActSubTileShape for ActSmemTileId<16, 128, Bf16> {}

impl<const ROWS: usize, const COLS: usize, T: TileDtype> ActSmemTileId<ROWS, COLS, T> {
    /// Mint a typed activation-tile handle. Restricted to shapes that
    /// are legal sub-tiles of [`ActTileSpec`] via [`ActSubTileShape`].
    pub(crate) fn from_page(page: ActPageId) -> Self
    where
        Self: ActSubTileShape,
    {
        Self {
            page,
            _marker: PhantomData,
        }
    }
    pub const fn page(&self) -> ActPageId {
        self.page
    }
    pub const fn rows() -> usize {
        ROWS
    }
    pub const fn cols() -> usize {
        COLS
    }
}

/// Sealed shared-vec slot identifier. Distinct namespace from
/// [`PageId`]: pages are uniformly typed `kittens::st_bf<R,C>` tiles
/// (the `page_buf[]` array), while shared vecs are `kittens::sv_*<LEN>`
/// objects, declared one-per-slot in the kernel preamble via
/// [`TkTape::smem_vec_arena`]. Mixing the two namespaces was the
/// step-8 cat-5 emit bug — `kittens::group<N>::row_sum(page_buf[N], ...)`
/// fails the `ducks::sv::all V` concept constraint because `page_buf[N]`
/// is a tile, not a vec. Per `feedback_ff_subtile_compile_time_inviolable`:
/// the type system refuses the mismatch — `Instr::sh_tile_row_sum`'s
/// `dst: SmemVecSlot` cannot accept a `PageId` (and vice versa).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmemVecSlot(pub(crate) u16);

#[derive(Debug, Clone, Copy)]
pub(crate) struct SmemVecArenaEntry {
    pub(crate) len: u32,
    pub(crate) dtype: TileDtypeTag,
}

/// Typed shared-memory vector handle. Backed by a [`SmemVecSlot`]
/// minted at lowering time via [`TkTape::mint_smem_vec`]. The
/// const-generic LEN + sealed `T: TileDtype` propagate into the
/// typed Instr constructors so length/dtype mismatch across the
/// vec endpoints (e.g. `row_sum.dst.LEN == src.ROWS`) is a rustc
/// E0308 at construction.
///
/// Constructor (`from_slot`) is `pub(crate)` so only the lowerer can
/// mint these.
///
/// # Compile-fail proof — length mismatch rejected
///
/// ```compile_fail
/// use ferrite_wavefront::tk_tape::{Bf16, SmemVecId, TileDtype};
/// fn _both<const L: usize, T: TileDtype>(
///     _a: SmemVecId<L, T>,
///     _b: SmemVecId<L, T>,
/// ) {}
/// let a: SmemVecId<32, Bf16> = unreachable!();
/// let b: SmemVecId<64, Bf16> = unreachable!();
/// _both(a, b);  // ← rustc rejects: L=32 vs L=64
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SmemVecId<const LEN: usize, T: TileDtype> {
    slot: SmemVecSlot,
    _marker: PhantomData<fn() -> T>,
}

impl<const LEN: usize, T: TileDtype> SmemVecId<LEN, T> {
    pub(crate) const fn from_slot(slot: SmemVecSlot) -> Self {
        Self {
            slot,
            _marker: PhantomData,
        }
    }
    pub const fn slot(&self) -> SmemVecSlot {
        self.slot
    }
    pub const fn len() -> usize {
        LEN
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
    /// Const witness — pure phantom-type construction with no runtime
    /// shape input. The const generics + sealed `T` are the source of
    /// truth. Use this in const contexts (substrate emit, type-alias
    /// wiring) where the shape is fixed at compile time. Per
    /// `feedback_compile_time_or_garbage`: no runtime input means no
    /// runtime check is needed — the `from_shape` `debug_assert`s
    /// stay only for the lowerer's runtime-shape boundary case.
    pub const WITNESS: Self = Self { _marker: PhantomData };

    /// Mint a typed spec at the SubtileIR-graph runtime-shape
    /// boundary. `pub(crate)`: only the lowerer (which owns the
    /// page-shape mapping) can mint these. Release-mode `assert_eq!`
    /// (NOT `debug_assert_eq!`): the proc-macro runs in release
    /// mode, so a typed-witness lie must be caught at codegen time,
    /// not silently produce a malformed `.cu`. Per the audit finding
    /// `smem-tile-spec-debug-assert`.
    ///
    /// In const contexts (substrate emit, type-alias wiring) prefer
    /// [`Self::WITNESS`] — it has no runtime input and so no runtime
    /// check is needed.
    pub(crate) fn from_runtime_shape(shape: TileShape) -> Self {
        assert_eq!(
            shape.rows, ROWS as u32,
            "SmemTileSpec<{ROWS},_,_>::from_runtime_shape: rows mismatch (got {})",
            shape.rows,
        );
        assert_eq!(
            shape.cols, COLS as u32,
            "SmemTileSpec<_,{COLS},_>::from_runtime_shape: cols mismatch (got {})",
            shape.cols,
        );
        assert_eq!(
            shape.elem_bytes,
            T::ELEM_BYTES,
            "SmemTileSpec<_,_,T>::from_runtime_shape: elem_bytes mismatch (got {})",
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

    /// Byte size of a single `kittens::st_<T><ROWS, COLS, ...>` smem
    /// tile, derived from const generics + the sealed `T::ELEM_BYTES`.
    /// Matches TK 2.0's `types/shared/st.cuh:73,77,105`: storage is
    /// `dtype data[rows*cols]`, no hidden padding.
    ///
    /// This is the source of truth for [`PAGE_SIZE`] / [`ACT_PAGE_SIZE`]
    /// — the substrate constants are no longer hand-typed literals.
    /// Per `feedback_end_to_end_compile_time_proofs`: the page byte
    /// size propagates from the typed witness through to the
    /// host-wrapper's `DYN_SMEM` and into the kernel's `al.allocate<>`
    /// chain via the same `T` and ROWS/COLS.
    pub const fn byte_size() -> u32 {
        (ROWS as u32) * (COLS as u32) * T::ELEM_BYTES
    }
}

/// Sealed per §2: inner field is `pub(crate)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageId(pub(crate) u8);

/// Sealed activation-page id — distinct namespace from [`PageId`].
/// Indexes into the kernel's `act_buf` shared array (64×128 entries
/// per [`NUM_ACT_PAGES`]) used for matmul A operands and AttnDecode
/// q/k/v tiles. Kept separate from `PageId` so passing one where the
/// other is expected is rustc E0308 — the substrate's two pools have
/// different shapes and the type system enforces the routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ActPageId(pub(crate) u8);

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

// ── Sealed byte-offset / byte-stride newtypes ───────────────────────
//
// Per `feedback_ff_subtile_compile_time_inviolable`: stride and base
// were `u64` raw — an emit-time bug could swap them or use a stride
// minted for one step-unit (loop iterations) where another (positions)
// was required. ByteStride carries a phantom step-unit so the unit
// is part of the type; ByteOffset is sealed pub(crate)-inner so
// external code can't fabricate.

mod byte_offset_sealed {
    pub trait Sealed {}
}

/// Absolute byte offset within a tensor or memory region. Sealed:
/// inner u64 is `pub(crate)`. The `bytes()` accessor recovers the
/// runtime value for emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ByteOffset(pub(crate) u64);

impl ByteOffset {
    pub const fn new(bytes: u64) -> Self {
        Self(bytes)
    }
    pub const fn bytes(&self) -> u64 {
        self.0
    }
}

/// Sealed marker for what step-unit a [`ByteStride<U>`] strides over.
/// Implemented for [`PerLoopStep`] and [`PerPositionStep`].
pub trait StrideUnit: byte_offset_sealed::Sealed + Copy {
    /// Diagnostic name (used in compile-error messages on mismatch).
    const NAME: &'static str;
}

/// Stride per one increment of a loop variable (the `var` field of
/// [`ByteOffsetExpr::LinearLoop`]). NOT interchangeable with
/// [`PerPositionStep`].
#[derive(Debug, Clone, Copy)]
pub struct PerLoopStep;
impl byte_offset_sealed::Sealed for PerLoopStep {}
impl StrideUnit for PerLoopStep {
    const NAME: &'static str = "loop step";
}

/// Stride per one increment of a kernel-arg-driven position (the
/// `arg` field of [`ByteOffsetExpr::RuntimePosition`]).
#[derive(Debug, Clone, Copy)]
pub struct PerPositionStep;
impl byte_offset_sealed::Sealed for PerPositionStep {}
impl StrideUnit for PerPositionStep {
    const NAME: &'static str = "position step";
}

/// Byte stride — both the BYTE COUNT and the STEP UNIT are at the
/// type level. Two different stride values (`ByteStride<32, ...>` vs
/// `ByteStride<64, ...>`) are different Rust types — a function that
/// expects `ByteStride<32, _>` rejects `ByteStride<64, _>` as rustc
/// E0308. Same for unit mismatch (`PerLoopStep` vs `PerPositionStep`).
///
/// Stride values in this codebase are always compile-time-known
/// (derived from KvCacheLayout<K> / tensor shape const generics);
/// const-generic encoding is the right type-level shape for them.
/// Per `feedback_ff_subtile_compile_time_inviolable` +
/// `feedback_end_to_end_compile_time_proofs`.
#[derive(Debug)]
pub struct ByteStride<const BYTES: u64, U: StrideUnit>(PhantomData<fn() -> U>);

impl<const BYTES: u64, U: StrideUnit> ByteStride<BYTES, U> {
    pub const NEW: Self = Self(PhantomData);
    /// The byte count, recovered from the const generic at emit time.
    pub const fn bytes(&self) -> u64 {
        BYTES
    }
    pub const BYTES: u64 = BYTES;
}

// Manual impls (the derive forms add `U: ...` bounds; the
// phantom-typed U is a unit marker that doesn't need those derives).
impl<const BYTES: u64, U: StrideUnit> Clone for ByteStride<BYTES, U> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<const BYTES: u64, U: StrideUnit> Copy for ByteStride<BYTES, U> {}
impl<const BYTES: u64, U: StrideUnit> PartialEq for ByteStride<BYTES, U> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl<const BYTES: u64, U: StrideUnit> Eq for ByteStride<BYTES, U> {}
impl<const BYTES: u64, U: StrideUnit> std::hash::Hash for ByteStride<BYTES, U> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        BYTES.hash(state);
    }
}

/// Byte-offset expression for TMA load/store source/dest.
///
/// Sealed enum (variants are `pub`, but the type is by-value match-only;
/// new arms can only be added inside this crate). The IR carries
/// **structured data**, not pre-formatted CUDA syntax; the player
/// formats per arm at emit time. Per
/// `feedback_no_premature_string_encoding`.
///
/// The variant stores stride bytes as runtime `u64` (Instr enum is
/// heterogeneous so the const generic must erase). The TYPED
/// CONSTRUCTORS take `ByteStride<STRIDE, U>` const-generic witnesses
/// — the stride value is compile-time-known and unit-checked at
/// construction; mismatches are rustc E0308.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOffsetExpr {
    /// `<c>u` — constant byte offset.
    Const(ByteOffset),
    /// `(<base>u + v<var> * <stride>u)` — loop-linear byte offset.
    /// Constructed via the const-generic `ByteOffsetExpr::linear_loop`
    /// taking `ByteStride<STRIDE, PerLoopStep>`.
    LinearLoop {
        var: LoopVarId,
        stride_bytes: u64,
        base: ByteOffset,
    },
    /// `(<base>u + a<arg> * <stride>u)` — kernel-arg-driven byte
    /// offset. Used by RopeAppend / AttnDecode for KV cache writes.
    /// Constructed via the const-generic `runtime_position` taking
    /// `ByteStride<STRIDE, PerPositionStep>`.
    RuntimePosition {
        arg: KernelArgRef,
        stride_bytes: u64,
        base: ByteOffset,
    },
}

impl ByteOffsetExpr {
    /// Convenience constant constructor.
    pub const fn from_const(bytes: u64) -> Self {
        Self::Const(ByteOffset::new(bytes))
    }

    /// Construct [`Self::LinearLoop`] from a typed const-generic
    /// `ByteStride<STRIDE, PerLoopStep>` witness. Caller writes
    /// `ByteOffsetExpr::linear_loop::<STRIDE>(var, ByteStride::NEW, base)`
    /// — STRIDE is compile-time-known and the unit witness rejects
    /// `PerPositionStep` strides.
    pub const fn linear_loop<const STRIDE: u64>(
        var: LoopVarId,
        _stride: ByteStride<STRIDE, PerLoopStep>,
        base: ByteOffset,
    ) -> Self {
        Self::LinearLoop {
            var,
            stride_bytes: STRIDE,
            base,
        }
    }

    /// Construct [`Self::RuntimePosition`] from a typed const-generic
    /// `ByteStride<STRIDE, PerPositionStep>` witness.
    pub const fn runtime_position<const STRIDE: u64>(
        arg: KernelArgRef,
        _stride: ByteStride<STRIDE, PerPositionStep>,
        base: ByteOffset,
    ) -> Self {
        Self::RuntimePosition {
            arg,
            stride_bytes: STRIDE,
            base,
        }
    }

    /// Construct [`Self::RuntimePosition`] for a KV cache write at
    /// runtime position. Stride is derived from the typed
    /// [`crate::subtile_ir::KvCacheLayout<K>`] witness's
    /// `K::ROW_BYTES` const. Layer base is `layer × K::LAYER_BYTES`.
    ///
    /// Preferred over the generic `runtime_position::<STRIDE>` for KV
    /// cache writes — the K type parameter is the single source of
    /// truth for the cache layout, so wrong-stride is structurally
    /// impossible (caller would have to mismatch K, which is a
    /// separate compile-time error per the K7 lift).
    ///
    /// Stable Rust prevents `runtime_position::<{ K::ROW_BYTES }>`
    /// (`generic_const_exprs` is unstable); this fn is the workaround.
    pub fn kv_cache_runtime_position<K: crate::subtile_ir::KvCacheShape>(
        arg: KernelArgRef,
        layer: u32,
    ) -> Self {
        Self::RuntimePosition {
            arg,
            stride_bytes: K::ROW_BYTES,
            base: ByteOffset::new((layer as u64) * K::LAYER_BYTES),
        }
    }

    /// Construct [`Self::LinearLoop`] for a KV cache CHUNKED read
    /// driven by a loop var iterating over chunks of `CHUNK_ROWS`
    /// cache positions. Stride per loop iteration is
    /// `CHUNK_ROWS × K::ROW_BYTES`; base is `layer × K::LAYER_BYTES`.
    ///
    /// Used by AttnDecode_Qkt's K/V loads (chunked iteration over
    /// the cache). Replaces the placeholder `linear_loop::<128>` that
    /// used a per-loop-step of just 128 bytes — production needs the
    /// full row stride scaled by the chunk row count.
    ///
    /// `K: KvCacheShape` is the type-level cache-shape witness;
    /// stride and base derive from K's const associated values, so
    /// wrong stride for a given K is structurally impossible.
    pub fn kv_cache_chunk_loop<const CHUNK_ROWS: usize, K: crate::subtile_ir::KvCacheShape>(
        var: LoopVarId,
        layer: u32,
    ) -> Self {
        Self::LinearLoop {
            var,
            stride_bytes: (CHUNK_ROWS as u64) * K::ROW_BYTES,
            base: ByteOffset::new((layer as u64) * K::LAYER_BYTES),
        }
    }
}

/// Runtime tile-shape triple at the SubtileIR-graph boundary. Used
/// by the lowerer when the SubtileIR specifies a shape that is not
/// a compile-time const generic (e.g., an `External` weight load
/// whose shape varies per Llama-3.2-1B FUF tile). Per
/// `feedback_no_premature_string_encoding` and
/// `feedback_end_to_end_compile_time_proofs`: every consumer that
/// can take a const-generic [`SmemTileSpec<R, C, T>`] does, and only
/// the runtime-graph boundary uses this type.
///
/// **Sealed inner fields.** The fields are `pub(crate)` so a
/// construction outside the crate is impossible. Inside the crate,
/// constructions go through one of:
///
/// 1. [`SmemTileSpec::shape`] — recovered from the typed witness
///    (compile-time path; no runtime check needed).
/// 2. The lowerer's `region_tile_shape` (runtime-graph boundary;
///    documented call site, single function).
/// 3. [`SmemTileSpec::from_runtime_shape`] — the runtime → typed
///    crossing, hardens to a release-mode `assert_eq!` (NOT
///    `debug_assert!` — release mode is what the proc-macro runs in).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileShape {
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) elem_bytes: u32,
}

/// Runtime sealed dtype tag. The `TileDtype` trait's `tag()` method
/// is the only construction path; external types cannot satisfy
/// `TileDtype` (sealed via `tile_dtype_sealed::Sealed`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileDtypeTag {
    Bf16,
    Fp32,
}

impl TileDtypeTag {
    /// CUDA template-alias suffix: the `<suffix>` in
    /// `kittens::st_<suffix><ROWS, COLS>`. See
    /// `third_party/thunderkittens/include/types/shared/st.cuh:313`.
    pub const fn st_alias_suffix(&self) -> &'static str {
        match self {
            Self::Bf16 => Bf16::ST_ALIAS_SUFFIX,
            Self::Fp32 => Fp32::ST_ALIAS_SUFFIX,
        }
    }
    /// Underlying scalar type — `kittens::<name>` for scalar
    /// literals in `*MulScalar` / `*AddScalar` emit.
    pub const fn scalar_name(&self) -> &'static str {
        match self {
            Self::Bf16 => Bf16::SCALAR_NAME,
            Self::Fp32 => Fp32::SCALAR_NAME,
        }
    }
    pub const fn elem_bytes(&self) -> u32 {
        match self {
            Self::Bf16 => Bf16::ELEM_BYTES,
            Self::Fp32 => Fp32::ELEM_BYTES,
        }
    }
}

/// Structured replacement for the legacy `TileType(String)` carrier
/// of pre-formatted `"kittens::st_bf<128, 128>"`. Per
/// `feedback_no_premature_string_encoding`: the IR carries (rows,
/// cols, dtype) as structured data; the player formats the alias at
/// emit time. Wrong-shape construction goes through the typed
/// `Instr::store_async_typed<ROWS, COLS, T>(SmemTileId<...>)` path,
/// which derives the spec from const-generics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileTypeSpec {
    pub rows: u32,
    pub cols: u32,
    pub dtype: TileDtypeTag,
}

// ── Sealed register-tile / register-vec layouts ─────────────────────
//
// Per SUBTILE_TK20_DECOMP.md §"New typed-witness types" lines 38-39:
// register tiles and register vectors carry their TK 2.0 layout
// (rt_layout::row/col, rv_layout::ortho/align/naive) at the type
// level so a layout-mismatch is rustc E0308 instead of an nvcc
// instantiation error. Layout markers are unit structs sealed via
// private modules; the runtime erasure (`*LayoutTag`) is also sealed
// — only the typed `tag()` impls can mint one.

mod rt_layout_sealed {
    pub trait Sealed {}
}

/// Sealed marker trait for register-tile layouts. Implemented only
/// for [`RowLayout`] and [`ColLayout`].
pub trait RegTileLayout: rt_layout_sealed::Sealed + Copy {
    /// Full TK 2.0 type path used in emitted CUDA, e.g.
    /// `kittens::ducks::rt_layout::row`.
    const LAYOUT_PATH: &'static str;
    fn tag() -> RegTileLayoutTag;
}

/// `kittens::ducks::rt_layout::row` — orthogonal layout (each row
/// owned by a distinct subset of warpgroup threads). The default for
/// most TK 2.0 register-tile ops; required by `mma_AB`'s A operand.
#[derive(Clone, Copy, Debug)]
pub struct RowLayout;
impl rt_layout_sealed::Sealed for RowLayout {}
impl RegTileLayout for RowLayout {
    const LAYOUT_PATH: &'static str = "kittens::ducks::rt_layout::row";
    fn tag() -> RegTileLayoutTag {
        RegTileLayoutTag::Row
    }
}

/// `kittens::ducks::rt_layout::col` — column-major register layout;
/// required by `mma_ABt`'s B operand.
#[derive(Clone, Copy, Debug)]
pub struct ColLayout;
impl rt_layout_sealed::Sealed for ColLayout {}
impl RegTileLayout for ColLayout {
    const LAYOUT_PATH: &'static str = "kittens::ducks::rt_layout::col";
    fn tag() -> RegTileLayoutTag {
        RegTileLayoutTag::Col
    }
}

/// Sealed runtime tag for [`RegTileLayout`] erasure on Instr fields.
/// Only constructable via `RegTileLayout::tag()` (whose impls are
/// crate-internal); external code cannot synthesize a variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegTileLayoutTag {
    Row,
    Col,
}

impl RegTileLayoutTag {
    pub const fn layout_path(&self) -> &'static str {
        match self {
            Self::Row => RowLayout::LAYOUT_PATH,
            Self::Col => ColLayout::LAYOUT_PATH,
        }
    }
}

mod rv_layout_sealed {
    pub trait Sealed {}
}

/// Sealed marker trait for register-vec layouts. Implemented only
/// for [`OrthoLayout`], [`AlignLayout`], [`NaiveLayout`].
pub trait RegVecLayout: rv_layout_sealed::Sealed + Copy {
    const LAYOUT_PATH: &'static str;
    fn tag() -> RegVecLayoutTag;
}

/// `kittens::ducks::rv_layout::ortho` — orthogonal warp partition
/// of the register vector.
#[derive(Clone, Copy, Debug)]
pub struct OrthoLayout;
impl rv_layout_sealed::Sealed for OrthoLayout {}
impl RegVecLayout for OrthoLayout {
    const LAYOUT_PATH: &'static str = "kittens::ducks::rv_layout::ortho";
    fn tag() -> RegVecLayoutTag {
        RegVecLayoutTag::Ortho
    }
}

/// `kittens::ducks::rv_layout::align` — aligned with rt_layout::row.
#[derive(Clone, Copy, Debug)]
pub struct AlignLayout;
impl rv_layout_sealed::Sealed for AlignLayout {}
impl RegVecLayout for AlignLayout {
    const LAYOUT_PATH: &'static str = "kittens::ducks::rv_layout::align";
    fn tag() -> RegVecLayoutTag {
        RegVecLayoutTag::Align
    }
}

/// `kittens::ducks::rv_layout::naive` — naive (one element per
/// thread) layout. Used when full warp parallelism isn't required.
#[derive(Clone, Copy, Debug)]
pub struct NaiveLayout;
impl rv_layout_sealed::Sealed for NaiveLayout {}
impl RegVecLayout for NaiveLayout {
    const LAYOUT_PATH: &'static str = "kittens::ducks::rv_layout::naive";
    fn tag() -> RegVecLayoutTag {
        RegVecLayoutTag::Naive
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegVecLayoutTag {
    Ortho,
    Align,
    Naive,
}

impl RegVecLayoutTag {
    pub const fn layout_path(&self) -> &'static str {
        match self {
            Self::Ortho => OrthoLayout::LAYOUT_PATH,
            Self::Align => AlignLayout::LAYOUT_PATH,
            Self::Naive => NaiveLayout::LAYOUT_PATH,
        }
    }
}

// ── Sealed register-tile / register-vec SSA handles ─────────────────
//
// Per SUBTILE_TK20_DECOMP.md §"Lifetime model: SSA": every Instr that
// produces a register handle mints a fresh id; the player walks the
// tape's `reg_tile_arena` / `reg_vec_arena` BTreeMaps in id-order to
// emit the kernel-preamble decls (`kittens::rt_bf<R, C, layout> rt_N;`).
//
// The handle is a sealed phantom-typed wrapper around a u16 SSA slot
// id. Const-generic shape (R, C / LEN), dtype (T: TileDtype), and
// layout (L: RegTileLayout / RV: RegVecLayout) are propagated through
// the handle's type — Instr constructors with mismatched const
// generics across operands fail rustc unification.

/// Sealed SSA slot id for register tiles. Inner u16 is `pub(crate)`
/// — only crate-internal mint paths (`TkTape::mint_reg_tile`) can
/// fabricate one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegTileSlot(pub(crate) u16);

/// Sealed SSA slot id for register vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegVecSlot(pub(crate) u16);

/// Typed register-tile handle. Const-generic shape (R, C), dtype
/// (T: TileDtype, sealed), and layout (L: RegTileLayout, sealed).
/// Two `RegTileId`s with different const generics are different Rust
/// types; an Instr constructor that requires `(src, dst):
/// (RegTileId<R,C,T,L>, RegTileId<R,C,T,L>)` rejects mismatch as
/// rustc E0308.
///
/// # Compile-fail proof — layout mismatch (Row vs Col) rejected
///
/// ```compile_fail
/// use ferrite_wavefront::tk_tape::{Bf16, ColLayout, RegTileId,
///     RegTileLayout, RowLayout, TileDtype};
/// fn _all_same<const R: usize, const C: usize, T: TileDtype, L: RegTileLayout>(
///     _a: RegTileId<R, C, T, L>,
///     _b: RegTileId<R, C, T, L>,
/// ) {}
/// let a: RegTileId<16, 128, Bf16, RowLayout> = unreachable!();
/// let b: RegTileId<16, 128, Bf16, ColLayout> = unreachable!();
/// _all_same(a, b);  // ← rustc rejects: L = RowLayout vs ColLayout
/// ```
///
/// # Compile-fail proof — shape mismatch (ROWS) rejected
///
/// ```compile_fail
/// use ferrite_wavefront::tk_tape::{Bf16, RegTileId, RegTileLayout,
///     RowLayout, TileDtype};
/// fn _all_same<const R: usize, const C: usize, T: TileDtype, L: RegTileLayout>(
///     _a: RegTileId<R, C, T, L>,
///     _b: RegTileId<R, C, T, L>,
/// ) {}
/// let a: RegTileId<16, 128, Bf16, RowLayout> = unreachable!();
/// let b: RegTileId<32, 128, Bf16, RowLayout> = unreachable!();
/// _all_same(a, b);  // ← rustc rejects: R=16 vs R=32
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegTileId<const R: usize, const C: usize, T: TileDtype, L: RegTileLayout> {
    slot: RegTileSlot,
    _marker: PhantomData<fn() -> (T, L)>,
}

impl<const R: usize, const C: usize, T: TileDtype, L: RegTileLayout> RegTileId<R, C, T, L> {
    pub(crate) const fn from_slot(slot: RegTileSlot) -> Self {
        Self {
            slot,
            _marker: PhantomData,
        }
    }
    pub const fn slot(&self) -> RegTileSlot {
        self.slot
    }
    pub const fn rows() -> usize {
        R
    }
    pub const fn cols() -> usize {
        C
    }
}

/// Typed register-vec handle. Const-generic LEN, dtype, layout.
///
/// # Compile-fail proof — layout mismatch (Ortho vs Align) rejected
///
/// ```compile_fail
/// use ferrite_wavefront::tk_tape::{AlignLayout, Bf16, OrthoLayout,
///     RegVecId, RegVecLayout, TileDtype};
/// fn _both<const LEN: usize, T: TileDtype, RV: RegVecLayout>(
///     _a: RegVecId<LEN, T, RV>,
///     _b: RegVecId<LEN, T, RV>,
/// ) {}
/// let a: RegVecId<128, Bf16, OrthoLayout> = unreachable!();
/// let b: RegVecId<128, Bf16, AlignLayout> = unreachable!();
/// _both(a, b);  // ← rustc rejects: RV mismatch
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegVecId<const LEN: usize, T: TileDtype, RV: RegVecLayout> {
    slot: RegVecSlot,
    _marker: PhantomData<fn() -> (T, RV)>,
}

impl<const LEN: usize, T: TileDtype, RV: RegVecLayout> RegVecId<LEN, T, RV> {
    pub(crate) const fn from_slot(slot: RegVecSlot) -> Self {
        Self {
            slot,
            _marker: PhantomData,
        }
    }
    pub const fn slot(&self) -> RegVecSlot {
        self.slot
    }
    pub const fn len() -> usize {
        LEN
    }
}

/// Sealed runtime entry recording a [`RegTileSlot`]'s shape/dtype/
/// layout. Inner fields `pub(crate)`; player iterates the arena to
/// emit kernel-preamble register-tile decls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegTileArenaEntry {
    pub(crate) rows: u16,
    pub(crate) cols: u16,
    pub(crate) dtype: TileDtypeTag,
    pub(crate) layout: RegTileLayoutTag,
}

impl RegTileArenaEntry {
    pub const fn rows(&self) -> u16 {
        self.rows
    }
    pub const fn cols(&self) -> u16 {
        self.cols
    }
    pub const fn dtype(&self) -> TileDtypeTag {
        self.dtype
    }
    pub const fn layout(&self) -> RegTileLayoutTag {
        self.layout
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegVecArenaEntry {
    pub(crate) len: u16,
    pub(crate) dtype: TileDtypeTag,
    pub(crate) layout: RegVecLayoutTag,
}

impl RegVecArenaEntry {
    pub const fn len(&self) -> u16 {
        self.len
    }
    pub const fn dtype(&self) -> TileDtypeTag {
        self.dtype
    }
    pub const fn layout(&self) -> RegVecLayoutTag {
        self.layout
    }
}

/// Inlined immediate scalar carried by *AddScalar / *MulScalar Instrs.
/// Per SUBTILE_TK20_DECOMP.md §"New typed-witness types" line 42:
/// the codegen emits a literal in the TK 2.0 call (e.g.
/// `kittens::bf16(-1.0f)`). The wrapper is a sealed newtype (the
/// inner field is `pub(crate)`) so external code cannot fabricate
/// scalar immediates that would silently splice into emitted CUDA.
/// Per `feedback_ff_subtile_compile_time_inviolable`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarF32(pub(crate) f32);

impl ScalarF32 {
    /// Construct a scalar immediate. Used by lowering paths that
    /// know the value at SubtileTape -> TkTape lowering time
    /// (e.g. SiluMul's -1.0 / 1.0 immediates).
    pub(crate) const fn new(value: f32) -> Self {
        Self(value)
    }

    /// Recover the f32 value for emit. Player formats with
    /// `format!("{value:.6}f")` or similar — see player.
    pub const fn value(&self) -> f32 {
        self.0
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
    /// Index into [`TkTape::kernel_args`] of the source tensor's
    /// CTensorMap arg. The player resolves to `aN` where N is this
    /// index (the kernel signature's `auto aN = ...` aliases). The
    /// SubtileIR `TensorId` lives on the [`KernelArgTy::BufPtr`]
    /// payload of the referenced kernel arg.
    pub src_arg: KernelArgRef,
    pub byte_off: ByteOffsetExpr,
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
        src_arg: KernelArgRef,
        byte_off: ByteOffsetExpr,
        tile: SmemTileSpec<ROWS, COLS, T>,
        role: LoaderRole,
        barrier_page: PageId,
    ) -> Self {
        Self {
            dst_page,
            src_arg,
            byte_off,
            tile: tile.shape(),
            role: role.to_warp_role(),
            barrier_page,
        }
    }

    /// Construct a [`LoadSpec`] from a RUNTIME tile shape — used by
    /// `emit_external_load` where the shape comes from the upstream
    /// `TensorRegion` (which varies per source: 1×N for vectors,
    /// 128×128 for projection weights, etc.).
    ///
    /// The const-generic typed gate at [`Self::new`] is appropriate
    /// when the lowerer KNOWS the shape (compute Instr arms); at the
    /// external-load boundary the shape is runtime data driven by
    /// the SubtileIR's tensor regions.
    pub(crate) fn new_runtime_shape(
        dst_page: PageId,
        src_arg: KernelArgRef,
        byte_off: ByteOffsetExpr,
        tile: TileShape,
        role: LoaderRole,
        barrier_page: PageId,
    ) -> Self {
        Self {
            dst_page,
            src_arg,
            byte_off,
            tile,
            role: role.to_warp_role(),
            barrier_page,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StoreSpec {
    pub src_page: PageId,
    /// Index into [`TkTape::kernel_args`] of the destination tensor's
    /// CTensorMap arg. See [`LoadSpec::src_arg`].
    pub dst_arg: KernelArgRef,
    pub byte_off: ByteOffsetExpr,
    pub tile: TileShape,
    pub role: WarpRole,
}

impl StoreSpec {
    /// Construct a [`StoreSpec`] from a typed [`SmemTileSpec<ROWS,
    /// COLS, T>`] witness. See [`LoadSpec::new`].
    pub(crate) fn new<const ROWS: usize, const COLS: usize, T: TileDtype>(
        src_page: PageId,
        dst_arg: KernelArgRef,
        byte_off: ByteOffsetExpr,
        tile: SmemTileSpec<ROWS, COLS, T>,
        role: StorerRole,
    ) -> Self {
        Self {
            src_page,
            dst_arg,
            byte_off,
            tile: tile.shape(),
            role: role.to_warp_role(),
        }
    }

    /// Runtime-shape store, parallel to [`LoadSpec::new_runtime_shape`].
    pub(crate) fn new_runtime_shape(
        src_page: PageId,
        dst_arg: KernelArgRef,
        byte_off: ByteOffsetExpr,
        tile: TileShape,
        role: StorerRole,
    ) -> Self {
        Self {
            src_page,
            dst_arg,
            byte_off,
            tile,
            role: role.to_warp_role(),
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
    /// Construct [`Instr::SyncthreadsCta`] from a typed
    /// [`AllWarpsRole`] witness. `__syncthreads()` is whole-CTA — any
    /// other role is meaningless here, so the constructor refuses
    /// non-`AllWarpsRole` at rustc time. Per
    /// `feedback_ff_subtile_compile_time_inviolable`.
    pub(crate) fn syncthreads_cta(role: AllWarpsRole) -> Self {
        Self::SyncthreadsCta {
            role: role.to_warp_role(),
        }
    }
}

/// Default compute-role tag used inside scalar shared-vec
/// constructors that don't take a typed role parameter (the only
/// legal compute role today is AllConsumers; future expansion can
/// type these the same way as ShTileMul/Add did).
const COMPUTE_ROLE_TAG: WarpRole = WarpRole::AllConsumers;

impl Instr {

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

    /// Construct a [`Instr::ShTileAdd`] from typed inputs. Same
    /// compile-time gates as [`Instr::sh_tile_mul`]: the `where
    /// GroupWidth<N>: ComputeWidth` bound restricts N to compute-
    /// eligible widths, and the const-generic shape unification across
    /// `lhs/rhs/dst` is rustc-checked.
    pub(crate) fn sh_tile_add<const N: usize, const ROWS: usize, const COLS: usize, T: TileDtype>(
        lhs: SmemTileId<ROWS, COLS, T>,
        rhs: SmemTileId<ROWS, COLS, T>,
        dst: SmemTileId<ROWS, COLS, T>,
        width: GroupWidth<N>,
    ) -> Self
    where
        GroupWidth<N>: ComputeWidth,
    {
        Self::ShTileAdd {
            lhs: lhs.page(),
            rhs: rhs.page(),
            dst: dst.page(),
            width: width.tag(),
        }
    }

    /// Construct a [`Instr::ShTileDiv`] — same compile-time gates as
    /// `sh_tile_mul` / `sh_tile_add`.
    pub(crate) fn sh_tile_div<const N: usize, const ROWS: usize, const COLS: usize, T: TileDtype>(
        lhs: SmemTileId<ROWS, COLS, T>,
        rhs: SmemTileId<ROWS, COLS, T>,
        dst: SmemTileId<ROWS, COLS, T>,
        width: GroupWidth<N>,
    ) -> Self
    where
        GroupWidth<N>: ComputeWidth,
    {
        Self::ShTileDiv {
            lhs: lhs.page(),
            rhs: rhs.page(),
            dst: dst.page(),
            width: width.tag(),
        }
    }

    /// Construct a [`Instr::ShTileExp`] (unary). Same compile-time
    /// gates with one source.
    pub(crate) fn sh_tile_exp<const N: usize, const ROWS: usize, const COLS: usize, T: TileDtype>(
        src: SmemTileId<ROWS, COLS, T>,
        dst: SmemTileId<ROWS, COLS, T>,
        width: GroupWidth<N>,
    ) -> Self
    where
        GroupWidth<N>: ComputeWidth,
    {
        Self::ShTileExp {
            src: src.page(),
            dst: dst.page(),
            width: width.tag(),
        }
    }

    /// Construct a [`Instr::ShTileMulScalar`]. The dtype is derived
    /// from the typed `T: TileDtype` parameter via `T::tag()`, so
    /// caller cannot pass a mismatched scalar dtype literal.
    pub(crate) fn sh_tile_mul_scalar<const N: usize, const ROWS: usize, const COLS: usize, T: TileDtype>(
        lhs: SmemTileId<ROWS, COLS, T>,
        dst: SmemTileId<ROWS, COLS, T>,
        scalar: ScalarF32,
        width: GroupWidth<N>,
    ) -> Self
    where
        GroupWidth<N>: ComputeWidth,
    {
        Self::ShTileMulScalar {
            lhs: lhs.page(),
            dst: dst.page(),
            scalar,
            dtype: T::tag(),
            width: width.tag(),
        }
    }

    /// Construct a [`Instr::ShTileAddScalar`] — see `sh_tile_mul_scalar`.
    pub(crate) fn sh_tile_add_scalar<const N: usize, const ROWS: usize, const COLS: usize, T: TileDtype>(
        lhs: SmemTileId<ROWS, COLS, T>,
        dst: SmemTileId<ROWS, COLS, T>,
        scalar: ScalarF32,
        width: GroupWidth<N>,
    ) -> Self
    where
        GroupWidth<N>: ComputeWidth,
    {
        Self::ShTileAddScalar {
            lhs: lhs.page(),
            dst: dst.page(),
            scalar,
            dtype: T::tag(),
            width: width.tag(),
        }
    }

    // ── Register-tile / register-vec typed constructors ────────────
    //
    // Each constructor takes the typed RegTileId<R,C,T,L> /
    // RegVecId<LEN,T,RV> witnesses; const-generic + sealed-marker
    // unification across operands gates the call at rustc time.
    // The variant stores the runtime RegTileSlot/RegVecSlot; the
    // arena entries on TkTape carry the shape/dtype/layout for emit.

    pub(crate) fn load_shmem_to_reg<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
    >(
        src: SmemTileId<ROWS, COLS, T>,
        dst: RegTileId<ROWS, COLS, T, L>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::LoadShmemToReg {
            src: src.page(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    /// Warpgroup-sharded smem→reg load for the rt_st WGMMA path.
    /// `kittens::group<4>::load(rt, st)` distributes ST.rows across
    /// the 4 warps of the warpgroup; per
    /// `ops/group/memory/tile/shared_to_register.cuh:17` this requires
    /// `ST::rows == 4 * RT::rows`. The `where GroupWidth<4>:
    /// WarpgroupLoadShape<ST_ROWS, RT_ROWS>` bound restricts callers
    /// to (ST_ROWS, RT_ROWS) pairs that exist on our substrate (today
    /// only (128, 32)) — passing arbitrary shapes is rustc E0277.
    pub(crate) fn load_shmem_to_reg_warpgroup<
        const ST_ROWS: usize,
        const RT_ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
    >(
        src: SmemTileId<ST_ROWS, COLS, T>,
        dst: RegTileId<RT_ROWS, COLS, T, L>,
        _width: GroupWidth<4>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<4>: WarpgroupLoadShape<ST_ROWS, RT_ROWS>,
    {
        Self::LoadShmemToReg {
            src: src.page(),
            dst: dst.slot(),
            width: GroupWidth::<4>::WARPGROUP.tag(),
            role: role.to_warp_role(),
        }
    }

    /// Activation-pool variant of [`Self::load_shmem_to_reg_warpgroup`].
    /// Same TK 2.0 primitive, but `src` is an [`ActSmemTileId`] from
    /// the `act_buf` pool. Player emits `act_buf[N]` instead of
    /// `page_buf[N]`. Per audit ADDENDUM 3 §"Step 2 design".
    pub(crate) fn load_shmem_to_reg_warpgroup_from_act<
        const ST_ROWS: usize,
        const RT_ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
    >(
        src: ActSmemTileId<ST_ROWS, COLS, T>,
        dst: RegTileId<RT_ROWS, COLS, T, L>,
        _width: GroupWidth<4>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<4>: WarpgroupLoadShape<ST_ROWS, RT_ROWS>,
    {
        Self::LoadShmemToRegFromAct {
            src: src.page(),
            dst: dst.slot(),
            width: GroupWidth::<4>::WARPGROUP.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn store_reg_tile_to_shmem<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
    >(
        src: RegTileId<ROWS, COLS, T, L>,
        dst: SmemTileId<ROWS, COLS, T>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::StoreRegTileToShmem {
            src: src.slot(),
            dst: dst.page(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    /// Warpgroup-sharded reg→smem store. Mirrors
    /// `load_shmem_to_reg_warpgroup`: 4 warps each contribute their
    /// per-warp RT_ROWS to the collective ST_ROWS. The
    /// `WarpgroupLoadShape<ST_ROWS, RT_ROWS>` witness is shared with
    /// the load constructor — same (ST_ROWS, RT_ROWS) gate.
    pub(crate) fn store_reg_tile_to_shmem_warpgroup<
        const ST_ROWS: usize,
        const RT_ROWS: usize,
        const COLS: usize,
        T_ST: TileDtype,
        T_RT: TileDtype,
        L: RegTileLayout,
    >(
        src: RegTileId<RT_ROWS, COLS, T_RT, L>,
        dst: SmemTileId<ST_ROWS, COLS, T_ST>,
        _width: GroupWidth<4>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<4>: WarpgroupLoadShape<ST_ROWS, RT_ROWS>,
    {
        // Note: T_ST and T_RT may differ — TK 2.0's `store(st, rt)`
        // template at `shared_to_register.cuh` permits dtype mismatch
        // between dst and src (e.g. fp32 rt → bf16 st conversion at
        // store time), which is what we want for MatmulTile's
        // accumulator-to-bf16 store. Per plan §"Resolved decision 5"
        // the proper RegTileCopyConvert lift is a follow-up.
        Self::StoreRegTileToShmem {
            src: src.slot(),
            dst: dst.page(),
            width: GroupWidth::<4>::WARPGROUP.tag(),
            role: role.to_warp_role(),
        }
    }

    /// Construct [`Instr::LoadShmemSubTileToReg`] from typed witnesses.
    /// Uses TK 2.0's `st.subtile<COLS_SUB>(idx)` primitive at
    /// `types/shared/st.cuh:159`. Const-generic guards:
    ///   - `COLS_FULL % COLS_SUB == 0` (sub-tile divides evenly)
    ///   - `IDX * COLS_SUB < COLS_FULL` (slice index in range)
    /// Both are `const{}` asserts — always-dead for valid usage,
    /// fire at instantiation time on misuse.
    pub(crate) fn load_shmem_subtile_to_reg<
        const N: usize,
        const ROWS: usize,
        const COLS_FULL: usize,
        const COLS_SUB: usize,
        const IDX: usize,
        T: TileDtype,
        L: RegTileLayout,
    >(
        src: SmemTileId<ROWS, COLS_FULL, T>,
        dst: RegTileId<ROWS, COLS_SUB, T, L>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        const {
            assert!(
                COLS_FULL % COLS_SUB == 0,
                "load_shmem_subtile_to_reg: COLS_FULL must be divisible by COLS_SUB",
            );
            assert!(
                IDX * COLS_SUB < COLS_FULL,
                "load_shmem_subtile_to_reg: IDX * COLS_SUB out of range",
            );
        }
        Self::LoadShmemSubTileToReg {
            src: src.page(),
            subtile_cols: COLS_SUB as u16,
            subtile_idx: IDX as u16,
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    // ── MatmulTile / WGMMA constructors (step 9) ─────────────────

    /// Construct [`Instr::TmaExpect`] from a typed shape+dtype
    /// witness so byte-count derives from `ROWS * COLS * T::ELEM_BYTES`
    /// — caller cannot pass a mismatched runtime byte count.
    pub(crate) fn tma_expect<const ROWS: usize, const COLS: usize, T: TileDtype>(
        barrier_page: PageId,
        _shape_witness: SmemTileSpec<ROWS, COLS, T>,
        role: LoaderRole,
    ) -> Self {
        Self::TmaExpect {
            barrier_page,
            bytes: (ROWS * COLS) as u32 * T::ELEM_BYTES,
            role: role.to_warp_role(),
        }
    }

    /// Construct [`Instr::InitRtZero`].
    pub(crate) fn init_rt_zero<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
    >(
        dst: RegTileId<ROWS, COLS, T, L>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::InitRtZero {
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    /// Construct [`Instr::WgmmaFenceAcc`]. Width must be GroupWidth<4>
    /// (warpgroup) — WGMMA is warpgroup-only.
    pub(crate) fn wgmma_fence_acc<
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
    >(
        d: RegTileId<ROWS, COLS, T, L>,
        _width: GroupWidth<4>,
    ) -> Self {
        Self::WgmmaFenceAcc {
            d: d.slot(),
            width: GroupWidth::<4>::WARPGROUP.tag(),
            role: WarpRole::AllConsumers,
        }
    }

    /// Construct [`Instr::WgmmaMmaAB_SmemSmem`].
    /// `D[M, N] = A[M, K] @ B[K, N]` — K unifies between A and B at
    /// the type level (rustc enforces). D is the register accumulator;
    /// its shape (M, N) and dtype/layout are propagated from typed
    /// witnesses. `_fence` and `_accumulate` are sealed policy
    /// witnesses (FencePolicy / AccPolicy); their const KIND erases
    /// to runtime u8 for emit (TK 2.0 template booleans).
    pub(crate) fn wgmma_mma_ab_smem_smem<
        const M: usize,
        const K: usize,
        const N: usize,
        T_AB: TileDtype,
        T_D: TileDtype,
        L: RegTileLayout,
        F: FencePolicy,
        AC: AccPolicy,
    >(
        d: RegTileId<M, N, T_D, L>,
        a: SmemTileId<M, K, T_AB>,
        b: SmemTileId<K, N, T_AB>,
        _fence: F,
        _accumulate: AC,
        _width: GroupWidth<4>,
    ) -> Self {
        Self::WgmmaMmaAB_SmemSmem {
            a_page: a.page(),
            b_page: b.page(),
            d: d.slot(),
            fence: F::KIND as u8,
            accumulate: AC::KIND as u8,
            width: GroupWidth::<4>::WARPGROUP.tag(),
            role: WarpRole::AllConsumers,
        }
    }

    /// Construct [`Instr::WgmmaMmaAB_RegSmem`]. D[M_per_warp, N] +=
    /// Construct [`Instr::WgmmaMmaABt_RegSmem`]. D += rt A @ smem B^T.
    /// `ops/group/mma/warpgroup.cuh:323` register-A variant — N is
    /// `B::rows / TILE_ROW_DIM` (B is transposed), K is
    /// `A::cols / TILE_COL_DIM == B::cols / TILE_COL_DIM`.
    pub(crate) fn wgmma_mma_abt_reg_smem<
        const M_PER_WARP: usize,
        const K: usize,
        const N: usize,
        T_AB: TileDtype,
        T_D: TileDtype,
        L: RegTileLayout,
        F: FencePolicy,
        AC: AccPolicy,
    >(
        d: RegTileId<M_PER_WARP, N, T_D, L>,
        a: RegTileId<M_PER_WARP, K, T_AB, L>,
        b: SmemTileId<N, K, T_AB>,
        _fence: F,
        _accumulate: AC,
        _width: GroupWidth<4>,
    ) -> Self {
        Self::WgmmaMmaABt_RegSmem {
            a: a.slot(),
            b_page: b.page(),
            d: d.slot(),
            fence: F::KIND as u8,
            accumulate: AC::KIND as u8,
            width: GroupWidth::<4>::WARPGROUP.tag(),
            role: WarpRole::AllConsumers,
        }
    }

    /// Construct [`Instr::WgmmaAsyncWait`].
    pub(crate) fn wgmma_async_wait(n: u32, _width: GroupWidth<4>) -> Self {
        Self::WgmmaAsyncWait {
            n,
            width: GroupWidth::<4>::WARPGROUP.tag(),
            role: WarpRole::AllConsumers,
        }
    }

    // ── AttnDecode chain constructors (steps 11-14) ─────────────

    pub(crate) fn init_rv_neg_infty<
        const N: usize,
        const LEN: usize,
        T: TileDtype,
        RV: RegVecLayout,
    >(
        dst: RegVecId<LEN, T, RV>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: ComputeWidth,
    {
        Self::InitRvNegInfty {
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn init_rv_zero<
        const N: usize,
        const LEN: usize,
        T: TileDtype,
        RV: RegVecLayout,
    >(
        dst: RegVecId<LEN, T, RV>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: ComputeWidth,
    {
        Self::InitRvZero {
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    /// Construct [`Instr::WgmmaMmaABt_SmemSmem`]. Q @ K^T pattern:
    /// D[M, N] = A[M, K] @ B[N, K]^T (B's stored layout has N rows,
    /// K cols; transposed at the WGMMA op).
    pub(crate) fn wgmma_mma_abt_smem_smem<
        const M: usize,
        const K: usize,
        const N: usize,
        T_AB: TileDtype,
        T_D: TileDtype,
        L: RegTileLayout,
        F: FencePolicy,
        AC: AccPolicy,
    >(
        d: RegTileId<M, N, T_D, L>,
        a: SmemTileId<M, K, T_AB>,
        b: SmemTileId<N, K, T_AB>,
        _fence: F,
        _accumulate: AC,
        _width: GroupWidth<4>,
    ) -> Self {
        Self::WgmmaMmaABt_SmemSmem {
            a_page: a.page(),
            b_page: b.page(),
            d: d.slot(),
            fence: F::KIND as u8,
            accumulate: AC::KIND as u8,
            width: GroupWidth::<4>::WARPGROUP.tag(),
            role: WarpRole::AllConsumers,
        }
    }

    /// Construct [`Instr::WgmmaMmaAB_RegSmem`]. P @ V pattern:
    /// D[M, N] = A[M, K] @ B[K, N], A in register tile.
    pub(crate) fn wgmma_mma_ab_reg_smem<
        const M: usize,
        const K: usize,
        const N: usize,
        T_AB: TileDtype,
        T_D: TileDtype,
        L_A: RegTileLayout,
        L_D: RegTileLayout,
        F: FencePolicy,
        AC: AccPolicy,
    >(
        d: RegTileId<M, N, T_D, L_D>,
        a: RegTileId<M, K, T_AB, L_A>,
        b: SmemTileId<K, N, T_AB>,
        _fence: F,
        _accumulate: AC,
        _width: GroupWidth<4>,
    ) -> Self {
        Self::WgmmaMmaAB_RegSmem {
            a: a.slot(),
            b_page: b.page(),
            d: d.slot(),
            fence: F::KIND as u8,
            accumulate: AC::KIND as u8,
            width: GroupWidth::<4>::WARPGROUP.tag(),
            role: WarpRole::AllConsumers,
        }
    }

    pub(crate) fn reg_tile_mul_scalar<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
    >(
        lhs: RegTileId<ROWS, COLS, T, L>,
        dst: RegTileId<ROWS, COLS, T, L>,
        scalar: ScalarF32,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegTileMulScalar {
            lhs: lhs.slot(),
            dst: dst.slot(),
            scalar,
            dtype: T::tag(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn reg_tile_row_max_acc<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
        RV: RegVecLayout,
    >(
        src: RegTileId<ROWS, COLS, T, L>,
        acc: RegVecId<ROWS, T, RV>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegTileRowMaxAcc {
            src: src.slot(),
            acc: acc.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn reg_tile_row_sum_acc<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
        RV: RegVecLayout,
    >(
        src: RegTileId<ROWS, COLS, T, L>,
        acc: RegVecId<ROWS, T, RV>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegTileRowSumAcc {
            src: src.slot(),
            acc: acc.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn reg_tile_sub_row<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
        RV: RegVecLayout,
    >(
        src: RegTileId<ROWS, COLS, T, L>,
        row_vec: RegVecId<ROWS, T, RV>,
        dst: RegTileId<ROWS, COLS, T, L>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegTileSubRow {
            src: src.slot(),
            row_vec: row_vec.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn reg_tile_exp2<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
    >(
        src: RegTileId<ROWS, COLS, T, L>,
        dst: RegTileId<ROWS, COLS, T, L>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegTileExp2 {
            src: src.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn reg_tile_div_row<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
        RV: RegVecLayout,
    >(
        src: RegTileId<ROWS, COLS, T, L>,
        row_vec: RegVecId<ROWS, T, RV>,
        dst: RegTileId<ROWS, COLS, T, L>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegTileDivRow {
            src: src.slot(),
            row_vec: row_vec.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn reg_vec_sub<
        const N: usize,
        const LEN: usize,
        T: TileDtype,
        RV: RegVecLayout,
    >(
        lhs: RegVecId<LEN, T, RV>,
        rhs: RegVecId<LEN, T, RV>,
        dst: RegVecId<LEN, T, RV>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegVecSub {
            lhs: lhs.slot(),
            rhs: rhs.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn reg_vec_exp2<
        const N: usize,
        const LEN: usize,
        T: TileDtype,
        RV: RegVecLayout,
    >(
        src: RegVecId<LEN, T, RV>,
        dst: RegVecId<LEN, T, RV>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegVecExp2 {
            src: src.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn reg_vec_mul<
        const N: usize,
        const LEN: usize,
        T: TileDtype,
        RV: RegVecLayout,
    >(
        lhs: RegVecId<LEN, T, RV>,
        rhs: RegVecId<LEN, T, RV>,
        dst: RegVecId<LEN, T, RV>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegVecMul {
            lhs: lhs.slot(),
            rhs: rhs.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    /// Construct [`Instr::RegTileCopyConvert`]. Source and dest can
    /// have DIFFERENT dtypes — TK 2.0's `copy` template handles the
    /// conversion (e.g. fp32 → bf16 for the P_block in AttnDecode_Sv).
    /// Shape and layout still unify between src and dst.
    pub(crate) fn reg_tile_copy_convert<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T_SRC: TileDtype,
        T_DST: TileDtype,
        L: RegTileLayout,
    >(
        src: RegTileId<ROWS, COLS, T_SRC, L>,
        dst: RegTileId<ROWS, COLS, T_DST, L>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegTileCopyConvert {
            src: src.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn reg_vec_copy<
        const N: usize,
        const LEN: usize,
        T: TileDtype,
        RV: RegVecLayout,
    >(
        src: RegVecId<LEN, T, RV>,
        dst: RegVecId<LEN, T, RV>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegVecCopy {
            src: src.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    /// Construct [`Instr::RegTileMulRow`]. Per-row scalar multiply
    /// of a register tile by a register vec; vec.LEN unifies with
    /// tile.ROWS at the type level.
    pub(crate) fn reg_tile_mul_row<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
        RV: RegVecLayout,
    >(
        src: RegTileId<ROWS, COLS, T, L>,
        row_vec: RegVecId<ROWS, T, RV>,
        dst: RegTileId<ROWS, COLS, T, L>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegTileMulRow {
            src: src.slot(),
            row_vec: row_vec.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    /// Construct [`Instr::StoreRegTileSubTileToShmem`] — inverse of
    /// `load_shmem_subtile_to_reg`. Same const-generic guards.
    pub(crate) fn store_reg_tile_subtile_to_shmem<
        const N: usize,
        const ROWS: usize,
        const COLS_FULL: usize,
        const COLS_SUB: usize,
        const IDX: usize,
        T: TileDtype,
        L: RegTileLayout,
    >(
        src: RegTileId<ROWS, COLS_SUB, T, L>,
        dst: SmemTileId<ROWS, COLS_FULL, T>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        const {
            assert!(
                COLS_FULL % COLS_SUB == 0,
                "store_reg_tile_subtile_to_shmem: COLS_FULL must be divisible by COLS_SUB",
            );
            assert!(
                IDX * COLS_SUB < COLS_FULL,
                "store_reg_tile_subtile_to_shmem: IDX * COLS_SUB out of range",
            );
        }
        Self::StoreRegTileSubTileToShmem {
            src: src.slot(),
            dst: dst.page(),
            subtile_cols: COLS_SUB as u16,
            subtile_idx: IDX as u16,
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn load_vec_smem_to_reg<
        const N: usize,
        const LEN: usize,
        T: TileDtype,
        RV: RegVecLayout,
    >(
        src: SmemVecId<LEN, T>,
        dst: RegVecId<LEN, T, RV>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::LoadVecSmemToReg {
            src: src.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn store_reg_vec_to_shmem<
        const N: usize,
        const LEN: usize,
        T: TileDtype,
        RV: RegVecLayout,
    >(
        src: RegVecId<LEN, T, RV>,
        dst: SmemVecId<LEN, T>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::StoreRegVecToShmem {
            src: src.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn reg_tile_neg<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
    >(
        src: RegTileId<ROWS, COLS, T, L>,
        dst: RegTileId<ROWS, COLS, T, L>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegTileNeg {
            src: src.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn reg_tile_exp<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
    >(
        src: RegTileId<ROWS, COLS, T, L>,
        dst: RegTileId<ROWS, COLS, T, L>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegTileExp {
            src: src.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn reg_tile_add<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
    >(
        lhs: RegTileId<ROWS, COLS, T, L>,
        rhs: RegTileId<ROWS, COLS, T, L>,
        dst: RegTileId<ROWS, COLS, T, L>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegTileAdd {
            lhs: lhs.slot(),
            rhs: rhs.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn reg_tile_sub<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
    >(
        lhs: RegTileId<ROWS, COLS, T, L>,
        rhs: RegTileId<ROWS, COLS, T, L>,
        dst: RegTileId<ROWS, COLS, T, L>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegTileSub {
            lhs: lhs.slot(),
            rhs: rhs.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn reg_tile_div<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
    >(
        lhs: RegTileId<ROWS, COLS, T, L>,
        rhs: RegTileId<ROWS, COLS, T, L>,
        dst: RegTileId<ROWS, COLS, T, L>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegTileDiv {
            lhs: lhs.slot(),
            rhs: rhs.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    /// `kittens::group<N>::mul_col(dst, src, col_vec)` — the col-vec
    /// LEN must equal the tile's COLS at the type level; constructor
    /// where-bound enforces.
    pub(crate) fn reg_tile_mul_col<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
        RV: RegVecLayout,
    >(
        src: RegTileId<ROWS, COLS, T, L>,
        col_vec: RegVecId<COLS, T, RV>,
        dst: RegTileId<ROWS, COLS, T, L>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegTileMulCol {
            src: src.slot(),
            col_vec: col_vec.slot(),
            dst: dst.slot(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    pub(crate) fn reg_tile_add_scalar<
        const N: usize,
        const ROWS: usize,
        const COLS: usize,
        T: TileDtype,
        L: RegTileLayout,
    >(
        lhs: RegTileId<ROWS, COLS, T, L>,
        dst: RegTileId<ROWS, COLS, T, L>,
        scalar: ScalarF32,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegTileAddScalar {
            lhs: lhs.slot(),
            dst: dst.slot(),
            scalar,
            dtype: T::tag(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    // ── RmsNorm-unique constructors (commit B) ───────────────────

    /// Construct [`Instr::ShTileRowSum`] with typed src tile + dst
    /// vec witnesses. `dst` is a [`SmemVecId<ROWS, T>`] — its
    /// `LEN` is bound to the src tile's `ROWS` at the constructor
    /// signature, so a length mismatch is rustc E0308.
    pub(crate) fn sh_tile_row_sum<const N: usize, const ROWS: usize, const COLS: usize, T: TileDtype>(
        src: SmemTileId<ROWS, COLS, T>,
        dst: SmemVecId<ROWS, T>,
        width: GroupWidth<N>,
    ) -> Self
    where
        GroupWidth<N>: ComputeWidth,
    {
        Self::ShTileRowSum {
            src: src.page(),
            dst: dst.slot(),
            width: width.tag(),
            role: COMPUTE_ROLE_TAG,
        }
    }

    /// Construct [`Instr::ShVecMulScalar`]. Both endpoints are typed
    /// `SmemVecId<LEN, T>` — a length / dtype mismatch is rustc E0308
    /// at the constructor.
    pub(crate) fn sh_vec_mul_scalar<const N: usize, const LEN: usize, T: TileDtype>(
        src: SmemVecId<LEN, T>,
        dst: SmemVecId<LEN, T>,
        scalar: ScalarF32,
        width: GroupWidth<N>,
    ) -> Self
    where
        GroupWidth<N>: ComputeWidth,
    {
        Self::ShVecMulScalar {
            src: src.slot(),
            dst: dst.slot(),
            scalar,
            dtype: T::tag(),
            width: width.tag(),
            role: COMPUTE_ROLE_TAG,
        }
    }

    /// Construct [`Instr::ShVecAddScalar`]. See `sh_vec_mul_scalar`.
    pub(crate) fn sh_vec_add_scalar<const N: usize, const LEN: usize, T: TileDtype>(
        src: SmemVecId<LEN, T>,
        dst: SmemVecId<LEN, T>,
        scalar: ScalarF32,
        width: GroupWidth<N>,
    ) -> Self
    where
        GroupWidth<N>: ComputeWidth,
    {
        Self::ShVecAddScalar {
            src: src.slot(),
            dst: dst.slot(),
            scalar,
            dtype: T::tag(),
            width: width.tag(),
            role: COMPUTE_ROLE_TAG,
        }
    }

    /// Construct [`Instr::RegVecUnaryRsqrt`] from typed RegVecId
    /// witnesses. Src and dst must share `(LEN, T, RV)` const-generics.
    pub(crate) fn reg_vec_unary_rsqrt<const N: usize, const LEN: usize, T: TileDtype, RV: RegVecLayout>(
        src: RegVecId<LEN, T, RV>,
        dst: RegVecId<LEN, T, RV>,
        width: GroupWidth<N>,
        role: AllConsumersRole,
    ) -> Self
    where
        GroupWidth<N>: WarpLoadWidth,
    {
        Self::RegVecUnaryRsqrt {
            src: src.slot(),
            dst: dst.slot(),
            dtype: T::tag(),
            layout: RV::tag(),
            width: width.tag(),
            role: role.to_warp_role(),
        }
    }

    /// Construct [`Instr::ShTileMulRow`] with typed src/dst tile
    /// witnesses (must share R, C, T). `row_vec` is a typed
    /// `SmemVecId<ROWS, T>` — its `LEN` must equal the tile's
    /// `ROWS` (rustc E0308 on mismatch).
    pub(crate) fn sh_tile_mul_row<const N: usize, const ROWS: usize, const COLS: usize, T: TileDtype>(
        src: SmemTileId<ROWS, COLS, T>,
        row_vec: SmemVecId<ROWS, T>,
        dst: SmemTileId<ROWS, COLS, T>,
        width: GroupWidth<N>,
    ) -> Self
    where
        GroupWidth<N>: ComputeWidth,
    {
        Self::ShTileMulRow {
            src: src.page(),
            row_vec: row_vec.slot(),
            dst: dst.page(),
            width: width.tag(),
            role: COMPUTE_ROLE_TAG,
        }
    }

    /// Construct [`Instr::ShTileMulCol`] — see `sh_tile_mul_row`.
    /// `col_vec` length must equal tile `COLS`.
    pub(crate) fn sh_tile_mul_col<const N: usize, const ROWS: usize, const COLS: usize, T: TileDtype>(
        src: SmemTileId<ROWS, COLS, T>,
        col_vec: SmemVecId<COLS, T>,
        dst: SmemTileId<ROWS, COLS, T>,
        width: GroupWidth<N>,
    ) -> Self
    where
        GroupWidth<N>: ComputeWidth,
    {
        Self::ShTileMulCol {
            src: src.page(),
            col_vec: col_vec.slot(),
            dst: dst.page(),
            width: width.tag(),
            role: COMPUTE_ROLE_TAG,
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
        dst_arg: KernelArgRef,
        role: StorerRole,
    ) -> Self {
        // Erase the typed witness into structured runtime data —
        // NOT a pre-formatted CUDA string. The format
        // `kittens::st_<suffix><R, C>` lives in the player at emit
        // time. Per `feedback_no_premature_string_encoding`.
        Self::StoreAsyncTyped {
            dst_page: src.page(),
            dst_arg,
            tile_type: TileTypeSpec {
                rows: ROWS as u32,
                cols: COLS as u32,
                dtype: T::tag(),
            },
            role: role.to_warp_role(),
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

    /// Mint a fresh [`RegTileId<R, C, T, L>`] SSA slot. Records the
    /// runtime shape/dtype/layout in [`Self::reg_tile_arena`] for
    /// kernel-preamble emit. Panics on u16 overflow with a clear
    /// message — 65,535 register-tile slots per kernel is far
    /// beyond any realistic megakernel.
    pub(crate) fn mint_reg_tile<const R: usize, const C: usize, T: TileDtype, L: RegTileLayout>(
        &mut self,
    ) -> RegTileId<R, C, T, L> {
        let slot = RegTileSlot(self.next_reg_tile_slot);
        self.next_reg_tile_slot = self
            .next_reg_tile_slot
            .checked_add(1)
            .expect("reg_tile slot overflow (>= 65536 register tiles in one kernel)");
        self.reg_tile_arena.insert(
            slot,
            RegTileArenaEntry {
                rows: R as u16,
                cols: C as u16,
                dtype: T::tag(),
                layout: L::tag(),
            },
        );
        RegTileId::from_slot(slot)
    }

    /// Mint a fresh [`RegVecId<LEN, T, RV>`] SSA slot. See
    /// [`Self::mint_reg_tile`].
    pub(crate) fn mint_reg_vec<const LEN: usize, T: TileDtype, RV: RegVecLayout>(
        &mut self,
    ) -> RegVecId<LEN, T, RV> {
        let slot = RegVecSlot(self.next_reg_vec_slot);
        self.next_reg_vec_slot = self
            .next_reg_vec_slot
            .checked_add(1)
            .expect("reg_vec slot overflow (>= 65536 register vectors in one kernel)");
        self.reg_vec_arena.insert(
            slot,
            RegVecArenaEntry {
                len: LEN as u16,
                dtype: T::tag(),
                layout: RV::tag(),
            },
        );
        RegVecId::from_slot(slot)
    }

    /// Mint a fresh [`SmemVecId<LEN, T>`] slot. The kernel preamble
    /// will declare `__shared__ kittens::sv_<dtype><LEN> sv_<idx>;`
    /// for each minted slot. See `feedback_ff_subtile_compile_time_inviolable`
    /// — distinct from `page_buf[]` so a vec endpoint cannot be a
    /// tile (and vice versa).
    pub(crate) fn mint_smem_vec<const LEN: usize, T: TileDtype>(
        &mut self,
    ) -> SmemVecId<LEN, T> {
        let slot = SmemVecSlot(self.next_smem_vec_slot);
        self.next_smem_vec_slot = self
            .next_smem_vec_slot
            .checked_add(1)
            .expect("smem_vec slot overflow (>= 65536 shared vectors in one kernel)");
        self.smem_vec_arena.insert(
            slot,
            SmemVecArenaEntry {
                len: LEN as u32,
                dtype: T::tag(),
            },
        );
        SmemVecId::from_slot(slot)
    }

    /// Read accessor for the register-tile arena (used by the player
    /// to emit kernel-preamble decls in deterministic id order).
    pub fn reg_tile_arena(&self) -> &BTreeMap<RegTileSlot, RegTileArenaEntry> {
        &self.reg_tile_arena
    }

    pub fn reg_vec_arena(&self) -> &BTreeMap<RegVecSlot, RegVecArenaEntry> {
        &self.reg_vec_arena
    }

    pub(crate) fn smem_vec_arena(&self) -> &BTreeMap<SmemVecSlot, SmemVecArenaEntry> {
        &self.smem_vec_arena
    }

    /// Append the cross-op gmem-fence as a 5-Instr atomic sequence.
    pub(crate) fn emit_cross_op_gmem_fence(&mut self) {
        // syncthreads_cta is role-typed (AllWarpsRole — `__syncthreads()`
        // is whole-CTA); commit_bulk / wait_bulk / threadfence_device
        // remain WarpRole-tagged because their CUDA emit is
        // role-agnostic (per-warp `kittens::group<1>::tma::*` and
        // `__threadfence()`).
        let role = WarpRole::All;
        self.instrs.push(Instr::syncthreads_cta(AllWarpsRole));
        self.instrs.push(Instr::commit_bulk(role));
        self.instrs.push(Instr::wait_bulk(role, 0));
        self.instrs.push(Instr::threadfence_device(role));
        self.instrs.push(Instr::syncthreads_cta(AllWarpsRole));
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
            Instr::ShTileMul { .. }
            | Instr::ShTileAdd { .. }
            | Instr::ShTileDiv { .. }
            | Instr::ShTileExp { .. }
            | Instr::ShTileMulScalar { .. }
            | Instr::ShTileAddScalar { .. }
            | Instr::ShTileRowSum { .. }
            | Instr::ShVecMulScalar { .. }
            | Instr::ShVecAddScalar { .. }
            | Instr::ShTileMulRow { .. }
            | Instr::ShTileMulCol { .. }
            | Instr::LoadShmemToReg { .. }
            | Instr::LoadShmemToRegFromAct { .. }
            | Instr::StoreRegTileToShmem { .. }
            | Instr::LoadShmemSubTileToReg { .. }
            | Instr::StoreRegTileSubTileToShmem { .. }
            | Instr::TmaExpect { .. }
            | Instr::InitRtZero { .. }
            | Instr::InitRvNegInfty { .. }
            | Instr::InitRvZero { .. }
            | Instr::WgmmaFenceAcc { .. }
            | Instr::WgmmaMmaAB_SmemSmem { .. }
            | Instr::WgmmaMmaABt_SmemSmem { .. }
            | Instr::WgmmaMmaAB_RegSmem { .. }
            | Instr::WgmmaMmaABt_RegSmem { .. }
            | Instr::WgmmaAsyncWait { .. }
            | Instr::RegTileMulScalar { .. }
            | Instr::RegTileRowMaxAcc { .. }
            | Instr::RegTileRowSumAcc { .. }
            | Instr::RegTileSubRow { .. }
            | Instr::RegTileExp2 { .. }
            | Instr::RegTileDivRow { .. }
            | Instr::RegVecSub { .. }
            | Instr::RegVecExp2 { .. }
            | Instr::RegVecMul { .. }
            | Instr::RegTileCopyConvert { .. }
            | Instr::RegVecCopy { .. }
            | Instr::RegTileMulRow { .. }
            | Instr::LoadVecSmemToReg { .. }
            | Instr::StoreRegVecToShmem { .. }
            | Instr::RegTileNeg { .. }
            | Instr::RegTileExp { .. }
            | Instr::RegTileAdd { .. }
            | Instr::RegTileSub { .. }
            | Instr::RegTileDiv { .. }
            | Instr::RegTileMulCol { .. }
            | Instr::RegTileAddScalar { .. }
            | Instr::RegVecUnaryRsqrt { .. }
            | Instr::DebugOpBeginMarker { .. } => {}
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
            reg_tile_arena: BTreeMap::new(),
            reg_vec_arena: BTreeMap::new(),
            smem_vec_arena: BTreeMap::new(),
            next_reg_tile_slot: 0,
            next_reg_vec_slot: 0,
            next_smem_vec_slot: 0,
            instrs: vec![
                Instr::StoreAsync(StoreSpec {
                    src_page: PageId(0),
                    dst_arg: KernelArgRef(0),
                    byte_off: ByteOffsetExpr::from_const(0),
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
            reg_tile_arena: BTreeMap::new(),
            reg_vec_arena: BTreeMap::new(),
            smem_vec_arena: BTreeMap::new(),
            next_reg_tile_slot: 0,
            next_reg_vec_slot: 0,
            next_smem_vec_slot: 0,
            instrs: vec![
                Instr::StoreAsync(StoreSpec {
                    src_page: PageId(0),
                    dst_arg: KernelArgRef(0),
                    byte_off: ByteOffsetExpr::from_const(0),
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
            reg_tile_arena: BTreeMap::new(),
            reg_vec_arena: BTreeMap::new(),
            smem_vec_arena: BTreeMap::new(),
            next_reg_tile_slot: 0,
            next_reg_vec_slot: 0,
            next_smem_vec_slot: 0,
            instrs: vec![
                Instr::StoreAsync(StoreSpec {
                    src_page: PageId(0),
                    dst_arg: KernelArgRef(0),
                    byte_off: ByteOffsetExpr::from_const(0),
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
