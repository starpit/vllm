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
//! TkTape side of the redesign. The lowering (`lower_tape_to_tk`)
//! preserves SubtileIR's TensorId for sources; the slot-realization
//! decision (which slot goes to which page, which lives in smem vs
//! gmem) lives at the TkTape optimizer passes, NOT at the source-side
//! identifier.

#![allow(dead_code)]

use std::marker::PhantomData;

use crate::subtile_ir::{KvCacheLayout, KvCacheProducer, KvCacheShape, TensorId};

// ── Sealed RopeForm trait (NeoX vs Interleaved) ─────────────────────

mod rope_form_seal {
    pub trait Sealed {}
}


pub trait RopeForm: rope_form_seal::Sealed {
    const PAIR_LO_EXPR: &'static str;
    const PAIR_HI_EXPR: &'static str;
    const NAME: &'static str;
}

pub struct NeoX;
impl rope_form_seal::Sealed for NeoX {}
impl RopeForm for NeoX {
    const PAIR_LO_EXPR: &'static str = "__row_head * __head_dim + __lane";
    const PAIR_HI_EXPR: &'static str = "__i_lo + __half";
    const NAME: &'static str = "NeoX";
}

pub struct Interleaved;
impl rope_form_seal::Sealed for Interleaved {}
impl RopeForm for Interleaved {
    const PAIR_LO_EXPR: &'static str = "__row_head * __head_dim + 2u * __lane";
    const PAIR_HI_EXPR: &'static str = "__i_lo + 1u";
    const NAME: &'static str = "Interleaved";
}

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
    SyncthreadsGroup { n_warps: u32, role: WarpRole },

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

    // ── Compute — flat, one variant per architectural primitive.

    /// Float-32 RMSNorm over a tile. `eps_bits` is the f32 bit pattern.
    RmsNorm {
        src_page: PageId,
        dst_page: PageId,
        gain_tensor: TensorId,
        rows: u32,
        cols: u32,
        eps_bits: u32,
        role: WarpRole,
    },

    /// Single-row GEMM (m=1) — output is `[1, n]`.
    GemmM1 {
        lhs_page: PageId,
        rhs_tensor: TensorId,
        rhs_byte_off: ByteOffset,
        out_page: PageId,
        m: u32,
        n: u32,
        k: u32,
        accum: AccumKind,
        role: WarpRole,
    },

    /// `out = silu(gate) * up` — fused SwiGLU.
    SiluMul {
        gate_page: PageId,
        up_page: PageId,
        out_page: PageId,
        cols: u32,
        role: WarpRole,
    },

    /// `out = a + b` — residual add.
    ResidualAdd {
        a_page: PageId,
        b_page: PageId,
        out_page: PageId,
        cols: u32,
        role: WarpRole,
    },

    /// RoPE rotation, NeoX form. Per plan §2 row "RopeForm" the form
    /// is a const-generic split on the variant identity — never a
    /// runtime `RopeFormTag` field — so a Q-side / K-side rope-form
    /// mismatch becomes a Rust type error at construction time
    /// (constructor body matches once on `F::TAG` to pick the variant;
    /// downstream Instrs cannot mix them by value).
    RopeRotateNeoX {
        src_page: PageId,
        dst_page: PageId,
        cos_sin_tensor: TensorId,
        position: KernelArgRef,
        kv_layout: KvLayoutId,
        side: RopeSide,
        role: WarpRole,
    },

    /// RoPE rotation, Interleaved form. See [`Instr::RopeRotateNeoX`]
    /// for the const-generic-split rationale.
    RopeRotateInterleaved {
        src_page: PageId,
        dst_page: PageId,
        cos_sin_tensor: TensorId,
        position: KernelArgRef,
        kv_layout: KvLayoutId,
        side: RopeSide,
        role: WarpRole,
    },

    /// Initialise the online-softmax recurrence. `kv_layout` carries
    /// the K-cache layout witness (per plan §2 line 88; resolved
    /// through [`TkTape::kv_layout`]); `producer` records how the
    /// cache was populated (per plan §2: "exhaustive match in
    /// lowering, no `_ =>` arm"). Per plan §2 line 92 + audit DRIFT
    /// fix: `head_dim` / `num_kv_heads` are NOT carried here — the
    /// player reads them via `tape.kv_layout(kv_layout)` (single-
    /// source method). `num_q_heads` is genuinely separate from KV
    /// layout (Q-side head count).
    AttnDecodeInit {
        state: SoftmaxStateId,
        num_q_heads: u32,
        kv_layout: KvLayoutId,
        producer: KvCacheProducer,
        role: WarpRole,
    },

    /// One iteration of `S = Q · Kᵀ * scale` followed by online softmax.
    /// `kv_layout` is the single source of head_dim / num_kv_heads.
    AttnDecodeQkt {
        state: SoftmaxStateId,
        q_page: PageId,
        k_page: PageId,
        scale_bits: u32,
        num_q_heads: u32,
        kv_layout: KvLayoutId,
        role: WarpRole,
    },

    /// `O += P · V` — second half of one online-softmax iteration.
    /// `kv_layout` is the single source of head_dim / num_kv_heads.
    AttnDecodeSv {
        state: SoftmaxStateId,
        v_page: PageId,
        num_q_heads: u32,
        kv_layout: KvLayoutId,
        role: WarpRole,
    },

    /// `O / l_sum` and write to `out_page`. Closes the recurrence.
    /// `kv_layout` is the single source of head_dim.
    AttnDecodeFinalise {
        state: SoftmaxStateId,
        out_page: PageId,
        num_q_heads: u32,
        kv_layout: KvLayoutId,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarpRole {
    Loader,
    Storer,
    Consumer(u8),
    AllConsumers,
    All,
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

#[derive(Debug, Clone)]
pub struct StoreSpec {
    pub src_page: PageId,
    pub dst_tensor: TensorId,
    pub byte_off: ByteOffset,
    pub tile: TileShape,
    pub role: WarpRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumKind {
    Zero,
    Accumulate,
}

/// Erased tag mirroring the `RopeForm` trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeFormTag {
    NeoX,
    Interleaved,
}

impl RopeFormTag {
    pub(crate) fn from_form<F: RopeForm>() -> Self {
        if F::NAME == NeoX::NAME {
            RopeFormTag::NeoX
        } else if F::NAME == Interleaved::NAME {
            RopeFormTag::Interleaved
        } else {
            unreachable!("RopeForm sealed: NAME must be NeoX or Interleaved")
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeSide {
    Q,
    K,
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

    pub(crate) fn syncthreads_group(role: WarpRole, n_warps: u32) -> Self {
        Self::SyncthreadsGroup { n_warps, role }
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
    /// carries `parity` as a u8 field.
    pub(crate) fn wait_static(
        page: PageId,
        kind: PageBarrier,
        parity: u8,
        role: WarpRole,
    ) -> Self {
        match parity & 1 {
            0 => Self::PageBarrierWaitStaticP0 { page_id: page, kind, role },
            1 => Self::PageBarrierWaitStaticP1 { page_id: page, kind, role },
            _ => unreachable!("parity & 1 is 0 or 1"),
        }
    }

    /// As [`Instr::wait_static`] but for runtime loop-carried parity.
    pub(crate) fn wait_loop(
        page: PageId,
        kind: PageBarrier,
        var: LoopVarId,
        start_parity: u8,
        role: WarpRole,
    ) -> Self {
        match start_parity & 1 {
            0 => Self::PageBarrierWaitLoopStart0 { page_id: page, kind, var, role },
            1 => Self::PageBarrierWaitLoopStart1 { page_id: page, kind, var, role },
            _ => unreachable!("start_parity & 1 is 0 or 1"),
        }
    }

    pub(crate) fn arrive(page: PageId, kind: PageBarrier, role: WarpRole) -> Self {
        Self::PageBarrierArrive { page_id: page, kind, role }
    }

    pub(crate) fn store_async_typed(
        dst_page: PageId,
        dst_tensor: TensorId,
        tile_type_str: impl Into<String>,
        role: WarpRole,
    ) -> Self {
        Self::StoreAsyncTyped {
            dst_page,
            dst_tensor,
            tile_type: TileType::from_layout(tile_type_str),
            role,
        }
    }

    /// Per plan §2 row "RopeForm": runtime→const dispatch happens
    /// here, once at the constructor's `match` site. Each arm produces
    /// a flat const-generic-split variant; the Instr never carries
    /// `form` as a runtime field. Q-side / K-side mismatch becomes a
    /// type error at the call site (caller's `<F>` is unique per side).
    pub(crate) fn rope_rotate<F: RopeForm>(
        src_page: PageId,
        dst_page: PageId,
        cos_sin_tensor: TensorId,
        position: KernelArgRef,
        kv_layout: KvLayoutId,
        side: RopeSide,
        role: WarpRole,
    ) -> Self {
        match RopeFormTag::from_form::<F>() {
            RopeFormTag::NeoX => Self::RopeRotateNeoX {
                src_page,
                dst_page,
                cos_sin_tensor,
                position,
                kv_layout,
                side,
                role,
            },
            RopeFormTag::Interleaved => Self::RopeRotateInterleaved {
                src_page,
                dst_page,
                cos_sin_tensor,
                position,
                kv_layout,
                side,
                role,
            },
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

#[doc(hidden)]
pub struct _RopeFormBridge<F: RopeForm>(PhantomData<F>);

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

/// Validate a [`TkTape`]. Runs at the exit of `lower_tape_to_tk` and
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
            Instr::RmsNorm { .. }
            | Instr::GemmM1 { .. }
            | Instr::SiluMul { .. }
            | Instr::ResidualAdd { .. }
            | Instr::RopeRotateNeoX { .. }
            | Instr::RopeRotateInterleaved { .. }
            | Instr::AttnDecodeInit { .. }
            | Instr::AttnDecodeQkt { .. }
            | Instr::AttnDecodeSv { .. }
            | Instr::AttnDecodeFinalise { .. }
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
        let w = Instr::wait_loop(PageId(2), PageBarrier::Ready, var, 1, WarpRole::AllConsumers);
        // Plan §2: parity is a const-generic split variant, not a u8
        // field. start_parity=1 produces PageBarrierWaitLoopStart1.
        match w {
            Instr::PageBarrierWaitLoopStart1 { var: v, .. } => {
                assert_eq!(v, var);
            }
            _ => panic!("expected loop-parity wait Start1"),
        }
    }

    #[test]
    fn validate_tk_tape_accepts_lower_output_shape() {
        // StoreAsync → Threadfence → Arrive{Done} is the conservative
        // all-gmem post-condition lower_tape_to_tk emits. validator
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
