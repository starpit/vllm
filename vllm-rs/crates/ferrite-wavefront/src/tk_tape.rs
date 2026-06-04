// SPDX-License-Identifier: Apache-2.0
//! `TkTape` — the flat instruction tape that backs the dumb tape player.
//!
//! Per `SUBTILE_IR_REDESIGN.md` the contract is:
//!
//! - Every operation the megakernel performs at runtime is an
//!   [`Instr`] in this tape: TMA loads, TMA stores, barrier
//!   inits/waits/arrives, every fence (`commit_group`, `wait_group`,
//!   `threadfence`, `__syncthreads`), persistent-state declarations,
//!   loops, kernel-arg declarations, kernel entry, kernel exit, every
//!   compute body. If the kernel runs it, an `Instr` encodes it.
//!
//! - Tape generation (the walker in `tk_lower.rs`) reads `LoweredOp`
//!   nodes and pushes `Instr`s. The walker cannot decide "I will
//!   insert a fence here" — a [`Instr::Fence`] is already a node
//!   in the input IR by the time the walker runs.
//!
//! - The tape player ([`crate::tk_player`]) is dumb transcription:
//!   one `match` arm per [`Instr`] kind, each ≤5 lines of
//!   `format!()` whose placeholders are filled from fields on the
//!   instruction. No ambient-state lookups. No formula computation.
//!
//! Phase 0 lands the empty scaffold. Subsequent phases fill in
//! variants as migration proceeds.

#![allow(dead_code)]

use crate::subtile_ir::BufId;

/// The flat instruction tape — produced by the walker, consumed by
/// the dumb player.
///
/// Phase 0: empty scaffold. Phases 1+ fill in fields per the
/// migration plan in `SUBTILE_IR_REDESIGN.md`.
#[derive(Debug, Default)]
pub struct TkTape {
    /// Kernel-arg declarations in ABI order. Order is the C++
    /// kernel signature's parameter order; the dispatcher's runtime
    /// `bufs[]` and `u32_args[]` follow this.
    pub kernel_args: Vec<KernelArg>,

    /// Persistent declarations emitted at function-scope before the
    /// instruction stream. Replaces `TkProgram::prelude: String`.
    pub prelude: Vec<PreludeDecl>,

    /// The instruction stream — flat, role-gated per Instr.
    pub instrs: Vec<Instr>,

    /// Tail drain emitted at kernel exit. Separate from the body so
    /// the kernel-end-drain witness pattern stays type-explicit.
    pub end_drain: FenceSpec,
}

// ── kernel-arg declarations ───────────────────────────────────────

/// One kernel-arg declaration. Drives the C++ kernel signature
/// parameter list and the dispatcher's `bufs[]` / `u32_args[]`
/// indices.
#[derive(Debug, Clone)]
pub struct KernelArg {
    pub name: KernelArgName,
    pub ty: KernelArgTy,
}

#[derive(Debug, Clone)]
pub enum KernelArgName {
    /// Stable identifier baked into the kernel signature.
    Fixed(&'static str),
}

#[derive(Debug, Clone)]
pub enum KernelArgTy {
    /// `uint32_t` runtime arg sourced from a typed per-call value.
    U32 { source: U32Source },
    /// Pointer to a typed buffer (weights / activations / kv cache /
    /// cos-sin cache).
    BufPtr(BufId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum U32Source {
    NumKvPages,
    DecodePosition,
    DecodeSlot,
}

/// Index into [`TkTape::kernel_args`]. The Instr stream references
/// kernel args by id, never by name string — name resolution lives
/// in the player.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelArgRef(pub u16);

// ── prelude declarations ──────────────────────────────────────────

/// One persistent-state declaration emitted before the instruction
/// stream. Replaces the freeform `TkProgram::prelude: String` —
/// every persistent state is typed.
#[derive(Debug, Clone)]
pub enum PreludeDecl {
    /// Per-warp `float[len]` — e.g. softmax `__l_sum`.
    PerWarpFloatArray {
        name: PreludeName,
        len: u32,
        owner: ComputeBodyOwner,
    },
    /// Per-warp `float[rows][cols]` — e.g. softmax `__o_accum`.
    PerWarpFloatMatrix {
        name: PreludeName,
        rows: u32,
        cols: u32,
        owner: ComputeBodyOwner,
    },
    /// `void* page_buf[NUM_PAGES]` aliasing.
    SmemTilePtr {
        name: PreludeName,
        page: PageId,
    },
}

/// Sealed name for a prelude decl. Constructed only via the
/// tape-builder API (Phase 5+).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreludeName(pub u32);

/// Identifier for the [`Instr::Compute`] body that owns this
/// prelude decl. Connecting decl ↔ body at type level prevents
/// orphan decls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeBodyOwner(pub u32);

// ── instruction stream ────────────────────────────────────────────

/// One instruction the kernel executes. Each variant maps 1:1 to a
/// TK 2.0 / CUDA primitive call. Phase 0 lands the variant list as
/// a stub; subsequent phases populate fields per
/// `SUBTILE_IR_REDESIGN.md` §2.2.
#[derive(Debug, Clone)]
pub enum Instr {
    /// `__syncthreads()` (or scoped variant).
    Syncthreads { scope: SyncScope },

    /// `__threadfence()` / `__threadfence_block()` /
    /// `__threadfence_system()`.
    Threadfence { scope: FenceScope },

    /// `cp.async.bulk.commit_group;` (sm90+) or the non-bulk
    /// equivalent.
    CommitGroup { kind: CommitKind },

    /// `cp.async.bulk.wait_group N;`.
    WaitGroup { kind: CommitKind, n: u32 },

    /// Cross-op gmem fence + kernel-end drain — consolidated. See
    /// [`FenceSpec`].
    Fence(FenceSpec),

    /// `mbarrier.init` for a named barrier.
    BarrierInit { id: BarrierId, count: u32 },

    /// `mbarrier.try_wait_parity` / `kittens::wait` on a barrier
    /// with a typed parity.
    BarrierWait {
        id: BarrierId,
        parity: ParityExpr,
        role: WarpRole,
    },

    /// `mbarrier.arrive` on a barrier from a warp role.
    BarrierArrive { id: BarrierId, role: WarpRole },

    /// `kittens::group<1>::tma::expect_bytes` + `tma::load_async`.
    LoadAsync(LoadSpec),

    /// `kittens::group<1>::tma::store_async` (+ optional inline
    /// commit/wait per [`StoreCommitStrategy`]).
    StoreAsync(StoreSpec),

    /// One ComputeBody emit — body_id keys a sealed CUDA template;
    /// fields on the variant fill placeholders.
    Compute {
        body_id: ComputeBodyId,
        role: WarpRole,
    },

    /// `for (uint var = 0; var < count; ++var) { body... }`. Body
    /// is inlined into the tape (no shared-body indirection per
    /// SUBTILE_IR_REDESIGN.md Q3).
    ForLoop {
        var: LoopVarId,
        count: KernelArgRef,
        body: Vec<Instr>,
    },
}

// ── instruction field types ───────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncScope {
    /// `__syncthreads()` — full CTA.
    Cta,
    /// `kittens::group<N>::sync()` — N-warp group sync.
    GroupOf(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FenceScope {
    /// `__threadfence_block()` — CTA-scope.
    Block,
    /// `__threadfence()` — device-scope.
    #[default]
    Device,
    /// `__threadfence_system()` — system-scope.
    System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitKind {
    /// `cp.async.bulk.{commit,wait}_group` — sm90+ TMA bulk path.
    BulkStore,
    /// `cp.async.{commit,wait}_group` — sm80 plain async copy. Not
    /// used on Hopper for K/V cache writes; provided for future
    /// non-TMA paths.
    NonBulk,
}

/// Cross-op fence specification — fully fielded so the player can
/// emit the fence with no ambient lookups. Replaces today's
/// `cross_op_gmem_fence_body` (hardcoded 5-line string) and
/// `KernelEndDrain` (separate hardcoded shape).
#[derive(Debug, Clone, Default)]
pub struct FenceSpec {
    pub scope: FenceScope,
    pub wait: WaitMode,
    /// Which warp roles' stores this fence drains.
    pub producer_role: WarpRoleSet,
    /// Which warp roles' loads must observe drained writes after
    /// this fence.
    pub consumer_role: WarpRoleSet,
    /// `__syncthreads()` before the commit/wait pair.
    pub bracket_pre_sync: bool,
    /// `__syncthreads()` after the threadfence.
    pub bracket_post_sync: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaitMode {
    /// `wait_group 0;` — drain all groups.
    #[default]
    DrainAll,
    /// `wait_group N;` — drain all but the most-recent N.
    DrainN(u32),
}

/// Bitmask over [`WarpRole`] discriminants. Allows a fence to be
/// declared "all roles" or "loader+storer only" without a Vec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WarpRoleSet(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpRole {
    Loader,
    Storer,
    Consumer(u8),
    AllConsumers,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BarrierId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoopVarId(pub u32);

/// Static (compile-time) or runtime parity for a barrier wait.
/// Static is a u8 baked at tape-build time; LoopParity is the
/// `(loop_var & 1)` expression for in-loop waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityExpr {
    Static(u8),
    LoopParity(LoopVarId),
}

/// Byte-offset expression for TMA load/store source/dest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOffsetExpr {
    /// Constant byte offset baked at tape-build time.
    Const(u64),
    /// `base + var * stride` — for in-loop TMA loads of paged K/V
    /// cache.
    LinearLoop {
        var: LoopVarId,
        stride: u64,
        base: u64,
    },
}

#[derive(Debug, Clone)]
pub struct LoadSpec {
    pub dst_page: PageId,
    pub src_buf: BufId,
    pub src_byte_off: ByteOffsetExpr,
    pub bytes: u32,
    pub role: WarpRole,
    pub barrier: BarrierId,
}

#[derive(Debug, Clone)]
pub struct StoreSpec {
    pub src_page: PageId,
    pub dst_buf: BufId,
    pub dst_byte_off: ByteOffsetExpr,
    pub bytes: u32,
    pub role: WarpRole,
    pub commit_strategy: StoreCommitStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreCommitStrategy {
    /// `store_async + store_commit_group + store_async_wait<0>`
    /// inline. Today's raw-bulk `tma_store_async` shape.
    InlineCommitWait,
    /// Just `store_async`; commit/wait emitted by a later
    /// [`Instr::Fence`] with matching producer_role.
    DeferredToFence(FenceId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FenceId(pub u32);

/// Sealed identifier for one ComputeBody template. Each variant
/// maps to a fixed `&'static str` CUDA template in
/// [`crate::tk_player`]; field substitution is mechanical.
///
/// Phase 0: enum stub. Phases 5+ populate variants as compute
/// bodies migrate from `tk_codegen.rs`'s body-string functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBodyId {
    /// Placeholder — concrete body variants land in later phases.
    Placeholder,
}
