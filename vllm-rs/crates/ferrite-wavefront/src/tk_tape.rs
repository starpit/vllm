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
/// A multi-step CUDA sequence (e.g. a "fence") is a SEQUENCE of
/// primitive Instrs in `instrs`, never one Instr that expands into
/// many lines. Same goes for the kernel-end drain: the walker pushes
/// the drain's primitive Instrs at the tail of `instrs`.
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

    /// `kittens::group<1>::tma::store_commit_group()`.
    CommitGroup,

    /// `kittens::group<1>::tma::store_async_wait<N>()`.
    WaitGroup { n: u32 },

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
}

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
