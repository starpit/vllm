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

use crate::subtile_ir::{KvCacheLayout, KvCacheProducer, TensorId};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelArgRef(pub u16);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreludeName(pub u32);

/// Identifier for the [`Instr`] compute that owns a prelude decl.
/// Connecting decl ↔ body at type level prevents orphan decls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeOwner(pub u32);

// ── instruction stream ──────────────────────────────────────────────

/// One instruction the kernel executes. Each variant maps 1:1 to a
/// TK 2.0 / CUDA primitive call. Compute variants alone expand to
/// a fixed `&'static str` template with field-substitution by the
/// dumb player.
#[derive(Debug, Clone)]
pub enum Instr {
    // ── Sync primitives — every variant carries a WarpRole so the
    //    player has zero ambient lookups.

    /// `__syncthreads()` (or scoped variant). The CTA-scope and the
    /// `kittens::group<N>::sync()` group-scope variants are distinct.
    Syncthreads { scope: SyncScope, role: WarpRole },

    /// `__threadfence()` / `__threadfence_block()` /
    /// `__threadfence_system()`.
    Threadfence { scope: FenceScope, role: WarpRole },

    /// `kittens::group<1>::tma::store_commit_group()`.
    CommitGroup { kind: CommitKind, role: WarpRole },

    /// `kittens::group<1>::tma::store_async_wait<N>()`. `n=0` drains
    /// all groups.
    WaitGroup { kind: CommitKind, n: u32, role: WarpRole },

    // ── Page barriers — TK 2.0 mbarrier handshake ────────────────

    /// `mbarrier.init` for a named page barrier. `count` is the
    /// expected arrival count.
    BarrierInit {
        page_id: PageId,
        kind: PageBarrier,
        count: u32,
    },

    /// Wait on `page_<kind>[page_id]` at the captured `parity`.
    PageBarrierWait {
        page_id: PageId,
        kind: PageBarrier,
        parity: ParityExpr,
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
        rhs_byte_off: ByteOffsetExpr,
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

    /// RoPE rotation. The `RopeFormTag` variant is constructed via
    /// `RopeFormTag::from_form::<F>()` so a Q-side / K-side mismatch
    /// is a compile error upstream.
    RopeRotate {
        src_page: PageId,
        dst_page: PageId,
        cos_sin_tensor: TensorId,
        position: KernelArgRef,
        kv_layout: KvLayoutId,
        head_dim: u32,
        num_heads: u32,
        form: RopeFormTag,
        side: RopeSide,
        role: WarpRole,
    },

    /// Initialise the online-softmax recurrence.
    AttnDecodeInit {
        state: SoftmaxStateId,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        role: WarpRole,
    },

    /// One iteration of `S = Q · Kᵀ * scale` followed by online softmax.
    AttnDecodeQkt {
        state: SoftmaxStateId,
        q_page: PageId,
        k_page: PageId,
        scale_bits: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        role: WarpRole,
    },

    /// `O += P · V` — second half of one online-softmax iteration.
    AttnDecodeSv {
        state: SoftmaxStateId,
        v_page: PageId,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        role: WarpRole,
    },

    /// `O / l_sum` and write to `out_page`. Closes the recurrence.
    AttnDecodeFinalise {
        state: SoftmaxStateId,
        out_page: PageId,
        num_q_heads: u32,
        head_dim: u32,
        role: WarpRole,
    },

    /// Inert marker the orchestrator emits at the start of an op
    /// when `EmitOpts::debug_handshake` is on.
    DebugOpBeginMarker { op_index: u32 },

    // ── Control flow — only here, never implicit.

    /// `for (uint var = 0; var < count; ++var) { body... }`.
    ForLoop {
        var: LoopVarId,
        count: LoopCount,
        body: Vec<Instr>,
    },
}

// ── instruction field types ─────────────────────────────────────────

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

/// `commit_group` / `wait_group` come in TK 2.0's bulk-store flavour
/// (TMA store async) and the legacy non-bulk flavour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitKind {
    BulkStore,
    NonBulk,
}

/// Which warp role inside the persistent CTA owns an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarpRole {
    Loader,
    Storer,
    Consumer(u8),
    AllConsumers,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub u8);

/// Which TK 2.0 mbarrier of a page slot a wait/arrive talks to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageBarrier {
    Ready,
    Done,
    Consumed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoopVarId(pub u32);

/// Sealed identifier for a piece of online-softmax recurrence state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SoftmaxStateId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopCount {
    Const(u32),
    KernelArg(KernelArgRef),
}

/// Static (compile-time) or runtime parity for a barrier wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityExpr {
    Static(u8),
    LoopParity { var: LoopVarId, start: u8 },
}

/// Byte-offset expression for TMA load/store source/dest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOffsetExpr {
    Const(u64),
    LinearLoop {
        var: LoopVarId,
        stride: u64,
        base: u64,
    },
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
    pub byte_off: ByteOffsetExpr,
    pub tile: TileShape,
    pub role: WarpRole,
    /// Which page barrier `expect_bytes` arms.
    pub barrier_page: PageId,
}

#[derive(Debug, Clone)]
pub struct StoreSpec {
    pub src_page: PageId,
    pub dst_tensor: TensorId,
    pub byte_off: ByteOffsetExpr,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvLayoutId(pub u32);

// ── Sealed constructors ─────────────────────────────────────────────

impl Instr {
    pub(crate) fn syncthreads_cta(role: WarpRole) -> Self {
        Self::Syncthreads { scope: SyncScope::Cta, role }
    }

    pub(crate) fn syncthreads_group(role: WarpRole, n: u32) -> Self {
        Self::Syncthreads { scope: SyncScope::GroupOf(n), role }
    }

    pub(crate) fn threadfence_device(role: WarpRole) -> Self {
        Self::Threadfence { scope: FenceScope::Device, role }
    }

    pub(crate) fn commit_bulk(role: WarpRole) -> Self {
        Self::CommitGroup { kind: CommitKind::BulkStore, role }
    }

    pub(crate) fn wait_bulk(role: WarpRole, n: u32) -> Self {
        Self::WaitGroup { kind: CommitKind::BulkStore, n, role }
    }

    pub(crate) fn wait_static(
        page: PageId,
        kind: PageBarrier,
        parity: u8,
        role: WarpRole,
    ) -> Self {
        Self::PageBarrierWait {
            page_id: page,
            kind,
            parity: ParityExpr::Static(parity),
            role,
        }
    }

    pub(crate) fn wait_loop(
        page: PageId,
        kind: PageBarrier,
        var: LoopVarId,
        start_parity: u8,
        role: WarpRole,
    ) -> Self {
        Self::PageBarrierWait {
            page_id: page,
            kind,
            parity: ParityExpr::LoopParity {
                var,
                start: start_parity & 1,
            },
            role,
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

    pub(crate) fn rope_rotate<F: RopeForm>(
        src_page: PageId,
        dst_page: PageId,
        cos_sin_tensor: TensorId,
        position: KernelArgRef,
        kv_layout: KvLayoutId,
        head_dim: u32,
        num_heads: u32,
        side: RopeSide,
        role: WarpRole,
    ) -> Self {
        Self::RopeRotate {
            src_page,
            dst_page,
            cos_sin_tensor,
            position,
            kv_layout,
            head_dim,
            num_heads,
            form: RopeFormTag::from_form::<F>(),
            side,
            role,
        }
    }
}

// ── Witness handles surfacing tape-side dataflow ────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayoutEntry {
    pub layout: KvCacheLayout,
}

impl KvLayoutEntry {
    pub fn cache_tensor(&self) -> TensorId {
        self.layout.cache_tensor()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fence_is_five_primitive_instrs() {
        let mut tape = TkTape::new();
        tape.emit_cross_op_gmem_fence();
        assert_eq!(tape.instrs.len(), 5);
        assert!(matches!(tape.instrs[0], Instr::Syncthreads { .. }));
        assert!(matches!(tape.instrs[1], Instr::CommitGroup { .. }));
        assert!(matches!(tape.instrs[2], Instr::WaitGroup { n: 0, .. }));
        assert!(matches!(tape.instrs[3], Instr::Threadfence { .. }));
        assert!(matches!(tape.instrs[4], Instr::Syncthreads { .. }));
    }

    #[test]
    fn parity_loop_carries_start() {
        let var = LoopVarId(0);
        let w = Instr::wait_loop(PageId(2), PageBarrier::Ready, var, 1, WarpRole::AllConsumers);
        match w {
            Instr::PageBarrierWait { parity: ParityExpr::LoopParity { var: v, start }, .. } => {
                assert_eq!(v, var);
                assert_eq!(start, 1);
            }
            _ => panic!("expected loop-parity wait"),
        }
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
