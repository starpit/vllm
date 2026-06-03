// SPDX-License-Identifier: Apache-2.0
//! Cross-op gmem-sync substrate (Step E.13).
//!
//! Within the persistent megakernel CTA, ops execute sequentially in
//! the IR but warps run in parallel. Op A's storer warp can issue a
//! `tma::store_async + store_async_wait` to gmem, then op B's loader
//! warp can issue `tma::load_async` on the same gmem address — the
//! per-page mbarrier handshakes do NOT enforce cross-op gmem
//! ordering, so the loader may not observe the storer's writes.
//! E.12 hit this: RopeAppend writes K to the paged K_cache,
//! AttnDecode reads from the same K_cache, and the reads come up
//! stale because no fence sits between them in the IR.
//!
//! This module provides the typed witness that makes the bug class a
//! Rust compile error:
//!
//! - [`GmemHandle<Buf>`] — a handle on a gmem buffer. Produced by
//!   any TMA-store lowering (e.g. [`crate::tk_lower::lower_rope_append`]).
//! - [`Fenced<H>`] — a sealed wrapper proving the gmem region's
//!   pending writes have been committed and made CTA-visible by an
//!   explicit fence emit.
//! - [`emit_fence_after_op`] — the only path to construct
//!   `Fenced<...>`. Appends a [`crate::tk_warp_ir::TkInstr::CrossOpGmemFence`]
//!   to the program, which the codegen lowers to the actual CUDA
//!   primitive.
//!
//! Lowerings that read a previously-written gmem buffer (e.g.
//! [`crate::tk_lower::lower_attn_decode`]) take `Fenced<GmemHandle<...>>`
//! by signature. The orchestrator can't wire a producer's
//! [`GmemHandle<...>`] directly to such a consumer — the call fails
//! to typecheck. The only way to satisfy the consumer is to call
//! [`emit_fence_after_op`], which guarantees the fence instruction
//! lands in the IR.
//!
//! `Buf` is a phantom type-level marker for the buffer kind ([`KCache`],
//! [`VCache`]) — it stops a [`Fenced<GmemHandle<KCache>>`] from being
//! handed to a consumer that expected [`Fenced<GmemHandle<VCache>>`].

use std::marker::PhantomData;

use crate::subtile_ir::BufId;

// ── Buffer-kind phantom markers ─────────────────────────────────────

/// Marker for the paged K cache pool (per-layer `PrefixK`).
#[derive(Clone, Copy, Debug)]
pub struct KCache;

/// Marker for the paged V cache pool (per-layer `PrefixV`).
#[derive(Clone, Copy, Debug)]
pub struct VCache;

/// Marker for an op's per-arena gmem staging slot
/// (`buf{n_sources + op_idx}`) — the orchestrator's `op_out_buf`
/// entries point at these. Phantom only; carried by the typed
/// [`GmemHandle<ArenaSlot>`] / [`Carried<ArenaSlot>`] so the
/// type system can distinguish "an arena edge between two ops"
/// from a paged-cache edge ([`KCache`] / [`VCache`]) or an
/// external source ([`Ext`]).
#[derive(Clone, Copy, Debug)]
pub struct ArenaSlot;

/// Marker for an external source buffer (the orchestrator's
/// `InputRef::Ext(e)` entries; weights, embeddings, the rotary
/// tables, the pre-existing decode-position scalars). Always
/// gmem-loaded — there is no carry-forward path for an external
/// source today, so a [`Carried<Ext>`] is constructively
/// uninhabited from the orchestrator's call sites (the
/// `from_handle` is `pub(crate)` and the orchestrator never
/// constructs one for an `Ext` input).
#[derive(Clone, Copy, Debug)]
pub struct Ext;

// ── GmemHandle ──────────────────────────────────────────────────────

/// Typed handle on a gmem buffer. The `Buf` phantom type tags the
/// buffer kind so a `GmemHandle<KCache>` can't be confused with a
/// `GmemHandle<VCache>`. Carries the [`BufId`] at the value level
/// for the emitter to reference.
///
/// **Constructibility**: [`GmemHandle::new_initial`] makes a fresh
/// handle for a pre-existing gmem buffer (e.g. the K_cache pool's
/// initial state at kernel entry, populated by per-op forward
/// prefill). Lowerings that write to a gmem buffer take a handle in
/// and return a fresh one out — the value is unfenced (pending writes
/// have not yet been ordered with respect to subsequent reads).
#[derive(Clone, Copy, Debug)]
pub struct GmemHandle<Buf> {
    buf_id: BufId,
    _marker: PhantomData<Buf>,
}

impl<Buf> GmemHandle<Buf> {
    /// Construct an initial handle for a gmem buffer. Used by the
    /// orchestrator to seed K_cache / V_cache handles at kernel
    /// entry — those buffers are pre-populated by per-op forward
    /// prefill, so the initial handle is conceptually "fenced
    /// already" by virtue of the kernel-launch boundary, but the
    /// type is plain [`GmemHandle`] for uniformity (the first reader
    /// in the kernel must still go through [`emit_fence_after_op`]).
    pub fn new_initial(buf_id: BufId) -> Self {
        Self {
            buf_id,
            _marker: PhantomData,
        }
    }

    /// The CUDA `bufN` index this handle references.
    pub fn buf_id(&self) -> BufId {
        self.buf_id
    }
}

// ── Fenced<H> + sealed FenceProof ───────────────────────────────────

/// Sealed token witnessing that a [`crate::tk_warp_ir::TkInstr::CrossOpGmemFence`]
/// has been emitted into the program. Only [`emit_fence_after_op`]
/// constructs one — the field is private to this crate, and there
/// is no `pub fn new()` / `pub const`.
#[derive(Clone, Copy, Debug)]
pub struct FenceProof(());

/// Wrapper proving the wrapped handle's pending gmem writes have
/// been committed and made CTA-visible by an explicit fence emit.
/// Constructible only via [`emit_fence_after_op`].
///
/// # Compile-fail proofs
///
/// `FenceProof` field is private — cannot be constructed outside
/// this crate's `emit_fence_after_op` path:
///
/// ```compile_fail
/// use ferrite_wavefront::tk_gmem::{Fenced, FenceProof, GmemHandle, KCache};
/// use ferrite_wavefront::subtile_ir::BufId;
/// let h: GmemHandle<KCache> = GmemHandle::new_initial(BufId(0));
/// // FenceProof's field is private — can't construct one.
/// let bogus = Fenced(h, FenceProof(()));
/// ```
///
/// Free `Fenced` field access from outside is also blocked
/// (constructors / fields are pub(crate)):
///
/// ```compile_fail
/// use ferrite_wavefront::tk_gmem::Fenced;
/// // Fenced's tuple fields are private outside the crate.
/// fn extract<H>(f: Fenced<H>) -> H { f.0 }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Fenced<H>(pub(crate) H, pub(crate) FenceProof);

impl<H> Fenced<H> {
    /// Drop the fenced wrapper to access the underlying handle.
    /// Used by lowerings (e.g. AttnDecode) to read the buf id for
    /// the TMA load emit.
    pub fn into_inner(self) -> H {
        self.0
    }
}

// ── emit_fence_after_op ─────────────────────────────────────────────

/// Emit the cross-op gmem-fence primitive into `prog` and lift the
/// handle to [`Fenced<H>`]. The fence body is a single
/// [`crate::tk_warp_ir::TkInstr::CrossOpGmemFence`] instruction; the
/// codegen ([`crate::tk_codegen`]) lowers it to the actual CUDA
/// primitive (commit/wait + threadfence + syncthreads).
///
/// This is the ONLY public path to a [`Fenced<H>`] value. Any
/// lowering that requires `Fenced<GmemHandle<...>>` for safety
/// (e.g. an AttnDecode reading from the K_cache pool) cannot be
/// fed an unfenced handle without going through this function —
/// the orchestrator's call site fails to typecheck otherwise.
pub fn emit_fence_after_op<H>(
    prog: &mut crate::tk_warp_ir::TkProgram,
    handle: H,
) -> Fenced<H> {
    prog.emit_cross_op_gmem_fence();
    Fenced(handle, FenceProof(()))
}

// ── Carried<H> — typed witness for smem carry-forward edges ────────

/// Sealed token witnessing that the orchestrator's routing analysis
/// classified an edge as `InputRouting::CarryForward` and that the
/// producer's smem page will be handed to this consumer via the
/// cross-IType mbarrier handshake (the producer's storer skips the
/// gmem drain; the consumer's loader skips the gmem TMA load and
/// emits a bare `arrive(Ready)` to flip its own page barrier).
///
/// Only the orchestrator's per-op dispatch can construct one — the
/// `CarriedProof::mint()` path is `pub(crate)`.
#[derive(Clone, Copy, Debug)]
pub struct CarriedProof(());

impl CarriedProof {
    /// Crate-private constructor. The orchestrator calls this at the
    /// point where the routing analysis says an edge is carry-
    /// forward AND the page allocator has the producer's slot in
    /// `in_use=true` with a recorded parity.
    pub(crate) fn mint() -> Self {
        Self(())
    }
}

/// Typed carry-forward handle. Wraps a [`crate::tk_lower::CarriedHandle`]
/// (slot id + parity) plus the [`CarriedProof`] witness. Per-buffer-
/// kind phantom marker keeps a `Carried<KCache>` from being passed
/// where a `Carried<ArenaSlot>` is required, etc.
///
/// # Compile-fail proof
///
/// Cannot construct a `Carried` outside the substrate (the inner
/// fields are private):
///
/// ```compile_fail
/// use ferrite_wavefront::tk_gmem::{Carried, CarriedProof, ArenaSlot};
/// use ferrite_wavefront::tk_lower::CarriedHandle;
/// // CarriedProof's tuple field is private outside the crate.
/// let bogus = Carried::<ArenaSlot>(
///     CarriedHandle { id: 0, phase: 0 },
///     CarriedProof(()),
///     std::marker::PhantomData,
/// );
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Carried<Buf>(
    pub(crate) crate::tk_lower::CarriedHandle,
    pub(crate) CarriedProof,
    pub(crate) std::marker::PhantomData<Buf>,
);

impl<Buf> Carried<Buf> {
    /// Crate-private constructor. The orchestrator wraps a
    /// [`crate::tk_lower::CarriedHandle`] from the routing analysis
    /// + the page allocator's `in_use` table.
    pub(crate) fn from_handle(
        h: crate::tk_lower::CarriedHandle,
        proof: CarriedProof,
    ) -> Self {
        Self(h, proof, std::marker::PhantomData)
    }

    /// Drop the wrapper to access the underlying [`CarriedHandle`]
    /// for `pages.consume_carried::<P>(...)` inside a lowering body.
    pub fn into_handle(self) -> crate::tk_lower::CarriedHandle {
        self.0
    }
}

// ── CrossOpInput<Buf> — the per-input routing decision ─────────────

/// One input slot for any per-op lowering. The orchestrator's
/// routing analysis picks the variant per edge:
///
/// - `Carried(...)` — smem carry-forward. The producer's storer
///   skipped the gmem drain; the consumer's loader emits a bare
///   `arrive(Ready)` instead of `tma::load_async`.
/// - `Fenced(...)` — gmem TMA load post a [`emit_fence_after_op`]
///   emit. The consumer issues the standard `tma::load_async`
///   knowing the producer's gmem store has been committed and
///   made CTA-visible.
///
/// Lowering bodies match exhaustively on the two arms — adding a
/// new edge kind (e.g. `BlockTableIndirected` for paged-cache
/// reads in multi-block sequences) becomes a Rust compile error
/// at every existing `match input { ... }` until that arm is
/// added.
///
/// # Compile-fail proof
///
/// Cannot construct a `CrossOpInput::Carried` from a fabricated
/// `Carried<Buf>` (the wrapper's fields are private outside the
/// crate; the orchestrator's `from_handle` + `mint` is the only
/// route):
///
/// ```compile_fail
/// use ferrite_wavefront::tk_gmem::{Carried, CarriedProof, ArenaSlot, CrossOpInput};
/// use ferrite_wavefront::tk_lower::CarriedHandle;
/// // Cannot fabricate a CarriedProof.
/// let proof = CarriedProof(());
/// ```
#[derive(Clone, Copy, Debug)]
pub enum CrossOpInput<Buf> {
    /// Smem carry-forward — producer's smem page is the consumer's
    /// input.
    Carried(Carried<Buf>),
    /// Gmem TMA load post-fence.
    Fenced(Fenced<GmemHandle<Buf>>),
}

// ── OpOutput<Buf> — the per-op output routing classification ──────

/// One per-output slot result returned by a lowering. Every per-op
/// `lower_X` returns an `OpOutput` per output buffer; the
/// orchestrator stashes it in its per-`BufId` handle table for the
/// next op to consume as a [`CrossOpInput`].
///
/// - `Gmem(...)` — output drained to gmem (`op.out` arena slot).
///   The next consumer will TMA-load it post-fence
///   (orchestrator wraps via `emit_fence_after_op`).
/// - `Carried(...)` — output kept in smem; the page slot's storer
///   skipped the drain. The orchestrator threads the
///   [`crate::tk_lower::CarriedHandle`] to the next op.
#[derive(Clone, Copy, Debug)]
pub enum OpOutput<Buf> {
    Gmem(GmemHandle<Buf>),
    Carried(Carried<Buf>),
}
