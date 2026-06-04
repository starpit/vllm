// SPDX-License-Identifier: Apache-2.0
//! Per-op lowering: high-level decode op → [`TkProgram`].
//!
//! Each public function here is the entire "blank-filling" the old ff-
//! mega-codegen `cuda_emit::render_*` did at *emit* time, hoisted up
//! to *lowering* time. The codegen ([`crate::tk_codegen`]) then walks
//! the resulting [`TkProgram`] without making any decisions.
//!
//! Today this file ports one op (RmsNorm) end-to-end. The remaining
//! decode ops follow the same pattern: each takes a [`SubtileIr`]
//! [`Dispatch`] (or the equivalent pre-resolved op descriptor) plus a
//! [`PageAllocator`] and a [`ScratchAllocator`], and returns a
//! [`TkProgram`] fragment. The persistent megakernel body is the
//! concatenation of those fragments.
//!
//! # PageAllocator
//!
//! The page allocator is the only piece of state in the lowering: it
//! tracks the *latest typed phase* of every page slot. Allocating a
//! page returns a fresh `PageHandle<Phase0>`; releasing one doesn't
//! reset the phase (the next op that grabs the slot picks up where
//! the previous round left off, which is exactly what the TK 2.0
//! ping-pong needs). Phase advance happens through `arrive` returning
//! a flipped `PageHandle<P::Next>` — the allocator only stores the
//! *current* handle as a type-erased `(id, phase_bit)` so the next
//! caller can ask for "the page whose phase is P0" or "P1" by the
//! same const-generic discipline.

#![allow(dead_code)]

use crate::subtile_ir::{BufId, RegionRef};
use crate::tk_warp_ir::{
    LoopBound, PageBarrier, PageHandle, Phase, Phase0, Phase1, TileShape, TkProgram, WarpRole,
    NUM_CONSUMER_WARPS, NUM_PAGES, PAGE_SIZE,
};

// ── Allocators ──────────────────────────────────────────────────────

/// Tracks page-slot ownership across a [`TkProgram`] build. Hands out
/// fresh `PageHandle<Phase0>`s for each new round; the IR threads the
/// phase advance through `arrive` from there.
#[derive(Clone, Debug)]
pub struct PageAllocator {
    /// `phase_bit[id]` is the current parity (`0 ↔ Phase0`,
    /// `1 ↔ Phase1`). Reset to 0 by [`PageAllocator::new`]; advanced
    /// each time the *caller* calls `bump`.
    phase_bit: [u32; NUM_PAGES as usize],
    /// `in_use[id]` — true if the slot is currently owned by an
    /// in-flight op. Released by [`PageAllocator::release`].
    in_use: [bool; NUM_PAGES as usize],
}

impl Default for PageAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl PageAllocator {
    pub fn new() -> Self {
        Self {
            phase_bit: [0; NUM_PAGES as usize],
            in_use: [false; NUM_PAGES as usize],
        }
    }

    /// Allocate the next free page slot, returning the typed handle at
    /// its current phase. Caller MUST also call `release` once the
    /// page's round is complete (the storer's `Consumed` arrive).
    ///
    /// Returns `None` if every page is in use; callers handle the
    /// out-of-pages case explicitly (the megakernel scheduler decides
    /// whether to spill to scratch or stall the consumer).
    /// Allocate a free slot whose current parity matches the requested
    /// `P` (the caller's typed phase). Returns `None` if no free slot is
    /// at that parity — the orchestrator should then probe the OTHER
    /// parity and dispatch to the matching `lower_X::<P::Next>`.
    pub fn alloc_at<P: Phase>(&mut self) -> Option<PageHandle<P>> {
        let id = (0..NUM_PAGES as usize)
            .find(|&i| !self.in_use[i] && self.phase_bit[i] == P::VALUE)?
            as u8;
        self.in_use[id as usize] = true;
        Some(P::fresh_handle(id))
    }

    /// How many slots are currently free at the given runtime parity.
    /// Used by the orchestrator to pick `Phase0` vs `Phase1` per-op so
    /// page reuse keeps working across many ops in one forward.
    pub fn count_at(&self, phase_value: u32) -> usize {
        (0..NUM_PAGES as usize)
            .filter(|&i| !self.in_use[i] && self.phase_bit[i] == phase_value)
            .count()
    }

    /// Convenience wrapper for `alloc_at::<Phase0>()`.
    pub fn alloc_p0(&mut self) -> Option<PageHandle<Phase0>> {
        self.alloc_at::<Phase0>()
    }

    /// Convenience wrapper for `alloc_at::<Phase1>()`.
    pub fn alloc_p1(&mut self) -> Option<PageHandle<Phase1>> {
        self.alloc_at::<Phase1>()
    }

    /// Release a page slot whose round has completed. The handle's
    /// type encodes the slot's *current* phase parity; we store it as
    /// a runtime bit for the next allocator decision.
    pub fn release<P: crate::tk_warp_ir::Phase>(&mut self, page: PageHandle<P>) {
        self.in_use[page.id() as usize] = false;
        self.phase_bit[page.id() as usize] = page.phase();
    }

    /// Phase 12: carry a page slot across an op boundary without
    /// releasing it. The producer lowering uses this in place of
    /// `release` for an internal output whose smem page is reused
    /// directly by the next consumer (no gmem round-trip).
    /// `phase_bit[id]` is updated to the post-round parity (so a
    /// future allocator probe sees the right value if for some
    /// reason the carry is dropped); `in_use[id]` stays `true` so
    /// no other op can claim the slot before the consumer takes it.
    pub fn carry_forward<P: crate::tk_warp_ir::Phase>(
        &mut self,
        page: PageHandle<P>,
    ) -> CarriedHandle {
        // Slot stays in_use=true; only the parity is recorded.
        self.phase_bit[page.id() as usize] = page.phase();
        CarriedHandle {
            id: page.id(),
            phase: page.phase(),
        }
    }

    /// Phase 12: consume a carried-forward handle. The consuming op's
    /// lowering uses this in place of `alloc_at::<P>()` for the input
    /// slot whose data was produced by the upstream op's smem page.
    /// Returns a typed `PageHandle<P>` at the carried-forward parity;
    /// callers MUST then `release` (or carry-forward again) the page
    /// at end-of-round, same as for an `alloc_at` allocation.
    ///
    /// The caller specifies the parity-typed handle they want; the
    /// underlying `phase_bit` must match `P::VALUE`. Mismatch is a
    /// caller error (the orchestrator chooses the consumer's lowering
    /// generic to align with the carried parity).
    pub fn consume_carried<P: crate::tk_warp_ir::Phase>(
        &mut self,
        carried: CarriedHandle,
    ) -> PageHandle<P> {
        debug_assert_eq!(
            carried.phase, P::VALUE,
            "consume_carried: carried phase {} != P::VALUE {}",
            carried.phase, P::VALUE,
        );
        debug_assert!(
            self.in_use[carried.id as usize],
            "consume_carried: slot {} is not in_use (carry-forward dropped?)",
            carried.id,
        );
        P::fresh_handle(carried.id)
    }
}

/// Phase 12: handle for a page slot carried across an op boundary
/// without a `release`/`alloc_at` round-trip. Produced by
/// `PageAllocator::carry_forward(page)` and consumed by
/// `PageAllocator::consume_carried::<P>(carried)`. Encodes the slot
/// id + the post-producer-round parity bit so the consumer can pick
/// the matching `lower_*<P>` generic.
#[derive(Clone, Copy, Debug)]
pub struct CarriedHandle {
    pub id: u8,
    pub phase: u32,
}

/// Phase 12: routing hints threaded by the orchestrator into a per-op
/// lowering. `inputs[i] = Some(carried)` means the op's i-th input
/// slot is provided as a carry-forward handle (loader skips TMA load,
/// emits a bare `arrive(Ready)` instead — the producer's storer's
/// `arrive Consumed` already advanced the slot's parity, and the page
/// already contains the producer's data). `inputs[i] = None` means
/// the op allocates a fresh page and TMA-loads from gmem (existing
/// path).
///
/// `output_internal = true` means the op's output is consumed only
/// by another op in this same kernel; the storer skips TMA store
/// (the consuming op reads directly from this op's smem page) and
/// the lowering returns a `CarriedHandle` for the orchestrator to
/// thread to the consuming op's `inputs` list. `false` means the
/// existing storer-drain-to-gmem path.
#[derive(Clone, Debug, Default)]
pub struct RoutingHints {
    pub inputs: Vec<Option<CarriedHandle>>,
    pub output_internal: bool,
}

/// Phase 12: result returned by a routing-aware lowering. When
/// `output_internal` was `true` in the hints, the lowering populates
/// `output_carried` with the slot it kept reserved for the consumer.
#[derive(Clone, Copy, Debug, Default)]
pub struct RoutingResult {
    pub output_carried: Option<CarriedHandle>,
}

// ── RmsNorm — the vertical slice ───────────────────────────────────

/// Inputs to lower one decode RmsNorm into a `TkProgram` fragment.
/// Mirrors `LoweredOp::RmsNorm` + the `[x, weight]` operand pair from
/// `subtile_ir`; the caller (proc macro / forward) supplies the resolved
/// `BufId`s + the activation column extent.
#[derive(Clone, Copy, Debug)]
pub struct RmsNormOp {
    /// `[m, hidden]` activation (read AND written when the layer wants
    /// the residual side-effect; pure read for layer-0 init).
    pub x: BufId,
    /// `[1, hidden]` rms gain weight. External — never validated by
    /// the dataflow checker.
    pub weight: BufId,
    /// Output `[m, hidden]`. May alias `x` for in-place rms (see
    /// `init` flag).
    pub out: BufId,
    /// Hidden size (decode: a known compile-time constant per arch).
    pub hidden: u32,
    /// Number of decode rows (m). Decode = 1; spec-decode m>1 follows
    /// the same shape but tile differently — out-of-scope for this
    /// vertical slice.
    pub m: u32,
    /// Activation element bytes (`bf16` = 2, `f16` = 2, `f32` = 4).
    pub act_elem: u32,
    /// rms epsilon, baked into the consumer's compute body as a
    /// literal `f`-suffixed float.
    pub eps: f32,
    /// True when this is layer-0 (no residual side-effect, no delta
    /// channel; `x_norm = rms_scale(x) * weight`).
    pub init: bool,
}

/// Lower one RmsNorm op into a `TkProgram` fragment.
///
/// Tape shape for one round (TK 2.0 mbarrier semantics — every wait in
/// the round reads the *same* parity `R & 1`; only `complete_round`
/// flips the typed parity for the next round):
///   1. Loader: wait `Consumed[id]@P` → TMA load `x[..]` into page →
///      arrive `Ready[id]`.
///   2. AllConsumers: wait `Ready[id]@P` → inline RMS reduce + scale
///      `Compute` → arrive `Done[id]`.
///   3. Storer: wait `Done[id]@P` → TMA store page → arrive
///      `Consumed[id]`.
///   4. Round boundary: `complete_round` flips `P → P::Next`.
///
/// Phase parity is *threaded through the type system*: the only way
/// each `wait` sees the right parity is that the IR consumed a typed
/// `PageHandle<P>` whose phase came from the round boundary. Codegen
/// emits the captured `P::VALUE`; phase drift is unrepresentable.
pub fn lower_rmsnorm<P: Phase>(op: RmsNormOp, pages: &mut PageAllocator, prog: &mut TkProgram) {
    let x_page = pages
        .alloc_at::<P>()
        .expect("page exhaustion: out of mbarrier slots (x)");
    let w_page = pages
        .alloc_at::<P>()
        .expect("page exhaustion: out of mbarrier slots (weight)");
    let x_id = x_page.id();
    let w_id = w_page.id();

    // x's region for the TMA: full hidden columns, m rows.
    let x_region = RegionRef::rows_cols(op.x, op.m, 0, op.hidden);
    let out_region = RegionRef::rows_cols(op.out, op.m, 0, op.hidden);
    let weight_region = RegionRef::rows_cols(op.weight, 1, 0, op.hidden);
    let x_tile = TileShape {
        rows: op.m,
        cols: op.hidden,
        elem_bytes: op.act_elem,
    };
    let weight_tile = TileShape {
        rows: 1,
        cols: op.hidden,
        elem_bytes: op.act_elem,
    };

    // ── Loader: fill x and weight pages ──
    // Round 0: wait on `Consumed` at Phase0. (Persistent-CTA scaffold
    // pre-arrives `page_consumed` at init so its first wait reads 1 —
    // and our typed parity for a fresh page slot starts at Phase0,
    // so `wait(consumed, 0)` returns immediately because the barrier's
    // observed phase is 1, != expected 0.)
    //
    // No explicit `arrive(Ready)` — `tma::load_async` ITSELF signals
    // the page_ready barrier when the load completes. Emitting an
    // arrive in addition to the load would over-count arrivals and
    // shift the barrier's phase off the round parity, breaking the
    // consumer's `wait(ready, 0)`.
    let x_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, x_page);
    prog.load_async(x_id, op.x, x_region, x_tile);

    let w_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, w_page);
    prog.load_async(w_id, op.weight, weight_region, weight_tile);

    // ── Consumer: wait both Ready, RMS reduce + scale + multiply weight,
    //    arrive Done on both. ──
    let x_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, x_page);
    let w_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, w_page);
    prog.compute_calls(
        WarpRole::AllConsumers,
        crate::tk_codegen::rmsnorm_compute_calls(x_id, w_id, op.hidden, op.eps),
    );
    let x_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, x_page);
    let w_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, w_page);

    // ── Storer: drain x_page → out, free weight slot. ──
    let x_page = prog.wait(WarpRole::Storer, PageBarrier::Done, x_page);
    prog.store_async(x_id, op.out, out_region, x_tile);
    let x_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, x_page);

    let w_page = prog.wait(WarpRole::Storer, PageBarrier::Done, w_page);
    let w_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, w_page);

    let x_page = prog.complete_round(x_page);
    let w_page = prog.complete_round(w_page);
    pages.release(x_page);
    pages.release(w_page);
}

/// Phase 12: routing-aware variant of `lower_rmsnorm`. Same protocol
/// as `lower_residual_add_routed`. The weight slot is always
/// gmem-loaded (weight is `InputRef::Ext` in every Llama / Mistral /
/// Qwen / Phi / Gemma forward). Only x can carry-forward in.
/// Output is in-place on x's page (the consumer body writes to
/// `__x_smem` and the storer drains x_page → op.out); when
/// `output_internal=true`, the storer skips the drain and the slot
/// is carried forward to the consuming op.
///
/// Default-hints path is byte-identical to `lower_rmsnorm`.
pub fn lower_rmsnorm_routed<P: Phase>(
    op: RmsNormOp,
    hints: &RoutingHints,
    pages: &mut PageAllocator,
    prog: &mut TkProgram,
) -> RoutingResult {
    debug_assert!(
        hints.inputs.is_empty() || hints.inputs.len() == 2,
        "lower_rmsnorm_routed: hints.inputs must be empty or len 2"
    );
    let in_x_carried = hints.inputs.first().and_then(|x| x.as_ref());
    let in_w_carried = hints.inputs.get(1).and_then(|x| x.as_ref());

    let x_page: PageHandle<P> = match in_x_carried {
        Some(c) => pages.consume_carried::<P>(*c),
        None => pages
            .alloc_at::<P>()
            .expect("page exhaustion: out of mbarrier slots (x)"),
    };
    let w_page: PageHandle<P> = match in_w_carried {
        Some(c) => pages.consume_carried::<P>(*c),
        None => pages
            .alloc_at::<P>()
            .expect("page exhaustion: out of mbarrier slots (weight)"),
    };
    let x_id = x_page.id();
    let w_id = w_page.id();

    let x_region = RegionRef::rows_cols(op.x, op.m, 0, op.hidden);
    let out_region = RegionRef::rows_cols(op.out, op.m, 0, op.hidden);
    let weight_region = RegionRef::rows_cols(op.weight, 1, 0, op.hidden);
    let x_tile = TileShape {
        rows: op.m,
        cols: op.hidden,
        elem_bytes: op.act_elem,
    };
    let weight_tile = TileShape {
        rows: 1,
        cols: op.hidden,
        elem_bytes: op.act_elem,
    };

    // ── Loader: fill x (or skip for carry-forward) ──
    let x_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, x_page);
    if in_x_carried.is_some() {
        prog.arrive(WarpRole::Loader, PageBarrier::Ready, x_page);
    } else {
        prog.load_async(x_id, op.x, x_region, x_tile);
    }

    // ── Loader: fill weight (or skip for carry-forward; usually Ext) ──
    let w_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, w_page);
    if in_w_carried.is_some() {
        prog.arrive(WarpRole::Loader, PageBarrier::Ready, w_page);
    } else {
        prog.load_async(w_id, op.weight, weight_region, weight_tile);
    }

    // ── Consumer: RMS reduce + scale + apply weight (compute body unchanged) ──
    let x_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, x_page);
    let w_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, w_page);
    prog.compute_calls(
        WarpRole::AllConsumers,
        crate::tk_codegen::rmsnorm_compute_calls(x_id, w_id, op.hidden, op.eps),
    );
    let x_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, x_page);
    let w_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, w_page);

    // ── Storer: drain x_page → out (or skip for internal output); free weight ──
    let x_page = prog.wait(WarpRole::Storer, PageBarrier::Done, x_page);
    if !hints.output_internal {
        prog.store_async(x_id, op.out, out_region, x_tile);
    }
    let x_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, x_page);

    let w_page = prog.wait(WarpRole::Storer, PageBarrier::Done, w_page);
    let w_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, w_page);

    let x_advanced = prog.complete_round(x_page);
    let w_advanced = prog.complete_round(w_page);

    let result = if hints.output_internal {
        let carried = pages.carry_forward(x_advanced);
        pages.release(w_advanced);
        RoutingResult {
            output_carried: Some(carried),
        }
    } else {
        pages.release(x_advanced);
        pages.release(w_advanced);
        RoutingResult::default()
    };
    result
}

// ── AttnDecode — the m=1 deadlock canonical ────────────────────────

/// Inputs to lower one decode-attention op into a `TkProgram` fragment.
/// Decode = one query token per CTA, sweeping a runtime number of
/// paged K/V tiles.
///
/// This is *the* canonical that today's ff-mega-codegen mis-bumps:
/// `MegaDispatchState::page_rounds` re-derives the loop's parity per
/// op call site, and one mis-counted op silently flips the consumer's
/// `wait` parity → m=1 megakernel hangs in production. Here the
/// outside-loop handshakes are typed-phase (compile-time correct), and
/// the inside-loop parity is bound to the loop variable directly via
/// [`TkProgram::wait_loop_parity`] — so codegen never invents the
/// parity at emit time.
/// **Paged K/V cache layout** (paris invariant
/// `kv-cache-write-slot-offset-correctness`).
///
/// Single source of truth for the byte stride of one token's K
/// (or V) row in the paged cache (`[num_blocks, block_size,
/// num_kv_heads, head_dim]` — one row =
/// `num_kv_heads * head_dim * act_elem` bytes) AND for the cos/sin
/// rotary cache row stride (`head_dim * act_elem`).
///
/// Both the WRITE side (RopeAppend's K/V cache writes inside its
/// storer) and the READ side (AttnDecode's per-iter K/V loads
/// inside its `for_loop_runtime` body) MUST use the same row
/// stride, or the reads observe rotated/shifted slots vs the
/// writes — silently wrong attention, the empirical
/// `Paris!!!!!!!!!` decode-degenerate stream's structural class.
///
/// Today (E.13 baseline) the formula is recomputed at four call
/// sites in `tk_lower.rs` (lines 625 / 832 / 1716 / 1887) — silent
/// drift in any future edit re-introduces the bug. This newtype
/// collapses them to one definition.
///
/// **Phase 3 (paris invariant `kv-layout-witness-binds-cache-bufid`)**:
/// Fields are PRIVATE; the only way to construct a [`KvCacheLayout`]
/// is [`KvCacheLayout::for_buf_id`], which binds the K-cache BufId
/// into the witness. This makes the layout a typed witness that
/// PAIRED RopeAppend (writer) and AttnDecode (reader) ops dereference
/// — both ops' `kv_layout()` constructors call `for_buf_id` with
/// THEIR OWN `k_cache` field; the orchestrator wires those k_cache
/// fields from the same source so the two layouts compare PartialEq-
/// equal by construction. A future divergence (e.g. a contributor
/// hand-rolling a layout with `num_kv_heads = 16` while the producer
/// used 8) is structurally impossible from outside this module:
/// `KvCacheLayout::*` constructors all live here, and the field
/// privacy plus sealed inner module forbid external `KvCacheLayout {
/// num_kv_heads: 16, ... }` literals.
///
/// # Compile-fail proof — no public constructor besides for_buf_id
///
/// ```compile_fail
/// use ferrite_wavefront::tk_lower::KvCacheLayout;
/// use ferrite_wavefront::subtile_ir::BufId;
/// let bogus = KvCacheLayout {
///     cache_buf_id: BufId(0),
///     num_kv_heads: 8,
///     head_dim: 64,
///     act_elem: 2,
/// };  // ERROR: fields private
/// ```
///
/// ```compile_fail
/// use ferrite_wavefront::tk_lower::KvCacheLayout;
/// let bogus = KvCacheLayout::new(8, 64, 2);  // ERROR: `new` removed
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvCacheLayout {
    cache_buf_id: BufId,
    num_kv_heads: u32,
    head_dim: u32,
    act_elem: u32,
}

impl KvCacheLayout {
    /// Sealed constructor binding `cache_buf_id` into the witness.
    /// Two layouts with the same (num_kv_heads, head_dim, act_elem)
    /// but different `cache_buf_id` compare PartialEq-UN-equal,
    /// catching cross-cache witness reuse at the orchestrator level.
    pub const fn for_buf_id(
        cache_buf_id: BufId,
        num_kv_heads: u32,
        head_dim: u32,
        act_elem: u32,
    ) -> Self {
        Self {
            cache_buf_id,
            num_kv_heads,
            head_dim,
            act_elem,
        }
    }

    /// The K-cache BufId this layout binds. Used at lowering time to
    /// `debug_assert!` that the AttnDecode/RopeAppend op consuming
    /// this layout has a matching `k_cache` BufId — catches cross-
    /// cache witness reuse at lower-time before it can drift.
    pub const fn cache_buf_id(&self) -> BufId {
        self.cache_buf_id
    }

    /// Per-token K (or V) head count.
    pub const fn num_kv_heads(&self) -> u32 {
        self.num_kv_heads
    }

    /// Per-head dimensionality.
    pub const fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// Bytes per element (2 for bf16, 4 for fp32).
    pub const fn act_elem(&self) -> u32 {
        self.act_elem
    }

    /// Per-token K (or V) row stride in bytes.
    /// `num_kv_heads * head_dim * act_elem`.
    pub const fn row_bytes(&self) -> u64 {
        (self.num_kv_heads as u64) * (self.head_dim as u64) * (self.act_elem as u64)
    }

    /// Per-position cos/sin row stride in bytes.
    /// `head_dim * act_elem`.
    pub const fn cos_sin_row_bytes(&self) -> u64 {
        (self.head_dim as u64) * (self.act_elem as u64)
    }

    /// CUDA expression for the K/V cache slot byte offset:
    /// `(<slot_arg> * row_bytes)`. Both the RopeAppend write and the
    /// AttnDecode read must reach for this method on the SAME
    /// [`KvCacheLayout`] value, which is constructed from a single
    /// (num_kv_heads, head_dim, act_elem) tuple per op.
    pub fn slot_offset_expr(&self, slot_arg: impl std::fmt::Display) -> String {
        format!("({} * {}u)", slot_arg, self.row_bytes())
    }

    /// CUDA expression for the cos/sin TMA load byte offset:
    /// `<position_arg> * cos_sin_row_bytes`.
    pub fn cos_sin_offset_expr(&self, position_arg: impl std::fmt::Display) -> String {
        format!("{} * {}u", position_arg, self.cos_sin_row_bytes())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AttnDecodeOp {
    /// `[1, num_q_heads * head_dim]` query (post-RoPE). Loaded once
    /// outside the loop.
    pub q: BufId,
    /// Paged KV cache: `[num_blocks, page_size, num_kv_heads, head_dim]`.
    /// The page-block table is a separate runtime arg the kernel
    /// scaffold reads; the codegen here only needs the buffer id.
    pub k_cache: BufId,
    pub v_cache: BufId,
    /// `[1, num_q_heads * head_dim]` output. One head_dim slice per
    /// q-head, concatenated.
    pub out: BufId,
    /// Per-head dimensionality. Llama-3.2-1B = 64.
    pub head_dim: u32,
    /// Number of query heads. Llama-3.2-1B = 32. GQA: q-heads are
    /// grouped onto kv-heads at ratio `num_q_heads / num_kv_heads`.
    /// MUST be divisible by `NUM_CONSUMER_WARPS` (today: 8) — the
    /// lowering distributes q-heads across consumer warps evenly.
    pub num_q_heads: u32,
    /// Number of KV heads. Llama-3.2-1B = 8. MUST equal
    /// `NUM_CONSUMER_WARPS` so each consumer warp owns exactly one
    /// kv-head; relaxing this (to support models with `N_kv != 8`)
    /// is a follow-up that distributes kv-heads round-robin across
    /// warps and adds an inner-loop over the warp's kv-head set.
    pub num_kv_heads: u32,
    pub act_elem: u32,
    /// `softmax_scale = 1 / sqrt(head_dim)`. Baked literal.
    pub softmax_scale: f32,
    /// Typed ZST for the runtime u32 arg the kernel scaffold provides
    /// for the number of KV iterations this query streams. Display
    /// fmt emits the canonical name `"__num_kv_pages"` — the SOLE
    /// source of truth shared with `fixtures::orchestrator_kernel_args`'s
    /// kernel-sig declaration. Drift between emit and sig is a
    /// compile error (sealed trait, ZST alone constructs the name).
    pub num_kv_pages_arg: crate::tk_warp_ir::NumKvPagesSym,
    /// Per-AttnDecode unique id, used as a suffix on the function-
    /// scope prelude variable names (`__q_smem_a0`, `__m_max_a0`, …)
    /// so multiple AttnDecode ops in one TkProgram (e.g. one per
    /// transformer block in a multi-layer Llama forward) don't
    /// collide on the same identifiers. The orchestrator passes the
    /// LoweringInput op index — unique across all ops in a forward.
    pub unique_id: u32,
}

impl AttnDecodeOp {
    /// Bind this op's K/V cache layout to the single
    /// [`KvCacheLayout`] source. The same layout instance is
    /// produced by the matching `RopeAppendOp::kv_layout` of the
    /// upstream RopeAppend; both READ and WRITE sides agree by
    /// construction.
    pub const fn kv_layout(&self) -> KvCacheLayout {
        KvCacheLayout::for_buf_id(self.k_cache, self.num_kv_heads, self.head_dim, self.act_elem)
    }
}

/// Lower one decode-attention op into a `TkProgram` fragment.
///
/// Tape shape:
///   1. Q-load (one round on the Q page; outside the loop). Loader
///      TMA-loads Q, consumer waits on Ready, the consumer body
///      initialises the softmax accumulator, arrive Done. Storer is
///      not used for Q (Q is read-only inside the loop).
///   2. KV sweep `for (i = 0; i < num_kv_pages; ++i)`: each iteration
///      is one *complete round* on the K page and one on the V page.
///      Inside the loop the wait parity is `(i & 1)` — the typed
///      model can't track per-iteration flips so the lowering binds
///      the parity to the loop variable directly.
///   3. Final softmax-normalise + O-store. Consumer divides the
///      accumulator by `l_sum`, storer TMA-stores the result.
pub fn lower_attn_decode<P: Phase>(
    op: AttnDecodeOp,
    k_cache: crate::tk_gmem::Fenced<crate::tk_gmem::GmemHandle<crate::tk_gmem::KCache>>,
    v_cache: crate::tk_gmem::Fenced<crate::tk_gmem::GmemHandle<crate::tk_gmem::VCache>>,
    pages: &mut PageAllocator,
    prog: &mut TkProgram,
) {
    // E.13: typed fenced handles confirm that
    // `tk_gmem::emit_fence_after_op` was emitted between the producing
    // op (RopeAppend) and this read. The handles' buf_ids must match
    // op.k_cache / op.v_cache — debug_assert at runtime, but the
    // fence emit is guaranteed by Rust types.
    let k_cache = k_cache.into_inner();
    let v_cache = v_cache.into_inner();
    debug_assert_eq!(op.k_cache, k_cache.buf_id());
    debug_assert_eq!(op.v_cache, v_cache.buf_id());
    // Phase 3 (paris invariant `kv-layout-witness-binds-cache-bufid`):
    // assert the K-side layout witness binds the same BufId as the
    // op's k_cache field. `op.kv_layout()` mints `for_buf_id(op.k_cache, ...)`
    // so this is by construction; verify in case a future refactor.
    debug_assert_eq!(op.kv_layout().cache_buf_id(), op.k_cache);
    let _ = (k_cache, v_cache);
    // Phase 7 GQA shape preconditions:
    // - `num_kv_heads <= NUM_CONSUMER_WARPS` (each kv-head owned by
    //   one consumer warp; warps with `__consumer_idx >= num_kv_heads`
    //   skip AttnDecode compute but still arrive on barriers).
    // - `num_q_heads % num_kv_heads == 0` (GQA grouping; per-kv
    //   q-heads identical per warp).
    //
    // Llama-3.2-1B (32 q, 8 kv) at NUM_CONSUMER_WARPS=16: warps 0..7
    // each take one kv-head + 4 q-heads; warps 8..15 idle on AttnDecode
    // (they still do barrier arrives). Phase 9 (register-tile online
    // softmax) revisits this sharding to use all 16 warps.
    debug_assert!(
        op.num_kv_heads <= NUM_CONSUMER_WARPS as u32,
        "lower_attn_decode: num_kv_heads ({}) must be <= NUM_CONSUMER_WARPS ({})",
        op.num_kv_heads,
        NUM_CONSUMER_WARPS,
    );
    debug_assert!(
        op.num_q_heads % op.num_kv_heads == 0,
        "lower_attn_decode: num_q_heads ({}) must be divisible by num_kv_heads ({}) for GQA",
        op.num_q_heads,
        op.num_kv_heads,
    );

    // Allocate all three page slots up front so the function-scope
    // prelude (typed page views, persistent compute accumulators) can
    // bind to known ids before any handshake instruction is emitted.
    let q_page = pages.alloc_at::<P>().expect("Q page");
    let k_page = pages.alloc_at::<P>().expect("K page");
    let v_page = pages.alloc_at::<P>().expect("V page");
    let q_id = q_page.id();
    let k_id = k_page.id();
    let v_id = v_page.id();
    populate_attn_decode_prelude(prog, &op, q_id, k_id, v_id);

    // Tile shapes:
    //   Q  : `[1, num_q_heads * head_dim]`  — full row, all heads.
    //   K,V: `[1, num_kv_heads * head_dim]` — one row per KV iter, all kv-heads.
    //   O  : same as Q.
    let q_cols = op.num_q_heads * op.head_dim;
    let kv_cols = op.num_kv_heads * op.head_dim;

    // ── Q page (one-shot) ──
    let q_region = RegionRef::rows_cols(op.q, 1, 0, q_cols);
    let q_tile = TileShape {
        rows: 1,
        cols: q_cols,
        elem_bytes: op.act_elem,
    };

    let q_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, q_page);
    prog.load_async(q_id, op.q, q_region, q_tile);
    // No `arrive(Ready)` — `tma::load_async` signals page_ready itself.

    let q_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, q_page);
    prog.compute_calls(
        WarpRole::AllConsumers,
        crate::tk_codegen::attn_decode_init_softmax_compute_calls(op.unique_id),
    );
    // No `arrive(Done)` here — the Q+O slot is ONE round: loader fills
    // (TMA load_async signals page_ready), the consumer holds the page
    // through the KV loop, writes O into the same page slot at the
    // finalise step, then signals page_done; the storer waits on done,
    // drains O, signals page_consumed. Two consumer arrives on
    // page_done would flip the barrier twice, breaking the round
    // invariant (storer's wait would hang on the wrong parity).

    // ── KV sweep ──
    // K and V tiles carry ALL kv-heads of the current iteration's
    // token (not a single head): each consumer warp reads its own
    // kv-head's slice from the tile via `__kv_head * __head_dim`
    // offsetting in the compute body. The per-page byte budget at
    // Llama-3.2-1B is 8 * 64 * 2 = 1024 B per K (and V) tile —
    // far below PAGE_SIZE.
    let k_tile = TileShape {
        rows: 1,
        cols: kv_cols,
        elem_bytes: op.act_elem,
    };
    let v_tile = k_tile;

    let loop_var = "__kv_i";
    // Capture K and V pages' static phase at loop entry. See the
    // `wait_loop_parity` doc on `tk_warp_ir.rs` for why iter-0 must
    // read the page's static phase, not a hardcoded 0 — same root
    // cause as the op4/Gemm deadlock when slot reuse straddled an
    // odd number of prior cycles.
    let start = P::VALUE;
    // Per-iteration K/V row stride. Each token's K (or V) row
    // occupies `kv_cols * act_elem` bytes (Llama-1B: 8 kv_heads *
    // 64 head_dim * 2 = 1024 B). `body.iter_offset` constructs the
    // typed [`LoopOffset`] = `(__kv_i * stride)` — the only
    // expression load_async_dyn accepts inside a for_loop body.
    // NOTE: proper paged-cache port indirects through block_table —
    // `block_table[__kv_i / block_size] * block_bytes + (__kv_i %
    // block_size) * row_bytes`. Single-block sequences match the
    // flat stride; multi-block needs block-table indirection.
    //
    // `kv_row_bytes` comes from the typed [`KvCacheLayout`] so the
    // READ stride here matches the WRITE stride RopeAppend uses
    // for the matching K/V cache slot — a single-source-of-truth
    // construction (paris invariant
    // `kv-cache-write-slot-offset-correctness`).
    let kv_row_bytes = op.kv_layout().row_bytes();
    // Structural enforcement (Gap 17): for_loop_runtime is the ONLY
    // path that constructs `LoopBound::RuntimeU32` (the constructor
    // is sealed). Plain `prog.for_loop(_, LoopBound::RuntimeU32(...))`
    // is now a Rust compile error — verifying that the substrate
    // catches the legacy buggy form structurally.
    let post_pages = prog.for_loop_runtime(
        loop_var,
        op.num_kv_pages_arg,
        vec![k_page, v_page],
        |body| {
            // Per-iteration K round. Parity = (__kv_i & 1) ^ P::VALUE.
            body.wait_loop_parity(WarpRole::Loader, PageBarrier::Consumed, k_id, loop_var, start);
            body.load_async_dyn(
                k_id,
                op.k_cache,
                RegionRef::rows_cols(op.k_cache, 1, 0, kv_cols),
                k_tile,
                body.iter_offset(kv_row_bytes),
            );
            // No `arrive(Ready)` — `tma::load_async` signals page_ready.

            body.wait_loop_parity(WarpRole::AllConsumers, PageBarrier::Ready, k_id, loop_var, start);
            body.compute_calls(
                WarpRole::AllConsumers,
                crate::tk_codegen::attn_decode_qkt_softmax_step_compute_calls(
                    op.unique_id,
                    op.head_dim,
                ),
            );
            body.arrive_loop(WarpRole::AllConsumers, PageBarrier::Done, k_id);

            body.wait_loop_parity(WarpRole::Storer, PageBarrier::Done, k_id, loop_var, start);
            body.arrive_loop(WarpRole::Storer, PageBarrier::Consumed, k_id);

            // Per-iteration V round (same row_bytes per iter).
            body.wait_loop_parity(WarpRole::Loader, PageBarrier::Consumed, v_id, loop_var, start);
            body.load_async_dyn(
                v_id,
                op.v_cache,
                RegionRef::rows_cols(op.v_cache, 1, 0, kv_cols),
                v_tile,
                body.iter_offset(kv_row_bytes),
            );
            // No `arrive(Ready)` — `tma::load_async` signals page_ready.

            body.wait_loop_parity(WarpRole::AllConsumers, PageBarrier::Ready, v_id, loop_var, start);
            body.compute_calls(
                WarpRole::AllConsumers,
                crate::tk_codegen::attn_decode_sv_accum_compute_calls(op.unique_id),
            );
            body.arrive_loop(WarpRole::AllConsumers, PageBarrier::Done, v_id);

            body.wait_loop_parity(WarpRole::Storer, PageBarrier::Done, v_id, loop_var, start);
            body.arrive_loop(WarpRole::Storer, PageBarrier::Consumed, v_id);
        },
    );

    // After the loop the K/V pages have been ping-ponged a RUNTIME
    // number of times (`__num_kv_pages`). Each barrier flips ONCE
    // per iter; for even N hardware doesn't net-flip, for odd N it
    // does. The legacy `complete_round` blindly advances typed phase
    // by 1 — for even N this corrupts state and the next op deadlocks.
    //
    // **Compile-time enforcement (Gap 17, fixes FERRITE_WAVEFRONT_GPU=1
    // first-decode hang)**: wrap the K and V handles in
    // `PageHandleAfterRuntimeLoop` and release ONLY through
    // `complete_round_with_parity_correction`, which emits the
    // runtime phantom round automatically. Calling plain
    // `complete_round(k_page)` on `PageHandleAfterRuntimeLoop` would
    // be a Rust compile error — there's no impl that accepts the
    // post-loop handle without the parity correction.
    // for_loop_runtime returned post-loop wrappers in the same order
    // as input: [k_page_post, v_page_post]. The wrappers are the only
    // type complete_round_with_parity_correction accepts — and that
    // function emits the phantom round automatically.
    let mut post_pages_iter = post_pages.into_iter();
    let k_page_post = post_pages_iter.next().expect("k post-loop handle");
    let v_page_post = post_pages_iter.next().expect("v post-loop handle");
    let k_page = prog.complete_round_with_parity_correction(k_page_post, op.num_kv_pages_arg);
    let v_page = prog.complete_round_with_parity_correction(v_page_post, op.num_kv_pages_arg);
    pages.release(k_page);
    pages.release(v_page);

    // ── Final O-store ──
    // Output is `[1, num_q_heads * head_dim]` (matches Q tile width).
    let o_page = q_page; // reuse Q's page slot for O after the loop.
    let o_region = RegionRef::rows_cols(op.out, 1, 0, q_cols);
    let o_tile = TileShape {
        rows: 1,
        cols: q_cols,
        elem_bytes: op.act_elem,
    };
    // No `wait(Consumed)` here — the consumer has been holding this
    // page slot since the Q-load Ready handshake; the page is already
    // owned. Going straight to compute + arrive(Done) closes the slot's
    // single round (loader Ready → consumer Done → storer Consumed).
    prog.compute_calls(
        WarpRole::AllConsumers,
        crate::tk_codegen::attn_decode_finalise_softmax_norm_compute_calls(op.unique_id),
    );
    let o_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, o_page);

    let o_page = prog.wait(WarpRole::Storer, PageBarrier::Done, o_page);
    prog.store_async(q_id, op.out, o_region, o_tile);
    let o_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, o_page);

    let o_page = prog.complete_round(o_page);
    pages.release(o_page);
}

/// Phase 12: routing-aware variant of `lower_attn_decode`.
///
/// Inputs (3 slots): `q`, `k_cache`, `v_cache`. Only `q` is a
/// carry-forward candidate; `k_cache` / `v_cache` are the paged KV
/// cache, always external (`InputRef::Ext`) and gmem-loaded.
///
/// Output: in-place on the Q+O slot (Q's page is reused as O after
/// the KV-sweep finalise step). When `output_internal=true`, the
/// storer skips `tma::store_async` and the Q+O slot is carried-
/// forward to the consuming op (typically `out_proj`).
///
/// Default-hints path is byte-identical to `lower_attn_decode`.
pub fn lower_attn_decode_routed<P: Phase>(
    op: AttnDecodeOp,
    k_cache: crate::tk_gmem::Fenced<crate::tk_gmem::GmemHandle<crate::tk_gmem::KCache>>,
    v_cache: crate::tk_gmem::Fenced<crate::tk_gmem::GmemHandle<crate::tk_gmem::VCache>>,
    hints: &RoutingHints,
    pages: &mut PageAllocator,
    prog: &mut TkProgram,
) -> RoutingResult {
    let k_cache = k_cache.into_inner();
    let v_cache = v_cache.into_inner();
    debug_assert_eq!(op.k_cache, k_cache.buf_id());
    debug_assert_eq!(op.v_cache, v_cache.buf_id());
    // Phase 3 (paris invariant `kv-layout-witness-binds-cache-bufid`):
    // assert the K-side layout witness binds the same BufId as the
    // op's k_cache field. `op.kv_layout()` mints `for_buf_id(op.k_cache, ...)`
    // so this is by construction; verify in case a future refactor.
    debug_assert_eq!(op.kv_layout().cache_buf_id(), op.k_cache);
    let _ = (k_cache, v_cache);
    debug_assert!(
        op.num_kv_heads <= NUM_CONSUMER_WARPS as u32,
        "lower_attn_decode_routed: num_kv_heads ({}) must be <= NUM_CONSUMER_WARPS ({})",
        op.num_kv_heads,
        NUM_CONSUMER_WARPS,
    );
    debug_assert!(
        op.num_q_heads % op.num_kv_heads == 0,
        "lower_attn_decode_routed: num_q_heads ({}) must be divisible by num_kv_heads ({}) for GQA",
        op.num_q_heads,
        op.num_kv_heads,
    );
    debug_assert!(
        hints.inputs.is_empty() || hints.inputs.len() == 3,
        "lower_attn_decode_routed: hints.inputs must be empty or len 3 (q, k_cache, v_cache)"
    );
    let in_q_carried = hints.inputs.first().and_then(|x| x.as_ref());
    // K/V cache always Ext; ignore those hints if any.

    let q_page: PageHandle<P> = match in_q_carried {
        Some(c) => pages.consume_carried::<P>(*c),
        None => pages.alloc_at::<P>().expect("Q page"),
    };
    let k_page = pages.alloc_at::<P>().expect("K page");
    let v_page = pages.alloc_at::<P>().expect("V page");
    let q_id = q_page.id();
    let k_id = k_page.id();
    let v_id = v_page.id();
    populate_attn_decode_prelude(prog, &op, q_id, k_id, v_id);

    let q_cols = op.num_q_heads * op.head_dim;
    let kv_cols = op.num_kv_heads * op.head_dim;

    let q_region = RegionRef::rows_cols(op.q, 1, 0, q_cols);
    let q_tile = TileShape {
        rows: 1,
        cols: q_cols,
        elem_bytes: op.act_elem,
    };

    // ── Q page (one-shot): load or skip-and-arrive-Ready ──
    let q_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, q_page);
    if in_q_carried.is_some() {
        prog.arrive(WarpRole::Loader, PageBarrier::Ready, q_page);
    } else {
        prog.load_async(q_id, op.q, q_region, q_tile);
    }

    let q_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, q_page);
    prog.compute_calls(
        WarpRole::AllConsumers,
        crate::tk_codegen::attn_decode_init_softmax_compute_calls(op.unique_id),
    );

    // ── KV sweep — unchanged from legacy (K/V always Ext) ──
    let k_tile = TileShape {
        rows: 1,
        cols: kv_cols,
        elem_bytes: op.act_elem,
    };
    let v_tile = k_tile;
    let loop_var = "__kv_i";
    let start = P::VALUE;
    // Per-iter K/V row stride; see `lower_attn_decode` for the
    // block-table-indirection caveat. Bound to the typed
    // [`KvCacheLayout`] so READ stride here matches WRITE stride
    // in the matching RopeAppend (paris invariant
    // `kv-cache-write-slot-offset-correctness`).
    let kv_row_bytes = op.kv_layout().row_bytes();
    // Structural enforcement (Gap 17) for the routed AttnDecode too.
    let post_pages = prog.for_loop_runtime(
        loop_var,
        op.num_kv_pages_arg,
        vec![k_page, v_page],
        |body| {
            body.wait_loop_parity(WarpRole::Loader, PageBarrier::Consumed, k_id, loop_var, start);
            body.load_async_dyn(
                k_id,
                op.k_cache,
                RegionRef::rows_cols(op.k_cache, 1, 0, kv_cols),
                k_tile,
                body.iter_offset(kv_row_bytes),
            );
            body.wait_loop_parity(WarpRole::AllConsumers, PageBarrier::Ready, k_id, loop_var, start);
            body.compute_calls(
                WarpRole::AllConsumers,
                crate::tk_codegen::attn_decode_qkt_softmax_step_compute_calls(
                    op.unique_id,
                    op.head_dim,
                ),
            );
            body.arrive_loop(WarpRole::AllConsumers, PageBarrier::Done, k_id);
            body.wait_loop_parity(WarpRole::Storer, PageBarrier::Done, k_id, loop_var, start);
            body.arrive_loop(WarpRole::Storer, PageBarrier::Consumed, k_id);

            body.wait_loop_parity(WarpRole::Loader, PageBarrier::Consumed, v_id, loop_var, start);
            body.load_async_dyn(
                v_id,
                op.v_cache,
                RegionRef::rows_cols(op.v_cache, 1, 0, kv_cols),
                v_tile,
                body.iter_offset(kv_row_bytes),
            );
            body.wait_loop_parity(WarpRole::AllConsumers, PageBarrier::Ready, v_id, loop_var, start);
            body.compute_calls(
                WarpRole::AllConsumers,
                crate::tk_codegen::attn_decode_sv_accum_compute_calls(op.unique_id),
            );
            body.arrive_loop(WarpRole::AllConsumers, PageBarrier::Done, v_id);
            body.wait_loop_parity(WarpRole::Storer, PageBarrier::Done, v_id, loop_var, start);
            body.arrive_loop(WarpRole::Storer, PageBarrier::Consumed, v_id);
        },
    );
    // for_loop_runtime returned post-loop wrappers; structurally
    // forces complete_round_with_parity_correction (Gap 17).
    let mut post_pages_iter = post_pages.into_iter();
    let k_page_post = post_pages_iter.next().expect("k post-loop handle");
    let v_page_post = post_pages_iter.next().expect("v post-loop handle");
    let k_page = prog.complete_round_with_parity_correction(k_page_post, op.num_kv_pages_arg);
    let v_page = prog.complete_round_with_parity_correction(v_page_post, op.num_kv_pages_arg);
    pages.release(k_page);
    pages.release(v_page);

    // ── Final O — write output to Q+O slot; skip storer drain when internal ──
    let o_page = q_page;
    let o_region = RegionRef::rows_cols(op.out, 1, 0, q_cols);
    let o_tile = TileShape {
        rows: 1,
        cols: q_cols,
        elem_bytes: op.act_elem,
    };
    prog.compute_calls(
        WarpRole::AllConsumers,
        crate::tk_codegen::attn_decode_finalise_softmax_norm_compute_calls(op.unique_id),
    );
    let o_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, o_page);

    let o_page = prog.wait(WarpRole::Storer, PageBarrier::Done, o_page);
    if !hints.output_internal {
        prog.store_async(q_id, op.out, o_region, o_tile);
    }
    let o_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, o_page);

    let o_advanced = prog.complete_round(o_page);
    if hints.output_internal {
        let carried = pages.carry_forward(o_advanced);
        RoutingResult {
            output_carried: Some(carried),
        }
    } else {
        pages.release(o_advanced);
        RoutingResult::default()
    }
}

/// Function-scope state for the multi-head GQA AttnDecode lowering.
/// Declared once at kernel entry; every consumer compute step
/// references these names from inside its own role-arm block.
///
/// **Per-warp head sharding.** With `num_q_heads = N_q` and 8 consumer
/// warps, each warp owns `N_q / 8` q-heads. Per-warp accumulator
/// arrays (`__m_max`, `__l_sum`, `__o_accum`, …) are sized to that
/// per-warp count — every consumer warp keeps its own softmax state
/// and writes its own slice of the output buffer.
///
/// **GQA grouping.** Each consumer warp `c ∈ [0, num_kv_heads)` is
/// responsible for kv-head `c`. The `q_per_kv = num_q_heads /
/// num_kv_heads` q-heads it sees all read from kv-head `c`. For
/// Llama-3.2-1B (`N_q=32, N_kv=8, q_per_kv=4`): warp 0 handles q-heads
/// 0..3 against kv-head 0; warp 1 handles q-heads 4..7 against
/// kv-head 1; … warp 7 handles q-heads 28..31 against kv-head 7.
///
/// **Constraint:** `num_q_heads % 8 == 0` and `num_kv_heads == 8` —
/// pre-checked at the orchestrator entry; the lowering panics in
/// `lower_attn_decode` if violated.
fn populate_attn_decode_prelude(
    prog: &mut TkProgram,
    op: &AttnDecodeOp,
    q_id: u8,
    k_id: u8,
    v_id: u8,
) {
    // Phase 7: each ACTIVE consumer warp (`__consumer_idx <
    // num_kv_heads`) owns one kv-head + `q_per_kv` q-heads. Idle
    // warps (`__consumer_idx >= num_kv_heads`) skip the body. With
    // NUM_CONSUMER_WARPS=16 and Llama-1B (8 kv, 32 q), 8 active
    // warps × 4 q-heads each.
    let q_per_kv = op.num_q_heads / op.num_kv_heads;
    let q_heads_per_warp = q_per_kv;
    let prelude = crate::tk_codegen::tk20::attn_decode_prelude(
        q_id,
        k_id,
        v_id,
        op.head_dim,
        op.num_q_heads,
        op.num_kv_heads,
        q_heads_per_warp,
        q_per_kv,
        op.softmax_scale,
        op.unique_id,
    );
    prog.add_prelude(prelude);
}

// Phase 5 cutover: legacy AttnDecode body fns deleted. Canonical
// emit lives at `tk_codegen::tk20::attn_decode_{init_softmax_body,
// qkt_softmax_step_body, sv_accum_step_body,
// finalise_softmax_norm_body}`, invoked through the corresponding
// Tk20Call::AttnDecode* variants.

// ── Residual Add — element-wise add into TkProgram ─────────────────

/// Inputs to lower one decode-shape element-wise add (e.g. the
/// post-attention residual `o + x_residual`). Mirrors `LoweredOp::Add`.
#[derive(Clone, Copy, Debug)]
pub struct AddOp {
    /// `[m, hidden]` first input.
    pub a: BufId,
    /// `[m, hidden]` second input.
    pub b: BufId,
    /// `[m, hidden]` output (may alias `a` for in-place residual).
    pub out: BufId,
    pub hidden: u32,
    pub m: u32,
    pub act_elem: u32,
}

/// Lower one element-wise Add into a `TkProgram` fragment.
///
/// Tape shape — two pages (one per input), same round parity, single
/// consumer step. Output is written to `a`'s page in place; the storer
/// drains that page to `out`. The B page is released a half-round
/// earlier (after the consumer's read), and lives in a sub-round whose
/// closing arrive is on the *consumed* barrier so it's free again for
/// the next op.
///
/// We emit ONE round's worth of barriers per page slot:
/// - A page (slot α): loader fills, consumer reads + writes A+B back,
///   storer drains.
/// - B page (slot β): loader fills, consumer reads, storer arrives
///   `Consumed` to release the slot. (Storer does NO TMA store on B —
///   B is read-only; we use the storer warp purely to flip the
///   `Consumed` barrier so the next op can reuse the slot.)
pub fn lower_residual_add<P: Phase>(
    op: AddOp,
    pages: &mut PageAllocator,
    prog: &mut TkProgram,
) {
    let a_page = pages.alloc_at::<P>().expect("residual add: A page");
    let b_page = pages.alloc_at::<P>().expect("residual add: B page");
    let a_id = a_page.id();
    let b_id = b_page.id();

    let region = |buf, rows, hidden| RegionRef::rows_cols(buf, rows, 0, hidden);
    let tile = TileShape {
        rows: op.m,
        cols: op.hidden,
        elem_bytes: op.act_elem,
    };

    // ── Loader fills A ──
    let a_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, a_page);
    prog.load_async(a_id, op.a, region(op.a, op.m, op.hidden), tile);

    // ── Loader fills B ──
    let b_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, b_page);
    prog.load_async(b_id, op.b, region(op.b, op.m, op.hidden), tile);

    // ── Consumer waits on Ready for both, computes A+B in place on A,
    //    arrives Done on both. ──
    let a_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, a_page);
    let b_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, b_page);
    prog.compute_calls(
        WarpRole::AllConsumers,
        crate::tk_codegen::residual_add_compute_calls(
            a_id,
            b_id,
            op.hidden as u64 * op.m as u64,
        ),
    );
    let a_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, a_page);
    let b_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, b_page);

    // ── Storer drains A → out, then frees both pages. ──
    let a_page = prog.wait(WarpRole::Storer, PageBarrier::Done, a_page);
    prog.store_async(a_id, op.out, region(op.out, op.m, op.hidden), tile);
    let a_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, a_page);

    let b_page = prog.wait(WarpRole::Storer, PageBarrier::Done, b_page);
    let b_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, b_page);

    pages.release(prog.complete_round(a_page));
    pages.release(prog.complete_round(b_page));
}

// Phase 5 cutover: legacy `residual_add_compute_body` deleted.
// Canonical emit at `tk_codegen::tk20::residual_add_consumer_body`.

/// Phase 12: routing-aware variant of `lower_residual_add`. Same
/// compute body, same intra-IType barrier protocol, but:
///
/// - For each input slot whose `hints.inputs[i] = Some(carried)`,
///   the lowering uses `pages.consume_carried::<P>(carried)` instead
///   of `alloc_at::<P>()`, and the loader emits a bare
///   `arrive(Ready[id])` instead of `tma::load_async`. The page
///   already holds the producer's data; the producer's storer's
///   `arrive Consumed` already flipped the slot's parity to `P`, so
///   the consumer's standard `wait Consumed[id]@P` passes through
///   without re-loading.
///
/// - When `hints.output_internal = true`, the storer skips
///   `tma::store_async` (the consuming op reads from this op's smem
///   page directly) and the lowering returns a `CarriedHandle` so
///   the orchestrator can thread it to the consuming op's `inputs`.
///
/// The non-routing path (no carry-forward inputs, no internal
/// output) is byte-identical to `lower_residual_add` — same instr
/// count, same emit, same page allocator state.
pub fn lower_residual_add_routed<P: Phase>(
    op: AddOp,
    hints: &RoutingHints,
    pages: &mut PageAllocator,
    prog: &mut TkProgram,
) -> RoutingResult {
    debug_assert!(
        hints.inputs.is_empty() || hints.inputs.len() == 2,
        "lower_residual_add_routed: hints.inputs must be empty or len 2"
    );
    let in_a_carried = hints.inputs.first().and_then(|x| x.as_ref());
    let in_b_carried = hints.inputs.get(1).and_then(|x| x.as_ref());

    // ── Allocate or consume A ──
    let a_page: PageHandle<P> = match in_a_carried {
        Some(c) => pages.consume_carried::<P>(*c),
        None => pages.alloc_at::<P>().expect("residual add: A page"),
    };
    let b_page: PageHandle<P> = match in_b_carried {
        Some(c) => pages.consume_carried::<P>(*c),
        None => pages.alloc_at::<P>().expect("residual add: B page"),
    };
    let a_id = a_page.id();
    let b_id = b_page.id();

    let region = |buf, rows, hidden| RegionRef::rows_cols(buf, rows, 0, hidden);
    let tile = TileShape {
        rows: op.m,
        cols: op.hidden,
        elem_bytes: op.act_elem,
    };

    // ── Loader fills A (or skips for carry-forward) ──
    let a_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, a_page);
    if in_a_carried.is_some() {
        // Page already holds producer's data; loader just flips the
        // Ready barrier so the consumer's wait Ready passes.
        prog.arrive(WarpRole::Loader, PageBarrier::Ready, a_page);
    } else {
        prog.load_async(a_id, op.a, region(op.a, op.m, op.hidden), tile);
    }

    // ── Loader fills B (or skips for carry-forward) ──
    let b_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, b_page);
    if in_b_carried.is_some() {
        prog.arrive(WarpRole::Loader, PageBarrier::Ready, b_page);
    } else {
        prog.load_async(b_id, op.b, region(op.b, op.m, op.hidden), tile);
    }

    // ── Consumer reads + computes (compute body is unchanged). ──
    let a_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, a_page);
    let b_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, b_page);
    prog.compute_calls(
        WarpRole::AllConsumers,
        crate::tk_codegen::residual_add_compute_calls(
            a_id,
            b_id,
            op.hidden as u64 * op.m as u64,
        ),
    );
    let a_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, a_page);
    let b_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, b_page);

    // ── Storer drains A → out (or skips for internal output). ──
    let a_page = prog.wait(WarpRole::Storer, PageBarrier::Done, a_page);
    if !hints.output_internal {
        prog.store_async(a_id, op.out, region(op.out, op.m, op.hidden), tile);
    }
    let a_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, a_page);

    let b_page = prog.wait(WarpRole::Storer, PageBarrier::Done, b_page);
    let b_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, b_page);

    let a_advanced = prog.complete_round(a_page);
    let b_advanced = prog.complete_round(b_page);

    // ── Output: carry-forward A if internal, else release. ──
    let result = if hints.output_internal {
        let carried = pages.carry_forward(a_advanced);
        // B is read-only; always release (B's data isn't the
        // op's output).
        pages.release(b_advanced);
        RoutingResult {
            output_carried: Some(carried),
        }
    } else {
        pages.release(a_advanced);
        pages.release(b_advanced);
        RoutingResult::default()
    };

    result
}

// ── SiluMul — fused SwiGLU element-wise ────────────────────────────

/// Inputs to lower one fused `silu(gate) * up` op (the SwiGLU
/// activation), mirroring `LoweredOp::SiluMul`.
#[derive(Clone, Copy, Debug)]
pub struct SiluMulOp {
    /// `[m, intermediate]` gate projection.
    pub gate: BufId,
    /// `[m, intermediate]` up projection.
    pub up: BufId,
    /// `[m, intermediate]` output (may alias `gate`).
    pub out: BufId,
    /// Intermediate size (the inner dim of the SwiGLU MLP).
    pub intermediate: u32,
    pub m: u32,
    pub act_elem: u32,
}

/// Lower one `silu(gate) * up` op into a `TkProgram` fragment.
///
/// Two pages — gate page is overwritten in place with the result and
/// drained to `out`; up page is read-only (storer arrives Consumed only).
/// One round on both slots, parity 0.
pub fn lower_silu_mul<P: Phase>(
    op: SiluMulOp,
    pages: &mut PageAllocator,
    prog: &mut TkProgram,
) {
    let g_page = pages.alloc_at::<P>().expect("silu_mul: gate page");
    let u_page = pages.alloc_at::<P>().expect("silu_mul: up page");
    let g_id = g_page.id();
    let u_id = u_page.id();

    let region = |buf, rows, cols| RegionRef::rows_cols(buf, rows, 0, cols);
    let tile = TileShape {
        rows: op.m,
        cols: op.intermediate,
        elem_bytes: op.act_elem,
    };

    let g_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, g_page);
    prog.load_async(g_id, op.gate, region(op.gate, op.m, op.intermediate), tile);

    let u_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, u_page);
    prog.load_async(u_id, op.up, region(op.up, op.m, op.intermediate), tile);

    let g_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, g_page);
    let u_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, u_page);
    prog.compute_calls(
        WarpRole::AllConsumers,
        crate::tk_codegen::silu_mul_compute_calls(
            g_id,
            u_id,
            op.intermediate as u64 * op.m as u64,
        ),
    );

    let g_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, g_page);
    let u_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, u_page);

    let g_page = prog.wait(WarpRole::Storer, PageBarrier::Done, g_page);
    prog.store_async(g_id, op.out, region(op.out, op.m, op.intermediate), tile);
    let g_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, g_page);

    let u_page = prog.wait(WarpRole::Storer, PageBarrier::Done, u_page);
    let u_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, u_page);

    pages.release(prog.complete_round(g_page));
    pages.release(prog.complete_round(u_page));
}

/// Phase 12: routing-aware variant of `lower_silu_mul`. Same protocol
/// as `lower_residual_add_routed`. Both gate and up are typically
/// produced by upstream Gemms (gate_proj, up_proj) and consumed only
/// by this op + the downstream down_proj — so both can carry-forward
/// in. Output is in-place on gate's page; the storer drains gate_page
/// to op.out (the down_proj's input) — when output_internal=true,
/// drain skipped and gate_page is carried-forward.
///
/// Default-hints path is byte-identical to `lower_silu_mul`.
pub fn lower_silu_mul_routed<P: Phase>(
    op: SiluMulOp,
    hints: &RoutingHints,
    pages: &mut PageAllocator,
    prog: &mut TkProgram,
) -> RoutingResult {
    debug_assert!(
        hints.inputs.is_empty() || hints.inputs.len() == 2,
        "lower_silu_mul_routed: hints.inputs must be empty or len 2"
    );
    let in_g_carried = hints.inputs.first().and_then(|x| x.as_ref());
    let in_u_carried = hints.inputs.get(1).and_then(|x| x.as_ref());

    let g_page: PageHandle<P> = match in_g_carried {
        Some(c) => pages.consume_carried::<P>(*c),
        None => pages.alloc_at::<P>().expect("silu_mul: gate page"),
    };
    let u_page: PageHandle<P> = match in_u_carried {
        Some(c) => pages.consume_carried::<P>(*c),
        None => pages.alloc_at::<P>().expect("silu_mul: up page"),
    };
    let g_id = g_page.id();
    let u_id = u_page.id();

    let region = |buf, rows, cols| RegionRef::rows_cols(buf, rows, 0, cols);
    let tile = TileShape {
        rows: op.m,
        cols: op.intermediate,
        elem_bytes: op.act_elem,
    };

    let g_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, g_page);
    if in_g_carried.is_some() {
        prog.arrive(WarpRole::Loader, PageBarrier::Ready, g_page);
    } else {
        prog.load_async(g_id, op.gate, region(op.gate, op.m, op.intermediate), tile);
    }

    let u_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, u_page);
    if in_u_carried.is_some() {
        prog.arrive(WarpRole::Loader, PageBarrier::Ready, u_page);
    } else {
        prog.load_async(u_id, op.up, region(op.up, op.m, op.intermediate), tile);
    }

    let g_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, g_page);
    let u_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, u_page);
    prog.compute_calls(
        WarpRole::AllConsumers,
        crate::tk_codegen::silu_mul_compute_calls(
            g_id,
            u_id,
            op.intermediate as u64 * op.m as u64,
        ),
    );

    let g_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, g_page);
    let u_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, u_page);

    let g_page = prog.wait(WarpRole::Storer, PageBarrier::Done, g_page);
    if !hints.output_internal {
        prog.store_async(g_id, op.out, region(op.out, op.m, op.intermediate), tile);
    }
    let g_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, g_page);

    let u_page = prog.wait(WarpRole::Storer, PageBarrier::Done, u_page);
    let u_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, u_page);

    let g_advanced = prog.complete_round(g_page);
    let u_advanced = prog.complete_round(u_page);

    if hints.output_internal {
        let carried = pages.carry_forward(g_advanced);
        pages.release(u_advanced);
        RoutingResult {
            output_carried: Some(carried),
        }
    } else {
        pages.release(g_advanced);
        pages.release(u_advanced);
        RoutingResult::default()
    }
}

// Phase 5 cutover: legacy `silu_mul_compute_body` deleted.
// Canonical emit at `tk_codegen::tk20::silu_mul_consumer_body`.

// ── RoPE rotate — NeoX rotary on Q or K ────────────────────────────

/// Inputs to lower one NeoX-style RoPE rotate, mirroring
/// `LoweredOp::RopeRotate { head_dim }`.
///
/// Acts on `[m, num_heads * head_dim]` x in place. cos/sin are
/// `[1, head_dim]` (the model's per-token RoPE frequencies for the
/// current decode position).
#[derive(Clone, Copy, Debug)]
pub struct RopeRotateOp {
    pub x: BufId,
    pub cos: BufId,
    pub sin: BufId,
    /// Output buffer (typically aliases `x`).
    pub out: BufId,
    /// Per-head dim (the rotary dimension; same for Q and K/V).
    pub head_dim: u32,
    /// Number of heads sharing this RoPE invocation (Q heads when
    /// rotating Q; KV heads when rotating K).
    pub num_heads: u32,
    pub m: u32,
    pub act_elem: u32,
}

/// Lower one NeoX-RoPE rotate into a `TkProgram` fragment.
///
/// Three pages — x (in-place + drained), cos (read-only), sin (read-only).
pub fn lower_rope_rotate<P: Phase>(
    op: RopeRotateOp,
    pages: &mut PageAllocator,
    prog: &mut TkProgram,
) {
    let x_page = pages.alloc_at::<P>().expect("rope: x page");
    let c_page = pages.alloc_at::<P>().expect("rope: cos page");
    let s_page = pages.alloc_at::<P>().expect("rope: sin page");
    let x_id = x_page.id();
    let c_id = c_page.id();
    let s_id = s_page.id();

    let region = |buf, rows, cols| RegionRef::rows_cols(buf, rows, 0, cols);
    let x_cols = op.num_heads * op.head_dim;
    let x_tile = TileShape {
        rows: op.m,
        cols: x_cols,
        elem_bytes: op.act_elem,
    };
    let cs_tile = TileShape {
        rows: 1,
        cols: op.head_dim,
        elem_bytes: op.act_elem,
    };

    // Loader fills all three pages. Cos/sin are runtime-positional:
    // the dispatcher passes `cos_sin_cache.base` (cos) and
    // `cos_sin_cache.base + half_offset` (sin); the kernel adds
    // `__decode_position * row_bytes` so each decode token reads its
    // own row. `row_bytes = head_dim * act_elem` (one full row of the
    // packed `[max_pos, head_dim]` cache; the load deliberately fetches
    // 128 B = full row even though the rotation only touches the first
    // half_dim, since TMA loads benefit from row-aligned sizes).
    let row_bytes = op.head_dim * op.act_elem;
    let pos_off = format!("__decode_position * {row_bytes}u");
    let x_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, x_page);
    prog.load_async(x_id, op.x, region(op.x, op.m, x_cols), x_tile);
    let c_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, c_page);
    prog.load_async_dyn(
        c_id,
        op.cos,
        region(op.cos, 1, op.head_dim),
        cs_tile,
        pos_off.clone(),
    );
    let s_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, s_page);
    prog.load_async_dyn(
        s_id,
        op.sin,
        region(op.sin, 1, op.head_dim),
        cs_tile,
        pos_off,
    );

    let x_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, x_page);
    let c_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, c_page);
    let s_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, s_page);
    let half = op.head_dim / 2;
    let total_pairs = (op.m as u64) * (op.num_heads as u64) * (half as u64);
    prog.compute_calls(
        WarpRole::AllConsumers,
        crate::tk_codegen::rope_compute_calls::<crate::tk_codegen::NeoX>(
            x_id,
            c_id,
            s_id,
            op.head_dim,
            total_pairs,
        ),
    );
    let x_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, x_page);
    let c_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, c_page);
    let s_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, s_page);

    let x_page = prog.wait(WarpRole::Storer, PageBarrier::Done, x_page);
    prog.store_async(x_id, op.out, region(op.out, op.m, x_cols), x_tile);
    let x_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, x_page);

    let c_page = prog.wait(WarpRole::Storer, PageBarrier::Done, c_page);
    let c_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, c_page);

    let s_page = prog.wait(WarpRole::Storer, PageBarrier::Done, s_page);
    let s_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, s_page);

    pages.release(prog.complete_round(x_page));
    pages.release(prog.complete_round(c_page));
    pages.release(prog.complete_round(s_page));
}

/// Phase 12: routing-aware variant of `lower_rope_rotate`. Same
/// protocol as `lower_residual_add_routed`. Only x is a carry-
/// forward candidate (cos/sin are always Ext — the per-position
/// rotary tables, loaded via `__decode_position * row_bytes`).
/// Output is in-place on x's page; storer drains x_page → op.out
/// when output is external.
///
/// Default-hints path is byte-identical to `lower_rope_rotate`.
pub fn lower_rope_rotate_routed<P: Phase>(
    op: RopeRotateOp,
    hints: &RoutingHints,
    pages: &mut PageAllocator,
    prog: &mut TkProgram,
) -> RoutingResult {
    debug_assert!(
        hints.inputs.is_empty() || hints.inputs.len() == 3,
        "lower_rope_rotate_routed: hints.inputs must be empty or len 3"
    );
    let in_x_carried = hints.inputs.first().and_then(|x| x.as_ref());
    let in_c_carried = hints.inputs.get(1).and_then(|x| x.as_ref());
    let in_s_carried = hints.inputs.get(2).and_then(|x| x.as_ref());

    let x_page: PageHandle<P> = match in_x_carried {
        Some(c) => pages.consume_carried::<P>(*c),
        None => pages.alloc_at::<P>().expect("rope: x page"),
    };
    let c_page: PageHandle<P> = match in_c_carried {
        Some(c) => pages.consume_carried::<P>(*c),
        None => pages.alloc_at::<P>().expect("rope: cos page"),
    };
    let s_page: PageHandle<P> = match in_s_carried {
        Some(c) => pages.consume_carried::<P>(*c),
        None => pages.alloc_at::<P>().expect("rope: sin page"),
    };
    let x_id = x_page.id();
    let c_id = c_page.id();
    let s_id = s_page.id();

    let region = |buf, rows, cols| RegionRef::rows_cols(buf, rows, 0, cols);
    let x_cols = op.num_heads * op.head_dim;
    let x_tile = TileShape {
        rows: op.m,
        cols: x_cols,
        elem_bytes: op.act_elem,
    };
    let cs_tile = TileShape {
        rows: 1,
        cols: op.head_dim,
        elem_bytes: op.act_elem,
    };

    let row_bytes = op.head_dim * op.act_elem;
    let pos_off = format!("__decode_position * {row_bytes}u");

    let x_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, x_page);
    if in_x_carried.is_some() {
        prog.arrive(WarpRole::Loader, PageBarrier::Ready, x_page);
    } else {
        prog.load_async(x_id, op.x, region(op.x, op.m, x_cols), x_tile);
    }
    let c_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, c_page);
    if in_c_carried.is_some() {
        prog.arrive(WarpRole::Loader, PageBarrier::Ready, c_page);
    } else {
        prog.load_async_dyn(
            c_id,
            op.cos,
            region(op.cos, 1, op.head_dim),
            cs_tile,
            pos_off.clone(),
        );
    }
    let s_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, s_page);
    if in_s_carried.is_some() {
        prog.arrive(WarpRole::Loader, PageBarrier::Ready, s_page);
    } else {
        prog.load_async_dyn(
            s_id,
            op.sin,
            region(op.sin, 1, op.head_dim),
            cs_tile,
            pos_off,
        );
    }

    let x_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, x_page);
    let c_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, c_page);
    let s_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, s_page);
    let half = op.head_dim / 2;
    let total_pairs = (op.m as u64) * (op.num_heads as u64) * (half as u64);
    prog.compute_calls(
        WarpRole::AllConsumers,
        crate::tk_codegen::rope_compute_calls::<crate::tk_codegen::NeoX>(
            x_id,
            c_id,
            s_id,
            op.head_dim,
            total_pairs,
        ),
    );
    let x_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, x_page);
    let c_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, c_page);
    let s_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, s_page);

    let x_page = prog.wait(WarpRole::Storer, PageBarrier::Done, x_page);
    if !hints.output_internal {
        prog.store_async(x_id, op.out, region(op.out, op.m, x_cols), x_tile);
    }
    let x_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, x_page);

    let c_page = prog.wait(WarpRole::Storer, PageBarrier::Done, c_page);
    let c_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, c_page);

    let s_page = prog.wait(WarpRole::Storer, PageBarrier::Done, s_page);
    let s_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, s_page);

    let x_advanced = prog.complete_round(x_page);
    let c_advanced = prog.complete_round(c_page);
    let s_advanced = prog.complete_round(s_page);

    if hints.output_internal {
        let carried = pages.carry_forward(x_advanced);
        pages.release(c_advanced);
        pages.release(s_advanced);
        RoutingResult {
            output_carried: Some(carried),
        }
    } else {
        pages.release(x_advanced);
        pages.release(c_advanced);
        pages.release(s_advanced);
        RoutingResult::default()
    }
}

// Phase 5 cutover: legacy `rope_compute_body` deleted. Canonical
// emit at `tk_codegen::tk20::rope_consumer_body`.

// ── RoPE append — rotate K + write K/V to paged KV cache ───────────

/// Inputs to lower one decode `RopeAppend` op.
///
/// Mirrors `LoweredOp::RopeAppend { head_dim, layer }`. RopeAppend
/// rotates K (NeoX) and **writes** rotated K + un-rotated V to the
/// paged KV cache pools (`PrefixK` / `PrefixV` per layer) at the new
/// decode token's slot. The slot is `__decode_slot` (a kernel runtime
/// u32 arg, populated by `KernelU32ArgsBuilder::push_decode_slot`
/// from `ctx.slot_mapping[0]`).
///
/// Without this op, AttnDecode reads only prompt-prefill K/V from
/// the cache; new decode tokens never land in the cache and
/// subsequent decode steps softmax against stale tail rows
/// (the `Paris!!!!!!!!!` regression Step E.12 fixes).
#[derive(Clone, Copy, Debug)]
pub struct RopeAppendOp {
    /// `[m, num_kv_heads * head_dim]` K activation (k_proj output).
    pub k: BufId,
    /// `[1, head_dim]` rotary cos for the current position
    /// (TMA-loaded with `__decode_position * row_bytes` offset).
    pub cos: BufId,
    /// `[1, head_dim]` rotary sin (same TMA offset).
    pub sin: BufId,
    /// `[m, num_kv_heads * head_dim]` V activation (v_proj output) —
    /// passed through to the cache un-rotated.
    pub v: BufId,
    /// Output buffer for rotated K (existing arena edge — kept so
    /// the dataflow validator's downstream-of-RopeAppend edge
    /// doesn't break, even though AttnDecode in the megakernel
    /// reads from the cache, not this buffer).
    pub out: BufId,
    /// Paged K cache pool (`[num_blocks, block_size, num_kv_heads,
    /// head_dim]` BF16 — same layout as `kernels::reshape_and_cache`).
    /// Bridge-supplied via `PrefixK { layer }` source.
    pub k_cache: BufId,
    /// Paged V cache pool (same layout).
    pub v_cache: BufId,
    /// Per-head dim.
    pub head_dim: u32,
    /// Number of KV heads (the `num_heads` dim in the cache layout).
    pub num_kv_heads: u32,
    /// Decode rows. m=1 for standard decode.
    pub m: u32,
    pub act_elem: u32,
    /// Typed ZST for the kernel scaffold's decode-slot u32 arg.
    /// Display fmt emits `"__decode_slot"` — the SOLE source of
    /// truth shared with `fixtures::orchestrator_kernel_args`'s
    /// kernel-sig declaration. Drift between emit and sig is a
    /// compile error (sealed trait, ZST alone constructs the name).
    pub decode_slot_arg: crate::tk_warp_ir::DecodeSlotSym,
}

impl RopeAppendOp {
    /// Bind this op's K/V cache layout to the single
    /// [`KvCacheLayout`] source. Both the WRITE side (this
    /// lowering's storer cache writes) and the matching
    /// AttnDecode's READ side (`AttnDecodeOp::kv_layout`) reach
    /// for `kv_layout().slot_offset_expr(...)` /
    /// `kv_layout().row_bytes()` so the formula is a single
    /// definition shared between sites.
    pub const fn kv_layout(&self) -> KvCacheLayout {
        KvCacheLayout::for_buf_id(self.k_cache, self.num_kv_heads, self.head_dim, self.act_elem)
    }
}

/// Lower one RoPE-append into a `TkProgram` fragment.
///
/// Tape shape (4 page slots):
///   1. Loader: TMA-loads K (no offset), cos/sin (with
///      `__decode_position * row_bytes`), V (no offset).
///   2. AllConsumers: rotates K in place using the standard
///      `RopeConsumerBody` (V is held but untouched).
///   3. Storer drains:
///      - rotated K → `op.out` (existing arena edge).
///      - rotated K → `op.k_cache + (__decode_slot * row_bytes)`
///        (NEW cache write).
///      - V → `op.v_cache + (__decode_slot * row_bytes)`
///        (NEW cache write).
///   4. Round boundary on all four pages.
///
/// `row_bytes = num_kv_heads * head_dim * act_elem`. The dispatcher
/// has already resolved block-table indirection into a flat slot
/// index, so the kernel just multiplies.
pub fn lower_rope_append<P: Phase>(
    op: RopeAppendOp,
    k_cache_handle: crate::tk_gmem::GmemHandle<crate::tk_gmem::KCache>,
    v_cache_handle: crate::tk_gmem::GmemHandle<crate::tk_gmem::VCache>,
    pages: &mut PageAllocator,
    prog: &mut TkProgram,
) -> (
    crate::tk_gmem::GmemHandle<crate::tk_gmem::KCache>,
    crate::tk_gmem::GmemHandle<crate::tk_gmem::VCache>,
) {
    debug_assert_eq!(
        op.k_cache,
        k_cache_handle.buf_id(),
        "lower_rope_append: op.k_cache != handle.buf_id"
    );
    debug_assert_eq!(
        op.v_cache,
        v_cache_handle.buf_id(),
        "lower_rope_append: op.v_cache != handle.buf_id"
    );
    // Phase 3 (paris invariant `kv-layout-witness-binds-cache-bufid`):
    // the layout witness binds op.k_cache by construction.
    debug_assert_eq!(op.kv_layout().cache_buf_id(), op.k_cache);
    let k_page = pages.alloc_at::<P>().expect("rope_append: K page");
    let c_page = pages.alloc_at::<P>().expect("rope_append: cos page");
    let s_page = pages.alloc_at::<P>().expect("rope_append: sin page");
    let v_page = pages.alloc_at::<P>().expect("rope_append: V page");
    let k_id = k_page.id();
    let c_id = c_page.id();
    let s_id = s_page.id();
    let v_id = v_page.id();

    let region = |buf, rows, cols| RegionRef::rows_cols(buf, rows, 0, cols);
    let kv_cols = op.num_kv_heads * op.head_dim;
    let kv_tile = TileShape {
        rows: op.m,
        cols: kv_cols,
        elem_bytes: op.act_elem,
    };
    let cs_tile = TileShape {
        rows: 1,
        cols: op.head_dim,
        elem_bytes: op.act_elem,
    };

    // Per-token row stride for both the cos/sin cache and the paged
    // K/V cache, bound through the typed [`KvCacheLayout`] (paris
    // invariant `kv-cache-write-slot-offset-correctness`). The
    // matching AttnDecode read uses the SAME `op.kv_layout()` value
    // by construction — no drift between WRITE here and READ there.
    let layout = op.kv_layout();
    let pos_off = layout.cos_sin_offset_expr("__decode_position");
    let slot_off = layout.slot_offset_expr(op.decode_slot_arg);

    // ── Loader: fill K, cos, sin, V ──
    let k_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, k_page);
    prog.load_async(k_id, op.k, region(op.k, op.m, kv_cols), kv_tile);

    let c_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, c_page);
    prog.load_async_dyn(
        c_id,
        op.cos,
        region(op.cos, 1, op.head_dim),
        cs_tile,
        pos_off.clone(),
    );

    let s_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, s_page);
    prog.load_async_dyn(
        s_id,
        op.sin,
        region(op.sin, 1, op.head_dim),
        cs_tile,
        pos_off,
    );

    let v_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, v_page);
    prog.load_async(v_id, op.v, region(op.v, op.m, kv_cols), kv_tile);

    // ── Consumer: rotate K in place; V held read-only ──
    let k_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, k_page);
    let c_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, c_page);
    let s_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, s_page);
    let v_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, v_page);
    let half = op.head_dim / 2;
    let total_pairs = (op.m as u64) * (op.num_kv_heads as u64) * (half as u64);
    // Reuse the standard RopeConsumerBody — it acts on a single page
    // (`x_id` slot) using cos/sin slots; we point it at K. V is not
    // touched.
    prog.compute_calls(
        WarpRole::AllConsumers,
        crate::tk_codegen::rope_compute_calls::<crate::tk_codegen::NeoX>(
            k_id,
            c_id,
            s_id,
            op.head_dim,
            total_pairs,
        ),
    );
    let _ = half; // silence unused-binding warning if linter complains
    let k_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, k_page);
    let c_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, c_page);
    let s_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, s_page);
    let v_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, v_page);

    // ── Storer: drain rotated K → op.out + K_cache; drain V → V_cache ──
    let k_page = prog.wait(WarpRole::Storer, PageBarrier::Done, k_page);
    // Existing arena edge — keeps the dataflow validator + any future
    // arena-reading consumer happy.
    prog.store_async(k_id, op.out, region(op.out, op.m, kv_cols), kv_tile);
    // NEW: paged K cache write at __decode_slot.
    prog.store_async_dyn(
        k_id,
        op.k_cache,
        region(op.k_cache, op.m, kv_cols),
        kv_tile,
        slot_off.clone(),
    );
    let k_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, k_page);

    let c_page = prog.wait(WarpRole::Storer, PageBarrier::Done, c_page);
    let c_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, c_page);

    let s_page = prog.wait(WarpRole::Storer, PageBarrier::Done, s_page);
    let s_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, s_page);

    let v_page = prog.wait(WarpRole::Storer, PageBarrier::Done, v_page);
    // NEW: paged V cache write at __decode_slot. (V is otherwise
    // read-only inside the kernel — no arena drain.)
    prog.store_async_dyn(
        v_id,
        op.v_cache,
        region(op.v_cache, op.m, kv_cols),
        kv_tile,
        slot_off,
    );
    let v_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, v_page);

    pages.release(prog.complete_round(k_page));
    pages.release(prog.complete_round(c_page));
    pages.release(prog.complete_round(s_page));
    pages.release(prog.complete_round(v_page));

    // Cross-op gmem ordering for the K/V cache writes is emitted by
    // the orchestrator via `tk_gmem::emit_fence_after_op` between
    // this op and the next reader (typically AttnDecode). The typed
    // `Fenced<GmemHandle<...>>` substrate enforces fence emission
    // structurally; the fence is a 5-Instr atomic sequence
    // (Sync/TmaStoreCommitGroup/TmaStoreAsyncWait/Threadfence/Sync)
    // pushed by `TkProgram::emit_cross_op_gmem_fence`, with one
    // one-line codegen arm per Instr.
    (k_cache_handle, v_cache_handle)
}

/// Phase 12: routing-aware variant of `lower_rope_append`.
///
/// Inputs (4 carry-forward candidates: K, cos, sin, V — though
/// cos/sin are always `InputRef::Ext` and never carry-forward in
/// practice). K_cache / V_cache are external sources by definition
/// (per-layer paged cache pools), never carry-forward.
/// Output (rotated K) is in-place on the K page slot's smem; the
/// `op.out` arena drain stays as the carry-forward target when
/// `output_internal=true`.
///
/// Default-hints path is byte-identical to `lower_rope_append`.
pub fn lower_rope_append_routed<P: Phase>(
    op: RopeAppendOp,
    k_cache_handle: crate::tk_gmem::GmemHandle<crate::tk_gmem::KCache>,
    v_cache_handle: crate::tk_gmem::GmemHandle<crate::tk_gmem::VCache>,
    hints: &RoutingHints,
    pages: &mut PageAllocator,
    prog: &mut TkProgram,
) -> (
    RoutingResult,
    crate::tk_gmem::GmemHandle<crate::tk_gmem::KCache>,
    crate::tk_gmem::GmemHandle<crate::tk_gmem::VCache>,
) {
    debug_assert_eq!(op.k_cache, k_cache_handle.buf_id());
    debug_assert_eq!(op.v_cache, v_cache_handle.buf_id());
    debug_assert!(
        hints.inputs.is_empty() || hints.inputs.len() == 4,
        "lower_rope_append_routed: hints.inputs must be empty or len 4"
    );
    let in_k_carried = hints.inputs.first().and_then(|x| x.as_ref());
    let in_c_carried = hints.inputs.get(1).and_then(|x| x.as_ref());
    let in_s_carried = hints.inputs.get(2).and_then(|x| x.as_ref());
    let in_v_carried = hints.inputs.get(3).and_then(|x| x.as_ref());

    let k_page: PageHandle<P> = match in_k_carried {
        Some(c) => pages.consume_carried::<P>(*c),
        None => pages.alloc_at::<P>().expect("rope_append: K page"),
    };
    let c_page: PageHandle<P> = match in_c_carried {
        Some(c) => pages.consume_carried::<P>(*c),
        None => pages.alloc_at::<P>().expect("rope_append: cos page"),
    };
    let s_page: PageHandle<P> = match in_s_carried {
        Some(c) => pages.consume_carried::<P>(*c),
        None => pages.alloc_at::<P>().expect("rope_append: sin page"),
    };
    let v_page: PageHandle<P> = match in_v_carried {
        Some(c) => pages.consume_carried::<P>(*c),
        None => pages.alloc_at::<P>().expect("rope_append: V page"),
    };
    let k_id = k_page.id();
    let c_id = c_page.id();
    let s_id = s_page.id();
    let v_id = v_page.id();

    let region = |buf, rows, cols| RegionRef::rows_cols(buf, rows, 0, cols);
    let kv_cols = op.num_kv_heads * op.head_dim;
    let kv_tile = TileShape {
        rows: op.m,
        cols: kv_cols,
        elem_bytes: op.act_elem,
    };
    let cs_tile = TileShape {
        rows: 1,
        cols: op.head_dim,
        elem_bytes: op.act_elem,
    };
    // Per-token row stride bound through [`KvCacheLayout`] (paris
    // invariant `kv-cache-write-slot-offset-correctness`).
    let layout = op.kv_layout();
    let pos_off = layout.cos_sin_offset_expr("__decode_position");
    let slot_off = layout.slot_offset_expr(op.decode_slot_arg);

    let k_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, k_page);
    if in_k_carried.is_some() {
        prog.arrive(WarpRole::Loader, PageBarrier::Ready, k_page);
    } else {
        prog.load_async(k_id, op.k, region(op.k, op.m, kv_cols), kv_tile);
    }

    let c_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, c_page);
    if in_c_carried.is_some() {
        prog.arrive(WarpRole::Loader, PageBarrier::Ready, c_page);
    } else {
        prog.load_async_dyn(
            c_id,
            op.cos,
            region(op.cos, 1, op.head_dim),
            cs_tile,
            pos_off.clone(),
        );
    }

    let s_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, s_page);
    if in_s_carried.is_some() {
        prog.arrive(WarpRole::Loader, PageBarrier::Ready, s_page);
    } else {
        prog.load_async_dyn(
            s_id,
            op.sin,
            region(op.sin, 1, op.head_dim),
            cs_tile,
            pos_off,
        );
    }

    let v_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, v_page);
    if in_v_carried.is_some() {
        prog.arrive(WarpRole::Loader, PageBarrier::Ready, v_page);
    } else {
        prog.load_async(v_id, op.v, region(op.v, op.m, kv_cols), kv_tile);
    }

    let k_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, k_page);
    let c_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, c_page);
    let s_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, s_page);
    let v_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, v_page);
    let half = op.head_dim / 2;
    let total_pairs = (op.m as u64) * (op.num_kv_heads as u64) * (half as u64);
    prog.compute_calls(
        WarpRole::AllConsumers,
        crate::tk_codegen::rope_compute_calls::<crate::tk_codegen::NeoX>(
            k_id,
            c_id,
            s_id,
            op.head_dim,
            total_pairs,
        ),
    );
    let _ = half;
    let k_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, k_page);
    let c_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, c_page);
    let s_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, s_page);
    let v_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, v_page);

    let k_page = prog.wait(WarpRole::Storer, PageBarrier::Done, k_page);
    if !hints.output_internal {
        prog.store_async(k_id, op.out, region(op.out, op.m, kv_cols), kv_tile);
    }
    // Cache writes ALWAYS happen (the cache is the load-bearing
    // edge for downstream attention; arena drain is a separate
    // bookkeeping store, gated by output_internal like other ops).
    prog.store_async_dyn(
        k_id,
        op.k_cache,
        region(op.k_cache, op.m, kv_cols),
        kv_tile,
        slot_off.clone(),
    );
    let k_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, k_page);

    let c_page = prog.wait(WarpRole::Storer, PageBarrier::Done, c_page);
    let c_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, c_page);

    let s_page = prog.wait(WarpRole::Storer, PageBarrier::Done, s_page);
    let s_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, s_page);

    let v_page = prog.wait(WarpRole::Storer, PageBarrier::Done, v_page);
    prog.store_async_dyn(
        v_id,
        op.v_cache,
        region(op.v_cache, op.m, kv_cols),
        kv_tile,
        slot_off,
    );
    let v_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, v_page);

    let k_advanced = prog.complete_round(k_page);
    let c_advanced = prog.complete_round(c_page);
    let s_advanced = prog.complete_round(s_page);
    let v_advanced = prog.complete_round(v_page);

    let result = if hints.output_internal {
        let carried = pages.carry_forward(k_advanced);
        pages.release(c_advanced);
        pages.release(s_advanced);
        pages.release(v_advanced);
        RoutingResult {
            output_carried: Some(carried),
        }
    } else {
        pages.release(k_advanced);
        pages.release(c_advanced);
        pages.release(s_advanced);
        pages.release(v_advanced);
        RoutingResult::default()
    };

    // E.13: cross-op gmem ordering for the K/V cache writes is
    // emitted by the orchestrator via
    // `tk_gmem::emit_fence_after_op`. The `__syncthreads()` E.12.B
    // emitted here is removed.
    (result, k_cache_handle, v_cache_handle)
}

// ── GemmM1 — M=1 vec-mat decode GEMM ───────────────────────────────

/// Inputs to lower one M=1 vec-mat decode GEMM, mirroring
/// `LoweredOp::Gemm` for the decode path. Computes
/// `out[1, n] = x[1, k] @ w[n, k]^T`.
///
/// The decoder uses this for q/k/v/o, gate/up/down, and lm_head. Caller
/// picks `bn` such that one `[bn, k]` W tile fits in a single TK 2.0
/// page (`bn * k * act_elem <= PAGE_SIZE`); the orchestrator's
/// [`crate::tk_orchestrate::pick_bn`] derives this purely from `k` and
/// `PAGE_SIZE` — no per-model knowledge.
///
/// Page layout (3 slots): `x_page` loaded once before the loop and
/// held read-only by the consumer through every iteration; `w_page`
/// streams a fresh `[bn, k]` tile per iteration; `y_page` stages the
/// `[1, bn]` output values which the storer drains to `out` per
/// iteration. Output streaming is required because the full `[1, n]`
/// output (e.g. `n = 128256` for lm_head) does not fit in a single
/// page.
#[derive(Clone, Copy, Debug)]
pub struct GemmM1Op {
    /// `[1, k]` input vector.
    pub x: BufId,
    /// `[n, k]` weight (row-major: `w[row, col]` with row in
    /// `[0, n)`, col in `[0, k)`).
    pub w: BufId,
    /// `[1, n]` output (must NOT alias `x`; output streaming reuses a
    /// dedicated page slot).
    pub out: BufId,
    pub k: u32,
    pub n: u32,
    /// N-tile rows per W page load. Caller MUST satisfy
    /// `bn * k * act_elem <= PAGE_SIZE`.
    pub bn: u32,
    pub act_elem: u32,
}

/// Lower one M=1 GEMM into a `TkProgram` fragment.
///
/// Tape shape:
///   1. Load X once before the N-block loop (one round on `x_page`,
///      consumer waits Ready and holds the slot through the whole
///      loop).
///   2. `for (__n_i = 0; __n_i < ceil(n/bn); ++__n_i)` — each
///      iteration is one complete round on `w_page` and one on
///      `y_page`. Wait parities use `(__n_i & 1)`.
///      - Loader streams W[bn, k] from `w[n_i*bn .. (n_i+1)*bn, :]`.
///      - Consumer (one warp per output value: warp `c` computes
///        `y[c]` if `c < bn`, else idles) does a lane-parallel dot
///        product `acc[c] = sum_k x[k] * w[n_i*bn+c, k]`, butterfly-
///        reduces inside the warp, lane 0 writes `y_page[c]`.
///      - Storer drains `y_page` to `out[n_i*bn .. (n_i+1)*bn]`.
///   3. Close X's single round (consumer arrives Done, storer waits
///      Done + arrives Consumed without storing — X is read-only).
pub fn lower_gemm_m1<P: Phase>(op: GemmM1Op, pages: &mut PageAllocator, prog: &mut TkProgram) {
    debug_assert!(
        op.bn.saturating_mul(op.k).saturating_mul(op.act_elem) <= PAGE_SIZE,
        "lower_gemm_m1: W tile {}x{} ({} bytes) exceeds PAGE_SIZE={}",
        op.bn,
        op.k,
        op.bn * op.k * op.act_elem,
        PAGE_SIZE,
    );

    // Phase 8: drop the Y staging page. The per-iter Y store was
    // `bn * act_elem` = 8 bytes for Llama-1B (bn=4, bf16), which
    // violates `cp.async.bulk`'s 16-byte minimum and silently dropped
    // the output. Consumer body now writes directly to gmem
    // (`buf{op.out}[__n_i * bn + row]`) from lane 0 of the producing
    // warp; storer no longer touches Y. Eliminates one page allocation
    // and the per-iter Y wait/store/arrive triplet.
    let x_page = pages.alloc_at::<P>().expect("gemm_m1: x page");
    let w_page = pages.alloc_at::<P>().expect("gemm_m1: w page");
    let x_id = x_page.id();
    let w_id = w_page.id();

    let region = |buf, rows, cols| RegionRef::rows_cols(buf, rows, 0, cols);
    let x_tile = TileShape {
        rows: 1,
        cols: op.k,
        elem_bytes: op.act_elem,
    };
    let w_tile = TileShape {
        rows: op.bn,
        cols: op.k,
        elem_bytes: op.act_elem,
    };

    // ── Load X once before the N-block loop ──
    let x_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, x_page);
    prog.load_async(x_id, op.x, region(op.x, 1, op.k), x_tile);
    let x_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, x_page);

    // ── N-block loop ──
    let n_blocks = op.n.div_ceil(op.bn);
    let w_byte_step = (op.bn as u64) * (op.k as u64) * (op.act_elem as u64);
    let loop_var = "__n_i";

    // Capture the page's static phase at loop entry — fed to every
    // `wait_loop_parity` so iter-0 reads the page's actual mbarrier
    // parity (the page was just allocated at type P, which the
    // allocator records as `phase_bit[id] = P::VALUE`; the underlying
    // mbarrier parity matches when the prior user closed the slot
    // cleanly via `complete_round + release`).
    let start = P::VALUE;
    prog.for_loop(loop_var, LoopBound::Const(n_blocks), |body| {
        // Loader streams the next W tile.
        body.wait_loop_parity(WarpRole::Loader, PageBarrier::Consumed, w_id, loop_var, start);
        body.load_async_dyn(
            w_id,
            op.w,
            region(op.w, op.bn, op.k),
            w_tile,
            body.iter_offset(w_byte_step),
        );

        // Consumer: wait W ready, compute (writes directly to gmem
        // `buf{op.out}[__n_i * bn + row]`), signal W done.
        body.wait_loop_parity(WarpRole::AllConsumers, PageBarrier::Ready, w_id, loop_var, start);
        body.compute_calls(
            WarpRole::AllConsumers,
            crate::tk_codegen::gemm_m1_compute_calls(x_id, w_id, op.out.0, op.k, op.bn),
        );
        body.arrive_loop(WarpRole::AllConsumers, PageBarrier::Done, w_id);

        // Storer: free W slot (read-only — no actual store).
        body.wait_loop_parity(WarpRole::Storer, PageBarrier::Done, w_id, loop_var, start);
        body.arrive_loop(WarpRole::Storer, PageBarrier::Consumed, w_id);
    });

    // After the loop the W page slot has been ping-ponged exactly
    // `n_blocks` times. Each iteration flips each barrier
    // (consumed/ready/done) once, so the post-loop runtime parity
    // is `start_parity XOR (n_blocks & 1)`:
    //   - odd  n_blocks → parity flipped from start ⇒ advance type by 1
    //                     (== `complete_round`).
    //   - even n_blocks → parity back at start ⇒ DON'T advance the
    //                     type, just release at the unchanged phase.
    //                     `complete_round` here would record
    //                     `phase_bit = !P::VALUE` while the actual
    //                     mbarrier sits at `P::VALUE`, and the next
    //                     op picking up the slot would emit waits
    //                     with one parity while the barrier polls the
    //                     other → deadlock at the op boundary.
    //                     This was the op5/Add hang post-op4/Gemm.
    if n_blocks % 2 == 1 {
        let w_page = prog.complete_round(w_page);
        pages.release(w_page);
    } else {
        pages.release(w_page);
    }

    // Close X's single round. X was loaded once (loader Ready), held by
    // the consumer through every loop iteration, and is now released.
    // No store — X is read-only — but the storer still flips Consumed
    // so the slot is freed for the next op.
    let x_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, x_page);
    let x_page = prog.wait(WarpRole::Storer, PageBarrier::Done, x_page);
    let x_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, x_page);
    let x_page = prog.complete_round(x_page);
    pages.release(x_page);
}

// Phase 5 cutover: legacy `gemm_m1_compute_body` deleted. Canonical
// emit at `tk_codegen::tk20::gemm_m1_consumer_body`.

/// Phase 12: routing-aware variant of `lower_gemm_m1`. Same per-warp
/// K-axis matvec; routing controls the input/output paths:
///
/// - `hints.inputs[0] = Some(carried)` — X is carry-forward; loader
///   skips TMA load + emits bare `arrive(Ready[x])`.
/// - `hints.inputs[1] = Some(carried)` — W is carry-forward (rare;
///   weights are usually `InputRef::Ext` and gmem-loaded). Bound for
///   completeness.
/// - `hints.output_internal = false` — Phase 8 gmem-direct-write path
///   (`gemm_m1_consumer_body` writes `buf{op.out}[__n_i * bn + row]`
///   from the consumer body). Storer doesn't touch Y. Default.
/// - `hints.output_internal = true` — restore a smem Y page slot
///   (Phase-8 originally dropped it for the cp.async.bulk-min issue
///   with TMA store; THAT issue doesn't apply here because we don't
///   TMA-store at all in the carry-forward case — the Y page stays
///   in smem). Consumer body uses `gemm_m1_consumer_body_internal`
///   (writes `page_buf[y_id][__n_i * bn + row]`); storer skips TMA
///   store; lowering returns `RoutingResult { output_carried: Some(y) }`.
///
/// Default-hints path (no carry-forward inputs, output external) is
/// byte-identical to `lower_gemm_m1`.
pub fn lower_gemm_m1_routed<P: Phase>(
    op: GemmM1Op,
    hints: &RoutingHints,
    pages: &mut PageAllocator,
    prog: &mut TkProgram,
) -> RoutingResult {
    debug_assert!(
        op.bn.saturating_mul(op.k).saturating_mul(op.act_elem) <= PAGE_SIZE,
        "lower_gemm_m1_routed: W tile {}x{} ({} bytes) exceeds PAGE_SIZE={}",
        op.bn,
        op.k,
        op.bn * op.k * op.act_elem,
        PAGE_SIZE,
    );
    if hints.output_internal {
        // Y page sized as [1, n] bf16 = `n * act_elem` bytes; must
        // fit in a single PAGE_SIZE slot. Llama-1B: n=2048 (4 KB) for
        // q/k/v/o/up/gate; n=8192 (16 KB) for down_proj — both ≤
        // PAGE_SIZE=16384.
        debug_assert!(
            op.n.saturating_mul(op.act_elem) <= PAGE_SIZE,
            "lower_gemm_m1_routed: internal Y tile [1,{}] ({} bytes) exceeds PAGE_SIZE={}",
            op.n,
            op.n * op.act_elem,
            PAGE_SIZE,
        );
    }
    debug_assert!(
        hints.inputs.is_empty() || hints.inputs.len() == 2,
        "lower_gemm_m1_routed: hints.inputs must be empty or len 2"
    );
    let in_x_carried = hints.inputs.first().and_then(|x| x.as_ref());
    let in_w_carried = hints.inputs.get(1).and_then(|x| x.as_ref());

    let x_page: PageHandle<P> = match in_x_carried {
        Some(c) => pages.consume_carried::<P>(*c),
        None => pages.alloc_at::<P>().expect("gemm_m1: x page"),
    };
    let w_page: PageHandle<P> = match in_w_carried {
        Some(c) => pages.consume_carried::<P>(*c),
        None => pages.alloc_at::<P>().expect("gemm_m1: w page"),
    };
    let x_id = x_page.id();
    let w_id = w_page.id();

    // Y page only allocated when the output is internal (carry-
    // forward target). For external output, Phase 8's direct-gmem
    // write path is used (no Y page, no Y barriers).
    let y_page: Option<PageHandle<P>> = if hints.output_internal {
        Some(pages.alloc_at::<P>().expect("gemm_m1: y page (internal)"))
    } else {
        None
    };
    let y_id_opt: Option<u8> = y_page.as_ref().map(|p| p.id());

    let region = |buf, rows, cols| RegionRef::rows_cols(buf, rows, 0, cols);
    let x_tile = TileShape {
        rows: 1,
        cols: op.k,
        elem_bytes: op.act_elem,
    };
    let w_tile = TileShape {
        rows: op.bn,
        cols: op.k,
        elem_bytes: op.act_elem,
    };

    // ── Load X once before the N-block loop (or skip for carry-forward) ──
    let x_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, x_page);
    if in_x_carried.is_some() {
        prog.arrive(WarpRole::Loader, PageBarrier::Ready, x_page);
    } else {
        prog.load_async(x_id, op.x, region(op.x, 1, op.k), x_tile);
    }
    let x_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, x_page);

    // ── For internal output: open Y page round (no loader fill —
    //    consumer is the producer) BEFORE the loop. ──
    let y_page = y_page.map(|y| {
        let y = prog.wait(WarpRole::Loader, PageBarrier::Consumed, y);
        // No TMA load: Y is filled by the consumer body inside the
        // loop. Loader emits a bare arrive Ready so the consumer's
        // wait Ready passes through.
        prog.arrive(WarpRole::Loader, PageBarrier::Ready, y);
        prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, y)
    });

    // ── N-block loop ──
    let n_blocks = op.n.div_ceil(op.bn);
    let w_byte_step = (op.bn as u64) * (op.k as u64) * (op.act_elem as u64);
    let loop_var = "__n_i";
    let start = P::VALUE;
    prog.for_loop(loop_var, LoopBound::Const(n_blocks), |body| {
        // Loader streams the next W tile.
        body.wait_loop_parity(WarpRole::Loader, PageBarrier::Consumed, w_id, loop_var, start);
        body.load_async_dyn(
            w_id,
            op.w,
            region(op.w, op.bn, op.k),
            w_tile,
            body.iter_offset(w_byte_step),
        );

        // Consumer compute body — switch between gmem and smem
        // output based on routing hint.
        body.wait_loop_parity(WarpRole::AllConsumers, PageBarrier::Ready, w_id, loop_var, start);
        let compute_calls = if let Some(y_id) = y_id_opt {
            crate::tk_codegen::gemm_m1_compute_calls_internal(x_id, w_id, y_id, op.k, op.bn)
        } else {
            crate::tk_codegen::gemm_m1_compute_calls(x_id, w_id, op.out.0, op.k, op.bn)
        };
        body.compute_calls(WarpRole::AllConsumers, compute_calls);
        body.arrive_loop(WarpRole::AllConsumers, PageBarrier::Done, w_id);

        // Storer: free W slot.
        body.wait_loop_parity(WarpRole::Storer, PageBarrier::Done, w_id, loop_var, start);
        body.arrive_loop(WarpRole::Storer, PageBarrier::Consumed, w_id);
    });

    // Close W slot's parity (same logic as legacy lower_gemm_m1).
    if n_blocks % 2 == 1 {
        let w_page = prog.complete_round(w_page);
        pages.release(w_page);
    } else {
        pages.release(w_page);
    }

    // ── Close Y page round (internal output only). After the loop,
    //    the consumer has finished writing into the Y slot. Arrive
    //    Done from the consumer; storer arrives Consumed (no TMA
    //    store); carry_forward the slot for the downstream consumer. ──
    let y_carried = if let Some(y_page) = y_page {
        let y_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, y_page);
        let y_page = prog.wait(WarpRole::Storer, PageBarrier::Done, y_page);
        let y_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, y_page);
        let y_advanced = prog.complete_round(y_page);
        Some(pages.carry_forward(y_advanced))
    } else {
        None
    };

    // Close X's single round.
    let x_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, x_page);
    let x_page = prog.wait(WarpRole::Storer, PageBarrier::Done, x_page);
    let x_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, x_page);
    let x_page = prog.complete_round(x_page);
    pages.release(x_page);

    RoutingResult {
        output_carried: y_carried,
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tk_codegen::emit_body;
    use crate::tk_warp_ir::TkInstr;

    fn op() -> RmsNormOp {
        RmsNormOp {
            x: BufId(0),
            weight: BufId(1),
            out: BufId(2),
            hidden: 2048,
            m: 1,
            act_elem: 2,
            eps: 1e-5,
            init: true,
        }
    }

    /// The lowered IR has the exact 13-instruction handshake covering
    /// the X page AND the weight page in one round (no blank-filling at
    /// emit time, no explicit `arrive(Loader,Ready)`: `tma::load_async`
    /// signals page_ready itself).
    ///
    /// Layout:
    ///   loader   (4): wait Consumed[x], load[x],
    ///                 wait Consumed[w], load[w]
    ///   consumer (5): wait Ready[x], wait Ready[w], compute,
    ///                 arrive Done[x], arrive Done[w]
    ///   storer   (5): wait Done[x], store[x], arrive Consumed[x],
    ///                 wait Done[w], arrive Consumed[w]
    /// (Total 14; weight is read-only so the storer arrives Consumed
    /// without a TMA store.)
    #[test]
    fn rmsnorm_lowers_to_two_page_handshake() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_rmsnorm::<Phase0>(op(), &mut pages, &mut prog);

        assert_eq!(prog.instrs.len(), 14, "{prog:?}");

        // Phase parities the lowering picked: every wait within one
        // round reads `R & 1` (round 0 → all 0). Six waits total.
        let phases: Vec<String> = prog
            .instrs
            .iter()
            .filter_map(|i| match i {
                TkInstr::Wait { phase, .. } => Some(phase.cuda_expr()),
                _ => None,
            })
            .collect();
        assert_eq!(
            phases,
            vec!["0", "0", "0", "0", "0", "0"],
            "round 0: every wait reads 0"
        );
    }

    /// Phase 12: carry-forward keeps the slot in_use so a follow-up
    /// op can claim it via `consume_carried` without going through
    /// alloc_at's free-slot probe. The `phase_bit` is recorded at
    /// the carrier's parity so the next consumer's parity-typed
    /// dispatch (`lower_X::<Phase0>` vs `<Phase1>`) sees the right
    /// value.
    #[test]
    fn carry_forward_keeps_slot_in_use_and_records_parity() {
        let mut pages = PageAllocator::new();
        // Allocate at Phase0, advance through one round to Phase1,
        // carry-forward instead of releasing.
        let p = pages.alloc_at::<Phase0>().expect("alloc");
        assert!(pages.in_use[p.id() as usize]);
        let p_advanced = p.advance(); // Phase0 → Phase1 typed advance
        let carried = pages.carry_forward(p_advanced);
        assert!(
            pages.in_use[carried.id as usize],
            "carry_forward keeps slot in_use"
        );
        assert_eq!(carried.phase, 1, "carried at Phase1");
        assert_eq!(
            pages.phase_bit[carried.id as usize], 1,
            "phase_bit recorded at carried parity"
        );
        // Consume — get a typed handle back at Phase1.
        let p2: PageHandle<crate::tk_warp_ir::Phase1> = pages.consume_carried(carried);
        assert_eq!(p2.id(), carried.id);
        assert_eq!(p2.phase(), 1);
        // Slot is still in_use after consume_carried (the consuming
        // op's lowering is now responsible for end-of-round release).
        assert!(pages.in_use[carried.id as usize]);
        // Standard release at end of consumer's round.
        pages.release(p2);
        assert!(!pages.in_use[carried.id as usize]);
    }

    /// Carry-forward + consume_carried preserves the slot id (no
    /// re-probe of the free pool happens between producer and
    /// consumer). Two carry-forwards in flight occupy two distinct
    /// slot ids.
    #[test]
    fn carry_forward_pins_slot_id() {
        let mut pages = PageAllocator::new();
        let p_a = pages.alloc_at::<Phase0>().expect("alloc a");
        let p_b = pages.alloc_at::<Phase0>().expect("alloc b");
        let id_a = p_a.id();
        let id_b = p_b.id();
        assert_ne!(id_a, id_b);
        let carried_a = pages.carry_forward(p_a.advance());
        let carried_b = pages.carry_forward(p_b.advance());
        assert_eq!(carried_a.id, id_a);
        assert_eq!(carried_b.id, id_b);
        // A fresh alloc_at probe must NOT return either pinned id.
        let p_c = pages.alloc_at::<Phase0>().expect("alloc c");
        assert_ne!(p_c.id(), id_a);
        assert_ne!(p_c.id(), id_b);
        // Cleanup.
        let _: PageHandle<crate::tk_warp_ir::Phase1> = pages.consume_carried(carried_a);
        let _: PageHandle<crate::tk_warp_ir::Phase1> = pages.consume_carried(carried_b);
    }

    /// The page is released back to the allocator at the right parity:
    /// the storer's last arrive flipped it once more, so each slot's
    /// next round starts at Phase1.
    #[test]
    fn page_released_at_correct_parity() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_rmsnorm::<Phase0>(op(), &mut pages, &mut prog);
        // Both x (slot 0) and weight (slot 1) live on Phase1 after one round.
        assert_eq!(pages.phase_bit[0], 1, "x slot on Phase1 after one round");
        assert_eq!(pages.phase_bit[1], 1, "weight slot on Phase1 after one round");
        assert!(!pages.in_use[0]);
        assert!(!pages.in_use[1]);
    }

    /// Codegen on the lowered program produces source containing the
    /// expected three role-gated arms and the expected three TK 2.0
    /// barrier names.
    #[test]
    fn rmsnorm_codegen_walks_match_arms() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_rmsnorm::<Phase0>(op(), &mut pages, &mut prog);
        let src = emit_body(&prog);

        // Three role-routed arms.
        assert!(
            src.contains("if (__role == ROLE_LOADER)"),
            "loader arm\n{src}"
        );
        assert!(
            src.contains("if (__role == ROLE_CONSUMER)"),
            "consumer arm\n{src}"
        );
        assert!(
            src.contains("if (__role == ROLE_STORER)"),
            "storer arm\n{src}"
        );

        // Three barriers, each on the page slot the allocator picked (0).
        assert!(src.contains("page_consumed[0]"), "{src}");
        assert!(src.contains("page_ready[0]"), "{src}");
        assert!(src.contains("page_done[0]"), "{src}");

        // Phase parities match TK 2.0 round semantics: every wait in
        // one round reads `R & 1` (round 0 → all 0).
        assert!(src.contains("page_consumed[0], 0"), "{src}");
        assert!(src.contains("page_ready[0], 0"), "{src}");
        assert!(src.contains("page_done[0], 0"), "{src}");

        // Body fragment pasted verbatim, including the weight multiply.
        assert!(src.contains("rsqrtf"), "compute body present\n{src}");
        assert!(
            src.contains("__v * __scale * __g"),
            "weight gain applied to normalised value\n{src}"
        );
    }

    /// Two adjacent RmsNorm ops use distinct page slots — the
    /// allocator does not stomp on an in-use slot.
    #[test]
    fn two_rmsnorms_use_distinct_pages() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_rmsnorm::<Phase0>(op(), &mut pages, &mut prog);
        // Pretend the first op's slot is still owned (skip the release).
        // Re-simulate by manually marking slot 0 as in_use:
        let mut pages2 = PageAllocator::new();
        pages2.in_use[0] = true;
        let page = pages2.alloc_p0().unwrap();
        assert_eq!(page.id(), 1, "second alloc takes the next free slot");
    }

    fn attn_op() -> AttnDecodeOp {
        AttnDecodeOp {
            q: BufId(10),
            k_cache: BufId(11),
            v_cache: BufId(12),
            out: BufId(13),
            head_dim: 128,
            // Test shape: 8 q-heads × 8 kv-heads (q_per_kv=1) — must
            // match `NUM_CONSUMER_WARPS` for the per-warp kv-head
            // sharding the lowering does. Real Llama-3.2-1B is
            // 32 q × 8 kv (q_per_kv=4), exercised end-to-end via
            // `fixtures::one_layer_input`.
            num_q_heads: 8,
            num_kv_heads: 8,
            act_elem: 2,
            softmax_scale: 0.088388_35,
            num_kv_pages_arg: crate::tk_warp_ir::NumKvPagesSym,
            unique_id: 0,
        }
    }

    /// AttnDecode lowers to: Q-handshake (outside) + ForLoop + final
    /// O-handshake. The structure has exactly one ForLoop and the
    /// loop body has 12 instrs (3 K-handshake steps × 3 roles + 3 V-
    /// handshake steps × 3 roles, with K-load and V-load inside).
    #[test]
    fn attn_decode_lowers_to_typed_outer_plus_kv_loop() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        let op = attn_op();
        let k = crate::tk_gmem::GmemHandle::<crate::tk_gmem::KCache>::new_initial(op.k_cache);
        let v = crate::tk_gmem::GmemHandle::<crate::tk_gmem::VCache>::new_initial(op.v_cache);
        let kf = crate::tk_gmem::emit_fence_after_op(&mut prog, k);
        let vf = crate::tk_gmem::emit_fence_after_op(&mut prog, v);
        lower_attn_decode::<Phase0>(op, kf, vf, &mut pages, &mut prog);

        // Count ForLoops and their body lengths.
        let loops: Vec<&Vec<TkInstr>> = prog
            .instrs
            .iter()
            .filter_map(|i| match i {
                TkInstr::ForLoop { body, .. } => Some(body),
                _ => None,
            })
            .collect();
        assert_eq!(loops.len(), 1, "exactly one KV sweep loop");

        // K-handshake: wait+load (loader), wait+compute+arrive (consumer),
        //              wait+arrive (storer) → 7 instrs.
        // V-handshake: same → 7 instrs.
        // Total per iteration: 14. (No `arrive(Ready)` — load signals it.)
        assert_eq!(
            loops[0].len(),
            14,
            "loop body has K + V handshakes, 14 instrs"
        );
    }

    /// All static (outside-loop) waits use compile-time parities; all
    /// in-loop waits use runtime `(__kv_i & 1)` parities.
    #[test]
    fn attn_decode_phase_kinds_match_loop_structure() {
        use crate::tk_warp_ir::WaitPhase;
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        let op = attn_op();
        let k = crate::tk_gmem::GmemHandle::<crate::tk_gmem::KCache>::new_initial(op.k_cache);
        let v = crate::tk_gmem::GmemHandle::<crate::tk_gmem::VCache>::new_initial(op.v_cache);
        let kf = crate::tk_gmem::emit_fence_after_op(&mut prog, k);
        let vf = crate::tk_gmem::emit_fence_after_op(&mut prog, v);
        lower_attn_decode::<Phase0>(op, kf, vf, &mut pages, &mut prog);

        let mut outer_phases = Vec::<WaitPhase>::new();
        let mut inner_phases = Vec::<WaitPhase>::new();
        for instr in &prog.instrs {
            match instr {
                TkInstr::Wait { phase, .. } => outer_phases.push(phase.clone()),
                TkInstr::ForLoop { body, .. } => {
                    for inner in body {
                        if let TkInstr::Wait { phase, .. } = inner {
                            inner_phases.push(phase.clone());
                        }
                    }
                }
                _ => {}
            }
        }

        for p in &outer_phases {
            assert!(
                matches!(p, WaitPhase::Static(_)),
                "outer waits are typed-phase, got {p:?}"
            );
        }
        for p in &inner_phases {
            assert!(
                matches!(p, WaitPhase::Runtime(_)),
                "in-loop waits are runtime parities, got {p:?}"
            );
        }
        assert!(!outer_phases.is_empty() && !inner_phases.is_empty());
    }

    /// Codegen on AttnDecode emits a `for (uint __kv_i = 0; __kv_i <
    /// __num_kv_pages; ...)` and a `(__kv_i & 1)` parity inside.
    #[test]
    fn attn_decode_codegen_emits_for_and_runtime_parity() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        let op = attn_op();
        let k = crate::tk_gmem::GmemHandle::<crate::tk_gmem::KCache>::new_initial(op.k_cache);
        let v = crate::tk_gmem::GmemHandle::<crate::tk_gmem::VCache>::new_initial(op.v_cache);
        let kf = crate::tk_gmem::emit_fence_after_op(&mut prog, k);
        let vf = crate::tk_gmem::emit_fence_after_op(&mut prog, v);
        lower_attn_decode::<Phase0>(op, kf, vf, &mut pages, &mut prog);
        let src = emit_body(&prog);

        assert!(
            src.contains("for (uint __kv_i = 0; __kv_i < __num_kv_pages; ++__kv_i)"),
            "KV sweep loop\n{src}"
        );
        assert!(
            src.contains("(__kv_i & 1)"),
            "runtime KV parity\n{src}"
        );
        assert!(src.contains("if (__role == ROLE_LOADER)"), "{src}");
        assert!(src.contains("if (__role == ROLE_CONSUMER)"), "{src}");
        assert!(src.contains("if (__role == ROLE_STORER)"), "{src}");
        // Body fragments pasted verbatim.
        assert!(src.contains("__o_accum"), "softmax accum present\n{src}");
        assert!(src.contains("expf"), "online softmax present\n{src}");
    }

    /// Three distinct page slots are claimed: Q, K, V (no stomping).
    #[test]
    fn attn_decode_uses_three_distinct_pages() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        let op = attn_op();
        let k = crate::tk_gmem::GmemHandle::<crate::tk_gmem::KCache>::new_initial(op.k_cache);
        let v = crate::tk_gmem::GmemHandle::<crate::tk_gmem::VCache>::new_initial(op.v_cache);
        let kf = crate::tk_gmem::emit_fence_after_op(&mut prog, k);
        let vf = crate::tk_gmem::emit_fence_after_op(&mut prog, v);
        lower_attn_decode::<Phase0>(op, kf, vf, &mut pages, &mut prog);

        let mut load_pages = Vec::<u8>::new();
        for instr in &prog.instrs {
            if let TkInstr::LoadAsync { page_id, .. } = instr {
                load_pages.push(*page_id);
            }
            if let TkInstr::ForLoop { body, .. } = instr {
                for inner in body {
                    if let TkInstr::LoadAsync { page_id, .. } = inner {
                        load_pages.push(*page_id);
                    }
                }
            }
        }
        let mut uniq = load_pages.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), 3, "Q + K + V → three distinct page slots; loads: {load_pages:?}");
    }

    fn add_op() -> AddOp {
        AddOp {
            a: BufId(20),
            b: BufId(21),
            out: BufId(22),
            hidden: 2048,
            m: 1,
            act_elem: 2,
        }
    }

    /// Residual Add lowers to two-page handshake: A is loaded + drained
    /// (full round), B is loaded + read + freed (no TMA store on B).
    #[test]
    fn add_lowers_to_two_page_handshake() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_residual_add::<Phase0>(add_op(), &mut pages, &mut prog);

        let mut load_pages = Vec::<u8>::new();
        let mut store_pages = Vec::<u8>::new();
        for instr in &prog.instrs {
            match instr {
                TkInstr::LoadAsync { page_id, .. } => load_pages.push(*page_id),
                TkInstr::StoreAsync { page_id, .. } => store_pages.push(*page_id),
                _ => {}
            }
        }
        // Two distinct loads (A + B), one store (out via A's slot).
        assert_eq!(load_pages.len(), 2, "two loads (A + B)");
        let mut uniq_loads = load_pages.clone();
        uniq_loads.sort();
        uniq_loads.dedup();
        assert_eq!(uniq_loads.len(), 2, "A and B on distinct pages");
        assert_eq!(store_pages.len(), 1, "one store (A → out)");
        assert_eq!(store_pages[0], load_pages[0], "store drains A's page");
    }

    /// Codegen on Add produces the expected role-routed arms for both
    /// pages plus an A+B compute body.
    #[test]
    fn add_codegen_emits_two_page_handshake() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_residual_add::<Phase0>(add_op(), &mut pages, &mut prog);
        let src = emit_body(&prog);

        // Both pages get loader+consumer+storer arms.
        assert!(src.contains("page_consumed[0]"), "{src}");
        assert!(src.contains("page_consumed[1]"), "{src}");
        assert!(src.contains("page_ready[0]"), "{src}");
        assert!(src.contains("page_ready[1]"), "{src}");
        assert!(src.contains("page_done[0]"), "{src}");
        assert!(src.contains("page_done[1]"), "{src}");
        // Compute body is in the consumer arm and references both pages.
        assert!(src.contains("__a_smem"), "{src}");
        assert!(src.contains("__b_smem"), "{src}");
        // Round 0: every wait reads parity 0.
        assert!(src.contains("page_consumed[0], 0"), "{src}");
        assert!(src.contains("page_consumed[1], 0"), "{src}");
        assert!(src.contains("page_ready[0], 0"), "{src}");
        assert!(src.contains("page_ready[1], 0"), "{src}");
        assert!(src.contains("page_done[0], 0"), "{src}");
        assert!(src.contains("page_done[1], 0"), "{src}");
    }

    /// Both pages release at parity 1 after one full round (storer's
    /// arrive on Consumed flips them once each).
    fn silu_mul_op_() -> SiluMulOp {
        SiluMulOp {
            gate: BufId(30),
            up: BufId(31),
            out: BufId(32),
            intermediate: 8192,
            m: 1,
            act_elem: 2,
        }
    }

    #[test]
    fn silu_mul_lowers_to_two_page_handshake() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_silu_mul::<Phase0>(silu_mul_op_(), &mut pages, &mut prog);

        let mut load_pages = Vec::<u8>::new();
        let mut store_pages = Vec::<u8>::new();
        for instr in &prog.instrs {
            match instr {
                TkInstr::LoadAsync { page_id, .. } => load_pages.push(*page_id),
                TkInstr::StoreAsync { page_id, .. } => store_pages.push(*page_id),
                _ => {}
            }
        }
        assert_eq!(load_pages.len(), 2, "two loads (gate + up)");
        let mut uniq = load_pages.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), 2);
        assert_eq!(store_pages.len(), 1, "one store (gate slot → out)");
    }

    #[test]
    fn silu_mul_codegen_emits_silu_and_mul() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_silu_mul::<Phase0>(silu_mul_op_(), &mut pages, &mut prog);
        let src = emit_body(&prog);
        // SiLU formula: g / (1 + exp(-g))
        assert!(src.contains("__silu_g"), "{src}");
        assert!(src.contains("expf(-"), "{src}");
        // Mul: silu(g) * u
        assert!(src.contains("__silu_g * __u"), "{src}");
    }

    fn rope_op() -> RopeRotateOp {
        RopeRotateOp {
            x: BufId(40),
            cos: BufId(41),
            sin: BufId(42),
            out: BufId(40), // alias x
            head_dim: 64,
            num_heads: 32, // Q heads
            m: 1,
            act_elem: 2,
        }
    }

    #[test]
    fn rope_lowers_to_three_page_handshake() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_rope_rotate::<Phase0>(rope_op(), &mut pages, &mut prog);

        let mut load_pages = Vec::<u8>::new();
        let mut store_pages = Vec::<u8>::new();
        for instr in &prog.instrs {
            match instr {
                TkInstr::LoadAsync { page_id, .. } => load_pages.push(*page_id),
                TkInstr::StoreAsync { page_id, .. } => store_pages.push(*page_id),
                _ => {}
            }
        }
        assert_eq!(load_pages.len(), 3, "x + cos + sin");
        let mut uniq = load_pages.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), 3, "three distinct pages");
        assert_eq!(store_pages.len(), 1, "x is the only store");
    }

    #[test]
    fn rope_codegen_emits_neox_rotation() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_rope_rotate::<Phase0>(rope_op(), &mut pages, &mut prog);
        let src = emit_body(&prog);
        assert!(src.contains("__cos_smem"), "{src}");
        assert!(src.contains("__sin_smem"), "{src}");
        // NeoX: lo = lo*c - hi*s; hi = lo*s + hi*c.
        assert!(src.contains("__x_lo * __c - __x_hi * __s"), "{src}");
        assert!(src.contains("__x_lo * __s + __x_hi * __c"), "{src}");
        // Cos/sin TMA loads add the per-decode-token row offset
        // `__decode_position * row_bytes` (rope_op is head_dim=64,
        // act_elem=2 → 128 B/row). Without this the kernel would read
        // position 0's cos/sin every decode step.
        assert!(
            src.contains("__decode_position * 128u"),
            "rope cos/sin TMA must add runtime row offset; got:\n{src}"
        );
    }

    #[test]
    fn add_releases_both_pages_at_phase1() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_residual_add::<Phase0>(add_op(), &mut pages, &mut prog);
        assert_eq!(pages.phase_bit[0], 1, "A page on Phase1 after one round");
        assert_eq!(pages.phase_bit[1], 1, "B page on Phase1 after one round");
        assert!(!pages.in_use[0]);
        assert!(!pages.in_use[1]);
    }

    fn gemm_qkv_op() -> GemmM1Op {
        // Llama-3.2-1B q_proj shape: x[1, 2048] @ W[2048, 2048]^T → out[1, 2048].
        // BN=4 keeps W tile = [4, 2048] bf16 = 16384 bytes (one full page).
        GemmM1Op {
            x: BufId(20),
            w: BufId(21),
            out: BufId(22),
            k: 2048,
            n: 2048,
            bn: 4,
            act_elem: 2,
        }
    }

    fn gemm_down_op() -> GemmM1Op {
        // Llama-3.2-1B down_proj shape: x[1, 8192] @ W[2048, 8192]^T → out[1, 2048].
        // BN=1 keeps W tile = [1, 8192] bf16 = 16384 bytes (one full page).
        GemmM1Op {
            x: BufId(30),
            w: BufId(31),
            out: BufId(32),
            k: 8192,
            n: 2048,
            bn: 1,
            act_elem: 2,
        }
    }

    /// GemmM1 lowers to: X-load (outside) + ForLoop over n_blocks +
    /// X-close (outside). Phase 8: Y staging page dropped — consumer
    /// writes directly to `buf{op.out}` from gmem; storer no longer
    /// touches Y.
    #[test]
    fn gemm_m1_lowers_to_xload_plus_loop_plus_xclose() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_gemm_m1::<Phase0>(gemm_qkv_op(), &mut pages, &mut prog);

        let loops: Vec<&Vec<TkInstr>> = prog
            .instrs
            .iter()
            .filter_map(|i| match i {
                TkInstr::ForLoop { body, .. } => Some(body),
                _ => None,
            })
            .collect();
        assert_eq!(loops.len(), 1, "one N-block loop");
        // Loop body: 7 instrs (Phase 8 — Y staging removed).
        //  loader  (2): wait Consumed[w], LoadAsync[w]
        //  consumer(3): wait Ready[w], Compute (direct gmem write),
        //               arrive Done[w]
        //  storer  (2): wait Done[w], arrive Consumed[w]   (W is read-only)
        assert_eq!(loops[0].len(), 7, "loop body has 7 instrs (see comment)");
    }

    /// GemmM1 uses two distinct page slots (x, w) — none alias.
    /// Phase 8: Y page dropped.
    #[test]
    fn gemm_m1_uses_two_distinct_pages() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_gemm_m1::<Phase0>(gemm_qkv_op(), &mut pages, &mut prog);

        // Find every page id referenced in any TMA load/store.
        let mut ids = Vec::<u8>::new();
        for instr in &prog.instrs {
            match instr {
                TkInstr::LoadAsync { page_id, .. } | TkInstr::StoreAsync { page_id, .. } => {
                    ids.push(*page_id);
                }
                TkInstr::ForLoop { body, .. } => {
                    for inner in body {
                        match inner {
                            TkInstr::LoadAsync { page_id, .. }
                            | TkInstr::StoreAsync { page_id, .. } => ids.push(*page_id),
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 2, "two distinct page slots: {ids:?}");
    }

    /// Codegen on GemmM1 emits the runtime parity `(__n_i & 1)` inside
    /// the loop and the runtime W byte-offset `(__n_i * <step>u)`.
    /// The Y store is gone — consumer body writes
    /// `__y_gmem[__n_i * __bn + __row]` directly.
    #[test]
    fn gemm_m1_codegen_emits_for_loop_runtime_offsets() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        let op = gemm_qkv_op();
        lower_gemm_m1::<Phase0>(op, &mut pages, &mut prog);
        let src = emit_body(&prog);

        let n_blocks = op.n.div_ceil(op.bn);
        assert!(
            src.contains(&format!("for (uint __n_i = 0; __n_i < {n_blocks};")),
            "compile-time N-block loop bound\n{src}"
        );
        assert!(
            src.contains("(__n_i & 1)"),
            "runtime parity inside the loop\n{src}"
        );
        // W byte step = bn * k * act_elem = 4 * 2048 * 2 = 16384.
        assert!(
            src.contains("(__n_i * 16384u)"),
            "W TMA load uses runtime byte-offset\n{src}"
        );
        // Phase 8: no Y TMA store.
        assert!(
            !src.contains("(__n_i * 8u)"),
            "Y TMA store removed (direct gmem write)\n{src}"
        );
        assert!(
            src.contains("__y_gmem[__n_i * __bn + __row]"),
            "consumer writes directly to gmem\n{src}"
        );
        // Phase 8: TK 2.0 register-vector K-reduce replaces shfl butterfly.
        assert!(
            src.contains("kittens::warp::sum(__x_rv_fl)"),
            "TK 2.0 register-vector sum reduction in compute body\n{src}"
        );
        assert!(
            !src.contains("__shfl_xor_sync"),
            "legacy shfl K-reduce removed\n{src}"
        );
    }

    /// down_proj shape (K=8192, BN=1) compiles cleanly and emits the
    /// expected page tile sizes.
    #[test]
    fn gemm_m1_down_proj_shape_lowers() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_gemm_m1::<Phase0>(gemm_down_op(), &mut pages, &mut prog);
        let src = emit_body(&prog);
        // W byte step = 1 * 8192 * 2 = 16384.
        assert!(src.contains("(__n_i * 16384u)"), "{src}");
        // Phase 8: Y store removed.
        assert!(
            !src.contains("(__n_i * 2u)"),
            "Y TMA store removed (direct gmem write)\n{src}"
        );
        assert!(src.contains("__y_gmem[__n_i * __bn + __row]"), "{src}");
    }

    fn add_op_for_routing_test() -> AddOp {
        AddOp {
            a: BufId(20),
            b: BufId(21),
            out: BufId(22),
            hidden: 2048,
            m: 1,
            act_elem: 2,
        }
    }

    /// Default hints (no carry-forward, no internal output) emit the
    /// SAME instruction sequence as the legacy `lower_residual_add`.
    /// Both load A from gmem, both load B from gmem, storer drains A
    /// to gmem.
    #[test]
    fn lower_residual_add_routed_default_hints_match_legacy_emit() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        let _result = lower_residual_add_routed::<Phase0>(
            add_op_for_routing_test(),
            &RoutingHints::default(),
            &mut pages,
            &mut prog,
        );
        let src_routed = emit_body(&prog);

        let mut pages2 = PageAllocator::new();
        let mut prog2 = TkProgram::new();
        lower_residual_add::<Phase0>(add_op_for_routing_test(), &mut pages2, &mut prog2);
        let src_legacy = emit_body(&prog2);

        assert_eq!(
            src_routed, src_legacy,
            "default hints must emit byte-identical CUDA to legacy"
        );
    }

    /// Carry-forward A: skip A's TMA load (loader emits bare arrive
    /// Ready) but keep B's TMA load. Storer's TMA store-A stays
    /// (output is external in this case). The carry-forward source
    /// is a synthetic upstream slot we set up by allocating all
    /// slots at Phase0, advancing them all to Phase1, releasing
    /// every slot except id=5 (which we carry-forward).
    #[test]
    fn lower_residual_add_routed_carry_forward_input_a_skips_load() {
        let mut pages = PageAllocator::new();
        // Allocate every slot at Phase0.
        let mut held: Vec<_> = (0..crate::tk_warp_ir::NUM_PAGES)
            .map(|_| pages.alloc_at::<Phase0>().expect("alloc"))
            .collect();
        // Pull out id=5 to carry-forward; advance + release the rest
        // at Phase1 so they're free for the lowering's alloc_at::<Phase1>().
        held.sort_by_key(|p| p.id());
        let target = held.remove(5);
        assert_eq!(target.id(), 5);
        for p in held {
            pages.release(p.advance());
        }
        let carried = pages.carry_forward(target.advance());

        let mut prog = TkProgram::new();
        let result = lower_residual_add_routed::<Phase1>(
            add_op_for_routing_test(),
            &RoutingHints {
                inputs: vec![Some(carried), None],
                output_internal: false,
            },
            &mut pages,
            &mut prog,
        );
        // No output carry-forward.
        assert!(result.output_carried.is_none());

        let src = emit_body(&prog);
        // The bare arrive Ready[5] from the loader is present (this
        // replaces the TMA load).
        assert!(
            src.contains("kittens::group<1>::arrive(page_ready[5])"),
            "loader emits bare arrive Ready[5] for carry-forward A\n{src}"
        );
        // A's TMA load_async is GONE — page 5 (the carried slot) is
        // never touched by tma::load_async.
        assert!(
            !src.contains("tma::load_async(page_ready[5]"),
            "no TMA load on carry-forward slot 5\n{src}"
        );
        // B's load_async is still there. (B's slot id is dynamic;
        // just check that some load_async exists.)
        assert!(
            src.contains("tma::load_async("),
            "B's TMA load is still present\n{src}"
        );
        // Storer's TMA store-A is still there (output_internal=false).
        assert!(
            src.contains("tma::store_async"),
            "storer drains A to gmem when output_internal=false\n{src}"
        );
    }

    /// `output_internal = true`: storer skips TMA store and the
    /// lowering returns a `CarriedHandle` for the orchestrator to
    /// thread to the consuming op.
    #[test]
    fn lower_residual_add_routed_internal_output_skips_store_and_carries() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        let result = lower_residual_add_routed::<Phase0>(
            add_op_for_routing_test(),
            &RoutingHints {
                inputs: vec![None, None],
                output_internal: true,
            },
            &mut pages,
            &mut prog,
        );
        let carried = result.output_carried.expect("output_carried set");
        // Slot stays in_use after the lowering returns.
        assert!(
            pages.in_use[carried.id as usize],
            "carry-forward slot stays in_use"
        );

        let src = emit_body(&prog);
        // Loaders DO load (no input carry-forward in this test).
        assert!(src.contains("tma::load_async("));
        // Storer does NOT TMA-store A.
        assert!(
            !src.contains("tma::store_async"),
            "storer skips TMA store when output_internal=true\n{src}"
        );
    }

    /// Default-hints byte-identity for `lower_rmsnorm_routed`.
    #[test]
    fn lower_rmsnorm_routed_default_hints_match_legacy_emit() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        let _ = lower_rmsnorm_routed::<Phase0>(
            op(),
            &RoutingHints::default(),
            &mut pages,
            &mut prog,
        );
        let src_routed = emit_body(&prog);

        let mut pages2 = PageAllocator::new();
        let mut prog2 = TkProgram::new();
        lower_rmsnorm::<Phase0>(op(), &mut pages2, &mut prog2);
        let src_legacy = emit_body(&prog2);

        assert_eq!(src_routed, src_legacy);
    }

    /// `lower_rmsnorm_routed` with `output_internal=true` skips the
    /// storer's TMA store and returns a CarriedHandle.
    #[test]
    fn lower_rmsnorm_routed_internal_output_skips_store() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        let result = lower_rmsnorm_routed::<Phase0>(
            op(),
            &RoutingHints {
                inputs: vec![None, None],
                output_internal: true,
            },
            &mut pages,
            &mut prog,
        );
        assert!(result.output_carried.is_some());
        let src = emit_body(&prog);
        assert!(!src.contains("tma::store_async"));
    }

    fn silu_mul_op_for_routing_test() -> SiluMulOp {
        SiluMulOp {
            gate: BufId(30),
            up: BufId(31),
            out: BufId(32),
            intermediate: 8192,
            m: 1,
            act_elem: 2,
        }
    }

    /// Default-hints byte-identity for `lower_silu_mul_routed`.
    #[test]
    fn lower_silu_mul_routed_default_hints_match_legacy_emit() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        let _ = lower_silu_mul_routed::<Phase0>(
            silu_mul_op_for_routing_test(),
            &RoutingHints::default(),
            &mut pages,
            &mut prog,
        );
        let src_routed = emit_body(&prog);

        let mut pages2 = PageAllocator::new();
        let mut prog2 = TkProgram::new();
        lower_silu_mul::<Phase0>(silu_mul_op_for_routing_test(), &mut pages2, &mut prog2);
        let src_legacy = emit_body(&prog2);

        assert_eq!(src_routed, src_legacy);
    }

    /// `lower_silu_mul_routed` with `output_internal=true`: storer skips drain.
    #[test]
    fn lower_silu_mul_routed_internal_output_skips_store() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        let result = lower_silu_mul_routed::<Phase0>(
            silu_mul_op_for_routing_test(),
            &RoutingHints {
                inputs: vec![None, None],
                output_internal: true,
            },
            &mut pages,
            &mut prog,
        );
        assert!(result.output_carried.is_some());
        let src = emit_body(&prog);
        assert!(!src.contains("tma::store_async"));
    }

    fn rope_op_for_routing_test() -> RopeRotateOp {
        RopeRotateOp {
            x: BufId(40),
            cos: BufId(41),
            sin: BufId(42),
            out: BufId(43),
            head_dim: 64,
            num_heads: 32,
            m: 1,
            act_elem: 2,
        }
    }

    /// Default-hints byte-identity for `lower_rope_rotate_routed`.
    #[test]
    fn lower_rope_rotate_routed_default_hints_match_legacy_emit() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        let _ = lower_rope_rotate_routed::<Phase0>(
            rope_op_for_routing_test(),
            &RoutingHints::default(),
            &mut pages,
            &mut prog,
        );
        let src_routed = emit_body(&prog);

        let mut pages2 = PageAllocator::new();
        let mut prog2 = TkProgram::new();
        lower_rope_rotate::<Phase0>(rope_op_for_routing_test(), &mut pages2, &mut prog2);
        let src_legacy = emit_body(&prog2);

        assert_eq!(src_routed, src_legacy);
    }

    /// `lower_rope_rotate_routed` with `output_internal=true`: storer
    /// skips drain.
    #[test]
    fn lower_rope_rotate_routed_internal_output_skips_store() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        let result = lower_rope_rotate_routed::<Phase0>(
            rope_op_for_routing_test(),
            &RoutingHints {
                inputs: vec![None, None, None],
                output_internal: true,
            },
            &mut pages,
            &mut prog,
        );
        assert!(result.output_carried.is_some());
        let src = emit_body(&prog);
        assert!(!src.contains("tma::store_async"));
    }
}
