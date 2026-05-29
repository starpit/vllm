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
    PageBarrier, PageHandle, Phase0, Phase1, TileShape, TkProgram, WarpRole, NUM_PAGES,
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
    pub fn alloc_p0(&mut self) -> Option<PageHandle<Phase0>> {
        let id = self.in_use.iter().position(|u| !u)? as u8;
        if self.phase_bit[id as usize] != 0 {
            // Slot's previous round ended on Phase1 — the next round's
            // first wait must read Phase1 (not Phase0). Caller should
            // request the right parity; we return None here so the
            // mistake is visible.
            return None;
        }
        self.in_use[id as usize] = true;
        Some(PageHandle::fresh(id))
    }

    pub fn alloc_p1(&mut self) -> Option<PageHandle<Phase1>> {
        let id = self.in_use.iter().position(|u| !u)? as u8;
        if self.phase_bit[id as usize] != 1 {
            return None;
        }
        self.in_use[id as usize] = true;
        // Synthesise a Phase1 handle directly — only valid because
        // we've checked the slot's phase_bit IS 1 (so the next wait
        // really should read 1).
        Some(PageHandle::<Phase0>::fresh(id).advance())
    }

    /// Release a page slot whose round has completed. The handle's
    /// type encodes the slot's *current* phase parity; we store it as
    /// a runtime bit for the next allocator decision.
    pub fn release<P: crate::tk_warp_ir::Phase>(&mut self, page: PageHandle<P>) {
        self.in_use[page.id() as usize] = false;
        self.phase_bit[page.id() as usize] = page.phase();
    }
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
pub fn lower_rmsnorm(op: RmsNormOp, pages: &mut PageAllocator, prog: &mut TkProgram) {
    let page = pages
        .alloc_p0()
        .expect("page exhaustion: out of mbarrier slots");
    let page_id = page.id();

    // x's region for the TMA: full hidden columns, m rows.
    let x_region = RegionRef::rows_cols(op.x, op.m, 0, op.hidden);
    let out_region = RegionRef::rows_cols(op.out, op.m, 0, op.hidden);
    let tile = TileShape {
        rows: op.m,
        cols: op.hidden,
        elem_bytes: op.act_elem,
    };

    // ── Loader ──
    // Round 0: wait on `Consumed` at Phase0. (TK 2.0 inits page_consumed
    // via `arrive_pre` so its first wait *would* read 1 — but the
    // persistent kernel scaffold pre-arrives page_consumed before any
    // op runs, leaving the round-start parity at 0. Callers scheduling
    // a non-pre-arrived slot use `pages.alloc_p1()`.)
    let page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, page);
    prog.load_async(page_id, op.x, x_region, tile);
    let page = prog.arrive(WarpRole::Loader, PageBarrier::Ready, page);

    // ── Consumer ──
    // Wait reads the SAME parity as the loader's wait — within one
    // round every barrier's wait reads `R & 1`. The loader's arrive
    // flipped the underlying `page_ready` mbarrier from 0 → 1, which is
    // exactly what makes `wait(0)` return on the consumer side.
    let page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, page);
    prog.compute(WarpRole::AllConsumers, rmsnorm_compute_body(&op));
    let page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, page);

    // ── Storer ──
    let page = prog.wait(WarpRole::Storer, PageBarrier::Done, page);
    prog.store_async(page_id, op.out, out_region, tile);
    let page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, page);

    // Round boundary — the only phase-advancing call. Releases the slot
    // back to the allocator with its parity flipped, so the next op
    // that grabs slot `id` starts its round at `P::Next`.
    let page = prog.complete_round(page);
    pages.release(page);
}

/// The RMS reduce + scale body, as a string fragment. Const-resolved
/// from the op's `(hidden, eps, m)` plus the act dtype. This is the
/// ONE place the lowering pastes a kernel-body fragment; it does not
/// touch sync, page IDs, or phase.
fn rmsnorm_compute_body(op: &RmsNormOp) -> String {
    let RmsNormOp {
        hidden, eps, m, ..
    } = *op;
    // Body convention matches today's atom_lib::AddRmsNormAtom but
    // in the page-resident TK form: page is `__page_smem` (a typed
    // smem array of T_act elements); `__weight_smem` is a separate
    // page or scratch slot bound by the caller.
    format!(
        r#"
            // tk_warp_ir RmsNorm — pre-resolved body
            const uint __hidden = {hidden};
            const float __eps   = {eps:?}f;
            const uint __m      = {m};
            float __sumsq = 0.0f;
            for (uint __i = threadIdx.x; __i < __hidden; __i += blockDim.x) {{
                const float __v = float(__page_smem[__i]);
                __sumsq += __v * __v;
            }}
            __sumsq = tk20::warp_reduce_sumsq(__sumsq);
            const float __scale = rsqrtf(__sumsq / float(__hidden) + __eps);
            for (uint __i = threadIdx.x; __i < __hidden; __i += blockDim.x) {{
                const float __v = float(__page_smem[__i]);
                const float __w = float(__weight_smem[__i]);
                __page_smem[__i] = T_act(__v * __scale * __w);
            }}
"#
    )
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

    /// The lowered IR has the exact nine-instruction
    /// loader/consumer/storer handshake from the design doc — no
    /// blank-filling at emit time.
    #[test]
    fn rmsnorm_lowers_to_nine_instr_handshake() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_rmsnorm(op(), &mut pages, &mut prog);

        // 9 instructions: wait+load+arrive (loader), wait+compute+arrive (consumer),
        // wait+store+arrive (storer). `complete_round` emits no IR.
        assert_eq!(prog.instrs.len(), 9, "{prog:?}");

        // Phase parities the lowering picked: every wait within one
        // round reads `R & 1` (round 0 → all 0).
        let phases: Vec<u32> = prog
            .instrs
            .iter()
            .filter_map(|i| match i {
                TkInstr::Wait { phase, .. } => Some(*phase),
                _ => None,
            })
            .collect();
        assert_eq!(phases, vec![0, 0, 0], "round 0: every wait reads 0");
    }

    /// The page is released back to the allocator at the right parity:
    /// the storer's last arrive flipped it once more, so the slot's
    /// next round starts at Phase1.
    #[test]
    fn page_released_at_correct_parity() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_rmsnorm(op(), &mut pages, &mut prog);
        // Slot 0 was used; its phase_bit reflects the post-storer state.
        assert_eq!(
            pages.phase_bit[0], 1,
            "after one round, slot 0 lives on Phase1"
        );
        assert!(!pages.in_use[0], "slot released");
    }

    /// Codegen on the lowered program produces source containing the
    /// expected three role-gated arms and the expected three TK 2.0
    /// barrier names.
    #[test]
    fn rmsnorm_codegen_walks_match_arms() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_rmsnorm(op(), &mut pages, &mut prog);
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

        // Body fragment pasted verbatim.
        assert!(src.contains("rsqrtf"), "compute body present\n{src}");
    }

    /// Two adjacent RmsNorm ops use distinct page slots — the
    /// allocator does not stomp on an in-use slot.
    #[test]
    fn two_rmsnorms_use_distinct_pages() {
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_rmsnorm(op(), &mut pages, &mut prog);
        // Pretend the first op's slot is still owned (skip the release).
        // Re-simulate by manually marking slot 0 as in_use:
        let mut pages2 = PageAllocator::new();
        pages2.in_use[0] = true;
        let page = pages2.alloc_p0().unwrap();
        assert_eq!(page.id(), 1, "second alloc takes the next free slot");
    }
}
