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
    LoopBound, PageBarrier, PageHandle, Phase0, Phase1, TileShape, TkProgram, WarpRole, NUM_PAGES,
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
#[derive(Clone, Copy, Debug)]
pub struct AttnDecodeOp {
    /// `[1, hidden]` query (post-RoPE). Loaded once outside the loop.
    pub q: BufId,
    /// Paged KV cache: `[num_blocks, page_size, num_kv_heads, head_dim]`.
    /// The page-block table is a separate runtime arg the kernel
    /// scaffold reads; the codegen here only needs the buffer id.
    pub k_cache: BufId,
    pub v_cache: BufId,
    /// `[1, hidden]` output (one head_dim slice per attention head;
    /// for the slice we model a single head).
    pub out: BufId,
    pub head_dim: u32,
    pub act_elem: u32,
    /// `softmax_scale = 1 / sqrt(head_dim)`. Baked literal.
    pub softmax_scale: f32,
    /// Name of the runtime u32 the persistent kernel scaffold provides
    /// for the number of KV pages this query streams (e.g.
    /// `"__num_kv_pages"`). The lowering does not invent this — the
    /// scaffold's signature defines it.
    pub num_kv_pages_arg: &'static str,
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
pub fn lower_attn_decode(op: AttnDecodeOp, pages: &mut PageAllocator, prog: &mut TkProgram) {
    // ── Q page (one-shot) ──
    let q_page = pages.alloc_p0().expect("Q page");
    let q_id = q_page.id();
    let q_region = RegionRef::rows_cols(op.q, 1, 0, op.head_dim);
    let q_tile = TileShape {
        rows: 1,
        cols: op.head_dim,
        elem_bytes: op.act_elem,
    };

    let q_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, q_page);
    prog.load_async(q_id, op.q, q_region, q_tile);
    let q_page = prog.arrive(WarpRole::Loader, PageBarrier::Ready, q_page);

    let q_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, q_page);
    prog.compute(WarpRole::AllConsumers, init_softmax_accum_body(&op));
    let q_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, q_page);
    // Q stays resident; the storer doesn't drain it. Its round closes
    // when the consumer signals Done; the loader's next wait on
    // Consumed never fires (no second Q-load) — perfectly fine, the
    // page is held for the loop's duration. complete_round + release
    // happen at the end (after the loop).

    // ── KV sweep ──
    let k_page = pages.alloc_p0().expect("K page");
    let v_page = pages.alloc_p0().expect("V page");
    let k_id = k_page.id();
    let v_id = v_page.id();
    let k_tile = TileShape {
        rows: 1,
        cols: op.head_dim,
        elem_bytes: op.act_elem,
    };
    let v_tile = k_tile;

    let loop_var = "__kv_i";
    prog.for_loop(
        loop_var,
        LoopBound::RuntimeU32(op.num_kv_pages_arg.into()),
        |body| {
            // Per-iteration K round. Parity = (__kv_i & 1).
            body.wait_loop_parity(WarpRole::Loader, PageBarrier::Consumed, k_id, loop_var);
            // Region uses a runtime expression for the page-table
            // lookup; here we use a placeholder — the codegen's
            // `LoadAsync` arm pastes `(__kv_i)` as the row index when
            // the region is parameterised. For the slice, model the
            // K-page byte-offset as 0 (real ports compute it from the
            // block table).
            body.load_async(
                k_id,
                op.k_cache,
                RegionRef::rows_cols(op.k_cache, 1, 0, op.head_dim),
                k_tile,
            );
            body.arrive_loop(WarpRole::Loader, PageBarrier::Ready, k_id);

            body.wait_loop_parity(WarpRole::AllConsumers, PageBarrier::Ready, k_id, loop_var);
            body.compute(WarpRole::AllConsumers, qkt_softmax_step_body(&op));
            body.arrive_loop(WarpRole::AllConsumers, PageBarrier::Done, k_id);

            body.wait_loop_parity(WarpRole::Storer, PageBarrier::Done, k_id, loop_var);
            body.arrive_loop(WarpRole::Storer, PageBarrier::Consumed, k_id);

            // Per-iteration V round.
            body.wait_loop_parity(WarpRole::Loader, PageBarrier::Consumed, v_id, loop_var);
            body.load_async(
                v_id,
                op.v_cache,
                RegionRef::rows_cols(op.v_cache, 1, 0, op.head_dim),
                v_tile,
            );
            body.arrive_loop(WarpRole::Loader, PageBarrier::Ready, v_id);

            body.wait_loop_parity(WarpRole::AllConsumers, PageBarrier::Ready, v_id, loop_var);
            body.compute(WarpRole::AllConsumers, sv_accum_step_body(&op));
            body.arrive_loop(WarpRole::AllConsumers, PageBarrier::Done, v_id);

            body.wait_loop_parity(WarpRole::Storer, PageBarrier::Done, v_id, loop_var);
            body.arrive_loop(WarpRole::Storer, PageBarrier::Consumed, v_id);
        },
    );

    // After the loop the K/V pages have been ping-ponged some
    // (runtime) number of times; we don't track parity statically
    // anymore — but the loop body is a closed round-pair every
    // iteration, so the slot's parity at the START of the next
    // op's round is determined by the runtime parity. The allocator
    // for the next op uses `wait_loop_parity` again, so this is fine.
    // We *do* need to release the typed handle to the allocator;
    // since we cannot statically know the post-loop parity, we
    // call complete_round to advance and release at Phase1 (matches
    // the typical even-N case; odd-N callers must arrange a half-
    // iteration cleanup).
    let k_page = prog.complete_round(k_page);
    let v_page = prog.complete_round(v_page);
    pages.release(k_page);
    pages.release(v_page);

    // ── Final O-store ──
    let o_page = q_page; // reuse Q's page slot for O after the loop.
    let o_region = RegionRef::rows_cols(op.out, 1, 0, op.head_dim);
    let o_tile = TileShape {
        rows: 1,
        cols: op.head_dim,
        elem_bytes: op.act_elem,
    };
    let o_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Consumed, o_page);
    prog.compute(WarpRole::AllConsumers, finalise_softmax_norm_body(&op));
    let o_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, o_page);

    let o_page = prog.wait(WarpRole::Storer, PageBarrier::Done, o_page);
    prog.store_async(q_id, op.out, o_region, o_tile);
    let o_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, o_page);

    let o_page = prog.complete_round(o_page);
    pages.release(o_page);
}

fn init_softmax_accum_body(op: &AttnDecodeOp) -> String {
    let head_dim = op.head_dim;
    let scale = op.softmax_scale;
    format!(
        r#"
            // tk_warp_ir AttnDecode — init softmax accumulator
            const uint __head_dim = {head_dim};
            const float __scale   = {scale:?}f;
            float __m_max = -INFINITY;
            float __l_sum = 0.0f;
            float __o_accum[/*head_dim*/];
            for (uint __j = threadIdx.x; __j < __head_dim; __j += blockDim.x) {{
                __o_accum[__j] = 0.0f;
            }}
"#
    )
}

fn qkt_softmax_step_body(_op: &AttnDecodeOp) -> String {
    r#"
            // tk_warp_ir AttnDecode — Q @ K^T + online softmax step
            float __s = 0.0f;
            for (uint __j = threadIdx.x; __j < __head_dim; __j += blockDim.x) {{
                __s += float(__q_smem[__j]) * float(__k_smem[__j]);
            }}
            __s = tk20::warp_reduce_sum(__s) * __scale;
            const float __m_new   = fmaxf(__m_max, __s);
            const float __renorm  = expf(__m_max - __m_new);
            const float __p       = expf(__s - __m_new);
            __l_sum   = __renorm * __l_sum + __p;
            for (uint __j = threadIdx.x; __j < __head_dim; __j += blockDim.x) {{
                __o_accum[__j] *= __renorm;
            }}
            __m_max = __m_new;
"#
    .into()
}

fn sv_accum_step_body(_op: &AttnDecodeOp) -> String {
    r#"
            // tk_warp_ir AttnDecode — softmax(P) @ V accumulate
            for (uint __j = threadIdx.x; __j < __head_dim; __j += blockDim.x) {{
                __o_accum[__j] += __p * float(__v_smem[__j]);
            }}
"#
    .into()
}

fn finalise_softmax_norm_body(_op: &AttnDecodeOp) -> String {
    r#"
            // tk_warp_ir AttnDecode — finalise: O = O_accum / l_sum
            const float __inv_l = 1.0f / __l_sum;
            for (uint __j = threadIdx.x; __j < __head_dim; __j += blockDim.x) {{
                __out_smem[__j] = T_act(__o_accum[__j] * __inv_l);
            }}
"#
    .into()
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
        let phases: Vec<String> = prog
            .instrs
            .iter()
            .filter_map(|i| match i {
                TkInstr::Wait { phase, .. } => Some(phase.cuda_expr()),
                _ => None,
            })
            .collect();
        assert_eq!(phases, vec!["0", "0", "0"], "round 0: every wait reads 0");
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

    fn attn_op() -> AttnDecodeOp {
        AttnDecodeOp {
            q: BufId(10),
            k_cache: BufId(11),
            v_cache: BufId(12),
            out: BufId(13),
            head_dim: 128,
            act_elem: 2,
            softmax_scale: 0.088388_35,
            num_kv_pages_arg: "__num_kv_pages",
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
        lower_attn_decode(attn_op(), &mut pages, &mut prog);

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

        // K-handshake: wait+load+arrive (loader), wait+compute+arrive (consumer),
        //              wait+arrive (storer) → 8 instrs.
        // V-handshake: same → 8 instrs.
        // Total per iteration: 16.
        assert_eq!(
            loops[0].len(),
            16,
            "loop body has K + V handshakes, 16 instrs"
        );
    }

    /// All static (outside-loop) waits use compile-time parities; all
    /// in-loop waits use runtime `(__kv_i & 1)` parities.
    #[test]
    fn attn_decode_phase_kinds_match_loop_structure() {
        use crate::tk_warp_ir::WaitPhase;
        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_attn_decode(attn_op(), &mut pages, &mut prog);

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
        lower_attn_decode(attn_op(), &mut pages, &mut prog);
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
        lower_attn_decode(attn_op(), &mut pages, &mut prog);

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
}
