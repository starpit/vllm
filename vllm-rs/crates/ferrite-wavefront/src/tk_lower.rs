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
    // Phase 1: route the compute body through a typed `Tk20Call`
    // atom. `FERRITE_NEW_RMSNORM=1` opts in to the typed path; the
    // emit is byte-identical to the legacy `format!()` body (the
    // typed atom delegates to `tk20::rmsnorm_consumer_body`, which
    // emits the same CUDA) — the gate is a safety net for the
    // cutover phase per the plan, not a behavior change.
    if std::env::var_os("FERRITE_NEW_RMSNORM").is_some() {
        prog.compute_calls(
            WarpRole::AllConsumers,
            vec![crate::tk_codegen::Tk20Call::RmsNormConsumerBody {
                x_id,
                w_id,
                hidden: op.hidden,
                eps: op.eps,
            }],
        );
    } else {
        prog.compute(WarpRole::AllConsumers, rmsnorm_compute_body(&op, x_id, w_id));
    }
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

/// The RMS reduce + scale body, as a string fragment. Const-resolved
/// from the op's `(hidden, eps, m)` plus the act dtype.
///
/// This is the slice version: weight multiply is omitted (allocate a
/// second page for weight in a follow-up). The compute is gated to
/// consumer warp 0 (`__consumer_idx == 0`); the other 7 consumer warps
/// fall through to the role-routed `arrive(Done)`, which is what gives
/// us the 8 arrivals the page_done init expects.
///
/// The body declares its own typed page view (`__page_smem`) and
/// `T_act` alias up front so the body is self-contained — codegen
/// pastes it inside `if (__role == ROLE_CONSUMER) { ... }` and that's
/// the only context required. Reduction is a single-warp shfl chain
/// (`__shfl_xor_sync` butterfly) — TK 2.0 ships warp-level register
/// reductions but the slice doesn't need them yet, and using bare
/// CUDA primitives keeps the emit independent of TK 2.0's typed-tile
/// machinery for now.
/// Test-only wrapper exposing the legacy `rmsnorm_compute_body` so
/// the byte-identity test in `tk_codegen` can compare typed-atom
/// output against the legacy `format!()` output without making the
/// private fn `pub`.
#[cfg(test)]
pub fn rmsnorm_compute_body_for_test(op: &RmsNormOp, x_id: u8, w_id: u8) -> String {
    rmsnorm_compute_body(op, x_id, w_id)
}

fn rmsnorm_compute_body(op: &RmsNormOp, x_id: u8, w_id: u8) -> String {
    let RmsNormOp { hidden, eps, .. } = *op;
    format!(
        r#"
            // tk_warp_ir RmsNorm — RMS reduce + scale + apply weight (consumer warp 0)
            using T_act = __nv_bfloat16;
            auto* __x_smem = reinterpret_cast<T_act*>(page_buf[{x_id}]);
            auto* __w_smem = reinterpret_cast<T_act*>(page_buf[{w_id}]);
            if (__consumer_idx == 0) {{
                const unsigned int __hidden = {hidden}u;
                const float __eps = {eps:?}f;
                const int __lane = static_cast<int>(threadIdx.x & 31);
                float __sumsq = 0.0f;
                for (unsigned int __i = static_cast<unsigned int>(__lane);
                     __i < __hidden; __i += 32u) {{
                    const float __v = __bfloat162float(__x_smem[__i]);
                    __sumsq += __v * __v;
                }}
                #pragma unroll
                for (int __o = 16; __o > 0; __o >>= 1) {{
                    __sumsq += __shfl_xor_sync(0xFFFFFFFFu, __sumsq, __o);
                }}
                const float __scale = rsqrtf(__sumsq / static_cast<float>(__hidden) + __eps);
                for (unsigned int __i = static_cast<unsigned int>(__lane);
                     __i < __hidden; __i += 32u) {{
                    const float __v = __bfloat162float(__x_smem[__i]);
                    const float __g = __bfloat162float(__w_smem[__i]);
                    __x_smem[__i] = __float2bfloat16(__v * __scale * __g);
                }}
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
    /// Name of the runtime u32 the persistent kernel scaffold provides
    /// for the number of KV pages this query streams (e.g.
    /// `"__num_kv_pages"`). The lowering does not invent this — the
    /// scaffold's signature defines it.
    pub num_kv_pages_arg: &'static str,
    /// Per-AttnDecode unique id, used as a suffix on the function-
    /// scope prelude variable names (`__q_smem_a0`, `__m_max_a0`, …)
    /// so multiple AttnDecode ops in one TkProgram (e.g. one per
    /// transformer block in a multi-layer Llama forward) don't
    /// collide on the same identifiers. The orchestrator passes the
    /// LoweringInput op index — unique across all ops in a forward.
    pub unique_id: u32,
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
    pages: &mut PageAllocator,
    prog: &mut TkProgram,
) {
    // GQA shape preconditions — the multi-head lowering distributes
    // q-heads across the 8 consumer warps (one kv-head per warp).
    // `num_q_heads` must be a multiple of NUM_CONSUMER_WARPS, and
    // `num_kv_heads` must equal NUM_CONSUMER_WARPS so every warp
    // owns exactly one kv-head. Llama-3.2-1B (32 q, 8 kv) satisfies
    // both. Llama-3.2-3B (24 q, 8 kv) and other ratios will need a
    // looser distribution; surface the constraint explicitly here
    // rather than silently producing wrong output.
    debug_assert!(
        op.num_q_heads % (NUM_CONSUMER_WARPS as u32) == 0,
        "lower_attn_decode: num_q_heads ({}) must be divisible by NUM_CONSUMER_WARPS ({})",
        op.num_q_heads,
        NUM_CONSUMER_WARPS,
    );
    debug_assert!(
        op.num_kv_heads == NUM_CONSUMER_WARPS as u32,
        "lower_attn_decode: num_kv_heads ({}) must equal NUM_CONSUMER_WARPS ({}) for the per-warp kv-head sharding",
        op.num_kv_heads,
        NUM_CONSUMER_WARPS,
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
    prog.compute(WarpRole::AllConsumers, init_softmax_accum_body(&op));
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
    prog.for_loop(
        loop_var,
        LoopBound::RuntimeU32(op.num_kv_pages_arg.into()),
        |body| {
            // Per-iteration K round. Parity = (__kv_i & 1) ^ P::VALUE.
            body.wait_loop_parity(WarpRole::Loader, PageBarrier::Consumed, k_id, loop_var, start);
            // Region uses a runtime expression for the page-table
            // lookup; here we use a placeholder — the codegen's
            // `LoadAsync` arm pastes `(__kv_i)` as the row index when
            // the region is parameterised. For the slice, model the
            // K-page byte-offset as 0 (real ports compute it from the
            // block table).
            body.load_async(
                k_id,
                op.k_cache,
                RegionRef::rows_cols(op.k_cache, 1, 0, kv_cols),
                k_tile,
            );
            // No `arrive(Ready)` — `tma::load_async` signals page_ready.

            body.wait_loop_parity(WarpRole::AllConsumers, PageBarrier::Ready, k_id, loop_var, start);
            body.compute(WarpRole::AllConsumers, qkt_softmax_step_body(&op));
            body.arrive_loop(WarpRole::AllConsumers, PageBarrier::Done, k_id);

            body.wait_loop_parity(WarpRole::Storer, PageBarrier::Done, k_id, loop_var, start);
            body.arrive_loop(WarpRole::Storer, PageBarrier::Consumed, k_id);

            // Per-iteration V round.
            body.wait_loop_parity(WarpRole::Loader, PageBarrier::Consumed, v_id, loop_var, start);
            body.load_async(
                v_id,
                op.v_cache,
                RegionRef::rows_cols(op.v_cache, 1, 0, kv_cols),
                v_tile,
            );
            // No `arrive(Ready)` — `tma::load_async` signals page_ready.

            body.wait_loop_parity(WarpRole::AllConsumers, PageBarrier::Ready, v_id, loop_var, start);
            body.compute(WarpRole::AllConsumers, sv_accum_step_body(&op));
            body.arrive_loop(WarpRole::AllConsumers, PageBarrier::Done, v_id);

            body.wait_loop_parity(WarpRole::Storer, PageBarrier::Done, v_id, loop_var, start);
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
    prog.compute(WarpRole::AllConsumers, finalise_softmax_norm_body(&op));
    let o_page = prog.arrive(WarpRole::AllConsumers, PageBarrier::Done, o_page);

    let o_page = prog.wait(WarpRole::Storer, PageBarrier::Done, o_page);
    prog.store_async(q_id, op.out, o_region, o_tile);
    let o_page = prog.arrive(WarpRole::Storer, PageBarrier::Consumed, o_page);

    let o_page = prog.complete_round(o_page);
    pages.release(o_page);
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
    let head_dim = op.head_dim;
    let num_q_heads = op.num_q_heads;
    let num_kv_heads = op.num_kv_heads;
    // 8 consumer warps; each owns N_q / 8 q-heads.
    let q_heads_per_warp = num_q_heads / (NUM_CONSUMER_WARPS as u32);
    let q_per_kv = num_q_heads / num_kv_heads;
    let scale = op.softmax_scale;
    let u = op.unique_id;
    let prelude = format!(
        r#"    // ── AttnDecode #{u} prelude (multi-head GQA; one kv-head per consumer warp) ──
    using T_act = __nv_bfloat16;
    auto* __q_smem_a{u}   = reinterpret_cast<T_act*>(page_buf[{q_id}]);
    auto* __k_smem_a{u}   = reinterpret_cast<T_act*>(page_buf[{k_id}]);
    auto* __v_smem_a{u}   = reinterpret_cast<T_act*>(page_buf[{v_id}]);
    auto* __out_smem_a{u} = reinterpret_cast<T_act*>(page_buf[{q_id}]);
    const unsigned int __head_dim_a{u} = {head_dim}u;
    const unsigned int __num_q_heads_a{u} = {num_q_heads}u;
    const unsigned int __num_kv_heads_a{u} = {num_kv_heads}u;
    const unsigned int __q_heads_per_warp_a{u} = {q_heads_per_warp}u;
    const unsigned int __q_per_kv_a{u} = {q_per_kv}u;
    const float __scale_a{u} = {scale:?}f;
    // Per-warp per-q-head softmax state. With Llama-3.2-1B
    // (N_q=32, 8 warps, qpw=4): each consumer warp keeps 4
    // independent softmax + accumulator states.
    float __m_max_a{u}[{q_heads_per_warp}];
    float __l_sum_a{u}[{q_heads_per_warp}];
    float __renorm_a{u}[{q_heads_per_warp}];
    float __p_a{u}[{q_heads_per_warp}];
    float __o_accum_a{u}[{q_heads_per_warp}][{head_dim}];
"#
    );
    prog.add_prelude(prelude);
}

fn init_softmax_accum_body(op: &AttnDecodeOp) -> String {
    let u = op.unique_id;
    format!(
        r#"
            // tk_warp_ir AttnDecode #{u} — init per-warp softmax state
            {{
                const int __lane = static_cast<int>(threadIdx.x & 31);
                for (unsigned int __h = 0u; __h < __q_heads_per_warp_a{u}; ++__h) {{
                    __m_max_a{u}[__h] = -INFINITY;
                    __l_sum_a{u}[__h] = 0.0f;
                    for (unsigned int __j = static_cast<unsigned int>(__lane);
                         __j < __head_dim_a{u}; __j += 32u) {{
                        __o_accum_a{u}[__h][__j] = 0.0f;
                    }}
                }}
            }}
"#
    )
}

fn qkt_softmax_step_body(op: &AttnDecodeOp) -> String {
    let u = op.unique_id;
    format!(
        r#"
            // tk_warp_ir AttnDecode #{u} — Q@K^T + online softmax
            {{
                const int __lane = static_cast<int>(threadIdx.x & 31);
                const unsigned int __kv_head = static_cast<unsigned int>(__consumer_idx);
                const unsigned int __q_head_base =
                    static_cast<unsigned int>(__consumer_idx) * __q_heads_per_warp_a{u};
                const unsigned int __k_off = __kv_head * __head_dim_a{u};
                for (unsigned int __h = 0u; __h < __q_heads_per_warp_a{u}; ++__h) {{
                    const unsigned int __q_off = (__q_head_base + __h) * __head_dim_a{u};
                    float __s = 0.0f;
                    for (unsigned int __j = static_cast<unsigned int>(__lane);
                         __j < __head_dim_a{u}; __j += 32u) {{
                        __s += __bfloat162float(__q_smem_a{u}[__q_off + __j])
                             * __bfloat162float(__k_smem_a{u}[__k_off + __j]);
                    }}
                    #pragma unroll
                    for (int __o = 16; __o > 0; __o >>= 1) {{
                        __s += __shfl_xor_sync(0xFFFFFFFFu, __s, __o);
                    }}
                    __s *= __scale_a{u};
                    const float __m_new = fmaxf(__m_max_a{u}[__h], __s);
                    __renorm_a{u}[__h] = expf(__m_max_a{u}[__h] - __m_new);
                    __p_a{u}[__h]      = expf(__s              - __m_new);
                    __l_sum_a{u}[__h]  = __renorm_a{u}[__h] * __l_sum_a{u}[__h] + __p_a{u}[__h];
                    for (unsigned int __j = static_cast<unsigned int>(__lane);
                         __j < __head_dim_a{u}; __j += 32u) {{
                        __o_accum_a{u}[__h][__j] *= __renorm_a{u}[__h];
                    }}
                    __m_max_a{u}[__h] = __m_new;
                }}
            }}
"#
    )
}

fn sv_accum_step_body(op: &AttnDecodeOp) -> String {
    let u = op.unique_id;
    format!(
        r#"
            // tk_warp_ir AttnDecode #{u} — softmax(P) @ V
            {{
                const int __lane = static_cast<int>(threadIdx.x & 31);
                const unsigned int __kv_head = static_cast<unsigned int>(__consumer_idx);
                const unsigned int __v_off = __kv_head * __head_dim_a{u};
                for (unsigned int __h = 0u; __h < __q_heads_per_warp_a{u}; ++__h) {{
                    for (unsigned int __j = static_cast<unsigned int>(__lane);
                         __j < __head_dim_a{u}; __j += 32u) {{
                        __o_accum_a{u}[__h][__j] += __p_a{u}[__h]
                            * __bfloat162float(__v_smem_a{u}[__v_off + __j]);
                    }}
                }}
            }}
"#
    )
}

fn finalise_softmax_norm_body(op: &AttnDecodeOp) -> String {
    let u = op.unique_id;
    format!(
        r#"
            // tk_warp_ir AttnDecode #{u} — finalise: O = O_accum / l_sum
            {{
                const int __lane = static_cast<int>(threadIdx.x & 31);
                const unsigned int __q_head_base =
                    static_cast<unsigned int>(__consumer_idx) * __q_heads_per_warp_a{u};
                for (unsigned int __h = 0u; __h < __q_heads_per_warp_a{u}; ++__h) {{
                    const unsigned int __out_off = (__q_head_base + __h) * __head_dim_a{u};
                    const float __inv_l = 1.0f / __l_sum_a{u}[__h];
                    for (unsigned int __j = static_cast<unsigned int>(__lane);
                         __j < __head_dim_a{u}; __j += 32u) {{
                        __out_smem_a{u}[__out_off + __j] =
                            __float2bfloat16(__o_accum_a{u}[__h][__j] * __inv_l);
                    }}
                }}
            }}
"#
    )
}

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
    // Phase 1: typed `Tk20Call::ResidualAddConsumerBody`. See
    // `lower_rmsnorm` for the env-gate rationale.
    if std::env::var_os("FERRITE_NEW_ADD").is_some() {
        prog.compute_calls(
            WarpRole::AllConsumers,
            vec![crate::tk_codegen::Tk20Call::ResidualAddConsumerBody {
                a_id,
                b_id,
                total: op.hidden as u64 * op.m as u64,
            }],
        );
    } else {
        prog.compute(WarpRole::AllConsumers, residual_add_compute_body(&op, a_id, b_id));
    }
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

/// Test-only wrapper for the legacy `residual_add_compute_body`. See
/// `rmsnorm_compute_body_for_test`.
#[cfg(test)]
pub fn residual_add_compute_body_for_test(op: &AddOp, a_id: u8, b_id: u8) -> String {
    residual_add_compute_body(op, a_id, b_id)
}

fn residual_add_compute_body(op: &AddOp, a_id: u8, b_id: u8) -> String {
    let AddOp { hidden, m, .. } = *op;
    let total = hidden as u64 * m as u64;
    format!(
        r#"
            // tk_warp_ir Residual Add — A+B in place on A's page (all consumer warps)
            using T_act = __nv_bfloat16;
            auto* __a_smem = reinterpret_cast<T_act*>(page_buf[{a_id}]);
            auto* __b_smem = reinterpret_cast<T_act*>(page_buf[{b_id}]);
            const unsigned int __total = {total}u;
            const int __tid_in_consumers =
                static_cast<int>(threadIdx.x) - 2 * 32;
            const int __consumer_threads = 8 * 32;
            for (unsigned int __i = static_cast<unsigned int>(__tid_in_consumers);
                 __i < __total; __i += static_cast<unsigned int>(__consumer_threads)) {{
                const float __a = __bfloat162float(__a_smem[__i]);
                const float __b = __bfloat162float(__b_smem[__i]);
                __a_smem[__i] = __float2bfloat16(__a + __b);
            }}
"#
    )
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
    // Phase 2: typed `Tk20Call::SiluMulConsumerBody`. See
    // `lower_rmsnorm` for the env-gate rationale.
    if std::env::var_os("FERRITE_NEW_SILU_MUL").is_some() {
        prog.compute_calls(
            WarpRole::AllConsumers,
            vec![crate::tk_codegen::Tk20Call::SiluMulConsumerBody {
                g_id,
                u_id,
                total: op.intermediate as u64 * op.m as u64,
            }],
        );
    } else {
        prog.compute(WarpRole::AllConsumers, silu_mul_compute_body(&op, g_id, u_id));
    }
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

/// Test-only wrapper for the legacy `silu_mul_compute_body`.
#[cfg(test)]
pub fn silu_mul_compute_body_for_test(op: &SiluMulOp, g_id: u8, u_id: u8) -> String {
    silu_mul_compute_body(op, g_id, u_id)
}

fn silu_mul_compute_body(op: &SiluMulOp, g_id: u8, u_id: u8) -> String {
    let SiluMulOp { intermediate, m, .. } = *op;
    let total = intermediate as u64 * m as u64;
    format!(
        r#"
            // tk_warp_ir SiluMul — silu(gate) * up in place on gate's page (all consumer warps)
            using T_act = __nv_bfloat16;
            auto* __g_smem = reinterpret_cast<T_act*>(page_buf[{g_id}]);
            auto* __u_smem = reinterpret_cast<T_act*>(page_buf[{u_id}]);
            const unsigned int __total = {total}u;
            const int __tid_in_consumers =
                static_cast<int>(threadIdx.x) - 2 * 32;
            const int __consumer_threads = 8 * 32;
            for (unsigned int __i = static_cast<unsigned int>(__tid_in_consumers);
                 __i < __total; __i += static_cast<unsigned int>(__consumer_threads)) {{
                const float __g = __bfloat162float(__g_smem[__i]);
                const float __u = __bfloat162float(__u_smem[__i]);
                const float __silu_g = __g / (1.0f + expf(-__g));
                __g_smem[__i] = __float2bfloat16(__silu_g * __u);
            }}
"#
    )
}

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
    // Phase 2: typed `Tk20Call::RopeConsumerBody`. See `lower_rmsnorm`
    // for the env-gate rationale.
    if std::env::var_os("FERRITE_NEW_ROPE").is_some() {
        let half = op.head_dim / 2;
        let total_pairs = (op.m as u64) * (op.num_heads as u64) * (half as u64);
        prog.compute_calls(
            WarpRole::AllConsumers,
            vec![crate::tk_codegen::Tk20Call::RopeConsumerBody {
                x_id,
                c_id,
                s_id,
                head_dim: op.head_dim,
                total_pairs,
            }],
        );
    } else {
        prog.compute(WarpRole::AllConsumers, rope_compute_body(&op, x_id, c_id, s_id));
    }
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

/// Test-only wrapper for the legacy `rope_compute_body`.
#[cfg(test)]
pub fn rope_compute_body_for_test(op: &RopeRotateOp, x_id: u8, c_id: u8, s_id: u8) -> String {
    rope_compute_body(op, x_id, c_id, s_id)
}

fn rope_compute_body(op: &RopeRotateOp, x_id: u8, c_id: u8, s_id: u8) -> String {
    let RopeRotateOp { head_dim, num_heads, m, .. } = *op;
    let half = head_dim / 2;
    let total_pairs = (m as u64) * (num_heads as u64) * (half as u64);
    format!(
        r#"
            // tk_warp_ir RoPE rotate (NeoX) — in place on x's page (all consumer warps)
            using T_act = __nv_bfloat16;
            auto* __x_smem   = reinterpret_cast<T_act*>(page_buf[{x_id}]);
            auto* __cos_smem = reinterpret_cast<T_act*>(page_buf[{c_id}]);
            auto* __sin_smem = reinterpret_cast<T_act*>(page_buf[{s_id}]);
            const unsigned int __head_dim = {head_dim}u;
            const unsigned int __half     = {half}u;
            const unsigned int __pairs    = {total_pairs}u;
            const int __tid_in_consumers =
                static_cast<int>(threadIdx.x) - 2 * 32;
            const int __consumer_threads = 8 * 32;
            for (unsigned int __p = static_cast<unsigned int>(__tid_in_consumers);
                 __p < __pairs; __p += static_cast<unsigned int>(__consumer_threads)) {{
                // Decompose pair index → (row * head, lane in head_dim/2).
                const unsigned int __row_head = __p / __half;
                const unsigned int __lane     = __p % __half;
                const unsigned int __i_lo     = __row_head * __head_dim + __lane;
                const unsigned int __i_hi     = __i_lo + __half;
                const float __c   = __bfloat162float(__cos_smem[__lane]);
                const float __s   = __bfloat162float(__sin_smem[__lane]);
                const float __x_lo = __bfloat162float(__x_smem[__i_lo]);
                const float __x_hi = __bfloat162float(__x_smem[__i_hi]);
                __x_smem[__i_lo] = __float2bfloat16(__x_lo * __c - __x_hi * __s);
                __x_smem[__i_hi] = __float2bfloat16(__x_lo * __s + __x_hi * __c);
            }}
"#
    )
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
    debug_assert!(
        op.bn.saturating_mul(op.act_elem) <= PAGE_SIZE,
        "lower_gemm_m1: Y tile [1,{}] ({} bytes) exceeds PAGE_SIZE={}",
        op.bn,
        op.bn * op.act_elem,
        PAGE_SIZE,
    );

    let x_page = pages.alloc_at::<P>().expect("gemm_m1: x page");
    let w_page = pages.alloc_at::<P>().expect("gemm_m1: w page");
    let y_page = pages.alloc_at::<P>().expect("gemm_m1: y page");
    let x_id = x_page.id();
    let w_id = w_page.id();
    let y_id = y_page.id();

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
    let y_tile = TileShape {
        rows: 1,
        cols: op.bn,
        elem_bytes: op.act_elem,
    };

    // ── Load X once before the N-block loop ──
    let x_page = prog.wait(WarpRole::Loader, PageBarrier::Consumed, x_page);
    prog.load_async(x_id, op.x, region(op.x, 1, op.k), x_tile);
    let x_page = prog.wait(WarpRole::AllConsumers, PageBarrier::Ready, x_page);

    // ── N-block loop ──
    let n_blocks = op.n.div_ceil(op.bn);
    let w_byte_step = (op.bn as u64) * (op.k as u64) * (op.act_elem as u64);
    let y_byte_step = (op.bn as u64) * (op.act_elem as u64);
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
            format!("(__n_i * {w_byte_step}u)"),
        );

        // Consumer: wait y free, wait W ready, compute, signal both done.
        body.wait_loop_parity(WarpRole::AllConsumers, PageBarrier::Consumed, y_id, loop_var, start);
        body.wait_loop_parity(WarpRole::AllConsumers, PageBarrier::Ready, w_id, loop_var, start);
        body.compute(WarpRole::AllConsumers, gemm_m1_compute_body(&op, x_id, w_id, y_id));
        body.arrive_loop(WarpRole::AllConsumers, PageBarrier::Done, w_id);
        body.arrive_loop(WarpRole::AllConsumers, PageBarrier::Done, y_id);

        // Storer: free W slot (read-only — no actual store), drain Y.
        body.wait_loop_parity(WarpRole::Storer, PageBarrier::Done, w_id, loop_var, start);
        body.arrive_loop(WarpRole::Storer, PageBarrier::Consumed, w_id);

        body.wait_loop_parity(WarpRole::Storer, PageBarrier::Done, y_id, loop_var, start);
        body.store_async_dyn(
            y_id,
            op.out,
            region(op.out, 1, op.bn),
            y_tile,
            format!("(__n_i * {y_byte_step}u)"),
        );
        body.arrive_loop(WarpRole::Storer, PageBarrier::Consumed, y_id);
    });

    // After the loop the W and Y page slots have been ping-ponged
    // exactly `n_blocks` times. Each iteration flips each barrier
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
        let y_page = prog.complete_round(y_page);
        pages.release(w_page);
        pages.release(y_page);
    } else {
        pages.release(w_page);
        pages.release(y_page);
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

fn gemm_m1_compute_body(op: &GemmM1Op, x_id: u8, w_id: u8, y_id: u8) -> String {
    let GemmM1Op { k, bn, .. } = *op;
    format!(
        r#"
            // tk_warp_ir GemmM1 — y[1, {bn}] = X[1, {k}] @ W[{bn}, {k}]^T
            // (per-warp output: warp c -> y[c] when c < bn; lane-parallel K reduce.)
            using T_act = __nv_bfloat16;
            auto* __x_smem = reinterpret_cast<T_act*>(page_buf[{x_id}]);
            auto* __w_smem = reinterpret_cast<T_act*>(page_buf[{w_id}]);
            auto* __y_smem = reinterpret_cast<T_act*>(page_buf[{y_id}]);
            const unsigned int __k  = {k}u;
            const unsigned int __bn = {bn}u;
            if (static_cast<unsigned int>(__consumer_idx) < __bn) {{
                const unsigned int __row = static_cast<unsigned int>(__consumer_idx);
                const int __lane = static_cast<int>(threadIdx.x & 31);
                float __acc = 0.0f;
                for (unsigned int __j = static_cast<unsigned int>(__lane);
                     __j < __k; __j += 32u) {{
                    __acc += __bfloat162float(__x_smem[__j])
                           * __bfloat162float(__w_smem[__row * __k + __j]);
                }}
                #pragma unroll
                for (int __o = 16; __o > 0; __o >>= 1) {{
                    __acc += __shfl_xor_sync(0xFFFFFFFFu, __acc, __o);
                }}
                if (__lane == 0) {{
                    __y_smem[__row] = __float2bfloat16(__acc);
                }}
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
            num_kv_pages_arg: "__num_kv_pages",
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
        lower_attn_decode::<Phase0>(attn_op(), &mut pages, &mut prog);

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
        lower_attn_decode::<Phase0>(attn_op(), &mut pages, &mut prog);

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
        lower_attn_decode::<Phase0>(attn_op(), &mut pages, &mut prog);
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
        lower_attn_decode::<Phase0>(attn_op(), &mut pages, &mut prog);

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
    /// X-close (outside). The loop body has 10 instrs:
    /// loader (wait+load) + consumer (wait+wait+compute+arrive+arrive)
    /// + storer-w (wait+arrive) + storer-y (wait+store+arrive).
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
        // Loop body: 12 instrs.
        //  loader  (2): wait Consumed[w], LoadAsync[w]
        //  consumer(5): wait Consumed[y], wait Ready[w], Compute,
        //               arrive Done[w], arrive Done[y]
        //  storer-w(2): wait Done[w], arrive Consumed[w]   (W is read-only)
        //  storer-y(3): wait Done[y], StoreAsync[y], arrive Consumed[y]
        assert_eq!(loops[0].len(), 12, "loop body has 12 instrs (see comment)");
    }

    /// GemmM1 uses three distinct page slots (x, w, y) — none alias.
    #[test]
    fn gemm_m1_uses_three_distinct_pages() {
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
        assert_eq!(ids.len(), 3, "three distinct page slots: {ids:?}");
    }

    /// Codegen on GemmM1 emits the runtime parity `(__n_i & 1)` inside
    /// the loop and the runtime W byte-offset `(__n_i * <step>u)`.
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
        // Y byte step = bn * act_elem = 4 * 2 = 8.
        assert!(
            src.contains("(__n_i * 8u)"),
            "Y TMA store uses runtime byte-offset\n{src}"
        );
        assert!(
            src.contains("__shfl_xor_sync"),
            "lane butterfly reduction in compute body\n{src}"
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
        // Y byte step = 1 * 2 = 2.
        assert!(src.contains("(__n_i * 2u)"), "{src}");
    }
}
