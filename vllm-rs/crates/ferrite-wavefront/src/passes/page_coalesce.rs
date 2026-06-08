// SPDX-License-Identifier: Apache-2.0
//! `page_coalesce_pass` — block-shared `PageId` linear-scan coalescing.
//!
//! ## Why
//!
//! The conservative lowering in `lower_subtile_tape_to_tk_tape` mints fresh
//! [`PageId`]s as it walks the SubtileTape. Even with the slot-recycling
//! free list, a Llama-3.2-1B decode tape ends up emitting `PageId`s far
//! beyond the substrate's [`NUM_PAGES`] = 13 — the conservative pass has no
//! global liveness view. Without coalescing, the emitted `.cu` references
//! `page_buf[60..68]` against a `page_buf[NUM_PAGES]` array, which is
//! out-of-bounds at runtime even though it compiles.
//!
//! Per plan §6.5 ("page coalescing" alongside "shmem promotion, fence
//! narrowing"), the architectural answer is a TkTape→TkTape pass that
//! walks the linearized Instr stream, computes per-PageId live ranges, and
//! reassigns PageIds onto a fixed pool of [`NUM_PAGES`] physical pages via
//! linear-scan greedy.
//!
//! Sister pass to [`crate::passes::rt_alias_pass`] (which does the same
//! for per-warp `RegTileSlot`s). The two passes are nearly structurally
//! identical; the differences:
//!
//! - **Pool cap.** `rt_alias_pass` is unbounded (ptxas regs/thread is the
//!   downstream gate). `page_coalesce_pass` has a HARD cap at
//!   [`NUM_PAGES`] — `__shared__` capacity is bounded by Hopper's 228 KB
//!   dynamic-smem-per-block ceiling, and the substrate (`page_buf`) is
//!   sized to a fixed [`NUM_PAGES`].
//! - **Shape key.** `rt_alias_pass` keys the free pool on
//!   [`crate::tk_tape::RegTileArenaEntry`] (per-warp register-tile shape).
//!   `page_coalesce_pass` does NOT shape-key — every page in `page_buf` is
//!   the same `kittens::st_bf<128, 128>` shape (the substrate is uniform),
//!   so any free physical page is a valid target.
//! - **Postcondition.** `page_coalesce_pass` panics with a liveness
//!   diagnostic if the tape's max-concurrent-live-page count exceeds the
//!   pool cap. That's the compile-time guard against silent OOB at
//!   `page_buf[NUM_PAGES]`.
//!
//! ## Algorithm
//!
//! 1. **Liveness** — walk Instrs in order. For every PageId reference
//!    (read, write, or sync/barrier), update `first_def` / `last_use` for
//!    that page. Loop nesting is tracked the same way as `rt_alias_pass`
//!    so a page first-defined inside a loop body has its effective
//!    lifetime extended to `[loop_open, loop_close]`.
//!
//! 2. **Greedy coalesce, hard cap at [`NUM_PAGES`]** — sort logical
//!    PageIds by `first_def`. Maintain a free pool of physical pages;
//!    on each logical page, free expired physicals back to the pool, then
//!    pick a free physical or panic if the pool is empty (and we're
//!    already at cap).
//!
//! 3. **Rewrite** — walk Instrs again, replacing every `PageId` field
//!    per the `page_remap`.
//!
//! ## Compile-time-or-garbage
//!
//! Per the INVIOLABLE rule (`feedback_compile_time_or_garbage`,
//! `feedback_ff_subtile_compile_time_inviolable`): runtime checks alone
//! are not sufficient. The pass enforces its postcondition statically by
//! construction:
//!
//! - The remap maps every logical PageId to a physical in `[0, NUM_PAGES)`.
//!   The greedy assignment never returns a physical id ≥ `NUM_PAGES` — if
//!   it would, the pass panics at codegen time (NOT at runtime).
//! - The remap is bijective on its domain so `dst_page` and `barrier_page`
//!   in a `LoadSpec` get rewritten consistently — the `page_buf[i]` and
//!   `page_<barrier>[i]` arrays stay in lockstep. The same bijection
//!   covers WGMMA `a_page`/`b_page`, ShTile `lhs/rhs/dst`, and the sync
//!   primitives' `page_id` / `barrier_page` fields.
//!
//! NOTE: ActPageId (act_buf pool) is NOT coalesced by this pass yet.
//! ActPageId currently has zero consumers in the lowering (substrate
//! split step 2c hasn't routed WGMMA A/D through act_buf yet). When it
//! does, a sibling `act_page_coalesce_pass` lands with cap
//! [`crate::tk_tape::NUM_ACT_PAGES`].

use std::collections::BTreeMap;

use crate::tk_tape::{Instr, NUM_PAGES, PageId, TkTape};

/// Kind of access an Instr makes to a `PageId`. Mirrors
/// `passes::rt_alias::SlotAccess`. Not currently used to disambiguate
/// remap decisions (the bijective remap rewrites read- and
/// write-sites identically), but kept on the API for future
/// shape-promotion / fence-narrowing passes that want write-only vs
/// read-only signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum PageAccess {
    Read,
    Write,
    /// E.g. a barrier_init followed by load + arrive — the page's
    /// data is overwritten and its barrier is touched. For liveness
    /// purposes the same as `Write` (defines new data) but with the
    /// expectation of a barrier handshake.
    BarrierDef,
    /// A sync-only reference (`PageBarrierWait*`, `PageBarrierArrive`)
    /// that doesn't read/write the page's tile data but still touches
    /// the page's barrier slot. Counts as a use for liveness.
    Sync,
}

/// Walk a single Instr and collect every `(PageId, PageAccess)`
/// reference. Field-name semantics:
///   - `dst_page` (LoadSpec, BarrierInit), `dst` (ShTile compute) → Write
///   - `src_page` (StoreSpec), `src`, `lhs`, `rhs`, `a_page`, `b_page`
///     (WGMMA), `barrier_page` (TmaExpect / LoadSpec.barrier_page) → Read
///   - `page_id` on PageBarrier* / BarrierInit / ArriveIfRuntimeEven → Sync
///
/// Variants are listed in the same order as `tk_tape::Instr` for ease
/// of audit. New PageId-bearing variants MUST add an arm here — the
/// fallback is `_ => {}` (no PageId), so a missed variant silently
/// makes the pass conservative (no coalesce) rather than incorrect,
/// but the test suite below should catch the obvious ones.
fn instr_page_accesses(instr: &Instr) -> Vec<(PageId, PageAccess)> {
    use Instr::*;
    let mut out: Vec<(PageId, PageAccess)> = Vec::new();
    match instr {
        // ── Sync / barrier primitives ────────────────────────────────
        BarrierInit { page_id, .. } => out.push((*page_id, PageAccess::BarrierDef)),
        PageBarrierWaitStaticP0 { page_id, .. }
        | PageBarrierWaitStaticP1 { page_id, .. }
        | PageBarrierWaitLoopStart0 { page_id, .. }
        | PageBarrierWaitLoopStart1 { page_id, .. }
        | PageBarrierArrive { page_id, .. }
        | ArriveIfRuntimeEven { page_id, .. } => out.push((*page_id, PageAccess::Sync)),

        // ── TMA bulk ─────────────────────────────────────────────────
        LoadAsync(spec) => {
            out.push((spec.dst_page, PageAccess::Write));
            // barrier_page may equal dst_page in current emit but is
            // logically a separate reference — track both so the remap
            // is consistent if a future pass decouples them.
            out.push((spec.barrier_page, PageAccess::Sync));
        }
        StoreAsync(spec) => out.push((spec.src_page, PageAccess::Read)),
        StoreAsyncTyped { dst_page, .. } => {
            // Despite the field name, this READS `page_buf[dst_page]`
            // (the smem source) and writes to gmem via the descriptor.
            out.push((*dst_page, PageAccess::Read));
        }
        TmaExpect { barrier_page, .. } => out.push((*barrier_page, PageAccess::Sync)),

        // ── Sh* compute (operate on smem tiles in pages) ─────────────
        ShTileMul { lhs, rhs, dst, .. } | ShTileAdd { lhs, rhs, dst, .. } | ShTileDiv { lhs, rhs, dst, .. } => {
            out.push((*lhs, PageAccess::Read));
            out.push((*rhs, PageAccess::Read));
            out.push((*dst, PageAccess::Write));
        }
        ShTileExp { src, dst, .. } => {
            out.push((*src, PageAccess::Read));
            out.push((*dst, PageAccess::Write));
        }
        ShTileMulScalar { lhs, dst, .. } | ShTileAddScalar { lhs, dst, .. } => {
            out.push((*lhs, PageAccess::Read));
            out.push((*dst, PageAccess::Write));
        }
        ShTileRowSum { src, .. } => out.push((*src, PageAccess::Read)),
        ShTileMulRow { src, dst, .. } | ShTileMulCol { src, dst, .. } => {
            out.push((*src, PageAccess::Read));
            out.push((*dst, PageAccess::Write));
        }
        // Note: ShVecMulScalar / ShVecAddScalar use SmemVecSlot, NOT
        // PageId. They live in a separate shared-vec arena
        // (`shared_sv_decl`) and are out of scope for this pass.

        // ── Smem ↔ register-file moves ──────────────────────────────
        // Note: LoadVecSmemToReg / StoreRegVecToShmem use SmemVecSlot,
        // not PageId. They live in the shared-vec arena and are out of
        // scope for this pass.
        LoadShmemToReg { src, .. } => out.push((*src, PageAccess::Read)),
        // LoadShmemToRegFromAct uses ActPageId — out of scope for this pass.
        StoreRegTileToShmem { dst, .. } => out.push((*dst, PageAccess::Write)),
        LoadShmemSubTileToReg { src, .. } => out.push((*src, PageAccess::Read)),
        StoreRegTileSubTileToShmem { dst, .. } => out.push((*dst, PageAccess::Write)),

        // ── WGMMA ───────────────────────────────────────────────────
        WgmmaMmaAB_SmemSmem { a_page, b_page, .. } | WgmmaMmaABt_SmemSmem { a_page, b_page, .. } => {
            out.push((*a_page, PageAccess::Read));
            out.push((*b_page, PageAccess::Read));
        }
        WgmmaMmaAB_RegSmem { b_page, .. } | WgmmaMmaABt_RegSmem { b_page, .. } => {
            out.push((*b_page, PageAccess::Read));
        }

        // ── No PageId-bearing fields — exhaustive listing per audit
        // finding `S2`. Catch-all `_ => {}` was a hazard: a future
        // Instr variant adding a PageId field but missing an arm above
        // would silently make the coalescer leave logical PageIds in
        // the tape (e.g., `page_buf[17]` on a NUM_PAGES=5 substrate)
        // → silent shared-memory OOB. The exhaustive match below
        // forces a build error when a new variant is added.
        SyncthreadsCta { .. }
        | SyncthreadsGroup { .. }
        | ThreadfenceBlock { .. }
        | ThreadfenceDevice { .. }
        | ThreadfenceSystem { .. }
        | CommitGroupBulk { .. }
        | WaitGroupBulk { .. }
        | InitRtZero { .. }
        | InitRvNegInfty { .. }
        | InitRvZero { .. }
        | WgmmaFenceAcc { .. }
        | WgmmaAsyncWait { .. }
        | RegTileMulScalar { .. }
        | RegTileRowMaxAcc { .. }
        | RegTileRowSumAcc { .. }
        | RegTileSubRow { .. }
        | RegTileExp2 { .. }
        | RegTileDivRow { .. }
        | RegVecSub { .. }
        | RegVecExp2 { .. }
        | RegVecMul { .. }
        | RegTileCopyConvert { .. }
        | RegVecCopy { .. }
        | RegTileMulRow { .. }
        | LoadVecSmemToReg { .. }
        | StoreRegVecToShmem { .. }
        | RegTileNeg { .. }
        | RegTileExp { .. }
        | RegTileAdd { .. }
        | RegTileSub { .. }
        | RegTileDiv { .. }
        | RegTileMulCol { .. }
        | RegTileAddScalar { .. }
        | ShVecMulScalar { .. }
        | ShVecAddScalar { .. }
        | RegVecUnaryRsqrt { .. }
        | LoadShmemToRegFromAct { .. }
        | DebugOpBeginMarker { .. }
        | ForLoopOpenConst { .. }
        | ForLoopOpenKernelArg { .. }
        | ForLoopClose { .. } => {}
    }
    out
}

/// Per-page live range = `[first_def, last_use]` indices into
/// `tape.instrs`. Mirrors `passes::rt_alias::LiveRange`.
#[derive(Debug, Clone, Copy)]
struct LiveRange {
    first_def: usize,
    last_use: usize,
    /// True if the page's first reference lies inside a loop body. A
    /// loop-local page is rewritten on every iteration; its effective
    /// free-by index is the matching `loop_close` (so an outer page
    /// defined after the close is the next legal user).
    def_in_loop: bool,
    loop_open: usize,
    loop_close: usize,
}

/// Compute live ranges for every `PageId` referenced in the
/// linearized Instr stream. Tracks loop nesting so loop-local pages
/// (defined inside a loop body) are flagged with their enclosing
/// `[loop_open, loop_close]` extent.
fn compute_liveness(instrs: &[Instr]) -> BTreeMap<PageId, LiveRange> {
    let mut ranges: BTreeMap<PageId, LiveRange> = BTreeMap::new();
    let mut loop_opens: Vec<usize> = Vec::new();

    struct InProgress {
        first_def: usize,
        last_use: usize,
        enclosing_open: Option<usize>,
    }
    let mut in_progress: BTreeMap<PageId, InProgress> = BTreeMap::new();
    let mut close_of_open: BTreeMap<usize, usize> = BTreeMap::new();

    for (idx, instr) in instrs.iter().enumerate() {
        match instr {
            Instr::ForLoopOpenConst { .. } | Instr::ForLoopOpenKernelArg { .. } => {
                loop_opens.push(idx);
                continue;
            }
            Instr::ForLoopClose { .. } => {
                if let Some(open) = loop_opens.pop() {
                    close_of_open.insert(open, idx);
                }
                continue;
            }
            _ => {}
        }
        let enclosing_open = loop_opens.last().copied();
        for (page, access) in instr_page_accesses(instr) {
            let entry = in_progress.entry(page).or_insert(InProgress {
                first_def: idx,
                last_use: idx,
                enclosing_open,
            });
            entry.last_use = idx;
            let _ = access;
        }
    }

    for (page, p) in in_progress {
        let (def_in_loop, loop_open, loop_close) = match p.enclosing_open {
            Some(open) => {
                let close = close_of_open.get(&open).copied().unwrap_or(p.last_use);
                (true, open, close)
            }
            None => (false, 0, 0),
        };
        ranges.insert(
            page,
            LiveRange {
                first_def: p.first_def,
                last_use: p.last_use,
                def_in_loop,
                loop_open,
                loop_close,
            },
        );
    }
    ranges
}

/// Greedy linear-scan with a hard physical-page pool of size `cap`.
/// Returns `page_remap: logical → physical` where every value is in
/// `[0, cap)`.
///
/// Panics with a liveness diagnostic if the tape's max-concurrent-live
/// page count exceeds `cap` — the compile-time guard against silent
/// `page_buf[NUM_PAGES]` OOB.
fn coalesce(ranges: &BTreeMap<PageId, LiveRange>, cap: u8) -> BTreeMap<PageId, PageId> {
    let mut remap: BTreeMap<PageId, PageId> = BTreeMap::new();

    // Sort logical pages by first_def so we visit in linear order.
    let mut by_def: Vec<(PageId, LiveRange)> =
        ranges.iter().map(|(p, r)| (*p, *r)).collect();
    by_def.sort_by_key(|(page, range)| (range.first_def, page.0));

    // Free pool of physical PageIds. Each entry: (physical, free_at_idx)
    // — the first linear instr index at which the physical is reusable.
    let mut free_pool: Vec<(PageId, usize)> = (0..cap).map(|i| (PageId(i), 0)).collect();
    // Free-by index for each currently-assigned physical (so we can
    // expire them as we scan further). Indexed by PageId.0.
    let mut in_use: BTreeMap<PageId, usize> = BTreeMap::new();

    for (logical, range) in by_def.iter().copied() {
        // Effective lifetime — same handling as rt_alias_pass: a page
        // first-defined inside a loop body is rewritten every
        // iteration; its predecessor must be dead by the loop_open
        // (not first_def of iter 0), and its successor cannot reuse
        // the physical until loop_close.
        let candidate_first = if range.def_in_loop {
            range.loop_open
        } else {
            range.first_def
        };
        let effective_last_use = if range.def_in_loop {
            range.loop_close
        } else {
            range.last_use
        };

        // Expire any in_use physicals whose free_at < candidate_first.
        let expired: Vec<PageId> = in_use
            .iter()
            .filter_map(|(p, last)| if *last < candidate_first { Some(*p) } else { None })
            .collect();
        for p in expired {
            in_use.remove(&p);
            free_pool.push((p, 0));
        }

        // Pick a free physical (any — the substrate is shape-uniform).
        // Preference: lowest physical id, deterministic emit.
        free_pool.sort_by_key(|(p, _)| p.0);
        let pick = if free_pool.is_empty() {
            None
        } else {
            Some(free_pool.remove(0))
        };

        match pick {
            Some((physical, _)) => {
                remap.insert(logical, physical);
                in_use.insert(physical, effective_last_use);
            }
            None => {
                // Cap exceeded — emit a liveness diagnostic and panic.
                let live: Vec<(PageId, usize)> =
                    in_use.iter().map(|(p, last)| (*p, *last)).collect();
                panic!(
                    "page_coalesce_pass: max-concurrent-live PageId count exceeds \
                     pool cap (NUM_PAGES = {cap}). Logical PageId {logical_id} \
                     at first_def={first_def} (effective_last_use={effective}) \
                     could not be assigned a physical page; in-use physicals \
                     and their effective_last_use indices: {live:?}. \
                     Either: (1) raise NUM_PAGES (cost: more dynamic shared \
                     memory; check Hopper 228 KB cap), (2) add a \
                     shmem-to-gmem spill pass (deferred §6.5), or (3) the \
                     lowering's slot_to_page recycling is incomplete \
                     (investigate `lower_subtile_tape_to_tk_tape::release_page`).",
                    logical_id = logical.0,
                    first_def = range.first_def,
                    effective = effective_last_use,
                );
            }
        }
    }

    remap
}

/// Rewrite every `PageId` reference in the Instr stream per
/// `page_remap`. Mirror of `rt_alias::rewrite_slots` for pages.
fn rewrite_pages(instrs: &mut [Instr], page_remap: &BTreeMap<PageId, PageId>) {
    let lookup = |p: &mut PageId| {
        if let Some(new_p) = page_remap.get(p) {
            *p = *new_p;
        }
    };
    for instr in instrs.iter_mut() {
        rewrite_instr_pages(instr, &lookup);
    }
}

/// Apply `f` to every `PageId` field of `instr`. Mirrors the structure
/// of `instr_page_accesses` — every variant covered there gets a
/// matching arm here. New PageId-bearing variants MUST be added to
/// both.
fn rewrite_instr_pages<F: Fn(&mut PageId)>(instr: &mut Instr, f: &F) {
    use Instr::*;
    match instr {
        // Sync / barrier primitives
        BarrierInit { page_id, .. }
        | PageBarrierWaitStaticP0 { page_id, .. }
        | PageBarrierWaitStaticP1 { page_id, .. }
        | PageBarrierWaitLoopStart0 { page_id, .. }
        | PageBarrierWaitLoopStart1 { page_id, .. }
        | PageBarrierArrive { page_id, .. }
        | ArriveIfRuntimeEven { page_id, .. } => f(page_id),
        // TMA bulk
        LoadAsync(spec) => {
            f(&mut spec.dst_page);
            f(&mut spec.barrier_page);
        }
        StoreAsync(spec) => f(&mut spec.src_page),
        StoreAsyncTyped { dst_page, .. } => f(dst_page),
        TmaExpect { barrier_page, .. } => f(barrier_page),
        // Sh* compute
        ShTileMul { lhs, rhs, dst, .. }
        | ShTileAdd { lhs, rhs, dst, .. }
        | ShTileDiv { lhs, rhs, dst, .. } => {
            f(lhs);
            f(rhs);
            f(dst);
        }
        ShTileExp { src, dst, .. } => {
            f(src);
            f(dst);
        }
        ShTileMulScalar { lhs, dst, .. } | ShTileAddScalar { lhs, dst, .. } => {
            f(lhs);
            f(dst);
        }
        ShTileRowSum { src, .. } => f(src),
        ShTileMulRow { src, dst, .. } | ShTileMulCol { src, dst, .. } => {
            f(src);
            f(dst);
        }
        // ShVecMulScalar / ShVecAddScalar / LoadVecSmemToReg /
        // StoreRegVecToShmem use SmemVecSlot, not PageId.
        // Smem ↔ register-file moves
        LoadShmemToReg { src, .. } => f(src),
        StoreRegTileToShmem { dst, .. } => f(dst),
        LoadShmemSubTileToReg { src, .. } => f(src),
        StoreRegTileSubTileToShmem { dst, .. } => f(dst),
        // WGMMA
        WgmmaMmaAB_SmemSmem { a_page, b_page, .. }
        | WgmmaMmaABt_SmemSmem { a_page, b_page, .. } => {
            f(a_page);
            f(b_page);
        }
        WgmmaMmaAB_RegSmem { b_page, .. } | WgmmaMmaABt_RegSmem { b_page, .. } => {
            f(b_page);
        }
        // Exhaustive non-PageId-bearing list. Same rationale as
        // `instr_page_accesses` — adding a new PageId-bearing
        // variant must cover BOTH places.
        SyncthreadsCta { .. }
        | SyncthreadsGroup { .. }
        | ThreadfenceBlock { .. }
        | ThreadfenceDevice { .. }
        | ThreadfenceSystem { .. }
        | CommitGroupBulk { .. }
        | WaitGroupBulk { .. }
        | InitRtZero { .. }
        | InitRvNegInfty { .. }
        | InitRvZero { .. }
        | WgmmaFenceAcc { .. }
        | WgmmaAsyncWait { .. }
        | RegTileMulScalar { .. }
        | RegTileRowMaxAcc { .. }
        | RegTileRowSumAcc { .. }
        | RegTileSubRow { .. }
        | RegTileExp2 { .. }
        | RegTileDivRow { .. }
        | RegVecSub { .. }
        | RegVecExp2 { .. }
        | RegVecMul { .. }
        | RegTileCopyConvert { .. }
        | RegVecCopy { .. }
        | RegTileMulRow { .. }
        | LoadVecSmemToReg { .. }
        | StoreRegVecToShmem { .. }
        | RegTileNeg { .. }
        | RegTileExp { .. }
        | RegTileAdd { .. }
        | RegTileSub { .. }
        | RegTileDiv { .. }
        | RegTileMulCol { .. }
        | RegTileAddScalar { .. }
        | ShVecMulScalar { .. }
        | ShVecAddScalar { .. }
        | RegVecUnaryRsqrt { .. }
        | LoadShmemToRegFromAct { .. }
        | DebugOpBeginMarker { .. }
        | ForLoopOpenConst { .. }
        | ForLoopOpenKernelArg { .. }
        | ForLoopClose { .. } => {}
    }
}

/// Rewrite [`PreludeDecl`] PageId references to track the rewrite
/// applied to the Instr stream. Currently relevant: `SmemTilePtr`'s
/// `page` field. Without this, `auto p<X> = &page_buf[X];` aliases in
/// the kernel preamble go stale relative to the body.
fn rewrite_prelude_pages(prelude: &mut [crate::tk_tape::PreludeDecl], remap: &BTreeMap<PageId, PageId>) {
    use crate::tk_tape::PreludeDecl;
    for d in prelude.iter_mut() {
        if let PreludeDecl::SmemTilePtr { page, .. } = d {
            if let Some(new_p) = remap.get(page) {
                *page = *new_p;
            }
        }
    }
}

/// `page_coalesce_pass` — coalesce `PageId`s with disjoint live
/// ranges onto a fixed pool of [`NUM_PAGES`] physical pages. Reduces
/// the substrate's `__shared__` capacity requirement by reusing
/// physical pages whose contents are dead.
///
/// **Postcondition (in-pass-checked, NOT validator-checked):** every
/// `PageId` referenced by an Instr (or by a `PreludeDecl::SmemTilePtr`)
/// is in `[0, NUM_PAGES)`. Enforced by construction in [`coalesce`]
/// which panics with a liveness diagnostic if the cap is exceeded,
/// plus a defensive walk at the end of this fn that asserts the
/// remap covers every reference. `validate_tk_tape` does NOT cross-
/// check this — it is stage-blind and pre-coalesce tapes legitimately
/// hold PageIds beyond NUM_PAGES (audit 2026-06-08 finding #13).
pub fn page_coalesce_pass(tape: &mut TkTape) {
    let ranges = compute_liveness(&tape.instrs);
    if ranges.is_empty() {
        return;
    }
    let cap: u8 = u8::try_from(NUM_PAGES).expect("NUM_PAGES fits u8 by substrate construction");
    let remap = coalesce(&ranges, cap);
    rewrite_pages(&mut tape.instrs, &remap);
    rewrite_prelude_pages(&mut tape.prelude, &remap);

    // Postcondition: every emitted PageId after rewrite is < cap.
    // This is structurally true by `coalesce` construction (we only
    // insert physicals from `0..cap`), but a defensive walk catches
    // any future variant that bears a PageId we forgot to add to
    // `rewrite_instr_pages`.
    for (idx, instr) in tape.instrs.iter().enumerate() {
        for (page, _) in instr_page_accesses(instr) {
            assert!(
                page.0 < cap,
                "page_coalesce_pass postcondition failed: Instr at index {idx} \
                 references PageId({pid}) >= NUM_PAGES ({cap}). The pass missed \
                 a PageId-bearing field — add the variant to `rewrite_instr_pages` \
                 in `passes::page_coalesce`.",
                pid = page.0,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tk_tape::{
        AllConsumersRole, ArrivalCount, Bf16, GroupWidth, PageBarrier, PageId, RegTileId,
        RoleWitness, RowLayout, SmemTileId, TkTape,
    };

    /// Two non-overlapping single-page chains coalesce onto the same
    /// physical page (PageId(0)).
    #[test]
    fn coalesces_disjoint_pages() {
        let mut tape = TkTape::default();
        let p0 = PageId(0);
        let p1 = PageId(1);
        // Chain 1: barrier-init p0, wait, arrive
        tape.instrs.push(Instr::BarrierInit {
            page_id: p0,
            kind: PageBarrier::Ready,
            count: ArrivalCount::One,
        });
        tape.instrs.push(Instr::PageBarrierWaitStaticP0 {
            page_id: p0,
            kind: PageBarrier::Ready,
            role: crate::tk_tape::AllWarpsRole.to_warp_role(),
        });
        tape.instrs.push(Instr::PageBarrierArrive {
            page_id: p0,
            kind: PageBarrier::Done,
            role: crate::tk_tape::AllWarpsRole.to_warp_role(),
        });
        // Chain 2: barrier-init p1, wait, arrive — fully after chain 1
        tape.instrs.push(Instr::BarrierInit {
            page_id: p1,
            kind: PageBarrier::Ready,
            count: ArrivalCount::One,
        });
        tape.instrs.push(Instr::PageBarrierWaitStaticP0 {
            page_id: p1,
            kind: PageBarrier::Ready,
            role: crate::tk_tape::AllWarpsRole.to_warp_role(),
        });
        tape.instrs.push(Instr::PageBarrierArrive {
            page_id: p1,
            kind: PageBarrier::Done,
            role: crate::tk_tape::AllWarpsRole.to_warp_role(),
        });

        page_coalesce_pass(&mut tape);

        // After: every reference is to physical PageId(0) — both
        // logical chains are disjoint and same-shape so they share.
        for instr in &tape.instrs {
            let accesses = instr_page_accesses(instr);
            for (p, _) in accesses {
                assert_eq!(
                    p,
                    PageId(0),
                    "expected coalesce to PageId(0), got PageId({})",
                    p.0
                );
            }
        }
    }

    /// Postcondition holds for a tape that already fits within
    /// NUM_PAGES (idempotency-ish — already-coalesced tapes are
    /// well-formed).
    #[test]
    fn idempotent_on_already_coalesced_tape() {
        let mut tape = TkTape::default();
        for i in 0..(NUM_PAGES as u8) {
            tape.instrs.push(Instr::BarrierInit {
                page_id: PageId(i),
                kind: PageBarrier::Ready,
                count: ArrivalCount::One,
            });
            tape.instrs.push(Instr::PageBarrierArrive {
                page_id: PageId(i),
                kind: PageBarrier::Done,
                role: crate::tk_tape::AllWarpsRole.to_warp_role(),
            });
        }
        page_coalesce_pass(&mut tape);
        // Postcondition: every PageId in [0, NUM_PAGES).
        for instr in &tape.instrs {
            for (p, _) in instr_page_accesses(instr) {
                assert!(p.0 < (NUM_PAGES as u8));
            }
        }
    }

    /// Many fully-disjoint single-page chains — far more than
    /// NUM_PAGES — all collapse onto one physical page.
    #[test]
    fn many_disjoint_pages_collapse_to_pool() {
        let mut tape = TkTape::default();
        // 3 × NUM_PAGES disjoint chains.
        let n_chains = (NUM_PAGES as u8).saturating_mul(3);
        for i in 0..n_chains {
            let p = PageId(i);
            tape.instrs.push(Instr::BarrierInit {
                page_id: p,
                kind: PageBarrier::Ready,
                count: ArrivalCount::One,
            });
            tape.instrs.push(Instr::PageBarrierArrive {
                page_id: p,
                kind: PageBarrier::Done,
                role: crate::tk_tape::AllWarpsRole.to_warp_role(),
            });
        }
        page_coalesce_pass(&mut tape);
        for instr in &tape.instrs {
            for (p, _) in instr_page_accesses(instr) {
                assert!(
                    p.0 < (NUM_PAGES as u8),
                    "post-pass PageId({}) >= NUM_PAGES",
                    p.0
                );
            }
        }
    }

    /// Cap exceeded → panic. A tape with NUM_PAGES + 1 simultaneously
    /// live pages cannot fit into the pool. The pass must panic with
    /// the liveness diagnostic, NOT silently produce an OOB tape.
    #[test]
    #[should_panic(expected = "page_coalesce_pass: max-concurrent-live PageId count exceeds")]
    fn panics_when_concurrent_live_exceeds_pool_cap() {
        let mut tape = TkTape::default();
        // Initialize NUM_PAGES + 1 distinct pages, all simultaneously
        // live (no arrive in between).
        for i in 0..((NUM_PAGES as u8) + 1) {
            tape.instrs.push(Instr::BarrierInit {
                page_id: PageId(i),
                kind: PageBarrier::Ready,
                count: ArrivalCount::One,
            });
        }
        // All used at the very end — every page is live across
        // [first_def=its-init, last_use=its-arrive].
        for i in 0..((NUM_PAGES as u8) + 1) {
            tape.instrs.push(Instr::PageBarrierArrive {
                page_id: PageId(i),
                kind: PageBarrier::Done,
                role: crate::tk_tape::AllWarpsRole.to_warp_role(),
            });
        }
        page_coalesce_pass(&mut tape);
    }

    /// `LoadShmemToReg` PageId field is rewritten consistently with
    /// the surrounding sync ops.
    #[test]
    fn rewrites_load_shmem_to_reg_src() {
        let mut tape = TkTape::default();
        let dst: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        let src_high = SmemTileId::<16, 128, Bf16>::from_page(PageId(42));
        // First chain on page 0, then a load from page 42 — the
        // pass should remap page 42 onto a physical < NUM_PAGES.
        tape.instrs.push(Instr::BarrierInit {
            page_id: PageId(0),
            kind: PageBarrier::Ready,
            count: ArrivalCount::One,
        });
        tape.instrs.push(Instr::PageBarrierArrive {
            page_id: PageId(0),
            kind: PageBarrier::Done,
            role: crate::tk_tape::AllWarpsRole.to_warp_role(),
        });
        tape.instrs.push(Instr::BarrierInit {
            page_id: PageId(42),
            kind: PageBarrier::Ready,
            count: ArrivalCount::One,
        });
        tape.instrs.push(Instr::load_shmem_to_reg(
            src_high,
            dst,
            GroupWidth::<1>::PER_WARP,
            AllConsumersRole,
        ));

        page_coalesce_pass(&mut tape);

        // Every PageId in the rewritten tape is < NUM_PAGES.
        for instr in &tape.instrs {
            for (p, _) in instr_page_accesses(instr) {
                assert!(p.0 < (NUM_PAGES as u8));
            }
        }
    }
}
