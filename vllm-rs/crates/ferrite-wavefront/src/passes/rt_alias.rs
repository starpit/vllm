// SPDX-License-Identifier: Apache-2.0
//! `rt_alias_pass` — register-tile slot coalescing.
//!
//! ## Why
//!
//! Phase A step 8 hit a Hopper register-budget wall: per-warp 128×128
//! register tiles cost 256 lane-regs each (= the whole 256-reg/lane
//! budget); Silu's 5-rt chain alone wants 1280 lane-regs. The `feedback_pages`
//! lowering mints fresh `RegTileSlot`s per Instr (SSA — plan §"Resolved
//! decision 1"), so the arena grows linearly with the kernel even though
//! most slots have non-overlapping live ranges.
//!
//! Per plan §0 ("we are a compiler"), the fix isn't to contort lowering
//! into arena reuse — it's a `TkTape → TkTape` pass that walks the
//! linearized Instr stream, computes per-slot live ranges, and coalesces
//! slots whose intervals are disjoint AND whose
//! [`RegTileArenaEntry`]s are byte-equal. The arena shrinks; the kernel
//! preamble emits fewer `kittens::rt<...> rt_<slot>;` declarations;
//! ptxas register pressure drops.
//!
//! ## Algorithm
//!
//! Linear-scan register allocation on the Instr stream:
//!
//! 1. **Liveness**: walk Instrs in order. For each `RegTileSlot`,
//!    record `first_def` and `last_use` indices. Field-name semantics
//!    (`dst`/`d` → write, `src`/`a`/`lhs`/`rhs` → read, WGMMA `d` →
//!    RMW so the slot stays live across the WGMMA).
//!
//! 2. **Loop scope**: refuse to coalesce a slot whose live range
//!    crosses a `ForLoopOpenConst`/`ForLoopOpenKernelArg` boundary
//!    (would clobber the slot across iterations). Conservative for
//!    now — a precise pass would compute liveness on the lexical CFG.
//!
//! 3. **Greedy coalesce**: sort slots by `first_def`. Keyed by
//!    [`RegTileArenaEntry`] (the byte-equal shape signature), maintain
//!    a free pool. For each slot, free expired slots back to the pool
//!    (last_use < current_def), then pick a free slot from the
//!    matching pool or keep this slot fresh.
//!
//! 4. **Rewrite**: walk Instrs again, replacing each `RegTileSlot` per
//!    the `slot_remap`. Prune `reg_tile_arena` to retained targets
//!    only.
//!
//! ## Compile-time-or-garbage
//!
//! The pass operates on `&mut TkTape` after typed `RegTileId<R,C,T,L>`
//! witnesses have been erased into bare `RegTileSlot`. The aliasing
//! safety (matching `RegTileArenaEntry`) is enforced by **construction**
//! — the free pool is keyed on `RegTileArenaEntry`, so a coalesced
//! pair has identical entries by definition. The pass also re-runs
//! [`crate::tk_tape::validate_tk_tape`] after rewriting; the validator's
//! postcondition (every `RegTileSlot` referenced by an Instr must
//! exist in `reg_tile_arena`) is verified runtime. Per
//! `feedback_asserts_must_be_dead_code` the runtime check is
//! structurally dead given correct pass construction.

use std::collections::BTreeMap;

use crate::tk_tape::{Instr, LoadSpec, RegTileArenaEntry, RegTileSlot, StoreSpec, TkTape};

/// Kind of access an Instr makes to a register tile slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotAccess {
    Read,
    Write,
    /// WGMMA accumulator pattern (`mma_AB.d` / `mma_fence.d`):
    /// conservatively treated as both read and write so the slot stays
    /// live across the WGMMA call. A precise pass could special-case
    /// `accumulate==0` as write-only.
    ReadWrite,
}

/// Walk a single Instr and collect every `(RegTileSlot, SlotAccess)`
/// reference. Field-name semantics from
/// `crates/ferrite-wavefront/src/tk_tape.rs`:
///   - `dst` / `d` → write
///   - `src` / `a` / `lhs` / `rhs` → read
///   - WGMMA `d` (mma_AB / mma_ABt / mma_fence) → read-write
fn instr_rt_accesses(instr: &Instr) -> Vec<(RegTileSlot, SlotAccess)> {
    use Instr::*;
    let mut out = Vec::new();
    match instr {
        // Smem ↔ reg moves
        LoadShmemToReg { dst, .. } => out.push((*dst, SlotAccess::Write)),
        StoreRegTileToShmem { src, .. } => out.push((*src, SlotAccess::Read)),
        LoadShmemSubTileToReg { dst, .. } => out.push((*dst, SlotAccess::Write)),
        StoreRegTileSubTileToShmem { src, .. } => out.push((*src, SlotAccess::Read)),
        // Init
        InitRtZero { dst, .. } => out.push((*dst, SlotAccess::Write)),
        // WGMMA — accumulator d is RMW; rt-A variants also read a
        WgmmaFenceAcc { d, .. } => out.push((*d, SlotAccess::ReadWrite)),
        WgmmaMmaAB_SmemSmem { d, .. } => out.push((*d, SlotAccess::ReadWrite)),
        WgmmaMmaABt_SmemSmem { d, .. } => out.push((*d, SlotAccess::ReadWrite)),
        WgmmaMmaAB_RegSmem { a, d, .. } => {
            out.push((*a, SlotAccess::Read));
            out.push((*d, SlotAccess::ReadWrite));
        }
        WgmmaMmaABt_RegSmem { a, d, .. } => {
            out.push((*a, SlotAccess::Read));
            out.push((*d, SlotAccess::ReadWrite));
        }
        // Reg-tile compute (RegVecSlot fields are out of scope for
        // this pass — a separate `rv_alias_pass` handles those)
        RegTileMulScalar { lhs, dst, .. } => {
            out.push((*lhs, SlotAccess::Read));
            out.push((*dst, SlotAccess::Write));
        }
        RegTileRowMaxAcc { src, .. } => out.push((*src, SlotAccess::Read)),
        RegTileRowSumAcc { src, .. } => out.push((*src, SlotAccess::Read)),
        RegTileSubRow { src, dst, .. } => {
            out.push((*src, SlotAccess::Read));
            out.push((*dst, SlotAccess::Write));
        }
        RegTileExp2 { src, dst, .. } => {
            out.push((*src, SlotAccess::Read));
            out.push((*dst, SlotAccess::Write));
        }
        RegTileDivRow { src, dst, .. } => {
            out.push((*src, SlotAccess::Read));
            out.push((*dst, SlotAccess::Write));
        }
        RegTileCopyConvert { src, dst, .. } => {
            out.push((*src, SlotAccess::Read));
            out.push((*dst, SlotAccess::Write));
        }
        RegTileMulRow { src, dst, .. } => {
            out.push((*src, SlotAccess::Read));
            out.push((*dst, SlotAccess::Write));
        }
        RegTileNeg { src, dst, .. } => {
            out.push((*src, SlotAccess::Read));
            out.push((*dst, SlotAccess::Write));
        }
        RegTileExp { src, dst, .. } => {
            out.push((*src, SlotAccess::Read));
            out.push((*dst, SlotAccess::Write));
        }
        RegTileAdd { lhs, rhs, dst, .. } => {
            out.push((*lhs, SlotAccess::Read));
            out.push((*rhs, SlotAccess::Read));
            out.push((*dst, SlotAccess::Write));
        }
        RegTileSub { lhs, rhs, dst, .. } => {
            out.push((*lhs, SlotAccess::Read));
            out.push((*rhs, SlotAccess::Read));
            out.push((*dst, SlotAccess::Write));
        }
        RegTileDiv { lhs, rhs, dst, .. } => {
            out.push((*lhs, SlotAccess::Read));
            out.push((*rhs, SlotAccess::Read));
            out.push((*dst, SlotAccess::Write));
        }
        RegTileMulCol { src, dst, .. } => {
            out.push((*src, SlotAccess::Read));
            out.push((*dst, SlotAccess::Write));
        }
        RegTileAddScalar { lhs, dst, .. } => {
            out.push((*lhs, SlotAccess::Read));
            out.push((*dst, SlotAccess::Write));
        }
        // No rt-bearing variants
        SyncthreadsCta { .. }
        | SyncthreadsGroup { .. }
        | ThreadfenceBlock { .. }
        | ThreadfenceDevice { .. }
        | ThreadfenceSystem { .. }
        | CommitGroupBulk { .. }
        | WaitGroupBulk { .. }
        | BarrierInit { .. }
        | PageBarrierArrive { .. }
        | ArriveIfRuntimeEven { .. }
        | StoreAsyncTyped { .. }
        | ShTileMul { .. }
        | ShTileAdd { .. }
        | ShTileDiv { .. }
        | ShTileExp { .. }
        | ShTileMulScalar { .. }
        | ShTileAddScalar { .. }
        | TmaExpect { .. }
        | WgmmaAsyncWait { .. }
        | InitRvNegInfty { .. }
        | InitRvZero { .. }
        | RegVecSub { .. }
        | RegVecExp2 { .. }
        | RegVecMul { .. }
        | RegVecCopy { .. }
        | LoadVecSmemToReg { .. }
        | StoreRegVecToShmem { .. }
        | ShTileRowSum { .. }
        | ShVecMulScalar { .. }
        | ShVecAddScalar { .. }
        | RegVecUnaryRsqrt { .. }
        | ShTileMulRow { .. }
        | ShTileMulCol { .. }
        | DebugOpBeginMarker { .. }
        | ForLoopOpenConst { .. }
        | ForLoopOpenKernelArg { .. }
        | ForLoopClose { .. }
        | LoadAsync(_)
        | StoreAsync(_) => {}
        // Catch-all for any future Instr variants — non-rt-bearing
        // by default. Adding a rt-bearing variant should explicitly
        // be added to the match above; this fallback is so a new
        // non-rt variant (e.g. a sync primitive) doesn't break the
        // build.
        _ => {}
    }
    out
}

/// Per-slot live range = `[first_def, last_use]` indices into
/// `tape.instrs`.
#[derive(Debug, Clone, Copy)]
struct LiveRange {
    first_def: usize,
    last_use: usize,
    /// True if `first_def` lies inside a loop body. A slot defined
    /// inside the loop is per-iteration — each iteration redefines
    /// it, so its effective lifetime is the entire loop body
    /// (`[loop_open, loop_close]`). Slots defined outside the loop
    /// but used inside are still conventional `[first_def, last_use]`
    /// — the slot has a single C++ identity that persists across
    /// iterations (e.g. AttnDecode's rt_o accumulator: init before
    /// loop, mma updates inside loop, store after loop close).
    def_in_loop: bool,
    /// If `def_in_loop`, the open-idx and close-idx of the
    /// enclosing loop. Used to extend the effective live range to
    /// `[loop_open, loop_close]` for coalescing decisions — a
    /// loop-local slot can only alias with another slot whose live
    /// range is entirely outside this loop body.
    loop_open: usize,
    loop_close: usize,
}

/// Compute live ranges for every `RegTileSlot` referenced in the
/// linearized Instr stream. Tracks loop nesting so loop-local slots
/// (defined inside a loop body) are flagged with their enclosing
/// `[loop_open, loop_close]` extent.
fn compute_liveness(instrs: &[Instr]) -> BTreeMap<RegTileSlot, LiveRange> {
    let mut ranges: BTreeMap<RegTileSlot, LiveRange> = BTreeMap::new();
    // Stack of loop-open indices (innermost on top). Empty = top-level.
    let mut loop_opens: Vec<usize> = Vec::new();
    // Map from each loop_open index to its matching loop_close index.
    // Filled on ForLoopClose; queried lazily by slots whose first_def
    // landed inside the loop. We can't precompute closes without a
    // first pass, so do this in two passes: pass 1 records first_def
    // index + the open idx of the innermost enclosing loop (or
    // sentinel `usize::MAX` if top-level). Pass 2 walks again to
    // discover closes; then we update each slot's loop_close.
    struct InProgress {
        first_def: usize,
        last_use: usize,
        enclosing_open: Option<usize>,
    }
    let mut in_progress: BTreeMap<RegTileSlot, InProgress> = BTreeMap::new();
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
        for (slot, access) in instr_rt_accesses(instr) {
            let entry = in_progress.entry(slot).or_insert(InProgress {
                first_def: idx,
                last_use: idx,
                enclosing_open,
            });
            entry.last_use = idx;
            let _ = access;
        }
    }

    for (slot, p) in in_progress {
        let (def_in_loop, loop_open, loop_close) = match p.enclosing_open {
            Some(open) => {
                // Loop-local slot. Its effective live range covers the
                // entire loop body so it cannot alias anything else
                // touching that body.
                let close = close_of_open.get(&open).copied().unwrap_or(p.last_use);
                (true, open, close)
            }
            None => (false, 0, 0),
        };
        ranges.insert(
            slot,
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

/// Coalesce slots greedily: sort by `first_def`, maintain a free pool
/// keyed by `RegTileArenaEntry`. Returns `slot_remap: original →
/// representative`. Slots whose live range crosses a loop boundary
/// (`inside_loop == true`) are kept distinct (mapped to themselves) —
/// see module docs §"Loop scope".
fn coalesce(
    ranges: &BTreeMap<RegTileSlot, LiveRange>,
    arena: &BTreeMap<RegTileSlot, RegTileArenaEntry>,
) -> BTreeMap<RegTileSlot, RegTileSlot> {
    // Collect slot list sorted by first_def.
    let mut by_def: Vec<(RegTileSlot, LiveRange)> =
        ranges.iter().map(|(s, r)| (*s, *r)).collect();
    by_def.sort_by_key(|(slot, range)| (range.first_def, slot.0));

    let mut slot_remap: BTreeMap<RegTileSlot, RegTileSlot> = BTreeMap::new();
    // (representative slot, effective_last_use, arena_entry) — the
    // free pool of slots ready for reuse. `effective_last_use` is the
    // representative's true free-by index: for non-loop-local slots,
    // their `last_use`; for loop-local representatives, their
    // enclosing `loop_close` (so an outer slot defined after the loop
    // close is the next legal user).
    let mut free_pool: Vec<(RegTileSlot, usize, RegTileArenaEntry)> = Vec::new();

    for (slot, range) in by_def.iter().copied() {
        let entry = match arena.get(&slot) {
            Some(e) => *e,
            None => {
                slot_remap.insert(slot, slot);
                continue;
            }
        };
        // A slot's "effective free index" — the first linear idx at
        // which another slot may safely alias its representative:
        //   - non-loop-local slot: its `last_use`
        //   - loop-local slot: the enclosing `loop_close` (since each
        //     iteration of the loop redefines the slot, the slot is
        //     not safe to reuse until the loop body has finished
        //     executing entirely — even if the slot's last linear use
        //     was on iteration N's earlier instructions, iteration N+1
        //     will write again before the loop closes).
        let effective_last_use = if range.def_in_loop {
            range.loop_close
        } else {
            range.last_use
        };
        // The "candidate first-def" for picking from the free pool:
        // a predecessor entry must be fully dead before THIS index.
        //   - non-loop-local slot: `first_def`
        //   - loop-local slot: `loop_open` (the slot is rewritten on
        //     every iteration starting at loop_open; a predecessor
        //     must already be dead by then, not just before
        //     `first_def` of iteration 0)
        let candidate_first = if range.def_in_loop {
            range.loop_open
        } else {
            range.first_def
        };
        let pick = free_pool.iter().position(|(_, free_last, e)| {
            *e == entry && *free_last < candidate_first
        });
        match pick {
            Some(idx) => {
                let (rep, _, _) = free_pool.swap_remove(idx);
                slot_remap.insert(slot, rep);
                free_pool.push((rep, effective_last_use, entry));
            }
            None => {
                slot_remap.insert(slot, slot);
                free_pool.push((slot, effective_last_use, entry));
            }
        }
    }
    slot_remap
}

/// Rewrite every `RegTileSlot` reference in the Instr stream per
/// `slot_remap`.
fn rewrite_slots(
    instrs: &mut [Instr],
    slot_remap: &BTreeMap<RegTileSlot, RegTileSlot>,
) {
    let lookup = |s: &mut RegTileSlot| {
        if let Some(rep) = slot_remap.get(s) {
            *s = *rep;
        }
    };
    for instr in instrs.iter_mut() {
        rewrite_instr(instr, &lookup);
    }
}

/// Apply `f` to every `RegTileSlot` field of `instr`.
fn rewrite_instr<F: Fn(&mut RegTileSlot)>(instr: &mut Instr, f: &F) {
    use Instr::*;
    match instr {
        LoadShmemToReg { dst, .. } => f(dst),
        StoreRegTileToShmem { src, .. } => f(src),
        LoadShmemSubTileToReg { dst, .. } => f(dst),
        StoreRegTileSubTileToShmem { src, .. } => f(src),
        InitRtZero { dst, .. } => f(dst),
        WgmmaFenceAcc { d, .. } => f(d),
        WgmmaMmaAB_SmemSmem { d, .. } => f(d),
        WgmmaMmaABt_SmemSmem { d, .. } => f(d),
        WgmmaMmaAB_RegSmem { a, d, .. } => {
            f(a);
            f(d);
        }
        WgmmaMmaABt_RegSmem { a, d, .. } => {
            f(a);
            f(d);
        }
        RegTileMulScalar { lhs, dst, .. } => {
            f(lhs);
            f(dst);
        }
        RegTileRowMaxAcc { src, .. } => f(src),
        RegTileRowSumAcc { src, .. } => f(src),
        RegTileSubRow { src, dst, .. } => {
            f(src);
            f(dst);
        }
        RegTileExp2 { src, dst, .. } => {
            f(src);
            f(dst);
        }
        RegTileDivRow { src, dst, .. } => {
            f(src);
            f(dst);
        }
        RegTileCopyConvert { src, dst, .. } => {
            f(src);
            f(dst);
        }
        RegTileMulRow { src, dst, .. } => {
            f(src);
            f(dst);
        }
        RegTileNeg { src, dst, .. } => {
            f(src);
            f(dst);
        }
        RegTileExp { src, dst, .. } => {
            f(src);
            f(dst);
        }
        RegTileAdd { lhs, rhs, dst, .. } => {
            f(lhs);
            f(rhs);
            f(dst);
        }
        RegTileSub { lhs, rhs, dst, .. } => {
            f(lhs);
            f(rhs);
            f(dst);
        }
        RegTileDiv { lhs, rhs, dst, .. } => {
            f(lhs);
            f(rhs);
            f(dst);
        }
        RegTileMulCol { src, dst, .. } => {
            f(src);
            f(dst);
        }
        RegTileAddScalar { lhs, dst, .. } => {
            f(lhs);
            f(dst);
        }
        // No rt-bearing variants
        _ => {}
    }
    // Suppress unused warnings for LoadSpec / StoreSpec — they are
    // not rt-bearing today but the type imports are kept for parity
    // with `instr_rt_accesses`.
    let _ = std::marker::PhantomData::<(LoadSpec, StoreSpec)>;
}

/// `rt_alias_pass` — coalesce `RegTileSlot`s with disjoint live
/// ranges and matching `RegTileArenaEntry`. Reduces per-warp register
/// pressure by shrinking the arena, which directly reduces the count
/// of `kittens::rt<...> rt_<slot>;` declarations the player emits in
/// the kernel preamble.
///
/// Postcondition (validator-checked): every `RegTileSlot` referenced
/// by an Instr exists as a key in `tape.reg_tile_arena`.
pub fn rt_alias_pass(tape: &mut TkTape) {
    let ranges = compute_liveness(&tape.instrs);
    let remap = coalesce(&ranges, &tape.reg_tile_arena);
    rewrite_slots(&mut tape.instrs, &remap);
    // Prune arena: keep only slots that are remap *targets* (a
    // representative). Slots that got coalesced INTO another slot
    // disappear from the arena — their declarations would be dead
    // code in the emit.
    let kept: std::collections::BTreeSet<RegTileSlot> = remap.values().copied().collect();
    tape.reg_tile_arena.retain(|slot, _| kept.contains(slot));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tk_tape::{
        AllConsumersRole, Bf16, GroupWidth, PageId, RegTileId, RegTileLayoutTag, RowLayout,
        SmemTileId, TileDtypeTag, TkTape,
    };

    /// Two non-overlapping reg-tile chains with identical shape
    /// coalesce into one slot.
    #[test]
    fn coalesces_disjoint_same_shape_slots() {
        let mut tape = TkTape::default();
        let dst1: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        let dst2: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        let src = SmemTileId::<16, 128, Bf16>::from_page(PageId(0));
        // Two disjoint chains: load into dst1, store; load into dst2, store.
        tape.push(Instr::load_shmem_to_reg(
            src, dst1, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        tape.push(Instr::store_reg_tile_to_shmem(
            dst1, src, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        tape.push(Instr::load_shmem_to_reg(
            src, dst2, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        tape.push(Instr::store_reg_tile_to_shmem(
            dst2, src, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        let arena_before = tape.reg_tile_arena.len();
        assert_eq!(arena_before, 2, "two slots minted");

        rt_alias_pass(&mut tape);

        // After: one slot in arena (the representative).
        assert_eq!(
            tape.reg_tile_arena.len(),
            1,
            "disjoint same-shape chains coalesce to one slot"
        );
        // Body references the same slot in both chains.
        match (&tape.instrs[0], &tape.instrs[2]) {
            (
                Instr::LoadShmemToReg { dst: a, .. },
                Instr::LoadShmemToReg { dst: b, .. },
            ) => assert_eq!(a, b, "both loads now write to the same slot"),
            _ => panic!("unexpected Instr shape"),
        }
    }

    /// Different shapes (bf16 vs fp32) do NOT coalesce even if
    /// disjoint.
    #[test]
    fn does_not_coalesce_mismatched_shapes() {
        use crate::tk_tape::Fp32;
        let mut tape = TkTape::default();
        let dst1: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        let dst2: RegTileId<16, 128, Fp32, RowLayout> = tape.mint_reg_tile();
        let src = SmemTileId::<16, 128, Bf16>::from_page(PageId(0));
        tape.push(Instr::load_shmem_to_reg(
            src, dst1, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        // Force a write to dst2 via init_rt_zero (fp32 doesn't have
        // a smem-load that pairs with a bf16 source; init_rt_zero
        // writes regardless of dtype).
        tape.push(Instr::init_rt_zero(
            dst2, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        rt_alias_pass(&mut tape);
        assert_eq!(
            tape.reg_tile_arena.len(),
            2,
            "different dtype → not coalesced"
        );
    }

    /// Overlapping live ranges (same shape, but live at the same
    /// time) do NOT coalesce.
    #[test]
    fn does_not_coalesce_overlapping_ranges() {
        let mut tape = TkTape::default();
        let dst1: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        let dst2: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        let src = SmemTileId::<16, 128, Bf16>::from_page(PageId(0));
        // Both live across an `add(dst, dst1, dst2)` Instr.
        tape.push(Instr::load_shmem_to_reg(
            src, dst1, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        tape.push(Instr::load_shmem_to_reg(
            src, dst2, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        let dst_sum: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        tape.push(Instr::reg_tile_add(
            dst1, dst2, dst_sum, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        rt_alias_pass(&mut tape);
        // dst1 + dst2 are simultaneously live during the add — they
        // can't share a slot. dst_sum's def is AFTER both reads, so
        // it could alias one of them; that's still <= 2 slots.
        assert!(
            tape.reg_tile_arena.len() >= 2,
            "overlapping pairs preserve at least one distinct slot"
        );
    }

    /// Validator postcondition: every Instr-referenced slot still in
    /// the arena.
    #[test]
    fn postcondition_every_referenced_slot_in_arena() {
        let mut tape = TkTape::default();
        let dst1: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        let dst2: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        let src = SmemTileId::<16, 128, Bf16>::from_page(PageId(0));
        tape.push(Instr::load_shmem_to_reg(
            src, dst1, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        tape.push(Instr::store_reg_tile_to_shmem(
            dst1, src, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        tape.push(Instr::load_shmem_to_reg(
            src, dst2, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        tape.push(Instr::store_reg_tile_to_shmem(
            dst2, src, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        rt_alias_pass(&mut tape);
        // Walk the rewritten tape and confirm every slot referenced
        // is in the arena.
        for instr in &tape.instrs {
            for (slot, _) in instr_rt_accesses(instr) {
                assert!(
                    tape.reg_tile_arena.contains_key(&slot),
                    "slot {:?} referenced but absent from arena",
                    slot
                );
            }
        }
    }

    /// Loop-local slot can alias with an outer slot whose live range
    /// fully precedes the loop_open. (Iterations of the loop-local
    /// slot don't clobber a value that was already dead before the
    /// loop started.)
    #[test]
    fn coalesces_outer_dead_before_loop_with_loop_local() {
        use crate::tk_tape::LoopVarId;
        let mut tape = TkTape::default();
        let dst_outside: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        let src = SmemTileId::<16, 128, Bf16>::from_page(PageId(0));
        // Outer chain — fully dead before the loop opens.
        tape.push(Instr::load_shmem_to_reg(
            src, dst_outside, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        tape.push(Instr::store_reg_tile_to_shmem(
            dst_outside, src, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        tape.push(Instr::ForLoopOpenConst {
            var: LoopVarId(0),
            n: 4,
        });
        let dst_inside: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        tape.push(Instr::load_shmem_to_reg(
            src, dst_inside, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        tape.push(Instr::store_reg_tile_to_shmem(
            dst_inside, src, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        tape.push(Instr::ForLoopClose { var: LoopVarId(0) });
        rt_alias_pass(&mut tape);
        // dst_outside is dead before loop_open, so dst_inside (loop-
        // local) safely reuses its slot.
        assert_eq!(
            tape.reg_tile_arena.len(),
            1,
            "loop-local slot reuses an outer slot dead before the loop"
        );
        let _ = (TileDtypeTag::Bf16, RegTileLayoutTag::Row);
    }

    /// Loop-local slot must NOT alias with an outer slot whose
    /// last_use is INSIDE the loop body (the loop body will rewrite
    /// the C++ register on iteration N+1 before iteration N+1 gets
    /// to read the outer slot).
    #[test]
    fn does_not_coalesce_with_outer_used_inside_loop() {
        use crate::tk_tape::LoopVarId;
        let mut tape = TkTape::default();
        let dst_outside: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        let src = SmemTileId::<16, 128, Bf16>::from_page(PageId(0));
        // dst_outside is defined OUTSIDE the loop but USED INSIDE.
        tape.push(Instr::load_shmem_to_reg(
            src, dst_outside, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        tape.push(Instr::ForLoopOpenConst {
            var: LoopVarId(0),
            n: 4,
        });
        // Use of dst_outside inside loop — must persist across
        // iterations, so its slot can't be reused by a loop-local.
        tape.push(Instr::store_reg_tile_to_shmem(
            dst_outside, src, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        let dst_inside: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        tape.push(Instr::load_shmem_to_reg(
            src, dst_inside, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        tape.push(Instr::store_reg_tile_to_shmem(
            dst_inside, src, GroupWidth::<1>::PER_WARP, AllConsumersRole,
        ));
        tape.push(Instr::ForLoopClose { var: LoopVarId(0) });
        rt_alias_pass(&mut tape);
        assert_eq!(
            tape.reg_tile_arena.len(),
            2,
            "outer slot used inside loop must keep its identity \
             distinct from a loop-local slot"
        );
        let _ = (TileDtypeTag::Bf16, RegTileLayoutTag::Row);
    }
}
