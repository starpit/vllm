// SPDX-License-Identifier: Apache-2.0
//! `TkTape → TkTape` optimizer passes.
//!
//! Per `SUBTILE_IR_REDESIGN.md` §0 (architecture):
//!
//! > Two syntax-directed lowerings; a pipeline of `TkTape → TkTape`
//! > optimizer passes (shmem promotion, fence narrowing, page coalescing,
//! > …); two validators ... **Whether an edge can live in shmem is a
//! > target-specific question (smem capacity, mbar slot count,
//! > `NUM_CONSUMER_WARPS`, page-lifetime windows), so the analysis lives
//! > at TkTape — not at the target-agnostic SubtileTape, and not at
//! > lowering time. We are a compiler.**
//!
//! Each pass is a function `fn pass_name(tape: &mut TkTape)` that
//! mutates the tape in-place, idempotent, no return value. The
//! pass is correct by construction (typed witnesses where possible,
//! plus a validator postcondition the pass declares).
//!
//! After every pass, [`crate::tk_tape::validate_tk_tape`] runs to
//! re-check the tape's invariants. Per plan §3.2 "kill criterion
//! (push invariants up to types)": if a validator finds an invariant
//! the pass *should* have caught at the type system level, push the
//! invariant up.
//!
//! # Pass list (this commit)
//!
//! - [`rt_alias`]: register-tile slot coalescing. Reduces per-warp
//!   register pressure by sharing slots whose live ranges are
//!   disjoint and whose [`crate::tk_tape::RegTileArenaEntry`] is
//!   identical. Plan §6.5-class pass; the first concrete one to
//!   land.

pub mod rt_alias;
pub use rt_alias::rt_alias_pass;
