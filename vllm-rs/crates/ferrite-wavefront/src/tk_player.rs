// SPDX-License-Identifier: Apache-2.0
//! `tk_player` — the dumb transcription emitter for [`crate::tk_tape::TkTape`].
//!
//! # Contract (from `SUBTILE_IR_REDESIGN.md` §3)
//!
//! - **One match arm per [`Instr`] kind**, ≤5 lines of `format!()`.
//! - **No ambient state lookups.** Every `{...}` placeholder is a
//!   field on the instruction.
//! - **No conditionals beyond the kind dispatch.** Role gating is one
//!   `if (__role == ...)` line per Instr; no other branching.
//! - **No formula computation.** `head_dim * elem`, `slot * row_bytes`,
//!   etc. are tape-time fields, not emit-time arithmetic.
//! - **Hard line budget**: this file ≤800 lines. If we exceed, the IR
//!   is wrong: split an [`Instr`] kind into more granular kinds.
//!
//! # Status
//!
//! Phase 0: this file emits an empty kernel from any [`TkTape`]. The
//! match dispatch is a stub; no live callers. Phases 1+ fill in arms
//! as Instr kinds migrate from `tk_codegen.rs`.

use std::fmt::Write;

use crate::tk_tape::{Instr, TkTape};

/// Emit the full CUDA kernel body from a [`TkTape`].
///
/// Phase 0: returns an empty kernel skeleton; subsequent phases
/// populate arms as Instr kinds migrate. The walker in
/// `tk_lower.rs` does NOT call this yet — the existing `tk_codegen`
/// emit path remains in use.
pub fn emit_kernel(tape: &TkTape) -> String {
    let mut out = String::new();
    out.push_str("// emitted by tk_player\n");
    for instr in &tape.instrs {
        emit_instr(&mut out, instr);
    }
    out
}

/// Single match dispatch — the entire player is this function plus
/// a few small projection helpers.
fn emit_instr(out: &mut String, instr: &Instr) {
    match instr {
        // ── synchronization primitives ───────────────────────────
        Instr::Syncthreads { .. } => {
            // Phase 1: emit __syncthreads() (or scoped variant).
        }
        Instr::Threadfence { .. } => {
            // Phase 1: emit __threadfence{,_block,_system}().
        }
        Instr::CommitGroup { .. } => {
            // Phase 1: emit cp.async.bulk.commit_group;.
        }
        Instr::WaitGroup { .. } => {
            // Phase 1: emit cp.async.bulk.wait_group N;.
        }

        // ── consolidated cross-op + kernel-end fence ─────────────
        Instr::Fence(_spec) => {
            // Phase 2: emit pre-sync? + commit_group + wait_group +
            // threadfence + post-sync? per FenceSpec fields.
        }

        // ── named barrier ops ────────────────────────────────────
        Instr::BarrierInit { .. } => {
            // Phase 7: mbarrier::init equivalent.
        }
        Instr::BarrierWait { .. } => {
            // Phase 7: kittens::wait(barrier, parity).
        }
        Instr::BarrierArrive { .. } => {
            // Phase 7: kittens::arrive(barrier).
        }

        // ── memory ops ───────────────────────────────────────────
        Instr::LoadAsync(_spec) => {
            // Phase 7: tma::expect_bytes + tma::load_async pair.
        }
        Instr::StoreAsync(_spec) => {
            // Phase 7: tma::store_async (+ optional inline
            // commit/wait per StoreCommitStrategy).
        }

        // ── compute body ─────────────────────────────────────────
        Instr::Compute { .. } => {
            // Phase 7: lookup compute_body_template(body_id),
            // format with fields on Instr.
        }

        // ── control flow ─────────────────────────────────────────
        Instr::ForLoop { var, count, body } => {
            let _ = writeln!(
                out,
                "for (uint v{}_= 0; v{}_ < a{}; ++v{}_) {{",
                var.0,
                var.0,
                count.0,
                var.0,
            );
            for inner in body {
                emit_instr(out, inner);
            }
            out.push_str("}\n");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tape_emits_only_header() {
        let tape = TkTape::default();
        let out = emit_kernel(&tape);
        assert!(out.contains("// emitted by tk_player"));
        assert!(!out.contains("for ("));
    }
}
