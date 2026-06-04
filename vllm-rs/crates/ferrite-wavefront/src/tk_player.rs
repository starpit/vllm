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

use crate::tk_tape::{
    CommitKind, FenceScope, Instr, SyncScope, TkTape, WaitMode,
};

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
        Instr::Syncthreads { scope } => match scope {
            SyncScope::Cta => out.push_str("__syncthreads();\n"),
            SyncScope::GroupOf(n) => {
                let _ = writeln!(out, "kittens::group<{n}>::sync();");
            }
        },
        Instr::Threadfence { scope } => match scope {
            FenceScope::Block => out.push_str("__threadfence_block();\n"),
            FenceScope::Device => out.push_str("__threadfence();\n"),
            FenceScope::System => out.push_str("__threadfence_system();\n"),
        },
        Instr::CommitGroup { kind } => match kind {
            CommitKind::BulkStore => {
                out.push_str("asm volatile(\"cp.async.bulk.commit_group;\");\n");
            }
            CommitKind::NonBulk => {
                out.push_str("asm volatile(\"cp.async.commit_group;\");\n");
            }
        },
        Instr::WaitGroup { kind, n } => match kind {
            CommitKind::BulkStore => {
                let _ = writeln!(out, "asm volatile(\"cp.async.bulk.wait_group {n};\");");
            }
            CommitKind::NonBulk => {
                let _ = writeln!(out, "asm volatile(\"cp.async.wait_group {n};\");");
            }
        },

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
    use crate::tk_tape::{CommitKind, FenceScope, SyncScope};

    fn emit(instr: Instr) -> String {
        let mut out = String::new();
        emit_instr(&mut out, &instr);
        out
    }

    #[test]
    fn empty_tape_emits_only_header() {
        let tape = TkTape::default();
        let out = emit_kernel(&tape);
        assert!(out.contains("// emitted by tk_player"));
        assert!(!out.contains("for ("));
    }

    // ── Phase 1: sync primitives ──────────────────────────────────
    //
    // Each arm must emit byte-identical CUDA to today's
    // tk_codegen.rs hardcoded strings — anything else regresses
    // the megakernel.

    #[test]
    fn syncthreads_cta_matches_legacy() {
        assert_eq!(
            emit(Instr::Syncthreads { scope: SyncScope::Cta }),
            "__syncthreads();\n"
        );
    }

    #[test]
    fn syncthreads_group_emits_kittens_sync() {
        assert_eq!(
            emit(Instr::Syncthreads { scope: SyncScope::GroupOf(8) }),
            "kittens::group<8>::sync();\n"
        );
    }

    #[test]
    fn threadfence_device_matches_legacy() {
        assert_eq!(
            emit(Instr::Threadfence { scope: FenceScope::Device }),
            "__threadfence();\n"
        );
    }

    #[test]
    fn threadfence_block_emits_block_scope() {
        assert_eq!(
            emit(Instr::Threadfence { scope: FenceScope::Block }),
            "__threadfence_block();\n"
        );
    }

    #[test]
    fn threadfence_system_emits_system_scope() {
        assert_eq!(
            emit(Instr::Threadfence { scope: FenceScope::System }),
            "__threadfence_system();\n"
        );
    }

    #[test]
    fn commit_group_bulk_matches_legacy() {
        // Legacy: tk_codegen.rs:69 / :157.
        assert_eq!(
            emit(Instr::CommitGroup { kind: CommitKind::BulkStore }),
            "asm volatile(\"cp.async.bulk.commit_group;\");\n"
        );
    }

    #[test]
    fn wait_group_bulk_zero_matches_legacy() {
        // Legacy: tk_codegen.rs:70 / :158 — wait_group 0 = drain
        // all.
        assert_eq!(
            emit(Instr::WaitGroup { kind: CommitKind::BulkStore, n: 0 }),
            "asm volatile(\"cp.async.bulk.wait_group 0;\");\n"
        );
    }

    #[test]
    fn wait_group_bulk_nonzero_emits_n() {
        assert_eq!(
            emit(Instr::WaitGroup { kind: CommitKind::BulkStore, n: 3 }),
            "asm volatile(\"cp.async.bulk.wait_group 3;\");\n"
        );
    }

    #[test]
    fn commit_and_wait_nonbulk_emit_sm80_path() {
        assert_eq!(
            emit(Instr::CommitGroup { kind: CommitKind::NonBulk }),
            "asm volatile(\"cp.async.commit_group;\");\n"
        );
        assert_eq!(
            emit(Instr::WaitGroup { kind: CommitKind::NonBulk, n: 0 }),
            "asm volatile(\"cp.async.wait_group 0;\");\n"
        );
    }

    /// Sequence test: a five-instruction tape reproducing today's
    /// `cross_op_gmem_fence_body` byte-for-byte. Phase 2 will
    /// consolidate this into a single `Instr::Fence` — for now we
    /// prove the building blocks compose.
    #[test]
    fn cross_op_fence_sequence_matches_legacy_body() {
        let tape = TkTape {
            instrs: vec![
                Instr::Syncthreads { scope: SyncScope::Cta },
                Instr::CommitGroup { kind: CommitKind::BulkStore },
                Instr::WaitGroup { kind: CommitKind::BulkStore, n: 0 },
                Instr::Threadfence { scope: FenceScope::Device },
                Instr::Syncthreads { scope: SyncScope::Cta },
            ],
            ..Default::default()
        };
        let out = emit_kernel(&tape);
        // Strip the header to compare just the body.
        let body = out.strip_prefix("// emitted by tk_player\n").unwrap();
        let expected = concat!(
            "__syncthreads();\n",
            "asm volatile(\"cp.async.bulk.commit_group;\");\n",
            "asm volatile(\"cp.async.bulk.wait_group 0;\");\n",
            "__threadfence();\n",
            "__syncthreads();\n",
        );
        assert_eq!(body, expected);
    }
}
