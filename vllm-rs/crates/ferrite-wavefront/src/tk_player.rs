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

use crate::tk_tape::{FenceScope, Instr, SyncScope, TkTape};

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

/// Single match dispatch — the entire player is this function. No
/// helper fns: a multi-step CUDA sequence is multiple Instrs in the
/// tape, never one Instr expanding to many lines.
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
        Instr::CommitGroup => {
            out.push_str("kittens::group<1>::tma::store_commit_group();\n");
        }
        Instr::WaitGroup { n } => {
            let _ = writeln!(out, "kittens::group<1>::tma::store_async_wait<{n}>();");
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
        #[allow(clippy::needless_borrows_for_generic_args)]
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
    use crate::tk_tape::{FenceScope, SyncScope};

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
    fn commit_group_emits_tk20_wrapper() {
        assert_eq!(
            emit(Instr::CommitGroup),
            "kittens::group<1>::tma::store_commit_group();\n"
        );
    }

    #[test]
    fn wait_group_zero_emits_tk20_wrapper() {
        assert_eq!(
            emit(Instr::WaitGroup { n: 0 }),
            "kittens::group<1>::tma::store_async_wait<0>();\n"
        );
    }

    #[test]
    fn wait_group_nonzero_emits_n() {
        assert_eq!(
            emit(Instr::WaitGroup { n: 3 }),
            "kittens::group<1>::tma::store_async_wait<3>();\n"
        );
    }

    /// A "fence" is NOT one Instr — it's a SEQUENCE of primitive
    /// Instrs. The walker pushes the five sync/commit/wait/fence/sync
    /// Instrs into the tape; the player has one one-line arm per
    /// primitive. There is no Instr::Fence; if the IR ever grew one,
    /// the player would gain a fat helper that re-invents what the
    /// IR was supposed to encode (the very bug class this redesign
    /// kills).
    #[test]
    fn cross_op_fence_is_a_sequence_of_primitive_instrs() {
        let tape = TkTape {
            instrs: vec![
                Instr::Syncthreads { scope: SyncScope::Cta },
                Instr::CommitGroup,
                Instr::WaitGroup { n: 0 },
                Instr::Threadfence { scope: FenceScope::Device },
                Instr::Syncthreads { scope: SyncScope::Cta },
            ],
            ..Default::default()
        };
        let body = emit_kernel(&tape)
            .strip_prefix("// emitted by tk_player\n")
            .unwrap()
            .to_string();
        let expected = concat!(
            "__syncthreads();\n",
            "kittens::group<1>::tma::store_commit_group();\n",
            "kittens::group<1>::tma::store_async_wait<0>();\n",
            "__threadfence();\n",
            "__syncthreads();\n",
        );
        assert_eq!(body, expected);
    }
}
