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
//! Sync/fence/commit/wait + ForLoop arms are live. PageBarrier ops,
//! LoadAsync/StoreAsync, Compute body templates land next as the
//! walker switches to push TkTape and the old `tk_codegen` emit path
//! is deleted.

use std::fmt::Write;

use crate::tk_tape::{Instr, TkTape};

// ── tk20 — typed wrappers around TK 2.0 / kittens::* primitives ─────
//
// Inline-defined now that tk_codegen.rs is gone. Stays a thin shim:
// each fn returns the textual CUDA fragment for one TK 2.0 primitive
// call. Per `feedback_dogfood_tk20_rust` no inline `kittens::*` strings
// outside this module.
mod tk20 {
    pub fn sync(n_warps: u32) -> String {
        format!("kittens::group<{n_warps}>::sync();")
    }

    pub fn group_tma_store_commit_group() -> String {
        "kittens::group<1>::tma::store_commit_group();".into()
    }

    pub fn group_tma_store_async_wait(n: u32) -> String {
        format!("kittens::group<1>::tma::store_async_wait<{n}>();")
    }
}

/// Emit the full CUDA kernel body from a [`TkTape`].
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
        // ── synchronization primitives — one Instr per CUDA primitive,
        //    one tk20 call per arm (per plan §3 step 8 + memory
        //    feedback_tk_player_one_call_per_arm).
        Instr::SyncthreadsCta { role: _ } => out.push_str("__syncthreads();\n"),
        Instr::SyncthreadsGroup { n_warps, role: _ } => {
            let _ = writeln!(out, "{}", tk20::sync(*n_warps));
        }
        Instr::ThreadfenceBlock { role: _ } => out.push_str("__threadfence_block();\n"),
        Instr::ThreadfenceDevice { role: _ } => out.push_str("__threadfence();\n"),
        Instr::ThreadfenceSystem { role: _ } => out.push_str("__threadfence_system();\n"),
        Instr::CommitGroupBulk { role: _ } => {
            let _ = writeln!(out, "{}", tk20::group_tma_store_commit_group());
        }
        Instr::WaitGroupBulk { n, role: _ } => {
            let _ = writeln!(out, "{}", tk20::group_tma_store_async_wait(*n));
        }

        // ── named barrier ops — full impls land with walker cutover.
        Instr::BarrierInit { .. } => {}
        Instr::PageBarrierWait { .. } => {}
        Instr::PageBarrierArrive { .. } => {}
        Instr::ArriveIfRuntimeEven { .. } => {}

        // ── memory ops ───────────────────────────────────────────
        Instr::LoadAsync(_spec) => {}
        Instr::StoreAsync(_spec) => {}
        Instr::StoreAsyncTyped { .. } => {}

        // ── compute — flat, one arm per architectural primitive.
        //    Full impls land with the walker cutover.
        Instr::RmsNorm { .. } => {}
        Instr::GemmM1 { .. } => {}
        Instr::SiluMul { .. } => {}
        Instr::ResidualAdd { .. } => {}
        Instr::RopeRotate { .. } => {}
        Instr::AttnDecodeInit { .. } => {}
        Instr::AttnDecodeQkt { .. } => {}
        Instr::AttnDecodeSv { .. } => {}
        Instr::AttnDecodeFinalise { .. } => {}
        Instr::DebugOpBeginMarker { .. } => {}

        // ── control flow — flat: open / body / close are separate Instrs.
        Instr::ForLoopOpenConst { var, n } => {
            let _ = writeln!(out, "for (uint v{0} = 0; v{0} < {1}u; ++v{0}) {{", var.0, n);
        }
        Instr::ForLoopOpenKernelArg { var, arg } => {
            let _ = writeln!(out, "for (uint v{0} = 0; v{0} < a{1}; ++v{0}) {{", var.0, arg.0);
        }
        Instr::ForLoopClose { var: _ } => out.push_str("}\n"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tk_tape::WarpRole;

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

    #[test]
    fn syncthreads_cta_matches_legacy() {
        assert_eq!(
            emit(Instr::SyncthreadsCta { role: WarpRole::All }),
            "__syncthreads();\n"
        );
    }

    #[test]
    fn syncthreads_group_emits_kittens_sync() {
        assert_eq!(
            emit(Instr::SyncthreadsGroup { n_warps: 8, role: WarpRole::All }),
            "kittens::group<8>::sync();\n"
        );
    }

    #[test]
    fn threadfence_device_matches_legacy() {
        assert_eq!(
            emit(Instr::ThreadfenceDevice { role: WarpRole::All }),
            "__threadfence();\n"
        );
    }

    #[test]
    fn threadfence_block_emits_block_scope() {
        assert_eq!(
            emit(Instr::ThreadfenceBlock { role: WarpRole::All }),
            "__threadfence_block();\n"
        );
    }

    #[test]
    fn threadfence_system_emits_system_scope() {
        assert_eq!(
            emit(Instr::ThreadfenceSystem { role: WarpRole::All }),
            "__threadfence_system();\n"
        );
    }

    #[test]
    fn commit_group_emits_tk20_wrapper() {
        assert_eq!(
            emit(Instr::CommitGroupBulk { role: WarpRole::All }),
            "kittens::group<1>::tma::store_commit_group();\n"
        );
    }

    #[test]
    fn wait_group_zero_emits_tk20_wrapper() {
        assert_eq!(
            emit(Instr::WaitGroupBulk { n: 0, role: WarpRole::All }),
            "kittens::group<1>::tma::store_async_wait<0>();\n"
        );
    }

    #[test]
    fn wait_group_nonzero_emits_n() {
        assert_eq!(
            emit(Instr::WaitGroupBulk { n: 3, role: WarpRole::All }),
            "kittens::group<1>::tma::store_async_wait<3>();\n"
        );
    }

    #[test]
    fn cross_op_fence_is_a_sequence_of_primitive_instrs() {
        let mut tape = TkTape::default();
        tape.emit_cross_op_gmem_fence();
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
