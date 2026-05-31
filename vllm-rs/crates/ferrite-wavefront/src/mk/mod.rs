// SPDX-License-Identifier: Apache-2.0
//! Hopper-native fused-IType megakernel substrate (replaces the legacy
//! `tk_warp_ir` + `tk_codegen` + `tk_lower` String-emit stack).
//!
//! See `/Users/nickm/.claude/plans/resilient-honking-falcon.md` for the
//! design + phase plan. Phase 0 lands stubs only; orchestrator + per-
//! IType bodies follow in Phases 1-6.
//!
//! The substrate is gated end-to-end on `FERRITE_NEW_SUBSTRATE=1`. Off
//! by default → existing path is byte-identical to commit `c990b089bd`.

pub mod fusion;
pub mod instruction;
pub mod itype;
pub mod scheduler;
pub mod codegen;

/// Returns true if the new MK substrate is enabled via env var. The
/// orchestrator probes this once at macro-expansion time; the value is
/// baked into the per-canonical emit and is NOT re-read at runtime.
pub fn substrate_enabled() -> bool {
    std::env::var("FERRITE_NEW_SUBSTRATE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}
