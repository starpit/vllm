// SPDX-License-Identifier: Apache-2.0
//! Fusion pass — walks `LoweringInput.ops` and groups sequences into
//! the fused-IType invocations that the MK substrate emits.
//!
//! Phase 0: stubs only. Real fusion-pass arms land in Phases 2-6 in
//! lockstep with the per-IType .cuh files.

use crate::lower::LoweringInput;

#[derive(Debug, Clone)]
pub enum FusedOp {
    /// Fallback marker — fusion couldn't reduce this op into a known
    /// IType. The macro emits the existing skeleton-`None` body for
    /// the canonical and the worker hook silently falls back to per-op
    /// forward.
    Unfused { reason: &'static str },
}

#[derive(Debug)]
pub enum FusionError {
    /// Phase 0 stub — every input gets this until Phase 2 starts
    /// matching real op sequences.
    NotYetImplemented,
}

pub fn fuse(_input: &LoweringInput) -> Result<Vec<FusedOp>, FusionError> {
    Err(FusionError::NotYetImplemented)
}
