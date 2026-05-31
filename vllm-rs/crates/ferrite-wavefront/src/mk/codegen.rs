// SPDX-License-Identifier: Apache-2.0
//! Per-canonical .cu emission for the new substrate. Phase 0: stubs.
//! Real `emit_cu` lands in Phases 1-6 alongside the per-IType .cuh
//! files. Until then, calling this module is gated on
//! `FERRITE_NEW_SUBSTRATE=1`; with the gate off, the macro stays on
//! the legacy `tk_codegen::emit_kernel` path.

use super::fusion::FusedOp;

#[derive(Debug)]
pub enum CodegenError {
    NotYetImplemented,
}

/// Phase 0 stub. Real implementation will:
///   1. write `MKGlobals_<canonical>` struct body (typed gl<>s for
///      every TensorKind the canonical's fused ops touch);
///   2. emit `INSTRUCTIONS_<canonical>[num_sms][max_q][N_INT]` static
///      table from the scheduler output;
///   3. emit `__global__ void <kernel>(MKGlobals_<canonical> g)` body
///      that #include's scaffold.cuh + per-IType .cuh files;
///   4. emit `extern "C" cudaError_t launch_<kernel>(...)` host
///      wrapper that sets `cudaFuncAttributeMaxDynamicSharedMemorySize`
///      and launches with persistent multi-CTA grid dim.
pub fn emit_cu(
    _canonical_stem: &str,
    _fused: &[FusedOp],
) -> Result<String, CodegenError> {
    Err(CodegenError::NotYetImplemented)
}
