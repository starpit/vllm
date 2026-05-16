// SPDX-License-Identifier: Apache-2.0
//! Host-interpreter tape executor — universal fallback.
//!
//! Runs any tape by dispatching `Instruction::eval` per op at
//! runtime. No compile-time artifacts: the existing per-bucket
//! forward fn emitted by [`crate::codegen::emit_model`] already
//! does the work. This claimer exists so the tape-level DP has
//! an always-matching baseline to compare against specialised
//! executors (e.g. [`crate::tape::tk_mega::TkMegaTapeClaimer`]).

use crate::impl_lib::OpInstance;
use crate::solver::WorkloadPoint;
use crate::tape_claim::{TapeClaimer, TapeEmission, TapeEmitCtx, TapeMatchInfo};

#[derive(Debug)]
pub struct HostInterpreterTapeClaimer;

/// No evidence needed — host matches any tape. Empty unit type
/// still satisfies the `TapeMatchInfo` bound.
#[derive(Debug)]
pub struct HostMatch;

impl TapeClaimer for HostInterpreterTapeClaimer {
    fn name(&self) -> &'static str {
        "host"
    }

    fn matches(
        &self,
        _backbone: &[OpInstance],
        _lm_head: &[OpInstance],
        _ctx: &TapeEmitCtx<'_>,
    ) -> Option<Box<dyn TapeMatchInfo>> {
        Some(Box::new(HostMatch))
    }

    fn cost_us(
        &self,
        _backbone: &[OpInstance],
        _lm_head: &[OpInstance],
        _ctx: &TapeEmitCtx<'_>,
        _info: &dyn TapeMatchInfo,
    ) -> f64 {
        // High sentinel cost so any specialised executor that can
        // claim the tape outcompetes host. The exact number is
        // not measured — it just has to be higher than any real
        // executor's estimate.
        1.0e12
    }

    fn emit(
        &self,
        _canonical_name: &str,
        _wp: WorkloadPoint,
        _backbone: &[OpInstance],
        _lm_head: &[OpInstance],
        _terminal_slot: u32,
        _ctx: &TapeEmitCtx<'_>,
        _info: &dyn TapeMatchInfo,
    ) -> TapeEmission {
        // No compile-time artifacts — the existing per-bucket
        // host forward fn in `emit_model` carries this executor.
        TapeEmission::none()
    }
}
