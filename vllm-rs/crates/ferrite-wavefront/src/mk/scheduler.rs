// SPDX-License-Identifier: Apache-2.0
//! Persistent multi-CTA scheduler — packs `Vec<FusedOp>` into a
//! `[num_sms, max_per_sm_instructions, sizeof(Instruction)/4]` int32
//! tensor that the kernel reads via `gl<instruction_t, ...>`.
//!
//! Phase 0: stubs only. Real scheduler arms land in Phases 4-6.

use super::fusion::FusedOp;
use super::instruction::Instruction;

#[derive(Debug)]
pub enum SchedulerError {
    NotYetImplemented,
}

pub fn pack(_fused: &[FusedOp], _num_sms: u32) -> Result<Vec<Vec<Instruction>>, SchedulerError> {
    Err(SchedulerError::NotYetImplemented)
}
