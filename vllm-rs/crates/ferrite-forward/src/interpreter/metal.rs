// SPDX-License-Identifier: Apache-2.0
//! Metal executor for ferrite-forward instructions.
//!
//! This module provides a runtime interpreter that walks the `Instruction<W>[]`
//! tape and records Metal ICB commands by calling the existing `record_*()`
//! methods from `ferrite-metal-kernels/src/instruction_executor/`.
//!
//! Architecture parallel to CUDA's `run()`:
//! - CUDA: walks tape, calls `Instruction::eval()` on each
//! - Metal: walks tape, calls `record_*()` methods, then executes ICB
//!
//! No new methods needed on `Instruction<W>` - all recording infrastructure
//! exists from Phase 4.6.

#![cfg(feature = "metal")]

use crate::{CanonicalParams, ForwardCtx, Instruction};
use ferrite_metal_kernels::device::MetalDevice;
use ferrite_metal_kernels::instruction_executor::RecordingContext;
use std::sync::Arc;

/// Metal executor that pre-records instructions into an ICB at init time,
/// then executes the ICB on each forward pass.
pub struct MetalExecutor<W: CanonicalParams> {
    device: Arc<MetalDevice>,
    recording_ctx: RecordingContext,
    weights: W,
    num_slots: usize,
}

impl<W: CanonicalParams> MetalExecutor<W> {
    /// Create a new Metal executor by recording the instruction tape into an ICB.
    ///
    /// # Arguments
    /// * `instructions` - The instruction tape from the solver
    /// * `weights` - The canonical weights (Metal buffers)
    /// * `num_slots` - Number of tile slots needed
    ///
    /// # Returns
    /// A new executor with pre-recorded ICB, or error if recording fails
    pub fn new(
        instructions: &[Instruction<W>],
        weights: W,
        num_slots: usize,
    ) -> Result<Self, String> {
        let device = ferrite_metal_kernels::detect_device()
            .ok_or("No Metal device found")?;
        
        let device_arc = Arc::new(device);
        let mut recording_ctx = RecordingContext::new(
            Arc::new(device_arc.device.clone()),
            instructions.len(),
        )?;

        // Walk instruction tape and record to ICB
        // TODO: Implement instruction recording loop
        // This will match on each instruction variant and call the corresponding
        // record_*() method from ferrite-metal-kernels/src/instruction_executor/
        
        for instr in instructions {
            match instr {
                // Phase 5.1: Start with basic instructions
                Instruction::RmsNorm(_out, _inp, _layer, _wt_fn) => {
                    // TODO: Call rmsnorm::record_rmsnorm()
                    return Err("RmsNorm recording not yet implemented".to_string());
                }
                Instruction::Gemm(_in, _out, _layer, _wt_fn, _n, _k) => {
                    // TODO: Call gemm::record_gemm()
                    return Err("Gemm recording not yet implemented".to_string());
                }
                Instruction::Add(_delta, _residual) => {
                    // TODO: Call elementwise::record_add()
                    return Err("Add recording not yet implemented".to_string());
                }
                Instruction::Reshape(_in, _out, _dims_lit, _dims_nt_pow, _dims_div_lit, _ndim) => {
                    // TODO: Call reshape::record_reshape()
                    return Err("Reshape recording not yet implemented".to_string());
                }
                Instruction::Embed(_out, _wt_fn) => {
                    // TODO: Call embed::record_embed()
                    return Err("Embed recording not yet implemented".to_string());
                }
                Instruction::ScalarMul(_in, _out, _scale) => {
                    // TODO: Call elementwise::record_scalar_mul()
                    return Err("ScalarMul recording not yet implemented".to_string());
                }
                Instruction::FusedAddRmsNorm(_delta, _residual, _layer, _wt_fn) => {
                    // TODO: Call fused::record_fused_add_rmsnorm()
                    return Err("FusedAddRmsNorm recording not yet implemented".to_string());
                }
                Instruction::RopeAppend(
                    _q, _k, _v, _q_out, _k_out, _v_out, _layer, _cos_sin_fn, _interleaved
                ) => {
                    // TODO: Call rope::record_rope_append()
                    return Err("RopeAppend recording not yet implemented".to_string());
                }
                
                // Unsupported instructions (will be added incrementally)
                _ => {
                    return Err(format!(
                        "Instruction variant not yet supported in Metal executor: {:?}",
                        std::any::type_name_of_val(instr)
                    ));
                }
            }
        }

        Ok(Self {
            device: device_arc,
            recording_ctx,
            weights,
            num_slots,
        })
    }

    /// Execute the pre-recorded ICB for one forward pass.
    ///
    /// # Arguments
    /// * `ctx` - Forward context with input tensors and runtime state
    ///
    /// # Returns
    /// The output tensor (logits or hidden states)
    pub fn forward(&self, _ctx: &ForwardCtx) -> Result<(), String> {
        // TODO: Implement forward execution
        // 1. Create tile table (Metal equivalent of Vec<Option<TileEntry>>)
        // 2. Bind runtime buffers (input_ids, positions, etc.)
        // 3. Execute ICB on compute encoder
        // 4. Return output tensor
        
        Err("forward() not yet implemented".to_string())
    }
}

// TODO: Implement tile table management for Metal
// This is the Metal equivalent of CUDA's Vec<Option<TileEntry>>
// Challenges:
// 1. ICB records buffer pointers at init time
// 2. Runtime buffers (input_ids, positions) come from ForwardCtx
// 3. Need to map slot indices to Metal buffers dynamically

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires Metal device
    fn test_metal_executor_creation() {
        // TODO: Add test with synthetic instruction tape
    }
}
