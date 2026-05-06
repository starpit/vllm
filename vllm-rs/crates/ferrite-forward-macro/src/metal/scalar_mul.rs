// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal ScalarMul implementation adapter.
//!
//! Elementwise scalar multiplication: out = input * scalar

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpInstance, OpcodeShape,
    Resources, ScalarMulImpl, SlotMap, WeightAccessor, WorkloadConstraint,
    default_required_weights,
};
use crate::target::{Backend, TargetProfile};

/// Adapter for Metal elementwise ScalarMul operation.
///
/// Performs elementwise scalar multiplication: out[i] = input[i] * scalar
/// This is a memory-bound operation (1 read + 1 write per element).
#[derive(Debug)]
pub struct MetalScalarMulImpl {
    /// Data type (fp16 or bf16)
    dtype: &'static str,
}

impl MetalScalarMulImpl {
    pub fn new_fp16() -> Self {
        Self { dtype: "fp16" }
    }

    pub fn new_bf16() -> Self {
        Self { dtype: "bf16" }
    }

    /// Analytical cost model for ScalarMul (memory-bound operation).
    /// Cost = (bytes_read + bytes_written) / bandwidth
    fn analytical_cost_us(&self, num_elements: u32, bandwidth_gbps: f64) -> f64 {
        let bytes_per_element = 2.0; // fp16/bf16
        let total_elements = num_elements as f64;
        let bytes_read = total_elements * bytes_per_element; // 1 input
        let bytes_written = total_elements * bytes_per_element; // 1 output
        let total_bytes = bytes_read + bytes_written;
        let total_gb = total_bytes / 1e9;
        let time_seconds = total_gb / bandwidth_gbps;
        time_seconds * 1e6 // convert to microseconds
    }
}

impl Implementation for MetalScalarMulImpl {
    fn name(&self) -> &'static str {
        match self.dtype {
            "fp16" => "metal_scalar_mul_f16",
            "bf16" => "metal_scalar_mul_bf16",
            _ => "metal_scalar_mul",
        }
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // Only compatible with Metal targets
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Mirror `ScalarMulImpl::matches`: claim singleton `OpKind::Mul`
        // tiles whose two inputs are exactly one Tile + one Scalar
        // (the `x * scale` pattern, e.g. Gemma's embedding scale).
        // Tile+Tile Muls inside SwiGLU/GeluMLP fuse via
        // `MetalFusedGateUpSiluMulImpl` and aren't claimed here.
        let node = fuf.get(seed);
        if node.op != OpKind::Mul || node.inputs.len() != 2 {
            return None;
        }
        let mut has_tile = false;
        let mut has_scalar = false;
        for inp in &node.inputs {
            match inp {
                FufInput::Tile { .. } => has_tile = true,
                FufInput::Scalar(_) => has_scalar = true,
                _ => return None,
            }
        }
        if !(has_tile && has_scalar) {
            return None;
        }
        let tile_input: Vec<TileId> = node
            .inputs
            .iter()
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: tile_input,
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let tile = m.claimed_tiles[0];
        let node = ctx.fuf.get(tile);
        
        // Get shape: ScalarMul operates on tensors of any shape
        let shape = &node.outputs[0];
        let dims = ctx.eval_shape(shape);
        
        if let Some(dims) = dims {
            // Calculate total number of elements
            let num_elements: u32 = dims.iter().copied().product::<u64>() as u32;
            
            // Try empirical cost first (if we have benchmarks for this size)
            let kernel_name = match self.dtype {
                "fp16" => "scalar_mul_f16",
                "bf16" => "scalar_mul_bf16",
                _ => "scalar_mul_f16",
            };
            
            if let Some(cost) = ctx.profile.cost_us_for(kernel_name, num_elements, 1, 0) {
                return cost;
            }
            
            // Fall back to analytical model
            return self.analytical_cost_us(num_elements, ctx.profile.memory_bandwidth_gbps);
        }
        
        // Fallback: conservative estimate
        10.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 8,
            threads_per_cta: 256,
        }
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder]
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder]
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_outputs.len()]
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        default_required_weights(claimed_tiles, fuf, program)
    }

    // Delegate the codegen-affecting hooks to `ScalarMulImpl`. Unity
    // passthrough (`x * 1.0` → alias of source) and the in-place
    // `take_owned → kernel → reinsert` pattern are codegen contracts —
    // mismatched semantics here would break the drop pass.
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        ScalarMulImpl.output_alias(claimed_tiles, fuf)
    }
    fn consumes_input_tiles(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<(TileId, u8)> {
        ScalarMulImpl.consumes_input_tiles(claimed_tiles, fuf)
    }
    fn opcode_shape(&self) -> OpcodeShape {
        ScalarMulImpl.opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        ScalarMulImpl.fan_out(m, fuf, program, bounds, slots)
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::target::from_metal_profile;

    #[test]
    fn metal_scalar_mul_only_compatible_with_metal_targets() {
        let metal_impl = MetalScalarMulImpl::new_fp16();
        
        // Metal target - should be compatible
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));
        
        // CUDA target - should NOT be compatible
        let cuda_profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!metal_impl.target_compatible(&cuda_profile));
    }

    #[test]
    fn metal_scalar_mul_analytical_cost_scales_with_bandwidth() {
        let impl_fp16 = MetalScalarMulImpl::new_fp16();
        
        // 1M elements
        let num_elements = 1_000_000;
        
        // M1: 68.25 GB/s
        let cost_m1 = impl_fp16.analytical_cost_us(num_elements, 68.25);
        
        // M2: 100 GB/s (1.46× faster)
        let cost_m2 = impl_fp16.analytical_cost_us(num_elements, 100.0);
        
        // Cost should be inversely proportional to bandwidth
        let ratio = cost_m1 / cost_m2;
        assert!((ratio - 1.46).abs() < 0.01, "Expected ratio ~1.46, got {}", ratio);
    }

    #[test]
    fn metal_scalar_mul_cost_is_memory_bound() {
        let impl_fp16 = MetalScalarMulImpl::new_fp16();
        
        // ScalarMul reads 1 input + writes 1 output = 2× memory traffic
        let num_elements = 1_000_000;
        let bandwidth = 100.0; // GB/s
        
        let cost = impl_fp16.analytical_cost_us(num_elements, bandwidth);
        
        // Expected: (1M*2 + 1M*2) bytes / 100 GB/s = 4 MB / 100 GB/s = 40 µs
        let expected = 40.0;
        assert!((cost - expected).abs() < 1.0, "Expected ~{}µs, got {}µs", expected, cost);
    }
}
