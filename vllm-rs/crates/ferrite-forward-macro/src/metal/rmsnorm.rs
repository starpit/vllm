// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal RMSNorm implementation adapter.
//!
//! Wraps Metal RMSNorm kernels to satisfy ferrite's Implementation trait.

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpcodeShape, Resources,
    RmsNormRefImpl, SlotMap, WeightAccessor, WorkloadConstraint, default_required_weights,
};
use crate::target::{Backend, TargetProfile};

/// Adapter that wraps a Metal RMSNorm implementation to satisfy ferrite's Implementation trait.
///
/// This allows Metal RMSNorm kernels to be registered in the solver's implementation library
/// alongside CUDA kernels, with the solver selecting the appropriate backend based on the
/// target profile.
#[derive(Debug)]
pub struct MetalRmsNormImpl {
    /// Kernel name for cost table lookup
    kernel_name: &'static str,
    /// Data type (fp16 or bf16)
    dtype: &'static str,
}

impl MetalRmsNormImpl {
    pub fn new_fp16() -> Self {
        Self {
            kernel_name: "rmsnorm_f16",
            dtype: "fp16",
        }
    }

    pub fn new_bf16() -> Self {
        Self {
            kernel_name: "rmsnorm_bf16",
            dtype: "bf16",
        }
    }

    /// Analytical cost model for RMSNorm (memory-bound operation).
    /// Cost = (bytes_read + bytes_written) / bandwidth
    fn analytical_cost_us(&self, m: u32, n: u32, bandwidth_gbps: f64) -> f64 {
        let bytes_per_element = 2.0; // fp16/bf16
        let total_elements = (m as f64) * (n as f64);
        let bytes_read = total_elements * bytes_per_element; // input
        let bytes_written = total_elements * bytes_per_element; // output
        let total_bytes = bytes_read + bytes_written;
        let total_gb = total_bytes / 1e9;
        let time_seconds = total_gb / bandwidth_gbps;
        time_seconds * 1e6 // convert to microseconds
    }
}

impl Implementation for MetalRmsNormImpl {
    fn name(&self) -> &'static str {
        match self.dtype {
            "fp16" => "metal_rmsnorm_f16",
            "bf16" => "metal_rmsnorm_bf16",
            _ => "metal_rmsnorm",
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
        let node = fuf.get(seed);
        if node.op != OpKind::RmsNorm {
            return None;
        }

        // Singleton claim - just this RMSNorm tile
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![],
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let tile = m.claimed_tiles[0];
        let node = ctx.fuf.get(tile);

        // Get shape: RMSNorm operates on [M, N] where M=num_tokens, N=hidden_size
        let shape = &node.outputs[0];
        let dims = ctx.eval_shape(shape);

        if let Some(dims) = dims
            && dims.len() >= 2
        {
            let m = dims[0] as u32;
            let n = dims[1] as u32;

            // Try empirical cost first
            if let Some(cost) = ctx.profile.cost_us_for(self.kernel_name, m, n, 0) {
                return cost;
            }

            // Fall back to analytical model
            return self.analytical_cost_us(m, n, ctx.profile.memory_bandwidth_gbps);
        }

        // Fallback: conservative estimate
        100.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0, // Metal uses threadgroup memory, not CUDA shmem
            regs_per_thread: 32,
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

    // Delegate to the CUDA `RmsNormRefImpl` for opcode shape + fan_out.
    // `Instruction::RmsNorm { in_slot, out_slot, layer, weight_fn }` is
    // shared between backends; the emission depends only on the FUF
    // structure, not on cuda/metal.
    fn opcode_shape(&self) -> OpcodeShape {
        RmsNormRefImpl.opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<ferrite_forward::Instruction>> {
        RmsNormRefImpl.fan_out(m, fuf, program, bounds, slots)
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::target::from_metal_profile;

    #[test]
    fn metal_rmsnorm_only_compatible_with_metal_targets() {
        let metal_impl = MetalRmsNormImpl::new_fp16();

        // Metal target - should be compatible
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));

        // CUDA target - should NOT be compatible
        let cuda_profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!metal_impl.target_compatible(&cuda_profile));
    }

    #[test]
    fn metal_rmsnorm_analytical_cost_scales_with_bandwidth() {
        let impl_fp16 = MetalRmsNormImpl::new_fp16();

        // M1: 68.25 GB/s
        let cost_m1 = impl_fp16.analytical_cost_us(1024, 4096, 68.25);

        // M2: 100 GB/s (1.46× faster)
        let cost_m2 = impl_fp16.analytical_cost_us(1024, 4096, 100.0);

        // Cost should be inversely proportional to bandwidth
        let ratio = cost_m1 / cost_m2;
        assert!(
            (ratio - 1.46).abs() < 0.01,
            "Expected ratio ~1.46, got {}",
            ratio
        );
    }
}
