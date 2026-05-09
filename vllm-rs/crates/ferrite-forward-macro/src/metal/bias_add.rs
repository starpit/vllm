// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal BiasAdd implementation adapter.
//!
//! Broadcast addition: out = input + bias (where bias is broadcast across batch/sequence dims)

use crate::classified::Program;
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, Resources, WeightAccessor,
    WorkloadConstraint, default_required_weights,
};
use crate::target::{Backend, TargetProfile};

/// Adapter for Metal BiasAdd operation.
///
/// Performs broadcast addition: out[..., i] = input[..., i] + bias[i]
/// Bias is typically 1D and broadcast across batch/sequence dimensions.
/// This is a memory-bound operation (2 reads + 1 write per element).
/// Often fused with Gemm as an epilogue operation.
#[derive(Debug)]
pub struct MetalBiasAddImpl {
    /// Data type (fp16 or bf16)
    dtype: &'static str,
}

impl MetalBiasAddImpl {
    pub fn new_fp16() -> Self {
        Self { dtype: "fp16" }
    }

    pub fn new_bf16() -> Self {
        Self { dtype: "bf16" }
    }

    /// Analytical cost model for BiasAdd (memory-bound operation).
    /// Cost = (bytes_input + bytes_bias + bytes_output) / bandwidth
    /// Bias is typically much smaller than input, so dominated by input/output traffic
    fn analytical_cost_us(&self, num_elements: u32, bandwidth_gbps: f64) -> f64 {
        let bytes_per_element = 2.0; // fp16/bf16
        let total_elements = num_elements as f64;

        // Input read + output write (bias is small and often cached)
        let bytes_read = total_elements * bytes_per_element; // input
        let bytes_written = total_elements * bytes_per_element; // output
        let total_bytes = bytes_read + bytes_written;

        let total_gb = total_bytes / 1e9;
        let time_seconds = total_gb / bandwidth_gbps;
        time_seconds * 1e6 // convert to microseconds
    }
}

impl Implementation for MetalBiasAddImpl {
    fn name(&self) -> &'static str {
        match self.dtype {
            "fp16" => "metal_bias_add_f16",
            "bf16" => "metal_bias_add_bf16",
            _ => "metal_bias_add",
        }
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // Only compatible with Metal targets
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    fn matches(&self, _fuf: &Fuf, _seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Standalone BiasAdd has no `Instruction<W>` variant — DSL
        // BiasAdds always fold into `FusedGemmBias` / `MeanSubRms-
        // NormBiasAdd` fusions on the CUDA side. Until a Metal
        // counterpart for those fusions lands, a lone BiasAdd
        // surfaces as `SolveError::UnclaimedTile` rather than
        // getting wired to a non-existent kernel.
        None
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let tile = m.claimed_tiles[0];
        let node = ctx.fuf.get(tile);

        // Get shape: BiasAdd operates on tensors of any shape
        let shape = &node.outputs[0];
        let dims = ctx.eval_shape(shape);

        if let Some(dims) = dims {
            // Calculate total number of elements
            let num_elements: u32 = dims.iter().copied().product::<u64>() as u32;

            // Try empirical cost first (if we have benchmarks for this size)
            let kernel_name = match self.dtype {
                "fp16" => "bias_add_f16",
                "bf16" => "bias_add_bf16",
                _ => "bias_add_f16",
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
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::target::from_metal_profile;

    #[test]
    fn metal_bias_add_only_compatible_with_metal_targets() {
        let metal_impl = MetalBiasAddImpl::new_fp16();

        // Metal target - should be compatible
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));

        // CUDA target - should NOT be compatible
        let cuda_profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!metal_impl.target_compatible(&cuda_profile));
    }

    #[test]
    fn metal_bias_add_analytical_cost_scales_with_size() {
        let impl_fp16 = MetalBiasAddImpl::new_fp16();

        // Small: 1K elements
        let small_cost = impl_fp16.analytical_cost_us(1_000, 68.25);

        // Large: 1M elements (1000× larger)
        let large_cost = impl_fp16.analytical_cost_us(1_000_000, 68.25);

        // Cost should scale linearly with size
        let ratio = large_cost / small_cost;
        assert!(
            (ratio - 1000.0).abs() < 50.0,
            "Expected ratio ~1000, got {}",
            ratio
        );
    }

    #[test]
    fn metal_bias_add_cost_lower_than_add() {
        let bias_add_impl = MetalBiasAddImpl::new_fp16();

        // BiasAdd has less memory traffic than Add
        // BiasAdd: 1 read (input) + 1 write (output) = 2× traffic (bias is small/cached)
        // Add: 2 reads + 1 write = 3× traffic
        let num_elements = 1_000_000;
        let bandwidth = 100.0; // GB/s

        let cost = bias_add_impl.analytical_cost_us(num_elements, bandwidth);

        // Expected: (1M*2 + 1M*2) bytes / 100 GB/s = 4 MB / 100 GB/s = 40 µs
        let expected = 40.0;
        assert!(
            (cost - expected).abs() < 1.0,
            "Expected ~{}µs, got {}µs",
            expected,
            cost
        );
    }

    #[test]
    fn metal_bias_add_bf16_same_cost_as_fp16() {
        let impl_fp16 = MetalBiasAddImpl::new_fp16();
        let impl_bf16 = MetalBiasAddImpl::new_bf16();

        let num_elements = 1_000_000;
        let bandwidth = 68.25;

        let cost_fp16 = impl_fp16.analytical_cost_us(num_elements, bandwidth);
        let cost_bf16 = impl_bf16.analytical_cost_us(num_elements, bandwidth);

        // Both are 2 bytes per element, so costs should be identical
        assert!((cost_fp16 - cost_bf16).abs() < 0.01);
    }
}
