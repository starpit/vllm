// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal TanhSoftCap implementation adapter.
//!
//! Logit capping: out = cap * tanh(input / cap)
//! Used in Gemma2 to prevent logits from growing too large

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpInstance, OpcodeShape,
    Resources, SlotMap, TanhSoftCapImpl, WeightAccessor, WorkloadConstraint,
    default_required_weights,
};
use crate::target::{Backend, TargetProfile};

/// Adapter for Metal TanhSoftCap operation.
///
/// Performs logit capping: out[i] = cap * tanh(input[i] / cap)
/// This prevents logits from growing unboundedly large.
/// Used in Gemma2 architecture.
/// This is a compute-bound operation (tanh is expensive).
#[derive(Debug)]
pub struct MetalTanhSoftCapImpl {
    /// Data type (fp16 or bf16)
    dtype: &'static str,
}

impl MetalTanhSoftCapImpl {
    pub fn new_fp16() -> Self {
        Self { dtype: "fp16" }
    }

    pub fn new_bf16() -> Self {
        Self { dtype: "bf16" }
    }

    /// Analytical cost model for TanhSoftCap (compute-bound operation).
    /// Tanh requires ~20-30 FLOPs (exp, div, add, sub)
    /// Cost = (bytes_read + bytes_written) / bandwidth + compute_time
    fn analytical_cost_us(&self, num_elements: u32, bandwidth_gbps: f64, peak_tflops: f64) -> f64 {
        let bytes_per_element = 2.0; // fp16/bf16
        let total_elements = num_elements as f64;

        // Memory traffic: 1 read + 1 write
        let bytes_read = total_elements * bytes_per_element;
        let bytes_written = total_elements * bytes_per_element;
        let total_bytes = bytes_read + bytes_written;
        let total_gb = total_bytes / 1e9;
        let memory_time_seconds = total_gb / bandwidth_gbps;

        // Compute: ~25 FLOPs per element for tanh + scaling
        let flops = total_elements * 25.0;
        let compute_time_seconds = flops / (peak_tflops * 1e12);

        // Take max of memory and compute time (compute-bound)
        let time_seconds = memory_time_seconds.max(compute_time_seconds);
        time_seconds * 1e6 // convert to microseconds
    }
}

impl Implementation for MetalTanhSoftCapImpl {
    fn name(&self) -> &'static str {
        match self.dtype {
            "fp16" => "metal_tanh_softcap_f16",
            "bf16" => "metal_tanh_softcap_bf16",
            _ => "metal_tanh_softcap",
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
        if node.op != OpKind::TanhSoftCap {
            return None;
        }

        // Singleton claim - just this TanhSoftCap tile
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![],
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let tile = m.claimed_tiles[0];
        let node = ctx.fuf.get(tile);

        // Get shape: TanhSoftCap operates on tensors of any shape
        let shape = &node.outputs[0];
        let dims = ctx.eval_shape(shape);

        if let Some(dims) = dims {
            // Calculate total number of elements
            let num_elements: u32 = dims.iter().copied().product::<u64>() as u32;

            // Try empirical cost first (if we have benchmarks for this size)
            let kernel_name = match self.dtype {
                "fp16" => "tanh_softcap_f16",
                "bf16" => "tanh_softcap_bf16",
                _ => "tanh_softcap_f16",
            };

            if let Some(cost) = ctx.profile.cost_us_for(kernel_name, num_elements, 1, 0) {
                return cost;
            }

            // Fall back to analytical model
            return self.analytical_cost_us(
                num_elements,
                ctx.profile.memory_bandwidth_gbps,
                ctx.profile.peak_tflops_fp16,
            );
        }

        // Fallback: conservative estimate
        10.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 16, // More registers for tanh computation
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

    fn is_compute_bound(&self) -> bool {
        true // Tanh is compute-intensive
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        default_required_weights(claimed_tiles, fuf, program)
    }

    // `tanh_softcap_inplace` is consume-style — `take_owned → mutate
    // → reinsert`. Mirror the CUDA contract so the codegen drop pass
    // doesn't double-free the upstream buffer.
    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        TanhSoftCapImpl.consumes_input_tiles(claimed_tiles, fuf)
    }
    fn opcode_shape(&self) -> OpcodeShape {
        TanhSoftCapImpl.opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        TanhSoftCapImpl.fan_out(m, fuf, program, bounds, slots)
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::target::from_metal_profile;

    #[test]
    fn metal_tanh_softcap_only_compatible_with_metal_targets() {
        let metal_impl = MetalTanhSoftCapImpl::new_fp16();

        // Metal target - should be compatible
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));

        // CUDA target - should NOT be compatible
        let cuda_profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!metal_impl.target_compatible(&cuda_profile));
    }

    #[test]
    fn metal_tanh_softcap_is_compute_bound() {
        let impl_fp16 = MetalTanhSoftCapImpl::new_fp16();
        assert!(impl_fp16.is_compute_bound());
    }

    #[test]
    fn metal_tanh_softcap_cost_higher_than_simple_ops() {
        let impl_fp16 = MetalTanhSoftCapImpl::new_fp16();

        let num_elements = 1_000_000;
        let bandwidth = 100.0; // GB/s
        let peak_tflops = 10.0; // TFLOPS

        let cost = impl_fp16.analytical_cost_us(num_elements, bandwidth, peak_tflops);

        // TanhSoftCap should be more expensive than simple elementwise ops
        // due to tanh computation (25 FLOPs vs 1 FLOP for add/mul)
        // Expected: max(memory_time, compute_time)
        // Memory: 4 MB / 100 GB/s = 40 µs
        // Compute: 25M FLOPs / 10 TFLOPS = 2.5 µs
        // Result: max(40, 2.5) = 40 µs (memory-bound on fast hardware)
        assert!(
            cost > 30.0 && cost < 100.0,
            "Expected 30-100µs, got {}µs",
            cost
        );
    }

    #[test]
    fn metal_tanh_softcap_compute_bound_on_slow_compute() {
        let impl_fp16 = MetalTanhSoftCapImpl::new_fp16();

        let num_elements = 1_000_000;
        let bandwidth = 100.0; // GB/s (fast memory)
        let peak_tflops = 1.0; // TFLOPS (slow compute)

        let cost = impl_fp16.analytical_cost_us(num_elements, bandwidth, peak_tflops);

        // With slow compute, should be compute-bound
        // Memory: 4 MB / 100 GB/s = 40 µs
        // Compute: 25M FLOPs / 1 TFLOPS = 25 µs
        // Result: max(40, 25) = 40 µs (still memory-bound, but closer)

        // Now with even slower compute
        let slow_cost = impl_fp16.analytical_cost_us(num_elements, bandwidth, 0.1);
        // Compute: 25M FLOPs / 0.1 TFLOPS = 250 µs
        // Result: max(40, 250) = 250 µs (compute-bound)
        assert!(
            slow_cost > 200.0,
            "Expected >200µs on slow compute, got {}µs",
            slow_cost
        );
    }
}
