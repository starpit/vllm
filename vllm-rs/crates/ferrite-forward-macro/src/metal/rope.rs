// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal Implementation adapters for RoPE operations

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpcodeShape, Resources,
    RopeAppendRefImpl, SlotMap, WeightAccessor, WorkloadConstraint, default_required_weights,
};
use crate::target::{Backend, TargetProfile};

/// Metal implementation for RopeAppend (NeoX-style)
///
/// Applies rotary position embedding to query and key tensors.
/// NeoX style: pairs element i with i + half_dim.
///
/// Used by: Llama, GPT-NeoX, Mistral, Qwen, most models
#[derive(Debug)]
pub struct MetalRopeAppendImpl {
    dtype: &'static str, // "fp16" or "bf16"
}

impl MetalRopeAppendImpl {
    pub fn new_fp16() -> Self {
        Self { dtype: "fp16" }
    }

    pub fn new_bf16() -> Self {
        Self { dtype: "bf16" }
    }

    /// Analytical cost model for RoPE (memory-bound operation).
    /// Memory traffic: 2 reads (Q, K) + 1 read (cache) + 2 writes (Q', K')
    fn analytical_cost_us(&self, num_elements: u32, bandwidth_gbps: f64) -> f64 {
        let bytes_per_element = 2.0; // fp16/bf16
        let total_elements = num_elements as f64;

        // RoPE reads Q, K, cos_sin_cache and writes Q', K'
        // Assuming Q and K have same size
        let bytes_read = 3.0 * total_elements * bytes_per_element; // Q, K, cache
        let bytes_written = 2.0 * total_elements * bytes_per_element; // Q', K'
        let total_bytes = bytes_read + bytes_written;

        let total_gb = total_bytes / 1e9;
        let time_seconds = total_gb / bandwidth_gbps;

        // Add small compute overhead for rotation math (2 muls + 2 adds per pair)
        let num_pairs = num_elements / 2;
        let flops = (num_pairs * 4) as f64;
        let compute_overhead = flops / (400.0 * 1e12) * 0.1; // 10% of compute time

        (time_seconds + compute_overhead) * 1e6 // convert to microseconds
    }
}

impl Implementation for MetalRopeAppendImpl {
    fn name(&self) -> &'static str {
        match self.dtype {
            "fp16" => "metal_rope_append_f16",
            "bf16" => "metal_rope_append_bf16",
            _ => "metal_rope_append",
        }
    }

    fn kv_layer_io(
        &self,
        claimed_tiles: &[crate::fuf::TileId],
        fuf: &crate::fuf::Fuf,
    ) -> (Option<u32>, Option<u32>) {
        // RopeAppend writes the per-layer paged KV cache.
        (
            crate::impl_lib::kv_cache_extern_layer(claimed_tiles, fuf),
            None,
        )
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::RopeAppend {
            return None;
        }

        // Singleton claim - just this RopeAppend tile
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![],
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let tile = m.claimed_tiles[0];
        let node = ctx.fuf.get(tile);

        let shape = &node.outputs[0];
        let dims = ctx.eval_shape(shape);

        if let Some(dims) = dims {
            let num_elements: u32 = dims.iter().copied().product::<u64>() as u32;

            let kernel_name = match self.dtype {
                "fp16" => "rope_append_f16",
                "bf16" => "rope_append_bf16",
                _ => "rope_append_f16",
            };

            if let Some(cost) = ctx.profile.cost_us_for(kernel_name, num_elements, 1, 0) {
                return cost;
            }

            return self.analytical_cost_us(num_elements, ctx.profile.memory_bandwidth_gbps);
        }

        10.0 // conservative fallback
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 16,
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

    // RopeAppend's three outputs (q', k', v') are 3D `TensorView`
    // reshapes over the upstream Q/K/V buffers — no allocation, no
    // move. Mirror the CUDA contract.
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        RopeAppendRefImpl.output_alias(claimed_tiles, fuf)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        RopeAppendRefImpl.opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<ferrite_forward::Instruction>> {
        RopeAppendRefImpl.fan_out(m, fuf, program, bounds, slots)
    }

    fn as_atom(&self, _m: &MatchInfo, _fuf: &Fuf) -> Option<Box<dyn crate::atom::Atom>> {
        Some(Box::new(crate::atom_lib::RopeAppendAtom))
    }
}

/// Metal implementation for RopeAppendInterleaved (GPT-J style)
///
/// Applies rotary position embedding with interleaved pairing.
/// Interleaved style: pairs element 2i with 2i+1.
///
/// Used by: Cohere CommandR family
#[derive(Debug)]
pub struct MetalRopeAppendInterleavedImpl {
    dtype: &'static str,
}

impl MetalRopeAppendInterleavedImpl {
    pub fn new_fp16() -> Self {
        Self { dtype: "fp16" }
    }

    pub fn new_bf16() -> Self {
        Self { dtype: "bf16" }
    }

    fn analytical_cost_us(&self, num_elements: u32, bandwidth_gbps: f64) -> f64 {
        // Same cost model as standard RoPE - only element pairing differs
        let bytes_per_element = 2.0;
        let total_elements = num_elements as f64;
        let bytes_read = 3.0 * total_elements * bytes_per_element;
        let bytes_written = 2.0 * total_elements * bytes_per_element;
        let total_bytes = bytes_read + bytes_written;
        let total_gb = total_bytes / 1e9;
        let time_seconds = total_gb / bandwidth_gbps;
        let num_pairs = num_elements / 2;
        let flops = (num_pairs * 4) as f64;
        let compute_overhead = flops / (400.0 * 1e12) * 0.1;
        (time_seconds + compute_overhead) * 1e6
    }
}

impl Implementation for MetalRopeAppendInterleavedImpl {
    fn name(&self) -> &'static str {
        match self.dtype {
            "fp16" => "metal_rope_append_interleaved_f16",
            "bf16" => "metal_rope_append_interleaved_bf16",
            _ => "metal_rope_append_interleaved",
        }
    }

    fn kv_layer_io(
        &self,
        claimed_tiles: &[crate::fuf::TileId],
        fuf: &crate::fuf::Fuf,
    ) -> (Option<u32>, Option<u32>) {
        (
            crate::impl_lib::kv_cache_extern_layer(claimed_tiles, fuf),
            None,
        )
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::RopeAppendInterleaved {
            return None;
        }

        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![],
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let tile = m.claimed_tiles[0];
        let node = ctx.fuf.get(tile);
        let shape = &node.outputs[0];
        let dims = ctx.eval_shape(shape);

        if let Some(dims) = dims {
            let num_elements: u32 = dims.iter().copied().product::<u64>() as u32;
            let kernel_name = match self.dtype {
                "fp16" => "rope_append_interleaved_f16",
                "bf16" => "rope_append_interleaved_bf16",
                _ => "rope_append_interleaved_f16",
            };

            if let Some(cost) = ctx.profile.cost_us_for(kernel_name, num_elements, 1, 0) {
                return cost;
            }

            return self.analytical_cost_us(num_elements, ctx.profile.memory_bandwidth_gbps);
        }

        10.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 16,
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

    // Same alias contract + same `Instruction::RopeAppend` shape as
    // the NeoX variant; the `interleaved: bool` field on the variant
    // distinguishes them at codegen time, so one shape covers both.
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        RopeAppendRefImpl.output_alias(claimed_tiles, fuf)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        RopeAppendRefImpl.opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<ferrite_forward::Instruction>> {
        RopeAppendRefImpl.fan_out(m, fuf, program, bounds, slots)
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::target::from_metal_profile;

    #[test]
    fn metal_rope_only_compatible_with_metal_targets() {
        let metal_impl = MetalRopeAppendImpl::new_fp16();
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));

        let cuda_profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!metal_impl.target_compatible(&cuda_profile));
    }

    #[test]
    fn metal_rope_interleaved_only_compatible_with_metal() {
        let impl_interleaved = MetalRopeAppendInterleavedImpl::new_fp16();
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(impl_interleaved.target_compatible(&metal_profile));
    }

    #[test]
    fn metal_rope_analytical_cost_scales_with_size() {
        let impl_fp16 = MetalRopeAppendImpl::new_fp16();

        // Small: 4096 elements
        let small_cost = impl_fp16.analytical_cost_us(4096, 68.25);

        // Large: 262144 elements (64x larger)
        let large_cost = impl_fp16.analytical_cost_us(262144, 68.25);

        // Cost should scale roughly linearly with size
        let ratio = large_cost / small_cost;
        assert!(ratio > 50.0 && ratio < 80.0, "Cost ratio: {}", ratio);

        // Costs should be in reasonable microsecond range
        assert!(
            small_cost > 0.1 && small_cost < 100.0,
            "Small cost: {}",
            small_cost
        );
        assert!(
            large_cost > 1.0 && large_cost < 10000.0,
            "Large cost: {}",
            large_cost
        );
    }

    #[test]
    fn metal_rope_cost_accounts_for_memory_traffic() {
        let impl_fp16 = MetalRopeAppendImpl::new_fp16();

        // RoPE reads 3 buffers (Q, K, cache) + writes 2 (Q', K') = 5× traffic
        // vs Add which reads 2 + writes 1 = 3× traffic
        let num_elements = 1_000_000;
        let bandwidth = 100.0; // GB/s

        let cost = impl_fp16.analytical_cost_us(num_elements, bandwidth);

        // Expected: (3*1M*2 + 2*1M*2) bytes / 100 GB/s = 10 MB / 100 GB/s = 100 µs
        let expected = 100.0;
        assert!(
            (cost - expected).abs() < 10.0,
            "Expected ~{}µs, got {}µs",
            expected,
            cost
        );
    }
}
