// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal Reshape implementation adapter.
//!
//! Reshape is a metadata-only operation that doesn't require actual computation.
//! It just changes the view of the tensor without moving data.

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpcodeShape, ReshapeRefImpl,
    Resources, SlotMap, WeightAccessor, WorkloadConstraint, default_required_weights,
};
use crate::target::{Backend, TargetProfile};

/// Adapter for Metal Reshape operation.
///
/// Reshape is a view operation that changes tensor dimensions without copying data.
/// Cost is essentially zero (just metadata manipulation).
#[derive(Debug)]
pub struct MetalReshapeImpl;

impl MetalReshapeImpl {
    pub fn new() -> Self {
        Self
    }

    /// Fixed cost for reshape operation (metadata-only, no data movement).
    /// Reshape is a view operation that just changes tensor dimensions
    /// without copying data, so cost is essentially zero.
    fn analytical_cost_us(&self) -> f64 {
        0.1 // microseconds - negligible fixed cost
    }
}

impl Implementation for MetalReshapeImpl {
    fn name(&self) -> &'static str {
        "metal_reshape"
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
        if node.op != OpKind::Reshape {
            return None;
        }

        // Singleton claim - just this Reshape tile
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![],
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        self.analytical_cost_us()
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
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

    // Reshape's output is a `TensorView` aliasing the upstream tile.
    // Mirror `ReshapeRefImpl::output_alias` so the codegen drop pass
    // keeps the upstream `OwnedTensor` alive while any consumer of
    // this view is live. Without this override the default `None`
    // alias would drop the upstream prematurely.
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        ReshapeRefImpl.output_alias(claimed_tiles, fuf)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        ReshapeRefImpl.opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<ferrite_forward::Instruction>> {
        ReshapeRefImpl.fan_out(m, fuf, program, bounds, slots)
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::target::from_metal_profile;

    #[test]
    fn metal_reshape_only_compatible_with_metal_targets() {
        let metal_impl = MetalReshapeImpl::new();

        // Metal target - should be compatible
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));

        // CUDA target - should NOT be compatible
        let cuda_profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!metal_impl.target_compatible(&cuda_profile));
    }

    #[test]
    fn metal_reshape_has_fixed_negligible_cost() {
        let metal_impl = MetalReshapeImpl::new();

        // Reshape is metadata-only (no data movement), so cost is fixed
        let cost = metal_impl.analytical_cost_us();

        // Verify cost is negligible (< 1µs)
        assert!(cost < 1.0, "Reshape cost should be < 1µs, got {}µs", cost);

        // Verify it's the expected fixed value
        assert_eq!(cost, 0.1, "Reshape should have fixed 0.1µs cost");
    }
}
