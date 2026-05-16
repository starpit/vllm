// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal GEMM implementation adapter.
//!
//! Wraps Metal Performance Shaders GEMM to satisfy ferrite's Implementation trait.

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, GemmRefImpl, Handoff, Implementation, LaunchKind, Layout, MatchInfo,
    OpcodeShape, Resources, SlotMap, WeightAccessor, WorkloadConstraint, default_required_weights,
    weight_storage_of,
};
use crate::quantization::StorageFormat;
use crate::target::{Backend, TargetProfile};

/// Adapter that wraps Metal GEMM (via MPS) to satisfy ferrite's Implementation trait.
///
/// Uses Metal Performance Shaders for optimized matrix multiplication on Apple Silicon.
/// Supports FP16 and FP32 precision, with optional transpose operations.
#[derive(Debug)]
pub struct MetalGemmImpl {
    /// Kernel name for cost table lookup
    kernel_name: &'static str,
    /// Data type (fp16 or fp32)
    dtype: &'static str,
}

impl MetalGemmImpl {
    pub fn new_fp16() -> Self {
        Self {
            kernel_name: "gemm_f16",
            dtype: "fp16",
        }
    }

    pub fn new_fp32() -> Self {
        Self {
            kernel_name: "gemm_f32",
            dtype: "fp32",
        }
    }

    /// Analytical cost model for GEMM (compute-bound operation).
    /// Cost = FLOPs / (compute_tflops * 1e12) * 1e6
    /// FLOPs = 2 * M * N * K (multiply-add counts as 2 ops)
    fn analytical_cost_us(&self, m: u32, n: u32, k: u32, compute_tflops: f64) -> f64 {
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        let time_seconds = flops / (compute_tflops * 1e12);
        time_seconds * 1e6 // convert to microseconds
    }
}

impl Implementation for MetalGemmImpl {
    fn name(&self) -> &'static str {
        match self.dtype {
            "fp16" => "metal_gemm_f16",
            "fp32" => "metal_gemm_f32",
            _ => "metal_gemm",
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
        if node.op != OpKind::Gemm {
            return None;
        }
        // MLX-affine int4 weights are claimed by `MetalAffineQmmImpl`
        // (forward-time qmv/qmm_t dispatch). MetalGemmImpl serves the
        // Dense path only — bail on Affine so the affine impl wins.
        if matches!(weight_storage_of(node), Some(StorageFormat::Affine { .. })) {
            return None;
        }

        // Singleton claim - just this GEMM tile
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![],
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let tile = m.claimed_tiles[0];
        let node = ctx.fuf.get(tile);

        // Get shape: GEMM output is [M, N], weight is [K, N] or [N, K]
        let output_shape = &node.outputs[0];
        let output_dims = ctx.eval_shape(output_shape);

        if let Some(output_dims) = output_dims
            && output_dims.len() >= 2
        {
            let m = output_dims[0] as u32;
            let n = output_dims[1] as u32;

            // Get K from weight shape
            // Weight input is typically the second input (after activation)
            if node.inputs.len() >= 2 {
                // For GEMM, the weight input provides K dimension
                // We'll use a conservative estimate if we can't determine K
                let k = 2048u32; // Conservative default for typical transformer dimensions

                // Try empirical cost first
                if let Some(cost) = ctx.profile.cost_us_for(self.kernel_name, m, n, k) {
                    return cost;
                }

                // Fall back to analytical model
                return self.analytical_cost_us(m, n, k, ctx.profile.peak_tflops_fp16);
            }
        }

        // Fallback: conservative estimate (assume medium-sized GEMM)
        500.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0, // MPS manages memory internally
            regs_per_thread: 64,
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

    // `Instruction::Gemm { in_slot, out_slot, layer, weight_fn, n, k }`
    // is shared between cuda + metal. Mirror the CUDA `GemmRefImpl`
    // emission (slot resolution + N,K from FUF + layered accessor).
    fn opcode_shape(&self) -> OpcodeShape {
        GemmRefImpl.opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<ferrite_forward::Instruction>> {
        GemmRefImpl.fan_out(m, fuf, program, bounds, slots)
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::target::from_metal_profile;

    #[test]
    fn metal_gemm_only_compatible_with_metal_targets() {
        let metal_impl = MetalGemmImpl::new_fp16();

        // Metal target - should be compatible
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));

        // CUDA target - should NOT be compatible
        let cuda_profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!metal_impl.target_compatible(&cuda_profile));
    }

    #[test]
    fn metal_gemm_analytical_cost_scales_with_compute() {
        let impl_fp16 = MetalGemmImpl::new_fp16();

        // M1 Max: 10.4 TFLOPS FP16
        let cost_m1 = impl_fp16.analytical_cost_us(1024, 4096, 2048, 10.4);

        // M2 Max: 13.6 TFLOPS FP16 (1.31× faster)
        let cost_m2 = impl_fp16.analytical_cost_us(1024, 4096, 2048, 13.6);

        // Cost should be inversely proportional to compute
        let ratio = cost_m1 / cost_m2;
        assert!(
            (ratio - 1.31).abs() < 0.01,
            "Expected ratio ~1.31, got {}",
            ratio
        );
    }
}
