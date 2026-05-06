// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal fused kernel implementation adapters.
//!
//! Wraps Metal fused kernels (Add+RMSNorm, Gate-Up-SiLU-Mul) to satisfy ferrite's Implementation trait.

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, FusedAddRmsNormImpl, FusedGateUpGeluMulImpl, FusedGateUpSiluMulImpl, Handoff,
    Implementation, LaunchKind, Layout, MatchInfo, OpInstance, OpcodeShape, Resources, SlotMap,
    WeightAccessor, WorkloadConstraint, default_required_weights,
};
use crate::target::{Backend, TargetProfile};

/// Adapter that wraps Metal Fused Add+RMSNorm to satisfy ferrite's Implementation trait.
///
/// This fusion eliminates one memory round-trip by computing the residual add and
/// normalization in a single pass: output = rmsnorm(input + residual, weight, eps)
#[derive(Debug)]
pub struct MetalFusedAddRmsNormImpl {
    /// Kernel name for cost table lookup
    kernel_name: &'static str,
    /// Data type (fp16 or bf16)
    dtype: &'static str,
}

impl MetalFusedAddRmsNormImpl {
    pub fn new_fp16() -> Self {
        Self {
            kernel_name: "fused_add_rmsnorm_f16",
            dtype: "fp16",
        }
    }

    pub fn new_bf16() -> Self {
        Self {
            kernel_name: "fused_add_rmsnorm_bf16",
            dtype: "bf16",
        }
    }

    /// Analytical cost model for fused Add+RMSNorm (memory-bound operation).
    /// Cost = (bytes_read + bytes_written) / bandwidth
    /// Reads: input + residual + weight
    /// Writes: output (+ optional residual_out)
    fn analytical_cost_us(
        &self,
        m: u32,
        n: u32,
        bandwidth_gbps: f64,
        has_residual_out: bool,
    ) -> f64 {
        let bytes_per_element = 2.0; // fp16/bf16
        let total_elements = (m as f64) * (n as f64);

        // Reads: input [M,N] + residual [M,N] + weight [N]
        let bytes_read = total_elements * bytes_per_element * 2.0 + (n as f64) * bytes_per_element;

        // Writes: output [M,N] + optional residual_out [M,N]
        let write_multiplier = if has_residual_out { 2.0 } else { 1.0 };
        let bytes_written = total_elements * bytes_per_element * write_multiplier;

        let total_bytes = bytes_read + bytes_written;
        let total_gb = total_bytes / 1e9;
        let time_seconds = total_gb / bandwidth_gbps;
        time_seconds * 1e6 // convert to microseconds
    }
}

impl Implementation for MetalFusedAddRmsNormImpl {
    fn name(&self) -> &'static str {
        match self.dtype {
            "fp16" => "metal_fused_add_rmsnorm_f16",
            "bf16" => "metal_fused_add_rmsnorm_bf16",
            _ => "metal_fused_add_rmsnorm",
        }
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // Only compatible with Metal targets
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        // Defer the (Add, RmsNorm) claim shape to the canonical CUDA
        // impl — same tile pattern, same boundary inputs, same
        // both-tiles-aliased output semantics. Only the kernel cost
        // model and the bound `target_compatible` differ between
        // backends.
        FusedAddRmsNormImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, match_info: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Get the RMSNorm tile (second in claimed_tiles)
        let rmsnorm_tile = match_info.claimed_tiles[1];
        let node = ctx.fuf.get(rmsnorm_tile);

        // Get shape: RMSNorm operates on [M, N]
        let shape = &node.outputs[0];
        let dims = ctx.eval_shape(shape);

        if let Some(dims) = dims {
            if dims.len() >= 2 {
                let m = dims[0] as u32;
                let n = dims[1] as u32;

                // Check if Add has multiple consumers (indicates residual_out needed)
                let add_tile = match_info.claimed_tiles[0];
                let has_residual_out = ctx
                    .fuf
                    .nodes
                    .iter()
                    .filter(|node| {
                        node.inputs.iter().any(|input| {
                            if let crate::fuf::FufInput::Tile { id, slot: _ } = input {
                                id == &add_tile
                            } else {
                                false
                            }
                        })
                    })
                    .count()
                    > 1;

                // Try empirical cost first
                let k = if has_residual_out { 1 } else { 0 };
                if let Some(cost) = ctx.profile.cost_us_for(self.kernel_name, m, n, k) {
                    return cost;
                }

                // Fall back to analytical model
                return self.analytical_cost_us(
                    m,
                    n,
                    ctx.profile.memory_bandwidth_gbps,
                    has_residual_out,
                );
            }
        }

        // Fallback: conservative estimate
        150.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0, // Metal uses threadgroup memory
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

    // `fused_add_rms_norm_inplace` mutates both the residual buffer
    // (Add output → updated residual) and the delta buffer (RmsNorm
    // output, normed-in-place); both outputs are TensorView aliases
    // of the upstream Add inputs. Mirror the CUDA contract so the
    // codegen drop pass preserves both upstreams correctly.
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        FusedAddRmsNormImpl.output_alias(claimed_tiles, fuf)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        FusedAddRmsNormImpl.opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        FusedAddRmsNormImpl.fan_out(m, fuf, program, bounds, slots)
    }
}

/// Adapter that wraps Metal Fused Gate-Up-SiLU-Mul (SwiGLU) to satisfy ferrite's Implementation trait.
///
/// This fusion computes: output = silu(gate) * up
/// Where: silu(x) = x * sigmoid(x)
///
/// Eliminates intermediate memory traffic by fusing the activation and multiplication.
#[derive(Debug)]
pub struct MetalFusedGateUpSiluMulImpl {
    /// Kernel name for cost table lookup
    kernel_name: &'static str,
    /// Data type (fp16 or bf16)
    dtype: &'static str,
    /// Whether this is GELU variant (for Gemma models)
    is_gelu: bool,
}

impl MetalFusedGateUpSiluMulImpl {
    pub fn new_fp16() -> Self {
        Self {
            kernel_name: "fused_gate_up_silu_mul_f16",
            dtype: "fp16",
            is_gelu: false,
        }
    }

    pub fn new_bf16() -> Self {
        Self {
            kernel_name: "fused_gate_up_silu_mul_bf16",
            dtype: "bf16",
            is_gelu: false,
        }
    }

    pub fn new_gelu_fp16() -> Self {
        Self {
            kernel_name: "fused_gate_up_gelu_mul_f16",
            dtype: "fp16",
            is_gelu: true,
        }
    }

    /// Analytical cost model for fused Gate-Up-SiLU-Mul (memory-bound operation).
    /// Cost = (bytes_read + bytes_written) / bandwidth
    /// Reads: gate [M,N] + up [M,N]
    /// Writes: output [M,N]
    fn analytical_cost_us(&self, m: u32, n: u32, bandwidth_gbps: f64) -> f64 {
        let bytes_per_element = 2.0; // fp16/bf16
        let total_elements = (m as f64) * (n as f64);

        // Reads: gate [M,N] + up [M,N]
        let bytes_read = total_elements * bytes_per_element * 2.0;

        // Writes: output [M,N]
        let bytes_written = total_elements * bytes_per_element;

        let total_bytes = bytes_read + bytes_written;
        let total_gb = total_bytes / 1e9;
        let time_seconds = total_gb / bandwidth_gbps;
        time_seconds * 1e6 // convert to microseconds
    }
}

impl Implementation for MetalFusedGateUpSiluMulImpl {
    fn name(&self) -> &'static str {
        if self.is_gelu {
            match self.dtype {
                "fp16" => "metal_fused_gate_up_gelu_mul_f16",
                "bf16" => "metal_fused_gate_up_gelu_mul_bf16",
                _ => "metal_fused_gate_up_gelu_mul",
            }
        } else {
            match self.dtype {
                "fp16" => "metal_fused_gate_up_silu_mul_f16",
                "bf16" => "metal_fused_gate_up_silu_mul_bf16",
                _ => "metal_fused_gate_up_silu_mul",
            }
        }
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // Only compatible with Metal targets
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        // Defer to the canonical CUDA matcher's 4-tile (Gemm, Gemm,
        // Silu/Gelu, Mul) claim shape — seeding on the gate Gemm and
        // walking forward. The Metal impl's prior 1-or-2-tile claim
        // missed the Gemms, breaking the fused weight accessor that
        // `required_weights` (and hence `fan_out`) depend on.
        if self.is_gelu {
            FusedGateUpGeluMulImpl.matches(fuf, seed, profile)
        } else {
            FusedGateUpSiluMulImpl.matches(fuf, seed, profile)
        }
    }

    fn cost_us(&self, match_info: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Get the Mul tile (last in claimed_tiles)
        let mul_tile = *match_info.claimed_tiles.last().unwrap();
        let node = ctx.fuf.get(mul_tile);

        // Get shape: Mul operates on [M, N]
        let shape = &node.outputs[0];
        let dims = ctx.eval_shape(shape);

        if let Some(dims) = dims {
            if dims.len() >= 2 {
                let m = dims[0] as u32;
                let n = dims[1] as u32;

                // Try empirical cost first
                if let Some(cost) = ctx.profile.cost_us_for(self.kernel_name, m, n, 0) {
                    return cost;
                }

                // Fall back to analytical model
                return self.analytical_cost_us(m, n, ctx.profile.memory_bandwidth_gbps);
            }
        }

        // Fallback: conservative estimate
        120.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0, // Metal uses threadgroup memory
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
        // The fused MLP needs ONE accessor whose source aggregates
        // both the gate_proj and up_proj weight refs (the loader packs
        // them into a single `[gate|up]` LinearLayer at runtime).
        // Delegate to the CUDA impls' override — the default
        // per-tile-input accessor would emit two separate accessors
        // and the kernel would fail to find a packed weight at runtime.
        if self.is_gelu {
            FusedGateUpGeluMulImpl.required_weights(claimed_tiles, fuf, program)
        } else {
            FusedGateUpSiluMulImpl.required_weights(claimed_tiles, fuf, program)
        }
    }

    fn opcode_shape(&self) -> OpcodeShape {
        if self.is_gelu {
            FusedGateUpGeluMulImpl.opcode_shape()
        } else {
            FusedGateUpSiluMulImpl.opcode_shape()
        }
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        if self.is_gelu {
            FusedGateUpGeluMulImpl.fan_out(m, fuf, program, bounds, slots)
        } else {
            FusedGateUpSiluMulImpl.fan_out(m, fuf, program, bounds, slots)
        }
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::target::from_metal_profile;

    #[test]
    fn metal_fused_add_rmsnorm_only_compatible_with_metal_targets() {
        let metal_impl = MetalFusedAddRmsNormImpl::new_fp16();

        // Metal target - should be compatible
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));

        // CUDA target - should NOT be compatible
        let cuda_profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!metal_impl.target_compatible(&cuda_profile));
    }

    #[test]
    fn metal_fused_add_rmsnorm_cost_accounts_for_residual_out() {
        let impl_fp16 = MetalFusedAddRmsNormImpl::new_fp16();

        // Without residual_out
        let cost_no_res = impl_fp16.analytical_cost_us(1024, 4096, 68.25, false);

        // With residual_out (extra write)
        let cost_with_res = impl_fp16.analytical_cost_us(1024, 4096, 68.25, true);

        // Cost with residual_out should be higher
        assert!(
            cost_with_res > cost_no_res,
            "Expected cost_with_res ({}) > cost_no_res ({})",
            cost_with_res,
            cost_no_res
        );

        // Should be roughly 1.33× (4 reads + 2 writes vs 4 reads + 1 write)
        let ratio = cost_with_res / cost_no_res;
        assert!(
            (ratio - 1.2).abs() < 0.2,
            "Expected ratio ~1.2-1.4, got {}",
            ratio
        );
    }

    #[test]
    fn metal_fused_gate_up_silu_mul_only_compatible_with_metal_targets() {
        let metal_impl = MetalFusedGateUpSiluMulImpl::new_fp16();

        // Metal target - should be compatible
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));

        // CUDA target - should NOT be compatible
        let cuda_profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!metal_impl.target_compatible(&cuda_profile));
    }

    #[test]
    fn metal_fused_gate_up_silu_mul_analytical_cost() {
        let impl_fp16 = MetalFusedGateUpSiluMulImpl::new_fp16();

        // Cost should scale with M*N (memory-bound)
        let cost_small = impl_fp16.analytical_cost_us(512, 2048, 68.25);
        let cost_large = impl_fp16.analytical_cost_us(1024, 4096, 68.25);

        // Large should be ~8× more expensive (2× M, 2× N = 4× elements, 2× for gate+up)
        let ratio = cost_large / cost_small;
        assert!(
            (ratio - 8.0).abs() < 1.0,
            "Expected ratio ~8.0, got {}",
            ratio
        );
    }
}
