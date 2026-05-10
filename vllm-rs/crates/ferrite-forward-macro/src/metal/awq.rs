// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal AWQ (Activation-aware Weight Quantization) implementation adapter.
//!
//! Wraps Metal AWQ dequantization kernels to satisfy ferrite's Implementation trait.

use crate::classified::{OpKind, Program};
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, Resources, WeightAccessor,
    WorkloadConstraint, default_required_weights,
};
use crate::target::{Backend, TargetProfile};

/// Adapter that wraps Metal AWQ dequantization kernels to satisfy ferrite's Implementation trait.
///
/// AWQ uses 4-bit weight quantization with group-wise scaling and zero-point.
/// This implementation handles dequantization from INT4 to FP16/BF16 format.
#[derive(Debug)]
pub struct MetalAwqImpl {
    /// Kernel name for cost table lookup
    kernel_name: &'static str,
    /// Data type (fp16 or bf16)
    dtype: &'static str,
    /// Group size for quantization (typically 128)
    group_size: u32,
}

impl MetalAwqImpl {
    /// AWQ dequantization (fp16, group_size=128)
    pub fn new_fp16_g128() -> Self {
        Self {
            kernel_name: "awq_dequantize_f16_g128",
            dtype: "fp16",
            group_size: 128,
        }
    }

    /// AWQ dequantization (bf16, group_size=128)
    pub fn new_bf16_g128() -> Self {
        Self {
            kernel_name: "awq_dequantize_bf16_g128",
            dtype: "bf16",
            group_size: 128,
        }
    }

    /// Analytical cost model for AWQ dequantization (memory-bound).
    ///
    /// Cost = (bytes_read + bytes_written) / bandwidth
    /// Reads: quantized weights (4-bit) + scales (fp16) + zeros (fp16)
    /// Writes: dequantized weights (fp16/bf16)
    fn analytical_cost_us(&self, m: u32, n: u32, bandwidth_gbps: f64) -> f64 {
        let bytes_per_element = 2.0; // fp16/bf16
        let total_elements = (m as f64) * (n as f64);

        // Reads:
        // - Quantized weights: 4 bits per element = 0.5 bytes
        // - Scales: 1 per group = (total_elements / group_size) * 2 bytes
        // - Zeros: 1 per group = (total_elements / group_size) * 2 bytes
        let quantized_bytes = total_elements * 0.5;
        let num_groups = total_elements / (self.group_size as f64);
        let scales_bytes = num_groups * bytes_per_element;
        let zeros_bytes = num_groups * bytes_per_element;
        let bytes_read = quantized_bytes + scales_bytes + zeros_bytes;

        // Writes: dequantized weights (fp16/bf16)
        let bytes_written = total_elements * bytes_per_element;

        let total_bytes = bytes_read + bytes_written;
        let total_gb = total_bytes / 1e9;
        let time_seconds = total_gb / bandwidth_gbps;
        time_seconds * 1e6 // convert to microseconds
    }
}

impl Implementation for MetalAwqImpl {
    fn name(&self) -> &'static str {
        match (self.dtype, self.group_size) {
            ("fp16", 128) => "metal_awq_dequantize_f16_g128",
            ("bf16", 128) => "metal_awq_dequantize_bf16_g128",
            _ => "metal_awq_dequantize",
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

        // AWQ dequantization is typically part of a GEMM operation
        // For now, we don't match standalone dequantization tiles
        // This will be integrated with MetalGemmImpl for quantized weights
        if node.op != OpKind::Gemm {
            return None;
        }

        // TODO: Check if this GEMM uses AWQ-quantized weights
        // For now, return None (AWQ integration happens in GEMM impl)
        None
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let tile = m.claimed_tiles[0];
        let node = ctx.fuf.get(tile);

        // Get shape: GEMM output is [M, N]
        let shape = &node.outputs[0];
        let dims = ctx.eval_shape(shape);

        if let Some(dims) = dims
            && dims.len() >= 2 {
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
        200.0
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
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::target::from_metal_profile;

    #[test]
    fn metal_awq_only_compatible_with_metal_targets() {
        let metal_impl = MetalAwqImpl::new_fp16_g128();

        // Metal target - should be compatible
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));

        // CUDA target - should NOT be compatible
        let cuda_profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!metal_impl.target_compatible(&cuda_profile));
    }

    #[test]
    fn metal_awq_analytical_cost_accounts_for_compression() {
        let impl_fp16 = MetalAwqImpl::new_fp16_g128();

        // AWQ dequantization should be cheaper than full FP16 read
        // because quantized weights are 4-bit (8× smaller)
        let cost_awq = impl_fp16.analytical_cost_us(1024, 4096, 400.0);

        // Equivalent FP16 read would be: (1024*4096*2 read + 1024*4096*2 write) / 400e9 * 1e6
        let fp16_bytes = (1024.0 * 4096.0 * 2.0 * 2.0) / 1e9;
        let cost_fp16 = (fp16_bytes / 400.0) * 1e6;

        // AWQ should be significantly cheaper (quantized weights are 8× smaller)
        assert!(
            cost_awq < cost_fp16 * 0.7,
            "Expected AWQ cost ({}) < 70% of FP16 cost ({})",
            cost_awq,
            cost_fp16
        );
    }

    #[test]
    fn metal_awq_analytical_cost_scales_with_size() {
        let impl_fp16 = MetalAwqImpl::new_fp16_g128();

        // Small matrix
        let cost_small = impl_fp16.analytical_cost_us(512, 2048, 400.0);

        // Large matrix (4× bigger)
        let cost_large = impl_fp16.analytical_cost_us(1024, 4096, 400.0);

        // Cost should scale roughly linearly with size
        let ratio = cost_large / cost_small;
        assert!(
            (ratio - 4.0).abs() < 0.5,
            "Expected ratio ~4.0, got {}",
            ratio
        );
    }
}
