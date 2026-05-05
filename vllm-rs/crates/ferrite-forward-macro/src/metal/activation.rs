// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal Activation implementation adapter.
//!
//! Wraps Metal activation function kernels (SiLU, GELU, FatReLU, etc.)
//! to satisfy ferrite's Implementation trait.

use crate::classified::{OpKind, Program};
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, Resources,
    WeightAccessor, WorkloadConstraint, default_required_weights,
};
use crate::target::{Backend, TargetProfile};

/// Adapter that wraps Metal Activation kernels to satisfy ferrite's Implementation trait.
///
/// Supports multiple activation functions:
/// - SiLU (Swish): x * sigmoid(x)
/// - GELU: Gaussian Error Linear Unit (tanh approximation)
/// - GELU Tanh: Explicit tanh variant
/// - GELU Quick: Fast sigmoid approximation
/// - FatReLU: ReLU with threshold parameter
#[derive(Debug)]
pub struct MetalActivationImpl {
    /// Kernel name for cost table lookup
    kernel_name: &'static str,
    /// Data type (fp16, bf16, or fp32)
    dtype: &'static str,
    /// Activation function type
    activation_type: ActivationType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActivationType {
    Silu,
    Gelu,
    GeluTanh,
    GeluQuick,
    FatRelu,
}

impl MetalActivationImpl {
    /// SiLU activation (fp16)
    pub fn new_silu_fp16() -> Self {
        Self {
            kernel_name: "silu_f16",
            dtype: "fp16",
            activation_type: ActivationType::Silu,
        }
    }

    /// GELU activation (fp16, tanh approximation)
    pub fn new_gelu_fp16() -> Self {
        Self {
            kernel_name: "gelu_f16",
            dtype: "fp16",
            activation_type: ActivationType::Gelu,
        }
    }

    /// GELU Tanh activation (fp16)
    pub fn new_gelu_tanh_fp16() -> Self {
        Self {
            kernel_name: "gelu_tanh_f16",
            dtype: "fp16",
            activation_type: ActivationType::GeluTanh,
        }
    }

    /// GELU Quick activation (fp16, fast sigmoid approximation)
    pub fn new_gelu_quick_fp16() -> Self {
        Self {
            kernel_name: "gelu_quick_f16",
            dtype: "fp16",
            activation_type: ActivationType::GeluQuick,
        }
    }

    /// FatReLU activation (fp16)
    pub fn new_fatrelu_fp16() -> Self {
        Self {
            kernel_name: "fatrelu_f16",
            dtype: "fp16",
            activation_type: ActivationType::FatRelu,
        }
    }

    /// Analytical cost model for activation functions (memory-bound).
    /// Cost = (bytes_read + bytes_written) / bandwidth
    fn analytical_cost_us(&self, num_elements: u64, bandwidth_gbps: f64) -> f64 {
        let bytes_per_element = 2.0; // fp16/bf16
        let bytes_read = (num_elements as f64) * bytes_per_element;
        let bytes_written = (num_elements as f64) * bytes_per_element;
        let total_bytes = bytes_read + bytes_written;
        let total_gb = total_bytes / 1e9;
        let time_seconds = total_gb / bandwidth_gbps;
        time_seconds * 1e6 // convert to microseconds
    }

    /// Get the OpKind that this activation impl matches
    fn matches_op_kind(&self) -> OpKind {
        match self.activation_type {
            ActivationType::Silu => OpKind::Silu,
            ActivationType::Gelu | ActivationType::GeluTanh | ActivationType::GeluQuick => OpKind::Gelu,
            ActivationType::FatRelu => OpKind::TanhSoftCap, // FatReLU uses TanhSoftCap opcode
        }
    }
}

impl Implementation for MetalActivationImpl {
    fn name(&self) -> &'static str {
        match (self.activation_type, self.dtype) {
            (ActivationType::Silu, "fp16") => "metal_silu_f16",
            (ActivationType::Gelu, "fp16") => "metal_gelu_f16",
            (ActivationType::GeluTanh, "fp16") => "metal_gelu_tanh_f16",
            (ActivationType::GeluQuick, "fp16") => "metal_gelu_quick_f16",
            (ActivationType::FatRelu, "fp16") => "metal_fatrelu_f16",
            _ => "metal_activation",
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
        
        // Match the appropriate OpKind for this activation type
        if node.op != self.matches_op_kind() {
            return None;
        }

        // Singleton claim - just this activation tile
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![],
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let tile = m.claimed_tiles[0];
        let node = ctx.fuf.get(tile);
        
        // Get shape: Activation is shape-preserving
        let shape = &node.outputs[0];
        let dims = ctx.eval_shape(shape);
        
        if let Some(dims) = dims {
            // Calculate total number of elements
            let num_elements: u64 = dims.iter().product();
            
            // Try empirical cost first (using first two dims as M, N)
            if dims.len() >= 2 {
                let m = dims[0] as u32;
                let n = dims[1] as u32;
                if let Some(cost) = ctx.profile.cost_us_for(self.kernel_name, m, n, 0) {
                    return cost;
                }
            }
            
            // Fall back to analytical model
            return self.analytical_cost_us(num_elements, ctx.profile.memory_bandwidth_gbps);
        }
        
        // Fallback: conservative estimate
        50.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0, // Metal uses threadgroup memory
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
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::target::from_metal_profile;

    #[test]
    fn metal_activation_only_compatible_with_metal_targets() {
        let metal_impl = MetalActivationImpl::new_silu_fp16();
        
        // Metal target - should be compatible
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));
        
        // CUDA target - should NOT be compatible
        let cuda_profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!metal_impl.target_compatible(&cuda_profile));
    }

    #[test]
    fn metal_activation_analytical_cost_scales_with_size() {
        let impl_fp16 = MetalActivationImpl::new_silu_fp16();
        
        // Small tensor
        let cost_small = impl_fp16.analytical_cost_us(1024 * 1024, 400.0);
        
        // Large tensor (4× bigger)
        let cost_large = impl_fp16.analytical_cost_us(4 * 1024 * 1024, 400.0);
        
        // Cost should scale linearly with size
        let ratio = cost_large / cost_small;
        assert!((ratio - 4.0).abs() < 0.1, "Expected ratio ~4.0, got {}", ratio);
    }

    #[test]
    fn metal_activation_matches_correct_op_kind() {
        let silu_impl = MetalActivationImpl::new_silu_fp16();
        assert_eq!(silu_impl.matches_op_kind(), OpKind::Silu);
        
        let gelu_impl = MetalActivationImpl::new_gelu_fp16();
        assert_eq!(gelu_impl.matches_op_kind(), OpKind::Gelu);
        
        let fatrelu_impl = MetalActivationImpl::new_fatrelu_fp16();
        assert_eq!(fatrelu_impl.matches_op_kind(), OpKind::TanhSoftCap);
    }
}
