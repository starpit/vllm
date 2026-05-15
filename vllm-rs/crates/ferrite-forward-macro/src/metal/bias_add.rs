// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal BiasAdd implementation adapter.
//!
//! Broadcast addition: out = input + bias (where bias is broadcast across batch/sequence dims)

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::codegen::split_base_layer;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpInstance, OpcodeShape,
    Resources, SlotMap, WeightAccessor, WorkloadConstraint, first_weight_ref, fused_accessor_name,
    gemm_nk_from_fuf, weight_storage_of,
};
use crate::quantization::StorageFormat;
use crate::target::{Backend, TargetProfile};

use quote::quote;

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

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::BiasAdd {
            return None;
        }
        // Bias-after-gemm singleton claim: first input is the upstream
        // Gemm's output tile, second input is the linear-layer bias
        // weight ref. Reject lone BiasAdds whose upstream is anything
        // else — there's no production DSL site for that shape today
        // and the LinearLayer accessor (built off the upstream Gemm)
        // is what wires `<prefix>.bias` / affine `linear_bias`.
        let upstream_id = match node.inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            _ => return None,
        };
        if fuf.get(upstream_id).op != OpKind::Gemm {
            return None;
        }
        if !matches!(node.inputs.get(1), Some(FufInput::Weight { .. })) {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            // Boundary input is the upstream Gemm's output tile — that's
            // the slot the kernel reads at `buffer(0)`.
            boundary_inputs: vec![upstream_id],
            boundary_outputs: vec![seed],
        })
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
        // Share the upstream Gemm's `LinearLayer` accessor — the bias
        // rides on `LinearLayer::Dense.bias` (auto-detected by
        // `load_dense`) or `LinearLayer::AffineQuant.linear_bias`
        // (`load_affine_quant` pulls `<prefix>.bias` when present).
        // `fused_accessor_name` produces the same Ident as the
        // upstream's `MetalAffineQmmImpl` / `MetalGemmImpl`, so the
        // loader dedupes and one `LinearLayer` field covers both.
        let bias_id = claimed_tiles[0];
        let bias_node = fuf.get(bias_id);
        let upstream_id = match bias_node.inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            _ => panic!(
                "MetalBiasAddImpl::required_weights: BiasAdd's first input \
                 must be a Tile — `matches` should have rejected otherwise"
            ),
        };
        let upstream_node = fuf.get(upstream_id);
        let gemm_weight = first_weight_ref(upstream_node)
            .expect("MetalBiasAddImpl: upstream Gemm must carry a weight ref");
        let sources = vec![gemm_weight];
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { ::ferrite_kernels::layers::LinearLayer },
            source_weights: sources,
        }]
    }

    fn opcode_shape(&self) -> OpcodeShape {
        // Mirrors `Instruction::MetalBiasAdd(u32, u32, u32, WtFn<W,
        // LinearLayer>, u32, bool)`. `n` is the bias broadcast dim;
        // `is_affine` selects `affine_linear_bias()` vs `dense_bias()`
        // at worker resolve time.
        OpcodeShape::new(
            "MetalBiasAdd",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                ("n", syn::parse_quote!(u32)),
                ("is_affine", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let bias_id = m.claimed_tiles[0];
        let bias_node = fuf.get(bias_id);
        let (upstream_id, upstream_slot) = match bias_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "MetalBiasAdd::fan_out: BiasAdd's first input must be a Tile \
                 (got {other:?}); `matches` is out of sync"
            ),
        };
        let upstream_node = fuf.get(upstream_id);
        let in_slot_idx = slots.of(upstream_id, upstream_slot);
        let out_slot_idx = slots.of(bias_id, 0);

        // Resolve the shared `LinearLayer` accessor (same name the
        // upstream Gemm's `required_weights` declared).
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("MetalBiasAdd::fan_out: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());

        // `n` = bias broadcast dim = the Gemm's output-N.
        let (n, _k) = gemm_nk_from_fuf(fuf, upstream_node, bounds).expect(
            "MetalBiasAdd::fan_out: upstream Gemm (N, K) must resolve from \
             FUF + bounds at fan_out time",
        );
        let is_affine = matches!(
            weight_storage_of(upstream_node),
            Some(StorageFormat::Affine { .. })
        );

        Some(vec![OpInstance::new(
            syn::Ident::new("MetalBiasAdd", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { #n },
                quote! { #is_affine },
            ],
        )])
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
