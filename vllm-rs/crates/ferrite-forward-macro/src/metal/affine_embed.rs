// SPDX-License-Identifier: Apache-2.0
//
//! Metal implementation for `OpKind::Embed` on MLX-affine int4
//! quantized embeddings (P6 of `INT4_PARITY_PLAN.md`).
//!
//! Pairs with [`crate::metal::MetalEmbedImpl`] (the dense path):
//! `MetalEmbedImpl::matches` rejects `StorageFormat::Affine` so the
//! solver routes quantized embeddings here. The fan_out emits an
//! `AffineEmbed` opcode whose runtime ferrite-forward Instruction
//! variant runs the fused gather + dequant kernel
//! (`affine_embed_<dtype>_gs_<gs>_b_4` in
//! `quantized_dequantize.metal`).
//!
//! Unlike `MetalEmbedImpl` (which delegates to `EmbedRefImpl` for the
//! codegen pieces), this Impl owns its own `opcode_shape` /
//! `required_weights` / `fan_out` because the opcode shape diverges
//! from `Embed` — three extra fields (`weight_fn` resolves to
//! `AffineQuantEmbedding`, plus `group_size: u32` and `bits: u32`).

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::codegen::split_base_layer;
use crate::emit::weight_field_name;
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpcodeShape, Resources,
    SlotMap, WeightAccessor, WorkloadConstraint, weight_storage_of,
};
use crate::quantization::StorageFormat;
use crate::target::{Backend, TargetProfile};

use proc_macro2::TokenStream;
use quote::quote;

/// Metal MLX-affine int4 quantized embedding implementation.
///
/// Matches: `OpKind::Embed` AND
/// `weight_storage_of(node) == StorageFormat::Affine`.
///
/// Cost model: same shape as `MetalEmbedImpl` (memory-bound gather)
/// with a `(bytes_per_element_packed)` adjustment — packed U32
/// weights are `bits/8` bytes per element, 4× smaller than BF16
/// dense at bits=4. Total bytes still scale with `num_tokens *
/// hidden_size` for the output write, which dominates anyway.
#[derive(Debug)]
pub struct MetalAffineEmbedImpl {
    dtype: &'static str,
}

impl MetalAffineEmbedImpl {
    pub fn fp16() -> Self {
        Self { dtype: "fp16" }
    }

    pub fn bf16() -> Self {
        Self { dtype: "bf16" }
    }

    fn analytical_cost_us(&self, num_tokens: u32, hidden_size: u32, bandwidth_gbps: f64) -> f64 {
        let bytes_per_element = 2.0; // fp16/bf16 output
        let output_elements = (num_tokens * hidden_size) as f64;
        // Weight read is ~bytes_per_element/4 (bits=4 packed) per
        // output element + scale/bias group reads (negligible at
        // gs >= 32). Output write dominates.
        let bytes_read = output_elements * (bytes_per_element / 4.0 + bytes_per_element / 16.0);
        let bytes_written = output_elements * bytes_per_element;
        let total_bytes = bytes_read + bytes_written;
        let total_gb = total_bytes / 1e9;
        let time_seconds = total_gb / bandwidth_gbps;
        time_seconds * 1e6
    }
}

impl Implementation for MetalAffineEmbedImpl {
    fn name(&self) -> &'static str {
        match self.dtype {
            "fp16" => "metal_affine_embed_f16",
            "bf16" => "metal_affine_embed_bf16",
            _ => "metal_affine_embed",
        }
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::Embed {
            return None;
        }
        // Reject anything but Affine storage — peer `MetalEmbedImpl`
        // handles the dense path.
        let storage = weight_storage_of(node)?;
        if !matches!(storage, StorageFormat::Affine { .. }) {
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
        if let Some(dims) = dims
            && dims.len() == 2
        {
            let num_tokens = dims[0] as u32;
            let hidden_size = dims[1] as u32;
            return self.analytical_cost_us(
                num_tokens,
                hidden_size,
                ctx.profile.memory_bandwidth_gbps,
            );
        }
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

    /// Override the default `WeightAccessor::rust_type` (which is
    /// `Embedding` per `rust_type_for_weight_consumed_by(OpKind::Embed)`)
    /// because the quantized path stores three tensors per embedding
    /// (`weight`, `scales`, `affine_biases`) bundled into
    /// `AffineQuantEmbedding` — `Embedding` only holds the dense
    /// `weight`.
    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &tid in claimed_tiles {
            let node = fuf.get(tid);
            for input in &node.inputs {
                if let crate::fuf::FufInput::Weight { id, index, .. } = input {
                    let name = weight_field_name(program, *id, *index);
                    if !seen.insert(name.to_string()) {
                        continue;
                    }
                    out.push(WeightAccessor {
                        name,
                        rust_type: quote! { ::ferrite_kernels::layers::AffineQuantEmbedding },
                        source_weights: vec![(*id, *index)],
                    });
                }
            }
        }
        out
    }

    /// Variant `AffineEmbed { out_slot, weight_fn, group_size, bits }`
    /// — payload mirrors `ferrite_forward::Instruction::AffineEmbed`.
    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "AffineEmbed",
            vec![
                ("out_slot", syn::parse_quote!(u32)),
                ("group_size", syn::parse_quote!(u32)),
                ("bits", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<ferrite_forward::Instruction>> {
        let tile = m.claimed_tiles[0];
        let out_slot = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("AffineEmbed: required_weights returned empty");
        let (base, _layer) = split_base_layer(&acc.name.to_string());
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());

        // Read (group_size, bits) off the weight input's storage. The
        // matcher already guaranteed it's Affine, so unreachable!
        // covers the impossible cases.
        let node = fuf.get(tile);
        let (group_size, bits) = match weight_storage_of(node) {
            Some(StorageFormat::Affine { group_size, bits }) => (*group_size, *bits),
            other => unreachable!(
                "MetalAffineEmbedImpl::fan_out called with non-Affine storage: {other:?} \
                 — matches() should have rejected this"
            ),
        };
        // AffineQuantEmbedding accessor flows through
        // `required_weights()`; the macro injects the
        // `(tape_index, op_idx, slot=0)` arm into the per-arch
        // `WeightAccessors::affine_quant_embedding_at`.
        let _ = base_ident;
        Some(vec![ferrite_forward::Instruction::AffineEmbed(
            out_slot, group_size, bits,
        )])
    }
}

/// Keep this list of expected accessors when reasoning about the
/// `AffineQuantEmbedding` field type emission in
/// `emit_weights_accessor_methods` / `group_accessors_by_base`. Used
/// only for documentation; the codegen branches on `WeightAccessor::
/// rust_type` directly, so no enum or trait gates the type.
#[allow(dead_code)]
fn _expected_affine_embed_field_type() -> TokenStream {
    quote! { ::ferrite_kernels::layers::AffineQuantEmbedding }
}
