// SPDX-License-Identifier: Apache-2.0

//! Metal NVIDIA ModelOpt NVFP4 int4 quantized GEMM implementation adapter.
//!
//! Singleton GEMM claim for `OpKind::Gemm` tiles whose weight is
//! `StorageFormat::Nvfp4 { group_size }`. Emits `Instruction::Nvfp4Qmm`
//! so the metal interpreter's lowering pass (`lower_one`) picks the
//! `nvfp4_qmv` (decode) or `nvfp4_qmm_t` (prefill) kernel per-bucket.
//!
//! Structurally identical to `MetalAffineQmmImpl` — the only differences
//! are the storage gate (`Nvfp4` vs `Affine`) and the emitted opcode
//! (`Nvfp4Qmm` vs `AffineQmm`). NVFP4 group_size is always 16 and bits
//! is always 4. First cut: analytical roofline cost only (no NVFP4
//! cost-sweep rows yet) and no synthesis-atom participation (`as_atom`
//! returns `None`); both are Phase-2 follow-ons.

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::codegen::split_base_layer;
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpcodeShape, Resources,
    SlotMap, WeightAccessor, WorkloadConstraint, default_required_weights, gemm_nk_from_fuf,
    weight_storage_of,
};
use crate::metal::affine_qmm::affine_qmm_vector_limit;
use crate::quantization::StorageFormat;
use crate::target::{Backend, TargetProfile};

/// Solver-side singleton claim for NVFP4 int4 GEMMs. Mirrors
/// `MetalAffineQmmImpl`; differs only in the storage gate (Nvfp4 vs
/// Affine) and the emitted opcode (`Nvfp4Qmm` vs `AffineQmm`).
#[derive(Debug)]
pub struct MetalNvfp4QmmImpl {
    /// Activation dtype (`"fp16"` or `"bf16"`). The quantized weight
    /// dtype (U8 packed E2M1 + F16 folded scales) is fixed by the NVFP4
    /// format and not a knob.
    dtype: &'static str,
}

impl MetalNvfp4QmmImpl {
    pub fn new_fp16() -> Self {
        Self { dtype: "fp16" }
    }

    pub fn new_bf16() -> Self {
        Self { dtype: "bf16" }
    }

    /// Analytical compute-bound cost — same roofline as
    /// `MetalAffineQmmImpl` (`2*M*N*K` FLOPs at `peak_tflops_fp16`). The
    /// nvfp4 kernels dequantize in-register, so peak compute throughput
    /// is the right roofline.
    fn analytical_cost_us(&self, m: u32, n: u32, k: u32, compute_tflops: f64) -> f64 {
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        let time_seconds = flops / (compute_tflops * 1e12);
        time_seconds * 1e6
    }
}

impl Implementation for MetalNvfp4QmmImpl {
    fn name(&self) -> &'static str {
        match self.dtype {
            "fp16" => "metal_nvfp4_qmm_f16",
            "bf16" => "metal_nvfp4_qmm_bf16",
            _ => "metal_nvfp4_qmm",
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
        if node.op != OpKind::Gemm {
            return None;
        }
        // NVFP4-only: Dense Gemms fall through to `MetalGemmImpl`,
        // affine Gemms to `MetalAffineQmmImpl`.
        if !matches!(weight_storage_of(node), Some(StorageFormat::Nvfp4 { .. })) {
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
        let output_dims = node.outputs.first().and_then(|s| ctx.eval_shape(s));
        if let Some(dims) = output_dims
            && dims.len() >= 2
        {
            let mm = dims[0] as u32;
            let nn = dims[1] as u32;
            let kk = node
                .inputs
                .iter()
                .find_map(|i| match i {
                    crate::fuf::FufInput::Tile { id, slot } => ctx
                        .fuf
                        .get(*id)
                        .outputs
                        .get(*slot as usize)
                        .and_then(|s| ctx.eval_shape(s))
                        .and_then(|v| v.last().copied())
                        .map(|x| x as u32),
                    _ => None,
                })
                .unwrap_or(2048);
            // First cut: analytical roofline only — the ferrite-metal
            // cost sweep has no nvfp4 rows yet (Phase 2 adds them, then
            // an `empirical_cost_us` mirroring `affine_qmm.rs`).
            return self.analytical_cost_us(mm, nn, kk, ctx.profile.peak_tflops_fp16);
        }
        500.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
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

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        default_required_weights(claimed_tiles, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        nvfp4_qmm_opcode_shape()
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<ferrite_forward::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(crate::fuf::FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "Nvfp4Qmm: first input must be a Tile (got {other:?}); the FUF tile shape \
                 doesn't match what fan_out expects"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Nvfp4Qmm: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        let (n, k) = gemm_nk_from_fuf(fuf, node, bounds)
            .expect("Nvfp4Qmm: weight (N, K) must resolve from FUF + bounds at fan_out time");
        let group_size = match weight_storage_of(node) {
            Some(StorageFormat::Nvfp4 { group_size }) => *group_size,
            other => panic!(
                "Nvfp4Qmm: matched a Gemm with non-Nvfp4 storage at fan_out time ({other:?}) \
                 — `matches` gate is out of sync with `fan_out` inspection"
            ),
        };
        // NVFP4 is always 4-bit.
        let bits = 4;
        let vector_limit = affine_qmm_vector_limit(k, n);
        // Weight (LinearLayer) flows through `required_weights()`; the
        // macro injects the `(tape_index, op_idx, slot=0)` arm into the
        // per-arch `WeightAccessors::linear_at`.
        let _ = base_ident;
        Some(vec![ferrite_forward::Instruction::Nvfp4Qmm(
            in_slot_idx,
            out_slot_idx,
            layer,
            n,
            k,
            group_size,
            bits,
            vector_limit,
        )])
    }
}

/// `Instruction::Nvfp4Qmm` variant shape. Must match the
/// `ferrite_forward::Instruction::Nvfp4Qmm` tuple (8 × u32), bit-for-bit
/// the same field list as `AffineQmm`.
pub(crate) fn nvfp4_qmm_opcode_shape() -> OpcodeShape {
    OpcodeShape::new(
        "Nvfp4Qmm",
        vec![
            ("in_slot", syn::parse_quote!(u32)),
            ("out_slot", syn::parse_quote!(u32)),
            ("layer", syn::parse_quote!(u32)),
            ("n", syn::parse_quote!(u32)),
            ("k", syn::parse_quote!(u32)),
            ("group_size", syn::parse_quote!(u32)),
            ("bits", syn::parse_quote!(u32)),
            ("vector_limit", syn::parse_quote!(u32)),
        ],
    )
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::target::from_metal_profile;

    #[test]
    fn metal_nvfp4_qmm_only_compatible_with_metal_targets() {
        let metal_impl = MetalNvfp4QmmImpl::new_fp16();
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));
    }
}
