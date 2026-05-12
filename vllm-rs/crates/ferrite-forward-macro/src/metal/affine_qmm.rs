// SPDX-License-Identifier: Apache-2.0

//! Metal MLX-affine int4 quantized GEMM implementation adapter.
//!
//! Singleton GEMM claim for `OpKind::Gemm` tiles whose weight is
//! `StorageFormat::Affine { group_size, bits }`. Emits
//! `Instruction::AffineQmm` so the metal interpreter's lowering pass
//! (`lower_one`) picks `qmv_*` (decode) or `qmm_t_*` (prefill)
//! per-bucket against the qmv/qmm_t kernels landed in P3/P4.
//!
//! Sits beside `MetalGemmImpl` (Dense storage). The Dense impl's
//! `matches` gates out Affine so this impl wins on Affine tiles.

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::codegen::split_base_layer;
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpInstance, OpcodeShape,
    Resources, SlotMap, WeightAccessor, WorkloadConstraint, default_required_weights,
    gemm_nk_from_fuf, weight_storage_of,
};
use crate::quantization::StorageFormat;
use crate::target::{Backend, TargetProfile};

use quote::quote;

/// Solver-side singleton claim for MLX-affine int4 GEMMs. Mirrors
/// `MetalGemmImpl` shape; differs only in the storage gate (Affine vs
/// Dense) and the emitted opcode (`AffineQmm` vs `Gemm`).
#[derive(Debug)]
pub struct MetalAffineQmmImpl {
    /// Activation dtype (`"fp16"` or `"bf16"`). The quantized weight
    /// dtype (U32 packed words + F16 scales/biases) is fixed by the
    /// MLX-affine format and not a knob.
    dtype: &'static str,
}

impl MetalAffineQmmImpl {
    pub fn new_fp16() -> Self {
        Self { dtype: "fp16" }
    }

    pub fn new_bf16() -> Self {
        Self { dtype: "bf16" }
    }

    /// Analytical compute-bound cost — same shape as `MetalGemmImpl`
    /// (`2*M*N*K` FLOPs at `peak_tflops_fp16`). The qmv/qmm_t kernels
    /// dequantize in-register, so peak compute throughput is the
    /// right roofline; the int4 BW saving doesn't change `FLOPs`.
    fn analytical_cost_us(&self, m: u32, n: u32, k: u32, compute_tflops: f64) -> f64 {
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        let time_seconds = flops / (compute_tflops * 1e12);
        time_seconds * 1e6
    }
}

impl Implementation for MetalAffineQmmImpl {
    fn name(&self) -> &'static str {
        match self.dtype {
            "fp16" => "metal_affine_qmm_f16",
            "bf16" => "metal_affine_qmm_bf16",
            _ => "metal_affine_qmm",
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
        // Affine-only: Dense Gemms fall through to `MetalGemmImpl`.
        if !matches!(weight_storage_of(node), Some(StorageFormat::Affine { .. })) {
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
        let output_dims = node
            .outputs
            .first()
            .and_then(|s| ctx.eval_shape(s));
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

            // Consult empirical cost table first. Reconstruct the
            // kernel-variant key by re-running the dispatcher decision
            // (vector_limit + qmv/qmm_t pick) on `(M, N, K, gs, bits)`.
            // Names match `ferrite-metal-cost-sweep::affine_q{mv,mm}_sweep::
            // csv_kernel_name`. Falls back to the analytical roofline when
            // the chip's profile has no row for this shape (chip ships
            // empty cost_table or the sweep didn't cover this (M, N, K)).
            if let Some(cost) = empirical_cost_us(self.dtype, mm, nn, kk, node, ctx) {
                return cost;
            }
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
        affine_qmm_opcode_shape()
    }

    fn as_atom(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
    ) -> Option<Box<dyn crate::atom::Atom>> {
        // Only the decode branch (M < vector_limit) participates in
        // synthesis today. Prefill stays on the qmm_t hand-written
        // kernels until Phase 5 (mk_mma) lands.
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (group_size, _bits) = match weight_storage_of(node) {
            Some(StorageFormat::Affine { group_size, bits }) => (*group_size, *bits),
            _ => return None,
        };
        Some(Box::new(crate::atom_lib::AffineQmvAtom {
            group_size,
            local_head_expr: "__head",
        }))
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(crate::fuf::FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "AffineQmm: first input must be a Tile (got {other:?}); the FUF tile shape \
                 doesn't match what fan_out expects"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("AffineQmm: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        let (n, k) = gemm_nk_from_fuf(fuf, node, bounds)
            .expect("AffineQmm: weight (N, K) must resolve from FUF + bounds at fan_out time");
        let (group_size, bits) = match weight_storage_of(node) {
            Some(StorageFormat::Affine { group_size, bits }) => (*group_size, *bits),
            other => panic!(
                "AffineQmm: matched a Gemm with non-Affine storage at fan_out time ({other:?}) \
                 — `matches` gate is out of sync with `fan_out` inspection"
            ),
        };
        let vector_limit = affine_qmm_vector_limit(k, n);
        Some(vec![OpInstance::new(
            syn::Ident::new("AffineQmm", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { #n },
                quote! { #k },
                quote! { #group_size },
                quote! { #bits },
                quote! { #vector_limit },
            ],
        )])
    }
}

/// Look up the empirical kernel cost for an AffineQmm tile. Returns
/// `None` when the profile has no row for the (kernel, M, N, K) key —
/// either the chip ships an empty cost_table or the sweep didn't cover
/// this exact shape. Caller falls back to the analytical roofline in
/// either case.
///
/// Reconstructs the dispatcher's per-shape kernel pick:
///   * `M < vector_limit` → matvec branch (`pick_qmv_kernel`)
///   * else                → matmul branch (`pick_qmm_t_kernel`)
///
/// Kernel-name format mirrors the rows emitted by
/// `ferrite-metal-cost-sweep::affine_q{mv,mm}_sweep::csv_kernel_name`.
fn empirical_cost_us(
    dtype: &'static str,
    m: u32,
    n: u32,
    k: u32,
    node: &crate::fuf::FufNode,
    ctx: &CostCtx,
) -> Option<f64> {
    use ferrite_metal_kernels::quantized::{
        pick_qmm_t_kernel, pick_qmv_kernel, QmmTKernel, QmvKernel,
    };

    let dtype_str = match dtype {
        "fp16" => "f16",
        "bf16" => "bf16",
        _ => return None,
    };
    let (group_size, _bits) = match weight_storage_of(node) {
        Some(StorageFormat::Affine { group_size, bits }) => (*group_size, *bits),
        _ => return None,
    };

    let vector_limit = affine_qmm_vector_limit(k, n);
    let name = if m < vector_limit {
        let kernel = pick_qmv_kernel(n, k, 4);
        match kernel {
            QmvKernel::Quad { d } => {
                format!("affine_qmv_quad_{dtype_str}_gs{group_size}_d{d}")
            }
            QmvKernel::Fast => format!("affine_qmv_fast_{dtype_str}_gs{group_size}"),
            QmvKernel::Generic => format!("affine_qmv_{dtype_str}_gs{group_size}"),
        }
    } else {
        let kernel = pick_qmm_t_kernel(m, n, k, 1, group_size);
        match kernel {
            QmmTKernel::Standard => format!("affine_qmm_t_{dtype_str}_gs{group_size}"),
            QmmTKernel::SplitK { split_k, .. } => {
                format!("affine_qmm_t_splitk{split_k}_{dtype_str}_gs{group_size}")
            }
        }
    };
    // For matvec the sweep emits M=1 rows; the AffineQmm tile's bucket
    // M ≥ 1 in all decode cases. Force M=1 on the lookup so the right
    // row is found regardless of the bucket size — kernel cost is
    // M-independent in the matvec regime (one threadgroup per output
    // row, M only widens the input fetch).
    let lookup_m = if m < vector_limit { 1 } else { m };
    ctx.profile.cost_us_for(&name, lookup_m, n, k)
}

/// `Instruction::AffineQmm` variant shape used by both the standalone
/// `MetalAffineQmmImpl` and the decomposed q-MLP path inside
/// `MetalFusedGateUpSiluMulImpl::fan_out`. Both registrations must
/// declare bit-identical shapes (`arch_opcodes.register` panics on
/// shape disagreement).
pub(crate) fn affine_qmm_opcode_shape() -> OpcodeShape {
    OpcodeShape::new(
        "AffineQmm",
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
            ("k", syn::parse_quote!(u32)),
            ("group_size", syn::parse_quote!(u32)),
            ("bits", syn::parse_quote!(u32)),
            ("vector_limit", syn::parse_quote!(u32)),
        ],
    )
}

/// Compute the M boundary that separates qmv (decode) from qmm_t
/// (prefill) for the `(K, N, arch_gen)` triple.
///
/// **Limitation**: `arch_gen` is currently hardcoded to
/// `AppleSiliconGen::M4` — `fan_out` doesn't receive `&TargetProfile`,
/// so the macro can't read the live target's generation from here.
/// All current ferrite-metal dev/CI machines are M3/M4 (24 GiB cap;
/// see `feedback_machine_24gi_limit`). The M1/M2 vs M3/M4 split only
/// shifts the boundary by ±4 (matvec is correct at any M, just less
/// efficient near the high-M edge), so the worst-case impact is a
/// slight prefill regression on M1/M2 — acceptable until a proper
/// `&TargetProfile` thread lands on the `fan_out` signature.
pub(crate) fn affine_qmm_vector_limit(k: u32, n: u32) -> u32 {
    ferrite_metal_kernels::quantized::get_qmv_batch_limit(
        k,
        n,
        ferrite_metal_targets::AppleSiliconGen::M4,
    )
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::target::from_metal_profile;

    #[test]
    fn metal_affine_qmm_only_compatible_with_metal_targets() {
        let metal_impl = MetalAffineQmmImpl::new_fp16();
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));
    }

    #[test]
    fn affine_qmm_vector_limit_matches_m4_table() {
        // Sanity check that hardcoded arch_gen wires through MLX's
        // table. K=N=2048 → 18 on M3/M4 per `quantized.rs:get_qmv_batch_limit`.
        assert_eq!(affine_qmm_vector_limit(2048, 2048), 18);
    }
}
