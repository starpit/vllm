// SPDX-License-Identifier: Apache-2.0
//! Non-generic, hashable view of an [`Instruction<W>`] for backbone
//! introspection (`vllm ferrite info`).
//!
//! `Instruction<W>` is generic, carries function pointers, and lives
//! behind `cfg(feature = "cuda")`. None of that is useful when the
//! consumer wants to:
//! - tell two backbones apart for fusion-hunting
//! - hash a backbone for equivalence-class grouping
//! - print a backbone as ASCII without needing a live `&W`
//!
//! [`Instruction::normalize`] runs once per row to produce a
//! [`NormalizedStep`]: the variant tag as `&'static str`, plus a
//! flat field list with semantic kinds (slot / layer / const /
//! kernel-class). Function pointers (`WtFn<W, L>`, `CosSinFn<W>`)
//! collapse to `LayerKind(L_name)` / `RopeCosSin` — what they *call*,
//! not their address.
//!
//! The `match` is closed (no `_` arm). Adding a new
//! [`Instruction<W>`] variant fails the build until normalize is
//! taught the new arm — same closed-emitter invariant the
//! interpreter `eval` already enforces.

#![cfg(feature = "cuda")]

use crate::Instruction;

/// One backbone step in non-generic form. Equivalent to one row of a
/// `BACKBONE_M_<…>` / `LM_HEAD_M_<…>` static slice with the W
/// dependence projected out.
///
/// `Hash` / `Eq` ignore nothing — every produced field participates.
/// Consumers that want a coarser equivalence (e.g. `body` —
/// alpha-renamed slots, dropped loop count) build that as a
/// transformation on top of `Vec<NormalizedStep>`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NormalizedStep {
    /// PascalCase variant ident — e.g. `"FusedAddRmsNorm"`.
    pub kind: &'static str,
    /// Field values in declaration order (matches the
    /// [`Instruction<W>`] tuple-variant order).
    pub fields: Vec<NormalizedField>,
}

/// Kinded view of a single [`Instruction<W>`] field.
///
/// `f32` consts stored as `u32` bits so this enum stays `Eq + Hash`
/// — the renderer reconstructs via `f32::from_bits`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NormalizedField {
    /// Tile-table slot index. Subject to alpha-renaming under the
    /// `body` equivalence class.
    Slot(u32),
    /// Layer index. Inside a `Loop` body these are *baseline* values
    /// (the per-iter index is added by `InterpreterCtx::layer_offset`
    /// at dispatch).
    Layer(u32),
    /// Plain integer const — Cutlass tile dims, FlashInfer head_dim,
    /// Reshape ndim, etc.
    ConstU32(u32),
    /// `f32` const stored as bit pattern for `Eq + Hash`.
    ConstF32Bits(u32),
    /// Boolean flag — `interleaved`, `biased`,
    /// `use_logits_soft_cap`, …
    ConstBool(bool),
    /// Fixed-length integer array, truncated to its valid prefix
    /// (`Reshape::dims_lit[..ndim]`).
    ConstU32Array(Vec<u32>),
    /// Same, but the underlying field type was `[u8; N]`.
    ConstU8Array(Vec<u8>),
    /// `WtFn<W, L>` collapsed to the layer-kernel-class name (`L`'s
    /// type name) — e.g. `"RmsNorm"`, `"LinearLayer"`,
    /// `"MarlinLinear"`, `"Bnb4bitLinear"`, `"Fp8AnyLinear"`,
    /// `"CohereLayerNorm"`, `"Embedding"`, `"DeepSeekV2MoELayer"`.
    LayerKind(&'static str),
    /// `CosSinFn<W>` — the rotary cos/sin table accessor. Carries
    /// no other identity at this layer (different rope tables come
    /// from per-variant `CanonicalParams` consts, not the slice).
    RopeCosSin,
    /// `Loop(count, body_len)` — count of iterations.
    LoopCount(u32),
    /// `Loop(count, body_len)` — number of subsequent rows that form
    /// the body.
    LoopBodyLen(u32),
}

impl<W> Instruction<W> {
    /// Project this row to a [`NormalizedStep`]. Pure data — no
    /// `&W` needed, no GPU work, no allocator beyond the field
    /// `Vec`.
    ///
    /// The match is closed (no catch-all). Every `Instruction<W>`
    /// variant has exactly one arm.
    #[must_use]
    pub fn normalize(&self) -> NormalizedStep {
        use NormalizedField as F;
        let (kind, fields): (&'static str, Vec<NormalizedField>) = match *self {
            Instruction::Embed(out_slot, _wf) => {
                ("Embed", vec![F::Slot(out_slot), F::LayerKind("Embedding")])
            }
            Instruction::RmsNorm(in_slot, out_slot, layer, _wf) => (
                "RmsNorm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("RmsNorm"),
                ],
            ),
            Instruction::LayerNorm(in_slot, out_slot, layer, _wf) => (
                "LayerNorm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("CohereLayerNorm"),
                ],
            ),
            Instruction::Reshape(in_slot, out_slot, dims_lit, dims_nt_pow, ndim) => {
                let n = ndim as usize;
                (
                    "Reshape",
                    vec![
                        F::Slot(in_slot),
                        F::Slot(out_slot),
                        F::ConstU32Array(dims_lit[..n].to_vec()),
                        F::ConstU8Array(dims_nt_pow[..n].to_vec()),
                        F::ConstU32(u32::from(ndim)),
                    ],
                )
            }
            Instruction::Add(delta_slot, residual_slot) => {
                ("Add", vec![F::Slot(delta_slot), F::Slot(residual_slot)])
            }
            Instruction::ScalarMul(in_slot, out_slot, scale) => (
                "ScalarMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::ConstF32Bits(scale.to_bits()),
                ],
            ),
            Instruction::TanhSoftCap(in_slot, out_slot) => {
                ("TanhSoftCap", vec![F::Slot(in_slot), F::Slot(out_slot)])
            }
            Instruction::FusedAddRmsNorm(in_slot, out_slot, layer, _wf) => (
                "FusedAddRmsNorm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("RmsNorm"),
                ],
            ),
            Instruction::FusedAddRmsNormWithOffset(in_slot, out_slot, layer, offset, _wf) => (
                "FusedAddRmsNormWithOffset",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::ConstF32Bits(offset.to_bits()),
                    F::LayerKind("RmsNorm"),
                ],
            ),
            Instruction::ScalarOffsetRmsNorm(in_slot, out_slot, layer, offset, _wf) => (
                "ScalarOffsetRmsNorm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::ConstF32Bits(offset.to_bits()),
                    F::LayerKind("RmsNorm"),
                ],
            ),
            Instruction::Gemm(in_slot, out_slot, layer, _wf) => (
                "Gemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::FusedGemmBias(in_slot, out_slot, layer, _wf) => (
                "FusedGemmBias",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::FusedGateUpSiluMul(in_slot, out_slot, layer, _wf) => (
                "FusedGateUpSiluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::FusedGateUpGeluMul(in_slot, out_slot, layer, _wf) => (
                "FusedGateUpGeluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::FusedQkvRopeCache(
                in_slot,
                out_slot,
                layer,
                _wf,
                _cs,
                biased,
                interleaved,
            ) => (
                "FusedQkvRopeCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::RopeCosSin,
                    F::ConstBool(biased),
                    F::ConstBool(interleaved),
                ],
            ),
            Instruction::FusedQkvQkNormRopeCache(
                in_slot,
                out_slot,
                layer,
                _qw,
                _kw,
                _vw,
                _qn,
                _kn,
                _cs,
                q_offset,
                k_offset,
            ) => (
                "FusedQkvQkNormRopeCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::LayerKind("LinearLayer"),
                    F::LayerKind("LinearLayer"),
                    F::LayerKind("RmsNorm"),
                    F::LayerKind("RmsNorm"),
                    F::RopeCosSin,
                    F::ConstF32Bits(q_offset.to_bits()),
                    F::ConstF32Bits(k_offset.to_bits()),
                ],
            ),
            Instruction::FusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                _wf,
                _cs,
                biased,
                interleaved,
            ) => (
                "FusedQkvRopePrefill",
                vec![
                    F::Slot(in_slot),
                    F::Slot(q_out_slot),
                    F::Slot(k_out_slot),
                    F::Slot(v_out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::RopeCosSin,
                    F::ConstBool(biased),
                    F::ConstBool(interleaved),
                ],
            ),
            Instruction::AttentionViaCache(in_slot, out_slot, layer, _cs, interleaved) => (
                "AttentionViaCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::RopeCosSin,
                    F::ConstBool(interleaved),
                ],
            ),
            Instruction::AttentionPrefillContiguous(
                q_slot,
                k_slot,
                v_slot,
                out_slot,
                interleaved,
            ) => (
                "AttentionPrefillContiguous",
                vec![
                    F::Slot(q_slot),
                    F::Slot(k_slot),
                    F::Slot(v_slot),
                    F::Slot(out_slot),
                    F::ConstBool(interleaved),
                ],
            ),
            Instruction::SlidingAttentionViaCache(in_slot, out_slot, layer, _cs, interleaved) => (
                "SlidingAttentionViaCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::RopeCosSin,
                    F::ConstBool(interleaved),
                ],
            ),
            Instruction::SlidingAttentionPrefillContiguous(
                q_slot,
                k_slot,
                v_slot,
                out_slot,
                interleaved,
            ) => (
                "SlidingAttentionPrefillContiguous",
                vec![
                    F::Slot(q_slot),
                    F::Slot(k_slot),
                    F::Slot(v_slot),
                    F::Slot(out_slot),
                    F::ConstBool(interleaved),
                ],
            ),
            Instruction::FlashInferAttentionDecode(
                in_slot,
                out_slot,
                layer,
                _cs,
                head_dim,
                use_logits_soft_cap,
            ) => (
                "FlashInferAttentionDecode",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::RopeCosSin,
                    F::ConstU32(head_dim),
                    F::ConstBool(use_logits_soft_cap),
                ],
            ),
            Instruction::FlashInferAttentionPrefill(
                q_slot,
                k_slot,
                v_slot,
                out_slot,
                layer,
                head_dim,
                use_logits_soft_cap,
            ) => (
                "FlashInferAttentionPrefill",
                vec![
                    F::Slot(q_slot),
                    F::Slot(k_slot),
                    F::Slot(v_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::ConstU32(head_dim),
                    F::ConstBool(use_logits_soft_cap),
                ],
            ),
            Instruction::RopeAppend(
                q_slot,
                k_slot,
                v_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                _cs,
                interleaved,
            ) => (
                "RopeAppend",
                vec![
                    F::Slot(q_slot),
                    F::Slot(k_slot),
                    F::Slot(v_slot),
                    F::Slot(q_out_slot),
                    F::Slot(k_out_slot),
                    F::Slot(v_out_slot),
                    F::Layer(layer),
                    F::RopeCosSin,
                    F::ConstBool(interleaved),
                ],
            ),
            Instruction::MlaSplit(in_slot, kv_latent_slot, k_pe_slot) => (
                "MlaSplit",
                vec![
                    F::Slot(in_slot),
                    F::Slot(kv_latent_slot),
                    F::Slot(k_pe_slot),
                ],
            ),
            Instruction::MlaAttention(q_slot, kv_b_slot, k_pe_slot, out_slot, layer, _cs) => (
                "MlaAttention",
                vec![
                    F::Slot(q_slot),
                    F::Slot(kv_b_slot),
                    F::Slot(k_pe_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::RopeCosSin,
                ],
            ),
            Instruction::DeepSeekMoe(in_slot, out_slot, layer, _wf) => (
                "DeepSeekMoe",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("DeepSeekV2MoELayer"),
                ],
            ),
            Instruction::CutlassGemm(in_slot, out_slot, layer, _wf, tile_m, tile_n, stages) => (
                "CutlassGemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                ],
            ),
            Instruction::CutlassGemmSplitK(
                in_slot,
                out_slot,
                layer,
                _wf,
                tile_m,
                tile_n,
                stages,
                split_k,
            ) => (
                "CutlassGemmSplitK",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::ConstU32(split_k),
                ],
            ),
            Instruction::CutlassGemmAdd(
                in_slot,
                residual_slot,
                layer,
                _wf,
                tile_m,
                tile_n,
                stages,
            ) => (
                "CutlassGemmAdd",
                vec![
                    F::Slot(in_slot),
                    F::Slot(residual_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                ],
            ),
            Instruction::CutlassGemv(in_slot, out_slot, layer, _wf) => (
                "CutlassGemv",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::CutlassFusedGemmBias(in_slot, out_slot, layer, _wf) => (
                "CutlassFusedGemmBias",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::CutlassFusedGateUpSiluMul(in_slot, out_slot, layer, _wf) => (
                "CutlassFusedGateUpSiluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::MarlinGemm(in_slot, out_slot, layer, _wf) => (
                "MarlinGemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("MarlinLinear"),
                ],
            ),
            Instruction::MarlinFusedGateUpSiluMul(in_slot, out_slot, layer, _wf) => (
                "MarlinFusedGateUpSiluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("MarlinLinear"),
                ],
            ),
            Instruction::MarlinFusedGateUpGeluMul(in_slot, out_slot, layer, _wf) => (
                "MarlinFusedGateUpGeluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("MarlinLinear"),
                ],
            ),
            Instruction::MarlinFusedQkvRopeCache(in_slot, out_slot, layer, _wf, _cs) => (
                "MarlinFusedQkvRopeCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("MarlinLinear"),
                    F::RopeCosSin,
                ],
            ),
            Instruction::MarlinFusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                _wf,
                _cs,
            ) => (
                "MarlinFusedQkvRopePrefill",
                vec![
                    F::Slot(in_slot),
                    F::Slot(q_out_slot),
                    F::Slot(k_out_slot),
                    F::Slot(v_out_slot),
                    F::Layer(layer),
                    F::LayerKind("MarlinLinear"),
                    F::RopeCosSin,
                ],
            ),
            Instruction::Bnb4Gemm(in_slot, out_slot, layer, _wf) => (
                "Bnb4Gemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Bnb4bitLinear"),
                ],
            ),
            Instruction::Bnb4FusedGateUpSiluMul(in_slot, out_slot, layer, _wf) => (
                "Bnb4FusedGateUpSiluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Bnb4bitLinear"),
                ],
            ),
            Instruction::Bnb4FusedGateUpGeluMul(in_slot, out_slot, layer, _wf) => (
                "Bnb4FusedGateUpGeluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Bnb4bitLinear"),
                ],
            ),
            Instruction::Bnb4FusedQkvRopeCache(in_slot, out_slot, layer, _wf, _cs) => (
                "Bnb4FusedQkvRopeCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Bnb4bitLinear"),
                    F::RopeCosSin,
                ],
            ),
            Instruction::Bnb4FusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                _wf,
                _cs,
            ) => (
                "Bnb4FusedQkvRopePrefill",
                vec![
                    F::Slot(in_slot),
                    F::Slot(q_out_slot),
                    F::Slot(k_out_slot),
                    F::Slot(v_out_slot),
                    F::Layer(layer),
                    F::LayerKind("Bnb4bitLinear"),
                    F::RopeCosSin,
                ],
            ),
            Instruction::Fp8Gemm(in_slot, out_slot, layer, _wf) => (
                "Fp8Gemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Fp8AnyLinear"),
                ],
            ),
            Instruction::Fp8FusedGemmBias(in_slot, out_slot, layer, _wf) => (
                "Fp8FusedGemmBias",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Fp8AnyLinear"),
                ],
            ),
            Instruction::Fp8FusedGateUpSiluMul(in_slot, out_slot, layer, _wf) => (
                "Fp8FusedGateUpSiluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Fp8AnyLinear"),
                ],
            ),
            Instruction::Fp8FusedGateUpGeluMul(in_slot, out_slot, layer, _wf) => (
                "Fp8FusedGateUpGeluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Fp8AnyLinear"),
                ],
            ),
            Instruction::Fp8FusedQkvRopeCache(in_slot, out_slot, layer, _wf, _cs) => (
                "Fp8FusedQkvRopeCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Fp8AnyLinear"),
                    F::RopeCosSin,
                ],
            ),
            Instruction::Fp8FusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                _wf,
                _cs,
            ) => (
                "Fp8FusedQkvRopePrefill",
                vec![
                    F::Slot(in_slot),
                    F::Slot(q_out_slot),
                    F::Slot(k_out_slot),
                    F::Slot(v_out_slot),
                    F::Layer(layer),
                    F::LayerKind("Fp8AnyLinear"),
                    F::RopeCosSin,
                ],
            ),
            Instruction::Loop(count, body_len) => {
                ("Loop", vec![F::LoopCount(count), F::LoopBodyLen(body_len)])
            }
            Instruction::Alias(dst, src) => ("Alias", vec![F::Slot(dst), F::Slot(src)]),
            Instruction::Free(slot) => ("Free", vec![F::Slot(slot)]),
        };
        NormalizedStep { kind, fields }
    }
}

/// Walk a `&[Instruction<W>]` slice and collect [`NormalizedStep`]s
/// in source order. `Loop` rows stay in-line — consumers that want
/// to render them as scoped blocks scan for `kind == "Loop"` and
/// take the next `LoopBodyLen` rows.
#[must_use]
pub fn normalize_slice<W>(slice: &[Instruction<W>]) -> Vec<NormalizedStep> {
    slice.iter().map(Instruction::normalize).collect()
}

/// One bucket's worth of dumped backbone + lm_head, with the
/// workload-bucket bounds the macro emitted alongside the slice.
#[derive(Clone, Debug)]
pub struct BucketDump {
    pub m_min: u64,
    pub m_max_excl: u64,
    pub sk_min: u64,
    pub sk_max_excl: u64,
    pub backbone: Vec<NormalizedStep>,
    pub lm_head: Vec<NormalizedStep>,
}

/// Per-variant backbone dump. Built by macro-emitted `dump()` fns
/// that walk the variant's `FORWARD_TABLE` and call
/// [`normalize_slice`] on each entry's backbone + lm_head.
#[derive(Clone, Debug)]
pub struct VariantDump {
    pub variant_stem: &'static str,
    pub buckets: Vec<BucketDump>,
}

/// Per-arch entry in the backbone-dump registry. The `dump_all` fn
/// walks every variant in this arch's compiled set and returns a
/// flat list. One [`BackboneDumpRegistration`] per `#[forward]` —
/// inventory-collected, mirroring the existing
/// `FerriteArchRegistration` pattern.
pub struct BackboneDumpRegistration {
    pub arch_name: &'static str,
    pub dump_all: fn() -> Vec<VariantDump>,
}

inventory::collect!(BackboneDumpRegistration);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instr::CanonicalParams;
    use ferrite_kernels::layers::{Embedding, LinearLayer, RmsNorm};

    /// Stub Weights — `normalize` doesn't invoke any of these
    /// pointers, so unimplemented bodies are fine.
    struct W;

    impl CanonicalParams for W {
        const HEAD_DIM: u32 = 0;
        const NUM_Q_HEADS: u32 = 0;
        const NUM_KV_HEADS: u32 = 0;
        const Q_SIZE: usize = 0;
        const KV_SIZE: usize = 0;
        const INTERMEDIATE_SIZE: usize = 0;
        const ATTN_SCALE: f32 = 0.0;
        const ATTN_SOFTCAP: f32 = 0.0;
        const SLIDING_WINDOW: i32 = -1;
        const KV_LORA_RANK: usize = 0;
        const QK_NOPE_HEAD_DIM: usize = 0;
        const QK_ROPE_HEAD_DIM: usize = 0;
        const V_HEAD_DIM: usize = 0;
        const FINAL_LOGIT_SOFTCAPPING: f32 = 0.0;
        const QK_HEAD_DIM: usize = 0;
        const MLA_ATTN_SCALE: f32 = 0.0;
    }

    fn embed_wf(_: &W, _: u32) -> &Embedding {
        unimplemented!()
    }
    fn rmsnorm_wf(_: &W, _: u32) -> &RmsNorm {
        unimplemented!()
    }
    fn linear_wf(_: &W, _: u32) -> &LinearLayer {
        unimplemented!()
    }

    #[test]
    fn embed_normalizes() {
        let i: Instruction<W> = Instruction::Embed(7, embed_wf);
        let n = i.normalize();
        assert_eq!(n.kind, "Embed");
        assert_eq!(
            n.fields,
            vec![
                NormalizedField::Slot(7),
                NormalizedField::LayerKind("Embedding"),
            ]
        );
    }

    #[test]
    fn fused_add_rmsnorm_carries_layer_and_kernel_class() {
        let i: Instruction<W> = Instruction::FusedAddRmsNorm(3, 4, 12, rmsnorm_wf);
        let n = i.normalize();
        assert_eq!(n.kind, "FusedAddRmsNorm");
        assert_eq!(
            n.fields,
            vec![
                NormalizedField::Slot(3),
                NormalizedField::Slot(4),
                NormalizedField::Layer(12),
                NormalizedField::LayerKind("RmsNorm"),
            ]
        );
    }

    #[test]
    fn cutlass_gemm_add_keeps_tile_consts() {
        let i: Instruction<W> = Instruction::CutlassGemmAdd(5, 6, 0, linear_wf, 128, 128, 3);
        let n = i.normalize();
        assert_eq!(n.kind, "CutlassGemmAdd");
        assert_eq!(
            n.fields,
            vec![
                NormalizedField::Slot(5),
                NormalizedField::Slot(6),
                NormalizedField::Layer(0),
                NormalizedField::LayerKind("LinearLayer"),
                NormalizedField::ConstU32(128),
                NormalizedField::ConstU32(128),
                NormalizedField::ConstU32(3),
            ]
        );
    }

    #[test]
    fn loop_count_and_body_len_are_distinguishable() {
        let i: Instruction<W> = Instruction::Loop(32, 9);
        let n = i.normalize();
        assert_eq!(n.kind, "Loop");
        assert_eq!(
            n.fields,
            vec![
                NormalizedField::LoopCount(32),
                NormalizedField::LoopBodyLen(9),
            ]
        );
    }

    #[test]
    fn flashinfer_decode_keeps_head_dim_and_softcap_flag() {
        fn cs(_: &W, _: u32) -> ferrite_cuda_core::tensor::GpuTensor {
            unimplemented!()
        }
        let i: Instruction<W> = Instruction::FlashInferAttentionDecode(0, 1, 0, cs, 128, true);
        let n = i.normalize();
        assert_eq!(n.kind, "FlashInferAttentionDecode");
        assert_eq!(
            n.fields,
            vec![
                NormalizedField::Slot(0),
                NormalizedField::Slot(1),
                NormalizedField::Layer(0),
                NormalizedField::RopeCosSin,
                NormalizedField::ConstU32(128),
                NormalizedField::ConstBool(true),
            ]
        );
    }
}
