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

#![cfg(any(feature = "cuda", feature = "metal"))]

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
    /// Static weight matrix shape for a GEMM-class step, captured at
    /// codegen time from the FUF's `eval_shape`. `n` is output dim
    /// (LinearLayer's out_features), `k` is reduction dim
    /// (in_features). M is workload-dependent — the dump consumer
    /// derives it from the bucket's `m_min..m_max_excl`.
    ///
    /// Emitted by simple-GEMM Instruction variants only (one matmul,
    /// one weight) — `Gemm`, `CutlassGemm`, `CutlassGemmSplitK`,
    /// `CutlassGemmAdd`, `CutlassGemv`. Fused QKV / GateUp variants
    /// have multiple participating shapes and don't emit this.
    WeightShape { n: u32, k: u32 },
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
            Instruction::MeanSubRmsNorm(in_slot, out_slot, layer, _wf) => (
                "MeanSubRmsNorm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("RmsNorm"),
                ],
            ),
            Instruction::MeanSubRmsNormBiasAdd(in_slot, out_slot, layer, _wf) => (
                "MeanSubRmsNormBiasAdd",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LayerNorm"),
                ],
            ),
            Instruction::Reshape(in_slot, out_slot, dims_lit, dims_nt_pow, dims_div_lit, ndim) => {
                let n = ndim as usize;
                (
                    "Reshape",
                    vec![
                        F::Slot(in_slot),
                        F::Slot(out_slot),
                        F::ConstU32Array(dims_lit[..n].to_vec()),
                        F::ConstU8Array(dims_nt_pow[..n].to_vec()),
                        F::ConstU32Array(dims_div_lit[..n].to_vec()),
                        F::ConstU32(u32::from(ndim)),
                    ],
                )
            }
            Instruction::Add(delta_slot, residual_slot) => {
                ("Add", vec![F::Slot(delta_slot), F::Slot(residual_slot)])
            }
            #[cfg(feature = "nccl")]
            Instruction::AllReduce(slot) => ("AllReduce", vec![F::Slot(slot)]),
            #[cfg(feature = "nccl")]
            Instruction::AllGather(in_slot, out_slot) => {
                ("AllGather", vec![F::Slot(in_slot), F::Slot(out_slot)])
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
            Instruction::CutlassFusedRmsNormGemm(
                in_slot,
                out_slot,
                layer,
                _nwf,
                _gwf,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => (
                "CutlassFusedRmsNormGemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("RmsNorm"),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::CutlassFusedMeanSubRmsNormGemm(
                in_slot,
                out_slot,
                layer,
                _nwf,
                _gwf,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => (
                "CutlassFusedMeanSubRmsNormGemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("RmsNorm"),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::CutlassFusedAddRmsNormGemm(
                delta_slot,
                residual_slot,
                out_slot,
                layer,
                _nwf,
                _gwf,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => (
                "CutlassFusedAddRmsNormGemm",
                vec![
                    F::Slot(delta_slot),
                    F::Slot(residual_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("RmsNorm"),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::Gemm(in_slot, out_slot, layer, _wf, n, k) => (
                "Gemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::FusedCublasGemmAdd(in_slot, residual_slot, layer, _wf, n, k) => (
                "FusedCublasGemmAdd",
                vec![
                    F::Slot(in_slot),
                    F::Slot(residual_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::WeightShape { n, k },
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
            Instruction::AttentionPrefillPaged(q_slot, out_slot, layer, interleaved) => (
                "AttentionPrefillPaged",
                vec![
                    F::Slot(q_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::ConstBool(interleaved),
                ],
            ),
            Instruction::EncoderAttention(q_slot, k_slot, v_slot, out_slot) => (
                "EncoderAttention",
                vec![
                    F::Slot(q_slot),
                    F::Slot(k_slot),
                    F::Slot(v_slot),
                    F::Slot(out_slot),
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
            Instruction::VarlenAttention(q_slot, k_slot, v_slot, out_slot, cu_seqlens_kind) => (
                "VarlenAttention",
                vec![
                    F::Slot(q_slot),
                    F::Slot(k_slot),
                    F::Slot(v_slot),
                    F::Slot(out_slot),
                    F::ConstU32(u32::from(cu_seqlens_kind)),
                ],
            ),
            Instruction::VisionRope(q_slot, k_slot, q_out_slot, k_out_slot) => (
                "VisionRope",
                vec![
                    F::Slot(q_slot),
                    F::Slot(k_slot),
                    F::Slot(q_out_slot),
                    F::Slot(k_out_slot),
                ],
            ),
            Instruction::QuickGelu(in_slot, out_slot) => {
                ("QuickGelu", vec![F::Slot(in_slot), F::Slot(out_slot)])
            }
            Instruction::GeluErf(in_slot, out_slot) => {
                ("GeluErf", vec![F::Slot(in_slot), F::Slot(out_slot)])
            }
            Instruction::Gelu(in_slot, out_slot) => {
                ("Gelu", vec![F::Slot(in_slot), F::Slot(out_slot)])
            }
            Instruction::PosEmbed(out_slot, _wf) => (
                "PosEmbed",
                vec![F::Slot(out_slot), F::LayerKind("Embedding")],
            ),
            Instruction::LoadPixels(out_slot) => ("LoadPixels", vec![F::Slot(out_slot)]),
            Instruction::EmbeddingGather(in_slot, out_slot, indices_kind) => (
                "EmbeddingGather",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::ConstU32(u32::from(indices_kind)),
                ],
            ),
            Instruction::AvgPool2d(in_slot, out_slot) => {
                ("AvgPool2d", vec![F::Slot(in_slot), F::Slot(out_slot)])
            }
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
            Instruction::DeepSeekMoeFp8Block(in_slot, out_slot, layer, _wf) => (
                "DeepSeekMoeFp8Block",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("DeepSeekV2Fp8BlockMoELayer"),
                ],
            ),
            Instruction::DeepSeekMoeGgml(in_slot, out_slot, layer, _wf) => (
                "DeepSeekMoeGgml",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("DeepSeekV2GgmlMoELayer"),
                ],
            ),
            Instruction::FusedMoe(in_slot, out_slot, layer, _wf) => (
                "FusedMoe",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("FusedMoELayer"),
                ],
            ),
            Instruction::SharedFusedMoe(in_slot, out_slot, layer, _wf) => (
                "SharedFusedMoe",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("SharedFusedMoELayer"),
                ],
            ),
            Instruction::CutlassGemm(
                in_slot,
                out_slot,
                layer,
                _wf,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => (
                "CutlassGemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n, k },
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
                n,
                k,
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
                    F::WeightShape { n, k },
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
                n,
                k,
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
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::CutlassGemv(in_slot, out_slot, layer, _wf, n, k) => (
                "CutlassGemv",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::CutlassFusedGemmBias(
                in_slot,
                out_slot,
                layer,
                _wf,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => (
                "CutlassFusedGemmBias",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::CutlassFusedGateUpSiluMul(
                in_slot,
                out_slot,
                layer,
                _wf,
                tile_m,
                tile_n,
                stages,
            ) => (
                "CutlassFusedGateUpSiluMul",
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
            Instruction::CutlassFusedGateUpGeluMul(
                in_slot,
                out_slot,
                layer,
                _wf,
                tile_m,
                tile_n,
                stages,
                packed_n,
                k,
            ) => (
                "CutlassFusedGateUpGeluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n: packed_n, k },
                ],
            ),
            Instruction::CutlassFusedQkvRopeCache(
                in_slot,
                out_slot,
                layer,
                _wf,
                _cs,
                interleaved,
                tile_m,
                tile_n,
                stages,
                packed_n,
                k,
            ) => (
                "CutlassFusedQkvRopeCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::RopeCosSin,
                    F::ConstBool(interleaved),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n: packed_n, k },
                ],
            ),
            Instruction::CutlassFusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                _wf,
                _cs,
                interleaved,
                tile_m,
                tile_n,
                stages,
                packed_n,
                k,
            ) => (
                "CutlassFusedQkvRopePrefill",
                vec![
                    F::Slot(in_slot),
                    F::Slot(q_out_slot),
                    F::Slot(k_out_slot),
                    F::Slot(v_out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::RopeCosSin,
                    F::ConstBool(interleaved),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n: packed_n, k },
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
            Instruction::GgmlGemm(in_slot, out_slot, layer, _wf) => (
                "GgmlGemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::GgmlFusedGateUpSiluMul(in_slot, out_slot, layer, _wf) => (
                "GgmlFusedGateUpSiluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::GgmlFusedGateUpGeluMul(in_slot, out_slot, layer, _wf) => (
                "GgmlFusedGateUpGeluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::GgmlFusedQkvRopeCache(in_slot, out_slot, layer, _wf, _cs, _i) => (
                "GgmlFusedQkvRopeCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::RopeCosSin,
                ],
            ),
            Instruction::GgmlFusedQkvRopePrefill(in_slot, q_out, k_out, v_out, layer, _wf, _cs) => {
                (
                    "GgmlFusedQkvRopePrefill",
                    vec![
                        F::Slot(in_slot),
                        F::Slot(q_out),
                        F::Slot(k_out),
                        F::Slot(v_out),
                        F::Layer(layer),
                        F::LayerKind("LinearLayer"),
                        F::RopeCosSin,
                    ],
                )
            }
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
            // MLX-affine int4 family. `AffineQmm` is the metal-only
            // quantized matmul; CUDA `eval` is `unreachable!`, but the
            // variant is unconditional on `Instruction<W>` so info.rs
            // must cover it under either backend. `SiluMul` is the
            // P3-P4 C4a fused activation that pairs with two
            // `AffineQmm`s on the q-MLP decomposed branch.
            Instruction::AffineQmm(
                in_slot,
                out_slot,
                layer,
                _wf,
                n,
                k,
                _group_size,
                _bits,
                _vector_limit,
            ) => (
                "AffineQmm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::SynthPreAttn(
                residual_slot,
                delta_slot,
                out_slot,
                layer,
                _wf,
                _rms_wf,
                _cs_fn,
                _group_size,
                _bits,
                _symbol,
            ) => (
                "SynthPreAttn",
                vec![
                    F::Slot(residual_slot),
                    F::Slot(delta_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::SiluMul(gate_slot, up_slot, out_slot) => (
                "SiluMul",
                vec![F::Slot(gate_slot), F::Slot(up_slot), F::Slot(out_slot)],
            ),
            // Metal-only fused gather+dequant for `*-4bit` checkpoints
            // (P6). Variant is cfg-gated on `Instruction<W>` so the
            // arm matches its gate.
            #[cfg(feature = "metal")]
            Instruction::AffineEmbed(out_slot, _wf, _group_size, _bits) => (
                "AffineEmbed",
                vec![
                    F::Slot(out_slot),
                    F::LayerKind("AffineQuantEmbedding"),
                ],
            ),
            Instruction::Loop(count, body_len) => {
                ("Loop", vec![F::LoopCount(count), F::LoopBodyLen(body_len)])
            }
            Instruction::Alias(dst, src) => ("Alias", vec![F::Slot(dst), F::Slot(src)]),
            Instruction::Free(slot) => ("Free", vec![F::Slot(slot)]),
            Instruction::SpliceMmEmbeds(slot) => ("SpliceMmEmbeds", vec![F::Slot(slot)]),
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
    /// Tensor-parallel world size this variant was compiled for. At
    /// `--features nccl` the macro emits one `VariantDump` per (model,
    /// tp) tuple in `{1, 2, 4, 8}`; at default `--features cuda` only
    /// tp=1 is emitted. `vllm ferrite info` displays this on each row
    /// and includes it in its substring filter as `tp=N`.
    pub tp_world_size: u8,
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
        let i: Instruction<W> =
            Instruction::CutlassGemmAdd(5, 6, 0, linear_wf, 128, 128, 3, 4096, 11008);
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
                NormalizedField::WeightShape { n: 4096, k: 11008 },
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
