// SPDX-License-Identifier: Apache-2.0
//! Glue between the Metal interpreter's [`KernelId`] taxonomy and
//! `ferrite-metal-kernels`'s [`SpecializedPipelineCache`].
//!
//! Phase 5.B's deliverable: turn `(KernelId, bucket M, CanonicalParams,
//! per-kernel extras)` into a `MTLComputePipelineState` whose
//! `[[function_constant(N)]]` slots are pre-baked. The `MetalWorker`
//! (Phase 5.C) consults this layer once per `(model variant, bucket,
//! kernel)` at init time; recording into the ICB happens against the
//! resulting pipeline.
//!
//! # Function-constant index assignments
//!
//! These are the contract between the MSL shader files and this glue
//! layer. The Phase 5.B.3 / 5.B.4 shader rewrites declare each
//! `[[function_constant(N)]]` at the index spelled here.
//!
//! ## RmsNorm / FusedAddRmsNorm
//! - `0`: `M` (uint)              — bucket token count
//! - `1`: `N`/`HIDDEN_SIZE` (uint) — `W::Q_SIZE`
//! - `2`: `EPS` (float)            — per-layer `RmsNorm.eps` (caller-supplied)
//!
//! ## FusedGateUpSiluMul
//! - `0`: `M` (uint)
//! - `1`: `N`/`INTERMEDIATE_SIZE` (uint) — `W::INTERMEDIATE_SIZE`
//!
//! ## Embed
//! - `0`: `HIDDEN_SIZE` (uint) — `W::Q_SIZE`
//!
//! ## RopeAppend
//! - `0`: `HEAD_DIM` (uint)
//! - `1`: `NUM_Q_HEADS` (uint)
//! - `2`: `NUM_KV_HEADS` (uint)
//! - `3`: `ROT_DIM` (uint) — for partial-rope models; equals `HEAD_DIM` by default
//!
//! ## AttentionViaCache / AttentionPrefillContiguous
//! - `0`: `HEAD_DIM` (uint)
//! - `1`: `NUM_Q_HEADS` (uint)
//! - `2`: `NUM_KV_HEADS` (uint)
//! - `3`: `ATTN_SCALE` (float) — `W::ATTN_SCALE` (1/sqrt(head_dim) by default)
//!
//! ## Add / ScalarMul
//! - No function constants in 5.B (kernels are token-parallel and
//!   read total-element count from dispatch shape). Reserved for
//!   future per-bucket fast paths.

#![cfg(feature = "metal")]

use std::sync::Arc;

use crate::CanonicalParams;
use ferrite_metal_kernels::metal::ComputePipelineState;
use ferrite_metal_kernels::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use ferrite_metal_kernels::stream::MetalStreamError;

use super::lowered::KernelId;

/// Per-kernel scalars the lowering pass cannot infer from
/// `CanonicalParams` alone (per-layer `eps`, dynamic scales, etc.).
/// Carried into [`SpecializedPipelines::pipeline_for`] alongside the
/// `KernelId` and bucket M.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KernelExtras {
    /// RmsNorm / FusedAddRmsNorm: the per-layer `RmsNorm.eps`. Worker
    /// resolves the weight bundle and reads this at pipeline-build
    /// time. `0.0` if the kernel does not consume `eps`.
    pub eps: f32,
    /// Override for `W::ATTN_SCALE` if the kernel needs a different
    /// scale (MlaAttention uses `MLA_ATTN_SCALE`). `0.0` selects
    /// `W::ATTN_SCALE`.
    pub attn_scale: f32,
    /// Partial-rope rotation dim. `0` selects `W::HEAD_DIM`.
    pub rot_dim: u32,
}

impl KernelExtras {
    pub const NONE: Self = Self {
        eps: 0.0,
        attn_scale: 0.0,
        rot_dim: 0,
    };
}

/// Library-and-function name pair for a [`KernelId`].
///
/// Library name picks the compiled MSL source file (matches the
/// `&'static str` keys the cache was constructed with). Kernel name
/// picks the actual `kernel void` symbol inside that library.
///
/// We default to the f16 variants because that's what every model in
/// the TinyLlama-class lineage uses; bf16 / vec4 specializations
/// surface as additional `KernelId` variants when their cost is
/// proven.
fn kernel_msl_names(kernel: KernelId) -> Result<(&'static str, &'static str), PipelineLookupError> {
    Ok(match kernel {
        KernelId::Embed => ("embed", "embed_tokens_f16"),
        KernelId::RmsNorm => ("rmsnorm", "rmsnorm_f16"),
        KernelId::FusedAddRmsNorm => ("fused_add_rmsnorm", "fused_add_rmsnorm_f16"),
        KernelId::FusedGateUpSiluMul => {
            ("fused_gate_up_silu_mul", "fused_gate_up_silu_mul_f16")
        }
        KernelId::RopeAppend => ("rope", "rope_append_f16"),
        KernelId::AttentionViaCache => ("attention", "attention_via_cache_f16"),
        KernelId::AttentionPrefillContiguous => {
            ("attention", "attention_prefill_contiguous_f16")
        }
        KernelId::Add => ("elementwise", "residual_add_f16"),
        KernelId::ScalarMul => ("elementwise", "scalar_mul_f16"),
        // GEMM does not get a function-constant pipeline at this
        // layer — Metal Performance Shaders' matmul2d is opaque.
        // Surfaces as a hard error so the worker (5.C) routes GEMM
        // through MPS rather than this cache.
        KernelId::Gemm => return Err(PipelineLookupError::OpaqueKernel(KernelId::Gemm)),
        // Reshape is a metadata-only op and never reaches this layer
        // (the lowering pass drops it). Defensive error in case a
        // future caller forgets.
        KernelId::Reshape => return Err(PipelineLookupError::MetadataOnly(KernelId::Reshape)),
    })
}

/// Build the function-constant bag for `kernel` from `W` and the
/// supplied extras. Kept as a free function so unit tests can call it
/// without standing up the full cache.
pub fn constants_for<W: CanonicalParams>(
    kernel: KernelId,
    bucket_m: u32,
    extras: KernelExtras,
) -> Result<Vec<ConstantValue>, PipelineLookupError> {
    let cv = match kernel {
        KernelId::RmsNorm | KernelId::FusedAddRmsNorm => vec![
            ConstantValue::uint(0, bucket_m),
            ConstantValue::uint(1, W::Q_SIZE as u32),
            ConstantValue::float(2, extras.eps),
        ],
        KernelId::FusedGateUpSiluMul => vec![
            ConstantValue::uint(0, bucket_m),
            ConstantValue::uint(1, W::INTERMEDIATE_SIZE as u32),
        ],
        KernelId::Embed => vec![ConstantValue::uint(0, W::Q_SIZE as u32)],
        KernelId::RopeAppend => {
            let rot_dim = if extras.rot_dim == 0 {
                W::HEAD_DIM
            } else {
                extras.rot_dim
            };
            vec![
                ConstantValue::uint(0, W::HEAD_DIM),
                ConstantValue::uint(1, W::NUM_Q_HEADS),
                ConstantValue::uint(2, W::NUM_KV_HEADS),
                ConstantValue::uint(3, rot_dim),
            ]
        }
        KernelId::AttentionViaCache | KernelId::AttentionPrefillContiguous => {
            let scale = if extras.attn_scale == 0.0 {
                W::ATTN_SCALE
            } else {
                extras.attn_scale
            };
            vec![
                ConstantValue::uint(0, W::HEAD_DIM),
                ConstantValue::uint(1, W::NUM_Q_HEADS),
                ConstantValue::uint(2, W::NUM_KV_HEADS),
                ConstantValue::float(3, scale),
            ]
        }
        KernelId::Add | KernelId::ScalarMul => Vec::new(),
        KernelId::Gemm => return Err(PipelineLookupError::OpaqueKernel(KernelId::Gemm)),
        KernelId::Reshape => return Err(PipelineLookupError::MetadataOnly(KernelId::Reshape)),
    };
    Ok(cv)
}

/// High-level wrapper around [`SpecializedPipelineCache`] that knows
/// the function-constant index layout for every [`KernelId`].
///
/// One instance per `(MetalDevice)`. The pool stashes it in `Arc`
/// and hands a clone to every worker.
pub struct SpecializedPipelines {
    cache: Arc<SpecializedPipelineCache>,
}

impl SpecializedPipelines {
    /// Wrap an already-constructed cache. The cache should have been
    /// built with [`SpecializedPipelineCache::with_standard_shaders`]
    /// so every library this glue references is compiled.
    pub fn new(cache: Arc<SpecializedPipelineCache>) -> Self {
        Self { cache }
    }

    /// Return the specialized pipeline for `(kernel, bucket_m, W, extras)`.
    /// First call builds; subsequent calls hit the cache.
    pub fn pipeline_for<W: CanonicalParams>(
        &self,
        kernel: KernelId,
        bucket_m: u32,
        extras: KernelExtras,
    ) -> Result<ComputePipelineState, PipelineLookupError> {
        let (library, function) = kernel_msl_names(kernel)?;
        let constants = constants_for::<W>(kernel, bucket_m, extras)?;
        let key = PipelineKey::new(library, function, constants);
        self.cache
            .get_or_build(&key)
            .map_err(PipelineLookupError::Build)
    }

    /// For diagnostics / tests: how many pipelines are currently
    /// cached. Useful for asserting that subsequent calls hit.
    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }
}

#[derive(Debug)]
pub enum PipelineLookupError {
    /// `KernelId::Gemm` is intentionally opaque (MPS-backed). Callers
    /// must route it through the GEMM wrapper, not this cache.
    OpaqueKernel(KernelId),
    /// `KernelId::Reshape` is metadata-only. Callers should drop it
    /// before reaching this layer; the lowering pass already does so.
    MetadataOnly(KernelId),
    /// Cache build error (shader compile, function constant mismatch,
    /// pipeline state construction). The wrapped variant carries the
    /// underlying message verbatim.
    Build(MetalStreamError),
}

impl std::fmt::Display for PipelineLookupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpaqueKernel(k) => write!(
                f,
                "specialized pipeline lookup: {k:?} is opaque (route through MPS GEMM wrapper)"
            ),
            Self::MetadataOnly(k) => write!(
                f,
                "specialized pipeline lookup: {k:?} is a metadata-only op (lowering should drop it)"
            ),
            Self::Build(e) => write!(f, "specialized pipeline build: {e}"),
        }
    }
}

impl std::error::Error for PipelineLookupError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CanonicalParams;

    /// Pure-CPU stub of `CanonicalParams` modelled on TinyLlama-1.1B.
    /// Lets us exercise [`constants_for`] without standing up a Metal
    /// device or a real model variant.
    struct TinyLlamaProbe;
    impl CanonicalParams for TinyLlamaProbe {
        const HEAD_DIM: u32 = 64;
        const NUM_Q_HEADS: u32 = 32;
        const NUM_KV_HEADS: u32 = 4;
        const Q_SIZE: usize = 2048;
        const KV_SIZE: usize = 256;
        const INTERMEDIATE_SIZE: usize = 5632;
        const ATTN_SCALE: f32 = 0.125;
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

    #[test]
    fn rmsnorm_constants_are_well_formed() {
        let extras = KernelExtras {
            eps: 1e-5,
            ..KernelExtras::NONE
        };
        let bag =
            constants_for::<TinyLlamaProbe>(KernelId::RmsNorm, 8, extras).expect("rmsnorm bag");
        assert_eq!(bag.len(), 3);
        assert_eq!(bag[0], ConstantValue::uint(0, 8));
        assert_eq!(bag[1], ConstantValue::uint(1, 2048));
        assert_eq!(bag[2], ConstantValue::float(2, 1e-5));
    }

    #[test]
    fn attention_uses_canonical_scale_when_extras_is_zero() {
        let bag = constants_for::<TinyLlamaProbe>(
            KernelId::AttentionViaCache,
            1,
            KernelExtras::NONE,
        )
        .expect("attn bag");
        assert_eq!(bag.len(), 4);
        assert_eq!(bag[0], ConstantValue::uint(0, 64));
        assert_eq!(bag[3], ConstantValue::float(3, 0.125));
    }

    #[test]
    fn fused_silu_bag_has_no_eps() {
        let bag = constants_for::<TinyLlamaProbe>(
            KernelId::FusedGateUpSiluMul,
            64,
            KernelExtras::NONE,
        )
        .expect("silu bag");
        assert_eq!(bag.len(), 2);
        assert_eq!(bag[0], ConstantValue::uint(0, 64));
        assert_eq!(bag[1], ConstantValue::uint(1, 5632));
    }

    #[test]
    fn gemm_is_rejected_as_opaque() {
        let err =
            constants_for::<TinyLlamaProbe>(KernelId::Gemm, 1, KernelExtras::NONE).unwrap_err();
        assert!(matches!(err, PipelineLookupError::OpaqueKernel(_)));
    }

    #[test]
    fn reshape_is_rejected_as_metadata_only() {
        let err = constants_for::<TinyLlamaProbe>(KernelId::Reshape, 1, KernelExtras::NONE)
            .unwrap_err();
        assert!(matches!(err, PipelineLookupError::MetadataOnly(_)));
    }
}
