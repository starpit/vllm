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
//! ## AttentionViaCache
//! - `0`: `HEAD_DIM` (uint)
//! - `1`: `NUM_Q_HEADS` (uint)
//! - `2`: `NUM_KV_HEADS` (uint)
//! - `3`: `ATTN_SCALE` (float) — `W::ATTN_SCALE` (1/sqrt(head_dim) by default)
//! - `4`: `BLOCK_SIZE` (uint) — paged-cache block stride (e.g. 16); from `KernelExtras`
//! - `5`: `MAX_BLOCKS_PER_SEQ` (uint) — `block_table` row stride; from `KernelExtras`
//!
//! ## AttentionPrefillContiguous
//! - `0`: `HEAD_DIM` (uint)
//! - `1`: `NUM_Q_HEADS` (uint)
//! - `2`: `NUM_KV_HEADS` (uint)
//! - `3`: `ATTN_SCALE` (float) — `W::ATTN_SCALE` (1/sqrt(head_dim) by default)
//! - `6`: `PREFILL_TILE_Q` (uint) — Q-axis tile from the lowering pass
//!         (currently 16, baked from `KernelExtras::prefill_tile_q`).
//!         Index `6` (not `4`) so AttentionViaCache and
//!         AttentionPrefillContiguous can coexist in the same `.metal`
//!         file without a function-constant index collision.
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
    /// Paged-KV-cache block stride for `AttentionViaCache`. The cache
    /// layout this targets is `[num_blocks, num_kv_heads, block_size,
    /// head_dim]`; reading a token's K/V vector is one contiguous
    /// `head_dim` slice. `0` is invalid for cache-reading kernels;
    /// the model meta supplies `16` (vLLM's default) or whatever the
    /// allocator chose.
    pub block_size: u32,
    /// Row stride of the `block_table` runtime buffer (in u32s),
    /// equal to `ceil(MAX_SEQ_LEN / block_size)`. Baked per-bucket so
    /// the kernel can index `block_table[seq * MAX_BLOCKS_PER_SEQ +
    /// logical_block]` without a runtime divide. `0` is invalid for
    /// `AttentionViaCache`.
    pub max_blocks_per_seq: u32,
    /// Q-axis tile size for `AttentionPrefillContiguous`. The kernel
    /// processes `prefill_tile_q` query tokens per threadgroup. Must
    /// match the lowering pass's `PREFILL_TILE_Q` constant. `0`
    /// selects the default (16); the model meta usually leaves this
    /// at the default.
    pub prefill_tile_q: u32,
}

impl KernelExtras {
    pub const NONE: Self = Self {
        eps: 0.0,
        attn_scale: 0.0,
        rot_dim: 0,
        block_size: 0,
        max_blocks_per_seq: 0,
        prefill_tile_q: 0,
    };
}

/// Default Q-axis tile size for `AttentionPrefillContiguous` — must
/// match the lowering pass's `PREFILL_TILE_Q`.
pub(crate) const DEFAULT_PREFILL_TILE_Q: u32 = 16;

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
        KernelId::Embed => ("embed", "embed_f16_specialized"),
        KernelId::RmsNorm => ("rmsnorm", "rmsnorm_f16_specialized"),
        KernelId::FusedAddRmsNorm => {
            ("fused_add_rmsnorm", "fused_add_rmsnorm_f16_specialized")
        }
        KernelId::FusedGateUpSiluMul => (
            "fused_gate_up_silu_mul",
            "fused_gate_up_silu_mul_f16_specialized",
        ),
        KernelId::RopeAppend => ("rope", "rope_append_f16_specialized"),
        KernelId::AttentionViaCache => ("attention", "attention_via_cache_f16_specialized"),
        KernelId::AttentionPrefillContiguous => {
            ("attention", "attention_prefill_contiguous_f16_specialized")
        }
        KernelId::Add => ("elementwise", "residual_add_f16_specialized"),
        KernelId::ScalarMul => ("elementwise", "scalar_mul_f16_specialized"),
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
            // `block_size` reaches the kernel as a function constant
            // (index 4) so the paged-write index math folds at
            // pipeline-build time. The same value plumbs into
            // `AttentionViaCache` (index 4) — the model_meta supplies
            // it once per layer via `KernelExtras::block_size`.
            vec![
                ConstantValue::uint(0, W::HEAD_DIM),
                ConstantValue::uint(1, W::NUM_Q_HEADS),
                ConstantValue::uint(2, W::NUM_KV_HEADS),
                ConstantValue::uint(3, rot_dim),
                ConstantValue::uint(4, extras.block_size),
            ]
        }
        KernelId::AttentionViaCache => {
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
                ConstantValue::uint(4, extras.block_size),
                ConstantValue::uint(5, extras.max_blocks_per_seq),
            ]
        }
        KernelId::AttentionPrefillContiguous => {
            let scale = if extras.attn_scale == 0.0 {
                W::ATTN_SCALE
            } else {
                extras.attn_scale
            };
            let tile_q = if extras.prefill_tile_q == 0 {
                DEFAULT_PREFILL_TILE_Q
            } else {
                extras.prefill_tile_q
            };
            vec![
                ConstantValue::uint(0, W::HEAD_DIM),
                ConstantValue::uint(1, W::NUM_Q_HEADS),
                ConstantValue::uint(2, W::NUM_KV_HEADS),
                ConstantValue::float(3, scale),
                ConstantValue::uint(6, tile_q),
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
    fn attention_via_cache_uses_canonical_scale_when_extras_is_zero() {
        let extras = KernelExtras {
            block_size: 16,
            max_blocks_per_seq: 128,
            ..KernelExtras::NONE
        };
        let bag =
            constants_for::<TinyLlamaProbe>(KernelId::AttentionViaCache, 1, extras).expect("attn bag");
        assert_eq!(bag.len(), 6);
        assert_eq!(bag[0], ConstantValue::uint(0, 64));
        assert_eq!(bag[3], ConstantValue::float(3, 0.125));
        assert_eq!(bag[4], ConstantValue::uint(4, 16));
        assert_eq!(bag[5], ConstantValue::uint(5, 128));
    }

    #[test]
    fn attention_prefill_falls_back_to_default_tile_q() {
        let bag = constants_for::<TinyLlamaProbe>(
            KernelId::AttentionPrefillContiguous,
            16,
            KernelExtras::NONE,
        )
        .expect("prefill bag");
        assert_eq!(bag.len(), 5);
        assert_eq!(bag[0], ConstantValue::uint(0, 64));
        assert_eq!(bag[4], ConstantValue::uint(6, DEFAULT_PREFILL_TILE_Q));
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

    /// Device-bound smoke test for the Phase 5.B.3 specialized
    /// rmsnorm shader rewrite. Requires a Metal device, so it
    /// silently passes on non-Apple hardware.
    #[cfg(target_os = "macos")]
    #[test]
    fn rmsnorm_pipeline_builds_and_caches() {
        let Some(device) = ferrite_metal_kernels::detect_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let cache =
            ferrite_metal_kernels::specialized_pipeline_cache::SpecializedPipelineCache::with_standard_shaders(
                device.device.clone(),
            )
            .expect("compile standard shaders");
        let pipelines = SpecializedPipelines::new(std::sync::Arc::new(cache));

        // RmsNorm at bucket=1 (decode) and bucket=8 (small prefill).
        let extras = KernelExtras {
            eps: 1e-5,
            ..KernelExtras::NONE
        };
        let _p1 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::RmsNorm, 1, extras)
            .expect("rmsnorm bucket=1");
        let _p8 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::RmsNorm, 8, extras)
            .expect("rmsnorm bucket=8");
        let _p1_again = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::RmsNorm, 1, extras)
            .expect("rmsnorm bucket=1 cache hit");
        // Two pipelines (one per bucket); third call hits the cache.
        assert_eq!(pipelines.cached_count(), 2);

        // FusedAddRmsNorm at the same buckets, same eps. Adds two
        // more entries; full cache count should be 4.
        let _f1 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::FusedAddRmsNorm, 1, extras)
            .expect("fused_add_rmsnorm bucket=1");
        let _f8 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::FusedAddRmsNorm, 8, extras)
            .expect("fused_add_rmsnorm bucket=8");
        assert_eq!(pipelines.cached_count(), 4);

        // FusedGateUpSiluMul (Phase 5.B.4 scope — silu only; gemm is
        // opaque/MPS). Two buckets → six pipelines total.
        let _g1 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::FusedGateUpSiluMul, 1, KernelExtras::NONE)
            .expect("silu bucket=1");
        let _g8 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::FusedGateUpSiluMul, 8, KernelExtras::NONE)
            .expect("silu bucket=8");
        assert_eq!(pipelines.cached_count(), 6);
    }

    /// Device-bound smoke test for the Phase 5.C.4 specialized
    /// attention shader rewrite. Verifies both
    /// `attention_via_cache_f16_specialized` and
    /// `attention_prefill_contiguous_f16_specialized` compile against
    /// `MTLFunctionConstantValues` carrying their respective bag, and
    /// that the cache returns the same handle on a repeat lookup.
    #[cfg(target_os = "macos")]
    #[test]
    fn attention_pipelines_build_and_cache() {
        let Some(device) = ferrite_metal_kernels::detect_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let cache =
            ferrite_metal_kernels::specialized_pipeline_cache::SpecializedPipelineCache::with_standard_shaders(
                device.device.clone(),
            )
            .expect("compile standard shaders");
        let pipelines = SpecializedPipelines::new(std::sync::Arc::new(cache));

        // AttentionViaCache: bucket=1 (decode), block_size=16,
        // max_blocks_per_seq=128 (room for ~2k KV tokens).
        let decode_extras = KernelExtras {
            block_size: 16,
            max_blocks_per_seq: 128,
            ..KernelExtras::NONE
        };
        let _d1 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::AttentionViaCache, 1, decode_extras)
            .expect("attention_via_cache bucket=1");
        let _d1_again = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::AttentionViaCache, 1, decode_extras)
            .expect("attention_via_cache bucket=1 cache hit");
        assert_eq!(pipelines.cached_count(), 1);

        // AttentionPrefillContiguous at bucket=16 (one full
        // PREFILL_TILE_Q tile) and bucket=64.
        let _p16 = pipelines
            .pipeline_for::<TinyLlamaProbe>(
                KernelId::AttentionPrefillContiguous,
                16,
                KernelExtras::NONE,
            )
            .expect("attention_prefill bucket=16");
        let _p64 = pipelines
            .pipeline_for::<TinyLlamaProbe>(
                KernelId::AttentionPrefillContiguous,
                64,
                KernelExtras::NONE,
            )
            .expect("attention_prefill bucket=64");
        // PREFILL_TILE_Q is the only `M`-derived constant baked, so
        // the bucket axis is collapsed for prefill — both 16 and 64
        // share the same cache entry. Total now = 2.
        assert_eq!(pipelines.cached_count(), 2);
    }

    /// Device-bound smoke test for the Phase 5.G.3 specialized
    /// `rope_append_f16_specialized` shader. Verifies the kernel
    /// compiles against the five-element `MTLFunctionConstantValues`
    /// bag (`HEAD_DIM`, `NUM_Q_HEADS`, `NUM_KV_HEADS`, `ROT_DIM`,
    /// `BLOCK_SIZE`) the lowering pass emits, and that the cache
    /// returns the same handle on a repeat lookup.
    ///
    /// `bucket_m` is not part of RopeAppend's constant bag (the
    /// kernel reads `tg_pos.x` over the dispatch range), so different
    /// buckets share one pipeline entry — same collapse pattern as
    /// `AttentionPrefillContiguous`.
    #[cfg(target_os = "macos")]
    #[test]
    fn rope_append_pipeline_builds_and_caches() {
        let Some(device) = ferrite_metal_kernels::detect_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let cache =
            ferrite_metal_kernels::specialized_pipeline_cache::SpecializedPipelineCache::with_standard_shaders(
                device.device.clone(),
            )
            .expect("compile standard shaders");
        let pipelines = SpecializedPipelines::new(std::sync::Arc::new(cache));

        let extras = KernelExtras {
            block_size: 16,
            ..KernelExtras::NONE
        };
        let _r1 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::RopeAppend, 1, extras)
            .expect("rope_append bucket=1");
        let _r8 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::RopeAppend, 8, extras)
            .expect("rope_append bucket=8");
        let _r1_again = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::RopeAppend, 1, extras)
            .expect("rope_append bucket=1 cache hit");
        // Bucket axis collapsed (no `M`-derived constant); both 1 and
        // 8 share one entry. Repeat at bucket=1 hits cache.
        assert_eq!(pipelines.cached_count(), 1);
    }

    /// Device-bound numerical-correctness check for the Phase 5.G.3
    /// `rope_append_f16_specialized` kernel against
    /// `cpu_golden::rope_append`. Allocates synthetic Q/K/V/cos_sin/
    /// positions/slot_mapping/kv_cache buffers, dispatches the
    /// specialized kernel directly (no ICB/worker — just the pipeline
    /// + a fresh compute encoder), reads back the f16 outputs, and
    /// asserts max-abs error < 5e-3 vs the cpu_golden ref.
    ///
    /// 5e-3 tolerance covers f16 round-tripping (3 ULP at typical
    /// Q magnitudes) plus the MSL `half(...)` rounding mode, which
    /// is round-to-nearest-even on Apple Silicon — same as `f16::from_f32`.
    #[cfg(target_os = "macos")]
    #[test]
    fn rope_append_matches_cpu_golden() {
        use crate::cpu_golden;
        use ferrite_metal_kernels::metal::MTLSize;

        let Some(device_info) = ferrite_metal_kernels::detect_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = device_info.device.clone();
        let queue = device.new_command_queue();

        let cache =
            ferrite_metal_kernels::specialized_pipeline_cache::SpecializedPipelineCache::with_standard_shaders(
                device.clone(),
            )
            .expect("compile standard shaders");
        let pipelines = SpecializedPipelines::new(std::sync::Arc::new(cache));

        // Shape: TinyLlama-1.1B params, bucket_m = 2 (small for test),
        // BLOCK_SIZE = 16. The kernel is bucket-axis-independent.
        let bucket_m: usize = 2;
        let head_dim = TinyLlamaProbe::HEAD_DIM as usize;
        let half_dim = head_dim / 2;
        let num_q = TinyLlamaProbe::NUM_Q_HEADS as usize;
        let num_kv = TinyLlamaProbe::NUM_KV_HEADS as usize;
        let block_size: usize = 16;
        let num_blocks: usize = 4;
        let max_pos: usize = 32;

        let extras = KernelExtras {
            block_size: block_size as u32,
            ..KernelExtras::NONE
        };
        let pipeline = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::RopeAppend, bucket_m as u32, extras)
            .expect("rope_append pipeline");

        // Synthetic deterministic inputs.
        let q_data: Vec<f32> = (0..bucket_m * num_q * head_dim)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect();
        let k_data: Vec<f32> = (0..bucket_m * num_kv * head_dim)
            .map(|i| ((i as f32) * 0.002).cos())
            .collect();
        let v_data: Vec<f32> = (0..bucket_m * num_kv * head_dim)
            .map(|i| (i as f32) * 0.003)
            .collect();

        // cos_sin layout: [max_pos, head_dim] = [cos[half] | sin[half]].
        let mut cos_sin_data = vec![0.0f32; max_pos * head_dim];
        for pos in 0..max_pos {
            for d in 0..half_dim {
                let theta =
                    (pos as f32) / 10000.0_f32.powf((d as f32) / (half_dim as f32));
                cos_sin_data[pos * head_dim + d] = theta.cos();
                cos_sin_data[pos * head_dim + half_dim + d] = theta.sin();
            }
        }

        let positions = vec![5u32, 7];
        // Two distinct slots — one in block 0 (offset 3), one in block 1 (offset 3).
        let slot_mapping = vec![3u32, (block_size as u32) + 3];

        // Buffer helpers.
        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_f16(device: &Device, data: &[f32]) -> Buffer {
            let half_data: Vec<half::f16> =
                data.iter().map(|&v| half::f16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(half_data.as_slice());
            let buf = device
                .new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    half_data.as_ptr() as *const u8,
                    buf.contents() as *mut u8,
                    bytes,
                );
            }
            buf
        }
        fn alloc_u32(device: &Device, data: &[u32]) -> Buffer {
            let bytes = std::mem::size_of_val(data);
            let buf = device
                .new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    buf.contents() as *mut u8,
                    bytes,
                );
            }
            buf
        }
        fn alloc_zero_f16(device: &Device, n: usize) -> Buffer {
            let bytes = (n * std::mem::size_of::<half::f16>()).max(1);
            let buf = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes);
            }
            buf
        }

        let q_buf = alloc_f16(&device, &q_data);
        let k_buf = alloc_f16(&device, &k_data);
        let v_buf = alloc_f16(&device, &v_data);
        let cos_sin_buf = alloc_f16(&device, &cos_sin_data);
        let positions_buf = alloc_u32(&device, &positions);
        let slot_buf = alloc_u32(&device, &slot_mapping);
        let kv_k_buf = alloc_zero_f16(&device, num_blocks * num_kv * block_size * head_dim);
        let kv_v_buf = alloc_zero_f16(&device, num_blocks * num_kv * block_size * head_dim);

        // Encode + dispatch.
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&k_buf), 0);
        enc.set_buffer(2, Some(&v_buf), 0);
        enc.set_buffer(3, Some(&cos_sin_buf), 0);
        enc.set_buffer(4, Some(&positions_buf), 0);
        enc.set_buffer(5, Some(&slot_buf), 0);
        enc.set_buffer(6, Some(&kv_k_buf), 0);
        enc.set_buffer(7, Some(&kv_v_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(bucket_m as u64, num_q as u64, 1),
            MTLSize::new(head_dim as u64, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // Build separated cos/sin tables for cpu_golden::rope_append.
        let mut cos_table = vec![0.0_f32; max_pos * head_dim];
        let mut sin_table = vec![0.0_f32; max_pos * head_dim];
        for pos in 0..max_pos {
            for d in 0..half_dim {
                cos_table[pos * head_dim + d] = cos_sin_data[pos * head_dim + d];
                sin_table[pos * head_dim + d] = cos_sin_data[pos * head_dim + half_dim + d];
            }
        }

        let mut q_out_cpu = vec![0.0_f32; q_data.len()];
        let mut kv_k_cpu = vec![0.0_f32; num_blocks * num_kv * block_size * head_dim];
        let mut kv_v_cpu = vec![0.0_f32; num_blocks * num_kv * block_size * head_dim];
        cpu_golden::rope_append(
            &q_data,
            &k_data,
            &v_data,
            &positions,
            &slot_mapping,
            &cos_table,
            &sin_table,
            &mut q_out_cpu,
            &mut kv_k_cpu,
            &mut kv_v_cpu,
            num_q,
            num_kv,
            head_dim,
            block_size,
        );

        // Read back f16 buffers and convert to f32 for comparison.
        fn read_f16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const half::f16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let q_metal = read_f16(&q_buf, q_data.len());
        let kv_k_metal = read_f16(&kv_k_buf, kv_k_cpu.len());
        let kv_v_metal = read_f16(&kv_v_buf, kv_v_cpu.len());

        let tol: f32 = 5e-3;
        for i in 0..q_data.len() {
            let diff = (q_metal[i] - q_out_cpu[i]).abs();
            assert!(
                diff < tol,
                "q[{i}] metal={} cpu={} diff={}",
                q_metal[i],
                q_out_cpu[i],
                diff
            );
        }
        for i in 0..kv_k_cpu.len() {
            let diff = (kv_k_metal[i] - kv_k_cpu[i]).abs();
            assert!(
                diff < tol,
                "kv_k[{i}] metal={} cpu={} diff={}",
                kv_k_metal[i],
                kv_k_cpu[i],
                diff
            );
        }
        for i in 0..kv_v_cpu.len() {
            let diff = (kv_v_metal[i] - kv_v_cpu[i]).abs();
            assert!(
                diff < tol,
                "kv_v[{i}] metal={} cpu={} diff={}",
                kv_v_metal[i],
                kv_v_cpu[i],
                diff
            );
        }
    }
}
