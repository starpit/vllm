// SPDX-License-Identifier: Apache-2.0
//! Glue between the Metal interpreter's [`KernelId`] taxonomy and
//! `ferrite-metal-kernels`'s [`SpecializedPipelineCache`].
//!
//! Turns `(KernelId, bucket M, W: CanonicalParams)` into a
//! `MTLComputePipelineState` whose `[[function_constant(N)]]` slots
//! are pre-baked. The `MetalWorker` consults this layer once per
//! `(model variant, bucket, kernel)` at init time; recording into the
//! ICB happens against the resulting pipeline. Every per-layer scalar
//! (eps, attn_scale, paging strides) is a `CanonicalParams` constant
//! the macro emitted from the model JSON — no runtime extras struct.
//!
//! # Function-constant index assignments
//!
//! These are the contract between the MSL shader files and this glue
//! layer. The shader rewrites declare each `[[function_constant(N)]]`
//! at the index spelled here.
//!
//! ## RmsNorm / FusedAddRmsNorm
//! - `0`: `M` (uint) — bucket token count
//! - `1`: `N`/`HIDDEN_SIZE` (uint) — `W::Q_SIZE`
//! - `2`: `EPS` (float) — `W::RMS_NORM_EPS` (from model config)
//!
//! ## FusedGateUpSiluMul
//! - `0`: `M` (uint)
//! - `1`: `N`/`INTERMEDIATE_SIZE` (uint) — `W::INTERMEDIATE_SIZE`
//! - `2`: `K`/`HIDDEN_SIZE` (uint)         — `W::Q_SIZE`
//!
//! ## Embed
//! - `0`: `HIDDEN_SIZE` (uint) — `W::Q_SIZE`
//!
//! ## RopeAppend
//! - `0`: `HEAD_DIM` (uint)
//! - `1`: `NUM_Q_HEADS` (uint)
//! - `2`: `NUM_KV_HEADS` (uint)
//! - `3`: `ROT_DIM` (uint) — `W::ROT_DIM`; equals `HEAD_DIM` for full rope
//! - `4`: `BLOCK_SIZE` (uint) — `W::BLOCK_SIZE`
//!
//! ## AttentionViaCache
//! - `0`: `HEAD_DIM` (uint)
//! - `1`: `NUM_Q_HEADS` (uint)
//! - `2`: `NUM_KV_HEADS` (uint)
//! - `3`: `ATTN_SCALE` (float) — `W::ATTN_SCALE`
//! - `4`: `BLOCK_SIZE` (uint) — `W::BLOCK_SIZE`
//! - `5`: `MAX_BLOCKS_PER_SEQ` (uint) — `W::MAX_BLOCKS_PER_SEQ`
//!
//! ## AttentionPrefillContiguous
//! - `0`: `HEAD_DIM` (uint)
//! - `1`: `NUM_Q_HEADS` (uint)
//! - `2`: `NUM_KV_HEADS` (uint)
//! - `3`: `ATTN_SCALE` (float) — `W::ATTN_SCALE`
//! - `6`: `PREFILL_TILE_Q` (uint) — `W::PREFILL_TILE_Q`. Index `6`
//!   (not `4`) so AttentionViaCache and AttentionPrefillContiguous
//!   can coexist in the same `.metal` file without a function-
//!   constant index collision.
//!
//! ## Add / ScalarMul
//! - No function constants — kernels are token-parallel and read
//!   total-element count from dispatch shape.

#![cfg(feature = "metal")]

use std::sync::Arc;

use crate::CanonicalParams;
use ferrite_metal_kernels::metal::ComputePipelineState;
use ferrite_metal_kernels::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use ferrite_metal_kernels::stream::MetalStreamError;

use super::lowered::KernelId;

// `KernelExtras` and friends used to live here. Every field has been
// promoted to a `CanonicalParams` constant (`RMS_NORM_EPS`,
// `BLOCK_SIZE`, `MAX_BLOCKS_PER_SEQ`, `PREFILL_TILE_Q`, `ROT_DIM`)
// because the macro reads them from the model JSON at compile time
// and emits the per-canonical impl. `constants_for::<W>` reads
// directly from `W::*`; no runtime extras struct, no
// `MetalModelMeta::kernel_extras_for` callback, no plumbing.

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
fn kernel_msl_names(
    kernel: KernelId,
    bucket_m: u32,
) -> Result<(&'static str, &'static str), PipelineLookupError> {
    Ok(match kernel {
        KernelId::Embed => ("embed", "embed_f16_specialized"),
        KernelId::RmsNorm => ("rmsnorm", "rmsnorm_f16_specialized"),
        KernelId::FusedAddRmsNorm => ("fused_add_rmsnorm", "fused_add_rmsnorm_f16_specialized"),
        // Two specialized variants: the decode (M=1) variant uses
        // simd_sum dot products and avoids the simdgroup_matrix
        // overhead that would otherwise dominate at one-row inputs.
        // Prefill (M >= 2) uses the matrix variant.
        KernelId::FusedGateUpSiluMul if bucket_m == 1 => (
            "fused_gate_up_silu_mul",
            "fused_gate_up_silu_mul_decode_f16_specialized",
        ),
        KernelId::FusedGateUpSiluMul => (
            "fused_gate_up_silu_mul",
            "fused_gate_up_silu_mul_gemm_f16_specialized",
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
/// bucket M. Every per-layer scalar (eps, attn_scale, paging
/// strides) is a `CanonicalParams` constant the macro emitted from
/// the model config — there are no runtime "extras" to thread.
pub fn constants_for<W: CanonicalParams>(
    kernel: KernelId,
    bucket_m: u32,
) -> Result<Vec<ConstantValue>, PipelineLookupError> {
    let cv = match kernel {
        KernelId::RmsNorm | KernelId::FusedAddRmsNorm => vec![
            ConstantValue::uint(0, bucket_m),
            ConstantValue::uint(1, W::Q_SIZE as u32),
            ConstantValue::float(2, W::RMS_NORM_EPS),
        ],
        // Decode (M=1) variant carries (M, N, K) at indices 3/4/5
        // so the matrix variant's 0/1/2 don't clash in the shared
        // library when both kernels are compiled together.
        KernelId::FusedGateUpSiluMul if bucket_m == 1 => vec![
            ConstantValue::uint(3, bucket_m),
            ConstantValue::uint(4, W::INTERMEDIATE_SIZE as u32),
            ConstantValue::uint(5, W::Q_SIZE as u32),
        ],
        KernelId::FusedGateUpSiluMul => vec![
            ConstantValue::uint(0, bucket_m),
            ConstantValue::uint(1, W::INTERMEDIATE_SIZE as u32),
            ConstantValue::uint(2, W::Q_SIZE as u32),
        ],
        KernelId::Embed => vec![
            ConstantValue::uint(0, bucket_m),
            ConstantValue::uint(1, W::Q_SIZE as u32),
        ],
        KernelId::RopeAppend => vec![
            ConstantValue::uint(0, W::HEAD_DIM),
            ConstantValue::uint(1, W::NUM_Q_HEADS),
            ConstantValue::uint(2, W::NUM_KV_HEADS),
            ConstantValue::uint(3, W::ROT_DIM),
            ConstantValue::uint(4, W::BLOCK_SIZE),
        ],
        KernelId::AttentionViaCache => vec![
            ConstantValue::uint(0, W::HEAD_DIM),
            ConstantValue::uint(1, W::NUM_Q_HEADS),
            ConstantValue::uint(2, W::NUM_KV_HEADS),
            ConstantValue::float(3, W::ATTN_SCALE),
            ConstantValue::uint(4, W::BLOCK_SIZE),
            ConstantValue::uint(5, W::MAX_BLOCKS_PER_SEQ),
        ],
        KernelId::AttentionPrefillContiguous => vec![
            ConstantValue::uint(0, W::HEAD_DIM),
            ConstantValue::uint(1, W::NUM_Q_HEADS),
            ConstantValue::uint(2, W::NUM_KV_HEADS),
            ConstantValue::float(3, W::ATTN_SCALE),
            ConstantValue::uint(6, W::PREFILL_TILE_Q),
        ],
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

    /// Return the specialized pipeline for `(kernel, bucket_m, W)`.
    /// First call builds; subsequent calls hit the cache. Every
    /// function constant the kernel consumes is read from `W::*`
    /// (the macro-emitted `CanonicalParams` impl) so there is no
    /// runtime extras struct to thread.
    pub fn pipeline_for<W: CanonicalParams>(
        &self,
        kernel: KernelId,
        bucket_m: u32,
    ) -> Result<ComputePipelineState, PipelineLookupError> {
        let (library, function) = kernel_msl_names(kernel, bucket_m)?;
        let constants = constants_for::<W>(kernel, bucket_m)?;
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
        // `RMS_NORM_EPS` defaults to 1e-5 on `CanonicalParams`.
        let bag = constants_for::<TinyLlamaProbe>(KernelId::RmsNorm, 8).expect("rmsnorm bag");
        assert_eq!(bag.len(), 3);
        assert_eq!(bag[0], ConstantValue::uint(0, 8));
        assert_eq!(bag[1], ConstantValue::uint(1, 2048));
        assert_eq!(bag[2], ConstantValue::float(2, 1e-5));
    }

    #[test]
    fn attention_via_cache_pulls_consts_from_canonical_params() {
        let bag =
            constants_for::<TinyLlamaProbe>(KernelId::AttentionViaCache, 1).expect("attn bag");
        assert_eq!(bag.len(), 6);
        assert_eq!(bag[0], ConstantValue::uint(0, 64));
        assert_eq!(bag[3], ConstantValue::float(3, 0.125));
        assert_eq!(bag[4], ConstantValue::uint(4, 16)); // BLOCK_SIZE default
        assert_eq!(bag[5], ConstantValue::uint(5, 128)); // MAX_BLOCKS_PER_SEQ default
    }

    #[test]
    fn attention_prefill_pulls_tile_q_from_canonical_params() {
        let bag = constants_for::<TinyLlamaProbe>(KernelId::AttentionPrefillContiguous, 16)
            .expect("prefill bag");
        assert_eq!(bag.len(), 5);
        assert_eq!(bag[0], ConstantValue::uint(0, 64));
        assert_eq!(bag[4], ConstantValue::uint(6, 16)); // PREFILL_TILE_Q default
    }

    #[test]
    fn fused_silu_bag_has_no_eps() {
        let bag =
            constants_for::<TinyLlamaProbe>(KernelId::FusedGateUpSiluMul, 64).expect("silu bag");
        assert_eq!(bag.len(), 3);
        assert_eq!(bag[0], ConstantValue::uint(0, 64));
        assert_eq!(bag[1], ConstantValue::uint(1, 5632));
        assert_eq!(bag[2], ConstantValue::uint(2, 2048));
    }

    #[test]
    fn fused_silu_decode_bag_uses_indices_3_4_5() {
        // bucket_m == 1 selects the decode kernel; constants live at
        // indices 3, 4, 5 to avoid clashing with the matrix variant
        // in the same library.
        let bag =
            constants_for::<TinyLlamaProbe>(KernelId::FusedGateUpSiluMul, 1).expect("silu decode bag");
        assert_eq!(bag.len(), 3);
        assert_eq!(bag[0], ConstantValue::uint(3, 1));
        assert_eq!(bag[1], ConstantValue::uint(4, 5632));
        assert_eq!(bag[2], ConstantValue::uint(5, 2048));
    }

    #[test]
    fn gemm_is_rejected_as_opaque() {
        let err = constants_for::<TinyLlamaProbe>(KernelId::Gemm, 1).unwrap_err();
        assert!(matches!(err, PipelineLookupError::OpaqueKernel(_)));
    }

    #[test]
    fn reshape_is_rejected_as_metadata_only() {
        let err = constants_for::<TinyLlamaProbe>(KernelId::Reshape, 1).unwrap_err();
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
        // `RMS_NORM_EPS` baked from `TinyLlamaProbe::RMS_NORM_EPS`
        // (defaults to 1e-5).
        let _p1 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::RmsNorm, 1)
            .expect("rmsnorm bucket=1");
        let _p8 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::RmsNorm, 8)
            .expect("rmsnorm bucket=8");
        let _p1_again = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::RmsNorm, 1)
            .expect("rmsnorm bucket=1 cache hit");
        // Two pipelines (one per bucket); third call hits the cache.
        assert_eq!(pipelines.cached_count(), 2);

        // FusedAddRmsNorm at the same buckets. Adds two more entries.
        let _f1 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::FusedAddRmsNorm, 1)
            .expect("fused_add_rmsnorm bucket=1");
        let _f8 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::FusedAddRmsNorm, 8)
            .expect("fused_add_rmsnorm bucket=8");
        assert_eq!(pipelines.cached_count(), 4);

        // FusedGateUpSiluMul (Phase 5.B.4 scope — silu only; gemm is
        // opaque/MPS). Two buckets → six pipelines total.
        let _g1 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::FusedGateUpSiluMul, 1)
            .expect("silu bucket=1");
        let _g8 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::FusedGateUpSiluMul, 8)
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

        // AttentionViaCache: bucket=1 (decode). block_size=16,
        // max_blocks_per_seq=128 baked from
        // `TinyLlamaProbe::BLOCK_SIZE` / `MAX_BLOCKS_PER_SEQ`
        // (defaults).
        let _d1 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::AttentionViaCache, 1)
            .expect("attention_via_cache bucket=1");
        let _d1_again = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::AttentionViaCache, 1)
            .expect("attention_via_cache bucket=1 cache hit");
        assert_eq!(pipelines.cached_count(), 1);

        // AttentionPrefillContiguous at bucket=16 (one full
        // PREFILL_TILE_Q tile) and bucket=64.
        let _p16 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::AttentionPrefillContiguous, 16)
            .expect("attention_prefill bucket=16");
        let _p64 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::AttentionPrefillContiguous, 64)
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

        // BLOCK_SIZE baked from `TinyLlamaProbe::BLOCK_SIZE` (default 16).
        let _r1 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::RopeAppend, 1)
            .expect("rope_append bucket=1");
        let _r8 = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::RopeAppend, 8)
            .expect("rope_append bucket=8");
        let _r1_again = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::RopeAppend, 1)
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
        let block_size: usize = TinyLlamaProbe::BLOCK_SIZE as usize;
        let num_blocks: usize = 4;
        let max_pos: usize = 32;

        let pipeline = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::RopeAppend, bucket_m as u32)
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
                let theta = (pos as f32) / 10000.0_f32.powf((d as f32) / (half_dim as f32));
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
            let half_data: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(half_data.as_slice());
            let buf = device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
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
            let buf = device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
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

    /// Phase 5.G.4a numerical-correctness check for
    /// `attention_via_cache_f16_specialized` against
    /// `cpu_golden::attention_via_cache`. Synthetic 2-sequence decode
    /// (batch = bucket_m = 2) with mixed cache lengths spanning one
    /// and two logical blocks; deterministic Q + paged K/V buffers
    /// dispatched directly via the specialized pipeline (no
    /// ICB/worker), output read back as f16, max-abs error asserted
    /// < 5e-3 vs the cpu_golden ref. The cpu_golden uses the same
    /// `inv_sum = 1 / (sum_exp + 1e-6)` guard the shader uses
    /// (5.G.2's matching tweak), so any drift is f16 round-tripping.
    #[cfg(target_os = "macos")]
    #[test]
    fn attention_via_cache_matches_cpu_golden() {
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

        // TinyLlamaProbe params: HEAD_DIM=64, NUM_Q_HEADS=32,
        // NUM_KV_HEADS=4, ATTN_SCALE=0.125, BLOCK_SIZE=16,
        // MAX_BLOCKS_PER_SEQ=128 (defaults from `CanonicalParams`).
        // Decode: bucket_m == batch.
        let batch: usize = 2;
        let head_dim = TinyLlamaProbe::HEAD_DIM as usize;
        let num_q = TinyLlamaProbe::NUM_Q_HEADS as usize;
        let num_kv = TinyLlamaProbe::NUM_KV_HEADS as usize;
        let block_size: usize = TinyLlamaProbe::BLOCK_SIZE as usize;
        let max_blocks_per_seq: usize = TinyLlamaProbe::MAX_BLOCKS_PER_SEQ as usize;
        let num_blocks: usize = 4;

        let pipeline = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::AttentionViaCache, batch as u32)
            .expect("attention_via_cache pipeline");

        // Per-seq cache lengths: seq 0 fits in one logical block, seq
        // 1 spans two. Block table maps logical → physical with seq 0
        // owning physical block 0, seq 1 owning physical blocks 2,3.
        let seq_used_k: Vec<u32> = vec![10, 17];
        let mut block_table = vec![0u32; batch * max_blocks_per_seq];
        block_table[0 * max_blocks_per_seq + 0] = 0;
        block_table[1 * max_blocks_per_seq + 0] = 2;
        block_table[1 * max_blocks_per_seq + 1] = 3;

        // Synthetic deterministic Q + K/V cache. Q and K must be
        // small enough that the unscaled dot products fit in f16
        // without overflow when multiplied across head_dim=64; we
        // use ~0.05-magnitude values so |Q·K| ≲ 64*0.05*0.05 = 0.16.
        let q_data: Vec<f32> = (0..batch * num_q * head_dim)
            .map(|i| ((i as f32) * 0.013).sin() * 0.5)
            .collect();
        let kv_cache_size = num_blocks * num_kv * block_size * head_dim;
        let mut kv_k_data = vec![0.0_f32; kv_cache_size];
        let mut kv_v_data = vec![0.0_f32; kv_cache_size];
        // Only fill the live slots — physical blocks 0, 2, 3 — so the
        // shader sees deterministic data wherever the block_table
        // points, and zeros wherever it doesn't (defends against an
        // accidental over-read).
        for (live_block, kv_len) in [(0usize, 10usize), (2usize, 16usize), (3usize, 1usize)] {
            for tok in 0..kv_len {
                for kvh in 0..num_kv {
                    for d in 0..head_dim {
                        let idx = live_block * num_kv * block_size * head_dim
                            + kvh * block_size * head_dim
                            + tok * head_dim
                            + d;
                        let seed = (idx as f32) * 0.0017;
                        kv_k_data[idx] = (seed.sin()) * 0.5;
                        kv_v_data[idx] = (seed.cos()) * 0.5;
                    }
                }
            }
        }

        // Buffer helpers (mirror rope_append_matches_cpu_golden).
        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_f16(device: &Device, data: &[f32]) -> Buffer {
            let half_data: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(half_data.as_slice());
            let buf = device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
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
            let buf = device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
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
        let seq_used_buf = alloc_u32(&device, &seq_used_k);
        let block_table_buf = alloc_u32(&device, &block_table);
        let kv_k_buf = alloc_f16(&device, &kv_k_data);
        let kv_v_buf = alloc_f16(&device, &kv_v_data);
        let output_buf = alloc_zero_f16(&device, batch * num_q * head_dim);

        // Encode + dispatch.
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&output_buf), 0);
        enc.set_buffer(1, Some(&q_buf), 0);
        enc.set_buffer(2, Some(&seq_used_buf), 0);
        enc.set_buffer(3, Some(&block_table_buf), 0);
        enc.set_buffer(4, Some(&kv_k_buf), 0);
        enc.set_buffer(5, Some(&kv_v_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(batch as u64, num_q as u64, 1),
            MTLSize::new(head_dim as u64, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // Round-trip Q + cache through f16 to match the shader's
        // input precision before invoking cpu_golden.
        let q_f16: Vec<f32> = q_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let kv_k_f16: Vec<f32> = kv_k_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let kv_v_f16: Vec<f32> = kv_v_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();

        let mut output_cpu = vec![0.0_f32; batch * num_q * head_dim];
        cpu_golden::attention_via_cache(
            &q_f16,
            &kv_k_f16,
            &kv_v_f16,
            &block_table,
            &seq_used_k,
            &mut output_cpu,
            num_q,
            num_kv,
            head_dim,
            block_size,
            max_blocks_per_seq,
            TinyLlamaProbe::ATTN_SCALE,
        );

        // Read back f16 output and compare.
        fn read_f16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const half::f16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let output_metal = read_f16(&output_buf, output_cpu.len());

        let tol: f32 = 5e-3;
        for i in 0..output_cpu.len() {
            let diff = (output_metal[i] - output_cpu[i]).abs();
            assert!(
                diff < tol,
                "out[{i}] metal={} cpu={} diff={}",
                output_metal[i],
                output_cpu[i],
                diff
            );
        }
    }

    /// Phase 5.G.4b numerical-correctness check for
    /// `attention_prefill_contiguous_f16_specialized` against
    /// `cpu_golden::attention_prefill`. Synthetic 2-sequence prefill
    /// with `seqlens = [8, 8]` (total = 16 = one full
    /// `PREFILL_TILE_Q`); deterministic contiguous Q/K/V dispatched
    /// directly via the specialized pipeline. Tolerance is 5e-3:
    /// shader uses `inv_sum = 1/(sum_exp + 1e-6)` while cpu_golden
    /// divides by `sum_exp` directly — drift is ~1e-6/sum_exp,
    /// dominated by f16 round-tripping at 5e-3.
    #[cfg(target_os = "macos")]
    #[test]
    fn attention_prefill_contiguous_matches_cpu_golden() {
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

        let head_dim = TinyLlamaProbe::HEAD_DIM as usize;
        let num_q = TinyLlamaProbe::NUM_Q_HEADS as usize;
        let num_kv = TinyLlamaProbe::NUM_KV_HEADS as usize;
        let tile_q: usize = TinyLlamaProbe::PREFILL_TILE_Q as usize;

        // 2 sequences, each 8 tokens — total = 16 = exactly one
        // PREFILL_TILE_Q tile, so the dispatch grid has a single
        // x-axis threadgroup.
        let cu_seqlens_q: Vec<u32> = vec![0, 8, 16];
        let total: usize = *cu_seqlens_q.last().unwrap() as usize;

        let bucket_m = total as u32;
        let pipeline = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::AttentionPrefillContiguous, bucket_m)
            .expect("attention_prefill pipeline");

        // Synthetic deterministic Q/K/V — small magnitudes so f16
        // round-tripping doesn't dominate (0.5 * sin(...) ≈ ±0.5).
        let q_data: Vec<f32> = (0..total * num_q * head_dim)
            .map(|i| ((i as f32) * 0.011).sin() * 0.5)
            .collect();
        let k_data: Vec<f32> = (0..total * num_kv * head_dim)
            .map(|i| ((i as f32) * 0.017).cos() * 0.5)
            .collect();
        let v_data: Vec<f32> = (0..total * num_kv * head_dim)
            .map(|i| ((i as f32) * 0.023).sin() * 0.5)
            .collect();

        // Buffer helpers.
        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_f16(device: &Device, data: &[f32]) -> Buffer {
            let half_data: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(half_data.as_slice());
            let buf = device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
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
            let buf = device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
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
        let cu_buf = alloc_u32(&device, &cu_seqlens_q);
        let output_buf = alloc_zero_f16(&device, total * num_q * head_dim);

        let num_q_tiles = (total + tile_q - 1) / tile_q;
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&output_buf), 0);
        enc.set_buffer(1, Some(&q_buf), 0);
        enc.set_buffer(2, Some(&k_buf), 0);
        enc.set_buffer(3, Some(&v_buf), 0);
        enc.set_buffer(4, Some(&cu_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(num_q_tiles as u64, num_q as u64, 1),
            MTLSize::new(head_dim as u64, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // Round-trip Q/K/V through f16 to match the shader's input
        // precision before invoking cpu_golden.
        let q_f16: Vec<f32> = q_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let k_f16: Vec<f32> = k_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let v_f16: Vec<f32> = v_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();

        let seq_starts: Vec<usize> = cu_seqlens_q.iter().map(|&v| v as usize).collect();
        let mut output_cpu = vec![0.0_f32; total * num_q * head_dim];
        cpu_golden::attention_prefill(
            &q_f16,
            &k_f16,
            &v_f16,
            &mut output_cpu,
            &seq_starts,
            num_q,
            num_kv,
            head_dim,
            TinyLlamaProbe::ATTN_SCALE,
        );

        fn read_f16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const half::f16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let output_metal = read_f16(&output_buf, output_cpu.len());

        let tol: f32 = 5e-3;
        for i in 0..output_cpu.len() {
            let diff = (output_metal[i] - output_cpu[i]).abs();
            assert!(
                diff < tol,
                "out[{i}] metal={} cpu={} diff={}",
                output_metal[i],
                output_cpu[i],
                diff
            );
        }
    }

    /// Small `CanonicalParams` probe with `Q_SIZE` (= K, hidden) and
    /// `INTERMEDIATE_SIZE` (= N, MLP intermediate) sized to multiples
    /// of the fused-MLP shader's 8-element tile. Used by
    /// [`fused_mlp_matches_cpu_golden`].
    struct MlpProbe;
    impl CanonicalParams for MlpProbe {
        const HEAD_DIM: u32 = 64;
        const NUM_Q_HEADS: u32 = 1;
        const NUM_KV_HEADS: u32 = 1;
        const Q_SIZE: usize = 16;
        const KV_SIZE: usize = 16;
        const INTERMEDIATE_SIZE: usize = 16;
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

    /// Numerical-correctness check for `rmsnorm_f16_specialized`
    /// against `cpu_golden::rmsnorm`. Hardens the binding contract
    /// (out=0, in=1, weight=2) the in/out swap fix in 3bb5b9c89 put
    /// in place, and catches any future arithmetic regression in the
    /// per-row sum-of-squares reduction.
    #[cfg(target_os = "macos")]
    #[test]
    fn rmsnorm_matches_cpu_golden() {
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

        let m: usize = 4;
        let hidden = TinyLlamaProbe::Q_SIZE;
        let pipeline = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::RmsNorm, m as u32)
            .expect("rmsnorm pipeline");

        let input_data: Vec<f32> = (0..m * hidden)
            .map(|i| ((i as f32) * 0.011).sin() * 0.5)
            .collect();
        let weight_data: Vec<f32> = (0..hidden)
            .map(|i| 1.0 + ((i as f32) * 0.017).cos() * 0.05)
            .collect();

        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_f16(device: &Device, data: &[f32]) -> Buffer {
            let half_data: Vec<half::f16> =
                data.iter().map(|&v| half::f16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(half_data.as_slice());
            let buf =
                device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    half_data.as_ptr() as *const u8,
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

        let input_buf = alloc_f16(&device, &input_data);
        let weight_buf = alloc_f16(&device, &weight_data);
        let output_buf = alloc_zero_f16(&device, m * hidden);

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&output_buf), 0);
        enc.set_buffer(1, Some(&input_buf), 0);
        enc.set_buffer(2, Some(&weight_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(m as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // Per-row CPU reference, f16-round-tripped to match the kernel.
        let input_f16: Vec<f32> = input_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let weight_f16: Vec<f32> = weight_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let mut output_cpu = vec![0.0_f32; m * hidden];
        let eps = 1e-5_f32; // CanonicalParams::RMS_NORM_EPS default.
        for row in 0..m {
            let base = row * hidden;
            cpu_golden::rmsnorm(
                &input_f16[base..base + hidden],
                &weight_f16,
                &mut output_cpu[base..base + hidden],
                eps,
            );
        }

        fn read_f16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const half::f16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let output_metal = read_f16(&output_buf, output_cpu.len());

        let tol: f32 = 5e-3;
        for i in 0..output_cpu.len() {
            let diff = (output_metal[i] - output_cpu[i]).abs();
            assert!(
                diff < tol,
                "rmsnorm[{i}] metal={} cpu={} diff={}",
                output_metal[i],
                output_cpu[i],
                diff
            );
        }
    }

    /// Numerical-correctness check for `fused_add_rmsnorm_f16_specialized`
    /// against `cpu_golden::fused_add_rmsnorm`. Verifies the in-place
    /// `residual += delta` step lands in buffer(0) and the
    /// `rmsnorm(residual_after_add, weight)` lands in buffer(1).
    #[cfg(target_os = "macos")]
    #[test]
    fn fused_add_rmsnorm_matches_cpu_golden() {
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

        let m: usize = 4;
        let hidden = TinyLlamaProbe::Q_SIZE;
        let pipeline = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::FusedAddRmsNorm, m as u32)
            .expect("fused_add_rmsnorm pipeline");

        let residual_data: Vec<f32> = (0..m * hidden)
            .map(|i| ((i as f32) * 0.013).sin() * 0.5)
            .collect();
        let delta_data: Vec<f32> = (0..m * hidden)
            .map(|i| ((i as f32) * 0.017).cos() * 0.5)
            .collect();
        let weight_data: Vec<f32> = (0..hidden)
            .map(|i| 1.0 + ((i as f32) * 0.019).sin() * 0.05)
            .collect();

        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_f16(device: &Device, data: &[f32]) -> Buffer {
            let half_data: Vec<half::f16> =
                data.iter().map(|&v| half::f16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(half_data.as_slice());
            let buf =
                device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    half_data.as_ptr() as *const u8,
                    buf.contents() as *mut u8,
                    bytes,
                );
            }
            buf
        }

        let residual_buf = alloc_f16(&device, &residual_data);
        let delta_buf = alloc_f16(&device, &delta_data);
        let weight_buf = alloc_f16(&device, &weight_data);

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&residual_buf), 0);
        enc.set_buffer(1, Some(&delta_buf), 0);
        enc.set_buffer(2, Some(&weight_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(m as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // CPU reference: f16-round-trip inputs, then run cpu_golden.
        let mut residual_cpu: Vec<f32> = residual_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let mut delta_cpu: Vec<f32> = delta_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let weight_f16: Vec<f32> = weight_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let eps = 1e-5_f32;
        cpu_golden::fused_add_rmsnorm(
            &mut residual_cpu,
            &mut delta_cpu,
            &weight_f16,
            eps,
            hidden,
        );

        fn read_f16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const half::f16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let residual_metal = read_f16(&residual_buf, residual_cpu.len());
        let delta_metal = read_f16(&delta_buf, delta_cpu.len());

        let tol: f32 = 5e-3;
        for i in 0..residual_cpu.len() {
            let diff = (residual_metal[i] - residual_cpu[i]).abs();
            assert!(
                diff < tol,
                "residual[{i}] metal={} cpu={} diff={}",
                residual_metal[i],
                residual_cpu[i],
                diff
            );
        }
        for i in 0..delta_cpu.len() {
            let diff = (delta_metal[i] - delta_cpu[i]).abs();
            assert!(
                diff < tol,
                "delta[{i}] metal={} cpu={} diff={}",
                delta_metal[i],
                delta_cpu[i],
                diff
            );
        }
    }

    /// Same as [`fused_mlp_matches_cpu_golden`] but at TinyLlama's
    /// actual decode shape (M=1, N=5632, K=2048). The smaller golden
    /// only exercises 2 K-tiles and 2 N-tiles; this case exercises 256
    /// K-tiles and 704 N-tiles, including the last tile boundary
    /// `n_base = N - 8` for both gate and up halves of the weight.
    /// Catches any K-loop / boundary bug that the small golden misses.
    ///
    /// Runs serially on CPU so the reference is slow (~22M MAC / variant);
    /// kept lean by computing a single output row.
    #[cfg(target_os = "macos")]
    #[test]
    fn fused_mlp_tinyllama_shape_matches_cpu_golden() {
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

        let m: usize = 1;
        let n: usize = TinyLlamaProbe::INTERMEDIATE_SIZE; // 5632
        let k: usize = TinyLlamaProbe::Q_SIZE;            // 2048

        let pipeline = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::FusedGateUpSiluMul, m as u32)
            .expect("fused_mlp pipeline");

        // Magnitudes ~0.05 so K=2048 sums stay in [-50, 50] (within
        // f16 max 65504 with margin).
        let input_data: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32) * 0.013).sin() * 0.05)
            .collect();
        let weight_data: Vec<f32> = (0..(2 * n) * k)
            .map(|i| ((i as f32) * 0.019).cos() * 0.05)
            .collect();

        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_f16(device: &Device, data: &[f32]) -> Buffer {
            let half_data: Vec<half::f16> =
                data.iter().map(|&v| half::f16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(half_data.as_slice());
            let buf =
                device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    half_data.as_ptr() as *const u8,
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

        let input_buf = alloc_f16(&device, &input_data);
        let weight_buf = alloc_f16(&device, &weight_data);
        let output_buf = alloc_zero_f16(&device, m * n);

        // M=1 → decode kernel dispatch shape.
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&output_buf), 0);
        enc.set_buffer(1, Some(&input_buf), 0);
        enc.set_buffer(2, Some(&weight_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(8), 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // CPU reference: round-trip inputs/weights through f16 first.
        let input_f16: Vec<f32> = input_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let weight_f16: Vec<f32> = weight_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let gate_w = &weight_f16[0..n * k];
        let up_w = &weight_f16[n * k..2 * n * k];
        let mut gate = vec![0.0_f32; m * n];
        let mut up = vec![0.0_f32; m * n];
        cpu_golden::gemm(&input_f16, gate_w, &mut gate, m, k, n);
        cpu_golden::gemm(&input_f16, up_w, &mut up, m, k, n);
        let mut output_cpu = vec![0.0_f32; m * n];
        cpu_golden::fused_gate_up_silu_mul(&gate, &up, &mut output_cpu);

        fn read_f16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const half::f16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let output_metal = read_f16(&output_buf, output_cpu.len());

        // K=2048 inner-product accumulates ~2048 f16-rounded products.
        // f32 accumulator drift dominates; allow 5e-2.
        let tol: f32 = 5e-2;
        let mut max_diff: f32 = 0.0;
        for i in 0..output_cpu.len() {
            let diff = (output_metal[i] - output_cpu[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < tol,
                "out[{i}] metal={} cpu={} diff={} max_so_far={}",
                output_metal[i],
                output_cpu[i],
                diff,
                max_diff
            );
        }
        eprintln!("fused_mlp_tinyllama_shape: max_abs_diff={max_diff}");
    }

    /// Numerical-correctness check for
    /// `fused_gate_up_silu_mul_gemm_f16_specialized`. Computes
    /// `output = silu(input @ W_gate^T) * (input @ W_up^T)` with the
    /// new fused kernel and compares against a CPU reference built
    /// from `cpu_golden::gemm` + `cpu_golden::fused_gate_up_silu_mul`.
    ///
    /// Inputs/weights use small magnitudes (~0.3) so the K-inner-
    /// product magnitudes stay well within f16 range — drift is
    /// dominated by f16 round-tripping at 5e-3.
    ///
    /// Two cases:
    /// - `m=16`: full 8×8 tiles on every axis (M=16, N=K=16).
    ///   Multiple K-tiles exercises the inner accumulation loop;
    ///   multiple M/N tiles exercises the threadgroup grid.
    /// - `m=1`:  decode-shaped, partial M tile. Forces the
    ///   threadgroup-scratch zero-pad path for the A fragment.
    #[cfg(target_os = "macos")]
    #[test]
    fn fused_mlp_matches_cpu_golden() {
        for m in [16usize, 1usize] {
            run_fused_mlp_check(m);
        }
    }

    #[cfg(target_os = "macos")]
    fn run_fused_mlp_check(m: usize) {
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

        let n: usize = MlpProbe::INTERMEDIATE_SIZE;
        let k: usize = MlpProbe::Q_SIZE;
        assert_eq!(n % 8, 0);
        assert_eq!(k % 8, 0);

        let pipeline = pipelines
            .pipeline_for::<MlpProbe>(KernelId::FusedGateUpSiluMul, m as u32)
            .expect("fused_mlp pipeline");

        // Synthetic deterministic input [M, K] and packed weight
        // [2*N, K]. Magnitudes ~0.3 so K=16 sums stay in [-5, 5].
        let input_data: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32) * 0.013).sin() * 0.3)
            .collect();
        let weight_data: Vec<f32> = (0..(2 * n) * k)
            .map(|i| ((i as f32) * 0.019).cos() * 0.3)
            .collect();

        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_f16(device: &Device, data: &[f32]) -> Buffer {
            let half_data: Vec<half::f16> =
                data.iter().map(|&v| half::f16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(half_data.as_slice());
            let buf =
                device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    half_data.as_ptr() as *const u8,
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

        let input_buf = alloc_f16(&device, &input_data);
        let weight_buf = alloc_f16(&device, &weight_data);
        let output_buf = alloc_zero_f16(&device, m * n);

        // Dispatch shape depends on which kernel variant was picked.
        // M=1 → decode kernel: 256 threads/group, 8 outputs/group.
        // M>=2 → matrix kernel: 32 threads/group, 8x8 output tile.
        let (threadgroups, threads_per_threadgroup) = if m == 1 {
            (
                MTLSize::new((n as u64).div_ceil(8), 1, 1),
                MTLSize::new(256, 1, 1),
            )
        } else {
            (
                MTLSize::new((n as u64).div_ceil(8), (m as u64).div_ceil(8), 1),
                MTLSize::new(32, 1, 1),
            )
        };
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&output_buf), 0);
        enc.set_buffer(1, Some(&input_buf), 0);
        enc.set_buffer(2, Some(&weight_buf), 0);
        enc.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // CPU reference: round-trip inputs/weights through f16 to
        // match what the kernel actually sees.
        let input_f16: Vec<f32> = input_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let weight_f16: Vec<f32> = weight_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();

        let gate_w = &weight_f16[0..n * k];
        let up_w = &weight_f16[n * k..2 * n * k];
        let mut gate = vec![0.0_f32; m * n];
        let mut up = vec![0.0_f32; m * n];
        cpu_golden::gemm(&input_f16, gate_w, &mut gate, m, k, n);
        cpu_golden::gemm(&input_f16, up_w, &mut up, m, k, n);
        let mut output_cpu = vec![0.0_f32; m * n];
        cpu_golden::fused_gate_up_silu_mul(&gate, &up, &mut output_cpu);

        fn read_f16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const half::f16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let output_metal = read_f16(&output_buf, output_cpu.len());

        let tol: f32 = 5e-3;
        for i in 0..output_cpu.len() {
            let diff = (output_metal[i] - output_cpu[i]).abs();
            assert!(
                diff < tol,
                "m={m}: out[{i}] metal={} cpu={} diff={}",
                output_metal[i],
                output_cpu[i],
                diff
            );
        }
    }
}
