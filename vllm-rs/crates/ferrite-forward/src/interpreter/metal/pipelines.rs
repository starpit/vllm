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

use super::lowered::{KernelId, MetalDtype};

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
/// `dtype` selects between the `_f16_specialized` / `_bf16_specialized`
/// symbol variants in each shader. Llama-3.x ships bf16 on disk and
/// cuda runs them natively in bf16; the metal backend follows suit
/// on M3+ (native bf16 MMA). `MetalDtype::Int4` is reserved for the
/// AWQ / GPTQ dequant path and is rejected here until that wiring
/// lands — int4 weights need a different binding shape (packed u32s
/// + group scales) so a single symbol picker can't transparently
/// model it.
fn kernel_msl_names(
    kernel: KernelId,
    bucket_m: u32,
    dtype: MetalDtype,
) -> Result<(&'static str, &'static str), PipelineLookupError> {
    if matches!(dtype, MetalDtype::Int4) {
        return Err(PipelineLookupError::DtypeNotYetWired(kernel, dtype));
    }
    Ok(match (kernel, dtype) {
        (KernelId::Embed, MetalDtype::F16) => ("embed", "embed_f16_specialized"),
        (KernelId::Embed, MetalDtype::Bf16) => ("embed", "embed_bf16_specialized"),
        (KernelId::RmsNorm, MetalDtype::F16) => ("rmsnorm", "rmsnorm_f16_specialized"),
        (KernelId::RmsNorm, MetalDtype::Bf16) => ("rmsnorm", "rmsnorm_bf16_specialized"),
        (KernelId::FusedAddRmsNorm, MetalDtype::F16) => {
            ("fused_add_rmsnorm", "fused_add_rmsnorm_f16_specialized")
        }
        (KernelId::FusedAddRmsNorm, MetalDtype::Bf16) => {
            ("fused_add_rmsnorm", "fused_add_rmsnorm_bf16_specialized")
        }
        // Two specialized variants per dtype: the decode (M=1) variant
        // uses simd_sum dot products and avoids simdgroup_matrix
        // overhead at one-row inputs. Prefill (M >= 2) uses the matrix
        // variant.
        (KernelId::FusedGateUpSiluMul, MetalDtype::F16) if bucket_m == 1 => (
            "fused_gate_up_silu_mul",
            "fused_gate_up_silu_mul_decode_f16_specialized",
        ),
        (KernelId::FusedGateUpSiluMul, MetalDtype::F16) => (
            "fused_gate_up_silu_mul",
            "fused_gate_up_silu_mul_gemm_f16_specialized",
        ),
        (KernelId::FusedGateUpSiluMul, MetalDtype::Bf16) if bucket_m == 1 => (
            "fused_gate_up_silu_mul",
            "fused_gate_up_silu_mul_decode_bf16_specialized",
        ),
        (KernelId::FusedGateUpSiluMul, MetalDtype::Bf16) => (
            "fused_gate_up_silu_mul",
            "fused_gate_up_silu_mul_gemm_bf16_specialized",
        ),
        (KernelId::RopeAppend, MetalDtype::F16) => ("rope", "rope_append_f16_specialized"),
        (KernelId::RopeAppend, MetalDtype::Bf16) => ("rope", "rope_append_bf16_specialized"),
        // v2 (online-softmax port of MLX `sdpa_vector`) is now the
        // production picker for both dtypes. v1 (2-pass softmax with
        // shared_logits[]) accumulated bf16 rounding error per layer
        // on Llama-3.2 decode; v2 fixes that. Both v2 variants assume
        // 1024 threads/group dispatch — see lowering.rs.
        (KernelId::AttentionViaCache, MetalDtype::F16) => {
            ("attention", "attention_via_cache_v2_f16_specialized")
        }
        (KernelId::AttentionViaCache, MetalDtype::Bf16) => {
            ("attention", "attention_via_cache_v2_bf16_specialized")
        }
        (KernelId::AttentionPrefillContiguous, MetalDtype::F16) => {
            ("attention", "attention_prefill_contiguous_f16_specialized")
        }
        (KernelId::AttentionPrefillContiguous, MetalDtype::Bf16) => {
            ("attention", "attention_prefill_contiguous_bf16_specialized")
        }
        (KernelId::Add, MetalDtype::F16) => ("elementwise", "residual_add_f16_specialized"),
        (KernelId::Add, MetalDtype::Bf16) => ("elementwise", "residual_add_bf16_specialized"),
        (KernelId::ScalarMul, MetalDtype::F16) => ("elementwise", "scalar_mul_f16_specialized"),
        (KernelId::ScalarMul, MetalDtype::Bf16) => ("elementwise", "scalar_mul_bf16_specialized"),
        // GEMM: f16 routes through MPS' `MPSMatrixMultiplication`
        // (opaque to this cache). bf16 has no MPS path —
        // `MPSMatrixMultiplication` rejects `MPSDataTypeBFloat16` at
        // runtime — so we ship a custom `gemm_bf16_specialized`
        // kernel using `simdgroup_bfloat8x8` MMA tiles.
        (KernelId::Gemm, MetalDtype::F16) => {
            return Err(PipelineLookupError::OpaqueKernel(KernelId::Gemm));
        }
        (KernelId::Gemm, MetalDtype::Bf16) => ("gemm", "gemm_bf16_specialized"),
        // Reshape is metadata-only.
        (KernelId::Reshape, _) => {
            return Err(PipelineLookupError::MetadataOnly(KernelId::Reshape));
        }
        // Int4 was filtered out above.
        (_, MetalDtype::Int4) => unreachable!("Int4 filtered at fn entry"),
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
        // GEMM: only the bf16 (custom) path uses function constants.
        // The f16 path goes through MPS and never queries this fn.
        // The shape constants (M, N, K) come from the lowering pass'
        // `gemm_dims`; we don't have those here, so the GEMM-side
        // pipeline build is done out-of-band by the worker (which has
        // `gemm_dims` in hand). This arm is unreachable for `Gemm`
        // when the worker uses the dedicated builder; defensive error
        // for any future caller that forgets.
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
    /// Defaults the dtype to [`MetalDtype::F16`] for backward
    /// compatibility while the bf16 wiring lands across the worker /
    /// macro-emission seams. New call sites should prefer
    /// [`Self::pipeline_for_dtype`].
    pub fn pipeline_for<W: CanonicalParams>(
        &self,
        kernel: KernelId,
        bucket_m: u32,
    ) -> Result<ComputePipelineState, PipelineLookupError> {
        self.pipeline_for_dtype::<W>(kernel, bucket_m, MetalDtype::F16)
    }

    /// Return the specialized pipeline for `(kernel, bucket_m, W,
    /// dtype)`. First call builds; subsequent calls hit the cache.
    /// Function constants read from `W::*` (the macro-emitted
    /// `CanonicalParams` impl); dtype picks the symbol variant
    /// (`_f16_specialized` vs `_bf16_specialized`).
    pub fn pipeline_for_dtype<W: CanonicalParams>(
        &self,
        kernel: KernelId,
        bucket_m: u32,
        dtype: MetalDtype,
    ) -> Result<ComputePipelineState, PipelineLookupError> {
        let (library, function) = kernel_msl_names(kernel, bucket_m, dtype)?;
        let constants = constants_for::<W>(kernel, bucket_m)?;
        let key = PipelineKey::new(library, function, constants);
        self.cache
            .get_or_build(&key)
            .map_err(PipelineLookupError::Build)
    }

    /// Bf16 GEMM pipeline keyed on the dynamic `(M, N, K)` triple.
    /// MPS' `MPSMatrixMultiplication` doesn't accept
    /// `MPSDataTypeBFloat16`, so the bf16 path uses the custom
    /// `gemm_bf16_specialized` kernel (uses `simdgroup_bfloat8x8`
    /// MMA tiles, native on M3+). Shape goes through function
    /// constants 0 / 1 / 2 = M / N / K.
    ///
    /// Caller supplies the dims directly (the lowering pass has them
    /// on `LoweredCommand.gemm_dims`); they don't sit on
    /// `CanonicalParams` since each GEMM step has its own shape.
    pub fn pipeline_for_gemm_bf16(
        &self,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<ComputePipelineState, PipelineLookupError> {
        let constants = vec![
            ConstantValue::uint(0, m),
            ConstantValue::uint(1, n),
            ConstantValue::uint(2, k),
        ];
        let key = PipelineKey::new("gemm", "gemm_bf16_specialized", constants);
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
    /// Caller asked for a dtype the pipeline cache can't resolve yet.
    /// Currently used for `MetalDtype::Int4` — the AWQ / GPTQ dequant
    /// path needs different binding shapes (packed u32 + group scales)
    /// so a single symbol picker can't transparently model it.
    DtypeNotYetWired(KernelId, MetalDtype),
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
            Self::DtypeNotYetWired(k, d) => write!(
                f,
                "specialized pipeline lookup: kernel {k:?} dtype {d:?} not yet wired"
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

    /// Pure-CPU stub modelled on Llama-3.2-3B-Instruct. Exists only so
    /// the bf16 attention-prefill golden can exercise HEAD_DIM=128
    /// (the variant TinyLlamaProbe doesn't cover) before the runtime
    /// hits the same shape on a 3-GiB checkpoint.
    struct Llama32Probe;
    impl CanonicalParams for Llama32Probe {
        const HEAD_DIM: u32 = 128;
        const NUM_Q_HEADS: u32 = 24;
        const NUM_KV_HEADS: u32 = 8;
        const Q_SIZE: usize = 3072;
        const KV_SIZE: usize = 1024;
        const INTERMEDIATE_SIZE: usize = 8192;
        // 1 / sqrt(128) ≈ 0.08838834764831845
        const ATTN_SCALE: f32 = 0.088388347;
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

    /// Pure-CPU stub modelled on Llama-3.2-1B-Instruct. Same GQA
    /// ratio as `Llama32Probe` (8 KV heads) but at HEAD_DIM=64 and
    /// 32 Q heads — the exact decode shape the runtime hits on
    /// Llama-3.2-1B (the working set the existing bf16 attention
    /// golden never covers, since `Llama32Probe` is the 3B's shape).
    /// Lets us pin down whether the bf16 attention_via_cache kernel
    /// is correct at this specific (HEAD_DIM=64, GQA 4:1) shape.
    struct Llama32_1BProbe;
    impl CanonicalParams for Llama32_1BProbe {
        const HEAD_DIM: u32 = 64;
        const NUM_Q_HEADS: u32 = 32;
        const NUM_KV_HEADS: u32 = 8;
        const Q_SIZE: usize = 2048;
        const KV_SIZE: usize = 512;
        const INTERMEDIATE_SIZE: usize = 8192;
        // 1 / sqrt(64) = 0.125
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
        #[cfg(feature = "metal")]
        const METAL_DTYPE: crate::interpreter::metal::MetalDtype =
            crate::interpreter::metal::MetalDtype::Bf16;
    }

    /// Same as `Llama32Probe` but pinned to f16 so we can route the
    /// f16 prefill kernel at HEAD_DIM=128. Isolates bf16-vs-f16
    /// from HEAD_DIM=128-vs-HEAD_DIM=64 when debugging the
    /// stale-shared-logits collapse seen with bf16+HEAD_DIM=128.
    struct Llama32F16Probe;
    impl CanonicalParams for Llama32F16Probe {
        const HEAD_DIM: u32 = 128;
        const NUM_Q_HEADS: u32 = 24;
        const NUM_KV_HEADS: u32 = 8;
        const Q_SIZE: usize = 3072;
        const KV_SIZE: usize = 1024;
        const INTERMEDIATE_SIZE: usize = 8192;
        const ATTN_SCALE: f32 = 0.088388347;
        const ATTN_SOFTCAP: f32 = 0.0;
        const SLIDING_WINDOW: i32 = -1;
        const KV_LORA_RANK: usize = 0;
        const QK_NOPE_HEAD_DIM: usize = 0;
        const QK_ROPE_HEAD_DIM: usize = 0;
        const V_HEAD_DIM: usize = 0;
        const FINAL_LOGIT_SOFTCAPPING: f32 = 0.0;
        const QK_HEAD_DIM: usize = 0;
        const MLA_ATTN_SCALE: f32 = 0.0;
        #[cfg(feature = "metal")]
        const METAL_DTYPE: crate::interpreter::metal::MetalDtype =
            crate::interpreter::metal::MetalDtype::F16;
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
        let bag = constants_for::<TinyLlamaProbe>(KernelId::FusedGateUpSiluMul, 1)
            .expect("silu decode bag");
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

    /// HEAD_DIM=128 + bf16 rope_append golden — exercises the path
    /// Llama-3.2 hits at every layer. The existing rope_append golden
    /// only covers TinyLlama (HEAD_DIM=64, f16); decode KV-cache writes
    /// at Llama-3.2 shapes have no other test until this one.
    #[cfg(target_os = "macos")]
    #[test]
    fn rope_append_bf16_matches_cpu_golden_llama32() {
        use crate::cpu_golden;
        use ferrite_metal_kernels::metal::MTLSize;
        use half::bf16;

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

        let bucket_m: usize = 2;
        let head_dim = Llama32Probe::HEAD_DIM as usize;
        let half_dim = head_dim / 2;
        let num_q = Llama32Probe::NUM_Q_HEADS as usize;
        let num_kv = Llama32Probe::NUM_KV_HEADS as usize;
        let block_size: usize = Llama32Probe::BLOCK_SIZE as usize;
        let num_blocks: usize = 4;
        let max_pos: usize = 32;

        let pipeline = pipelines
            .pipeline_for_dtype::<Llama32Probe>(
                KernelId::RopeAppend,
                bucket_m as u32,
                MetalDtype::Bf16,
            )
            .expect("rope_append bf16 pipeline");

        let q_data: Vec<f32> = (0..bucket_m * num_q * head_dim)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect();
        let k_data: Vec<f32> = (0..bucket_m * num_kv * head_dim)
            .map(|i| ((i as f32) * 0.002).cos())
            .collect();
        let v_data: Vec<f32> = (0..bucket_m * num_kv * head_dim)
            .map(|i| (i as f32) * 0.003)
            .collect();

        // cos_sin layout: [max_pos, head_dim] = [cos[half] | sin[half]]
        // — matches what RotaryCache::new_from_gpuweights writes and
        // what the bf16 shader reads.
        let mut cos_sin_data = vec![0.0f32; max_pos * head_dim];
        for pos in 0..max_pos {
            for d in 0..half_dim {
                let theta = (pos as f32) / 10000.0_f32.powf((d as f32) / (half_dim as f32));
                cos_sin_data[pos * head_dim + d] = theta.cos();
                cos_sin_data[pos * head_dim + half_dim + d] = theta.sin();
            }
        }

        let positions = vec![5u32, 7];
        let slot_mapping = vec![3u32, (block_size as u32) + 3];

        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_bf16(device: &Device, data: &[f32]) -> Buffer {
            let bf: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(bf.as_slice());
            let buf = device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bf.as_ptr() as *const u8,
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
        fn alloc_zero_bf16(device: &Device, n: usize) -> Buffer {
            let bytes = (n * std::mem::size_of::<bf16>()).max(1);
            let buf = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes);
            }
            buf
        }

        let q_buf = alloc_bf16(&device, &q_data);
        let k_buf = alloc_bf16(&device, &k_data);
        let v_buf = alloc_bf16(&device, &v_data);
        let cos_sin_buf = alloc_bf16(&device, &cos_sin_data);
        let positions_buf = alloc_u32(&device, &positions);
        let slot_buf = alloc_u32(&device, &slot_mapping);
        let kv_k_buf = alloc_zero_bf16(&device, num_blocks * num_kv * block_size * head_dim);
        let kv_v_buf = alloc_zero_bf16(&device, num_blocks * num_kv * block_size * head_dim);

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

        let mut cos_table = vec![0.0_f32; max_pos * head_dim];
        let mut sin_table = vec![0.0_f32; max_pos * head_dim];
        for pos in 0..max_pos {
            for d in 0..half_dim {
                cos_table[pos * head_dim + d] = cos_sin_data[pos * head_dim + d];
                sin_table[pos * head_dim + d] = cos_sin_data[pos * head_dim + half_dim + d];
            }
        }

        // Round-trip inputs through bf16 to match shader precision.
        let bf16_round = |data: &[f32]| -> Vec<f32> {
            data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect()
        };
        let q_b = bf16_round(&q_data);
        let k_b = bf16_round(&k_data);
        let v_b = bf16_round(&v_data);
        let cos_b = bf16_round(&cos_table);
        let sin_b = bf16_round(&sin_table);

        let mut q_out_cpu = vec![0.0_f32; q_data.len()];
        let mut kv_k_cpu = vec![0.0_f32; num_blocks * num_kv * block_size * head_dim];
        let mut kv_v_cpu = vec![0.0_f32; num_blocks * num_kv * block_size * head_dim];
        cpu_golden::rope_append(
            &q_b,
            &k_b,
            &v_b,
            &positions,
            &slot_mapping,
            &cos_b,
            &sin_b,
            &mut q_out_cpu,
            &mut kv_k_cpu,
            &mut kv_v_cpu,
            num_q,
            num_kv,
            head_dim,
            block_size,
        );

        fn read_bf16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const bf16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let q_metal = read_bf16(&q_buf, q_data.len());
        let kv_k_metal = read_bf16(&kv_k_buf, kv_k_cpu.len());
        let kv_v_metal = read_bf16(&kv_v_buf, kv_v_cpu.len());

        let tol: f32 = 5e-2;
        for i in 0..q_data.len() {
            let diff = (q_metal[i] - q_out_cpu[i]).abs();
            assert!(
                diff < tol,
                "rope_append_bf16_l32 q[{i}] metal={} cpu={} diff={}",
                q_metal[i],
                q_out_cpu[i],
                diff
            );
        }
        for i in 0..kv_k_cpu.len() {
            let diff = (kv_k_metal[i] - kv_k_cpu[i]).abs();
            assert!(
                diff < tol,
                "rope_append_bf16_l32 kv_k[{i}] metal={} cpu={} diff={}",
                kv_k_metal[i],
                kv_k_cpu[i],
                diff
            );
        }
        for i in 0..kv_v_cpu.len() {
            let diff = (kv_v_metal[i] - kv_v_cpu[i]).abs();
            assert!(
                diff < tol,
                "rope_append_bf16_l32 kv_v[{i}] metal={} cpu={} diff={}",
                kv_v_metal[i],
                kv_v_cpu[i],
                diff
            );
        }
    }

    /// Reproduces the EXACT runtime binding pattern observed in
    /// `FERRITE_METAL_BAKE_DEBUG=1` for Llama-3.2-1B decode: the
    /// `attention_via_cache_bf16_specialized` dispatch binds the SAME
    /// MTLBuffer at index 0 (output) and index 1 (Q input). The
    /// existing 1B golden uses separate buffers; this one aliases
    /// them. The kernel claims to be safe under aliasing because Q is
    /// loaded into `threadgroup q_local[]` before any writes to
    /// `output`, but if there's a path where the threadgroup load
    /// races against another threadgroup's write, the bug shows up
    /// here and not in the unaliased golden.
    #[cfg(target_os = "macos")]
    #[test]
    fn attention_via_cache_bf16_with_aliased_q_output_llama32_1b() {
        use crate::cpu_golden;
        use ferrite_metal_kernels::metal::MTLSize;
        use half::bf16;

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

        let batch: usize = 1;
        let head_dim = Llama32_1BProbe::HEAD_DIM as usize;
        let num_q = Llama32_1BProbe::NUM_Q_HEADS as usize;
        let num_kv = Llama32_1BProbe::NUM_KV_HEADS as usize;
        let block_size: usize = Llama32_1BProbe::BLOCK_SIZE as usize;
        let max_blocks_per_seq: usize = Llama32_1BProbe::MAX_BLOCKS_PER_SEQ as usize;
        let num_blocks: usize = 4;

        let pipeline = pipelines
            .pipeline_for_dtype::<Llama32_1BProbe>(
                KernelId::AttentionViaCache,
                batch as u32,
                MetalDtype::Bf16,
            )
            .expect("attention_via_cache bf16 pipeline (1B aliased)");

        // Q has 32 heads × 64 head_dim = 2048 elements; the runtime's
        // q-projection output buffer is sized [bucket_m, num_q,
        // head_dim] = [1, 32, 64]. The aliased binding gives the
        // ATTENTION output the SAME 2048-element layout.
        let q_elems = batch * num_q * head_dim;

        let seq_used_k: Vec<u32> = vec![37];
        let mut block_table = vec![0u32; batch * max_blocks_per_seq];
        block_table[0] = 0;
        block_table[1] = 1;
        block_table[2] = 2;

        let q_data: Vec<f32> = (0..q_elems)
            .map(|i| ((i as f32) * 0.013).sin() * 0.5)
            .collect();
        let kv_cache_size = num_blocks * num_kv * block_size * head_dim;
        let mut kv_k_data = vec![0.0_f32; kv_cache_size];
        let mut kv_v_data = vec![0.0_f32; kv_cache_size];
        for (live_block, kv_len) in [(0usize, 16usize), (1usize, 16usize), (2usize, 5usize)] {
            for tok in 0..kv_len {
                for kvh in 0..num_kv {
                    for d in 0..head_dim {
                        let idx = live_block * num_kv * block_size * head_dim
                            + kvh * block_size * head_dim
                            + tok * head_dim
                            + d;
                        let seed = (idx as f32) * 0.0017;
                        kv_k_data[idx] = seed.sin() * 0.5;
                        kv_v_data[idx] = seed.cos() * 0.5;
                    }
                }
            }
        }

        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_bf16(device: &Device, data: &[f32]) -> Buffer {
            let bf: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(bf.as_slice());
            let buf = device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bf.as_ptr() as *const u8,
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

        // Single shared buffer that holds Q on input and gets
        // overwritten with attention output. This is what the runtime
        // does (see FERRITE_METAL_BAKE_DEBUG output: idx=0 == idx=1
        // for AttentionViaCache).
        let qo_buf = alloc_bf16(&device, &q_data);
        let seq_used_buf = alloc_u32(&device, &seq_used_k);
        let block_table_buf = alloc_u32(&device, &block_table);
        let kv_k_buf = alloc_bf16(&device, &kv_k_data);
        let kv_v_buf = alloc_bf16(&device, &kv_v_data);

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&qo_buf), 0); // output
        enc.set_buffer(1, Some(&qo_buf), 0); // Q input — SAME buffer
        enc.set_buffer(2, Some(&seq_used_buf), 0);
        enc.set_buffer(3, Some(&block_table_buf), 0);
        enc.set_buffer(4, Some(&kv_k_buf), 0);
        enc.set_buffer(5, Some(&kv_v_buf), 0);
        // v2 sdpa_vector port requires (1024, 1, 1) = 32 simdgroups × 32
        // lanes; matches lowering.rs:438 for AttentionViaCache.
        enc.dispatch_thread_groups(
            MTLSize::new(batch as u64, num_q as u64, 1),
            MTLSize::new(1024, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let bf16_round = |data: &[f32]| -> Vec<f32> {
            data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect()
        };
        let q_b = bf16_round(&q_data);
        let kv_k_b = bf16_round(&kv_k_data);
        let kv_v_b = bf16_round(&kv_v_data);

        let mut output_cpu = vec![0.0_f32; q_elems];
        cpu_golden::attention_via_cache(
            &q_b,
            &kv_k_b,
            &kv_v_b,
            &block_table,
            &seq_used_k,
            &mut output_cpu,
            num_q,
            num_kv,
            head_dim,
            block_size,
            max_blocks_per_seq,
            Llama32_1BProbe::ATTN_SCALE,
        );

        let output_metal: Vec<f32> =
            unsafe { std::slice::from_raw_parts(qo_buf.contents() as *const bf16, q_elems) }
                .iter()
                .map(|&v| v.to_f32())
                .collect();

        let tol: f32 = 5e-2;
        for i in 0..output_cpu.len() {
            let diff = (output_metal[i] - output_cpu[i]).abs();
            assert!(
                diff < tol,
                "via_cache_bf16_aliased_qo[{i}] (head {} dim {}) metal={} cpu={} diff={}",
                i / head_dim,
                i % head_dim,
                output_metal[i],
                output_cpu[i],
                diff
            );
        }
    }

    /// rope_append golden at Llama-3.2-1B's EXACT decode-1 shape:
    /// HEAD_DIM=64, 32 Q heads, 8 KV heads (GQA 4:1), bucket_m=1
    /// (single decode token), position=36 (matches the runtime
    /// trace's first-decode position on the "Hi" prompt). Fills the
    /// gap between the existing rope_append golden (Llama-3.2-3B
    /// shape, bucket_m=2, position 5/7) and the runtime call the
    /// 1B failure case actually makes.
    #[cfg(target_os = "macos")]
    #[test]
    fn rope_append_bf16_matches_cpu_golden_llama32_1b_decode() {
        use crate::cpu_golden;
        use ferrite_metal_kernels::metal::MTLSize;
        use half::bf16;

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

        let bucket_m: usize = 1;
        let head_dim = Llama32_1BProbe::HEAD_DIM as usize;
        let half_dim = head_dim / 2;
        let num_q = Llama32_1BProbe::NUM_Q_HEADS as usize;
        let num_kv = Llama32_1BProbe::NUM_KV_HEADS as usize;
        let block_size: usize = Llama32_1BProbe::BLOCK_SIZE as usize;
        let num_blocks: usize = 4;
        let max_pos: usize = 64;

        let pipeline = pipelines
            .pipeline_for_dtype::<Llama32_1BProbe>(
                KernelId::RopeAppend,
                bucket_m as u32,
                MetalDtype::Bf16,
            )
            .expect("rope_append bf16 pipeline (1B shape)");

        let q_data: Vec<f32> = (0..bucket_m * num_q * head_dim)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect();
        let k_data: Vec<f32> = (0..bucket_m * num_kv * head_dim)
            .map(|i| ((i as f32) * 0.002).cos())
            .collect();
        let v_data: Vec<f32> = (0..bucket_m * num_kv * head_dim)
            .map(|i| (i as f32) * 0.003)
            .collect();

        let mut cos_sin_data = vec![0.0f32; max_pos * head_dim];
        for pos in 0..max_pos {
            for d in 0..half_dim {
                let theta = (pos as f32) / 10000.0_f32.powf((d as f32) / (half_dim as f32));
                cos_sin_data[pos * head_dim + d] = theta.cos();
                cos_sin_data[pos * head_dim + half_dim + d] = theta.sin();
            }
        }

        // Position 36 — what the runtime feeds at decode-1 after a
        // 36-token prefill (matches the FERRITE_METAL_TRACE log on
        // the "Hi" prompt). Slot 36 = block 2, offset 4.
        let positions = vec![36u32];
        let slot_mapping = vec![(2u32) * (block_size as u32) + 4u32];

        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_bf16(device: &Device, data: &[f32]) -> Buffer {
            let bf: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(bf.as_slice());
            let buf = device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bf.as_ptr() as *const u8,
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
        fn alloc_zero_bf16(device: &Device, n: usize) -> Buffer {
            let bytes = (n * std::mem::size_of::<bf16>()).max(1);
            let buf = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes);
            }
            buf
        }

        let q_buf = alloc_bf16(&device, &q_data);
        let k_buf = alloc_bf16(&device, &k_data);
        let v_buf = alloc_bf16(&device, &v_data);
        let cos_sin_buf = alloc_bf16(&device, &cos_sin_data);
        let positions_buf = alloc_u32(&device, &positions);
        let slot_buf = alloc_u32(&device, &slot_mapping);
        let kv_k_buf = alloc_zero_bf16(&device, num_blocks * num_kv * block_size * head_dim);
        let kv_v_buf = alloc_zero_bf16(&device, num_blocks * num_kv * block_size * head_dim);

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

        let mut cos_table = vec![0.0_f32; max_pos * head_dim];
        let mut sin_table = vec![0.0_f32; max_pos * head_dim];
        for pos in 0..max_pos {
            for d in 0..half_dim {
                cos_table[pos * head_dim + d] = cos_sin_data[pos * head_dim + d];
                sin_table[pos * head_dim + d] = cos_sin_data[pos * head_dim + half_dim + d];
            }
        }

        let bf16_round = |data: &[f32]| -> Vec<f32> {
            data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect()
        };
        let q_b = bf16_round(&q_data);
        let k_b = bf16_round(&k_data);
        let v_b = bf16_round(&v_data);
        let cos_b = bf16_round(&cos_table);
        let sin_b = bf16_round(&sin_table);

        let mut q_out_cpu = vec![0.0_f32; q_data.len()];
        let mut kv_k_cpu = vec![0.0_f32; num_blocks * num_kv * block_size * head_dim];
        let mut kv_v_cpu = vec![0.0_f32; num_blocks * num_kv * block_size * head_dim];
        cpu_golden::rope_append(
            &q_b,
            &k_b,
            &v_b,
            &positions,
            &slot_mapping,
            &cos_b,
            &sin_b,
            &mut q_out_cpu,
            &mut kv_k_cpu,
            &mut kv_v_cpu,
            num_q,
            num_kv,
            head_dim,
            block_size,
        );

        fn read_bf16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const bf16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let q_metal = read_bf16(&q_buf, q_data.len());
        let kv_k_metal = read_bf16(&kv_k_buf, kv_k_cpu.len());
        let kv_v_metal = read_bf16(&kv_v_buf, kv_v_cpu.len());

        let tol: f32 = 5e-2;
        for i in 0..q_data.len() {
            let diff = (q_metal[i] - q_out_cpu[i]).abs();
            assert!(
                diff < tol,
                "rope_append_bf16_l32_1b q[{i}] metal={} cpu={} diff={}",
                q_metal[i],
                q_out_cpu[i],
                diff
            );
        }
        for i in 0..kv_k_cpu.len() {
            let diff = (kv_k_metal[i] - kv_k_cpu[i]).abs();
            assert!(
                diff < tol,
                "rope_append_bf16_l32_1b kv_k[{i}] metal={} cpu={} diff={}",
                kv_k_metal[i],
                kv_k_cpu[i],
                diff
            );
        }
        for i in 0..kv_v_cpu.len() {
            let diff = (kv_v_metal[i] - kv_v_cpu[i]).abs();
            assert!(
                diff < tol,
                "rope_append_bf16_l32_1b kv_v[{i}] metal={} cpu={} diff={}",
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
        // v2 kernel uses 1024 threads/group (32 simdgroups × 32 lanes).
        enc.dispatch_thread_groups(
            MTLSize::new(batch as u64, num_q as u64, 1),
            MTLSize::new(1024, 1, 1),
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

    /// HEAD_DIM=128 + bf16 attention_via_cache golden — the runtime
    /// path used at decode time on Llama-3.2 / Llama-3.1 8B / Qwen-7B.
    /// Dispatches with the runtime's actual `(HEAD_DIM, 1, 1)`
    /// thread-per-tg shape, not the test-only 1024-thread shape, so a
    /// failure here means the kernel itself is wrong at HEAD_DIM=128.
    #[cfg(target_os = "macos")]
    #[test]
    fn attention_via_cache_bf16_matches_cpu_golden_llama32() {
        use crate::cpu_golden;
        use ferrite_metal_kernels::metal::MTLSize;
        use half::bf16;

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

        let batch: usize = 2;
        let head_dim = Llama32Probe::HEAD_DIM as usize;
        let num_q = Llama32Probe::NUM_Q_HEADS as usize;
        let num_kv = Llama32Probe::NUM_KV_HEADS as usize;
        let block_size: usize = Llama32Probe::BLOCK_SIZE as usize;
        let max_blocks_per_seq: usize = Llama32Probe::MAX_BLOCKS_PER_SEQ as usize;
        let num_blocks: usize = 4;

        let pipeline = pipelines
            .pipeline_for_dtype::<Llama32Probe>(
                KernelId::AttentionViaCache,
                batch as u32,
                MetalDtype::Bf16,
            )
            .expect("attention_via_cache bf16 pipeline");

        let seq_used_k: Vec<u32> = vec![10, 17];
        let mut block_table = vec![0u32; batch * max_blocks_per_seq];
        block_table[0 * max_blocks_per_seq + 0] = 0;
        block_table[1 * max_blocks_per_seq + 0] = 2;
        block_table[1 * max_blocks_per_seq + 1] = 3;

        let q_data: Vec<f32> = (0..batch * num_q * head_dim)
            .map(|i| ((i as f32) * 0.013).sin() * 0.5)
            .collect();
        let kv_cache_size = num_blocks * num_kv * block_size * head_dim;
        let mut kv_k_data = vec![0.0_f32; kv_cache_size];
        let mut kv_v_data = vec![0.0_f32; kv_cache_size];
        for (live_block, kv_len) in [(0usize, 10usize), (2usize, 16usize), (3usize, 1usize)] {
            for tok in 0..kv_len {
                for kvh in 0..num_kv {
                    for d in 0..head_dim {
                        let idx = live_block * num_kv * block_size * head_dim
                            + kvh * block_size * head_dim
                            + tok * head_dim
                            + d;
                        let seed = (idx as f32) * 0.0017;
                        kv_k_data[idx] = seed.sin() * 0.5;
                        kv_v_data[idx] = seed.cos() * 0.5;
                    }
                }
            }
        }

        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_bf16(device: &Device, data: &[f32]) -> Buffer {
            let bf: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(bf.as_slice());
            let buf = device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bf.as_ptr() as *const u8,
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
        fn alloc_zero_bf16(device: &Device, n: usize) -> Buffer {
            let bytes = (n * std::mem::size_of::<bf16>()).max(1);
            let buf = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes);
            }
            buf
        }

        let q_buf = alloc_bf16(&device, &q_data);
        let seq_used_buf = alloc_u32(&device, &seq_used_k);
        let block_table_buf = alloc_u32(&device, &block_table);
        let kv_k_buf = alloc_bf16(&device, &kv_k_data);
        let kv_v_buf = alloc_bf16(&device, &kv_v_data);
        let output_buf = alloc_zero_bf16(&device, batch * num_q * head_dim);

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&output_buf), 0);
        enc.set_buffer(1, Some(&q_buf), 0);
        enc.set_buffer(2, Some(&seq_used_buf), 0);
        enc.set_buffer(3, Some(&block_table_buf), 0);
        enc.set_buffer(4, Some(&kv_k_buf), 0);
        enc.set_buffer(5, Some(&kv_v_buf), 0);
        // v2 sdpa_vector port requires (1024, 1, 1) = 32 simdgroups × 32
        // lanes; matches lowering.rs:438 for AttentionViaCache.
        enc.dispatch_thread_groups(
            MTLSize::new(batch as u64, num_q as u64, 1),
            MTLSize::new(1024, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let bf16_round = |data: &[f32]| -> Vec<f32> {
            data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect()
        };
        let q_b = bf16_round(&q_data);
        let kv_k_b = bf16_round(&kv_k_data);
        let kv_v_b = bf16_round(&kv_v_data);

        let mut output_cpu = vec![0.0_f32; batch * num_q * head_dim];
        cpu_golden::attention_via_cache(
            &q_b,
            &kv_k_b,
            &kv_v_b,
            &block_table,
            &seq_used_k,
            &mut output_cpu,
            num_q,
            num_kv,
            head_dim,
            block_size,
            max_blocks_per_seq,
            Llama32Probe::ATTN_SCALE,
        );

        fn read_bf16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const bf16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let output_metal = read_bf16(&output_buf, output_cpu.len());

        let tol: f32 = 5e-2; // bf16 has 7-bit mantissa, more rounding noise than f16
        for i in 0..output_cpu.len() {
            let diff = (output_metal[i] - output_cpu[i]).abs();
            assert!(
                diff < tol,
                "via_cache_bf16_l32[{i}] (seq {} head {} dim {}) metal={} cpu={} diff={}",
                i / (num_q * head_dim),
                (i / head_dim) % num_q,
                i % head_dim,
                output_metal[i],
                output_cpu[i],
                diff
            );
        }
    }

    /// bf16 attention_via_cache golden at Llama-3.2-1B's EXACT decode
    /// shape: HEAD_DIM=64, 32 Q heads, 8 KV heads (GQA 4:1), batch=1.
    /// The existing `attention_via_cache_bf16_matches_cpu_golden_llama32`
    /// runs against the 3B's shape (HEAD_DIM=128, 24q/8kv, batch=2);
    /// no golden so far has covered the 1B's shape, and the runtime
    /// is broken on Llama-3.2-1B (decode collapses to repeated token 0
    /// after the first decode token). This test answers: does the
    /// kernel itself work at the 1B's GQA + HEAD_DIM=64 + batch=1
    /// combination?
    #[cfg(target_os = "macos")]
    #[test]
    fn attention_via_cache_bf16_matches_cpu_golden_llama32_1b_decode() {
        use crate::cpu_golden;
        use ferrite_metal_kernels::metal::MTLSize;
        use half::bf16;

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

        // Llama-3.2-1B decode shape — batch=1 (single decode step).
        let batch: usize = 1;
        let head_dim = Llama32_1BProbe::HEAD_DIM as usize;
        let num_q = Llama32_1BProbe::NUM_Q_HEADS as usize;
        let num_kv = Llama32_1BProbe::NUM_KV_HEADS as usize;
        let block_size: usize = Llama32_1BProbe::BLOCK_SIZE as usize;
        let max_blocks_per_seq: usize = Llama32_1BProbe::MAX_BLOCKS_PER_SEQ as usize;
        let num_blocks: usize = 4;

        let pipeline = pipelines
            .pipeline_for_dtype::<Llama32_1BProbe>(
                KernelId::AttentionViaCache,
                batch as u32,
                MetalDtype::Bf16,
            )
            .expect("attention_via_cache bf16 pipeline (1B shape)");

        // 37-token cache (matches the Llama-3.2-1B "Hi" prompt's actual
        // decode-1 state: 36 prefill tokens + 1 decode-0 token already
        // appended). Spans physical blocks 0..2 (block_size=16, so
        // 37 → blocks 0,1,2 with 5 used in block 2).
        let seq_used_k: Vec<u32> = vec![37];
        let mut block_table = vec![0u32; batch * max_blocks_per_seq];
        block_table[0] = 0;
        block_table[1] = 1;
        block_table[2] = 2;

        let q_data: Vec<f32> = (0..batch * num_q * head_dim)
            .map(|i| ((i as f32) * 0.013).sin() * 0.5)
            .collect();
        let kv_cache_size = num_blocks * num_kv * block_size * head_dim;
        let mut kv_k_data = vec![0.0_f32; kv_cache_size];
        let mut kv_v_data = vec![0.0_f32; kv_cache_size];
        // Fill physical blocks 0,1,2 with deterministic data; block 3
        // stays zero so an over-read into it shows up as drift.
        for (live_block, kv_len) in [(0usize, 16usize), (1usize, 16usize), (2usize, 5usize)] {
            for tok in 0..kv_len {
                for kvh in 0..num_kv {
                    for d in 0..head_dim {
                        let idx = live_block * num_kv * block_size * head_dim
                            + kvh * block_size * head_dim
                            + tok * head_dim
                            + d;
                        let seed = (idx as f32) * 0.0017;
                        kv_k_data[idx] = seed.sin() * 0.5;
                        kv_v_data[idx] = seed.cos() * 0.5;
                    }
                }
            }
        }

        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_bf16(device: &Device, data: &[f32]) -> Buffer {
            let bf: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(bf.as_slice());
            let buf = device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bf.as_ptr() as *const u8,
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
        fn alloc_zero_bf16(device: &Device, n: usize) -> Buffer {
            let bytes = (n * std::mem::size_of::<bf16>()).max(1);
            let buf = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes);
            }
            buf
        }

        let q_buf = alloc_bf16(&device, &q_data);
        let seq_used_buf = alloc_u32(&device, &seq_used_k);
        let block_table_buf = alloc_u32(&device, &block_table);
        let kv_k_buf = alloc_bf16(&device, &kv_k_data);
        let kv_v_buf = alloc_bf16(&device, &kv_v_data);
        let output_buf = alloc_zero_bf16(&device, batch * num_q * head_dim);

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&output_buf), 0);
        enc.set_buffer(1, Some(&q_buf), 0);
        enc.set_buffer(2, Some(&seq_used_buf), 0);
        enc.set_buffer(3, Some(&block_table_buf), 0);
        enc.set_buffer(4, Some(&kv_k_buf), 0);
        enc.set_buffer(5, Some(&kv_v_buf), 0);
        // v2 sdpa_vector port requires (1024, 1, 1) = 32 simdgroups × 32
        // lanes; matches lowering.rs:438 for AttentionViaCache.
        enc.dispatch_thread_groups(
            MTLSize::new(batch as u64, num_q as u64, 1),
            MTLSize::new(1024, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let bf16_round = |data: &[f32]| -> Vec<f32> {
            data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect()
        };
        let q_b = bf16_round(&q_data);
        let kv_k_b = bf16_round(&kv_k_data);
        let kv_v_b = bf16_round(&kv_v_data);

        let mut output_cpu = vec![0.0_f32; batch * num_q * head_dim];
        cpu_golden::attention_via_cache(
            &q_b,
            &kv_k_b,
            &kv_v_b,
            &block_table,
            &seq_used_k,
            &mut output_cpu,
            num_q,
            num_kv,
            head_dim,
            block_size,
            max_blocks_per_seq,
            Llama32_1BProbe::ATTN_SCALE,
        );

        fn read_bf16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const bf16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let output_metal = read_bf16(&output_buf, output_cpu.len());

        let tol: f32 = 5e-2;
        for i in 0..output_cpu.len() {
            let diff = (output_metal[i] - output_cpu[i]).abs();
            assert!(
                diff < tol,
                "via_cache_bf16_l32_1b[{i}] (head {} dim {}) metal={} cpu={} diff={}",
                i / head_dim,
                i % head_dim,
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

    /// M=64 single-sequence prefill test, matching the runtime
    /// bucket size and `cu_seqlens_q = [0, num_real_tokens]` shape.
    /// Catches bugs where the kernel works at small M but breaks at
    /// the actual prefill scale (4 Q-tiles, 32 heads).
    #[cfg(target_os = "macos")]
    #[test]
    fn attention_prefill_contiguous_matches_cpu_golden_m64_single_seq() {
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

        // Single sequence, 23 real tokens, padded to bucket_m=64.
        // cu_seqlens_q = [0, 23]; padding tokens (23..63) get
        // in_range=false → output 0.
        let real_tokens: usize = 23;
        let bucket_m: usize = 64;
        let cu_seqlens_q: Vec<u32> = vec![0, real_tokens as u32];

        let pipeline = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::AttentionPrefillContiguous, bucket_m as u32)
            .expect("attention_prefill pipeline");

        // Q/K/V at the bucket size — only the first `real_tokens` rows
        // matter; padding rows can hold anything (kernel masks them).
        let q_data: Vec<f32> = (0..bucket_m * num_q * head_dim)
            .map(|i| ((i as f32) * 0.011).sin() * 0.5)
            .collect();
        let k_data: Vec<f32> = (0..bucket_m * num_kv * head_dim)
            .map(|i| ((i as f32) * 0.017).cos() * 0.5)
            .collect();
        let v_data: Vec<f32> = (0..bucket_m * num_kv * head_dim)
            .map(|i| ((i as f32) * 0.023).sin() * 0.5)
            .collect();

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
        let output_buf = alloc_zero_f16(&device, bucket_m * num_q * head_dim);

        let num_q_tiles = bucket_m.div_ceil(tile_q);
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

        // Round-trip Q/K/V through f16 for the reference, but truncate
        // to `real_tokens` rows — the cpu_golden's seq_starts shape
        // matches the runtime's single-sequence layout.
        let f16_round = |data: &[f32]| -> Vec<f32> {
            data.iter()
                .map(|&v| half::f16::from_f32(v).to_f32())
                .collect()
        };
        let q_f16 = f16_round(&q_data);
        let k_f16 = f16_round(&k_data);
        let v_f16 = f16_round(&v_data);

        let seq_starts: Vec<usize> = cu_seqlens_q.iter().map(|&v| v as usize).collect();
        let mut output_cpu = vec![0.0_f32; bucket_m * num_q * head_dim];
        // The cpu_golden expects only `total = real_tokens` rows, but
        // we sized the output buffer for `bucket_m`. Run the reference
        // on the truncated view; padding rows stay 0 (matches kernel).
        let q_real = &q_f16[..real_tokens * num_q * head_dim];
        let k_real = &k_f16[..real_tokens * num_kv * head_dim];
        let v_real = &v_f16[..real_tokens * num_kv * head_dim];
        let mut output_cpu_real = vec![0.0_f32; real_tokens * num_q * head_dim];
        cpu_golden::attention_prefill(
            q_real,
            k_real,
            v_real,
            &mut output_cpu_real,
            &seq_starts,
            num_q,
            num_kv,
            head_dim,
            TinyLlamaProbe::ATTN_SCALE,
        );
        output_cpu[..real_tokens * num_q * head_dim].copy_from_slice(&output_cpu_real);

        fn read_f16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const half::f16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let output_metal = read_f16(&output_buf, output_cpu.len());

        // Sanity: row 0 and row 22's outputs should DIFFER (they see
        // different K/V context — row 0 only sees position 0, row 22
        // sees positions 0..22 with non-trivial softmax weights).
        let head0 = 0usize;
        let row0_off = (0 * num_q + head0) * head_dim;
        let row22_off = (22 * num_q + head0) * head_dim;
        let row0 = &output_metal[row0_off..row0_off + 4];
        let row22 = &output_metal[row22_off..row22_off + 4];
        assert_ne!(row0, row22, "row 0 == row 22 — attention collapsed");

        let tol: f32 = 1e-2; // larger tol — fp16 accumulation over 23 K positions
        for i in 0..real_tokens * num_q * head_dim {
            let diff = (output_metal[i] - output_cpu[i]).abs();
            assert!(
                diff < tol,
                "attn_prefill_m64[{i}] (row {} head {} dim {}) metal={} cpu={} diff={}",
                i / (num_q * head_dim),
                (i / head_dim) % num_q,
                i % head_dim,
                output_metal[i],
                output_cpu[i],
                diff
            );
        }
    }

    /// HEAD_DIM=128 + f16 isolation test — drives the same
    /// `attention_prefill_contiguous_f16_specialized` shader that
    /// HEAD_DIM=64 already passes against, but at the larger head
    /// dim. If this fails, the kernel itself has a HEAD_DIM=128
    /// bug independent of bf16; if it passes, the bf16 variant has
    /// a dtype-specific bug.
    #[cfg(target_os = "macos")]
    #[test]
    fn attention_prefill_contiguous_f16_matches_cpu_golden_head_dim_128() {
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

        let head_dim = Llama32F16Probe::HEAD_DIM as usize;
        let num_q = Llama32F16Probe::NUM_Q_HEADS as usize;
        let num_kv = Llama32F16Probe::NUM_KV_HEADS as usize;
        let tile_q: usize = Llama32F16Probe::PREFILL_TILE_Q as usize;

        let real_tokens: usize = 23;
        let bucket_m: usize = 64;
        let cu_seqlens_q: Vec<u32> = vec![0, real_tokens as u32];

        let pipeline = pipelines
            .pipeline_for::<Llama32F16Probe>(KernelId::AttentionPrefillContiguous, bucket_m as u32)
            .expect("attention_prefill f16 pipeline");

        let q_data: Vec<f32> = (0..bucket_m * num_q * head_dim)
            .map(|i| ((i as f32) * 0.011).sin() * 0.5)
            .collect();
        let k_data: Vec<f32> = (0..bucket_m * num_kv * head_dim)
            .map(|i| ((i as f32) * 0.017).cos() * 0.5)
            .collect();
        let v_data: Vec<f32> = (0..bucket_m * num_kv * head_dim)
            .map(|i| ((i as f32) * 0.023).sin() * 0.5)
            .collect();

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
        let output_buf = alloc_zero_f16(&device, bucket_m * num_q * head_dim);

        let num_q_tiles = bucket_m.div_ceil(tile_q);
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

        let f16_round = |data: &[f32]| -> Vec<f32> {
            data.iter()
                .map(|&v| half::f16::from_f32(v).to_f32())
                .collect()
        };
        let q_f16 = f16_round(&q_data);
        let k_f16 = f16_round(&k_data);
        let v_f16 = f16_round(&v_data);

        let seq_starts: Vec<usize> = cu_seqlens_q.iter().map(|&v| v as usize).collect();
        let q_real = &q_f16[..real_tokens * num_q * head_dim];
        let k_real = &k_f16[..real_tokens * num_kv * head_dim];
        let v_real = &v_f16[..real_tokens * num_kv * head_dim];
        let mut output_cpu_real = vec![0.0_f32; real_tokens * num_q * head_dim];
        cpu_golden::attention_prefill(
            q_real,
            k_real,
            v_real,
            &mut output_cpu_real,
            &seq_starts,
            num_q,
            num_kv,
            head_dim,
            Llama32F16Probe::ATTN_SCALE,
        );

        fn read_f16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const half::f16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let output_metal = read_f16(&output_buf, bucket_m * num_q * head_dim);

        eprintln!(
            "f16 row 0 d=0..3: {:?}",
            (0..4).map(|j| output_metal[j]).collect::<Vec<_>>()
        );
        eprintln!(
            "f16 row 1 d=0..3 (metal, cpu): {:?}",
            (0..4)
                .map(|j| (output_metal[24 * 128 + j], output_cpu_real[24 * 128 + j]))
                .collect::<Vec<_>>()
        );

        let tol: f32 = 1e-2;
        for i in 0..real_tokens * num_q * head_dim {
            let diff = (output_metal[i] - output_cpu_real[i]).abs();
            assert!(
                diff < tol,
                "f16 attn_prefill_l32[{i}] (row {} head {} dim {}) metal={} cpu={} diff={}",
                i / (num_q * head_dim),
                (i / head_dim) % num_q,
                i % head_dim,
                output_metal[i],
                output_cpu_real[i],
                diff
            );
        }
    }

    /// HEAD_DIM=128 + bf16 prefill golden, mirroring the M=64 f16 test
    /// at Llama-3.2-3B shapes. Catches HEAD_DIM=128-only bugs in
    /// `attention_prefill_contiguous_bf16_specialized` — the f16 path
    /// is goldenned at HEAD_DIM=64 (TinyLlama), so without this the
    /// bf16/HEAD_DIM=128 surface only got tested through end-to-end
    /// generation.
    #[cfg(target_os = "macos")]
    #[test]
    fn attention_prefill_contiguous_bf16_matches_cpu_golden_llama32() {
        use crate::cpu_golden;
        use ferrite_metal_kernels::metal::MTLSize;
        use half::bf16;

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

        let head_dim = Llama32Probe::HEAD_DIM as usize;
        let num_q = Llama32Probe::NUM_Q_HEADS as usize;
        let num_kv = Llama32Probe::NUM_KV_HEADS as usize;
        let tile_q: usize = Llama32Probe::PREFILL_TILE_Q as usize;

        // 2 real tokens, padded to bucket_m=16. Minimal repro for the
        // bf16 + HEAD_DIM=128 collapse — if row 1 still inherits
        // row 0's softmax in this 2-token case, the kernel state-
        // reset bug is independent of token count.
        let real_tokens: usize = 2;
        let bucket_m: usize = 16;
        let cu_seqlens_q: Vec<u32> = vec![0, real_tokens as u32];

        let pipeline = pipelines
            .pipeline_for_dtype::<Llama32Probe>(
                KernelId::AttentionPrefillContiguous,
                bucket_m as u32,
                MetalDtype::Bf16,
            )
            .expect("attention_prefill bf16 pipeline");

        let q_data: Vec<f32> = (0..bucket_m * num_q * head_dim)
            .map(|i| ((i as f32) * 0.011).sin() * 0.5)
            .collect();
        let k_data: Vec<f32> = (0..bucket_m * num_kv * head_dim)
            .map(|i| ((i as f32) * 0.017).cos() * 0.5)
            .collect();
        let v_data: Vec<f32> = (0..bucket_m * num_kv * head_dim)
            .map(|i| ((i as f32) * 0.023).sin() * 0.5)
            .collect();

        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_bf16(device: &Device, data: &[f32]) -> Buffer {
            let bf: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(bf.as_slice());
            let buf = device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bf.as_ptr() as *const u8,
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
        fn alloc_zero_bf16(device: &Device, n: usize) -> Buffer {
            let bytes = (n * std::mem::size_of::<bf16>()).max(1);
            let buf = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes);
            }
            buf
        }

        let q_buf = alloc_bf16(&device, &q_data);
        let k_buf = alloc_bf16(&device, &k_data);
        let v_buf = alloc_bf16(&device, &v_data);
        let cu_buf = alloc_u32(&device, &cu_seqlens_q);
        let output_buf = alloc_zero_bf16(&device, bucket_m * num_q * head_dim);

        let num_q_tiles = bucket_m.div_ceil(tile_q);
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

        // Round-trip Q/K/V through bf16 for the cpu reference.
        let bf16_round = |data: &[f32]| -> Vec<f32> {
            data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect()
        };
        let q_bf = bf16_round(&q_data);
        let k_bf = bf16_round(&k_data);
        let v_bf = bf16_round(&v_data);

        let seq_starts: Vec<usize> = cu_seqlens_q.iter().map(|&v| v as usize).collect();
        let q_real = &q_bf[..real_tokens * num_q * head_dim];
        let k_real = &k_bf[..real_tokens * num_kv * head_dim];
        let v_real = &v_bf[..real_tokens * num_kv * head_dim];
        let mut output_cpu_real = vec![0.0_f32; real_tokens * num_q * head_dim];
        cpu_golden::attention_prefill(
            q_real,
            k_real,
            v_real,
            &mut output_cpu_real,
            &seq_starts,
            num_q,
            num_kv,
            head_dim,
            Llama32Probe::ATTN_SCALE,
        );

        fn read_bf16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const bf16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let output_metal = read_bf16(&output_buf, bucket_m * num_q * head_dim);

        // Pre-assert dump: print row 0 / row 1 / row 22 head 0 dim
        // 0..3 so a failure tells us where the kernel diverges. Row 0
        // sees only K[0]; row 1 sees K[0..1]; row 22 sees K[0..22]. If
        // row 0 is right but row 1 is wrong, the kernel state isn't
        // reset between local_q iterations.
        let snap = |row: usize| -> [(f32, f32); 4] {
            let off = (row * num_q + 0) * head_dim;
            let mut out = [(0.0, 0.0); 4];
            for j in 0..4 {
                out[j] = (output_metal[off + j], output_cpu_real[off + j]);
            }
            out
        };
        eprintln!("bf16 row 0 head 0 dim 0..3 (metal, cpu): {:?}", snap(0));
        eprintln!("bf16 row 1 head 0 dim 0..3 (metal, cpu): {:?}", snap(1));

        // bf16 has 7-bit mantissa (~3-4 decimal digits). Tolerance is
        // larger than f16 by an order of magnitude.
        let tol: f32 = 5e-2;
        for i in 0..real_tokens * num_q * head_dim {
            let diff = (output_metal[i] - output_cpu_real[i]).abs();
            assert!(
                diff < tol,
                "attn_prefill_bf16_l32[{i}] (row {} head {} dim {}) metal={} cpu={} diff={}",
                i / (num_q * head_dim),
                (i / head_dim) % num_q,
                i % head_dim,
                output_metal[i],
                output_cpu_real[i],
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
        enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(256, 1, 1));
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

    /// BF16 RmsNorm at M=64. Compiles `rmsnorm_bf16_specialized` via
    /// the dtype-aware pipeline picker, dispatches against bf16 host
    /// data, and compares to the bf16-round-tripped CPU reference.
    /// First end-to-end exercise of the bf16 path on the metal
    /// backend.
    #[cfg(target_os = "macos")]
    #[test]
    fn rmsnorm_bf16_matches_cpu_golden_m64() {
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

        let m: usize = 64;
        let hidden = TinyLlamaProbe::Q_SIZE;
        let pipeline = pipelines
            .pipeline_for_dtype::<TinyLlamaProbe>(KernelId::RmsNorm, m as u32, MetalDtype::Bf16)
            .expect("rmsnorm bf16 pipeline");

        let input_data: Vec<f32> = (0..m * hidden)
            .map(|i| ((i as f32) * 0.011).sin() * 0.5)
            .collect();
        let weight_data: Vec<f32> = (0..hidden)
            .map(|i| 1.0 + ((i as f32) * 0.017).cos() * 0.05)
            .collect();

        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_bf16(device: &Device, data: &[f32]) -> Buffer {
            let bf16_data: Vec<half::bf16> =
                data.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(bf16_data.as_slice());
            let buf = device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bf16_data.as_ptr() as *const u8,
                    buf.contents() as *mut u8,
                    bytes,
                );
            }
            buf
        }
        fn alloc_zero_bf16(device: &Device, n: usize) -> Buffer {
            let bytes = (n * std::mem::size_of::<half::bf16>()).max(1);
            let buf = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes);
            }
            buf
        }

        let input_buf = alloc_bf16(&device, &input_data);
        let weight_buf = alloc_bf16(&device, &weight_data);
        let output_buf = alloc_zero_bf16(&device, m * hidden);

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&output_buf), 0);
        enc.set_buffer(1, Some(&input_buf), 0);
        enc.set_buffer(2, Some(&weight_buf), 0);
        enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // BF16-round-trip Q/K to match the kernel's input precision.
        let input_bf16: Vec<f32> = input_data
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        let weight_bf16: Vec<f32> = weight_data
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        let mut output_cpu = vec![0.0_f32; m * hidden];
        let eps = 1e-5_f32;
        for row in 0..m {
            let base = row * hidden;
            cpu_golden::rmsnorm(
                &input_bf16[base..base + hidden],
                &weight_bf16,
                &mut output_cpu[base..base + hidden],
                eps,
            );
        }

        fn read_bf16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const half::bf16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let output_metal = read_bf16(&output_buf, output_cpu.len());

        // Sanity: row 0 and row 22 should differ (input rows differ).
        let row0 = &output_metal[0..4];
        let row22 = &output_metal[22 * hidden..22 * hidden + 4];
        assert_ne!(
            row0, row22,
            "bf16 rmsnorm: row 0 == row 22 — kernel collapsed distinct inputs"
        );

        // bf16 has 7-bit mantissa vs fp16's 10-bit; loosen tolerance
        // to the bf16 ULP scale (~5e-3 for values near 1.0, scaling
        // with magnitude). The cpu reference is bf16-round-tripped on
        // input/weight but accumulates in f32, mirroring the kernel.
        let tol: f32 = 1e-2;
        for i in 0..output_cpu.len() {
            let diff = (output_metal[i] - output_cpu[i]).abs();
            assert!(
                diff < tol,
                "rmsnorm_bf16[{i}] (row {} col {}) metal={} cpu={} diff={}",
                i / hidden,
                i % hidden,
                output_metal[i],
                output_cpu[i],
                diff
            );
        }
    }

    /// Same as `rmsnorm_matches_cpu_golden` but at M=64 (the prefill
    /// bucket size used at TinyLlama runtime). Distinguishes a "kernel
    /// works at small M but breaks at large M" bug from a runtime
    /// dispatch / binding bug.
    #[cfg(target_os = "macos")]
    #[test]
    fn rmsnorm_matches_cpu_golden_m64() {
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

        let m: usize = 64;
        let hidden = TinyLlamaProbe::Q_SIZE;
        let pipeline = pipelines
            .pipeline_for::<TinyLlamaProbe>(KernelId::RmsNorm, m as u32)
            .expect("rmsnorm pipeline");

        // Per-row distinct: input[row, col] = sin((row*hidden + col)*k).
        // Different rows have very different sums-of-squares.
        let input_data: Vec<f32> = (0..m * hidden)
            .map(|i| ((i as f32) * 0.011).sin() * 0.5)
            .collect();
        let weight_data: Vec<f32> = (0..hidden)
            .map(|i| 1.0 + ((i as f32) * 0.017).cos() * 0.05)
            .collect();

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
        enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let input_f16: Vec<f32> = input_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let weight_f16: Vec<f32> = weight_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let mut output_cpu = vec![0.0_f32; m * hidden];
        let eps = 1e-5_f32;
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

        // Sanity: row 0 and row 22 outputs should DIFFER (the inputs do).
        let row0 = &output_metal[0..4];
        let row22 = &output_metal[22 * hidden..22 * hidden + 4];
        assert_ne!(
            row0, row22,
            "row 0 == row 22 — kernel produced identical output for distinct input rows"
        );

        let tol: f32 = 5e-3;
        for i in 0..output_cpu.len() {
            let diff = (output_metal[i] - output_cpu[i]).abs();
            assert!(
                diff < tol,
                "rmsnorm[{i}] (row {} col {}) metal={} cpu={} diff={}",
                i / hidden,
                i % hidden,
                output_metal[i],
                output_cpu[i],
                diff
            );
        }
    }

    /// Same as `rmsnorm_matches_cpu_golden_m64` but binds the SAME
    /// buffer for input and output (the runtime's hot path: the macro
    /// colors the embedding tile and the rmsnorm output tile to one
    /// arena slot, so the kernel runs in-place on slot 0). If this
    /// test passes but the runtime corrupts rows, the bug is in the
    /// runtime's bindings / dispatch, not the kernel.
    #[cfg(target_os = "macos")]
    #[test]
    fn rmsnorm_in_place_matches_cpu_golden_m64() {
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

        let m: usize = 64;
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

        // Single buffer used for both input and output. The CPU
        // reference reads from a separate copy so the comparison stays
        // valid after the kernel writes back.
        let inout_buf = alloc_f16(&device, &input_data);
        let weight_buf = alloc_f16(&device, &weight_data);

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&inout_buf), 0); // OUT = same buffer
        enc.set_buffer(1, Some(&inout_buf), 0); // IN  = same buffer
        enc.set_buffer(2, Some(&weight_buf), 0);
        enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let input_f16: Vec<f32> = input_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let weight_f16: Vec<f32> = weight_data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let mut output_cpu = vec![0.0_f32; m * hidden];
        let eps = 1e-5_f32;
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
        let output_metal = read_f16(&inout_buf, output_cpu.len());

        let row0 = &output_metal[0..4];
        let row22 = &output_metal[22 * hidden..22 * hidden + 4];
        assert_ne!(
            row0, row22,
            "in-place rmsnorm: row 0 == row 22 — kernel corrupted distinct input rows"
        );

        let tol: f32 = 5e-3;
        for i in 0..output_cpu.len() {
            let diff = (output_metal[i] - output_cpu[i]).abs();
            assert!(
                diff < tol,
                "in-place rmsnorm[{i}] (row {} col {}) metal={} cpu={} diff={}",
                i / hidden,
                i % hidden,
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
    /// Runs at M=4 (small smoke) AND M=64 (TinyLlama prefill bucket).
    #[cfg(target_os = "macos")]
    #[test]
    fn fused_add_rmsnorm_matches_cpu_golden() {
        for m in [64usize, 4usize] {
            run_fused_add_rmsnorm_check(m);
        }
    }

    #[cfg(target_os = "macos")]
    fn run_fused_add_rmsnorm_check(m: usize) {
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

        let residual_buf = alloc_f16(&device, &residual_data);
        let delta_buf = alloc_f16(&device, &delta_data);
        let weight_buf = alloc_f16(&device, &weight_data);

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&residual_buf), 0);
        enc.set_buffer(1, Some(&delta_buf), 0);
        enc.set_buffer(2, Some(&weight_buf), 0);
        enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(256, 1, 1));
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
        cpu_golden::fused_add_rmsnorm(&mut residual_cpu, &mut delta_cpu, &weight_f16, eps, hidden);

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
        let k: usize = TinyLlamaProbe::Q_SIZE; // 2048

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

        // M=1 → decode kernel dispatch shape (MLX gemv port: blockM=4).
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&output_buf), 0);
        enc.set_buffer(1, Some(&input_buf), 0);
        enc.set_buffer(2, Some(&weight_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(4), 1, 1),
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

    /// Same as `fused_mlp_tinyllama_shape_matches_cpu_golden` but
    /// drives the bf16 decode kernel at Llama-3.2-3B shapes (M=1,
    /// N=8192, K=3072). This is the path the runtime hits at decode;
    /// previously only goldenned at TinyLlama shape + f16.
    #[cfg(target_os = "macos")]
    #[test]
    fn fused_mlp_decode_bf16_matches_cpu_golden_llama32() {
        use crate::cpu_golden;
        use ferrite_metal_kernels::metal::MTLSize;
        use half::bf16;

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
        let n: usize = Llama32Probe::INTERMEDIATE_SIZE; // 8192
        let k: usize = Llama32Probe::Q_SIZE; // 3072

        let pipeline = pipelines
            .pipeline_for_dtype::<Llama32Probe>(
                KernelId::FusedGateUpSiluMul,
                m as u32,
                MetalDtype::Bf16,
            )
            .expect("fused_mlp bf16 pipeline");

        // Magnitudes ~0.05 so K=3072 sums stay within bf16 range.
        let input_data: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32) * 0.013).sin() * 0.05)
            .collect();
        let weight_data: Vec<f32> = (0..(2 * n) * k)
            .map(|i| ((i as f32) * 0.019).cos() * 0.05)
            .collect();

        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_bf16(device: &Device, data: &[f32]) -> Buffer {
            let bf: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(bf.as_slice());
            let buf = device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bf.as_ptr() as *const u8,
                    buf.contents() as *mut u8,
                    bytes,
                );
            }
            buf
        }
        fn alloc_zero_bf16(device: &Device, n: usize) -> Buffer {
            let bytes = (n * std::mem::size_of::<bf16>()).max(1);
            let buf = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes);
            }
            buf
        }

        let input_buf = alloc_bf16(&device, &input_data);
        let weight_buf = alloc_bf16(&device, &weight_data);
        let output_buf = alloc_zero_bf16(&device, m * n);

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&output_buf), 0);
        enc.set_buffer(1, Some(&input_buf), 0);
        enc.set_buffer(2, Some(&weight_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(4), 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let bf16_round = |data: &[f32]| -> Vec<f32> {
            data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect()
        };
        let input_b = bf16_round(&input_data);
        let weight_b = bf16_round(&weight_data);
        let gate_w = &weight_b[0..n * k];
        let up_w = &weight_b[n * k..2 * n * k];
        let mut gate = vec![0.0_f32; m * n];
        let mut up = vec![0.0_f32; m * n];
        cpu_golden::gemm(&input_b, gate_w, &mut gate, m, k, n);
        cpu_golden::gemm(&input_b, up_w, &mut up, m, k, n);
        let mut output_cpu = vec![0.0_f32; m * n];
        cpu_golden::fused_gate_up_silu_mul(&gate, &up, &mut output_cpu);

        fn read_bf16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const bf16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let output_metal = read_bf16(&output_buf, output_cpu.len());

        let tol: f32 = 8e-2;
        let mut max_diff: f32 = 0.0;
        for i in 0..output_cpu.len() {
            let diff = (output_metal[i] - output_cpu[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < tol,
                "fused_mlp_decode_bf16_l32[{i}] metal={} cpu={} diff={} max_so_far={}",
                output_metal[i],
                output_cpu[i],
                diff,
                max_diff
            );
        }
        eprintln!("fused_mlp_decode_bf16_l32: max_abs_diff={max_diff}");
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
        for m in [64usize, 16usize, 1usize] {
            run_fused_mlp_check(m);
        }
    }

    /// Generic bf16 GEMM golden at TinyLlama (M=64, N=2048, K=2048)
    /// AND Llama-3.2-3B (M=64, N=3072, K=3072) shapes — confirms the
    /// `gemm_bf16_specialized` kernel runs correctly across the range
    /// of shapes the runtime drives. This is the kernel that replaces
    /// MPS' `MPSMatrixMultiplication` for the bf16 path; MPS rejects
    /// `MPSDataTypeBFloat16` at runtime.
    #[cfg(target_os = "macos")]
    #[test]
    fn gemm_bf16_matches_cpu_golden() {
        for (m, n, k) in [
            (64usize, 2048usize, 2048usize), // TinyLlama Q/K/V/O shape K-side
            (64usize, 3072usize, 3072usize), // Llama-3.2-3B Q/O shape
            (64usize, 1024usize, 3072usize), // Llama-3.2-3B K/V shape
            (64usize, 3072usize, 8192usize), // Llama-3.2-3B down-proj shape
            // Decode shapes (M=1) — lm_head + per-layer projections.
            (1usize, 128256usize, 3072usize), // Llama-3.2-3B lm_head/embed (tied)
            (1usize, 3072usize, 3072usize),   // Llama-3.2-3B Q/O at decode
            (1usize, 1024usize, 3072usize),   // Llama-3.2-3B K/V at decode
            (1usize, 3072usize, 8192usize),   // Llama-3.2-3B down at decode
            (1usize, 512usize, 2048usize),    // Llama-3.2-1B K/V at decode
            (1usize, 2048usize, 8192usize),   // Llama-3.2-1B down at decode
            (1usize, 128256usize, 2048usize), // Llama-3.2-1B lm_head at decode
            (1usize, 2048usize, 2048usize),   // Llama-3.2-1B Q/O at decode (gap-fill)
            (1usize, 8192usize, 2048usize),   // Llama-3.2-1B gate/up at decode (gap-fill)
        ] {
            run_gemm_bf16_check(m, n, k);
        }
    }

    #[cfg(target_os = "macos")]
    fn run_gemm_bf16_check(m: usize, n: usize, k: usize) {
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

        let pipeline = pipelines
            .pipeline_for_gemm_bf16(m as u32, n as u32, k as u32)
            .expect("gemm_bf16 pipeline");

        let input_data: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32) * 0.013).sin() * 0.3)
            .collect();
        let weight_data: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.019).cos() * 0.3)
            .collect();

        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_bf16(device: &Device, data: &[f32]) -> Buffer {
            let bf16_data: Vec<half::bf16> =
                data.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(bf16_data.as_slice());
            let buf = device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bf16_data.as_ptr() as *const u8,
                    buf.contents() as *mut u8,
                    bytes,
                );
            }
            buf
        }
        fn alloc_zero_bf16(device: &Device, n: usize) -> Buffer {
            let bytes = (n * std::mem::size_of::<half::bf16>()).max(1);
            let buf = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes);
            }
            buf
        }

        let input_buf = alloc_bf16(&device, &input_data);
        let weight_buf = alloc_bf16(&device, &weight_data);
        let output_buf = alloc_zero_bf16(&device, m * n);

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&output_buf), 0);
        enc.set_buffer(1, Some(&input_buf), 0);
        enc.set_buffer(2, Some(&weight_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(8), (m as u64).div_ceil(8), 1),
            MTLSize::new(32, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let input_bf16: Vec<f32> = input_data
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        let weight_bf16: Vec<f32> = weight_data
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        let mut output_cpu = vec![0.0_f32; m * n];
        cpu_golden::gemm(&input_bf16, &weight_bf16, &mut output_cpu, m, k, n);

        fn read_bf16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const half::bf16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let output_metal = read_bf16(&output_buf, output_cpu.len());

        // bf16 reduction over K up to 8192: looser tol than the
        // matmul-into-f32 path strictly needs, but sized so a real
        // numerical break would still trigger.
        let tol: f32 = 5e-2;
        for i in 0..output_cpu.len() {
            let diff = (output_metal[i] - output_cpu[i]).abs();
            assert!(
                diff < tol,
                "gemm_bf16 m={m} n={n} k={k} [{i}] (row {} col {}) metal={} cpu={} diff={}",
                i / n,
                i % n,
                output_metal[i],
                output_cpu[i],
                diff
            );
        }
    }

    /// BF16 fused MLP golden — exercises both shader variants:
    /// `..._gemm_bf16_specialized` (M >= 2, uses `simdgroup_bfloat8x8`
    /// MMA tiles) and `..._decode_bf16_specialized` (M=1 GEMV-style).
    /// Validates that Metal's `simdgroup_matrix<bfloat,8,8>` runs on
    /// this hardware and produces values within bf16 tolerance of the
    /// cpu reference.
    #[cfg(target_os = "macos")]
    #[test]
    fn fused_mlp_bf16_matches_cpu_golden() {
        for m in [64usize, 16usize, 1usize] {
            run_fused_mlp_bf16_check(m);
        }
    }

    #[cfg(target_os = "macos")]
    fn run_fused_mlp_bf16_check(m: usize) {
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
            .pipeline_for_dtype::<MlpProbe>(
                KernelId::FusedGateUpSiluMul,
                m as u32,
                MetalDtype::Bf16,
            )
            .expect("fused_mlp bf16 pipeline");

        let input_data: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32) * 0.013).sin() * 0.3)
            .collect();
        let weight_data: Vec<f32> = (0..(2 * n) * k)
            .map(|i| ((i as f32) * 0.019).cos() * 0.3)
            .collect();

        use ferrite_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};
        fn alloc_bf16(device: &Device, data: &[f32]) -> Buffer {
            let bf16_data: Vec<half::bf16> =
                data.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let bytes = std::mem::size_of_val(bf16_data.as_slice());
            let buf = device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bf16_data.as_ptr() as *const u8,
                    buf.contents() as *mut u8,
                    bytes,
                );
            }
            buf
        }
        fn alloc_zero_bf16(device: &Device, n: usize) -> Buffer {
            let bytes = (n * std::mem::size_of::<half::bf16>()).max(1);
            let buf = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes);
            }
            buf
        }

        let input_buf = alloc_bf16(&device, &input_data);
        let weight_buf = alloc_bf16(&device, &weight_data);
        let output_buf = alloc_zero_bf16(&device, m * n);

        let (threadgroups, threads_per_threadgroup) = if m == 1 {
            (
                MTLSize::new((n as u64).div_ceil(4), 1, 1),
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

        let input_bf16: Vec<f32> = input_data
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        let weight_bf16: Vec<f32> = weight_data
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();

        let gate_w = &weight_bf16[0..n * k];
        let up_w = &weight_bf16[n * k..2 * n * k];
        let mut gate = vec![0.0_f32; m * n];
        let mut up = vec![0.0_f32; m * n];
        cpu_golden::gemm(&input_bf16, gate_w, &mut gate, m, k, n);
        cpu_golden::gemm(&input_bf16, up_w, &mut up, m, k, n);
        let mut output_cpu = vec![0.0_f32; m * n];
        cpu_golden::fused_gate_up_silu_mul(&gate, &up, &mut output_cpu);

        fn read_bf16(buf: &Buffer, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(buf.contents() as *const half::bf16, n) }
                .iter()
                .map(|&v| v.to_f32())
                .collect()
        }
        let output_metal = read_bf16(&output_buf, output_cpu.len());

        // bf16 has 7-bit mantissa vs fp16's 10-bit; loosen the
        // tolerance correspondingly. Reduction over K=16 tile gives
        // O(K * eps_bf16) cumulative error.
        let tol: f32 = 2e-2;
        for i in 0..output_cpu.len() {
            let diff = (output_metal[i] - output_cpu[i]).abs();
            assert!(
                diff < tol,
                "fused_mlp_bf16 m={m} [{i}] metal={} cpu={} diff={}",
                output_metal[i],
                output_cpu[i],
                diff
            );
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
                MTLSize::new((n as u64).div_ceil(4), 1, 1),
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
