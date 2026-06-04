// SPDX-License-Identifier: Apache-2.0
//! Typed per-kernel function-constants structs (Phase 2 of
//! `FERRITE_METAL_TYPE_SAFETY_PLAN.md`).
//!
//! Each kernel family that takes `[[function_constant(N)]]` slots gets
//! one struct. Constructing the struct lists every slot the kernel
//! reads — so adding a slot to a `.metal` source without adding a
//! field here, or vice versa, breaks every call site at compile time.
//!
//! Catches bug class #1 (the `ATTN_PAGED_DEBUG_MODE` slot-99 omission
//! that turned production output into `" pr formal formal ..."` —
//! see `feedback_never_stop_after_commit` + commit `b3ddb3b46`).
//!
//! Convention: each struct provides
//! `impl From<Self> for Vec<ConstantValue>` that emits the slots in
//! their declared order — same order as the matching `.metal` header.
//! Lowering arms construct the struct, then `.into()` for assignment
//! to `LoweredCommand::constants`.

use ferrite_metal_kernels::specialized_pipeline_cache::{ConstSlot, ConstantValue};

use super::ids::{
    AttnDebugMode, AttnScale, BlockSize, BlocksPerChunk, BucketM, HeadDim, HiddenSize,
    IntermediateSize, KDim, KDimI32, KPartitionSizeI32, MDimI32, MaxBlocksPerSeq, NDim, NDimI32,
    NumKvHeads, NumQHeads, QSize, RmsNormEps, RotDim, SplitK,
};

// ── Embed (token gather) ───────────────────────────────────────────

/// `KernelId::Embed` (`embed_<dtype>_specialized`).
pub struct EmbedConstants {
    pub bucket_m: BucketM,
    pub q_size: QSize,
}

impl From<EmbedConstants> for Vec<ConstantValue> {
    fn from(c: EmbedConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.bucket_m.get()),
            ConstantValue::uint(ConstSlot(1), c.q_size.get()),
        ]
    }
}

// ── RmsNorm / FusedAddRmsNorm ──────────────────────────────────────

/// `KernelId::RmsNorm` (`rmsnorm_<T_act>_s_<T_scale>_specialized`)
/// and `KernelId::FusedAddRmsNorm` (`fused_add_rmsnorm_<...>`).
/// Constants are identical across the two.
pub struct RmsNormConstants {
    pub bucket_m: BucketM,
    pub q_size: QSize,
    pub rms_norm_eps: RmsNormEps,
    /// Zero-centered (Gemma / Qwen3.5) gain offset: effective gain =
    /// `weight + weight_offset`. `1.0` for `(1 + weight)` arches, `0.0`
    /// for plain RMSNorm. Function constant 3 in `rmsnorm.metal` /
    /// `fused_add_rmsnorm.metal`.
    pub weight_offset: f32,
}

impl From<RmsNormConstants> for Vec<ConstantValue> {
    fn from(c: RmsNormConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.bucket_m.get()),
            ConstantValue::uint(ConstSlot(1), c.q_size.get()),
            ConstantValue::float(ConstSlot(2), c.rms_norm_eps.get()),
            ConstantValue::float(ConstSlot(3), c.weight_offset),
        ]
    }
}

// ── RopeAppend ─────────────────────────────────────────────────────

/// `KernelId::RopeAppend` (`rope_append_<dtype>_specialized`).
pub struct RopeAppendConstants {
    pub head_dim: HeadDim,
    pub num_q_heads: NumQHeads,
    pub num_kv_heads: NumKvHeads,
    pub rot_dim: RotDim,
    pub block_size: BlockSize,
    pub blocks_per_chunk: BlocksPerChunk,
}

impl From<RopeAppendConstants> for Vec<ConstantValue> {
    fn from(c: RopeAppendConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.head_dim.get()),
            ConstantValue::uint(ConstSlot(1), c.num_q_heads.get()),
            ConstantValue::uint(ConstSlot(2), c.num_kv_heads.get()),
            ConstantValue::uint(ConstSlot(3), c.rot_dim.get()),
            ConstantValue::uint(ConstSlot(4), c.block_size.get()),
            ConstantValue::uint(ConstSlot(5), c.blocks_per_chunk.get()),
        ]
    }
}

// ── FusedQkvRopeCache (dense BF16/F16) ─────────────────────────────

/// `KernelId::FusedQkvRopeCache`
/// (`fused_qkv_rope_cache_<dtype>_specialized`).
pub struct FusedQkvRopeCacheConstants {
    pub q_size: QSize,
    pub num_q_heads: NumQHeads,
    pub num_kv_heads: NumKvHeads,
    pub head_dim: HeadDim,
    pub rot_dim: RotDim,
    pub block_size: BlockSize,
    pub bucket_m: BucketM,
    pub blocks_per_chunk: BlocksPerChunk,
}

impl From<FusedQkvRopeCacheConstants> for Vec<ConstantValue> {
    fn from(c: FusedQkvRopeCacheConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.q_size.get()),
            ConstantValue::uint(ConstSlot(1), c.num_q_heads.get()),
            ConstantValue::uint(ConstSlot(2), c.num_kv_heads.get()),
            ConstantValue::uint(ConstSlot(3), c.head_dim.get()),
            ConstantValue::uint(ConstSlot(4), c.rot_dim.get()),
            ConstantValue::uint(ConstSlot(5), c.block_size.get()),
            ConstantValue::uint(ConstSlot(6), c.bucket_m.get()),
            ConstantValue::uint(ConstSlot(7), c.blocks_per_chunk.get()),
        ]
    }
}

// ── AttentionViaCache (decode) ─────────────────────────────────────

/// `KernelId::AttentionViaCache`
/// (`attention_via_cache_v2_<dtype>_specialized`).
pub struct AttentionViaCacheConstants {
    pub head_dim: HeadDim,
    pub num_q_heads: NumQHeads,
    pub num_kv_heads: NumKvHeads,
    pub attn_scale: AttnScale,
    pub block_size: BlockSize,
    pub max_blocks: MaxBlocksPerSeq,
    pub blocks_per_chunk: BlocksPerChunk,
}

impl From<AttentionViaCacheConstants> for Vec<ConstantValue> {
    fn from(c: AttentionViaCacheConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.head_dim.get()),
            ConstantValue::uint(ConstSlot(1), c.num_q_heads.get()),
            ConstantValue::uint(ConstSlot(2), c.num_kv_heads.get()),
            ConstantValue::float(ConstSlot(3), c.attn_scale.get()),
            ConstantValue::uint(ConstSlot(4), c.block_size.get()),
            ConstantValue::uint(ConstSlot(5), c.max_blocks.get()),
            ConstantValue::uint(ConstSlot(6), c.blocks_per_chunk.get()),
        ]
    }
}

// ── AttentionPrefillSdpaPaged (sdpa + steel) ───────────────────────

/// `KernelId::AttentionPrefillSdpaPaged` for both the
/// `attention_prefill_sdpa_v2_paged_*` and `attention_steel_paged_*`
/// kernel families.
///
/// Steel kernel reads slot 99 (`ATTN_PAGED_DEBUG_MODE`); the
/// sdpa_vector kernel ignores it. The lowering arm sets
/// `debug_mode = Some(AttnDebugMode(0))` for steel and `None` for sdpa
/// to keep the typed surface honest — Metal pipeline build tolerates
/// extra constants but failing to bind a declared slot is exactly the
/// `b3ddb3b46` regression.
pub struct AttentionPrefillPagedConstants {
    pub head_dim: HeadDim,
    pub num_q_heads: NumQHeads,
    pub num_kv_heads: NumKvHeads,
    pub attn_scale: AttnScale,
    pub block_size: BlockSize,
    pub max_blocks: MaxBlocksPerSeq,
    pub blocks_per_chunk: BlocksPerChunk,
    /// `Some(0)` for the steel kernel (production), `None` for the
    /// sdpa_vector kernel (declares no slot 99).
    pub debug_mode: Option<AttnDebugMode>,
}

impl From<AttentionPrefillPagedConstants> for Vec<ConstantValue> {
    fn from(c: AttentionPrefillPagedConstants) -> Self {
        let mut v = vec![
            ConstantValue::uint(ConstSlot(0), c.head_dim.get()),
            ConstantValue::uint(ConstSlot(1), c.num_q_heads.get()),
            ConstantValue::uint(ConstSlot(2), c.num_kv_heads.get()),
            ConstantValue::float(ConstSlot(3), c.attn_scale.get()),
            ConstantValue::uint(ConstSlot(4), c.block_size.get()),
            ConstantValue::uint(ConstSlot(5), c.max_blocks.get()),
            ConstantValue::uint(ConstSlot(6), c.blocks_per_chunk.get()),
        ];
        if let Some(dm) = c.debug_mode {
            v.push(ConstantValue::uint(ConstSlot(99), dm.get()));
        }
        v
    }
}

// ── MLX-affine QMV (decode matvec) ────────────────────────────────

/// `KernelId::AffineQmvQuad` / `AffineQmvFast` / `AffineQmv`
/// (`quantized_qmv` library; constants declared as signed `int`).
pub struct AffineQmvConstants {
    pub k: KDimI32,
    pub n: NDimI32,
}

impl From<AffineQmvConstants> for Vec<ConstantValue> {
    fn from(c: AffineQmvConstants) -> Self {
        vec![
            ConstantValue::int(ConstSlot(0), c.k.get()),
            ConstantValue::int(ConstSlot(1), c.n.get()),
        ]
    }
}

// ── MLX-affine QMM_T (prefill matmul) ─────────────────────────────

/// `KernelId::AffineQmmT` / `AffineQmmTNax`
/// (`quantized_qmm.metal` / `quantized_qmm_nax.metal`; constants
/// declared as signed `int`).
pub struct AffineQmmTConstants {
    pub k: KDimI32,
    pub n: NDimI32,
    pub m: MDimI32,
}

impl From<AffineQmmTConstants> for Vec<ConstantValue> {
    fn from(c: AffineQmmTConstants) -> Self {
        vec![
            ConstantValue::int(ConstSlot(0), c.k.get()),
            ConstantValue::int(ConstSlot(1), c.n.get()),
            ConstantValue::int(ConstSlot(2), c.m.get()),
        ]
    }
}

// ── MLX-affine QMM_T SplitK ───────────────────────────────────────

/// `KernelId::AffineQmmTSplitK`. The `k_partition_size` slot was the
/// historical mistake — omitting it leaves the partition stride
/// undefined and every layer's prefill output is garbage. Exhaustive
/// struct so it can't be omitted.
pub struct AffineQmmTSplitKConstants {
    pub k: KDimI32,
    pub n: NDimI32,
    pub m: MDimI32,
    pub k_partition_size: KPartitionSizeI32,
}

impl From<AffineQmmTSplitKConstants> for Vec<ConstantValue> {
    fn from(c: AffineQmmTSplitKConstants) -> Self {
        vec![
            ConstantValue::int(ConstSlot(0), c.k.get()),
            ConstantValue::int(ConstSlot(1), c.n.get()),
            ConstantValue::int(ConstSlot(2), c.m.get()),
            ConstantValue::int(ConstSlot(3), c.k_partition_size.get()),
        ]
    }
}

// ── SplitKReduceSum ────────────────────────────────────────────────

/// `KernelId::SplitKReduceSum`
/// (`quantized_splitk_reduce.metal::splitk_reduce_sum_<dtype>`).
pub struct SplitKReduceSumConstants {
    pub bucket_m: BucketM,
    pub n: NDim,
    pub split_k: SplitK,
}

impl From<SplitKReduceSumConstants> for Vec<ConstantValue> {
    fn from(c: SplitKReduceSumConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.bucket_m.get()),
            ConstantValue::uint(ConstSlot(1), c.n.get()),
            ConstantValue::uint(ConstSlot(2), c.split_k.get()),
        ]
    }
}

// ── SiluMul ────────────────────────────────────────────────────────

/// `KernelId::SiluMul` (`silu_mul.metal::silu_mul_<dtype>`).
/// `n = M * intermediate_size` — total output elements.
pub struct SiluMulConstants {
    pub n: HiddenSize,
}

impl From<SiluMulConstants> for Vec<ConstantValue> {
    fn from(c: SiluMulConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(0), c.n.get())]
    }
}

// ── GateApply / GateSplit (Qwen3.5 attention output gate) ─────────

/// `KernelId::GateApply` (`gate_apply.metal::gate_apply_<dtype>`).
/// `n = M * num_heads * head_dim` — total output elements. Same single
/// `n` constant as `SiluMulConstants`, kept distinct for clarity.
pub type GateApplyConstants = SiluMulConstants;

/// `KernelId::GateSplit` (`gate_split.metal::gate_split_<dtype>`).
/// `n` = per-output element count (`M * num_heads * head_dim`);
/// `head_dim` / `num_heads` drive the per-head interleaved source index.
pub struct GateSplitConstants {
    pub n: HiddenSize,
    pub head_dim: u32,
    pub num_heads: u32,
}

impl From<GateSplitConstants> for Vec<ConstantValue> {
    fn from(c: GateSplitConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.n.get()),
            ConstantValue::uint(ConstSlot(1), c.head_dim),
            ConstantValue::uint(ConstSlot(2), c.num_heads),
        ]
    }
}

/// `KernelId::GateScale` (`gate_scale.metal::gate_scale_<dtype>`).
/// `n` = total output elements (`M * hidden_size`); `cols` =
/// `hidden_size` — the gate's row index for the `[T, 1]` broadcast is
/// `gid / cols`.
pub struct GateScaleConstants {
    pub n: HiddenSize,
    pub cols: u32,
}

impl From<GateScaleConstants> for Vec<ConstantValue> {
    fn from(c: GateScaleConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.n.get()),
            ConstantValue::uint(ConstSlot(1), c.cols),
        ]
    }
}

// ── AffineEmbed (MLX-affine int4 embedding lookup) ────────────────

/// `KernelId::AffineEmbed`
/// (`quantized_dequantize.metal::affine_embed_<dtype>_gs_<gs>_b_4`).
pub struct AffineEmbedConstants {
    pub hidden_size: HiddenSize,
}

impl From<AffineEmbedConstants> for Vec<ConstantValue> {
    fn from(c: AffineEmbedConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(0), c.hidden_size.get())]
    }
}

// ── GatherLastToken / ScatterFirstToLastRow ───────────────────────

/// `KernelId::GatherLastToken` and `KernelId::ScatterFirstToLastRow`
/// share the single-`row_stride` constant layout
/// (`gather_last_token_<dtype>_specialized` and
/// `scatter_first_to_last_row_<dtype>_specialized`).
pub struct GatherLastTokenConstants {
    pub row_stride: HiddenSize,
}

impl From<GatherLastTokenConstants> for Vec<ConstantValue> {
    fn from(c: GatherLastTokenConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(0), c.row_stride.get())]
    }
}

// ── FusedGateUpSiluMul (decode + prefill) ─────────────────────────

/// `KernelId::FusedGateUpSiluMul` decode branch
/// (`fused_gate_up_silu_mul_..._specialized`, decode `bucket_m == 1`).
/// Slots are 3/4/5 — the prefill branch uses 6/7/8, so the two are
/// distinct struct types.
pub struct FusedGateUpSiluMulDecodeConstants {
    pub bucket_m: BucketM,
    pub intermediate_size: IntermediateSize,
    pub q_size: QSize,
}

impl From<FusedGateUpSiluMulDecodeConstants> for Vec<ConstantValue> {
    fn from(c: FusedGateUpSiluMulDecodeConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(3), c.bucket_m.get()),
            ConstantValue::uint(ConstSlot(4), c.intermediate_size.get()),
            ConstantValue::uint(ConstSlot(5), c.q_size.get()),
        ]
    }
}

/// `KernelId::FusedGateUpSiluMul` prefill branch
/// (`fused_gate_up_silu_mul_gemm_steel_..._specialized`).
pub struct FusedGateUpSiluMulPrefillConstants {
    pub bucket_m: BucketM,
    pub intermediate_size: IntermediateSize,
    pub q_size: QSize,
}

impl From<FusedGateUpSiluMulPrefillConstants> for Vec<ConstantValue> {
    fn from(c: FusedGateUpSiluMulPrefillConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(6), c.bucket_m.get()),
            ConstantValue::uint(ConstSlot(7), c.intermediate_size.get()),
            ConstantValue::uint(ConstSlot(8), c.q_size.get()),
        ]
    }
}

// ── Synth-* (compiler-emitted megakernels) ────────────────────────

/// `KernelId::SynthPreAttn` / `KernelId::SynthMlpPreDown` /
/// `KernelId::SynthGateUpSiluMul` — the macro-emitted megakernels.
///
/// All three bake every dim into MSL `constant constexpr` literals at
/// synth time. Only the per-bucket `M` stays a function constant
/// (slot 0).
pub struct SynthMegakernelConstants {
    pub bucket_m: BucketM,
}

impl From<SynthMegakernelConstants> for Vec<ConstantValue> {
    fn from(c: SynthMegakernelConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(0), c.bucket_m.get())]
    }
}

// Keep `KDim` / `NDim` re-exported even though the int32 siblings
// (`KDimI32`/`NDimI32`) cover the qmv/qmm_t shaders today. SplitK
// reduce and the synth-* family need the unsigned form.
#[allow(dead_code)]
const _: fn() = || {
    let _ = std::marker::PhantomData::<KDim>;
    let _ = std::marker::PhantomData::<NDim>;
};
