// SPDX-License-Identifier: Apache-2.0
//! Universal interpreter instruction set.
//!
//! `Instruction<W>` is a closed enum over every kernel-call shape
//! any solver-picked `Implementation` produces. The match on
//! `Instruction<W>::eval` lives ONCE in this module — not per
//! canonical, not per arch. Each arm calls existing kernels in
//! `ferrite-kernels` directly. Per-canonical specialization flows
//! through:
//!
//! 1. `W` — the per-canonical `Weights` type. Each variant's
//!    `weight_fn` field is `for<'a> fn(&'a W, u32) -> &'a Layer`,
//!    so the same enum dispatches against any concrete `Weights`.
//! 2. [`CanonicalParams`] — trait implemented per canonical with
//!    associated `const`s for model-wide values (head_dim,
//!    intermediate_size, attn_scale, …). Eval bodies read these
//!    as `W::HEAD_DIM` etc., never as runtime fields.
//!
//! Per canonical the macro emits ONLY: a `Weights` struct,
//! `impl CanonicalParams for Weights { … }`, static
//! `&[Instruction<Weights>]` slices for backbone + lm_head per
//! bucket, and a 1-line forward shim that delegates to [`run`].
//! No `__dispatch_one`, no `__interpret`, no per-canonical `Op`.

#![cfg(feature = "cuda")]

use crate::ForwardCtx;
use crate::tile_table::{TileEntry, take_owned, tile_ref, view};
use ferrite_cuda_core::alloc::OwnedTensor;
use ferrite_cuda_core::device::GpuDevice;
use ferrite_cuda_core::tensor::{GpuTensor, MAX_DIMS};
use ferrite_kernels::attention_helpers as ah;
use ferrite_kernels::cutlass;
use ferrite_kernels::flashinfer;
use ferrite_kernels::kernels;
use ferrite_kernels::layers::{
    Bnb4bitLinear, CohereLayerNorm, Embedding, Fp8AnyLinear, LinearLayer, MarlinLinear, RmsNorm,
};
use ferrite_kernels::layers_moe::{
    DeepSeekV2Fp8BlockMoELayer, DeepSeekV2GgmlMoELayer, DeepSeekV2MoELayer,
};

/// Per-canonical model parameters. Implemented by each canonical's
/// `Weights` so the universal `Instruction::eval` body can read
/// model constants without storing them on every variant instance.
/// Defaults to 0 / 0.0 / -1 for fields the canonical doesn't use.
pub trait CanonicalParams {
    const HEAD_DIM: u32;
    const NUM_Q_HEADS: u32;
    const NUM_KV_HEADS: u32;
    const Q_SIZE: usize;
    const KV_SIZE: usize;
    const INTERMEDIATE_SIZE: usize;
    const ATTN_SCALE: f32;
    const ATTN_SOFTCAP: f32;
    const SLIDING_WINDOW: i32;
    const KV_LORA_RANK: usize;
    const QK_NOPE_HEAD_DIM: usize;
    const QK_ROPE_HEAD_DIM: usize;
    const V_HEAD_DIM: usize;
    const FINAL_LOGIT_SOFTCAPPING: f32;
    /// `qk_nope_head_dim + qk_rope_head_dim` (MlaAttention).
    const QK_HEAD_DIM: usize;
    /// MlaAttention scale: `1/sqrt(qk_head_dim)` w/ YaRN correction.
    const MLA_ATTN_SCALE: f32;
}

/// Runtime state passed by `&mut` into every `op.eval(&mut ctx)`.
/// Constants live on `W: CanonicalParams`, NOT here.
pub struct InterpreterCtx<'a, W> {
    pub wm: &'a W,
    pub tiles: &'a mut Vec<Option<TileEntry>>,
    pub fwd: &'a ForwardCtx<'a>,
    pub device: &'a mut GpuDevice,
    /// Iter index of the enclosing `Op::Loop`, else 0.
    pub layer_offset: u32,
}

// Type aliases for variant fields.
pub type WtFn<W, L> = for<'a> fn(&'a W, u32) -> &'a L;
pub type CosSinFn<W> = for<'a> fn(&'a W, u32) -> GpuTensor;

/// Universal opcode set. Tuple variants throughout — keeps each
/// row in a per-canonical static slice on a single line of cargo
/// expand. Field order per variant matches the per-Impl
/// `OpcodeShape::fields` order in `impl_lib.rs`.
#[allow(clippy::type_complexity)]
pub enum Instruction<W> {
    Embed(u32, WtFn<W, Embedding>),
    RmsNorm(u32, u32, u32, WtFn<W, RmsNorm>),
    LayerNorm(u32, u32, u32, WtFn<W, CohereLayerNorm>),
    Reshape(u32, u32, [u32; MAX_DIMS], [u8; MAX_DIMS], u8),
    Add(u32, u32),
    /// Tensor-parallel all-reduce-sum on the slot in place. Inserted
    /// by the lowering pass after every gemm whose weight is
    /// row-parallel (`ShardDim1`) and after the vocab-parallel embed.
    /// At tp=1 the lowering pass emits zero of these.
    #[cfg(feature = "nccl")]
    AllReduce(u32),
    /// Tensor-parallel all-gather along the last dim: `(in_slot, out_slot)`.
    /// Inserted by the lowering pass after the lm_head Gemm at tp>1
    /// (lm_head is vocab-parallel `ShardDim0`).
    #[cfg(feature = "nccl")]
    AllGather(u32, u32),
    ScalarMul(u32, u32, f32),
    TanhSoftCap(u32, u32),
    FusedAddRmsNorm(u32, u32, u32, WtFn<W, RmsNorm>),
    FusedAddRmsNormWithOffset(u32, u32, u32, f32, WtFn<W, RmsNorm>),
    ScalarOffsetRmsNorm(u32, u32, u32, f32, WtFn<W, RmsNorm>),
    /// Norm→Gemm fusion: `cutlass_gemm(rms_norm(in), gemm_w)`. The
    /// CUTLASS tile is bucket-pickable per `CUTLASS_TILE_ZOO` entry.
    /// Reuses existing `kernels::rms_norm` + `cutlass::cutlass_gemm`
    /// — no new .cu file. Matches body norms whose only downstream
    /// consumer is a single dense Gemm.
    CutlassFusedRmsNormGemm(
        u32,
        u32,
        u32,
        WtFn<W, RmsNorm>,
        WtFn<W, LinearLayer>,
        u32,
        u32,
        u32,
        u32,
        u32,
    ),
    /// LayerNorm→Gemm sibling for Cohere-style architectures.
    CutlassFusedLayerNormGemm(
        u32,
        u32,
        u32,
        WtFn<W, CohereLayerNorm>,
        WtFn<W, LinearLayer>,
        u32,
        u32,
        u32,
        u32,
        u32,
    ),
    /// (Add, RmsNorm, Gemm) 3-tile fusion. Claim shape captures
    /// the lm_head canonical pattern `x = x + delta; logits =
    /// lm_head(rmsnorm(x))`. Runtime: `fused_add_rms_norm_inplace`
    /// then `cutlass_gemm`. The Add's residual update is exposed as
    /// a TensorView aliasing the residual upstream OwnedTensor (same
    /// alias semantics as `FusedAddRmsNorm`); the Gemm output is a
    /// fresh OwnedTensor.
    CutlassFusedAddRmsNormGemm(
        u32,
        u32,
        u32,
        u32,
        WtFn<W, RmsNorm>,
        WtFn<W, LinearLayer>,
        u32,
        u32,
        u32,
        u32,
        u32,
    ),
    Gemm(u32, u32, u32, WtFn<W, LinearLayer>, u32, u32),
    /// cuBLAS-side peer to `CutlassGemmAdd`. cuBLAS GEMM produces a
    /// delta; `add_inplace` then folds it into the residual buffer.
    /// Output is the residual upstream's OwnedTensor (aliased via
    /// the codegen prelude); no `out_slot` payload.
    FusedCublasGemmAdd(u32, u32, u32, WtFn<W, LinearLayer>, u32, u32),
    FusedGemmBias(u32, u32, u32, WtFn<W, LinearLayer>),
    FusedGateUpSiluMul(u32, u32, u32, WtFn<W, LinearLayer>),
    FusedGateUpGeluMul(u32, u32, u32, WtFn<W, LinearLayer>),
    FusedQkvRopeCache(u32, u32, u32, WtFn<W, LinearLayer>, CosSinFn<W>, bool, bool),
    FusedQkvQkNormRopeCache(
        u32,
        u32,
        u32,
        WtFn<W, LinearLayer>,
        WtFn<W, LinearLayer>,
        WtFn<W, LinearLayer>,
        WtFn<W, RmsNorm>,
        WtFn<W, RmsNorm>,
        CosSinFn<W>,
        f32,
        f32,
    ),
    FusedQkvRopePrefill(
        u32,
        u32,
        u32,
        u32,
        u32,
        WtFn<W, LinearLayer>,
        CosSinFn<W>,
        bool,
        bool,
    ),
    AttentionViaCache(u32, u32, u32, CosSinFn<W>, bool),
    AttentionPrefillContiguous(u32, u32, u32, u32, bool),
    SlidingAttentionViaCache(u32, u32, u32, CosSinFn<W>, bool),
    SlidingAttentionPrefillContiguous(u32, u32, u32, u32, bool),
    FlashInferAttentionDecode(u32, u32, u32, CosSinFn<W>, u32, bool),
    FlashInferAttentionPrefill(u32, u32, u32, u32, u32, u32, bool),
    RopeAppend(u32, u32, u32, u32, u32, u32, u32, CosSinFn<W>, bool),
    MlaSplit(u32, u32, u32),
    MlaAttention(u32, u32, u32, u32, u32, CosSinFn<W>),
    DeepSeekMoe(u32, u32, u32, WtFn<W, DeepSeekV2MoELayer>),
    DeepSeekMoeFp8Block(u32, u32, u32, WtFn<W, DeepSeekV2Fp8BlockMoELayer>),
    DeepSeekMoeGgml(u32, u32, u32, WtFn<W, DeepSeekV2GgmlMoELayer>),
    CutlassGemm(u32, u32, u32, WtFn<W, LinearLayer>, u32, u32, u32, u32, u32),
    CutlassGemmSplitK(
        u32,
        u32,
        u32,
        WtFn<W, LinearLayer>,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
    ),
    CutlassGemmAdd(u32, u32, u32, WtFn<W, LinearLayer>, u32, u32, u32, u32, u32),
    CutlassGemv(u32, u32, u32, WtFn<W, LinearLayer>, u32, u32),
    CutlassFusedGemmBias(u32, u32, u32, WtFn<W, LinearLayer>, u32, u32, u32, u32, u32),
    CutlassFusedGateUpSiluMul(u32, u32, u32, WtFn<W, LinearLayer>, u32, u32, u32),
    CutlassFusedGateUpGeluMul(u32, u32, u32, WtFn<W, LinearLayer>, u32, u32, u32, u32, u32),
    CutlassFusedQkvRopeCache(
        u32,
        u32,
        u32,
        WtFn<W, LinearLayer>,
        CosSinFn<W>,
        bool,
        u32,
        u32,
        u32,
        u32,
        u32,
    ),
    CutlassFusedQkvRopePrefill(
        u32,
        u32,
        u32,
        u32,
        u32,
        WtFn<W, LinearLayer>,
        CosSinFn<W>,
        bool,
        u32,
        u32,
        u32,
        u32,
        u32,
    ),
    MarlinGemm(u32, u32, u32, WtFn<W, MarlinLinear>),
    MarlinFusedGateUpSiluMul(u32, u32, u32, WtFn<W, MarlinLinear>),
    MarlinFusedGateUpGeluMul(u32, u32, u32, WtFn<W, MarlinLinear>),
    MarlinFusedQkvRopeCache(u32, u32, u32, WtFn<W, MarlinLinear>, CosSinFn<W>),
    MarlinFusedQkvRopePrefill(u32, u32, u32, u32, u32, WtFn<W, MarlinLinear>, CosSinFn<W>),
    Bnb4Gemm(u32, u32, u32, WtFn<W, Bnb4bitLinear>),
    Bnb4FusedGateUpSiluMul(u32, u32, u32, WtFn<W, Bnb4bitLinear>),
    Bnb4FusedGateUpGeluMul(u32, u32, u32, WtFn<W, Bnb4bitLinear>),
    Bnb4FusedQkvRopeCache(u32, u32, u32, WtFn<W, Bnb4bitLinear>, CosSinFn<W>),
    Bnb4FusedQkvRopePrefill(u32, u32, u32, u32, u32, WtFn<W, Bnb4bitLinear>, CosSinFn<W>),
    GgmlGemm(u32, u32, u32, WtFn<W, LinearLayer>),
    GgmlFusedGateUpSiluMul(u32, u32, u32, WtFn<W, LinearLayer>),
    GgmlFusedGateUpGeluMul(u32, u32, u32, WtFn<W, LinearLayer>),
    GgmlFusedQkvRopeCache(u32, u32, u32, WtFn<W, LinearLayer>, CosSinFn<W>, bool),
    GgmlFusedQkvRopePrefill(u32, u32, u32, u32, u32, WtFn<W, LinearLayer>, CosSinFn<W>),
    Fp8Gemm(u32, u32, u32, WtFn<W, Fp8AnyLinear>),
    Fp8FusedGemmBias(u32, u32, u32, WtFn<W, Fp8AnyLinear>),
    Fp8FusedGateUpSiluMul(u32, u32, u32, WtFn<W, Fp8AnyLinear>),
    Fp8FusedGateUpGeluMul(u32, u32, u32, WtFn<W, Fp8AnyLinear>),
    Fp8FusedQkvRopeCache(u32, u32, u32, WtFn<W, Fp8AnyLinear>, CosSinFn<W>),
    Fp8FusedQkvRopePrefill(u32, u32, u32, u32, u32, WtFn<W, Fp8AnyLinear>, CosSinFn<W>),
    /// Re-run the next `body_len` instructions `count` times.
    Loop(u32, u32),
    /// `tiles[dst] = Some(View(src))`.
    Alias(u32, u32),
    /// Drop the OwnedTensor at `slot`.
    Free(u32),
}

impl<W> Copy for Instruction<W> {}
impl<W> Clone for Instruction<W> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

/// Assert a runtime weight tensor's `[N, K]` shape matches the
/// codegen-time constants the `Instruction` was emitted with. The
/// constants come from the FUF's `eval_shape` at solve time; the
/// runtime tensor is whatever `WtFn` resolved to from the loaded
/// safetensors. A mismatch means the loader produced a weight whose
/// shape disagrees with the model the solver compiled against —
/// silent shape drift here corrupts every output. Real `assert!`,
/// not `debug_assert!`, because release builds need to fail loud
/// rather than march on with a K-mismatch.
///
/// At tp>1 the codegen-time (n, k) reflects the *unsharded* model
/// because shape inference unifies `num_q_heads * head_dim` with
/// `hidden_size` (numerically equal in most arches), losing the
/// distinction between sharded-axis dims and replicated-axis dims.
/// The runtime tensor is per-rank-sharded by the loader, so the
/// numbers legitimately disagree at tp>1. Skip the check there;
/// the kernel itself uses the runtime tensor's shapes directly,
/// so the assertion is purely a sanity check that's only sound at
/// tp=1.
#[track_caller]
fn assert_weight_shape(
    op: &'static str,
    weight: ferrite_cuda_core::tensor::GpuTensor,
    n: u32,
    k: u32,
    tp_active: bool,
) {
    if tp_active {
        return;
    }
    let actual_n = weight.dim(0) as u32;
    let actual_k = weight.dim(1) as u32;
    assert_eq!(
        actual_n, n,
        "{op}: weight N (out_features) mismatch — runtime={actual_n} codegen={n}"
    );
    assert_eq!(
        actual_k, k,
        "{op}: weight K (in_features) mismatch — runtime={actual_k} codegen={k}"
    );
}

/// Whether a TP group is attached on this forward pass (i.e. tp>1).
/// Used by `assert_weight_shape` to skip its check at tp>1 where
/// runtime per-rank shapes legitimately disagree with the codegen's
/// unified-bounds shapes.
#[inline]
fn tp_active<W>(_ctx: &InterpreterCtx<'_, W>) -> bool {
    #[cfg(feature = "nccl")]
    {
        _ctx.fwd.tp_group.is_some()
    }
    #[cfg(not(feature = "nccl"))]
    {
        false
    }
}

impl<W: CanonicalParams> Instruction<W> {
    /// Evaluate one instruction. Closed match (no `_` arm).
    /// `Loop` is dispatched by [`run`] — never reaches here.
    ///
    /// # Safety
    /// Tile slot indices in range; weight accessors produce live
    /// GPU memory; `ctx.device.compute_stream` is live.
    #[allow(clippy::too_many_lines)]
    pub unsafe fn eval(&self, ctx: &mut InterpreterCtx<'_, W>) {
        match *self {
            Instruction::Embed(out_slot, weight_fn) => unsafe {
                let weight = (weight_fn)(ctx.wm, 0u32).weight;
                // At tp=1 the embed weight covers the full vocab and
                // vocab_offset is 0 — the mask never trips. At tp>1
                // the weight is the per-rank `[vocab/tp, hidden]`
                // shard; vocab_offset = rank * vocab_per_rank gives
                // each rank a disjoint slice of the global vocab.
                // Per-Embed call follows with an AllReduce-sum
                // (injected by tp_lowering when shard-kind for the
                // embed weight is ShardDim0).
                let vocab_per_rank = weight.dim(0) as u32;
                #[cfg(feature = "nccl")]
                let vocab_offset = ctx
                    .fwd
                    .tp_group
                    .map(|g| (g.rank() as u32) * vocab_per_rank)
                    .unwrap_or(0);
                #[cfg(not(feature = "nccl"))]
                let vocab_offset: u32 = 0;
                let out = kernels::embedding_gather_masked(
                    weight,
                    *ctx.fwd.input_ids,
                    vocab_offset,
                    vocab_per_rank,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::RmsNorm(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let out = kernels::rms_norm(
                    *v,
                    w.weight,
                    w.eps,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::LayerNorm(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let out = kernels::cohere_layer_norm(
                    *v,
                    w.weight,
                    w.eps,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Reshape(in_slot, out_slot, dims_lit, dims_nt_pow, ndim) => {
                let upstream = tile_ref(ctx.tiles, in_slot).as_gpu_tensor(ctx.tiles);
                let nt = (*ctx.fwd.input_ids).dim(0);
                let mut shape = [0usize; MAX_DIMS];
                let nd = ndim as usize;
                for i in 0..nd {
                    let mut d = dims_lit[i] as usize;
                    for _ in 0..(dims_nt_pow[i] as usize) {
                        d *= nt;
                    }
                    shape[i] = d;
                }
                let reshaped = upstream.reshape(&shape[..nd]);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Reshaped {
                    ref_slot: in_slot,
                    tensor: reshaped,
                });
            }
            Instruction::Add(delta_slot, residual_slot) => unsafe {
                let delta = tile_ref(ctx.tiles, delta_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                kernels::add_inplace(*residual, *delta, ctx.device.compute_stream);
            },
            #[cfg(feature = "nccl")]
            Instruction::AllReduce(slot) => unsafe {
                let group = ctx.fwd.tp_group.expect(
                    "Instruction::AllReduce reached eval but \
                     ForwardCtx::tp_group is None — caller must \
                     attach an NcclGroup at tp_world_size > 1",
                );
                let gt = tile_ref(ctx.tiles, slot).as_gpu_tensor(ctx.tiles);
                group
                    .all_reduce_inplace_promote(gt, &mut ctx.device.caching)
                    .expect("NCCL all_reduce_inplace_promote failed");
            },
            #[cfg(feature = "nccl")]
            Instruction::AllGather(in_slot, out_slot) => unsafe {
                let group = ctx.fwd.tp_group.expect(
                    "Instruction::AllGather reached eval but \
                     ForwardCtx::tp_group is None — caller must \
                     attach an NcclGroup at tp_world_size > 1",
                );
                let v = tile_ref(ctx.tiles, in_slot).as_gpu_tensor(ctx.tiles);
                let out = group.all_gather_last_dim(v, &mut ctx.device.caching);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::ScalarMul(in_slot, out_slot, scale) => {
                let owned = take_owned(ctx.tiles, in_slot);
                unsafe {
                    kernels::scale_inplace(*owned, scale, &ctx.device.cublas);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            }
            Instruction::TanhSoftCap(in_slot, out_slot) => {
                let owned = take_owned(ctx.tiles, in_slot);
                unsafe {
                    kernels::tanh_softcap_inplace(
                        *owned,
                        W::FINAL_LOGIT_SOFTCAPPING,
                        ctx.device.compute_stream,
                    );
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            }
            Instruction::FusedAddRmsNorm(delta_slot, residual_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let delta = tile_ref(ctx.tiles, delta_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let _ = kernels::fused_add_rms_norm_inplace(
                    *delta,
                    *residual,
                    w.weight,
                    w.eps,
                    ctx.device.compute_stream,
                );
            },
            Instruction::FusedAddRmsNormWithOffset(
                delta_slot,
                residual_slot,
                layer,
                offset,
                weight_fn,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let delta = tile_ref(ctx.tiles, delta_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let _ = kernels::fused_add_rms_norm_inplace_with_offset(
                    *delta,
                    *residual,
                    w.weight,
                    w.eps,
                    offset,
                    ctx.device.compute_stream,
                );
            },
            Instruction::ScalarOffsetRmsNorm(in_slot, out_slot, layer, offset, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let out = kernels::rms_norm_with_offset(
                    *v,
                    w.weight,
                    w.eps,
                    offset,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassFusedRmsNormGemm(
                in_slot,
                out_slot,
                layer,
                norm_wf,
                gemm_wf,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let nw = (norm_wf)(ctx.wm, layer);
                let gw = (gemm_wf)(ctx.wm, layer);
                assert_weight_shape(
                    "CutlassFusedRmsNormGemm",
                    gw.dense_weight(),
                    n,
                    k,
                    tp_active(ctx),
                );
                let normed = kernels::rms_norm(
                    *v,
                    nw.weight,
                    nw.eps,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = cutlass::cutlass_gemm(
                    normed.as_gpu_tensor(),
                    gw.dense_weight(),
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassFusedLayerNormGemm(
                in_slot,
                out_slot,
                layer,
                norm_wf,
                gemm_wf,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let nw = (norm_wf)(ctx.wm, layer);
                let gw = (gemm_wf)(ctx.wm, layer);
                assert_weight_shape(
                    "CutlassFusedLayerNormGemm",
                    gw.dense_weight(),
                    n,
                    k,
                    tp_active(ctx),
                );
                let normed = kernels::cohere_layer_norm(
                    *v,
                    nw.weight,
                    nw.eps,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = cutlass::cutlass_gemm(
                    normed.as_gpu_tensor(),
                    gw.dense_weight(),
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassFusedAddRmsNormGemm(
                delta_slot,
                residual_slot,
                out_slot,
                layer,
                norm_wf,
                gemm_wf,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let delta = tile_ref(ctx.tiles, delta_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                let nw = (norm_wf)(ctx.wm, layer);
                let gw = (gemm_wf)(ctx.wm, layer);
                assert_weight_shape(
                    "CutlassFusedAddRmsNormGemm",
                    gw.dense_weight(),
                    n,
                    k,
                    tp_active(ctx),
                );
                // After this kernel: delta buffer = normed output;
                // residual buffer = updated residual. The residual
                // alias is set up by the codegen prelude (TileEntry::View
                // on (add_id, 0) → residual upstream).
                let (normed_view, _) = kernels::fused_add_rms_norm_inplace(
                    *delta,
                    *residual,
                    nw.weight,
                    nw.eps,
                    ctx.device.compute_stream,
                );
                let out = cutlass::cutlass_gemm(
                    normed_view,
                    gw.dense_weight(),
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Gemm(in_slot, out_slot, layer, weight_fn, n, k) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                assert_weight_shape("Gemm", w.dense_weight(), n, k, tp_active(ctx));
                let out = ctx
                    .device
                    .cublas
                    .gemm(*v, w.dense_weight(), &mut ctx.device.caching);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::FusedCublasGemmAdd(in_slot, residual_slot, layer, weight_fn, n, k) => unsafe {
                // cuBLAS gemm(activation, weight) → delta; then
                // add_inplace folds delta into the residual buffer.
                // The residual upstream's OwnedTensor is aliased to
                // the Add tile's slot via the codegen prelude — same
                // as CutlassGemmAdd.
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                assert_weight_shape("FusedCublasGemmAdd", w.dense_weight(), n, k, tp_active(ctx));
                let delta = ctx
                    .device
                    .cublas
                    .gemm(*v, w.dense_weight(), &mut ctx.device.caching);
                kernels::add_inplace(*residual, delta.as_gpu_tensor(), ctx.device.compute_stream);
            },
            Instruction::FusedGemmBias(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                debug_assert!(
                    w.dense_bias().is_some(),
                    "FusedGemmBias: DSL `bias_add` claimed but \
                     LinearLayer has no bias — check safetensors path"
                );
                let out = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::FusedGateUpSiluMul(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let gate_up = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::silu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::FusedGateUpGeluMul(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let gate_up = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::gelu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::FusedQkvRopeCache(
                in_slot,
                out_slot,
                layer,
                weight_fn,
                cos_sin_fn,
                biased,
                interleaved,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                if biased {
                    debug_assert!(
                        w.dense_bias().is_some(),
                        "FusedQkvRopeCache: DSL `bias_add` on QKV claimed but \
                         packed LinearLayer has no bias — check safetensors path"
                    );
                }
                let qkv_packed = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                let out = if interleaved {
                    kernels::fused_qkv_interleaved_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else if ctx.fwd.kv_cache.is_fp8() {
                    kernels::fused_qkv_rope_cache_fp8(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        ctx.fwd.kv_cache.k_scale_ptr(layer as usize),
                        ctx.fwd.kv_cache.v_scale_ptr(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else {
                    kernels::fused_qkv_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::FusedQkvQkNormRopeCache(
                in_slot,
                out_slot,
                layer,
                q_weight_fn,
                k_weight_fn,
                v_weight_fn,
                q_norm_fn,
                k_norm_fn,
                cos_sin_fn,
                q_offset,
                k_offset,
            ) => {
                let layer = ctx.layer_offset + layer;
                let mut q_out = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let qw = (q_weight_fn)(ctx.wm, layer);
                    let kw = (k_weight_fn)(ctx.wm, layer);
                    let vw = (v_weight_fn)(ctx.wm, layer);
                    let qnorm = (q_norm_fn)(ctx.wm, layer);
                    let knorm = (k_norm_fn)(ctx.wm, layer);
                    let nt = (*ctx.fwd.input_ids).dim(0);
                    let q = qw.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let k = kw.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let v_proj = vw.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let q_view = (*q).reshape(&[nt, W::NUM_Q_HEADS as usize, W::HEAD_DIM as usize]);
                    let k_view =
                        (*k).reshape(&[nt, W::NUM_KV_HEADS as usize, W::HEAD_DIM as usize]);
                    let v_view =
                        (*v_proj).reshape(&[nt, W::NUM_KV_HEADS as usize, W::HEAD_DIM as usize]);
                    let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                    kernels::qk_norm_rope_inplace(
                        q_view,
                        k_view,
                        qnorm.weight,
                        knorm.weight,
                        cos_sin,
                        *ctx.fwd.positions,
                        W::NUM_Q_HEADS as usize,
                        W::NUM_KV_HEADS as usize,
                        W::HEAD_DIM as usize,
                        qnorm.eps,
                        q_offset,
                        k_offset,
                        ctx.device.compute_stream,
                    );
                    kernels::reshape_and_cache(
                        k_view,
                        v_view,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        *ctx.fwd.slot_mapping,
                        ctx.fwd.kv_cache.block_size,
                        ctx.device.compute_stream,
                    );
                    q
                };
                unsafe {
                    let nt = (*q_out).dim(0);
                    let dt = (*q_out).dtype();
                    q_out.reshape(&[nt, W::NUM_Q_HEADS as usize, W::HEAD_DIM as usize], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(q_out));
            }
            Instruction::FusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                weight_fn,
                cos_sin_fn,
                biased,
                interleaved,
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = (weight_fn)(ctx.wm, layer);
                    if biased {
                        debug_assert!(
                            w.dense_bias().is_some(),
                            "FusedQkvRopePrefill: DSL `bias_add` on QKV claimed but \
                             packed LinearLayer has no bias — check safetensors path"
                        );
                    }
                    let qkv_packed = w.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                    if interleaved {
                        kernels::fused_qkv_interleaved_rope(
                            *qkv_packed,
                            *ctx.fwd.positions,
                            cos_sin,
                            W::Q_SIZE,
                            W::KV_SIZE,
                            W::NUM_Q_HEADS as usize,
                            W::NUM_KV_HEADS as usize,
                            W::HEAD_DIM as usize,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        )
                    } else {
                        kernels::fused_qkv_rope(
                            *qkv_packed,
                            *ctx.fwd.positions,
                            cos_sin,
                            W::Q_SIZE,
                            W::KV_SIZE,
                            W::NUM_Q_HEADS as usize,
                            W::NUM_KV_HEADS as usize,
                            W::HEAD_DIM as usize,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        )
                    }
                };
                unsafe {
                    ah::write_kv_cache(
                        k.view(),
                        v_out.view(),
                        ctx.fwd.slot_mapping,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.compute_stream,
                    );
                }
                ctx.tiles[q_out_slot as usize] = Some(TileEntry::Owned(q));
                ctx.tiles[k_out_slot as usize] = Some(TileEntry::Owned(k));
                ctx.tiles[v_out_slot as usize] = Some(TileEntry::Owned(v_out));
            }
            Instruction::AttentionViaCache(in_slot, out_slot, layer, cos_sin_fn, interleaved) => {
                let layer = ctx.layer_offset + layer;
                let mut out = unsafe {
                    let q = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                    let has_spans = !ctx.fwd.kv_cache.block_unrotated_gpu().is_null();
                    let (cos_sin_ptr, rotary_dim) = if has_spans {
                        (cos_sin.raw_ptr() as *const u8, cos_sin.dim(1))
                    } else {
                        (::std::ptr::null::<u8>(), 0)
                    };
                    ah::attention_decode_from_cache(
                        q,
                        ctx.fwd.cu_seqlens_q,
                        ctx.fwd.seqused_k,
                        ctx.fwd.block_table,
                        ctx.fwd.max_seqlen_q,
                        ctx.fwd.max_seqlen_k,
                        W::ATTN_SCALE,
                        W::ATTN_SOFTCAP,
                        -1,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.num_sm,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                        cos_sin_ptr,
                        rotary_dim,
                        interleaved,
                    )
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::AttentionPrefillContiguous(
                q_slot,
                k_slot,
                v_slot,
                out_slot,
                interleaved,
            ) => {
                let mut out = unsafe {
                    let q = tile_ref(ctx.tiles, q_slot).as_view(ctx.tiles);
                    let k = tile_ref(ctx.tiles, k_slot).as_view(ctx.tiles);
                    let v = tile_ref(ctx.tiles, v_slot).as_view(ctx.tiles);
                    kernels::flash_attn_contiguous(
                        *q,
                        *k,
                        *v,
                        *ctx.fwd.cu_seqlens_q,
                        *ctx.fwd.cu_seqlens_q,
                        ctx.fwd.max_seqlen_q,
                        ctx.fwd.max_seqlen_k,
                        W::ATTN_SCALE,
                        true,
                        W::ATTN_SOFTCAP,
                        -1,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                        ::std::ptr::null::<u8>(),
                        0,
                        interleaved,
                    )
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::SlidingAttentionViaCache(
                in_slot,
                out_slot,
                layer,
                cos_sin_fn,
                interleaved,
            ) => {
                let layer = ctx.layer_offset + layer;
                let mut out = unsafe {
                    let q = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                    let has_spans = !ctx.fwd.kv_cache.block_unrotated_gpu().is_null();
                    let (cos_sin_ptr, rotary_dim) = if has_spans {
                        (cos_sin.raw_ptr() as *const u8, cos_sin.dim(1))
                    } else {
                        (::std::ptr::null::<u8>(), 0)
                    };
                    ah::attention_decode_from_cache(
                        q,
                        ctx.fwd.cu_seqlens_q,
                        ctx.fwd.seqused_k,
                        ctx.fwd.block_table,
                        ctx.fwd.max_seqlen_q,
                        ctx.fwd.max_seqlen_k,
                        W::ATTN_SCALE,
                        W::ATTN_SOFTCAP,
                        W::SLIDING_WINDOW,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.num_sm,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                        cos_sin_ptr,
                        rotary_dim,
                        interleaved,
                    )
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::SlidingAttentionPrefillContiguous(
                q_slot,
                k_slot,
                v_slot,
                out_slot,
                interleaved,
            ) => {
                let mut out = unsafe {
                    let q = tile_ref(ctx.tiles, q_slot).as_view(ctx.tiles);
                    let k = tile_ref(ctx.tiles, k_slot).as_view(ctx.tiles);
                    let v = tile_ref(ctx.tiles, v_slot).as_view(ctx.tiles);
                    kernels::flash_attn_contiguous(
                        *q,
                        *k,
                        *v,
                        *ctx.fwd.cu_seqlens_q,
                        *ctx.fwd.cu_seqlens_q,
                        ctx.fwd.max_seqlen_q,
                        ctx.fwd.max_seqlen_k,
                        W::ATTN_SCALE,
                        true,
                        W::ATTN_SOFTCAP,
                        W::SLIDING_WINDOW,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                        ::std::ptr::null::<u8>(),
                        0,
                        interleaved,
                    )
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::FlashInferAttentionDecode(
                in_slot,
                out_slot,
                layer,
                cos_sin_fn,
                head_dim,
                use_logits_soft_cap,
            ) => {
                let layer = ctx.layer_offset + layer;
                let mut out = unsafe {
                    let q = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let fi_cfg = flashinfer::FlashInferConfig {
                        dtype: flashinfer::FiDType::Bf16,
                        head_dim,
                        use_logits_soft_cap,
                    };
                    let sk_bucket = ah::sk_bucket_for(ctx.fwd.max_seqlen_k);
                    let fi = ah::flashinfer_attention(
                        q,
                        ctx.fwd.cu_seqlens_q,
                        ctx.fwd.seqused_k,
                        ctx.fwd.block_table,
                        ctx.fwd.max_seqlen_q,
                        ctx.fwd.max_seqlen_k,
                        W::ATTN_SCALE,
                        W::ATTN_SOFTCAP,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.num_sm,
                        fi_cfg,
                        sk_bucket,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    match fi {
                        Some(t) => t,
                        None => {
                            let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                            let has_spans = !ctx.fwd.kv_cache.block_unrotated_gpu().is_null();
                            let (cos_sin_ptr, rotary_dim) = if has_spans {
                                (cos_sin.raw_ptr() as *const u8, cos_sin.dim(1))
                            } else {
                                (::std::ptr::null::<u8>(), 0)
                            };
                            ah::attention_decode_from_cache(
                                q,
                                ctx.fwd.cu_seqlens_q,
                                ctx.fwd.seqused_k,
                                ctx.fwd.block_table,
                                ctx.fwd.max_seqlen_q,
                                ctx.fwd.max_seqlen_k,
                                W::ATTN_SCALE,
                                W::ATTN_SOFTCAP,
                                -1,
                                ctx.fwd.kv_cache,
                                layer as usize,
                                ctx.device.num_sm,
                                &mut ctx.device.caching,
                                ctx.device.compute_stream,
                                cos_sin_ptr,
                                rotary_dim,
                                false,
                            )
                        }
                    }
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::FlashInferAttentionPrefill(
                q_slot,
                k_slot,
                v_slot,
                out_slot,
                layer,
                head_dim,
                use_logits_soft_cap,
            ) => {
                let layer = ctx.layer_offset + layer;
                let mut out = unsafe {
                    let q = tile_ref(ctx.tiles, q_slot).as_view(ctx.tiles);
                    let k = tile_ref(ctx.tiles, k_slot).as_view(ctx.tiles);
                    let v = tile_ref(ctx.tiles, v_slot).as_view(ctx.tiles);
                    let fi_cfg = flashinfer::FlashInferConfig {
                        dtype: flashinfer::FiDType::Bf16,
                        head_dim,
                        use_logits_soft_cap,
                    };
                    let sk_bucket = ah::sk_bucket_for(ctx.fwd.max_seqlen_k);
                    let fi = ah::flashinfer_attention(
                        q,
                        ctx.fwd.cu_seqlens_q,
                        ctx.fwd.seqused_k,
                        ctx.fwd.block_table,
                        ctx.fwd.max_seqlen_q,
                        ctx.fwd.max_seqlen_k,
                        W::ATTN_SCALE,
                        W::ATTN_SOFTCAP,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.num_sm,
                        fi_cfg,
                        sk_bucket,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    match fi {
                        Some(t) => t,
                        None => kernels::flash_attn_contiguous(
                            *q,
                            *k,
                            *v,
                            *ctx.fwd.cu_seqlens_q,
                            *ctx.fwd.cu_seqlens_q,
                            ctx.fwd.max_seqlen_q,
                            ctx.fwd.max_seqlen_k,
                            W::ATTN_SCALE,
                            true,
                            W::ATTN_SOFTCAP,
                            -1,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                            ::std::ptr::null::<u8>(),
                            0,
                            false,
                        ),
                    }
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::RopeAppend(
                q_slot,
                k_slot,
                v_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                cos_sin_fn,
                interleaved,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                let q_view = tile_ref(ctx.tiles, q_slot).as_view(ctx.tiles);
                let k_view = tile_ref(ctx.tiles, k_slot).as_view(ctx.tiles);
                let v_view = tile_ref(ctx.tiles, v_slot).as_view(ctx.tiles);
                if interleaved {
                    kernels::rotary_embedding_interleaved_inplace(
                        *q_view,
                        *k_view,
                        *ctx.fwd.positions,
                        cos_sin,
                        W::HEAD_DIM as usize,
                        ctx.device.compute_stream,
                    );
                } else {
                    kernels::rotary_embedding_inplace(
                        *q_view,
                        *k_view,
                        *ctx.fwd.positions,
                        cos_sin,
                        W::HEAD_DIM as usize,
                        ctx.device.compute_stream,
                    );
                }
                let nt = (*k_view).dim(0);
                let k_3d = k_view.reshape(&[nt, W::NUM_KV_HEADS as usize, W::HEAD_DIM as usize]);
                let v_3d = v_view.reshape(&[nt, W::NUM_KV_HEADS as usize, W::HEAD_DIM as usize]);
                kernels::reshape_and_cache(
                    *k_3d,
                    *v_3d,
                    *ctx.fwd.kv_cache.k_cache(layer as usize),
                    *ctx.fwd.kv_cache.v_cache(layer as usize),
                    *ctx.fwd.slot_mapping,
                    ctx.fwd.kv_cache.block_size,
                    ctx.device.compute_stream,
                );
                let nt_q = (*q_view).dim(0);
                let q_3d = q_view.reshape(&[nt_q, W::NUM_Q_HEADS as usize, W::HEAD_DIM as usize]);
                ctx.tiles[q_out_slot as usize] = Some(TileEntry::Reshaped {
                    ref_slot: q_slot,
                    tensor: *q_3d,
                });
                ctx.tiles[k_out_slot as usize] = Some(TileEntry::Reshaped {
                    ref_slot: k_slot,
                    tensor: *k_3d,
                });
                ctx.tiles[v_out_slot as usize] = Some(TileEntry::Reshaped {
                    ref_slot: v_slot,
                    tensor: *v_3d,
                });
            },
            Instruction::MlaSplit(in_slot, kv_latent_slot, k_pe_slot) => unsafe {
                let kv_a_tv = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let nt = (*kv_a_tv).dim(0);
                let dt = (*kv_a_tv).dtype();
                let kv_latent = ctx.device.caching.alloc_tensor(&[nt, W::KV_LORA_RANK], dt);
                let k_pe = ctx
                    .device
                    .caching
                    .alloc_tensor(&[nt, W::QK_ROPE_HEAD_DIM], dt);
                kernels::mla_split_kv_a(
                    *kv_a_tv,
                    *kv_latent.view(),
                    *k_pe.view(),
                    W::KV_LORA_RANK,
                    W::QK_ROPE_HEAD_DIM,
                    ctx.device.compute_stream,
                );
                ctx.tiles[kv_latent_slot as usize] = Some(TileEntry::Owned(kv_latent));
                ctx.tiles[k_pe_slot as usize] = Some(TileEntry::Owned(k_pe));
            },
            Instruction::MlaAttention(
                q_slot,
                kv_b_slot,
                k_pe_slot,
                out_slot,
                layer,
                cos_sin_fn,
            ) => {
                let layer = ctx.layer_offset + layer;
                let out = unsafe {
                    mla_attention_eval(ctx, q_slot, kv_b_slot, k_pe_slot, layer, cos_sin_fn)
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::DeepSeekMoe(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let out = w.forward(v, ctx.device);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::DeepSeekMoeFp8Block(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let out = w.forward(v, ctx.device);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::DeepSeekMoeGgml(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let out = w.forward(v, ctx.device);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassGemm(
                in_slot,
                out_slot,
                layer,
                weight_fn,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                assert_weight_shape("CutlassGemm", w.dense_weight(), n, k, tp_active(ctx));
                let out = cutlass::cutlass_gemm(
                    *v,
                    w.dense_weight(),
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassGemmSplitK(
                in_slot,
                out_slot,
                layer,
                weight_fn,
                tile_m,
                tile_n,
                stages,
                split_k,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                assert_weight_shape("CutlassGemmSplitK", w.dense_weight(), n, k, tp_active(ctx));
                let out = cutlass::cutlass_gemm_splitk(
                    *v,
                    w.dense_weight(),
                    cutlass::CutlassSplitKTile::new(tile_m, tile_n, stages, split_k),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassGemmAdd(
                in_slot,
                residual_slot,
                layer,
                weight_fn,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                assert_weight_shape("CutlassGemmAdd", w.dense_weight(), n, k, tp_active(ctx));
                cutlass::cutlass_gemm_add(
                    *v,
                    w.dense_weight(),
                    *residual,
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    ctx.device.compute_stream,
                );
            },
            Instruction::CutlassGemv(in_slot, out_slot, layer, weight_fn, n, k) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                assert_weight_shape("CutlassGemv", w.dense_weight(), n, k, tp_active(ctx));
                let out = cutlass::cutlass_gemv(
                    *v,
                    w.dense_weight(),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassFusedGemmBias(
                in_slot,
                out_slot,
                layer,
                weight_fn,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                assert_weight_shape(
                    "CutlassFusedGemmBias",
                    w.dense_weight(),
                    n,
                    k,
                    tp_active(ctx),
                );
                let bias = w.dense_bias().expect(
                    "CutlassFusedGemmBias: LinearLayer has no bias — check safetensors path",
                );
                let out = cutlass::cutlass_gemm_bias(
                    *v,
                    w.dense_weight(),
                    bias,
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassFusedGateUpSiluMul(
                in_slot,
                out_slot,
                layer,
                weight_fn,
                tile_m,
                tile_n,
                stages,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let packed = w.dense_weight();
                let gate_w = packed.narrow_dim0(0, W::INTERMEDIATE_SIZE);
                let up_w = packed.narrow_dim0(W::INTERMEDIATE_SIZE, W::INTERMEDIATE_SIZE);
                let up_out = cutlass::cutlass_gemm(
                    *v,
                    up_w,
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = cutlass::cutlass_gemm_silu_mul(
                    *v,
                    gate_w,
                    up_out,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassFusedGateUpGeluMul(
                in_slot,
                out_slot,
                layer,
                weight_fn,
                tile_m,
                tile_n,
                stages,
                packed_n,
                k,
            ) => unsafe {
                // Mirrors the cuBLAS-peer FusedGateUpGeluMul:
                //   1. ONE GEMM at packed (M, 2I, K) → [M, 2I] intermediate
                //   2. gelu_and_mul_fused over [M, 2I] → [M, I]
                // The GEMM here is a calibrated CUTLASS standalone tile
                // instead of cuBLAS; the elementwise step is identical.
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                assert_weight_shape(
                    "CutlassFusedGateUpGeluMul",
                    w.dense_weight(),
                    packed_n,
                    k,
                    tp_active(ctx),
                );
                let gate_up = cutlass::cutlass_gemm(
                    *v,
                    w.dense_weight(),
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::gelu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassFusedQkvRopeCache(
                in_slot,
                out_slot,
                layer,
                weight_fn,
                cos_sin_fn,
                interleaved,
                tile_m,
                tile_n,
                stages,
                packed_n,
                k,
            ) => unsafe {
                // Mirrors the cuBLAS-peer FusedQkvRopeCache: ONE GEMM at
                // packed (M, q+2*kv, K) → fused_qkv_rope_cache* writing
                // K/V to the paged cache and returning rotated Q. The
                // GEMM here is a calibrated CUTLASS standalone tile
                // instead of cuBLAS; the rope+cache step is identical.
                //
                // Non-biased only — claim is gated to `biased=false` in
                // CutlassFusedQkvRopeCacheImpl::matches; qwen2's biased
                // QKV stays on the cuBLAS peer until the bias-zoo CSV
                // gains shape-swept rows.
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                assert_weight_shape(
                    "CutlassFusedQkvRopeCache",
                    w.dense_weight(),
                    packed_n,
                    k,
                    tp_active(ctx),
                );
                let qkv_packed = cutlass::cutlass_gemm(
                    *v,
                    w.dense_weight(),
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                let out = if interleaved {
                    kernels::fused_qkv_interleaved_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else if ctx.fwd.kv_cache.is_fp8() {
                    kernels::fused_qkv_rope_cache_fp8(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        ctx.fwd.kv_cache.k_scale_ptr(layer as usize),
                        ctx.fwd.kv_cache.v_scale_ptr(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else {
                    kernels::fused_qkv_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassFusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                weight_fn,
                cos_sin_fn,
                interleaved,
                tile_m,
                tile_n,
                stages,
                packed_n,
                k_dim,
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k_tensor, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = (weight_fn)(ctx.wm, layer);
                    assert_weight_shape(
                        "CutlassFusedQkvRopePrefill",
                        w.dense_weight(),
                        packed_n,
                        k_dim,
                        tp_active(ctx),
                    );
                    let qkv_packed = cutlass::cutlass_gemm(
                        *view_in,
                        w.dense_weight(),
                        cutlass::CutlassTile::new(tile_m, tile_n, stages),
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                    if interleaved {
                        kernels::fused_qkv_interleaved_rope(
                            *qkv_packed,
                            *ctx.fwd.positions,
                            cos_sin,
                            W::Q_SIZE,
                            W::KV_SIZE,
                            W::NUM_Q_HEADS as usize,
                            W::NUM_KV_HEADS as usize,
                            W::HEAD_DIM as usize,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        )
                    } else {
                        kernels::fused_qkv_rope(
                            *qkv_packed,
                            *ctx.fwd.positions,
                            cos_sin,
                            W::Q_SIZE,
                            W::KV_SIZE,
                            W::NUM_Q_HEADS as usize,
                            W::NUM_KV_HEADS as usize,
                            W::HEAD_DIM as usize,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        )
                    }
                };
                unsafe {
                    ah::write_kv_cache(
                        k_tensor.view(),
                        v_out.view(),
                        ctx.fwd.slot_mapping,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.compute_stream,
                    );
                }
                ctx.tiles[q_out_slot as usize] = Some(TileEntry::Owned(q));
                ctx.tiles[k_out_slot as usize] = Some(TileEntry::Owned(k_tensor));
                ctx.tiles[v_out_slot as usize] = Some(TileEntry::Owned(v_out));
            }
            Instruction::MarlinGemm(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let out = w.forward(v, &mut ctx.device.caching, ctx.device.compute_stream);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::MarlinFusedGateUpSiluMul(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let gate_up = w.forward(v, &mut ctx.device.caching, ctx.device.compute_stream);
                let out = kernels::silu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::MarlinFusedGateUpGeluMul(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let gate_up = w.forward(v, &mut ctx.device.caching, ctx.device.compute_stream);
                let out = kernels::gelu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::MarlinFusedQkvRopeCache(
                in_slot,
                out_slot,
                layer,
                weight_fn,
                cos_sin_fn,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let qkv_packed = w.forward(v, &mut ctx.device.caching, ctx.device.compute_stream);
                let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                let out = if ctx.fwd.kv_cache.is_fp8() {
                    kernels::fused_qkv_rope_cache_fp8(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        ctx.fwd.kv_cache.k_scale_ptr(layer as usize),
                        ctx.fwd.kv_cache.v_scale_ptr(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else {
                    kernels::fused_qkv_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::MarlinFusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                weight_fn,
                cos_sin_fn,
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = (weight_fn)(ctx.wm, layer);
                    let qkv_packed =
                        w.forward(view_in, &mut ctx.device.caching, ctx.device.compute_stream);
                    let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                    kernels::fused_qkv_rope(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::NUM_KV_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                unsafe {
                    ah::write_kv_cache(
                        k.view(),
                        v_out.view(),
                        ctx.fwd.slot_mapping,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.compute_stream,
                    );
                }
                ctx.tiles[q_out_slot as usize] = Some(TileEntry::Owned(q));
                ctx.tiles[k_out_slot as usize] = Some(TileEntry::Owned(k));
                ctx.tiles[v_out_slot as usize] = Some(TileEntry::Owned(v_out));
            }
            Instruction::GgmlGemm(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let out = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::GgmlFusedGateUpSiluMul(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let gate_up = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::silu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::GgmlFusedGateUpGeluMul(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let gate_up = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::gelu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::GgmlFusedQkvRopeCache(
                in_slot,
                out_slot,
                layer,
                weight_fn,
                cos_sin_fn,
                interleaved,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let qkv_packed = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                let out = if interleaved {
                    kernels::fused_qkv_interleaved_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else if ctx.fwd.kv_cache.is_fp8() {
                    kernels::fused_qkv_rope_cache_fp8(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        ctx.fwd.kv_cache.k_scale_ptr(layer as usize),
                        ctx.fwd.kv_cache.v_scale_ptr(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else {
                    kernels::fused_qkv_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::GgmlFusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                weight_fn,
                cos_sin_fn,
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = (weight_fn)(ctx.wm, layer);
                    let qkv_packed = w.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                    kernels::fused_qkv_rope(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::NUM_KV_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                unsafe {
                    ah::write_kv_cache(
                        k.view(),
                        v_out.view(),
                        ctx.fwd.slot_mapping,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.compute_stream,
                    );
                }
                ctx.tiles[q_out_slot as usize] = Some(TileEntry::Owned(q));
                ctx.tiles[k_out_slot as usize] = Some(TileEntry::Owned(k));
                ctx.tiles[v_out_slot as usize] = Some(TileEntry::Owned(v_out));
            }
            Instruction::Bnb4Gemm(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let out = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Bnb4FusedGateUpSiluMul(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let gate_up = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::silu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Bnb4FusedGateUpGeluMul(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let gate_up = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::gelu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Bnb4FusedQkvRopeCache(in_slot, out_slot, layer, weight_fn, cos_sin_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let qkv_packed = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                let out = if ctx.fwd.kv_cache.is_fp8() {
                    kernels::fused_qkv_rope_cache_fp8(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        ctx.fwd.kv_cache.k_scale_ptr(layer as usize),
                        ctx.fwd.kv_cache.v_scale_ptr(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else {
                    kernels::fused_qkv_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Bnb4FusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                weight_fn,
                cos_sin_fn,
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = (weight_fn)(ctx.wm, layer);
                    let qkv_packed = w.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                    kernels::fused_qkv_rope(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::NUM_KV_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                unsafe {
                    ah::write_kv_cache(
                        k.view(),
                        v_out.view(),
                        ctx.fwd.slot_mapping,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.compute_stream,
                    );
                }
                ctx.tiles[q_out_slot as usize] = Some(TileEntry::Owned(q));
                ctx.tiles[k_out_slot as usize] = Some(TileEntry::Owned(k));
                ctx.tiles[v_out_slot as usize] = Some(TileEntry::Owned(v_out));
            }
            Instruction::Fp8Gemm(in_slot, out_slot, layer, weight_fn)
            | Instruction::Fp8FusedGemmBias(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let out = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Fp8FusedGateUpSiluMul(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let gate_up = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::silu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Fp8FusedGateUpGeluMul(in_slot, out_slot, layer, weight_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let gate_up = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::gelu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Fp8FusedQkvRopeCache(in_slot, out_slot, layer, weight_fn, cos_sin_fn) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = (weight_fn)(ctx.wm, layer);
                let qkv_packed = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                let out = if ctx.fwd.kv_cache.is_fp8() {
                    kernels::fused_qkv_rope_cache_fp8(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        ctx.fwd.kv_cache.k_scale_ptr(layer as usize),
                        ctx.fwd.kv_cache.v_scale_ptr(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else {
                    kernels::fused_qkv_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Fp8FusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                weight_fn,
                cos_sin_fn,
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = (weight_fn)(ctx.wm, layer);
                    let qkv_packed = w.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let cos_sin = (cos_sin_fn)(ctx.wm, layer);
                    kernels::fused_qkv_rope(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::NUM_KV_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                unsafe {
                    ah::write_kv_cache(
                        k.view(),
                        v_out.view(),
                        ctx.fwd.slot_mapping,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.compute_stream,
                    );
                }
                ctx.tiles[q_out_slot as usize] = Some(TileEntry::Owned(q));
                ctx.tiles[k_out_slot as usize] = Some(TileEntry::Owned(k));
                ctx.tiles[v_out_slot as usize] = Some(TileEntry::Owned(v_out));
            }
            Instruction::Loop(_, _) => {
                unreachable!("Instruction::Loop should be handled by run(), not eval()");
            }
            Instruction::Alias(dst, src) => {
                ctx.tiles[dst as usize] = Some(view(src));
            }
            Instruction::Free(slot) => {
                ctx.tiles[slot as usize] = None;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn mla_attention_eval<W: CanonicalParams>(
    ctx: &mut InterpreterCtx<'_, W>,
    q_slot: u32,
    kv_b_slot: u32,
    k_pe_slot: u32,
    layer: u32,
    cos_sin_fn: CosSinFn<W>,
) -> OwnedTensor {
    unsafe {
        let q_tv = tile_ref(ctx.tiles, q_slot).as_view(ctx.tiles);
        let kv_b_tv = tile_ref(ctx.tiles, kv_b_slot).as_view(ctx.tiles);
        let k_pe_tv = tile_ref(ctx.tiles, k_pe_slot).as_view(ctx.tiles);
        let cos_sin = (cos_sin_fn)(ctx.wm, layer);
        let nt = (*q_tv).dim(0);
        let dt = (*q_tv).dtype();

        let q_pe = ctx
            .device
            .caching
            .alloc_tensor(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_ROPE_HEAD_DIM], dt);
        kernels::mla_extract_q_pe(
            *q_tv,
            *q_pe
                .view()
                .reshape(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_ROPE_HEAD_DIM]),
            W::NUM_Q_HEADS as usize,
            W::QK_HEAD_DIM,
            W::QK_NOPE_HEAD_DIM,
            W::QK_ROPE_HEAD_DIM,
            ctx.device.compute_stream,
        );

        kernels::rotary_embedding_interleaved_inplace(
            *q_pe
                .view()
                .reshape(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_ROPE_HEAD_DIM]),
            *k_pe_tv,
            *ctx.fwd.positions,
            cos_sin,
            W::QK_ROPE_HEAD_DIM,
            ctx.device.compute_stream,
        );

        kernels::mla_write_q_pe(
            *q_pe
                .view()
                .reshape(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_ROPE_HEAD_DIM]),
            *q_tv,
            W::NUM_Q_HEADS as usize,
            W::QK_HEAD_DIM,
            W::QK_NOPE_HEAD_DIM,
            W::QK_ROPE_HEAD_DIM,
            ctx.device.compute_stream,
        );
        drop(q_pe);

        let k = ctx
            .device
            .caching
            .alloc_tensor(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_HEAD_DIM], dt);
        kernels::mla_assemble_k(
            *kv_b_tv,
            *k_pe_tv,
            *k.view()
                .reshape(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_HEAD_DIM]),
            W::NUM_Q_HEADS as usize,
            W::QK_NOPE_HEAD_DIM,
            W::QK_ROPE_HEAD_DIM,
            W::V_HEAD_DIM,
            W::QK_HEAD_DIM,
            ctx.device.compute_stream,
        );

        let v = ctx
            .device
            .caching
            .alloc_tensor(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_HEAD_DIM], dt);
        ferrite_cuda_core::driver::memset_d8(
            (*v.view()).raw_ptr(),
            0,
            (*v.view()).size_bytes(),
            ctx.device.compute_stream,
        )
        .expect("MLA: memset V");
        kernels::mla_assemble_v(
            *kv_b_tv,
            *v.view()
                .reshape(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_HEAD_DIM]),
            W::NUM_Q_HEADS as usize,
            W::QK_NOPE_HEAD_DIM,
            W::V_HEAD_DIM,
            W::QK_HEAD_DIM,
            ctx.device.compute_stream,
        );

        let k_tv = k.view();
        let k_3d = k_tv.reshape(&[nt, W::NUM_Q_HEADS as usize, W::QK_HEAD_DIM]);
        let v_tv = v.view();
        let v_3d = v_tv.reshape(&[nt, W::NUM_Q_HEADS as usize, W::QK_HEAD_DIM]);
        ah::write_kv_cache(
            k_3d,
            v_3d,
            ctx.fwd.slot_mapping,
            ctx.fwd.kv_cache,
            layer as usize,
            ctx.device.compute_stream,
        );

        let q_3d = q_tv.reshape(&[nt, W::NUM_Q_HEADS as usize, W::QK_HEAD_DIM]);
        let attn = ah::attention_standard(
            q_3d,
            k_3d,
            v_3d,
            ctx.fwd.cu_seqlens_q,
            ctx.fwd.seqused_k,
            ctx.fwd.block_table,
            ctx.fwd.max_seqlen_q,
            ctx.fwd.max_seqlen_k,
            W::MLA_ATTN_SCALE,
            ctx.fwd.kv_cache,
            layer as usize,
            ctx.device.num_sm,
            &mut ctx.device.caching,
            ctx.device.compute_stream,
            ::std::ptr::null(),
            0,
            false,
        );
        drop(k);
        drop(v);

        let sliced = ctx
            .device
            .caching
            .alloc_tensor(&[nt, (W::NUM_Q_HEADS as usize) * W::V_HEAD_DIM], dt);
        let attn_tv = attn.view();
        let attn_flat = attn_tv.reshape(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_HEAD_DIM]);
        kernels::mla_slice_attn_output(
            *attn_flat,
            *sliced.view(),
            W::NUM_Q_HEADS as usize,
            W::QK_HEAD_DIM,
            W::V_HEAD_DIM,
            ctx.device.compute_stream,
        );
        drop(attn);
        sliced
    }
}

/// Walk one slice in-place against ctx. `Instruction::Loop(count,
/// body_len)` re-runs the next `body_len` instructions `count`
/// times with `ctx.layer_offset` set to the iter index.
unsafe fn run_slice<W: CanonicalParams>(
    instructions: &[Instruction<W>],
    ctx: &mut InterpreterCtx<'_, W>,
) {
    let mut i = 0usize;
    while i < instructions.len() {
        match instructions[i] {
            Instruction::Loop(count, body_len) => {
                let body_start = i + 1;
                let body_end = body_start + body_len as usize;
                let body = &instructions[body_start..body_end];
                for l in 0..count {
                    ctx.layer_offset = l;
                    for instr in body {
                        unsafe {
                            instr.eval(ctx);
                        }
                    }
                }
                ctx.layer_offset = 0;
                i = body_end;
            }
            instr => {
                unsafe {
                    instr.eval(ctx);
                }
                i += 1;
            }
        }
    }
}

/// Run backbone followed by lm_head against one tile table.
///
/// # Safety
/// Both slices well-formed; tile slot indices in range; weight
/// accessor fns produce live GPU memory.
pub unsafe fn run<W: CanonicalParams>(
    backbone: &[Instruction<W>],
    lm_head: &[Instruction<W>],
    wm: &W,
    fwd: &ForwardCtx,
    device: &mut GpuDevice,
    num_slots: u32,
    terminal_slot: u32,
) -> OwnedTensor {
    let mut tiles: Vec<Option<TileEntry>> = (0..num_slots).map(|_| None).collect();
    let mut ctx = InterpreterCtx {
        wm,
        tiles: &mut tiles,
        fwd,
        device,
        layer_offset: 0,
    };
    unsafe {
        run_slice(backbone, &mut ctx);
        run_slice(lm_head, &mut ctx);
    }
    take_owned(&mut tiles, terminal_slot)
}

/// Backbone-only run: returns a memcpy'd backbone tile so it
/// outlives the per-call tile table.
///
/// # Safety
/// Same as [`run`].
pub unsafe fn run_backbone<W: CanonicalParams>(
    backbone: &[Instruction<W>],
    wm: &W,
    fwd: &ForwardCtx,
    device: &mut GpuDevice,
    num_slots: u32,
    backbone_slot: u32,
) -> OwnedTensor {
    let mut tiles: Vec<Option<TileEntry>> = (0..num_slots).map(|_| None).collect();
    let mut ctx = InterpreterCtx {
        wm,
        tiles: &mut tiles,
        fwd,
        device,
        layer_offset: 0,
    };
    unsafe {
        run_slice(backbone, &mut ctx);
    }
    let bb_view = unsafe { tile_ref(&tiles, backbone_slot).as_view(&tiles) };
    let bb_shape_u32: &[u32] = bb_view.shape();
    let bb_shape: Vec<usize> = bb_shape_u32.iter().map(|&d| d as usize).collect();
    let bb_out = device.caching.alloc_tensor(&bb_shape, bb_view.dtype());
    unsafe {
        ferrite_cuda_core::driver::memcpy_dtod_async(
            bb_out.raw_ptr(),
            bb_view.raw_ptr() as *const u8,
            bb_view.size_bytes(),
            device.compute_stream,
        )
        .expect("run_backbone: DtoD memcpy of output");
    }
    bb_out
}
