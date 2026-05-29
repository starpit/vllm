// SPDX-License-Identifier: Apache-2.0
//! Universal interpreter instruction set.
//!
//! `Instruction` is a closed enum over every kernel-call shape any
//! solver-picked `Implementation` produces. The match in
//! `Instruction::eval` lives ONCE in this module — not per canonical,
//! not per arch. Each arm calls existing kernels in `ferrite-kernels`
//! directly. Per-canonical specialization flows through method-level
//! generics:
//!
//! 1. `eval<W: CanonicalParams>` reads model constants via
//!    `W::HEAD_DIM` etc. and weight tensors via the
//!    [`WeightAccessors`] trait — `ctx.wm.<kind>_at(bucket, op_idx,
//!    slot, layer)`. The proc-macro emits a per-arch
//!    `WeightAccessors` impl whose body matches on
//!    `(bucket, op_idx, slot)` to resolve the named field on
//!    `Weights`. The variant itself carries no weight info.
//! 2. [`CanonicalParams`] supertype — trait implemented per
//!    canonical with associated `const`s for model-wide values
//!    (`HEAD_DIM`, `INTERMEDIATE_SIZE`, `ATTN_SCALE`, …).
//!
//! Per canonical the macro emits: a `Weights` struct,
//! `impl CanonicalParams for Weights { … }`,
//! `impl WeightAccessors for Weights { … }`, static
//! `&[Instruction]` slices for backbone + lm_head per bucket, and a
//! 1-line forward shim that delegates to [`run`].

// File is ungated at the module level so the proc-macro can use
// `Instruction` without a backend feature. The `Instruction` enum
// + `CanonicalParams` / `WeightAccessors` traits compile under any
// feature combination — only `GpuTensor` (always available) plus the
// layer struct *type names* (also always available; methods are
// cuda-gated inside `ferrite-kernels/src/layers.rs`) are referenced.
//
// `MAX_DIMS` is needed by the `Instruction::Reshape` variant which is
// ungated, so this import stays at the module level.
use ferrite_cuda_core::tensor::MAX_DIMS;
// `GpuTensor`, `AffineQuantEmbedding`, and the `ferrite_kernels` layer
// structs are referenced exclusively through fully-qualified paths
// (`::ferrite_cuda_core::tensor::GpuTensor`,
// `::ferrite_kernels::layers::LinearLayer`, etc.) in the
// `WeightAccessors` trait return types and the cuda eval body, so no
// `use` import is needed here. The `Instruction` enum itself is ungated
// and references only `u32` / scalar fields after the lift.
//
// Cuda-only imports (eval body, runtime entry points) — gated
// individually below so the metal-feature / no-backend builds only
// pull in the cross-backend pieces above.
#[cfg(feature = "cuda")]
use crate::ForwardCtx;
#[cfg(feature = "cuda")]
use crate::tile_table::{TileEntry, take_owned, tile_ref, view};
#[cfg(feature = "cuda")]
use ferrite_cuda_core::alloc::OwnedTensor;
#[cfg(feature = "cuda")]
use ferrite_cuda_core::device::GpuDevice;
#[cfg(feature = "cuda")]
use ferrite_kernels::attention_helpers as ah;
#[cfg(feature = "cuda")]
use ferrite_kernels::cutlass;
#[cfg(feature = "cuda")]
use ferrite_kernels::flashinfer;
#[cfg(feature = "cuda")]
use ferrite_kernels::kernels;

/// Per-canonical model parameters. Implemented by each canonical's
/// `Weights` so the universal `Instruction::eval` body can read
/// model constants without storing them on every variant instance.
/// Defaults to 0 / 0.0 / -1 for fields the canonical doesn't use.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub trait CanonicalParams: WeightAccessors {
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
    /// MRoPE (Qwen2-VL / Qwen2.5-VL) section split `[T, H, W]` (rotary
    /// pairs assigned to time / height / width axes; sum equals
    /// `head_dim/2`). `None` for every text-only arch — the rope kernel
    /// takes the legacy 1D-positions fast path. `Some([a,b,c])` selects
    /// the MRoPE path: kernel reads three position values per token
    /// (positions tensor shape `[3, n_tokens]`) and dispatches each
    /// rotary pair through the section that owns it. Default `None` so
    /// existing arches need no override; Qwen2-VL sets it via the
    /// proc-macro emit path. Plan: `~/.claude/plans/distributed-mapping-map.md`.
    const MROPE_SECTION: Option<[u32; 3]> = None;
    /// Vision-tower attention head count. Vision encoders run plain
    /// MHA (`num_kv_heads == num_heads`); only one head dim is needed.
    /// Default 0 for text-only arches that never produce
    /// `OpKind::VarlenAttention` / `OpKind::VisionRope` tiles, so the
    /// matching `Instruction` variants stay registered but unreachable.
    /// Set by `#[vision_forward]` from the vision config.
    const VISION_NUM_HEADS: u32 = 0;

    /// Metal-only: list of compiler-synthesized kernel sources (per
    /// `ferrite-forward-macro::fuse_pass`). Each entry is
    /// `(symbol_name, precompiled .metallib bytes)`. The proc-macro
    /// AOT-compiles synthesized MSL via `xcrun metal -c` +
    /// `xcrun metallib` at macro-expansion time and embeds the
    /// resulting bytes as `&'static [u8]`. The MetalWorkerPool
    /// registers each via
    /// `SpecializedPipelineCache::register_metallib_library`
    /// (`newLibraryWithData`) — same path used by every hand-written
    /// shader, NOT `newLibraryWithSource`. Default empty — the macro
    /// overrides this per Metal arch with the actual synthesized
    /// metallibs from the FUF analysis.
    fn synthesized_kernel_metallibs() -> &'static [(&'static str, &'static [u8])] {
        &[]
    }
    /// Vision-tower attention head dimension. Same defaults / set-by
    /// rule as [`Self::VISION_NUM_HEADS`].
    const VISION_HEAD_DIM: u32 = 0;
    /// `vision_num_heads * vision_head_dim` — the rank-2 last-dim of
    /// q/k/v at the FUF level (the kernel internally reshapes to
    /// rank-3). Same defaults / set-by rule as
    /// [`Self::VISION_NUM_HEADS`].
    const VISION_Q_SIZE: usize = 0;
    /// Vision-tower softmax scale: `1 / sqrt(vision_head_dim)`. Same
    /// defaults / set-by rule as [`Self::VISION_NUM_HEADS`].
    const VISION_ATTN_SCALE: f32 = 0.0;
    /// Patch-grid side length (square): `vision_image_size / vision_patch_size`.
    /// Used by `Instruction::AvgPool2d` to convert flat row index
    /// `[L = ph * ph]` → `(row, col)` and walk the k² source cells per
    /// output. Defaults to 0 for arches that never emit `OpKind::AvgPool2d`.
    const VISION_PATCH_GRID_SIDE: u32 = 0;
    /// Stride / kernel size of the post-encoder average pool (Gemma3-MM
    /// projector: k=4 over a 64×64 patch grid → 16×16 = 256 tokens). The
    /// pool is non-overlapping, so stride == kernel. Defaults to 0.
    const VISION_POOL_KERNEL: u32 = 0;

    /// RmsNorm epsilon — read from `rms_norm_eps` in the model
    /// config at macro-expand time. The metal `rmsnorm_*_specialized`
    /// kernels consume this via `[[function_constant]]` baked into
    /// the compiled pipeline; cuda's interpreter still reads it
    /// from `RmsNorm.eps` on the loaded layer struct (same value,
    /// same source). Default is the value Llama / Qwen / Phi
    /// canonically use; per-canonical macro impls override.
    const RMS_NORM_EPS: f32 = 1e-5;

    /// Paged-KV-cache block stride (the `block_size` function
    /// constant `attention_via_cache_*_specialized` and
    /// `rope_append_*_specialized` consume). Backend-fixed at 16
    /// (vLLM's default); per-canonical override only if a model
    /// chooses a different paging size.
    const BLOCK_SIZE: u32 = 16;

    /// Block-table row stride (in u32s), equal to
    /// `ceil(MAX_SEQ_LEN / BLOCK_SIZE)`. Baked into
    /// `attention_via_cache_*_specialized` so the kernel can index
    /// `block_table[seq * MAX_BLOCKS_PER_SEQ + logical_block]`
    /// without a runtime divide. Default sized for ~2k tokens; per-
    /// canonical macro impls override for longer-context models.
    const MAX_BLOCKS_PER_SEQ: u32 = 128;

    /// Q-axis tile size for the cuda contiguous-prefill kernel — the
    /// kernel processes this many query tokens per threadgroup.
    /// Metal post-Phase B always emits `Instruction::AttentionPrefillPaged`
    /// (1 Q per TG via `sdpa_vector` port) so this constant is
    /// cuda-only; backend-fixed and tuning requires kernel co-evolution.
    const PREFILL_TILE_Q: u32 = 16;

    /// Partial-rope rotation dim — for models where only the first
    /// `ROT_DIM` of `HEAD_DIM` get rotary applied (Qwen2-VL, GPT-J).
    /// Default equals `HEAD_DIM` (full rope, the common case).
    const ROT_DIM: u32 = Self::HEAD_DIM;

    /// Element dtype the metal backend should run this canonical in.
    /// Picks between the `_f16_specialized` / `_bf16_specialized`
    /// shader symbols and matching MPS GEMM data type. Default
    /// `MetalDtype::Bf16` matches every modern HF Llama / Qwen /
    /// Phi / Mistral checkpoint (`torch_dtype: bfloat16` on disk)
    /// and the cuda backend's native dtype. Per-canonical macro
    /// overrides set this from the model config's `torch_dtype`.
    ///
    /// Pre-bf16-rollout default was `F16`; that path lost exponent
    /// range over deep layer chains and produced incoherent outputs
    /// on Llama-3.x. The default is `Bf16` now; arches that ship
    /// fp16 on disk (rare) override.
    #[cfg(feature = "metal")]
    const METAL_DTYPE: crate::interpreter::metal::MetalDtype =
        crate::interpreter::metal::MetalDtype::Bf16;

    /// `hidden_size` (residual-stream width) — distinct from
    /// `Q_SIZE = num_q_heads * head_dim`. On Llama / Qwen2.5 these
    /// happen to be equal because each Q head is `hidden_size /
    /// num_heads`. On Qwen3 (and any GQA / different head_dim
    /// arch) they differ: e.g. Qwen3-30B-A3B has hidden=2048,
    /// num_q=32, head_dim=128 → Q_SIZE=4096. The metal interpreter
    /// previously baked `W::Q_SIZE` as `AFFINE_EMBED_HIDDEN_SIZE`
    /// and `RMSNORM_HIDDEN_SIZE`, which broke Qwen3 silently. The
    /// macro emits this const from `model.bounds["hidden_size"]`.
    /// Default value is invalid (0) so unset arches surface as
    /// compile-time-detectable bad values rather than silent
    /// corruption.
    const HIDDEN_SIZE: usize = 0;

    /// On-disk storage dtype for `*.scales` / `*.biases` tensors on
    /// mlx-affine-b4 checkpoints. The affine quant Metal kernels
    /// (`affine_qmv`, `affine_qmm_t`, `affine_qvm`, `affine_gather_qmv`,
    /// `affine_embed`) read these as `T_scale` and cast to `T_act` in-
    /// register. MLX templates `T_scale ∈ {half, bfloat16_t}` natively
    /// (`mlx/.../quantized.h INSTANTIATE_QUANTIZED_FUNCTIONS`);
    /// ferrite-metal mirrors that surface, picking the right symbol
    /// based on this const.
    ///
    /// `mlx-community` convention varies by arch family (probed across
    /// the cached HF snapshots): Llama-3.x / Qwen2.5 / SmolLM → F16,
    /// Qwen3 family → BF16. Default `F16` matches the
    /// majority-of-checkpoints convention; per-arch macro overrides
    /// flip it to BF16 (Qwen3 family). Sourced from the manifest's
    /// `scale_dtype` field on the matched `mlx-affine-*` preset.
    #[cfg(feature = "metal")]
    const SCALE_DTYPE: crate::interpreter::metal::ScaleDtype =
        crate::interpreter::metal::ScaleDtype::F16;

    // ── Backend-capability flags ────────────────────────────────
    //
    // Each `HAS_*` flag tells [`crate::backend_compat::BackendCompat`]
    // whether the arch's DSL emits a tile that requires a particular
    // backend-side `Impl`. The macro derives these by walking the
    // classified `Program` at expansion time — they're never
    // hand-set per canonical, and never read at runtime.
    //
    // Flags exist to make a missing-Impl-on-backend bug a compile
    // error instead of silent garbage output. See
    // `FERRITE_METAL_TYPE_SAFETY_PLAN.md` and
    // `crates/ferrite-forward/src/backend_compat.rs`.

    /// True iff the arch's DSL body emits `bias_add(...)` tile(s)
    /// (Qwen-family QKV biases, ModernBERT-style LayerNorm post-add,
    /// etc.). Set by the macro via DSL inspection; never written
    /// by hand.
    ///
    /// On CUDA the `(Gemm, BiasAdd)` and `(AffineQmm, BiasAdd)`
    /// fusions absorb every shipping pattern; on Metal the matching
    /// fused Impls (`MetalBiasAddImpl::matches() → None`,
    /// `FusedGemmBiasImpl` Affine-storage gate, missing
    /// `FusedAffineQkvRopeCacheWithBias`) leave bias_add tiles
    /// unclaimed, so a `true` value blocks
    /// `BackendCompat<Metal>` at compile time.
    const HAS_BIAS_ADD: bool = false;

    /// True iff the arch's DSL emits `OpKind::Moe` (router softmax,
    /// top-k, experts gather, and SwitchGLU bundled into one tile).
    /// MoE-on-Metal isn't landed yet — the router prereqs (softmax,
    /// argpartition, take_along_axis) ported in
    /// `project_metal_moe_router_kernels`, but `gather_qmm_rhs` and
    /// the pure-ICB SwitchGLU decomposition
    /// (`project_metal_moe_switchglu`) are open. MoE arches block
    /// `BackendCompat<Metal>` until those land.
    const HAS_MOE: bool = false;
}

/// Per-arch weight-accessor dispatch trait.
///
/// Each per-`L` method matches on `(bucket, op_idx)` against the
/// closed set of weight-consuming positions the proc-macro emitted
/// for this arch. Replaces the `WtFn<W, L>` / `CosSinFn<W>` fn-
/// pointer fields previously stored on `Instruction<W>` variants
/// AND the parallel `WeightAccessorIdx` array.
///
/// Why `(bucket, op_idx)`: the `Instruction` variant determines the
/// *kind* of weight (RmsNorm always consumes an RmsNorm). What
/// varies per call site is *which named field* on `Weights` the per-
/// arch impl returns. The bucket id + position-in-bucket fully
/// identify the call site; the layer disambiguates the slice.
///
/// Per-arch impls are macro-generated — see
/// `ferrite-forward-macro::codegen::emit_weight_accessors_impl`.
///
/// Sibling methods cover every weight kind: `rms_norm_at`,
/// `embedding_at`, `linear_at`, `layer_norm_at`, `marlin_at`,
/// `bnb4_at`, `fp8_at`, the four MoE getters, and `cos_sin_at`.
/// Default bodies all `unreachable!()` so per-arch impls only
/// override the methods their generated tape actually exercises.
/// Each method matches on `(bucket, op_idx, slot)` to resolve the
/// named weight at this tape call site. `slot` disambiguates
/// multiple accessors of the same kind at the same position
/// (e.g. `FusedQkvQkNormRopeCache` consumes 3 LinearLayers — q_proj,
/// k_proj, v_proj — all at the same `op_idx`; their `slot`s are
/// 0, 1, 2). For variants with one accessor per kind, `slot` is
/// always 0.
///
/// Default bodies all `unreachable!()` so per-arch impls only
/// override the methods their generated tape actually calls.
///
/// Cross-backend: both the cuda eval body and the metal worker call
/// into the same per-arch impl emitted by
/// `ferrite-forward-macro::codegen::emit_weight_accessors_impl`.
/// Gated on `any(cuda, metal)` because the return types reference
/// `ferrite_kernels::layers::*` structs which only compile under
/// one of those features.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub trait WeightAccessors {
    fn rms_norm_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::ferrite_kernels::layers::RmsNorm {
        unreachable!("rms_norm_at not implemented for this arch")
    }
    fn embedding_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::ferrite_kernels::layers::Embedding {
        unreachable!("embedding_at not implemented for this arch")
    }
    fn linear_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::ferrite_kernels::layers::LinearLayer {
        unreachable!("linear_at not implemented for this arch")
    }
    fn layer_norm_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::ferrite_kernels::layers::LayerNorm {
        unreachable!("layer_norm_at not implemented for this arch")
    }
    fn marlin_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::ferrite_kernels::layers::MarlinLinear {
        unreachable!("marlin_at not implemented for this arch")
    }
    fn bnb4_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::ferrite_kernels::layers::Bnb4bitLinear {
        unreachable!("bnb4_at not implemented for this arch")
    }
    fn fp8_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::ferrite_kernels::layers::Fp8AnyLinear {
        unreachable!("fp8_at not implemented for this arch")
    }
    fn deepseek_moe_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::ferrite_kernels::layers_moe::DeepSeekV2MoELayer {
        unreachable!("deepseek_moe_at not implemented for this arch")
    }
    fn deepseek_moe_fp8_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::ferrite_kernels::layers_moe::DeepSeekV2Fp8BlockMoELayer {
        unreachable!("deepseek_moe_fp8_at not implemented for this arch")
    }
    fn deepseek_moe_ggml_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::ferrite_kernels::layers_moe::DeepSeekV2GgmlMoELayer {
        unreachable!("deepseek_moe_ggml_at not implemented for this arch")
    }
    fn fused_moe_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::ferrite_kernels::layers_moe::FusedMoELayer {
        unreachable!("fused_moe_at not implemented for this arch")
    }
    fn shared_fused_moe_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::ferrite_kernels::layers_moe::SharedFusedMoELayer {
        unreachable!("shared_fused_moe_at not implemented for this arch")
    }
    fn cos_sin_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> ::ferrite_cuda_core::tensor::GpuTensor {
        unreachable!("cos_sin_at not implemented for this arch")
    }
    /// Metal-only: MLX-affine int4 quantized embedding. Resolves the
    /// `AffineQuantEmbedding` field on `Weights` for the
    /// `Instruction::AffineEmbed` lowering.
    #[cfg(feature = "metal")]
    fn affine_quant_embedding_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::ferrite_kernels::layers::AffineQuantEmbedding {
        unreachable!("affine_quant_embedding_at not implemented for this arch")
    }
}

/// Runtime state passed by `&mut` into every `op.eval(&mut ctx)`.
/// Constants live on `W: CanonicalParams`, NOT here.
///
/// **Reshape / RopeAppend `out_slot != in_slot` invariant.** Both
/// instructions write `TileEntry::Reshaped { ref_slot, tensor }` at
/// `out_slot`. The codegen `colored_slot_map` guarantees `out_slot !=
/// in_slot` for every Impl that declares `output_alias_is_view = true`
/// (today: `ReshapeRefImpl`, `RopeAppendRefImpl`) — so the overwrite
/// at `tiles[out_slot]` never drops the upstream's `OwnedTensor` at
/// `tiles[in_slot]`. The G.7(e.tail.2) bug fix had a runtime safety
/// net (`pinned_owned: Vec<OwnedTensor>`) that pinned the prior Owned
/// past the overwrite; once the codegen invariant is in place, the
/// safety net is unnecessary and was deleted in this commit.
#[cfg(feature = "cuda")]
pub struct InterpreterCtx<'a, W> {
    pub wm: &'a W,
    pub tiles: &'a mut Vec<Option<TileEntry>>,
    pub fwd: &'a ForwardCtx<'a>,
    pub device: &'a mut GpuDevice,
    /// Iter index of the enclosing `Op::Loop`, else 0.
    pub layer_offset: u32,
    /// `OwnedTensor`s removed from `tiles` to make room for a
    /// `Reshaped` at the same slot index. Keeping them here pins
    /// the underlying GPU memory for the rest of the run, so any
    /// `Reshaped { ref_slot, tensor }` that aliases this storage
    /// continues to point at live memory.
    ///
    /// **Why this exists.** When `Instruction::Reshape(in_slot,
    /// out_slot, ...)` is emitted with `out_slot == in_slot`, the
    /// previous `TileEntry::Owned(OwnedTensor)` at that slot would
    /// be dropped on overwrite — freeing its block back to the
    /// caching allocator. The new `TileEntry::Reshaped` still holds
    /// the original GPU pointer in its `tensor` field, but that
    /// memory is now in the free pool; subsequent `alloc_tensor`
    /// calls in the same forward (e.g., a downstream attention
    /// kernel's output) hand the same address back, the kernel
    /// writes there while still reading the Reshaped, and the
    /// supposedly-still-live "input" tile sees torn writes (NaN /
    /// extreme magnitudes). Pinning here is the minimum-invasive
    /// fix: the codegen contract that `Reshaped::ref_slot` "pins
    /// the slot whose `OwnedTensor` actually owns the storage" only
    /// works when the OwnedTensor still lives at `ref_slot`; in the
    /// in-place case it has already been overwritten. Stash it.
    pub pinned_owned: Vec<OwnedTensor>,
}

/// Universal opcode set. Tuple variants throughout — keeps each
/// row in a per-canonical static slice on a single line of cargo
/// expand. Field order per variant matches the per-Impl
/// `OpcodeShape::fields` order in `impl_lib.rs`.
///
/// No longer generic over `W` — every variant lost its `WtFn<W, L>` /
/// `CosSinFn<W>` field when fan_out moved weight resolution to the
/// per-arch `WeightAccessors` trait at the tape level. `W` lives only
/// on `eval<W: CanonicalParams>` and `run<W>` / `run_backbone<W>` as
/// method-level generics, parameterizing the interpreter's access to
/// per-canonical model constants without polluting every static row.
#[allow(clippy::type_complexity)]
pub enum Instruction {
    Embed(u32),
    /// `RmsNorm(in_slot, out_slot, layer, hidden_size, m_multiplier)`.
    /// Weight accessor lives at the tape level — see [`Tape`].
    ///
    /// `hidden_size` is the row width the kernel reduces over;
    /// `m_multiplier` is the number of normalization rows per input
    /// token. For the standard residual-stream RmsNorm:
    /// `(hidden_size = W::HIDDEN_SIZE, m_multiplier = 1)`. For
    /// per-head `q_norm` (Qwen3): `(W::HEAD_DIM, num_q_heads)`;
    /// for `k_norm`: `(W::HEAD_DIM, num_kv_heads)`. The metal kernel
    /// dispatches `bucket_m * m_multiplier` threadgroups, each
    /// reducing `hidden_size` elements. Pre-Qwen3 the lowering
    /// hardcoded `(W::Q_SIZE, 1)` — correct for Llama (Q_SIZE ==
    /// hidden) but wrong for Qwen3 (Q_SIZE != hidden, q_norm has
    /// per-head reduction). The cuda interpreter reads tile shape
    /// directly and ignores both fields.
    RmsNorm(u32, u32, u32, u32, u32),
    /// CohereLayerNorm-flavored norm (subtracts mean before scaling).
    /// Claimed from the `(mean, sub, rmsnorm)` math trio in the DSL.
    /// Weight is typed `RmsNorm` because the DSL author writes
    /// `rmsnorm(centered, w)`; the wrapper struct is structurally
    /// identical to the now-retired `CohereLayerNorm` (one `weight`
    /// + one `eps`), and the kernel reads only those two fields.
    MeanSubRmsNorm(u32, u32, u32),
    /// Torch-style LayerNorm with bias. Claimed from the 4-tile pattern
    /// `(mean, sub, rmsnorm, bias_add)` in the DSL — extends the
    /// `MeanSubRmsNorm` math trio with a learned bias addition. Weight
    /// is typed `LayerNorm` (not `RmsNorm`) so the loader auto-pulls
    /// `<prefix>.weight` AND `<prefix>.bias` together; the matcher's
    /// `required_weights` returns one accessor whose source is the
    /// rmsnorm's weight ref. The DSL bias-weight ref (e.g.
    /// `attn_norm.bias[layer]`) is structural-only — same trick the
    /// dense `(Gemm, BiasAdd)` fusion plays via `LinearLayer`.
    /// Used by encoder models like ModernBERT and vision towers like
    /// Qwen2-VL.
    MeanSubRmsNormBiasAdd(u32, u32, u32),
    /// `Reshape(in_slot, out_slot, dims_lit, dims_nt_pow, dims_div_lit, ndim)`.
    /// Output axis i is computed as
    /// `(dims_lit[i] * num_tokens^dims_nt_pow[i]) / dims_div_lit[i]`.
    /// `dims_div_lit` is `1` for every dim by default (G.5.f.a opens
    /// the divisor for DSL-authored arithmetic — the merger's
    /// `[num_tokens / vision_merge_factor, vision_merge_hidden]`
    /// is the first consumer, decomposing as
    /// `dims_div_lit = [vision_merge_factor, 1]`).
    Reshape(
        u32,
        u32,
        [u32; MAX_DIMS],
        [u8; MAX_DIMS],
        [u32; MAX_DIMS],
        u8,
    ),
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
    /// Multimodal embed splice — D2D-copy projected vision-encoder
    /// rows into the placeholder positions of the post-embed hidden
    /// states in-place. Always inserted by `tp_lowering::insert_mm_splices`
    /// after every `Instruction::Embed` (after the vocab-parallel
    /// `Instruction::AllReduce` at tp>1, so the splice runs on the
    /// fully-reduced embedding and its D2D overwrite is NOT summed
    /// across ranks). At runtime the op is a no-op when
    /// `ForwardCtx::embed_patches` is empty (text-only batches) —
    /// one extra slot check per forward pass, cost negligible.
    SpliceMmEmbeds(u32),
    ScalarMul(u32, u32, f32),
    TanhSoftCap(u32, u32),
    /// `FusedAddRmsNorm(delta_slot, residual_slot, layer, hidden_size, m_multiplier)`.
    /// See [`RmsNorm`](Instruction::RmsNorm) for the field semantics.
    /// FusedAddRmsNorm is always on the residual stream so
    /// `m_multiplier = 1` and `hidden_size = W::HIDDEN_SIZE`.
    FusedAddRmsNorm(u32, u32, u32, u32, u32),
    FusedAddRmsNormWithOffset(u32, u32, u32, f32),
    ScalarOffsetRmsNorm(u32, u32, u32, f32),
    /// Norm→Gemm fusion: `cutlass_gemm(rms_norm(in), gemm_w)`. The
    /// CUTLASS tile is bucket-pickable per `CUTLASS_TILE_ZOO` entry.
    /// Reuses existing `kernels::rms_norm` + `cutlass::cutlass_gemm`
    /// — no new .cu file. Matches body norms whose only downstream
    /// consumer is a single dense Gemm.
    CutlassFusedRmsNormGemm(u32, u32, u32, u32, u32, u32, u32, u32),
    /// Norm→Gemm fusion for CohereLayerNorm-flavored norms.
    /// Claimed from the `(mean, sub, rmsnorm, gemm)` 4-tile pattern
    /// where the Gemm is the sole consumer of the rmsnorm output.
    /// Runs `cohere_layer_norm` then `cutlass_gemm` sequentially.
    CutlassFusedMeanSubRmsNormGemm(u32, u32, u32, u32, u32, u32, u32, u32),
    /// (Add, RmsNorm, Gemm) 3-tile fusion. Claim shape captures
    /// the lm_head canonical pattern `x = x + delta; logits =
    /// lm_head(rmsnorm(x))`. Runtime: `fused_add_rms_norm_inplace`
    /// then `cutlass_gemm`. The Add's residual update is exposed as
    /// a TensorView aliasing the residual upstream OwnedTensor (same
    /// alias semantics as `FusedAddRmsNorm`); the Gemm output is a
    /// fresh OwnedTensor.
    CutlassFusedAddRmsNormGemm(u32, u32, u32, u32, u32, u32, u32, u32, u32),
    Gemm(u32, u32, u32, u32, u32),
    /// cuBLAS-side peer to `CutlassGemmAdd`. cuBLAS GEMM produces a
    /// delta; `add_inplace` then folds it into the residual buffer.
    /// Output is the residual upstream's OwnedTensor (aliased via
    /// the codegen prelude); no `out_slot` payload.
    FusedCublasGemmAdd(u32, u32, u32, u32, u32),
    FusedGemmBias(u32, u32, u32),
    /// Metal-only per-row bias broadcast add. Emitted by
    /// `MetalBiasAddImpl` when the synth-pre-attn megakernel doesn't
    /// claim the biased QKV chain (today: M ≥ 2 prefill, where the
    /// cost CSV picks unfused — see `project_metal_synth_solver_handoff`).
    /// CUDA's analogue is `FusedGemmBias`, which folds the bias into
    /// cuBLAS's gemm_bias epilog. On Metal the singleton path
    /// dispatches a separate `KernelId::BiasAdd` after the AffineQmm
    /// / Gemm; once the synth path absorbs biases the solver will
    /// prefer that over this singleton on decode workloads.
    ///
    /// Fields: `(in_slot, out_slot, layer, n, is_affine)`. Bias is
    /// resolved through the tape-level `WeightAccessors::linear_at`
    /// at the macro-emitted `(bucket, op_idx, slot=0, layer)` site;
    /// `is_affine` tells the worker whether to pull `LinearLayer::Dense
    /// .bias` or `LinearLayer::AffineQuant.linear_bias`. `n` is the
    /// bias's broadcast dimension — baked into the specialized
    /// pipeline as `function_constant(0)`.
    MetalBiasAdd(u32, u32, u32, u32, bool),
    FusedGateUpSiluMul(u32, u32, u32),
    FusedGateUpGeluMul(u32, u32, u32),
    FusedQkvRopeCache(u32, u32, u32, bool, bool),
    FusedQkvQkNormRopeCache(u32, u32, u32, f32, f32),
    FusedQkvRopePrefill(u32, u32, u32, u32, u32, bool, bool),
    AttentionViaCache(u32, u32, u32, bool),
    AttentionPrefillContiguous(u32, u32, u32, u32, bool),
    /// Prefill attention reading K/V from the paged KV cache via
    /// block_table indirection. `(q_slot, out_slot, layer, interleaved)`.
    /// Drops the (k_slot, v_slot) pair that
    /// [`Instruction::AttentionPrefillContiguous`] carries — the
    /// upstream `RopeAppend` already wrote rotated K + raw V into the
    /// per-layer paged cache slots, and the kernel reads them through
    /// `block_table` with the K-axis covering the FULL `seqused_k[seq]`
    /// (prefix + new). The per-Q causal mask shifts by
    /// `(seqused_k[seq] - new_q_for_seq)` so prior cached prefix
    /// contributes to attention. Required for chunked prefill, prefix
    /// caching, mixed prefill+decode batches, and multi-turn chat —
    /// scenarios `AttentionPrefillContiguous` cannot handle because
    /// its K-axis is bounded by `cu_seqlens_q` (new tokens only).
    /// Currently emitted only by the metal adapter; cuda continues to
    /// route prefill through `flash_attn_contiguous`.
    AttentionPrefillPaged(u32, u32, u32, bool),
    /// Bidirectional / encoder attention. Reads contiguous Q/K/V from
    /// the upstream tile slots; calls `flash_attn_contiguous` with
    /// `is_causal=false` and a null cos_sin pointer (RoPE applied
    /// separately upstream). No KV cache, no per-layer cos_sin —
    /// the encoder DSL form is `attention(q, k, v)` (3 args). Q/K/V
    /// must already be 3D `[T, heads, head_dim]` from the upstream
    /// projection chain. Output is reshaped to `[T, Q_SIZE]`.
    EncoderAttention(u32, u32, u32, u32),
    SlidingAttentionViaCache(u32, u32, u32, bool),
    SlidingAttentionPrefillContiguous(u32, u32, u32, u32, bool),
    /// Vision-tower variable-length attention: `(q_slot, k_slot,
    /// v_slot, out_slot, cu_seqlens_kind)`. The `cu_seqlens_kind`
    /// u8 discriminant selects which `(cu_seqlens, max_seqlen)` pair
    /// the kernel reads:
    ///
    /// - `0` (Default): `ForwardCtx::cu_seqlens_q` + `max_seqlen_q`.
    ///   Qwen2-VL's single-cu-seqlens path; the host wrapper
    ///   populates these from the per-batch concatenated boundaries.
    /// - `1` (Full): `ForwardCtx::vision_cu_seqlens_full` +
    ///   `vision_max_seqlen_full`. Qwen2.5-VL's full-frame layers
    ///   (`fullatt_block_indexes = [7, 15, 23, 31]`).
    /// - `2` (Window): `ForwardCtx::vision_cu_seqlens_window` +
    ///   `vision_max_seqlen_window`. Qwen2.5-VL's windowed-attn
    ///   layers (the other 28 of 32).
    ///
    /// Bidirectional (no causal mask), no softcap, no fused rope —
    /// rope is applied upstream by `Instruction::VisionRope`. Output
    /// reshaped to rank-2 `[L, vision_num_heads * vision_head_dim]`
    /// to match the FUF's q-shape contract.
    VarlenAttention(u32, u32, u32, u32, u8),
    /// Vision 2D RoPE pair-rotation: `(q_in, k_in, q_out, k_out)`.
    /// Reads cos / sin from `ForwardCtx::vision_rope_cos` /
    /// `vision_rope_sin`. In-place on q / k buffers; the output slots
    /// reinsert the same `OwnedTensor`s after mutation. Handles the
    /// rank-2 → rank-3 reshape internally; the kernel
    /// `vision_rope_apply` requires rank-3 `[L, H, D]`.
    VisionRope(u32, u32, u32, u32),
    /// Quick-GELU activation: `(in_slot, out_slot)`. In-place
    /// elementwise mutation, take-owned + kernel + reinsert. Same
    /// shape as `TanhSoftCap` / `ScalarMul` consume-pattern.
    QuickGelu(u32, u32),
    /// GELU tanh-approximation activation: `(in_slot, out_slot)`.
    /// In-place mutation. Mirrors `QuickGelu` / `GeluErf`; matches
    /// PyTorch `nn.GELU(approximate="tanh")`. Used by SigLIP /
    /// Gemma3-MM vision MLP.
    Gelu(u32, u32),
    /// Vision learned positional embedding lookup: `(out_slot,
    /// weight_fn)`. Reads `ctx.fwd.vision_position_ids` (rank-1 u32
    /// view of length `num_tokens`) and gathers rows from the
    /// per-arch positional table via the same
    /// `kernels::embedding_gather_masked` kernel
    /// `Instruction::Embed` calls (with `vocab_offset = 0` /
    /// `vocab_per_rank = num_positions` so the mask never trips).
    /// Used by SigLIP / Gemma3-MM. Distinct from `Embed` because the
    /// table dim 0 is `vision_num_positions` (not `vocab_size`); the
    /// `OpKind::PosEmbed` shape sig anchors on the vision bound names
    /// so the macro emits a `Weights` field of the right type.
    PosEmbed(u32),
    /// Materialize the vision-prelude `pixels` extern as a tile:
    /// `(out_slot)`. Reads `ctx.fwd.pixels` (the rank-2
    /// `[num_tokens, vision_in_features]` view the
    /// `vision_forward` host wrapper writes onto `ForwardCtx`
    /// before invoking the vision interpreter), allocates a fresh
    /// `OwnedTensor` of the same shape/dtype, D2D-copies the
    /// pixels view into it, and publishes the result at `out_slot`.
    ///
    /// The copy is what lets the consume-pattern vision Impls
    /// (`QuickGelu` / `GeluErf` / `VisionRope`) downstream of
    /// `pixels` read it as a tile-table `Owned` entry — `take_owned`
    /// requires `Owned`, and a borrowed `External` wrapper around
    /// `ctx.fwd.pixels` would panic on the first such consumer.
    /// In G.5.f's real encoder body the first op on pixels is a
    /// non-consuming `gemm` (`patch_embed`), so the copy is paid
    /// once per encoder invocation regardless. Synthesized
    /// exclusively by `vision_lowering::materialize_pixels` —
    /// never appears in any DSL.
    LoadPixels(u32),
    /// Erf-form GELU activation: same shape as `QuickGelu`. Distinct
    /// numerics (`0.5 * x * (1 + erf(x / sqrt(2)))`).
    GeluErf(u32, u32),
    /// Row-permutation gather: `(in_slot, out_slot, indices_kind)`.
    /// Reads the source rank-2 tile from `in_slot`, the rank-1 u32
    /// indices buffer from `ForwardCtx::vision_window_index` (kind=0)
    /// or `ForwardCtx::vision_reverse_indices` (kind=1), and writes a
    /// fresh `OwnedTensor` of the same shape as the source to
    /// `out_slot`. Output row `i` = source row `indices[i]`. Used by
    /// Qwen2.5-VL's window-attention dispatch — tokens, cos, sin are
    /// gather-permuted into window order on encoder entry, and the
    /// merger output is permuted back to natural order at exit.
    EmbeddingGather(u32, u32, u8),
    /// 2-D non-overlapping average pool: `(in_slot, out_slot)`. Reads
    /// the source rank-2 tile `[L = ph², e]` from `in_slot`, walks the
    /// flat row index as a `(row, col)` pair on a `ph × ph` grid (with
    /// `ph = W::VISION_PATCH_GRID_SIDE`), and averages each k×k cell
    /// (`k = W::VISION_POOL_KERNEL`) into one output row. Output is a
    /// fresh `OwnedTensor` of shape `[(ph/k)², e]`. Used by Gemma3-MM's
    /// SigLIP→text projector. Stride == kernel (non-overlapping).
    AvgPool2d(u32, u32),
    /// CLIP-class CLS-token strip: `(in_slot, out_slot)`. Reads the
    /// rank-2 tile `[L, e]` from `in_slot` (where the CLS row has been
    /// run through the encoder at row 0), copies rows `1..L` into a
    /// fresh `OwnedTensor` of shape `[L - 1, e]`, and publishes that
    /// at `out_slot`. No baked-in constant — input dims are read off
    /// the source tile at runtime, so a misconfigured `vision_in_seq_len
    /// = vision_num_positions - 1` invariant in the variant config
    /// surfaces as a downstream shape mismatch (caught at expansion
    /// time by the bound-resolved sig), not as a kernel panic. Used by
    /// LLaVA-1.5 family with `vision_feature_select_strategy =
    /// "default"`.
    StripCls(u32, u32),
    FlashInferAttentionDecode(u32, u32, u32, u32, bool),
    FlashInferAttentionPrefill(u32, u32, u32, u32, u32, u32, bool),
    /// Hopper-native FA3 paged decode. Selected by the solver when
    /// `cuda_arch >= 90` and the cost CSV calibrated FA3 below FI for
    /// this `(num_tokens, sk_bucket)` cell. No softcap field — the
    /// vendored FA3 build excludes softcap variants
    /// (`FLASHATTENTION_DISABLE_SOFTCAP`).
    /// Args: (in_slot, out_slot, layer, head_dim).
    #[cfg(fa3_built)]
    FlashAttention3Decode(u32, u32, u32, u32),
    RopeAppend(u32, u32, u32, u32, u32, u32, u32, bool),
    MlaSplit(u32, u32, u32),
    MlaAttention(u32, u32, u32, u32, u32),
    DeepSeekMoe(u32, u32, u32),
    DeepSeekMoeFp8Block(u32, u32, u32),
    DeepSeekMoeGgml(u32, u32, u32),
    /// Mixtral-style BF16 fused MoE (no shared expert). Top-k routing
    /// via softmax; experts are stacked `[E, 2*inter, hidden]` /
    /// `[E, hidden, inter]` and dispatched through the fused-MoE GEMM.
    FusedMoe(u32, u32, u32),
    /// Qwen-MoE-style BF16 fused MoE PLUS shared expert with sigmoid
    /// gate. Routed experts use the same fused-MoE pipeline as
    /// `FusedMoe` (with `renormalize=true`); the shared expert is a
    /// SwiGLU MLP gated by `sigmoid(shared_expert_gate(x))`.
    SharedFusedMoe(u32, u32, u32),
    /// Metal-only Mixtral-style fused MoE: carries the full structural
    /// shape baked at macro-expansion time from the model config, so
    /// the Metal lowering pass + worker can specialize pipelines
    /// (`function_constant`s for moe_weighted_sum, affine_gather_qmv),
    /// stamp `Binding::Inline` u32s (top_k, axis_size), and size the
    /// per-bucket MoE scratch buffer — all without a runtime
    /// shape-resolution detour.
    ///
    /// Tuple fields: `(in_slot, out_slot, layer, num_experts, top_k,
    /// moe_intermediate_size, hidden_size, group_size, bits)`.
    /// `group_size` + `bits` come from the per-expert weight
    /// `StorageFormat::Affine`; for the on-disk mlx-community 4bit
    /// MoE checkpoints those are `64` and `4` respectively.
    ///
    /// Softmax order is fixed by variant identity: Mixtral does
    /// `topk → softmax(scores)`, no renorm flag. (See `SharedFusedMoe`
    /// for the Qwen `softmax → topk → take_along_axis` order.)
    /// CUDA-target macros keep emitting [`Instruction::FusedMoe`];
    /// the Metal-target `MetalFusedMoeImpl` (in
    /// `ferrite-forward-macro::metal::moe`) emits this variant when
    /// `profile.backend == Backend::Metal` and the per-expert weight
    /// storage is `Affine`.
    MetalFusedMoe(u32, u32, u32, u32, u32, u32, u32, u32, u32),
    /// Metal-only Qwen-MoE-style fused MoE + optional shared expert.
    /// Same rationale as [`Instruction::MetalFusedMoe`]: macro-baked
    /// shape for the Metal lowering arm.
    ///
    /// Tuple fields: `(in_slot, out_slot, layer, num_experts, top_k,
    /// moe_intermediate_size, hidden_size, shared_intermediate_size,
    /// group_size, bits, norm_topk_prob)`.
    /// `shared_intermediate_size = 0` means no shared expert — the
    /// lowering arm skips the shared-expert tail entirely
    /// (modern Qwen3-MoE-30B-A3B-Instruct ships this). Qwen1.5-MoE /
    /// Qwen2-MoE ship non-zero. `norm_topk_prob = true` triggers the
    /// post-`take_along_axis` renorm of gathered scores (Qwen3-MoE's
    /// `norm_topk_prob` config flag).
    ///
    /// Softmax order is fixed by variant identity: Qwen-MoE does
    /// `softmax → topk → take_along_axis(scores)`.
    MetalSharedFusedMoe(u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, bool),
    CutlassGemm(u32, u32, u32, u32, u32, u32, u32, u32),
    CutlassGemmSplitK(u32, u32, u32, u32, u32, u32, u32, u32, u32),
    CutlassGemmAdd(u32, u32, u32, u32, u32, u32, u32, u32),
    CutlassGemv(u32, u32, u32, u32, u32),
    CutlassFusedGemmBias(u32, u32, u32, u32, u32, u32, u32, u32),
    CutlassFusedGateUpSiluMul(u32, u32, u32, u32, u32, u32),
    CutlassFusedGateUpGeluMul(u32, u32, u32, u32, u32, u32, u32, u32),
    CutlassFusedQkvRopeCache(u32, u32, u32, bool, u32, u32, u32, u32, u32),
    CutlassFusedQkvRopePrefill(u32, u32, u32, u32, u32, bool, u32, u32, u32, u32, u32),
    MarlinGemm(u32, u32, u32),
    MarlinFusedGateUpSiluMul(u32, u32, u32),
    MarlinFusedGateUpGeluMul(u32, u32, u32),
    MarlinFusedQkvRopeCache(u32, u32, u32),
    MarlinFusedQkvRopePrefill(u32, u32, u32, u32, u32),
    Bnb4Gemm(u32, u32, u32),
    Bnb4FusedGateUpSiluMul(u32, u32, u32),
    Bnb4FusedGateUpGeluMul(u32, u32, u32),
    Bnb4FusedQkvRopeCache(u32, u32, u32),
    Bnb4FusedQkvRopePrefill(u32, u32, u32, u32, u32),
    GgmlGemm(u32, u32, u32),
    GgmlFusedGateUpSiluMul(u32, u32, u32),
    GgmlFusedGateUpGeluMul(u32, u32, u32),
    GgmlFusedQkvRopeCache(u32, u32, u32, bool),
    GgmlFusedQkvRopePrefill(u32, u32, u32, u32, u32),
    Fp8Gemm(u32, u32, u32),
    Fp8FusedGemmBias(u32, u32, u32),
    Fp8FusedGateUpSiluMul(u32, u32, u32),
    Fp8FusedGateUpGeluMul(u32, u32, u32),
    Fp8FusedQkvRopeCache(u32, u32, u32),
    Fp8FusedQkvRopePrefill(u32, u32, u32, u32, u32),
    /// MLX-affine int4 matmul (transpose=true). Metal-only. The
    /// `LinearLayer` resolved through `WeightAccessors::linear_at`
    /// must be `AffineQuant` — the metal worker reads the quant
    /// accessors (packed weight, scales, per-group affine biases,
    /// optional fp linear bias) through the `WeightTensor::Affine*`
    /// arms.
    ///
    /// Tuple fields: `(in_slot, out_slot, layer, n, k, group_size,
    /// bits, vector_limit)`. Weight comes from
    /// `WeightAccessors::linear_at(bucket, op_idx, slot=0, layer)`.
    /// `vector_limit` is the matvec/matmul boundary from
    /// `get_qmv_batch_limit(K, N, arch_gen)` (mirrors MLX
    /// `quantized.cpp:84`); the macro bakes it at codegen time so
    /// `lower_one` can compare against `bucket_m` without reaching
    /// for the target profile. M < limit → qmv (decode-matvec);
    /// M ≥ limit → qmm_t (prefill-matmul, SplitK heuristic deferred
    /// to C3).
    ///
    /// CUDA eval is `unreachable!` — emit only on the metal forward.
    AffineQmm(u32, u32, u32, u32, u32, u32, u32, u32),
    /// NVIDIA ModelOpt NVFP4 int4 matmul (transpose=true). Metal-only.
    /// Identical shape and dispatch to `AffineQmm` — the only difference
    /// is the in-shader weight decode (E2M1 LUT, no per-group bias). The
    /// `LinearLayer` resolved through `WeightAccessors::linear_at` must be
    /// `Nvfp4`; the metal worker reads the quant accessors (packed E2M1
    /// weight, folded per-group scales, optional fp linear bias) through
    /// the `WeightTensor::Nvfp4*` arms.
    ///
    /// Tuple fields: `(in_slot, out_slot, layer, n, k, group_size, bits,
    /// vector_limit)` — `group_size` is 16, `bits` is 4. `vector_limit` is
    /// the matvec/matmul boundary baked at codegen time (same source as
    /// `AffineQmm`). M < limit → nvfp4_qmv (decode); M ≥ limit →
    /// nvfp4_qmm_t (prefill).
    ///
    /// CUDA eval is `unreachable!` — emit only on the metal forward.
    Nvfp4Qmm(u32, u32, u32, u32, u32, u32, u32, u32),
    /// Compiler-synthesized pre-attention chunk megakernel. Metal-only.
    /// Combines (Add → RmsNorm → 3×AffineQmv → RoPE → paged KV-cache
    /// write) into one dispatch. The kernel itself is generated at
    /// macro-expansion time by
    /// `ferrite-forward-macro/src/fuse_pass.rs` stitching MK primitive
    /// calls; the symbol name carried here resolves at runtime against
    /// a per-arch source-compiled library registered into the
    /// `SpecializedPipelineCache` at worker-pool init.
    ///
    /// Tuple fields: `(residual_slot, delta_slot, q_out_slot, layer,
    /// group_size, bits, kernel_symbol, has_linear_bias)`. Q/K/V
    /// LinearLayer accessors and the RmsNorm/CosSin pair resolve
    /// through `WeightAccessors::{linear_at, rms_norm_at, cos_sin_at}`
    /// at slots 0..3 of the same `(bucket, op_idx)` — avoiding the
    /// load-time packed-concat infrastructure that Phase 0's revert
    /// dropped.
    ///
    /// CUDA eval is `unreachable!`.
    SynthPreAttn(
        u32, // 0: residual_in_slot
        u32, // 1: delta_slot
        u32, // 2: q_out_slot
        /// 3: `residual_out_slot` — distinct arena slot for the updated
        /// residual (`residual_in + delta`). NOT written in place: the
        /// kernel is tiled across threadgroups and reads the whole
        /// `residual_in` for the rmsnorm sum, so an in-place write would
        /// race cross-threadgroup. The coloring keeps it distinct from
        /// `residual_in_slot`; downstream reads this. For the layer-0
        /// `_init` variant (no residual add) the `fan_out` sets it equal
        /// to `residual_in_slot` and the kernel leaves it untouched.
        u32,
        u32,          // 4: layer
        u32,          // 5: group_size
        u32,          // 6: bits
        &'static str, // 7: kernel_symbol
        /// 8: `has_linear_bias`. When `true`, the lowering arm appends 3
        /// extra `Binding::Weight { which: AffineLinearBias, .. }`
        /// entries for Q/K/V at buffers 18/19/20 and picks the
        /// `_bias` synth kernel symbol (set by `fan_out`). Qwen2/2.5
        /// quantized chains land here; Llama stays `false`.
        bool,
    ),
    /// Compiler-synthesized MLP pre-down chunk megakernel. Metal-only.
    /// Combines (FusedAddRmsNorm → AffineQmv gate → AffineQmv up →
    /// SiluMul) into one dispatch, leaving the down-projection as a
    /// standalone `AffineQmm` consumer of the synthesized output. The
    /// kernel itself is generated at macro-expansion time by
    /// `ferrite-forward-macro::fuse_pass::synthesize_mlp_pre_down_chunk`;
    /// the symbol name carried here resolves at runtime against a
    /// per-arch source-compiled library registered into the
    /// `SpecializedPipelineCache` at worker-pool init.
    ///
    /// Tuple fields: `(residual_in_slot, delta_slot, silu_mul_out_slot,
    /// residual_out_slot, layer, group_size, bits, kernel_symbol)`.
    /// Gate/up `LinearLayer`s and the RmsNorm resolve through
    /// `WeightAccessors::{linear_at, rms_norm_at}` at slots 0/1/2 of the
    /// same `(bucket, op_idx)`. The standalone `AffineQmm` for down_proj
    /// follows directly in the instruction stream as before.
    ///
    /// `residual_out_slot` is the distinct arena slot the fusion writes
    /// the updated residual (`residual_in + delta`) into — NOT in place,
    /// because the kernel is tiled across threadgroups and reads the
    /// whole `residual_in` for the rmsnorm sum (an in-place write would
    /// race cross-threadgroup). The coloring assigns it a slot distinct
    /// from `residual_in_slot` (fused-subgraph inputs stay live to the
    /// subgraph's end); downstream reads `residual_out_slot`.
    ///
    /// CUDA eval is `unreachable!`.
    SynthMlpPreDown(u32, u32, u32, u32, u32, u32, u32, &'static str),
    /// Fused elementwise `silu(gate) * up` for the decomposed q-MLP
    /// path (plan P12 branch (i)). The macro emits this after a pair
    /// of `AffineQmm` GEMMs when both gate_proj and up_proj are
    /// MLX-affine quantized — `MetalFusedGateUpSiluMulImpl::fan_out`
    /// produces (AffineQmm, AffineQmm, SiluMul) in that case rather
    /// than a single fused `FusedGateUpSiluMul` (which assumes Dense
    /// storage).
    ///
    /// Tuple fields: `(gate_slot, up_slot, out_slot)`. Both inputs
    /// must be `[M, intermediate_size]` in the activation dtype;
    /// output is the same shape. CUDA eval is `unreachable!` —
    /// metal-only (CUDA's q-MLP routes through Marlin/Bnb/etc).
    SiluMul(u32, u32, u32),
    /// Fused gate+up GEMM + SiluMul for large-M prefill (M ≥ 8).
    /// Metal-only; emitted by `MetalSynthGateUpSiluMulImpl`.
    /// Fields: `(x_norm_slot, out_slot, layer, group_size, bits,
    /// kernel_symbol)`. Gate/up LinearLayer accessors resolve through
    /// `WeightAccessors::linear_at` at slots 0/1 of the same
    /// `(bucket, op_idx)`.
    #[cfg(feature = "metal")]
    SynthGateUpSiluMul(u32, u32, u32, u32, u32, &'static str),
    /// MLX-affine int4 quantized embedding lookup (Metal-only).
    /// Replaces `Instruction::Embed` when `model.embed_tokens` ships
    /// as a quantized triple `(weight=U32, scales, biases)` — i.e.
    /// every `mlx-community/*-4bit` checkpoint.
    ///
    /// Faithful port of MLX's `nn.QuantizedEmbedding.__call__`
    /// (`python/mlx/nn/layers/quantized.py:144`), fused into one
    /// dispatch via `affine_embed_<dtype>_gs_<gs>_b_4` in
    /// `quantized_dequantize.metal`. Without this lift the embedding
    /// would CPU-dequantize at load (P2 deviation), burning ~2 GB of
    /// arena on Llama-3.2-1B.
    ///
    /// Tuple fields: `(out_slot, group_size, bits)`. The
    /// `AffineQuantEmbedding` weight resolves through a dedicated
    /// `WeightAccessors::affine_quant_embedding_at` accessor (mirrors
    /// `embedding_at` for dense embeds). The hidden_size rides
    /// through `W::Q_SIZE` (function constant baked at lower time).
    /// Metal-only — `AffineQuantEmbedding` is cfg-gated to the metal
    /// backend (mirrors `LinearLayer::AffineQuant`).
    #[cfg(feature = "metal")]
    AffineEmbed(u32, u32, u32),
    /// Re-run the next `body_len` instructions `count` times.
    Loop(u32, u32),
    /// `tiles[dst] = Some(View(src))`.
    Alias(u32, u32),
    /// Drop the OwnedTensor at `slot`.
    Free(u32),
}

impl Copy for Instruction {}
impl Clone for Instruction {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

/// Assert a runtime weight tensor's `[N, K]` shape matches the
/// codegen-time constants the `Instruction` was emitted with. The
/// constants come from the FUF's `eval_shape` at solve time; the
/// runtime tensor is whatever `WeightAccessors` resolved to from the loaded
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
#[cfg(feature = "cuda")]
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
#[cfg(feature = "cuda")]
#[inline]
#[cfg(feature = "cuda")]
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

/// Cold path for `Instruction::AttentionPrefillContiguous` when
/// `max_seqlen_q != max_seqlen_k` (chunked-prefill chunk 2+).
///
/// **Deliberately non-generic.** Callers pass HEAD_DIM / ATTN_SCALE /
/// ATTN_SOFTCAP as runtime args instead of `W`-typed const-generics, so
/// this compiles to ONE symbol regardless of how many model archs the
/// crate emits. With `#[cold]` + `#[inline(never)]` the symbol stays
/// out-of-line. eval<W>'s match arm reduces to a small forwarder; none
/// of the `flashinfer_attention` / `flash_attn_paged_ext` symbol
/// references end up in the per-arch eval<W> bodies, where they would
/// otherwise grow the function and disturb LLVM's code layout for the
/// hot fresh-prefill path.
///
/// Reads K/V from the paged cache (block_table + seqused_k); chunk 1's
/// writes plus this chunk's upstream `RopeAndCacheKV` mean the cache
/// spans the full sequence at this point, so q's q_len rows can attend
/// over the full seqlen_k.
#[cfg(feature = "cuda")]
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn attn_prefill_chunked_continuation(
    kv_cache: &ferrite_kernels::kv_cache::KvCachePool,
    cu_seqlens_q: ferrite_cuda_core::tensor::TensorView<'_>,
    seqused_k: ferrite_cuda_core::tensor::TensorView<'_>,
    block_table: ferrite_cuda_core::tensor::TensorView<'_>,
    q: ferrite_cuda_core::tensor::TensorView<'static>,
    max_q: usize,
    max_k: usize,
    layer: usize,
    head_dim: u32,
    attn_scale: f32,
    attn_softcap: f32,
    num_sm: i32,
    caching: &mut ferrite_cuda_core::CachingAllocator,
    stream: ferrite_cuda_core::CUstream,
    interleaved: bool,
) -> OwnedTensor {
    let fi_cfg = flashinfer::FlashInferConfig {
        dtype: flashinfer::FiDType::Bf16,
        head_dim,
        use_logits_soft_cap: attn_softcap > 0.0,
    };
    let sk_bucket = ah::sk_bucket_for(max_k);
    let fi = unsafe {
        ah::flashinfer_attention(
            q,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_q,
            max_k,
            attn_scale,
            attn_softcap,
            kv_cache,
            layer,
            num_sm,
            fi_cfg,
            sk_bucket,
            caching,
            stream,
        )
    };
    if let Some(t) = fi {
        return t;
    }
    unsafe {
        kernels::flash_attn_paged_ext(
            *q,
            *kv_cache.k_cache(layer),
            *kv_cache.v_cache(layer),
            *cu_seqlens_q,
            *seqused_k,
            *block_table,
            max_q,
            max_k,
            attn_scale,
            true,
            attn_softcap,
            -1,
            kv_cache.block_size,
            num_sm,
            caching,
            stream,
            ::std::ptr::null::<u8>(),
            0,
            interleaved,
            kv_cache.block_unrotated_gpu(),
        )
    }
}

#[cfg(feature = "cuda")]
impl Instruction {
    /// Evaluate one instruction. Closed match (no `_` arm).
    /// `Loop` is dispatched by [`run`] — never reaches here.
    ///
    /// `acc` is the weight accessor index for this op position in
    /// the [`Tape`]. Variants that don't consume a weight (e.g.
    /// `Add`, `BarrierSignal`, `Reshape`) ignore it; the
    /// proc-macro emits a sentinel value at the matching tape
    /// position so the parallel arrays stay aligned.
    ///
    /// # Safety
    /// Tile slot indices in range; weight accessors produce live
    /// GPU memory; `ctx.device.compute_stream` is live.
    #[allow(clippy::too_many_lines)]
    /// `bucket` and `op_idx` together identify the call site so the
    /// per-arch [`WeightAccessors`] impl can resolve which named
    /// weight to return. Variants that consume no weight ignore both.
    pub unsafe fn eval<W: CanonicalParams>(
        &self,
        ctx: &mut InterpreterCtx<'_, W>,
        bucket: u32,
        op_idx: u32,
    ) {
        let _ = (bucket, op_idx); // unused for variants that consume no weight
        match *self {
            Instruction::Embed(out_slot) => unsafe {
                let weight = ctx.wm.embedding_at(bucket, op_idx, 0, 0u32).weight;
                // At tp=1 the embed weight covers the full vocab and
                // vocab_offset is 0 — the mask never trips. At tp>1
                // the weight is the per-rank `[vocab/tp, hidden]`
                // shard; vocab_offset = rank * vocab_per_rank gives
                // each rank a disjoint slice of the global vocab.
                // Per-Embed call follows with an AllReduce-sum
                // (injected by tp_lowering when shard-kind for the
                // embed weight is ShardDim0), and then a
                // `SpliceMmEmbeds` pass (also injected by tp_lowering)
                // that D2D-copies the vision-encoder embeddings into
                // the image-placeholder rows. The splice MUST run
                // after the AllReduce so its overwrite doesn't get
                // multiplied by `tp_world_size` on the sum.
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
            Instruction::SpliceMmEmbeds(slot) => unsafe {
                // Text-only batches: skip. One branch per forward pass.
                if ctx.fwd.embed_patches.is_empty() {
                    return;
                }
                let out = tile_ref(ctx.tiles, slot).as_gpu_tensor(ctx.tiles);
                let mm = ctx
                    .fwd
                    .mm_embeds
                    .expect("mm_embeds required when embed_patches is non-empty");
                let hidden = out.dim(1);
                let elem_bytes = out.dtype().size_bytes();
                let row_bytes = hidden * elem_bytes;
                let dst_base = out.raw_ptr();
                let src_base = mm.raw_ptr();
                let mut src_row: usize = 0;
                for patch in ctx.fwd.embed_patches.iter() {
                    let length = patch.length as usize;
                    let dst = dst_base.add((patch.token_offset as usize) * row_bytes);
                    let src = src_base.add(src_row * row_bytes);
                    let _ = ferrite_cuda_core::driver::memcpy_dtod_async(
                        dst,
                        src,
                        length * row_bytes,
                        ctx.device.compute_stream,
                    );
                    src_row += length;
                }
            },
            Instruction::RmsNorm(in_slot, out_slot, layer, _hidden_size, _m_multiplier) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                let out = kernels::rms_norm(
                    *v,
                    w.weight,
                    w.eps,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::MeanSubRmsNorm(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                let out = kernels::cohere_layer_norm(
                    *v,
                    w.weight,
                    w.eps,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::MeanSubRmsNormBiasAdd(in_slot, out_slot, layer) => unsafe {
                // `LayerNorm`'s `bias` is `Option<GpuTensor>`; this Impl
                // is only emitted when the DSL has a downstream
                // `bias_add`, which means the loader is expected to
                // populate the bias. Empty bias is a loader/DSL
                // mismatch — surface it via `expect` rather than
                // silently dropping the bias term and producing wrong
                // numerics.
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let ln = ctx.wm.layer_norm_at(bucket, op_idx, 0, layer);
                let bias = ln
                    .bias
                    .expect("MeanSubRmsNormBiasAdd: LayerNorm.bias must be Some");
                let out = kernels::layer_norm_bias(
                    *v,
                    ln.weight,
                    bias,
                    ln.eps,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Reshape(in_slot, out_slot, dims_lit, dims_nt_pow, dims_div_lit, ndim) => {
                let upstream = tile_ref(ctx.tiles, in_slot).as_gpu_tensor(ctx.tiles);
                // `num_tokens` source: vision bodies set `pixels`
                // (and the input_ids ForwardCtx field is unused); the
                // decoder path keys off `input_ids`.
                let nt = match ctx.fwd.pixels {
                    Some(p) => (*p).dim(0),
                    None => (*ctx.fwd.input_ids).dim(0),
                };
                let mut shape = [0usize; MAX_DIMS];
                let nd = ndim as usize;
                for i in 0..nd {
                    let mut d = dims_lit[i] as usize;
                    for _ in 0..(dims_nt_pow[i] as usize) {
                        d *= nt;
                    }
                    let div = dims_div_lit[i] as usize;
                    debug_assert!(
                        div > 0 && d.is_multiple_of(div),
                        "Reshape: axis {i} numerator {d} not divisible by \
                         denominator {div} — codegen bug"
                    );
                    shape[i] = d / div;
                }
                let reshaped = upstream.reshape(&shape[..nd]);
                // Pin overwritten Owned: if `out_slot == in_slot`, the
                // existing `TileEntry::Owned` would otherwise drop here,
                // freeing the very memory the new `Reshaped` aliases.
                // See `pinned_owned` doc on `InterpreterCtx`. Take the
                // old entry out first so we can stash any Owned without
                // double-borrowing `ctx.tiles`.
                let prev = std::mem::take(&mut ctx.tiles[out_slot as usize]);
                if let Some(TileEntry::Owned(t)) = prev {
                    ctx.pinned_owned.push(t);
                }
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
            Instruction::FusedAddRmsNorm(
                delta_slot,
                residual_slot,
                layer,
                _hidden_size,
                _m_multiplier,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let delta = tile_ref(ctx.tiles, delta_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                let w = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                let _ = kernels::fused_add_rms_norm_inplace(
                    *delta,
                    *residual,
                    w.weight,
                    w.eps,
                    ctx.device.compute_stream,
                );
            },
            Instruction::FusedAddRmsNormWithOffset(delta_slot, residual_slot, layer, offset) => unsafe {
                let layer = ctx.layer_offset + layer;
                let delta = tile_ref(ctx.tiles, delta_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                let w = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                let _ = kernels::fused_add_rms_norm_inplace_with_offset(
                    *delta,
                    *residual,
                    w.weight,
                    w.eps,
                    offset,
                    ctx.device.compute_stream,
                );
            },
            Instruction::ScalarOffsetRmsNorm(in_slot, out_slot, layer, offset) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
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
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let nw = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                let gw = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
            Instruction::CutlassFusedMeanSubRmsNormGemm(
                in_slot,
                out_slot,
                layer,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let nw = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                let gw = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                assert_weight_shape(
                    "CutlassFusedMeanSubRmsNormGemm",
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
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let delta = tile_ref(ctx.tiles, delta_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                let nw = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                let gw = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
                // Last-token-per-seq narrow before lm_head GEMM. At
                // prefill, lm_head only needs the row at
                // `query_start_loc[i+1]-1` for each sequence; running
                // the GEMM at full M=bucket_m wastes ~num_tokens/num_seqs×
                // compute. Mirrors metal's `GatherLastToken` lowering
                // and Python vLLM's
                // `logits_indices = query_start_loc[1:] - 1`.
                // The gather is gated on `idx.dim(0) < bucket_m` so the
                // decode path (M=BS already) is byte-identical.
                // `LMHEAD_NARROW=0` disables (debug-only A/B).
                let narrow_disabled = matches!(
                    std::env::var("LMHEAD_NARROW").as_deref(),
                    Ok("0") | Ok("off") | Ok("false")
                );
                let gathered_owned: Option<OwnedTensor>;
                let gemm_input = match ctx.fwd.last_token_indices {
                    Some(idx) if !narrow_disabled && idx.dim(0) < normed_view.dim(0) => {
                        let owned = kernels::embedding_gather(
                            normed_view,
                            *idx,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        );
                        let view = owned.as_gpu_tensor();
                        gathered_owned = Some(owned);
                        view
                    }
                    _ => {
                        gathered_owned = None;
                        normed_view
                    }
                };
                let out = cutlass::cutlass_gemm(
                    gemm_input,
                    gw.dense_weight(),
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                drop(gathered_owned);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Gemm(in_slot, out_slot, layer, n, k) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                assert_weight_shape("Gemm", w.dense_weight(), n, k, tp_active(ctx));
                let out = ctx
                    .device
                    .cublas
                    .gemm(*v, w.dense_weight(), &mut ctx.device.caching);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::FusedCublasGemmAdd(in_slot, residual_slot, layer, n, k) => unsafe {
                // cuBLAS gemm(activation, weight) → delta; then
                // add_inplace folds delta into the residual buffer.
                // The residual upstream's OwnedTensor is aliased to
                // the Add tile's slot via the codegen prelude — same
                // as CutlassGemmAdd.
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                assert_weight_shape("FusedCublasGemmAdd", w.dense_weight(), n, k, tp_active(ctx));
                let delta = ctx
                    .device
                    .cublas
                    .gemm(*v, w.dense_weight(), &mut ctx.device.caching);
                kernels::add_inplace(*residual, delta.as_gpu_tensor(), ctx.device.compute_stream);
            },
            Instruction::FusedGemmBias(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
            Instruction::FusedGateUpSiluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
            Instruction::FusedGateUpGeluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
            Instruction::FusedQkvRopeCache(in_slot, out_slot, layer, biased, interleaved) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
                let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
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
                        W::MROPE_SECTION,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::FusedQkvQkNormRopeCache(in_slot, out_slot, layer, q_offset, k_offset) => {
                let layer = ctx.layer_offset + layer;
                let mut q_out = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    // FusedQkvQkNormRopeCache consumes 6 weights at this
                    // tape position: q_proj / k_proj / v_proj LinearLayers,
                    // q_norm / k_norm RmsNorms, and one CosSin. The per-
                    // arch impl returns each via its kind getter; the
                    // proc-macro records all 6 weight_slots in fan_out
                    // order so the per-(bucket, op_idx) match arms wire
                    // up to the right named accessors.
                    let qw = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                    let kw = ctx.wm.linear_at(bucket, op_idx, 1, layer);
                    let vw = ctx.wm.linear_at(bucket, op_idx, 2, layer);
                    let qnorm = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                    let knorm = ctx.wm.rms_norm_at(bucket, op_idx, 1, layer);
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
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
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
                biased,
                interleaved,
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
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
                            W::MROPE_SECTION,
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
            Instruction::AttentionViaCache(in_slot, out_slot, layer, interleaved) => {
                let layer = ctx.layer_offset + layer;
                let mut out = unsafe {
                    let q = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
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
                    // Cold-path gate: ONLY a chunked-prefill continuation
                    // (chunk 2+ of a prompt > max_num_batched_tokens) needs
                    // the paged read of prior K/V. That's the case when
                    // `q_len > 1` AND `q_len < seq_len` — q_len > 1 because
                    // it's a multi-token prefill chunk, q_len < seq_len
                    // because chunk 1's K/V already sits in the cache.
                    //
                    // The cost solver also picks AttentionPrefillContiguousImpl
                    // for DECODE workloads at bs>=2 (q=1, seq_len>1). Decode
                    // ran through `flash_attn_contiguous` pre-fix and worked
                    // because the K/V tiles get concatenated upstream — so the
                    // hot path below must catch decode too. A naive `q != k`
                    // gate would route every decode step through flashinfer
                    // paged attention, which is what slowed `vllm bench
                    // latency` from ~1.27s to ~1.31s before this guard.
                    if ctx.fwd.max_seqlen_q <= 1 || ctx.fwd.max_seqlen_q == ctx.fwd.max_seqlen_k {
                        // HOT path: fresh prefill (q == k) OR decode (q == 1).
                        // Byte-for-byte the pre-fix code; the cold branch's
                        // flashinfer/flash_attn_paged symbols live in a
                        // non-generic out-of-line helper, so eval<W>'s
                        // monomorphizations don't pull them into this body.
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
                    } else {
                        // COLD path: chunked-prefill continuation (chunk 2+).
                        attn_prefill_chunked_continuation(
                            ctx.fwd.kv_cache,
                            ctx.fwd.cu_seqlens_q,
                            ctx.fwd.seqused_k,
                            ctx.fwd.block_table,
                            q,
                            ctx.fwd.max_seqlen_q,
                            ctx.fwd.max_seqlen_k,
                            ctx.layer_offset as usize,
                            W::HEAD_DIM,
                            W::ATTN_SCALE,
                            W::ATTN_SOFTCAP,
                            ctx.device.num_sm,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                            interleaved,
                        )
                    }
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::AttentionPrefillPaged(_q_slot, _out_slot, _layer, _interleaved) => {
                unimplemented!(
                    "AttentionPrefillPaged is metal-only — cuda routes prefill through \
                     `Instruction::AttentionPrefillContiguous` (flash_attn_contiguous)."
                );
            }
            Instruction::EncoderAttention(q_slot, k_slot, v_slot, out_slot) => {
                // Encoder/bidirectional self-attention: same FA2 kernel
                // as the prefill prefill path but with `is_causal=false`
                // (every query attends to every key in its sequence). No
                // softcap (encoder models don't use it), no sliding
                // window (full attention), no fused RoPE (RoPE applied
                // upstream — null cos_sin pointer + rotary_dim=0 makes
                // flash-attn skip its fused rotary). cu_seqlens_q is
                // reused as cu_seqlens_k since K and V live alongside Q
                // for self-attention.
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
                        ctx.fwd.max_seqlen_q,
                        W::ATTN_SCALE,
                        false,
                        0.0,
                        -1,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                        ::std::ptr::null::<u8>(),
                        0,
                        false,
                    )
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::SlidingAttentionViaCache(in_slot, out_slot, layer, interleaved) => {
                let layer = ctx.layer_offset + layer;
                let mut out = unsafe {
                    let q = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
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
            Instruction::VarlenAttention(q_slot, k_slot, v_slot, out_slot, cu_seqlens_kind) => {
                let (cu_view, max_seqlen) = match cu_seqlens_kind {
                    0 => (ctx.fwd.cu_seqlens_q, ctx.fwd.max_seqlen_q),
                    1 => {
                        let cu = ctx.fwd.vision_cu_seqlens_full.expect(
                            "Instruction::VarlenAttention(kind=1) reached eval but \
                             ForwardCtx::vision_cu_seqlens_full is None — caller must \
                             populate it before invoking the vision interpreter",
                        );
                        let m = ctx.fwd.vision_max_seqlen_full.expect(
                            "Instruction::VarlenAttention(kind=1) reached eval but \
                             ForwardCtx::vision_max_seqlen_full is None",
                        );
                        (cu, m)
                    }
                    2 => {
                        let cu = ctx.fwd.vision_cu_seqlens_window.expect(
                            "Instruction::VarlenAttention(kind=2) reached eval but \
                             ForwardCtx::vision_cu_seqlens_window is None — caller must \
                             populate it before invoking the vision interpreter",
                        );
                        let m = ctx.fwd.vision_max_seqlen_window.expect(
                            "Instruction::VarlenAttention(kind=2) reached eval but \
                             ForwardCtx::vision_max_seqlen_window is None",
                        );
                        (cu, m)
                    }
                    other => panic!(
                        "Instruction::VarlenAttention: cu_seqlens_kind must be 0 \
                         (default) | 1 (full) | 2 (window); got {other}"
                    ),
                };
                let mut out = unsafe {
                    let q_view = tile_ref(ctx.tiles, q_slot).as_view(ctx.tiles);
                    let k_view = tile_ref(ctx.tiles, k_slot).as_view(ctx.tiles);
                    let v_view = tile_ref(ctx.tiles, v_slot).as_view(ctx.tiles);
                    let total_l = (*q_view).dim(0);
                    let nh = W::VISION_NUM_HEADS as usize;
                    let hd = W::VISION_HEAD_DIM as usize;
                    let q3 = q_view.reshape(&[total_l, nh, hd]);
                    let k3 = k_view.reshape(&[total_l, nh, hd]);
                    let v3 = v_view.reshape(&[total_l, nh, hd]);
                    kernels::flash_attn_contiguous(
                        *q3,
                        *k3,
                        *v3,
                        *cu_view,
                        *cu_view,
                        max_seqlen,
                        max_seqlen,
                        W::VISION_ATTN_SCALE,
                        false,
                        0.0,
                        -1,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                        ::std::ptr::null::<u8>(),
                        0,
                        false,
                    )
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::VISION_Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::VisionRope(q_slot, k_slot, q_out_slot, k_out_slot) => {
                let cos = ctx.fwd.vision_rope_cos.expect(
                    "Instruction::VisionRope reached eval but \
                     ForwardCtx::vision_rope_cos is None — caller must \
                     populate it before invoking the vision interpreter",
                );
                let sin = ctx.fwd.vision_rope_sin.expect(
                    "Instruction::VisionRope reached eval but \
                     ForwardCtx::vision_rope_sin is None — caller must \
                     populate it before invoking the vision interpreter",
                );
                let q_owned = take_owned(ctx.tiles, q_slot);
                let k_owned = take_owned(ctx.tiles, k_slot);
                let total_l = (*q_owned).dim(0);
                let nh = W::VISION_NUM_HEADS as usize;
                let hd = W::VISION_HEAD_DIM as usize;
                let q3 = (*q_owned).reshape(&[total_l, nh, hd]);
                let k3 = (*k_owned).reshape(&[total_l, nh, hd]);
                unsafe {
                    kernels::vision_rope_apply(q3, *cos, *sin, ctx.device.compute_stream);
                    kernels::vision_rope_apply(k3, *cos, *sin, ctx.device.compute_stream);
                }
                ctx.tiles[q_out_slot as usize] = Some(TileEntry::Owned(q_owned));
                ctx.tiles[k_out_slot as usize] = Some(TileEntry::Owned(k_owned));
            }
            Instruction::QuickGelu(in_slot, out_slot) => {
                let owned = take_owned(ctx.tiles, in_slot);
                unsafe {
                    kernels::quick_gelu_inplace(*owned, ctx.device.compute_stream);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            }
            Instruction::Gelu(in_slot, out_slot) => {
                let owned = take_owned(ctx.tiles, in_slot);
                unsafe {
                    kernels::gelu_tanh_inplace(*owned, ctx.device.compute_stream);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            }
            Instruction::PosEmbed(out_slot) => unsafe {
                let weight = ctx.wm.embedding_at(bucket, op_idx, 0, 0u32).weight;
                let position_ids = ctx.fwd.vision_position_ids.expect(
                    "Instruction::PosEmbed reached eval but \
                     ForwardCtx::vision_position_ids is None — caller \
                     (vision_forward host wrapper) must populate this \
                     view before driving the vision interpreter, \
                     mirroring the pixels / cu_seqlens / cos / sin \
                     contract",
                );
                let num_positions = weight.dim(0) as u32;
                // No tp sharding on the vision positional table —
                // vision is replicated per-rank; vocab_offset = 0 and
                // vocab_per_rank covers the full table so the mask
                // arm in `embedding_gather_masked` never trips.
                let out = kernels::embedding_gather_masked(
                    weight,
                    *position_ids,
                    0u32,
                    num_positions,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::GeluErf(in_slot, out_slot) => {
                let owned = take_owned(ctx.tiles, in_slot);
                unsafe {
                    kernels::gelu_erf_inplace(*owned, ctx.device.compute_stream);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            }
            Instruction::EmbeddingGather(in_slot, out_slot, indices_kind) => {
                let indices_view = match indices_kind {
                    0 => ctx.fwd.vision_window_index.expect(
                        "Instruction::EmbeddingGather(kind=0) reached eval but \
                         ForwardCtx::vision_window_index is None — caller must \
                         populate it before invoking the vision interpreter",
                    ),
                    1 => ctx.fwd.vision_reverse_indices.expect(
                        "Instruction::EmbeddingGather(kind=1) reached eval but \
                         ForwardCtx::vision_reverse_indices is None — caller must \
                         populate it before invoking the vision interpreter",
                    ),
                    other => panic!(
                        "Instruction::EmbeddingGather: indices_kind must be 0 \
                         (window_index) or 1 (reverse_indices); got {other}"
                    ),
                };
                let owned = unsafe {
                    let in_view = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    kernels::embedding_gather(
                        *in_view,
                        *indices_view,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            }
            Instruction::AvgPool2d(in_slot, out_slot) => {
                let ph = W::VISION_PATCH_GRID_SIDE;
                let k = W::VISION_POOL_KERNEL;
                if ph == 0 || k == 0 {
                    panic!(
                        "Instruction::AvgPool2d reached eval with \
                         VISION_PATCH_GRID_SIDE={ph} VISION_POOL_KERNEL={k} — \
                         the per-arch CanonicalParams bake must populate both \
                         (vision_patch_grid_side / vision_pool_kernel bounds in \
                         configs/<variant>.json)",
                    );
                }
                let owned = unsafe {
                    let in_view = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    kernels::avg_pool_2d(
                        *in_view,
                        ph,
                        k,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            }
            Instruction::LoadPixels(out_slot) => unsafe {
                let view = ctx.fwd.pixels.expect(
                    "Instruction::LoadPixels invoked without ForwardCtx::pixels — \
                     caller (vision_forward host wrapper) must populate this view \
                     before driving the vision interpreter, mirroring the \
                     vision_rope_cos / vision_rope_sin contract",
                );
                let raw = view.as_raw();
                let shape: Vec<usize> = raw.shape().iter().map(|&d| d as usize).collect();
                let owned = ctx.device.caching.alloc_tensor(&shape, raw.dtype());
                let bytes = raw.size_bytes();
                ferrite_cuda_core::driver::memcpy_dtod_async(
                    (*owned).raw_ptr(),
                    raw.raw_ptr(),
                    bytes,
                    ctx.device.compute_stream,
                )
                .expect("LoadPixels D2D copy");
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            },
            Instruction::FlashInferAttentionDecode(
                in_slot,
                out_slot,
                layer,
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
                    // The TP>1 path runs through `AttentionViaCacheImpl`
                    // (FA2) on sm<90 and `FlashAttention3DecodeImpl` (FA3)
                    // on sm>=90 — both graph-capture-safe. The solver
                    // doesn't emit `Instruction::FlashInferAttentionDecode`
                    // when `tp_world_size > 1` (gated in
                    // `FlashInferAttentionDecodeImpl::applies_to`), so any
                    // execution of this arm is a TP=1 path where FI's
                    // cooperative persistent kernel is the right choice.
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
                            let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
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
            #[cfg(fa3_built)]
            Instruction::FlashAttention3Decode(in_slot, out_slot, layer, _head_dim) => {
                // Solver-selected: this Instruction is emitted only when
                // `FlashAttention3DecodeImpl::target_compatible()` passed
                // (sm_90+, head_dim==128, no softcap, FA3 cost-CSV row
                // present). No fallback inside — if FA3 dispatch fails,
                // the bug is at the Impl/cost-CSV level, not runtime.
                if std::env::var("FERRITE_FA3_TRACE").ok().as_deref() == Some("1") {
                    use std::sync::atomic::{AtomicBool, Ordering};
                    static ANNOUNCED: AtomicBool = AtomicBool::new(false);
                    if !ANNOUNCED.swap(true, Ordering::Relaxed) {
                        eprintln!("[FA3] solver picked Instruction::FlashAttention3Decode");
                    }
                }
                let step_layer = layer as usize;
                let layer = ctx.layer_offset + layer;
                let mut out = unsafe {
                    let q = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    ah::flash_attn_3_decode(
                        q,
                        ctx.fwd.cu_seqlens_q,
                        ctx.fwd.seqused_k,
                        ctx.fwd.block_table,
                        ctx.fwd.max_seqlen_q,
                        ctx.fwd.max_seqlen_k,
                        W::ATTN_SCALE,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        step_layer,
                        ctx.device.num_sm,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
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
                    // Solver-gated to TP=1 in
                    // `FlashInferAttentionPrefillImpl::applies_to`. See
                    // FlashInferAttentionDecode arm above for the full
                    // explanation.
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
                interleaved,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
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
                // Pin any Owned overwritten by these three Reshaped
                // writes. See InterpreterCtx::pinned_owned for the why.
                for slot in [q_out_slot, k_out_slot, v_out_slot] {
                    let prev = std::mem::take(&mut ctx.tiles[slot as usize]);
                    if let Some(TileEntry::Owned(t)) = prev {
                        ctx.pinned_owned.push(t);
                    }
                }
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
            Instruction::MlaAttention(q_slot, kv_b_slot, k_pe_slot, out_slot, layer) => {
                let layer = ctx.layer_offset + layer;
                let out = unsafe {
                    mla_attention_eval(ctx, q_slot, kv_b_slot, k_pe_slot, layer, bucket, op_idx)
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::DeepSeekMoe(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.deepseek_moe_at(bucket, op_idx, 0, layer);
                let out = w.forward(v, ctx.device);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::DeepSeekMoeFp8Block(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.deepseek_moe_fp8_at(bucket, op_idx, 0, layer);
                let out = w.forward(v, ctx.device);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::DeepSeekMoeGgml(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.deepseek_moe_ggml_at(bucket, op_idx, 0, layer);
                let out = w.forward(v, ctx.device);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::FusedMoe(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.fused_moe_at(bucket, op_idx, 0, layer);
                let out = w.forward(v, ctx.device);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::SharedFusedMoe(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.shared_fused_moe_at(bucket, op_idx, 0, layer);
                let out = w.forward(v, ctx.device);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassGemm(in_slot, out_slot, layer, tile_m, tile_n, stages, n, k) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
                tile_m,
                tile_n,
                stages,
                split_k,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                assert_weight_shape("CutlassGemmAdd", w.dense_weight(), n, k, tp_active(ctx));
                cutlass::cutlass_gemm_add(
                    *v,
                    w.dense_weight(),
                    *residual,
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    ctx.device.compute_stream,
                );
            },
            Instruction::CutlassGemv(in_slot, out_slot, layer, n, k) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
                tile_m,
                tile_n,
                stages,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
                let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
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
                        W::MROPE_SECTION,
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
                    let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
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
                            W::MROPE_SECTION,
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
            Instruction::MarlinGemm(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.marlin_at(bucket, op_idx, 0, layer);
                let out = w.forward(v, &mut ctx.device.caching, ctx.device.compute_stream);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::MarlinFusedGateUpSiluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.marlin_at(bucket, op_idx, 0, layer);
                let gate_up = w.forward(v, &mut ctx.device.caching, ctx.device.compute_stream);
                let out = kernels::silu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::MarlinFusedGateUpGeluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.marlin_at(bucket, op_idx, 0, layer);
                let gate_up = w.forward(v, &mut ctx.device.caching, ctx.device.compute_stream);
                let out = kernels::gelu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::MarlinFusedQkvRopeCache(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.marlin_at(bucket, op_idx, 0, layer);
                let qkv_packed = w.forward(v, &mut ctx.device.caching, ctx.device.compute_stream);
                let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
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
                        W::MROPE_SECTION,
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
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = ctx.wm.marlin_at(bucket, op_idx, 0, layer);
                    let qkv_packed =
                        w.forward(view_in, &mut ctx.device.caching, ctx.device.compute_stream);
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                    kernels::fused_qkv_rope(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::NUM_KV_HEADS as usize,
                        W::HEAD_DIM as usize,
                        W::MROPE_SECTION,
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
            Instruction::GgmlGemm(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                let out = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::GgmlFusedGateUpSiluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
            Instruction::GgmlFusedGateUpGeluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
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
            Instruction::GgmlFusedQkvRopeCache(in_slot, out_slot, layer, interleaved) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                let qkv_packed = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
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
                        W::MROPE_SECTION,
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
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                    let qkv_packed = w.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                    kernels::fused_qkv_rope(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::NUM_KV_HEADS as usize,
                        W::HEAD_DIM as usize,
                        W::MROPE_SECTION,
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
            Instruction::Bnb4Gemm(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.bnb4_at(bucket, op_idx, 0, layer);
                let out = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Bnb4FusedGateUpSiluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.bnb4_at(bucket, op_idx, 0, layer);
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
            Instruction::Bnb4FusedGateUpGeluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.bnb4_at(bucket, op_idx, 0, layer);
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
            Instruction::Bnb4FusedQkvRopeCache(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.bnb4_at(bucket, op_idx, 0, layer);
                let qkv_packed = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
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
                        W::MROPE_SECTION,
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
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = ctx.wm.bnb4_at(bucket, op_idx, 0, layer);
                    let qkv_packed = w.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                    kernels::fused_qkv_rope(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::NUM_KV_HEADS as usize,
                        W::HEAD_DIM as usize,
                        W::MROPE_SECTION,
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
            Instruction::Fp8Gemm(in_slot, out_slot, layer)
            | Instruction::Fp8FusedGemmBias(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.fp8_at(bucket, op_idx, 0, layer);
                let out = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Fp8FusedGateUpSiluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.fp8_at(bucket, op_idx, 0, layer);
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
            Instruction::Fp8FusedGateUpGeluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.fp8_at(bucket, op_idx, 0, layer);
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
            Instruction::Fp8FusedQkvRopeCache(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.fp8_at(bucket, op_idx, 0, layer);
                let qkv_packed = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
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
                        W::MROPE_SECTION,
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
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = ctx.wm.fp8_at(bucket, op_idx, 0, layer);
                    let qkv_packed = w.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                    kernels::fused_qkv_rope(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::NUM_KV_HEADS as usize,
                        W::HEAD_DIM as usize,
                        W::MROPE_SECTION,
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
            Instruction::AffineQmm(..) => {
                unreachable!(
                    "Instruction::AffineQmm is metal-only — the macro must \
                     not emit it on the cuda forward (Affine weights stay \
                     in StorageFormat::Dense on cuda by the FUF downgrade)"
                );
            }
            Instruction::Nvfp4Qmm(..) => {
                unreachable!(
                    "Instruction::Nvfp4Qmm is metal-only — the macro must \
                     not emit it on the cuda forward (NVFP4 weights stay \
                     in StorageFormat::Dense on cuda by the FUF downgrade)"
                );
            }
            Instruction::SynthPreAttn(..) => {
                unreachable!(
                    "Instruction::SynthPreAttn is metal-only — emitted by the \
                     compiler-driven megakernel synthesis pass on the metal forward only"
                );
            }
            Instruction::SynthMlpPreDown(..) => {
                unreachable!(
                    "Instruction::SynthMlpPreDown is metal-only — emitted by the \
                     compiler-driven megakernel synthesis pass on the metal forward only"
                );
            }
            Instruction::SiluMul(..) => {
                unreachable!(
                    "Instruction::SiluMul is metal-only — emitted by the \
                     decomposed q-MLP path on Affine; cuda's q-MLP routes \
                     through Marlin/Bnb/Fp8/etc fused kernels"
                );
            }
            Instruction::MetalBiasAdd(..) => {
                unreachable!(
                    "Instruction::MetalBiasAdd is metal-only — emitted by \
                     MetalBiasAddImpl for QKV biases on the singleton \
                     (non-synth) path. cuda folds biases into cuBLAS \
                     gemm_bias via FusedGemmBias instead"
                );
            }
            Instruction::MetalFusedMoe(..) | Instruction::MetalSharedFusedMoe(..) => {
                unreachable!(
                    "Instruction::Metal{{,Shared}}FusedMoe is metal-only — \
                     emitted by Metal{{,Shared}}FusedMoeImpl carrying macro-baked \
                     MoE shape for the Metal lowering arm. cuda's MoE path emits \
                     Instruction::{{,Shared}}FusedMoe and reads shape from the \
                     runtime FusedMoELayer / SharedFusedMoELayer struct"
                );
            }
            #[cfg(feature = "metal")]
            Instruction::SynthGateUpSiluMul(..) => {
                unreachable!(
                    "Instruction::SynthGateUpSiluMul is metal-only — emitted by the \
                     compiler-driven megakernel synthesis pass on the metal forward only"
                );
            }
            #[cfg(feature = "metal")]
            Instruction::AffineEmbed(..) => {
                unreachable!(
                    "Instruction::AffineEmbed is metal-only — emitted by the \
                     P6 quantized embedding lift; cuda's quantized embeddings \
                     never lift to forward-time (Marlin/Bnb keep dense embed)"
                );
            }
            Instruction::StripCls(_, _) => {
                unimplemented!("Instruction::StripCls eval is unwired");
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

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
unsafe fn mla_attention_eval<W: CanonicalParams>(
    ctx: &mut InterpreterCtx<'_, W>,
    q_slot: u32,
    kv_b_slot: u32,
    k_pe_slot: u32,
    layer: u32,
    bucket: u32,
    op_idx: u32,
) -> OwnedTensor {
    unsafe {
        let q_tv = tile_ref(ctx.tiles, q_slot).as_view(ctx.tiles);
        let kv_b_tv = tile_ref(ctx.tiles, kv_b_slot).as_view(ctx.tiles);
        let k_pe_tv = tile_ref(ctx.tiles, k_pe_slot).as_view(ctx.tiles);
        let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
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
///
/// `bucket` is the static bucket id (e.g. `BACKBONE`, `LM_HEAD`)
/// the proc-macro emits; it is forwarded together with each op's
/// flat position in the bucket to `Instruction::eval`, which
/// passes both to the per-arch [`WeightAccessors`] impl.
///
/// Loop bodies use their absolute position in the bucket (not the
/// position within the body), so the per-arch match table is keyed
/// uniformly on `(bucket, flat_position)` regardless of whether
/// the op sits inside a `Loop` body.
#[cfg(feature = "cuda")]
unsafe fn run_slice<W: CanonicalParams>(
    instructions: &[Instruction],
    bucket: u32,
    ctx: &mut InterpreterCtx<'_, W>,
) {
    let dump = debug_dump::DumpHook::from_env();
    let mut i = 0usize;
    while i < instructions.len() {
        match instructions[i] {
            Instruction::Loop(count, body_len) => {
                let body_start = i + 1;
                let body_end = body_start + body_len as usize;
                let body = &instructions[body_start..body_end];
                for l in 0..count {
                    ctx.layer_offset = l;
                    for (j, instr) in body.iter().enumerate() {
                        let body_op_idx = (body_start + j) as u32;
                        unsafe {
                            instr.eval(ctx, bucket, body_op_idx);
                        }
                        if let Some(d) = dump.as_ref() {
                            d.dump(
                                &format!("loop_iter_{}_body_{}", l, j),
                                ctx.tiles,
                                ctx.device.compute_stream,
                            );
                        }
                    }
                }
                ctx.layer_offset = 0;
                i = body_end;
            }
            instr => {
                unsafe {
                    instr.eval(ctx, bucket, i as u32);
                }
                if let Some(d) = dump.as_ref() {
                    d.dump(
                        &format!("instr_{}", i),
                        ctx.tiles,
                        ctx.device.compute_stream,
                    );
                }
                i += 1;
            }
        }
    }
}

/// Per-instruction hidden-state dumper.
///
/// Activated by `FERRITE_DEBUG_DUMP_PATH=<file>`. After each
/// `Instruction::eval`, copies the first row of every Owned tile slot
/// to host, casts BF16/F16 → F32, and appends one JSONL row to the
/// path: `{"label": "...", "slot": N, "shape": [..], "first8":
/// [..]}`. Buffers are line-buffered so a process kill mid-forward
/// still yields usable diagnostics.
///
/// Cost: one D2H + stream sync per slot per instruction. Useful
/// only for single-request bisection runs; never enable in
/// production.
#[cfg(feature = "cuda")]
mod debug_dump {
    use super::TileEntry;
    use ferrite_cuda_core::CUstream;
    use ferrite_cuda_core::driver::{memcpy_dtoh_async, stream_synchronize};
    use ferrite_cuda_core::dtype::DType;
    use std::cell::RefCell;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::path::PathBuf;

    pub struct DumpHook {
        file: RefCell<std::fs::File>,
    }

    impl DumpHook {
        pub fn from_env() -> Option<Self> {
            let path: PathBuf = std::env::var_os("FERRITE_DEBUG_DUMP_PATH")?.into();
            let f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .ok()?;
            Some(Self {
                file: RefCell::new(f),
            })
        }

        pub fn dump(&self, label: &str, tiles: &[Option<TileEntry>], stream: CUstream) {
            // Sync first to ensure prior async ops have completed.
            unsafe {
                let _ = stream_synchronize(stream);
            }
            for (slot, entry) in tiles.iter().enumerate() {
                let Some(e) = entry else { continue };
                // Only dump Owned tiles — Views alias another slot,
                // and Reshaped is a metadata-only rewrap. Skipping
                // them avoids double-dumping the same storage.
                let TileEntry::Owned(t) = e else { continue };
                let gt = t.as_gpu_tensor();
                if gt.ndim() < 1 {
                    continue;
                }
                // First-row size in bytes.
                let row_elems: usize = (1..gt.ndim()).map(|d| gt.dim(d)).product();
                let row_elems = row_elems.max(1);
                let take = row_elems.min(64);
                let dt = gt.dtype();
                let bytes_per = dt.size_bytes();
                let buf_len = take * bytes_per;
                let mut host: Vec<u8> = vec![0u8; buf_len];
                unsafe {
                    let _ = memcpy_dtoh_async(host.as_mut_ptr(), gt.as_ptr(), buf_len, stream);
                    let _ = stream_synchronize(stream);
                }
                let first8: Vec<f32> = match dt {
                    DType::F32 => host
                        .chunks_exact(4)
                        .take(take)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect(),
                    // BF16 → F32: pad with two zero LSBs to widen to a
                    // 32-bit IEEE float. Lossless because BF16 shares
                    // F32's 8-bit exponent.
                    DType::BF16 => host
                        .chunks_exact(2)
                        .take(take)
                        .map(|b| f32::from_le_bytes([0, 0, b[0], b[1]]))
                        .collect(),
                    // F16 → F32: requires real conversion (different
                    // exponent width). Skip for the dump — every arch
                    // we care about uses BF16/F32.
                    _ => continue,
                };
                let shape: Vec<usize> = (0..gt.ndim()).map(|d| gt.dim(d)).collect();
                let line = format!(
                    "{{\"label\":\"{label}\",\"slot\":{slot},\"shape\":{shape:?},\"first8\":{first8:?}}}\n"
                );
                if let Ok(mut f) = self.file.try_borrow_mut() {
                    let _ = f.write_all(line.as_bytes());
                }
            }
        }
    }
}

/// Run backbone followed by lm_head against one tile table.
///
/// # Safety
/// Both slices well-formed; tile slot indices in range; weight
/// accessor fns produce live GPU memory.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn run<W: CanonicalParams>(
    backbone: &[Instruction],
    backbone_bucket: u32,
    lm_head: &[Instruction],
    lm_head_bucket: u32,
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
        pinned_owned: Vec::new(),
    };
    unsafe {
        run_slice(backbone, backbone_bucket, &mut ctx);
        run_slice(lm_head, lm_head_bucket, &mut ctx);
    }
    take_owned(&mut tiles, terminal_slot)
}

/// Backbone-only run: returns a memcpy'd backbone tile so it
/// outlives the per-call tile table.
///
/// # Safety
/// Same as [`run`].
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_backbone<W: CanonicalParams>(
    backbone: &[Instruction],
    backbone_bucket: u32,
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
        pinned_owned: Vec::new(),
    };
    unsafe {
        run_slice(backbone, backbone_bucket, &mut ctx);
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
