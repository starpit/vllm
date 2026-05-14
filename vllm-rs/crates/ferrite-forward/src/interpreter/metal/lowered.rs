// SPDX-License-Identifier: Apache-2.0
//! Buffer-pointer-free Metal tape produced by the lowering pass.
//!
//! `LoweredMetalTape` is computed once per `(model variant, bucket)` and
//! shared across every `MetalWorker` in the pool via `Arc`. It carries
//! pipeline keys, dispatch shapes, scalar constants, and slot ids only —
//! never raw `metal::Buffer` pointers. Each `MetalWorker` instantiates
//! the tape against its own arena at worker init time, resolving
//! `Binding::ArenaSlot` against `arena[slot]` and recording the result
//! into a private `IndirectCommandBuffer`.
//!
//! Per the architecture doc (`FERRITE_METAL_ARCHITECTURE.md` §1, §2):
//! the lowered tape is structurally a sequence of `LoweredCommand`s,
//! with `Loop` instructions already statically unrolled by the lowering
//! pass (CUDA's runtime `Loop` interpreter has no analogue on Metal —
//! the per-bucket ICB is fully baked).

use crate::CanonicalParams;
use ferrite_metal_kernels::specialized_pipeline_cache::ConstantValue;

/// One-of identifier for the kernel a `LoweredCommand` invokes.
///
/// The `MetalWorker` resolves `(KernelId, bucket)` against a
/// `SpecializedPipelineCache` (Phase 5.B) to find the right
/// `MTLComputePipelineState`. Each variant corresponds to one Metal
/// kernel under `crates/ferrite-metal-kernels/shaders/` and one
/// recorder under `crates/ferrite-metal-kernels/src/instruction_executor/`.
///
/// The set is closed and small on purpose: the TinyLlama-1.1B critical
/// path covers ~13 of these, with more added as additional models come
/// online. Variants the lowering pass cannot yet produce surface as a
/// `LoweringError::UnsupportedVariant` rather than appearing here as a
/// stub.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelId {
    /// Token embedding gather: rows of `embed_tokens.weight` indexed
    /// by `input_ids`. Output `[num_tokens, hidden_size]`.
    Embed,
    /// Standalone RMSNorm: `out = weight * x / sqrt(mean(x²) + eps)`.
    RmsNorm,
    /// Fused residual-add + RMSNorm: writes `residual += delta` and
    /// publishes `weight * residual / sqrt(mean(residual²) + eps)`.
    FusedAddRmsNorm,
    /// Generic dense GEMM: `y = x @ W^T`. Backed by Metal Performance
    /// Shaders' `matmul2d` (or a hand-rolled tile shader once the
    /// fused kernels land).
    Gemm,
    /// Fused gate-up SwiGLU MLP: `silu(gate) * up` after a single GEMM
    /// produces the stacked `[gate; up]` activation. TinyLlama-1.1B's
    /// MLP path.
    FusedGateUpSiluMul,
    /// Apply RoPE to a fresh QKV projection and write K/V to the
    /// paged KV cache at the per-request slot. Output: rotated Q
    /// only (K/V are sunk into cache).
    RopeAppend,
    /// Fused QKV matmul + NeoX-style RoPE + paged KV-cache write in
    /// one kernel. Replaces the four-dispatch
    /// `Q_proj + K_proj + V_proj + RopeAppend` chain on the dense
    /// (BF16 / F16) path. Affine-int4 / prefill variants land
    /// separately per `project_metal_fused_qkv_handoff`. Maps to
    /// `fused_qkv_rope_cache_<dtype>_specialized` in
    /// `fused_qkv_rope_cache.metallib`.
    FusedQkvRopeCache,
    /// Affine-int4 sibling of [`KernelId::FusedQkvRopeCache`]. Reads
    /// packed `u32` weights + per-group F16 scales/biases (mlx-community
    /// 4bit layout) for Q/K/V concatenated along the output axis, fuses
    /// the dequant→matmul→RoPE→paged-cache-write chain in a single
    /// launch. Maps to
    /// `fused_affine_qkv_rope_cache_<dtype>_s_<scale_dtype>_b_4_specialized`
    /// in `fused_affine_qkv_rope_cache.metallib`. Group size rides on
    /// function constant 7.
    FusedAffineQkvRopeCache,
    /// Decode-bucket attention reading from the paged KV cache.
    /// Single-query-token-per-sequence path.
    AttentionViaCache,
    /// Prefill-bucket attention reading from the paged KV cache.
    /// Faithful MLX `sdpa_vector` port: 1 Q per threadgroup,
    /// online softmax + per-simdgroup K-axis split (same outer
    /// structure as the decode kernel `AttentionViaCache`), with K/V
    /// access through `block_table` indirection. K-axis covers the
    /// FULL `seqused_k[seq]` (prefix + new), and the per-Q causal
    /// mask shifts by `(seqused_k[seq] - new_q_for_seq)` to account
    /// for prior cached prefix. Required for chunked prefill,
    /// prefix caching, mixed prefill/decode batches, and multi-turn
    /// chat continuation. The only prefill kernel emitted by metal
    /// post-Phase B (the legacy contiguous and non-paged sdpa
    /// kernels were retired once `Instruction::AttentionPrefillPaged`
    /// became the universal metal prefill emitter).
    AttentionPrefillSdpaPaged,
    /// Pure scalar broadcast multiply: `out = x * scale`.
    ScalarMul,
    /// Elementwise residual add: `lhs += rhs`. Output is the lhs slot
    /// rebound (in-place semantics).
    Add,
    /// Static reshape: rebinds a slot to a fresh logical shape; does
    /// not touch device memory. Lowering treats this as a metadata
    /// op — no `LoweredCommand` is emitted, only the slot's logical
    /// shape registers in the dispatcher.
    /// (Present in this enum for symmetry / future zero-copy ops.)
    Reshape,
    /// MLX-affine int4 decode matvec, K∈{64,128} ∧ pow2 bits.
    /// Maps to `affine_qmv_quad_<dtype>_gs_<gs>_b_4_d_<K>_batch_<batched>`
    /// in `quantized_qmv.metallib`. Faithful port of MLX's
    /// `affine_qmv_quad` (`quantized.h:1444`).
    AffineQmvQuad,
    /// MLX-affine int4 decode matvec, `N % 8 == 0 ∧ K % 512 == 0`.
    /// Maps to `affine_qmv_fast_<dtype>_gs_<gs>_b_4_batch_<batched>`.
    /// Faithful port of MLX's `affine_qmv_fast` (`quantized.h:1496`).
    AffineQmvFast,
    /// MLX-affine int4 decode matvec, generic shape fallback.
    /// Maps to `affine_qmv_<dtype>_gs_<gs>_b_4_batch_<batched>`.
    /// Faithful port of MLX's `affine_qmv` (`quantized.h:1548`).
    AffineQmv,
    /// MLX-affine int4 prefill matmul, transpose=true. Maps to
    /// `affine_qmm_t_<dtype>_gs_<gs>_b_4_alN_<bool>_batch_0` in
    /// `quantized_qmm.metallib`. Faithful port of MLX's
    /// `affine_qmm_t` (`quantized.h:1707`).
    AffineQmmT,
    /// MLX-affine int4 prefill matmul, transpose=true, split-K
    /// variant for small-M / B=1 shapes. Maps to
    /// `affine_qmm_t_splitk_<dtype>_gs_<gs>_b_4_alN_<bool>`. Faithful
    /// port of MLX's `affine_qmm_t_splitk` (`quantized.h:1780`).
    /// Downstream sum-reduce across the split_k partition axis is
    /// emitted by the lowering pass (mirroring
    /// `quantized.cpp:861 strided_reduce_general_dispatch`).
    AffineQmmTSplitK,
    /// NAX (Apple9 / M4+) prefill matmul — 64×64×64 MPP matmul2d tile.
    /// Maps to `affine_qmm_t_nax_<dtype>_gs_<gs>_b_4_alN_<bool>_batch_0`
    /// in `quantized_qmm_nax.metallib`. Only dispatched when
    /// `is_nax_capable(profile.generation)` and `K % 64 == 0`.
    AffineQmmTNax,
    /// Fused `silu(gate) * up` for the decomposed q-MLP path. The
    /// macro emits this after a pair of `AffineQmm` GEMMs when the
    /// gate/up Linears are MLX-affine quantized (plan P12 branch
    /// (i)). Maps to `silu_mul_<dtype>` in `silu_mul.metallib`.
    SiluMul,
    /// Sum-along-axis-0 reduce for the `[split_k, M, N]` intermediate
    /// `AffineQmmTSplitK` produces. Maps to
    /// `splitk_reduce_sum_<dtype>` in `quantized_splitk_reduce.metallib`.
    /// Lowered alongside `AffineQmmTSplitK` so the worker sees
    /// (qmm_t_splitk → scratch, reduce → out) as adjacent commands.
    SplitKReduceSum,
    /// MLX-affine int4 quantized embedding lookup: gather + dequant
    /// in one pass. Maps to
    /// `affine_embed_<dtype>_gs_<gs>_b_4` in
    /// `quantized_dequantize.metallib`. Faithful port of MLX's
    /// `nn.QuantizedEmbedding.__call__`
    /// (`python/mlx/nn/layers/quantized.py:144`).
    AffineEmbed,
    /// Compiler-synthesized pre-attention megakernel. Symbol resolves
    /// against a per-arch source-compiled library registered at worker
    /// init via `SpecializedPipelineCache::register_source_library`.
    /// Kernel body is generated at macro-expansion time by
    /// `ferrite-forward-macro::fuse_pass`.
    SynthPreAttn,
    /// Compiler-synthesized MLP pre-down megakernel. Symbol resolves
    /// against a per-arch source-compiled library registered at worker
    /// init via `SpecializedPipelineCache::register_source_library`.
    /// Kernel body is generated at macro-expansion time by
    /// `ferrite-forward-macro::fuse_pass::synthesize_mlp_pre_down_chunk`.
    /// Fuses `FusedAddRmsNorm + gate AffineQmv + up AffineQmv + SiluMul`
    /// into one dispatch; the standalone `AffineQmm` down_proj
    /// instruction follows immediately and consumes the device-buffer
    /// `silu_mul` output.
    SynthMlpPreDown,
    /// Fused gate+up GEMM + SiluMul large-M prefill kernel.
    SynthGateUpSiluMul,
    /// MLX `softmax_single_row` (precise variant). Used by the MoE
    /// router: `mx.softmax(gate(x), axis=-1, precise=True)` over
    /// `[N, num_experts]`. Maps to `softmax_<dtype>_specialized`
    /// in `softmax.metallib`. Faithful port of MLX
    /// `softmax_single_row_precise` (`mlx/.../softmax.metal`).
    Softmax,
    /// MLX `block_sort` in ARG_SORT mode, top-k along axis=-1.
    /// Used by the MoE router: `mx.argpartition(probs, kth=-k)[..., -k:]`
    /// over `[N, num_experts]` → `[N, top_k]` u32 indices. Maps to
    /// `c_arg_block_sort_<dt>_uint32_bn{32,64,128}_tn4`. Faithful
    /// port of MLX `block_sort` (`mlx/.../sort/sort.metal`).
    ArgPartitionTopK,
    /// MLX `gather_axis` specialized to 2D contiguous, axis=-1. Used
    /// by the MoE router to read top-k scores given top-k indices:
    /// `scores = mx.take_along_axis(probs, inds, axis=-1)`.
    TakeAlongAxis,
    /// MLX-affine int4 MoE per-expert matvec, transpose=true.
    /// `K%512==0` fast variant. Maps to
    /// `affine_gather_qmv_fast_<dt>_s_<scale>_gs_<gs>_b_4` in
    /// `quantized_qmv.metallib`. Used by the MoE decomposition:
    /// `mx.gather_qmm(x, W_expert, scales, biases, rhs_indices=inds,
    /// transpose=True)` over `(token, slot)` flat rows.
    AffineGatherQmvFast,
    /// MLX-affine int4 MoE per-expert matvec, transpose=true,
    /// generic K (non-multiple-of-512) variant. Maps to
    /// `affine_gather_qmv_<dt>_s_<scale>_gs_<gs>_b_4`.
    AffineGatherQmv,
    /// Plain row-gather: `out[m, d] = src[idx[m] / divisor, d]`.
    /// `divisor=1` is `_scatter_unsort` (`indices_flat[order]` and
    /// `x[inv_order]`); `divisor=top_k` is the SwitchGLU
    /// `x.flatten(0,-3)[order // K]` from `switch_layers.py:17`.
    /// Maps to `row_gather_<dtype>` in `row_gather.metallib`.
    RowGather,
    /// MoE final reduction: `out[n, d] = sum_k(expert[n, k, d] * scores[n, k])`.
    /// MLX expression: `(y * scores[..., None]).sum(axis=-2)` from
    /// qwen3_moe.py:137. One thread per (n, d) output element.
    /// Maps to `moe_weighted_sum_<dtype>` in `moe_weighted_sum.metallib`.
    MoeWeightedSum,
    /// Take trailing `top_k` columns of a 2D `[N, src_cols]` u32 buffer
    /// into a contiguous `[N, top_k]` u32 buffer. MLX expresses this as
    /// a strided view (`mx.argpartition(...)[..., -k:]` at
    /// qwen3_moe.py:131); ferrite-metal materializes it because MoE
    /// scratch regions are flat byte ranges. Maps to
    /// `slice_trailing_cols_uint32` in `slice_trailing_cols_u32.metallib`.
    SliceTrailingColsU32,
    /// In-place L1 row-renormalize on `[N, top_k]` T_act scores:
    /// `scores[n, k] /= sum(scores[n, :])`. Implements
    /// `norm_topk_prob` from qwen3_moe.py:134. One thread per row.
    /// Maps to `top_k_renormalize_<dtype>` in `top_k_renormalize.metallib`.
    TopKRenormalize,
}

/// Element dtype the metal pipeline should pick. The shader source
/// contains both `_f16_specialized` and `_bf16_specialized` symbols
/// per kernel; the lowering arms in `lowering::lower_one` consult
/// `W::METAL_DTYPE` to pick the right symbol when populating
/// [`LoweredCommand::function`].
///
/// Llama-3.x ships bf16 on disk; the cuda backend runs them in bf16
/// natively, and Apple Silicon (M3+) has hardware bf16 MMA. Casting
/// to fp16 — which the metal backend did originally — clips
/// exponent range and accumulates into nonsense output across deep
/// layer chains (28 layers for Llama-3.2-3B).
///
/// `Int4` is a placeholder for the upcoming AWQ / GPTQ dequant path
/// (group-wise int4 weights with bf16 scales / zeros). When that
/// lands the dtype routes through this enum just like bf16 does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MetalDtype {
    F16,
    Bf16,
    /// Reserved — int4 quantized weight path. Not yet wired through
    /// the lowering kernel-symbol pickers; placed here so callers
    /// can already speak in `MetalDtype` terms.
    Int4,
}

impl MetalDtype {
    /// `_f16_specialized` / `_bf16_specialized` infix used by every
    /// dtype-parameterized shader symbol. Centralized so symbol naming
    /// stays consistent across the on-path kernels.
    pub fn symbol_infix(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Int4 => "int4",
        }
    }
}

/// Per-axis dispatch grid: threadgroup count + threads per group.
///
/// The lowering pass computes both from `(bucket M, kernel-specific
/// tile dims)`. Kept as `(u32, u32, u32)` rather than Metal's `MTLSize`
/// so this type stays available without the `metal` crate (lowering is
/// pure CPU code).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchShape {
    /// (x, y, z) threadgroup count — baseline computed against
    /// `bucket_m` at lowering time.
    pub threadgroups: (u32, u32, u32),
    /// (x, y, z) threads per threadgroup.
    pub threads_per_threadgroup: (u32, u32, u32),
    /// If `Some`, the runtime rewrites the m-axis count using
    /// `actual_num_tokens` instead of `bucket_m`. The baked
    /// `threadgroups` field still holds the bucket_m-based count;
    /// runtime computes `axis_count = num_tokens.div_ceil(tile)` and
    /// patches `tg.{axis}` immediately before `dispatchThreadgroups`.
    ///
    /// Without this, GEMM-class kernels over-dispatch by up to 8×
    /// when actual_M sits at the low end of a wide bucket
    /// (e.g. bucket_m=4096 servicing num_tokens=1024) — every spare
    /// threadgroup pays the full per-tile compute cost on garbage
    /// rows in the arena past `num_tokens`.
    pub m_scaling: Option<MScaling>,
}

/// Tells the runtime how to shrink the dispatch grid for the actual
/// `num_tokens` of this forward pass. `axis` is 0/1/2 for x/y/z;
/// `bucket_m` is the bucket-M the baseline `threadgroups` count was
/// computed against.
///
/// Runtime formula:
///
/// ```text
/// new_count = ceil(baseline_count * num_tokens / bucket_m)
/// ```
///
/// Equivalent to "scale this axis proportionally with M". Works for
/// every kernel — tile-based GEMM grids
/// (`(n_tiles, ceil(M/tile), …)`), per-row dispatches
/// (`(M, n_heads, …)`), and 1D-over-`M*K` elementwise kernels alike
/// — because each is linear in M and the baseline is just that
/// linear function evaluated at `bucket_m`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MScaling {
    pub axis: u8,
    pub bucket_m: u32,
}

impl DispatchShape {
    /// 1D dispatch helper: `total_threads` rounded up by
    /// `threads_per_group`.
    pub fn dispatch_1d(total_threads: u32, threads_per_group: u32) -> Self {
        let groups = total_threads.div_ceil(threads_per_group);
        Self {
            threadgroups: (groups, 1, 1),
            threads_per_threadgroup: (threads_per_group, 1, 1),
            m_scaling: None,
        }
    }
}

/// Where the worker should source the buffer for a binding at worker
/// init time.
///
/// `LoweredMetalTape` is parameterized by `W: CanonicalParams` so the
/// lowering pass can carry weight-resolution thunks (`WtFn<W, L>`)
/// through to the worker without committing to a specific Metal weight
/// representation here. The worker resolves these thunks at ICB-record
/// time: it calls the `WtFn` against the loaded `&Weights` to get a
/// `&Layer` struct, pulls out the requested `GpuTensor`, and asks the
/// `MetalAllocator` which arena buffer + offset that pointer belongs
/// to.
pub enum Binding<W: CanonicalParams> {
    /// `MetalWorker.arena[slot]` — the worker's private tile-arena
    /// buffer for this slot. The arena is sized for the colored
    /// `NUM_TILES` post-FUF coloring (linear-scan reg allocation
    /// performed by `colored_slot_map()` in
    /// `ferrite-forward-macro/src/interpreter_codegen.rs`).
    ArenaSlot { slot: u32, binding_index: u8 },
    /// A weight bundle resolved at ICB-record time by calling the
    /// `WtFn` against `&Weights` and looking the resulting tensor's
    /// raw pointer up in the `MetalAllocator`'s arena registry. The
    /// thunk + layer index are carried verbatim from the source
    /// `Instruction<W>` variant; the worker walks them once and
    /// records the resulting buffer pointers into the ICB.
    ///
    /// `which` selects which of the bundle's tensors this binding
    /// targets — RmsNorm has only `weight`, but `LinearLayer` exposes
    /// `weight` + optional `bias`, and `RopeAppend` consumes the
    /// per-layer cos/sin pair. The worker resolves it.
    Weight {
        kind: WeightBundleKind<W>,
        which: WeightTensor,
        layer: u32,
        binding_index: u8,
    },
    /// A buffer drawn from `ForwardCtx`-equivalent runtime state at
    /// `forward()` time (input_ids, positions, KV cache pages,
    /// cu_seqlens, etc.). The worker's bucket-selection layer
    /// rebinds these on every forward — `executeCommandsInBuffer`
    /// does not re-record, so runtime bindings reach the GPU via a
    /// pre-`executeCommandsInBuffer` `setBuffer` on the encoder.
    Runtime {
        kind: RuntimeBindingKind,
        binding_index: u8,
    },
    /// The worker's shared SplitK scratch buffer — sized to
    /// `LoweredMetalTape::splitk_scratch_bytes` at worker init, used
    /// by the two-command lowering for `Instruction::AffineQmm` when
    /// the bucket picks `QmmTKernel::SplitK`. The first command
    /// (`affine_qmm_t_splitk_*`) writes the `[split_k, M, N]`
    /// partial here; the second (`splitk_reduce_sum_*`) reads it and
    /// reduces to `[M, N]` in the AffineQmm's arena slot.
    ///
    /// Only one scratch buffer is needed even when multiple AffineQmm
    /// tiles pick SplitK: ICB commands inside a single encoder are
    /// serialized, so the writer/reader pair fully completes before
    /// the next AffineQmm overwrites the scratch.
    Scratch { binding_index: u8 },
    /// Region inside the worker's shared **MoE scratch** buffer —
    /// allocated once per `MetalWorker` at init time, sized to
    /// `LoweredMetalTape::moe_scratch_bytes` (the max across all bucket
    /// tapes). The lowering arm for `Instruction::SharedFusedMoe`
    /// sub-divides the buffer into named regions (router probs, top-k
    /// indices, top-k scores, per-expert gate/up/down intermediates)
    /// by carving deterministic byte offsets — same trick as
    /// `Binding::Scratch` for SplitK, but with multiple regions.
    ///
    /// One MoE scratch buffer suffices across many `Instruction::SharedFusedMoe`
    /// (one per layer × num_layers): ICB commands inside a single
    /// encoder are serialized, so each MoE call fully completes before
    /// the next overwrites the regions.
    MoeScratch { binding_index: u8, byte_offset: u32 },
    /// Tiny inline scalar baked into a per-binding 4-byte MTLBuffer at
    /// bake time. Used for kernels (softmax, argpartition,
    /// take_along_axis, row_gather, affine_gather_qmv) whose `.metal`
    /// signature exposes a `constant int& X [[buffer(N)]]` parameter
    /// — the existing standalone dispatchers populate them via
    /// `setBytes_length_atIndex`, but ICB commands cannot use
    /// `setBytes`, so the baker allocates a tiny shared-storage
    /// `MTLBuffer`, writes `value` into it, retains it on the
    /// `BucketBaking`, and binds it at `binding_index`.
    Inline { binding_index: u8, value: u32 },
}

/// Discriminator over the typed weight thunks `Instruction<W>` carries.
///
/// Stays generic over `W` so the lowering pass doesn't have to convert
/// `WtFn<W, RmsNorm>` to a backend-neutral type; the worker (Phase
/// 5.C) handles the bridge to Metal weight buffers.
pub enum WeightBundleKind<W: CanonicalParams> {
    Embedding(crate::WtFn<W, ferrite_kernels::layers::Embedding>),
    RmsNorm(crate::WtFn<W, ferrite_kernels::layers::RmsNorm>),
    LinearLayer(crate::WtFn<W, ferrite_kernels::layers::LinearLayer>),
    /// RoPE cos/sin table lookup: `CosSinFn<W>` returns the per-layer
    /// table directly (no struct wrapper).
    CosSin(crate::CosSinFn<W>),
    /// MLX-affine int4 quantized embedding (Metal-only). Carries the
    /// packed U32 weight + F16 scales + F16 affine offsets the
    /// `affine_embed` kernel reads. P6 macro emission decides between
    /// this and `Embedding` per-layer based on safetensors layout
    /// (U32 weight ⇒ AffineQuantEmbedding, else Embedding).
    #[cfg(feature = "metal")]
    AffineQuantEmbedding(crate::WtFn<W, ferrite_kernels::layers::AffineQuantEmbedding>),
    /// Qwen-MoE / Qwen3-Next sparse MoE block bundle. Carries the
    /// dense BF16 router gate, per-expert SwitchGLU gate/up/down
    /// triples (4bit affine MLX layout), and an optional shared
    /// expert (Qwen3-Next only). The worker resolves a single
    /// `WeightTensor` per `Binding::Weight` against the layer's
    /// `MetalSwitchGluMoeWeights` field — see the `SharedFusedMoe`
    /// arm in `worker.rs::resolve_weight`.
    #[cfg(feature = "metal")]
    SharedFusedMoe(crate::WtFn<W, ferrite_kernels::layers_moe::SharedFusedMoELayer>),
}

/// Which tensor inside a multi-tensor weight bundle this binding
/// references. Most bundles have a single weight tensor (`Weight`);
/// MLX-affine `LinearLayer::AffineQuant` carries four (packed weight,
/// per-group scales, per-group affine offsets, optional fp linear bias).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightTensor {
    /// The bundle's primary weight (`RmsNorm.weight`,
    /// `LinearLayer::Dense(Linear { weight, .. })`,
    /// `LinearLayer::AffineQuant(AffineQuantLinear { weight, .. })`
    /// — the U32 packed weights, in the affine case —
    /// `Embedding.weight`).
    Weight,
    /// The bundle's bias, if present (`Linear.bias`,
    /// `LayerNorm.bias`). The worker treats absent bias as
    /// `LoweringError::MissingBias` if a binding asks for it.
    Bias,
    /// Per-group scales on an MLX-affine LinearLayer
    /// (`AffineQuantLinear.scales`, `[N, K / group_size]` F16).
    /// Only valid against `LinearLayer::AffineQuant`.
    AffineScales,
    /// Per-group affine offsets on an MLX-affine LinearLayer
    /// (`AffineQuantLinear.affine_biases`, `[N, K / group_size]` F16).
    /// MLX terminology calls these "biases" — they are NOT the
    /// linear-layer bias. Only valid against `LinearLayer::AffineQuant`.
    AffineBiases,
    /// Optional fp linear-layer bias on an MLX-affine LinearLayer
    /// (`AffineQuantLinear.linear_bias`, `[N]` in activation dtype).
    /// Worker reports `MissingBias` if the layer's `linear_bias` is
    /// `None`. Only valid against `LinearLayer::AffineQuant`.
    AffineLinearBias,
    // ── MoE bundle (`WeightBundleKind::SharedFusedMoe`) ───────────
    //
    // Per-expert SwitchGLU (4bit affine MLX layout): each variant
    // names one `[E, *, *]` slab on the layer's
    // `MetalSwitchGluMoeWeights`. The worker's `SharedFusedMoe`
    // resolver matches on the variant and pulls the right field.
    /// Dense `[num_experts, hidden_size]` router gate (`mlp.gate.weight`)
    /// in `T_act` dtype. Used by the I::SharedFusedMoe lowering arm
    /// for the router GEMM step.
    MoeRouterGate,
    /// Per-expert gate_proj packed u32 weight `[E, moe_inter, hidden / pack_factor]`.
    MoeExpertGateW,
    /// Per-expert gate_proj scales f16 `[E, moe_inter, hidden / group_size]`.
    MoeExpertGateS,
    /// Per-expert gate_proj affine biases f16 (per-group offset, NOT linear bias).
    MoeExpertGateB,
    MoeExpertUpW,
    MoeExpertUpS,
    MoeExpertUpB,
    /// Per-expert down_proj packed u32 `[E, hidden, moe_inter / pack_factor]`.
    MoeExpertDownW,
    MoeExpertDownS,
    MoeExpertDownB,
    /// Optional Qwen3-Next shared expert `gate_proj` packed u32
    /// `[shared_inter, hidden / pack_factor]`. Worker reports
    /// `WeightLookupFailed` if the layer ships no shared expert.
    MoeSharedGateW,
    MoeSharedGateS,
    MoeSharedGateB,
    MoeSharedUpW,
    MoeSharedUpS,
    MoeSharedUpB,
    MoeSharedDownW,
    MoeSharedDownS,
    MoeSharedDownB,
    /// Dense `[1, hidden_size]` shared expert sigmoid gate
    /// (`shared_expert_gate.weight`). `T_act` dtype. Optional.
    MoeSharedExpertGate,
}

/// Categories of buffers the worker rebinds per forward call.
///
/// These map 1:1 to fields on the runtime context the engine threads
/// into the Metal forward (the Metal analogue of `ForwardCtx`). The
/// worker's `forward()` consults `RuntimeBindingKind` to know which
/// runtime buffer to bind at which encoder slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeBindingKind {
    /// `[num_tokens]` u32 — tokens to embed.
    InputIds,
    /// `[num_tokens]` u32 — RoPE position per token (1D path) or
    /// `[3, num_tokens]` for MRoPE arches.
    Positions,
    /// `[num_tokens]` u32 — paged-cache slot per token.
    SlotMapping,
    /// `[batch+1]` u32 — prefill-only sequence boundaries.
    CuSeqlensQ,
    /// `[batch]` u32 — current K-axis used length per sequence.
    SeqUsedK,
    /// `[batch, max_blocks]` u32 — per-sequence block table.
    BlockTable,
    /// Paged KV cache pool (the worker resolves to the K and V
    /// halves at the right layer offset based on `layer`).
    KvCacheK {
        layer: u32,
    },
    KvCacheV {
        layer: u32,
    },
}

/// One ICB command: kernel + dispatch shape + bindings.
///
/// All bindings are by-slot/by-thunk references (no raw buffer
/// pointers) so this struct is shareable across workers via `Arc`.
///
/// `library` / `function` / `constants` together identify the exact
/// `MTLComputePipelineState` the worker should bind. The lowering
/// pass picks them per-kernel from `W` + bucket_m + dtype; the worker
/// just hashes them into a [`PipelineKey`] and queries the cache.
/// `KernelId` lingers for diagnostics (logs, debug formatting,
/// `BucketStep::Icb { kernel, .. }`) and for the GEMM special-case
/// the worker still routes around (`KernelId::Gemm` is opaque to
/// `pipeline_for_command` — f16 goes to MPS, bf16 has its own
/// dims-keyed builder).
///
/// [`PipelineKey`]: ferrite_metal_kernels::specialized_pipeline_cache::PipelineKey
pub struct LoweredCommand<W: CanonicalParams> {
    pub kernel: KernelId,
    /// Compiled-metallib name the kernel symbol lives in (matches the
    /// `&'static str` keys [`SpecializedPipelineCache::with_standard_shaders`]
    /// registers). Empty for `KernelId::Gemm` (no entry; routed
    /// out-of-band).
    ///
    /// [`SpecializedPipelineCache::with_standard_shaders`]: ferrite_metal_kernels::specialized_pipeline_cache::SpecializedPipelineCache::with_standard_shaders
    pub library: &'static str,
    /// MSL `kernel void` symbol the pipeline binds. Empty for
    /// `KernelId::Gemm`.
    pub function: &'static str,
    /// `[[function_constant(N)]]` bag the pipeline specializes on.
    /// Pre-baked at lowering time from `W::*` + bucket_m so the worker
    /// never reaches back into `CanonicalParams`. Empty Vec is valid
    /// (e.g. `Add`, `ScalarMul`); empty constants AND empty function
    /// name signals "this is the opaque GEMM path."
    pub constants: Vec<ConstantValue>,
    pub dispatch: DispatchShape,
    pub bindings: Vec<Binding<W>>,
    /// Dense-GEMM dimensions when `kernel == KernelId::Gemm`; `None`
    /// for every other kernel. The worker reads `(m, n, k)` from
    /// here when encoding the MPS dispatch (5.C.5 routing).
    /// Carried on the lowered command rather than baked into
    /// `DispatchShape` because MPS does not consume threadgroup
    /// counts — the dimensions are the actual API parameters.
    pub gemm_dims: Option<GemmDims>,
}

/// Dense-GEMM dimensions for `KernelId::Gemm`.
///
/// `out = in @ weight^T` for the canonical row-major Linear layer:
/// `in: [m, k]`, `weight: [n, k]`, `out: [m, n]`. Future quantized
/// or transposed variants get sibling structs once they land.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmDims {
    /// Rows of the activation / output (= bucket_m).
    pub m: u32,
    /// Output columns (= weight rows; from `Instruction::Gemm`'s `n`).
    pub n: u32,
    /// Inner dimension (= weight columns = activation columns).
    pub k: u32,
}

/// One bucket's lowered tape — the input the `MetalWorker` walks at
/// init time to record its per-bucket ICB.
pub struct LoweredMetalTape<W: CanonicalParams> {
    /// Bucket M (number of tokens this tape was specialized for).
    /// Used by the worker to pick the right specialized pipeline
    /// (Phase 5.B) and the right runtime-buffer shapes.
    pub bucket_m: u32,
    /// Number of arena slots the tape references (post-coloring tile
    /// count). The worker allocates exactly this many arena buffers
    /// per shape class.
    pub num_arena_slots: u32,
    pub commands: Vec<LoweredCommand<W>>,
    /// MTL4 encoder barrier-before flag per command, mirroring
    /// `commands.len()`. Sourced from the macro-emitted
    /// `MetalBucketSpec::{backbone,lm_head}_barriers` slice (one
    /// bool per `Instruction`) and expanded through loop
    /// unrolling — the macro's loop-compression body has the same
    /// barrier pattern across iterations (byte-equivalence is the
    /// compression precondition), so iteration N's body row i
    /// reuses iteration 0's flag at the same position. The bake
    /// pass propagates this into `Mtl4Step.barrier_before`; the
    /// runtime never re-derives the analysis.
    pub barrier_before: Vec<bool>,
    /// Byte size of the shared SplitK scratch buffer the worker
    /// allocates if any `Instruction::AffineQmm` in this tape was
    /// lowered to the SplitK two-command form. Computed as
    /// `max(split_k * bucket_m * N * elem_size)` across all such
    /// instructions. Zero when no AffineQmm picked SplitK (in which
    /// case `Binding::Scratch` never appears and the worker skips
    /// the buffer allocation).
    pub splitk_scratch_bytes: u32,
    /// Byte size of the worker's shared MoE scratch buffer — `max`
    /// across every `Instruction::SharedFusedMoe` in the tape, taken
    /// across all bucket tapes when the worker allocates. Zero when no
    /// `Instruction::SharedFusedMoe` lowered to `Binding::MoeScratch`
    /// (i.e. the model is not MoE), in which case the worker skips
    /// allocation. Layout convention: regions are laid out as
    /// `[router | up | indices | scores]`, each padded to 256-byte
    /// alignment to satisfy Metal's MTLBuffer offset alignment for
    /// shader access.
    pub moe_scratch_bytes: u32,
}

/// Errors produced by the lowering pass.
///
/// `LoweringError::UnsupportedVariant` is the most common case during
/// Phase 5.A: each model adds new `Instruction<W>` variants that the
/// TinyLlama-only lowering doesn't yet handle. The variant name and
/// the `Instruction<W>` index are surfaced so the model author knows
/// exactly which instruction to add support for.
#[derive(Debug)]
pub enum LoweringError {
    /// The lowering pass doesn't yet handle this `Instruction<W>`
    /// variant. Add a match arm in `lowering::lower_one()` and a
    /// matching kernel in `KernelId`.
    UnsupportedVariant {
        /// Index of the offending instruction in the source tape.
        index: usize,
        /// `std::any::type_name_of_val()` of the offending variant
        /// — gives the variant name without requiring `Debug`.
        variant_type: &'static str,
    },
    /// A `Loop(count, body_len)` instruction overran the source tape
    /// when unrolling: the body extended past the end of the slice.
    /// Indicates malformed codegen — every tape the macro produces
    /// has been validated by the time it reaches lowering.
    MalformedLoop {
        index: usize,
        count: u32,
        body_len: u32,
        remaining: usize,
    },
}

impl std::fmt::Display for LoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVariant {
                index,
                variant_type,
            } => write!(
                f,
                "lowering: unsupported instruction variant `{variant_type}` at tape index {index} \
                 (TinyLlama-1.1B critical path is the only set covered in Phase 5.A; \
                 add a match arm in `interpreter::metal::lowering::lower_one`)"
            ),
            Self::MalformedLoop {
                index,
                count,
                body_len,
                remaining,
            } => write!(
                f,
                "lowering: malformed Loop({count}, {body_len}) at tape index {index} \
                 — body extends past tape end (only {remaining} instructions remain)"
            ),
        }
    }
}

impl std::error::Error for LoweringError {}
