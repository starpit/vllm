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

use super::ids::{BucketM, LayerId};
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
    /// Per-row bias broadcast add: `out[m, n] = in[m, n] + bias[n]`.
    /// Singleton claim used by `MetalBiasAddImpl` for Qwen2/Qwen2.5
    /// (and any other) QKV biases that the synth megakernel doesn't
    /// absorb at the current bucket M (e.g. M ≥ 2 prefill where the
    /// solver's cost CSV picks unfused). Maps to
    /// `bias_add_{f16,bf16}_specialized` in `elementwise.metallib`;
    /// `num_cols` rides on `function_constant(0)`.
    BiasAdd,
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
    /// Persistent-envelope variant of `SynthPreAttn`. Same kernel
    /// shape and bindings, plus one appended buffer for the cross-TG
    /// barrier counter. Emitted only when
    /// `FERRITE_PERSISTENT_PREATTN=1`.
    SynthPreAttnPersistent,
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
    /// Slice the last-token row of a `[num_tokens, hidden]` activation
    /// to row 0 of the same buffer, in place. Inserted by the lowering
    /// pass before the lm_head GEMM so the GEMM runs at M=1 instead of
    /// M=num_tokens — only the last token's logits are ever consumed
    /// by the sampler. Maps to `gather_last_token_{f16,bf16}_specialized`
    /// in `gather_last_token.metallib`. Reads num_tokens from a 4-byte
    /// runtime buffer the worker re-writes each forward() — see
    /// `RuntimeBindingKind::NumTokensU32`.
    GatherLastToken,
    /// Paired with [`KernelId::GatherLastToken`]: copies row 0 of a
    /// `[num_tokens, vocab]` logits tensor BACK to row `num_tokens-1`
    /// in place, after the lm_head GEMM ran on its shrunk
    /// 1-m-tile grid. The worker's downstream
    /// `embedding_gather(logits, last_token_indices=[num_tokens-1])`
    /// then reads valid logits at row `num_tokens-1` without needing
    /// to know about the slice. Reads num_tokens from the same 4-byte
    /// runtime buffer the gather does
    /// (`RuntimeBindingKind::NumTokensU32`). Maps to
    /// `scatter_first_to_last_row_{f16,bf16}_specialized` in
    /// `gather_last_token.metallib`.
    ScatterFirstToLastRow,
    /// Row-wise precise softmax (MoE router prerequisite). Faithful
    /// port of MLX `softmax_single_row` from
    /// `mlx/backend/metal/kernels/softmax.h:10-98`. Bindings:
    /// `(in @ 0, out @ 1, axis_size_i32 inline @ 2)`. Dispatch shape:
    /// `(rows, 1, 1)` threadgroups × `(256, 1, 1)` threads. Maps to
    /// `block_softmax_precise_{float16,bfloat16}` in `softmax.metallib`.
    Softmax,
    /// Row-wise full ascending argsort. Used as the "argpartition+
    /// trailing-k slice" equivalent in the MoE router lowering for
    /// the small router widths (E ≤ 128) we target. Bindings:
    /// `(in @ 0, out_u32 @ 1, axis @ 2, one @ 3, one @ 4, stride_in @ 5,
    /// stride_out @ 6)`. Dispatch: `(1, rows, 1)` threadgroups ×
    /// `(bn, 1, 1)` threads where `bn ∈ {32,64}` per
    /// `argpartition::pick_pipeline_shape`. Symbol:
    /// `c_arg_block_sort_<dtype>_uint32_bn<bn>_tn4` in
    /// `argpartition.metallib`. Lowering pairs this with
    /// `SliceTrailingColsU32` to recover top-k indices.
    ArgPartitionTopK,
    /// 2-D contiguous take-along-axis gather: pulls the `[top_k]`
    /// scores per row from the `[num_experts]` softmax output via
    /// the `[num_tokens, top_k]` top-k index buffer. Faithful port
    /// of MLX `take_along_axis_2d_contig` (gather_axis.h). Bindings:
    /// `(src @ 0, idx_u32 @ 1, out @ 2, src_axis_i32 @ 3,
    /// idx_axis_i32 @ 4)`. Dispatch (converted to threadgroup form):
    /// `(ceil(idx_axis/tg_x), rows, 1)` × `(min(32, idx_axis), 1, 1)`.
    /// Symbol: `take_along_axis_2d_contig_{float16,bfloat16}` in
    /// `take_along_axis.metallib`.
    TakeAlongAxis,
    /// Per-row "drop everything but the trailing `top_k` columns"
    /// u32 slicer. Sits between [`KernelId::ArgPartitionTopK`] and
    /// [`KernelId::TakeAlongAxis`] to convert the full sorted-ascending
    /// `[rows, num_experts]` index tensor into `[rows, top_k]`.
    /// Bindings: `(src_u32 @ 0, dst_u32 @ 1, axis_size_i32 @ 2,
    /// top_k_i32 @ 3)`. Dispatch (threads-form converted to tg):
    /// `(ceil(top_k/tg_x), rows, 1)` × `(min(32, top_k), 1, 1)`.
    /// Symbol: `slice_trailing_cols_u32` in
    /// `slice_trailing_cols.metallib`.
    SliceTrailingColsU32,
    /// MoE per-expert gather-matvec, fast variant
    /// (`N % 8 == 0 && K % 512 == 0`). Faithful port of MLX
    /// `affine_gather_qmv_fast` (`quantized.h:1899`). Used for the
    /// 3× SwitchGLU gate/up/down projections inside one MoE block.
    /// Bindings: `(packed_w @ 0, scales @ 1, biases @ 2, x @ 3,
    /// rhs_indices @ 4, y @ 5, top_k_i32 inline @ 6)`. Function
    /// constants 0/1 carry K/N respectively. Dispatch:
    /// `(1, N/8, num_tokens*top_k)` threadgroups × `(32, 2, 1)` threads.
    /// Symbol: `affine_gather_qmv_fast_<dtype>_s_<sdtype>_gs_<gs>_b_4`
    /// in `quantized_qmv.metallib`.
    AffineGatherQmvFast,
    /// MLX-affine int4 gather-matvec generic-shape fallback. Same
    /// bindings + dispatch as [`KernelId::AffineGatherQmvFast`].
    /// Symbol: `affine_gather_qmv_<dtype>_s_<sdtype>_gs_<gs>_b_4`.
    AffineGatherQmv,
    /// `out[n, d] = Σ_k expert[n, k, d] * scores[n, k]` — the final
    /// MoE reduction. Function-constant specialization on top_k
    /// (constant 0) and hidden (constant 1). Bindings:
    /// `(expert_out @ 0, scores @ 1, out @ 2)`. Dispatch:
    /// `(ceil(hidden/tg_x), num_tokens, 1)` × `(min(64, hidden), 1, 1)`.
    /// Symbol: `moe_weighted_sum_{float16,bfloat16}` in
    /// `moe_weighted_sum.metallib`.
    MoeWeightedSum,
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
/// Axis of a threadgroup dispatch grid (`(x, y, z)`). Replaces the
/// historical `u8` field on [`MScaling`]: writing `axis: 3` would have
/// silently fallen through `worker::scale_tg_for_num_tokens`'s match
/// and returned the un-scaled grid; the enum variant set forces one
/// of the three legal options.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MScaleAxis {
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MScaling {
    pub axis: MScaleAxis,
    pub bucket_m: BucketM,
}

/// Runtime gate evaluated per dispatch — when `Some`, the worker
/// skips the dispatch unless the live `num_seqs` matches the gate.
/// `None` (the common case, used by every kernel except the
/// lm_head slice / fallback pair) means "always dispatch."
///
/// Mirrors the way `barrier_before` is a per-command parallel Vec
/// on [`LoweredMetalTape`]: cheap to encode, cheap to check at
/// dispatch time, no impact on the hot path for the 99% of
/// commands that aren't gated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeGate {
    /// Run only when `num_seqs == 1` (single-sequence forward —
    /// either pure-prefill of one seq or a single decode token).
    /// Used by the lm_head slice trio (gather/qmv/scatter).
    OnlyIfSingleSeq,
    /// Run only when `num_seqs > 1` (batched decode or mixed
    /// prefill+decode batch). Used by the full M=bucket_m lm_head
    /// fallback so the slice's per-seq-incorrect logits get
    /// overwritten by a correct multi-row GEMM.
    OnlyIfMultiSeq,
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
/// Weight bindings carry a `(bucket, op_idx, slot, kind)` locator the
/// worker passes into the per-arch [`crate::WeightAccessors`] impl at
/// ICB-record time to recover the `&Layer` struct (`linear_at`,
/// `rms_norm_at`, etc.). The resolved tensor's pointer is then looked
/// up against the `MetalAllocator`'s arena registry. No fn pointers
/// live here, so `Binding` is fully backend-neutral.
pub enum Binding {
    /// `MetalWorker.arena[slot]` — the worker's private tile-arena
    /// buffer for this slot. The arena is sized for the colored
    /// `NUM_TILES` post-FUF coloring (linear-scan reg allocation
    /// performed by `colored_slot_map()` in
    /// `ferrite-forward-macro/src/interpreter_codegen.rs`).
    ArenaSlot { slot: u32, binding_index: u8 },
    /// A weight bundle resolved at ICB-record time through the per-arch
    /// [`crate::WeightAccessors`] impl. `locator` is the
    /// `(bucket, op_idx, slot)` triple the macro baked at codegen time;
    /// the trait method picked by `kind` returns the named field on
    /// `Weights`. `which` then selects which tensor inside the bundle
    /// to bind (e.g. weight vs bias vs affine scales).
    Weight {
        kind: WeightBundleKind,
        which: WeightTensor,
        layer: LayerId,
        locator: WeightLocator,
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
    /// `setBytes_length_atIndex` of a `u32` immediate at the argument
    /// table slot `binding_index`. Used by the MoE lowering arms to
    /// pass scalar shape parameters (axis_size, top_k, etc.) that
    /// match each kernel's `constant int& [[buffer(N)]]` declaration.
    /// The worker writes the 4 bytes onto the encoder; no device
    /// buffer is allocated.
    Inline { binding_index: u8, value: u32 },
    /// A bound sub-region of the worker's shared MoE scratch buffer
    /// (`MetalWorker.moe_scratch`, sized to
    /// `LoweredMetalTape::moe_scratch_bytes`). Each lowered MoE
    /// command picks the named region it operates on by passing
    /// `byte_offset` into `setBuffer_offset_atIndex`. Sub-regions are
    /// 256-byte aligned by the lowering pass per Apple Silicon's
    /// `MTLBuffer.offset` alignment rule.
    ///
    /// Inside one bucket the regions are: router_logits, sorted_full
    /// (`[M, num_experts]` u32), topk_inds (`[M, top_k]` u32),
    /// topk_scores (`[M, top_k]` act), gate_up_out (`[M, top_k,
    /// intermediate]` act), down_out (`[M, top_k, hidden]` act),
    /// plus optional shared_expert scratch (gate_up / act / out /
    /// gate_logit) when the variant is SharedFusedMoe with
    /// `shared_expert_intermediate_size > 0`. Layout is computed at
    /// lowering time and stamped into the byte_offset field; the
    /// worker only sees opaque offsets.
    MoeScratch { binding_index: u8, byte_offset: u32 },
    /// Per-worker zero-initialized 4-byte atomic-counter buffer for
    /// the persistent-envelope cross-TG ticket-lock barrier. The
    /// worker pre-allocates ONE shared u32 buffer at init and
    /// `memset(0)`s it before each dispatch that references this
    /// binding. Used exclusively by `KernelId::SynthPreAttnPersistent`
    /// (and future persistent-envelope kernels).
    PersistentBarrierCounter { binding_index: u8 },
}

/// Per-bundle locator for the macro-emitted `WeightAccessors` impl.
/// The worker invokes the matching `<kind>_at(bucket, op_idx, slot,
/// layer)` method to recover the `&Layer` reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightLocator {
    /// Bucket id baked at codegen — matches the bucket axis of the
    /// per-arch `WeightAccessors` match arm.
    pub bucket: u32,
    /// Flat tape position of the source `Instruction` (post loop
    /// unrolling for backbone slices; absolute index in the lm_head
    /// slice for lm_head).
    pub op_idx: u32,
    /// Sub-position within the same `(bucket, op_idx)` for variants
    /// that resolve multiple accessors of the same kind — e.g.
    /// `SynthPreAttn` consumes 3 `LinearLayer`s (Q/K/V at slots 0/1/2).
    pub slot: u32,
}

/// Discriminator selecting which per-arch [`crate::WeightAccessors`]
/// method the worker should call to resolve this binding.
///
/// Pure tag — fn pointers are gone after the lift; weight resolution
/// lives at the tape level via `(bucket, op_idx, slot)` keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightBundleKind {
    /// Calls `WeightAccessors::embedding_at` → `&Embedding`.
    Embedding,
    /// Calls `WeightAccessors::rms_norm_at` → `&RmsNorm`.
    RmsNorm,
    /// Calls `WeightAccessors::linear_at` → `&LinearLayer`.
    LinearLayer,
    /// Calls `WeightAccessors::cos_sin_at` → `GpuTensor` (per-layer
    /// RoPE table; no struct wrapper).
    CosSin,
    /// MLX-affine int4 quantized embedding (Metal-only). Calls
    /// `WeightAccessors::affine_quant_embedding_at` →
    /// `&AffineQuantEmbedding`. P6 macro emission decides between this
    /// and `Embedding` per-layer based on safetensors layout (U32
    /// weight ⇒ AffineQuantEmbedding, else Embedding).
    #[cfg(feature = "metal")]
    AffineQuantEmbedding,
    /// Mixtral-style fused MoE bundle (no shared expert). Metal-only —
    /// resolves through `WeightAccessors::fused_moe_at(...)` to a
    /// `&FusedMoELayer` whose Metal arm carries packed per-expert
    /// {gate, up, down} weight slabs in MLX-affine int4 layout plus
    /// the dense router gate. The Metal worker treats this as the
    /// arbitrator for the 19-ish `WeightTensor::Moe*` variants below.
    #[cfg(feature = "metal")]
    FusedMoe,
    /// Qwen-MoE-style fused MoE + shared expert bundle. Metal-only —
    /// resolves through `WeightAccessors::shared_fused_moe_at(...)`.
    /// Carries the same per-expert slabs as `FusedMoe` plus the
    /// shared-expert {gate_up, down} affine slabs + the dense
    /// `shared_expert_gate` (`[1, hidden]` sigmoid gate). On variants
    /// where `shared_expert_intermediate_size == 0` (modern
    /// Qwen3-MoE-30B-A3B), the shared-expert tensors are absent and
    /// the lowering arm skips the shared-expert tail.
    #[cfg(feature = "metal")]
    SharedFusedMoe,
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
    // ── MoE bundle tensors ──────────────────────────────────────────
    //
    // Valid only against `WeightBundleKind::{FusedMoe, SharedFusedMoe}`.
    // The Metal worker resolves these against the MetalSwitchGluMoeWeights
    // payload on the FusedMoELayer / SharedFusedMoELayer struct. Each
    // names a concrete tensor; the worker reads its arena-backed pointer
    // and stamps it onto the encoder.
    /// `[num_experts, hidden_size]` dense router gate weight (BF16 /
    /// F16 — not quantized). Output of `Gemm(x, router_gate)` produces
    /// `[num_tokens, num_experts]` router logits.
    MoeRouterGate,
    /// `[num_experts, intermediate_size, hidden_size / 8]` packed
    /// per-expert gate_proj (gate half of SwitchGLU). U32 storage of
    /// int4 elements.
    MoeExpertGateW,
    /// `[num_experts, intermediate_size, hidden_size / group_size]`
    /// per-expert gate_proj scales (act-dtype).
    MoeExpertGateS,
    /// `[num_experts, intermediate_size, hidden_size / group_size]`
    /// per-expert gate_proj affine biases (act-dtype).
    MoeExpertGateB,
    /// Packed per-expert up_proj (`up` half of SwitchGLU).
    MoeExpertUpW,
    MoeExpertUpS,
    MoeExpertUpB,
    /// Packed per-expert down_proj. Reads `[num_tokens, top_k,
    /// intermediate_size]` × `[num_experts, hidden_size,
    /// intermediate_size]` → `[num_tokens, top_k, hidden_size]`.
    MoeExpertDownW,
    MoeExpertDownS,
    MoeExpertDownB,
    /// Shared-expert `gate_up` packed weight (`[2*shared_intermediate,
    /// hidden / 8]` U32) — present only when
    /// `shared_expert_intermediate_size > 0`. Lowering arm reads this
    /// through `LinearLayer::AffineQuant` semantics: one AffineQmm
    /// emits both halves stacked, then `SiluMul` splits.
    MoeSharedGateUpW,
    MoeSharedGateUpS,
    MoeSharedGateUpB,
    /// Shared-expert `down_proj` packed weight (`[hidden,
    /// shared_intermediate / 8]` U32).
    MoeSharedDownW,
    MoeSharedDownS,
    MoeSharedDownB,
    /// Dense `[1, hidden_size]` sigmoid gate that scales the shared-
    /// expert output. Stored as a `Linear` (not quantized) in MLX
    /// safetensors.
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
        layer: LayerId,
    },
    KvCacheV {
        layer: LayerId,
    },
    /// `[1]` u32 — actual `num_tokens` of this forward, written by
    /// the worker at `forward()` entry. Consumed by
    /// `KernelId::GatherLastToken` so the kernel can compute the
    /// source row index `num_tokens - 1` at runtime without needing
    /// a function constant (M varies per call).
    NumTokensU32,
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
pub struct LoweredCommand {
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
    pub bindings: Vec<Binding>,
    /// Dense-GEMM dimensions when `kernel == KernelId::Gemm`; `None`
    /// for every other kernel. The worker reads `(m, n, k)` from
    /// here when encoding the MPS dispatch (5.C.5 routing).
    /// Carried on the lowered command rather than baked into
    /// `DispatchShape` because MPS does not consume threadgroup
    /// counts — the dimensions are the actual API parameters.
    pub gemm_dims: Option<GemmDims>,
}

impl LoweredCommand {
    /// Construct from a [`MetalKernel`] ZST. The trait carries
    /// `LIBRARY`, `FUNCTION`, and `KERNEL_ID` so the trio can't drift
    /// out of sync. The typed Constants / BindingSet structs from
    /// Phases 2 and 3 lower to the existing `Vec` wire formats via
    /// `Into`.
    ///
    /// Use this for kernels whose symbol name is a single `&'static
    /// str` — attention, etc. Kernels whose symbol is composed at
    /// lowering time (qmv / qmm_t / synth_*) keep the struct-literal
    /// `LoweredCommand { kernel, library, function, ... }` form.
    pub fn for_kernel<K: super::kernel_identity::MetalKernel>(
        constants: K::Constants,
        bindings: K::BindingSet,
        dispatch: DispatchShape,
    ) -> Self {
        Self {
            kernel: K::KERNEL_ID,
            library: K::LIBRARY,
            function: K::FUNCTION,
            constants: constants.into(),
            dispatch,
            bindings: bindings.into(),
            gemm_dims: None,
        }
    }
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
pub struct LoweredMetalTape {
    /// Bucket M (number of tokens this tape was specialized for).
    /// Used by the worker to pick the right specialized pipeline
    /// (Phase 5.B) and the right runtime-buffer shapes.
    pub bucket_m: u32,
    /// Number of arena slots the tape references (post-coloring tile
    /// count). The worker allocates exactly this many arena buffers
    /// per shape class.
    pub num_arena_slots: u32,
    pub commands: Vec<LoweredCommand>,
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
    /// Runtime-gate flag per command, mirroring `commands.len()`.
    /// `None` (the common case) = always dispatch.
    /// `Some(OnlyIfSingleSeq)` = dispatch only when
    /// `cu_seqlens_q.len() - 1 == 1` (single-sequence forward).
    /// `Some(OnlyIfMultiSeq)` = dispatch only when
    /// `cu_seqlens_q.len() - 1 > 1` (batched / mixed forward).
    /// Used by the lm_head slice path so the cheap M=1 slice
    /// (gather/qmv/scatter) fires for single-seq prefill while a
    /// parallel full M=bucket_m lm_head qmm fires only when the
    /// bucket holds multiple sequences (the slice's gather/scatter
    /// only handles row `num_tokens-1`, so it produces stale logits
    /// for every seq except the last in a packed batch).
    pub runtime_gate: Vec<Option<RuntimeGate>>,
    /// Byte size of the shared SplitK scratch buffer the worker
    /// allocates if any `Instruction::AffineQmm` in this tape was
    /// lowered to the SplitK two-command form. Computed as
    /// `max(split_k * bucket_m * N * elem_size)` across all such
    /// instructions. Zero when no AffineQmm picked SplitK (in which
    /// case `Binding::Scratch` never appears and the worker skips
    /// the buffer allocation).
    pub splitk_scratch_bytes: u32,
    /// Byte size of the shared MoE scratch buffer the worker allocates
    /// if any `Instruction::{FusedMoe, SharedFusedMoe}` in this tape
    /// was lowered. The lowering pass packs router_logits / sorted_inds
    /// / topk_inds / topk_scores / gate_out / up_out / down_out (and
    /// shared-expert intermediates when present) into a single buffer
    /// region, each 256-byte aligned. Computed as the max
    /// per-MoE-block scratch footprint across all `I::FusedMoe` /
    /// `I::SharedFusedMoe` lowerings in this tape (MoE blocks within
    /// one bucket execute serially through the ICB, so they can share
    /// scratch). Zero when no MoE instruction was lowered.
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
