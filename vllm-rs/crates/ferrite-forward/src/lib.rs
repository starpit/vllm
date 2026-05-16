// SPDX-License-Identifier: Apache-2.0
//! Runtime support types the `#[forward]`-emitted code depends on:
//! most importantly [`ForwardCtx`], the ambient-args bundle the
//! emitted forward fn takes, and the [`Instruction`] enum the
//! generated tape rows construct.
//!
//! The `#[forward]` / `#[vision_forward]` attribute macros live in
//! [`ferrite_forward_macro`] — consumer crates import them
//! directly:
//!
//! ```ignore
//! use ferrite_forward_macro::{forward, vision_forward};
//! ```
//!
//! ferrite-forward intentionally does NOT re-export the proc-macro
//! crate so it can serve as a build-time dependency of
//! ferrite-forward-macro itself (per `MEGA_IR_PLAN.md` §9 step 4:
//! `Implementation::fan_out` returns `Vec<Instruction>` typed at
//! proc-macro time). Re-exporting the macro would re-introduce the
//! macro→forward→macro cycle.

#[cfg(feature = "cuda")]
pub mod attack_surface;
pub mod cpu_golden;
#[cfg(feature = "cuda")]
pub mod info;
pub mod instr;
#[cfg(feature = "cuda")]
pub mod interpreter;
#[cfg(feature = "cuda")]
pub mod loaders;
#[cfg(feature = "cuda")]
pub mod tile_table;
#[cfg(feature = "cuda")]
pub mod vision_arch;

#[cfg(feature = "cuda")]
pub use info::{
    BackboneDumpRegistration, BucketDump, NormalizedField, NormalizedStep, VariantDump,
    normalize_slice,
};

pub use instr::Instruction;
#[cfg(feature = "cuda")]
pub use instr::{CanonicalParams, InterpreterCtx, WeightAccessors, run, run_backbone};
#[cfg(feature = "cuda")]
pub use loaders::{
    load_layered_bnb4, load_layered_bnb4_concat, load_layered_embedding,
    load_layered_embedding_sharded, load_layered_fp8_block_linear,
    load_layered_fp8_block_linear_concat, load_layered_fp8_linear, load_layered_fp8_linear_concat,
    load_layered_layer_norm, load_layered_layer_norm_vision, load_layered_linear_dense,
    load_layered_linear_dense_concat, load_layered_linear_dense_concat_sharded,
    load_layered_linear_dense_concat_vision, load_layered_linear_dense_sharded,
    load_layered_linear_dense_vision, load_layered_marlin_linear,
    load_layered_marlin_linear_concat, load_layered_rms_norm, load_layered_rms_norm_vision,
};
#[cfg(feature = "cuda")]
pub use tile_table::{TileEntry, take_owned, tile_ref, view};
#[cfg(feature = "cuda")]
pub use vision_arch::{VisionArchWeights, VisionWrapper};

/// One row in a per-canonical forward dispatch table. Replaces the
/// O(N×M) nested-match `pub fn forward()` + per-bucket
/// `forward_m_<N>` / `forward_backbone_m_<N>` shim fns the
/// generated code used to emit. Each canonical's
/// `static FORWARD_TABLE: &[BucketEntry<Instruction<Weights>>]`
/// describes the workload-bucket boundaries
/// (`m_min` / `m_max_excl` / `sk_min` / `sk_max_excl`), the static
/// `Instruction` slices (backbone + lm_head), and the slot-map
/// metadata that varies per bucket because the solver picks
/// different Impls per workload point (e.g. `CutlassGemmAdd` fuses
/// the residual add into the GEMM at prefill, saving a slot vs the
/// decode-bucket's separate-Add path; that shifts the terminal
/// output's slot index).
///
/// Tuple-struct so prettyplease can collapse each entry to one
/// line in expanded source. Field order:
///
///   0 = m_min          (inclusive)
///   1 = m_max_excl     (exclusive; `u64::MAX` for the final bucket)
///   2 = sk_min         (inclusive; `0` when the model has no sk axis)
///   3 = sk_max_excl    (exclusive; `u64::MAX` when no sk axis)
///   4 = backbone       — static `Op` slice for this bucket's body
///   5 = lm_head        — static `Op` slice for this bucket's tail
///   6 = num_slots      — tile-table size for `run`/`run_backbone`
///   7 = backbone_slot  — slot `forward_backbone` returns
///   8 = terminal_slot  — slot `forward` returns (after lm_head)
#[cfg(feature = "cuda")]
pub struct BucketEntry<Op: 'static>(
    pub u64,
    pub u64,
    pub u64,
    pub u64,
    pub &'static [Op],
    pub &'static [Op],
    pub u32,
    pub u32,
    pub u32,
    /// Field 9: bucket id passed to `run_slice` for this row's
    /// backbone slice. The proc-macro emits a unique id per
    /// canonical lowered entry so the per-arch
    /// [`crate::instr::WeightAccessors`] match can disambiguate
    /// "same op_idx, different canonical".
    pub u32,
    /// Field 10: bucket id for this row's lm_head slice. See field 9.
    pub u32,
);

/// Linear-scan bucket lookup. Falls back to `table[0]` when no row
/// matches — the smallest bucket comes first by convention, so out-
/// of-range inputs route there (matches the old
/// `_ => unsafe { forward_m_<smallest>(...) }` arm).
#[cfg(feature = "cuda")]
pub fn find_bucket<Op: 'static>(
    table: &'static [BucketEntry<Op>],
    num_tokens: u64,
    sk: u64,
) -> &'static BucketEntry<Op> {
    &table[find_bucket_idx(table, num_tokens, sk)]
}

/// Linear-scan bucket lookup returning the matching row index.
/// Same convention as [`find_bucket`] (fallback to `0`); split out
/// so the per-model `forward()` can consult a parallel
/// `MEGA_FORWARD_TABLE` at the same index without walking the
/// table twice.
#[cfg(feature = "cuda")]
pub fn find_bucket_idx<Op: 'static>(
    table: &'static [BucketEntry<Op>],
    num_tokens: u64,
    sk: u64,
) -> usize {
    for (i, e) in table.iter().enumerate() {
        if e.0 <= num_tokens && num_tokens < e.1 && e.2 <= sk && sk < e.3 {
            return i;
        }
    }
    0
}

/// Phase 3f-2l-i runtime gate for the megakernel dispatch path.
/// Reads `FERRITE_MEGA` from the environment on the first call and
/// caches the result. Set `FERRITE_MEGA=1` (or any non-empty,
/// non-"0" value) at process start to route `forward()` through the
/// codegen'd `LAUNCH_FN_<VARIANT>` when the bucket has one; any
/// other value (unset, empty, "0") keeps the host interpreter.
///
/// Orthogonal from the build-time `FERRITE_MEGA=1` gate that drives
/// `.cu` emission: the build-time flag controls whether
/// `LAUNCH_FN_<VARIANT>` constants exist at all; this runtime flag
/// controls whether `forward()` actually calls them when they do.
/// Both must be truthy for the mega path to take over.
///
/// Unconditionally compiled in — the cost when disabled is one
/// atomic-load + branch per forward call, amortized across the
/// per-bucket lookup.
#[inline]
pub fn mega_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FERRITE_MEGA")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

/// Runtime gate for the per-op trace `Instruction::eval` opens
/// each match with. Reads `FERRITE_TRACE` from the environment on
/// the first call and caches the result. Set `FERRITE_TRACE=1`
/// (or any non-empty, non-"0" value) before launch to enable;
/// pair with `CUDA_LAUNCH_BLOCKING=1` so the trace lines align
/// with kernel completion order.
///
/// Unconditionally compiled in — the cost when disabled is one
/// atomic-load + branch per dispatched op. `Instruction<W>` carries
/// `#[derive(Debug)]` so the trace can pretty-print variants.
#[inline]
pub fn trace_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FERRITE_TRACE")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

/// Build a `model.layers.<layer>.<suffix>` weight-name string. The
/// emitted `Weights::load_with` body fires one of these per layered
/// accessor, per layer, per canonical (~9 accessors × 32 layers ×
/// 57 canonicals on llama). Routing through a helper fn instead
/// of inlining `format!("model.layers.{}.{suffix}", layer)` keeps
/// the expanded source one line per call site (a fn call) instead
/// of five (the post-expansion `format!` →
/// `::alloc::__export::must_use({ ::alloc::fmt::format(...) })`
/// pipeline). Returns a `String` because
/// `LinearLayer::load`/`RmsNorm::load`/etc. take `&str` and the
/// binding outlives the `&str` borrow.
#[inline]
pub fn layer_weight_path(layer: u32, suffix: &str) -> String {
    format!("model.layers.{layer}.{suffix}")
}

/// Multimodal-aware decoder layered key. `root` is the per-arch
/// `<prefix>.layers` template (e.g. `"model.layers"` for text-only and
/// Qwen-style VL, `"language_model.model.layers"` for Gemma3-MM-style
/// arches that nest the text decoder under `language_model.<...>`).
/// Mirrors [`vision_block_weight_path`] for the decoder side.
#[inline]
pub fn layer_weight_path_with_root(root: &str, layer: u32, suffix: &str) -> String {
    format!("{root}.{layer}.{suffix}")
}

/// Vision-tower analogue: `<root>.<layer>.<suffix>` where `root` is
/// the per-arch concatenation `<default_root>.<layered_subpath>`
/// derived from `vision_safetensors_layout` (e.g. `visual.blocks` for
/// Qwen2-VL / Qwen2.5-VL or `vision_tower.vision_model.encoder.layers`
/// for SigLIP-style encoders like Gemma3-MM). Used by the
/// `load_layered_*_vision` helpers when the codegen emits a
/// `#[vision_forward]` body — the per-block prefix differs from the
/// decoder's `model.layers.<L>.` convention.
#[inline]
pub fn vision_block_weight_path(root: &str, layer: u32, suffix: &str) -> String {
    format!("{root}.{layer}.{suffix}")
}

/// Deterministic hash of a `serde_json::Value` for `HfFingerprint`
/// content discrimination. Canonicalizes object key order and
/// hashes numbers as f64 bits so manifest-time and runtime produce
/// the same output regardless of how the JSON was parsed.
///
/// Used to distinguish checkpoints whose `rope_scaling` has the same
/// `type` + `max_position_embeddings` but different `short_factor` /
/// `long_factor` vectors (Phi-3.5-mini vs Phi-3-mini-128k,
/// Phi-4-mini-instruct vs Phi-4-mini-reasoning). The macro's
/// `emit_fingerprint_check` bakes the manifest's hash as a `u64`
/// literal; the executor passes the live HF config's hash via
/// `HfFingerprint::rope_scaling_hash`.
pub fn hash_json_value(v: &serde_json::Value) -> u64 {
    use std::hash::{Hash, Hasher};
    fn recurse<H: Hasher>(v: &serde_json::Value, h: &mut H) {
        match v {
            serde_json::Value::Null => 0u8.hash(h),
            serde_json::Value::Bool(b) => {
                1u8.hash(h);
                b.hash(h);
            }
            serde_json::Value::Number(n) => {
                2u8.hash(h);
                // Canonicalize as f64 bits so `10000` vs `10000.0`
                // produce the same hash across JSON parsers.
                let f = n.as_f64().unwrap_or(0.0);
                f.to_bits().hash(h);
            }
            serde_json::Value::String(s) => {
                3u8.hash(h);
                s.hash(h);
            }
            serde_json::Value::Array(arr) => {
                4u8.hash(h);
                arr.len().hash(h);
                for v in arr {
                    recurse(v, h);
                }
            }
            serde_json::Value::Object(obj) => {
                5u8.hash(h);
                let mut keys: Vec<&String> = obj.keys().collect();
                keys.sort();
                keys.len().hash(h);
                for k in keys {
                    k.hash(h);
                    recurse(&obj[k], h);
                }
            }
        }
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    recurse(v, &mut h);
    h.finish()
}

#[cfg(feature = "cuda")]
mod ctx {
    use ferrite_cuda_core::tensor::TensorView;
    use ferrite_kernels::kv_cache::KvCachePool;

    use super::EmbedPatch;
    use crate::interpreter::mega::{
        ActPtrs, I32MutPtr, I32Ptr, I64Ptr, KvPtrs, LaunchArgsAttn, LaunchArgsMultiStep, U32Ptr,
        WeightPtrs, block_table_ptr, input_ids_ptr, positions_ptr, seq_lens_ptr, slot_mapping_ptr,
    };

    /// Per-step GPU arrays for the multi-step cooperative kernel.
    /// Built by the executor for M=1 decode batches when FERRITE_MEGA=1
    /// at runtime and the bucket has a `_ms` variant. Attached to
    /// `ForwardCtx::multi_step`; `forward()` dispatch reads this to
    /// stage `LaunchArgsMultiStep` and call `launch_multi_step`.
    ///
    /// All pointers are device pointers. The executor owns the backing
    /// allocations and must keep them alive until the multi-step kernel
    /// completes (i.e., until after the stream is synchronized).
    ///
    /// Layout contract (all arrays have length `num_steps`, except
    /// `input_ids_multi` which is `num_steps + 1`):
    /// - `input_ids_multi[0]` — initial token seeded by the host.
    /// - `input_ids_multi[1..=num_steps]` — filled by in-kernel argmax.
    /// - `output_token_ids[0..num_steps]` — argmax results per step;
    ///   read by the executor after the kernel completes.
    pub struct MultiStepCtx {
        /// Mutable per-step input token IDs, length `num_steps + 1`.
        pub input_ids_multi: *mut u32,
        /// Per-step rotary positions, length `num_steps`.
        pub positions_multi: U32Ptr,
        /// Per-step paged-KV slot mappings, length `num_steps`.
        pub slot_mapping_multi: I64Ptr,
        /// Per-step sequence lengths (seqused_k per step), length `num_steps`.
        pub seq_lens_multi: I32Ptr,
        /// Output token IDs written by in-kernel argmax, length `num_steps`.
        pub output_token_ids: *mut u32,
        pub num_steps: i32,
    }

    // SAFETY: raw device pointers — the executor ensures the backing
    // allocations outlive concurrent access.
    unsafe impl Send for MultiStepCtx {}
    unsafe impl Sync for MultiStepCtx {}

    /// Ambient runtime args the emitted forward fn needs. The
    /// caller builds a `ForwardCtx` per forward call and passes
    /// it in. Fields are the union of what any ported kernel
    /// needs at invocation time; new kernels can reference new
    /// fields, which is the extension point. RoPE caches live on
    /// the emitted per-arch `Weights` struct (both global `rotary`
    /// and Gemma3's `rotary_local`), built inside `Weights::load`
    /// from manifest-driven bounds/scalars/rope_scaling — not
    /// threaded through here.
    pub struct ForwardCtx<'a> {
        pub input_ids: TensorView<'a>,
        /// Positions tensor. Shape contract:
        /// - Text arches (`W::MROPE_SECTION == None`): `[n_tokens]` u32.
        /// - MRoPE arches (Qwen2-VL etc. with `W::MROPE_SECTION == Some(_)`):
        ///   `[3, n_tokens]` u32 — rows are (T, H, W) coordinates;
        ///   the rope kernel reads each rotary index from the row that
        ///   `mrope_section` assigns it to. Must be set up by whoever
        ///   constructs the `ForwardCtx` (currently the `Self::Ferrite`
        ///   arm in `vllm-executor::cuda_worker`); ferrite-forward
        ///   itself is shape-agnostic past the kernel boundary.
        pub positions: TensorView<'a>,
        pub slot_mapping: TensorView<'a>,
        pub cu_seqlens_q: TensorView<'a>,
        pub seqused_k: TensorView<'a>,
        /// Per-token K lengths sized `[total_tokens]` (mega kernel path).
        /// Mega kernels read `seq_lens[token]` for each flat-batch token,
        /// while FA2 reads `seqused_k[seq]` per-batch. Both are populated
        /// by the executor; consumers pick based on which kernel they call.
        /// `None` is allowed for callers that only invoke FA2 paths.
        pub seqused_k_per_token: ::core::option::Option<TensorView<'a>>,
        pub block_table: TensorView<'a>,
        pub max_seqlen_q: usize,
        pub max_seqlen_k: usize,
        pub kv_cache: &'a KvCachePool,
        /// Multimodal embed splice. `mm_embeds` carries the projected
        /// vision-encoder output `[total_mm_tokens, hidden]` produced by
        /// [`super::MultimodalForward::vision_forward`]; `embed_patches`
        /// names the destination ranges in the input-id sequence. After
        /// the `Instruction::Embed` arm gathers `embed_tokens`, it
        /// D2D-copies each patch's slice from `mm_embeds` into the
        /// gather output's corresponding rows. Empty `embed_patches` =
        /// text-only batch, no splice — byte-identical to pre-MM
        /// behavior. `mm_embeds = None` is only valid when
        /// `embed_patches` is empty.
        pub mm_embeds: Option<TensorView<'a>>,
        pub embed_patches: &'a [EmbedPatch],
        /// Vision-tower 2D RoPE cos table, shape `[total_L, head_dim/2]`,
        /// bf16. Built host-side from `grid_thw` per vision-encoder call;
        /// the caller (`vision_forward`) uploads it and sets the field
        /// before invoking the vision interpreter. `None` for text-side
        /// forward calls — the `Instruction::VisionRope` arm panics on
        /// `expect` if reached without these set, mirroring the
        /// `tp_group` contract for `Instruction::AllReduce` at tp>1.
        pub vision_rope_cos: Option<TensorView<'a>>,
        /// Vision-tower 2D RoPE sin table. Same shape / population /
        /// invariants as [`Self::vision_rope_cos`].
        pub vision_rope_sin: Option<TensorView<'a>>,
        /// Vision-tower input patches buffer, shape `[num_tokens,
        /// vision_in_features]`, bf16. The vision encoder's
        /// `vision_forward` host wrapper packs per-image CHW pixels
        /// into this rank-2 layout (one row per patch, channels-times-
        /// patch-area columns), uploads it, and sets the field before
        /// invoking the vision interpreter. `None` for text-side
        /// forward calls — `Instruction::LoadPixels` panics on
        /// `expect` if reached without it set, mirroring the
        /// [`Self::vision_rope_cos`] contract.
        ///
        /// Synthesized by `vision_lowering::materialize_pixels` after
        /// `fuf::unroll`: every vision-prelude `pixels` extern in the
        /// DSL classifies into a `FufInput::Extern` and is rewritten
        /// to a `FufInput::Tile` whose producer is a single
        /// `OpKind::LoadPixels` node; that node's runtime
        /// counterpart copies this view into a tile-table OwnedTensor
        /// the rest of the encoder consumes. See
        /// [`crate::Instruction::LoadPixels`] for the eval body.
        pub pixels: Option<TensorView<'a>>,
        /// Qwen2.5-VL: cu_seqlens for the per-image **full-frame**
        /// segmentation. Populated by the vision wrapper for arches
        /// whose body calls `varlen_attention(..., cu_seqlens_full,
        /// max_seqlen_full)` at fullatt-layer indices; `None` for
        /// every text-side call and for vision arches that use a
        /// single `cu_seqlens_q` (Qwen2-VL).
        pub vision_cu_seqlens_full: Option<TensorView<'a>>,
        /// Qwen2.5-VL: cu_seqlens for the per-window segmentation.
        /// Populated by the vision wrapper for windowed-attention
        /// layers; same `None` semantics as
        /// [`Self::vision_cu_seqlens_full`].
        pub vision_cu_seqlens_window: Option<TensorView<'a>>,
        /// Qwen2.5-VL: max segment length under
        /// [`Self::vision_cu_seqlens_full`]. `None` when not in use.
        pub vision_max_seqlen_full: Option<usize>,
        /// Qwen2.5-VL: max segment length under
        /// [`Self::vision_cu_seqlens_window`]. `None` when not in use.
        pub vision_max_seqlen_window: Option<usize>,
        /// Qwen2.5-VL: per-merged-cell natural→window-grouped
        /// permutation `[L / spatial_merge_size²]` u32. Drives the
        /// entry-side `embedding_gather(x, window_index)` (and the
        /// matching `embedding_gather(cos/sin, window_index)`) so
        /// every windowed-attention layer reads contiguous segments.
        pub vision_window_index: Option<TensorView<'a>>,
        /// Qwen2.5-VL: inverse of [`Self::vision_window_index`] —
        /// per-merged-cell window-grouped→natural permutation that
        /// undoes the entry permute on the merger output before
        /// splice into the language-model embedding stream.
        pub vision_reverse_indices: Option<TensorView<'a>>,
        /// SigLIP-style learned positional embedding indices, shape
        /// `[num_tokens]` u32. Built host-side as `[0..num_pos,
        /// 0..num_pos, ...]` per image. Consumed by
        /// `Instruction::PosEmbed` via `kernels::embedding_gather_masked`
        /// (the same kernel `Instruction::Embed` calls). `None` for
        /// text-side forward calls and for vision arches that don't
        /// need a positional embedding (Qwen2-VL / Qwen2.5-VL use
        /// 2D RoPE via `vision_rope` instead).
        pub vision_position_ids: Option<TensorView<'a>>,
        // The TP communicator the `Instruction::AllReduce` arm calls
        // into. `None` at tp=1 (the lowering pass emits no AllReduce
        // rows, so the field is never read). `Some(_)` only when
        // built with `--features nccl` AND the worker constructed an
        // NCCL group for this rank — see vllm-executor::cuda_worker.
        #[cfg(feature = "nccl")]
        pub tp_group: Option<&'a std::sync::Arc<ferrite_cuda_core::NcclGroup>>,
        /// Multi-step cooperative kernel context. `Some(_)` when the
        /// executor has pre-staged per-step arrays and wants `forward()`
        /// to dispatch the `_ms` variant (N decode steps fused in one
        /// cooperative kernel launch). `None` for all other forward calls
        /// (prefill, single-step decode, non-mega paths).
        pub multi_step: Option<&'a MultiStepCtx>,
        /// Persistent-decode session pointer. Non-null activates the
        /// persistent-decode dispatch in `forward()`: on the first call
        /// (session.resources is None) the kernel is launched; on subsequent
        /// calls the step is submitted via the pinned protocol buffer and
        /// polled. The output token is stored in `session.last_output_token`.
        /// Null for all non-persistent-decode forward calls.
        /// Raw pointer so the generated `forward(&ctx, ...)` can mutate the
        /// session through a `&ForwardCtx` without requiring `&mut ForwardCtx`.
        pub persistent_decode_session: *mut crate::interpreter::mega::PersistentDecodeSession,
        /// Per-step CPU-side inputs for persistent-decode. Must be `Some`
        /// when `persistent_decode_session` is non-null; contains the data that
        /// `write_step_input` sends to the persistent kernel via the
        /// pinned protocol buffer.
        pub persistent_decode_step: Option<&'a PersistentDecodeStep>,
    }

    /// CPU-side per-step inputs for persistent-decode. Staged by the
    /// executor before each `forward()` call on the persistent-decode path.
    #[derive(Clone, Copy, Debug)]
    pub struct PersistentDecodeStep {
        pub input_id: u32,
        pub position: u32,
        pub seq_len: i32,
        pub slot_mapping: i64,
        pub block_table_stride: u32,
        pub block_ids: [u32; 512], // matches protocol_layout::MAX_BLOCKS
        pub num_block_ids: usize,
    }

    impl<'a> ForwardCtx<'a> {
        /// Mega-dispatch accessor: project `seqused_k` to the
        /// attention-ABI [`I32Ptr`] a [`LaunchArgsAttn::seq_lens`]
        /// field expects. Thin wrapper over [`seq_lens_ptr`]; the
        /// dtype-reinterpret rationale lives on that function.
        ///
        /// [`LaunchArgsAttn::seq_lens`]: crate::interpreter::mega::LaunchArgsAttn::seq_lens
        pub fn mega_seq_lens(&self) -> I32Ptr {
            // Mega reads seq_lens[token] per flat-batch token. Use the
            // per-token tensor if the executor populated it; fall back
            // to per-batch seqused_k for callers that don't (e.g. tests).
            // Per-token is required for mega prefill correctness — without
            // it, prefill out-of-bounds reads cause CUDA_ERROR_ILLEGAL_ADDRESS.
            match self.seqused_k_per_token {
                ::core::option::Option::Some(view) => seq_lens_ptr(view),
                ::core::option::Option::None => seq_lens_ptr(self.seqused_k),
            }
        }

        /// Mega-dispatch accessor: project `block_table` to the
        /// attention-ABI [`U32Ptr`] a [`LaunchArgsAttn::block_table`]
        /// field expects. Thin wrapper over [`block_table_ptr`]; the
        /// signed→unsigned reinterpret rationale lives on that
        /// function.
        ///
        /// [`LaunchArgsAttn::block_table`]: crate::interpreter::mega::LaunchArgsAttn::block_table
        pub fn mega_block_table(&self) -> U32Ptr {
            block_table_ptr(self.block_table)
        }

        /// Mega-dispatch accessor (Wave F): the row stride of
        /// `block_table` as a `u32`, for the
        /// [`LaunchArgsAttn::block_table_stride`] field. Read from
        /// the view's second dimension
        /// (`self.block_table.dim(1)`) — the host stages the table
        /// as `[num_tokens, max_blocks_per_seq_in_batch]` i32 in
        /// `cuda_worker::build_attention_tensors`. An empty block
        /// table (`max_blocks == 0`, configured for tests or a
        /// cold-start with no paged pages) returns a stride of `0`;
        /// the NUM_TOKENS==1 decode path doesn't dereference past
        /// index 0 so this is inert there, and the batched-decode
        /// path only enters with at least one page per sequence.
        ///
        /// [`LaunchArgsAttn::block_table_stride`]: crate::interpreter::mega::LaunchArgsAttn::block_table_stride
        pub fn mega_block_table_stride(&self) -> u32 {
            match (*self.block_table).shape() {
                [_, stride] => *stride,
                _ => 0,
            }
        }

        /// Mega-dispatch accessor: project `positions` to the QKV-
        /// tier ABI [`U32Ptr`] a [`LaunchArgsQkv::positions`] /
        /// [`LaunchArgsAttn::positions`] field expects. Thin wrapper
        /// over [`positions_ptr`]; the host-dtype rationale
        /// (`DType::U32` on the host, `const uint32_t*` on the
        /// mega kernel side) lives on that function.
        ///
        /// [`LaunchArgsQkv::positions`]: crate::interpreter::mega::LaunchArgsQkv::positions
        /// [`LaunchArgsAttn::positions`]: crate::interpreter::mega::LaunchArgsAttn::positions
        pub fn mega_positions(&self) -> U32Ptr {
            positions_ptr(self.positions)
        }

        /// Mega-dispatch accessor: project `input_ids` to the QKV-tier
        /// ABI [`U32Ptr`] a [`LaunchArgsQkv::input_ids`] /
        /// [`LaunchArgsAttn::input_ids`] field expects. Thin wrapper
        /// over [`input_ids_ptr`]; host dtype is `DType::U32` per
        /// `vllm-cuda/src/graph.rs`, kernel side reads
        /// `const uint32_t*` — zero-copy pointer reinterpret. Added
        /// at Phase 3f-2e-iii alongside the `Embed` op dispatch (the
        /// only consumer of this field in the schedule walker today).
        ///
        /// [`LaunchArgsQkv::input_ids`]: crate::interpreter::mega::LaunchArgsQkv::input_ids
        /// [`LaunchArgsAttn::input_ids`]: crate::interpreter::mega::LaunchArgsAttn::input_ids
        pub fn mega_input_ids(&self) -> U32Ptr {
            input_ids_ptr(self.input_ids)
        }

        /// Mega-dispatch accessor: project `slot_mapping` to the
        /// QKV-tier ABI [`I64Ptr`] a [`LaunchArgsQkv::slot_mapping`]
        /// / [`LaunchArgsAttn::slot_mapping`] field expects. Thin
        /// wrapper over [`slot_mapping_ptr`]; dtype matches both
        /// sides (`DType::I64` on the host, `const int64_t*` on
        /// the mega kernel side).
        ///
        /// [`LaunchArgsQkv::slot_mapping`]: crate::interpreter::mega::LaunchArgsQkv::slot_mapping
        /// [`LaunchArgsAttn::slot_mapping`]: crate::interpreter::mega::LaunchArgsAttn::slot_mapping
        pub fn mega_slot_mapping(&self) -> I64Ptr {
            slot_mapping_ptr(self.slot_mapping)
        }

        /// Mega-dispatch accessor: surface the QKV-tier ABI [`KvPtrs`]
        /// a [`LaunchArgsQkv::key_cache_ptrs`] /
        /// [`LaunchArgsAttn::key_cache_ptrs`] field expects — a device
        /// pointer to a `[num_layers]` bf16** array of per-layer K cache
        /// base pointers, laid out layer-major. Thin wrapper over the
        /// persistent array the [`KvCachePool`] owns (see
        /// [`KvCachePool::key_cache_ptrs_gpu`]); callers must keep the
        /// pool alive for the duration of the launch.
        ///
        /// The per-layer base pointers are populated once at pool
        /// construction and never change, so no host-side staging work
        /// is needed here.
        ///
        /// [`LaunchArgsQkv::key_cache_ptrs`]: crate::interpreter::mega::LaunchArgsQkv::key_cache_ptrs
        /// [`LaunchArgsAttn::key_cache_ptrs`]: crate::interpreter::mega::LaunchArgsAttn::key_cache_ptrs
        /// [`KvCachePool::key_cache_ptrs_gpu`]: ferrite_kernels::kv_cache::KvCachePool::key_cache_ptrs_gpu
        pub fn mega_key_cache_ptrs(&self) -> KvPtrs {
            self.kv_cache.key_cache_ptrs_gpu()
        }

        /// Mega-dispatch accessor: surface the QKV-tier ABI [`KvPtrs`]
        /// a [`LaunchArgsQkv::value_cache_ptrs`] /
        /// [`LaunchArgsAttn::value_cache_ptrs`] field expects. Same
        /// contract as [`Self::mega_key_cache_ptrs`], for the V side.
        ///
        /// [`LaunchArgsQkv::value_cache_ptrs`]: crate::interpreter::mega::LaunchArgsQkv::value_cache_ptrs
        /// [`LaunchArgsAttn::value_cache_ptrs`]: crate::interpreter::mega::LaunchArgsAttn::value_cache_ptrs
        pub fn mega_value_cache_ptrs(&self) -> KvPtrs {
            self.kv_cache.value_cache_ptrs_gpu()
        }

        /// Stage a [`LaunchArgsAttn`] from the seven `ForwardCtx`-owned
        /// pool/metadata fields plus the two variant-owned pointer
        /// arrays (`act_ptrs`, `weight_ptrs`) a caller has to supply.
        /// Closes the 2d-v → 2e-iii chain at a single call site: every
        /// `AttentionViaCache`-bearing variant's generated launch shim
        /// composes `act_ptrs` + `weight_ptrs` from its own slot /
        /// accessor tables, then calls this method to fold in the seven
        /// ctx-sourced fields and hands the result to
        /// [`dispatch_launch`].
        ///
        /// The seven ctx-sourced fields come through the matching
        /// `mega_*` accessor — `input_ids` via [`Self::mega_input_ids`],
        /// `positions` via [`Self::mega_positions`],
        /// `slot_mapping` via [`Self::mega_slot_mapping`],
        /// `key_cache_ptrs` / `value_cache_ptrs` via
        /// [`Self::mega_key_cache_ptrs`] / [`Self::mega_value_cache_ptrs`],
        /// `seq_lens` via [`Self::mega_seq_lens`], and `block_table`
        /// via [`Self::mega_block_table`]. Each accessor's doc has the
        /// dtype / layout rationale for the individual projection;
        /// this helper just groups them into the positional struct the
        /// emitted `LaunchFnAttn` expects.
        ///
        /// The returned struct borrows nothing with a named lifetime —
        /// it holds raw device pointers whose validity is bounded by
        /// the ctx's backing tensors + pool. Callers must keep `self`
        /// (and its `TensorView` / `KvCachePool` sources) alive for the
        /// duration of the downstream kernel launch.
        ///
        /// [`dispatch_launch`]: crate::interpreter::mega::dispatch_launch
        /// [`LaunchArgsAttn`]: crate::interpreter::mega::LaunchArgsAttn
        /// [`LaunchFnAttn`]: crate::interpreter::mega::LaunchFnAttn
        pub fn stage_launch_args_attn(
            &self,
            act_ptrs: ActPtrs,
            weight_ptrs: WeightPtrs,
            barriers: I32MutPtr,
            trace_level: i32,
        ) -> LaunchArgsAttn {
            LaunchArgsAttn {
                act_ptrs,
                weight_ptrs,
                input_ids: self.mega_input_ids(),
                positions: self.mega_positions(),
                slot_mapping: self.mega_slot_mapping(),
                key_cache_ptrs: self.mega_key_cache_ptrs(),
                value_cache_ptrs: self.mega_value_cache_ptrs(),
                seq_lens: self.mega_seq_lens(),
                block_table: self.mega_block_table(),
                block_table_stride: self.mega_block_table_stride(),
                barriers,
                trace_level,
            }
        }

        /// Stage a [`LaunchArgsMultiStep`] from this context's pool/metadata
        /// fields plus the supplied per-call pointers. Reads per-step arrays
        /// from `self.multi_step` (which must be `Some`); panics if called
        /// without a staged [`MultiStepCtx`].
        ///
        /// The returned struct holds raw device pointers. Same lifetime /
        /// aliasing contract as [`Self::stage_launch_args_attn`].
        pub fn stage_launch_args_multi_step(
            &self,
            act_ptrs: ActPtrs,
            weight_ptrs: WeightPtrs,
            barriers: I32MutPtr,
            trace_level: i32,
        ) -> LaunchArgsMultiStep {
            let ms = self
                .multi_step
                .expect("stage_launch_args_multi_step: multi_step ctx not set");
            LaunchArgsMultiStep {
                act_ptrs,
                weight_ptrs,
                input_ids_multi: ms.input_ids_multi,
                positions_multi: ms.positions_multi,
                slot_mapping_multi: ms.slot_mapping_multi,
                key_cache_ptrs: self.mega_key_cache_ptrs(),
                value_cache_ptrs: self.mega_value_cache_ptrs(),
                seq_lens_multi: ms.seq_lens_multi,
                block_table: self.mega_block_table(),
                block_table_stride: self.mega_block_table_stride(),
                barriers,
                trace_level,
                num_steps: ms.num_steps,
                output_token_ids: ms.output_token_ids,
            }
        }
    }
}
#[cfg(feature = "cuda")]
pub use ctx::{ForwardCtx, MultiStepCtx, PersistentDecodeStep};

// ── Cross-arch dispatcher (cuda only) ────────────────────────────
//
// Every `#[forward] fn <arch>()` macro invocation auto-emits an
// `impl FerriteWeights for Weights` + an `inventory::submit!`
// registration, so `try_load` discovers every compiled arch
// without a hand-written central list. Adding a new arch touches
// only its source file plus a single `pub mod <arch>;` in the
// calling crate's lib.rs (Rust module system requirement).

#[cfg(feature = "cuda")]
mod dispatcher {
    use ferrite_cuda_core::CUstream;
    use ferrite_cuda_core::alloc::OwnedTensor;
    use ferrite_cuda_core::device::GpuDevice;
    use ferrite_cuda_core::weights::GpuWeights;

    use super::ForwardCtx;
    pub use crate::interpreter::mega::PersistentDecodeResources;

    /// Arch-agnostic handle for a ferrite-loaded model. Every
    /// `#[forward] fn <arch>()` emits an `impl FerriteWeights` for
    /// its per-arch `Weights` type; callers hold
    /// `Box<dyn FerriteWeights>` and never need to know which arch
    /// they got.
    pub trait FerriteWeights: Send + Sync {
        fn arch_name(&self) -> &'static str;
        fn num_hidden_layers(&self) -> u64;
        fn hidden_size(&self) -> u64;
        fn intermediate_size(&self) -> u64;
        fn num_attention_heads(&self) -> u64;
        fn num_key_value_heads(&self) -> u64;
        fn head_dim(&self) -> u64;
        fn vocab_size(&self) -> u64;

        /// # Safety
        /// All tensors in `ctx` must be valid GPU memory; `device`
        /// must be the live CUDA device. Same invariants as each
        /// per-arch `forward`.
        unsafe fn forward(
            &self,
            ctx: &ForwardCtx,
            device: &mut GpuDevice,
            num_tokens: u64,
        ) -> OwnedTensor;

        /// Backbone-only forward (skips lm_head).
        ///
        /// # Safety
        /// Same as [`Self::forward`] — caller guarantees `ctx`
        /// tensors and `device` outlive the returned `OwnedTensor`
        /// and that kernel launches on `device.compute_stream` have
        /// completed before the output is read on another stream.
        unsafe fn forward_backbone(
            &self,
            ctx: &ForwardCtx,
            device: &mut GpuDevice,
            num_tokens: u64,
        ) -> OwnedTensor;
    }

    /// Minimal HF-config view threaded into `try_load` so per-variant
    /// `fingerprint_matches` can disambiguate checkpoints that share
    /// on-disk tensor shapes but differ in config-only fields.
    /// Phi-3-mini-4k (`max_position_embeddings=4096`, `rope_scaling=null`)
    /// and Phi-3.5-mini-128k (`131072`, `{type:"longrope", …}`) have
    /// identical weight shapes; without the config view the
    /// alphabetically-earlier variant's fingerprint wins and its
    /// (manifest-baked) RoPE cache gets used for the wrong model.
    ///
    /// Kept as a struct of plain `Option<primitive>` so ferrite
    /// stays decoupled from the caller's full HF-config parser —
    /// add fields here only when a future arch truly needs them to
    /// disambiguate.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct HfFingerprint<'a> {
        pub max_position_embeddings: Option<u64>,
        pub rope_scaling_type: Option<&'a str>,
        /// Deterministic hash of the full `rope_scaling` JSON
        /// subobject (or `None` when the checkpoint has no
        /// rope_scaling). Discriminates checkpoints that share
        /// `type` + `max_position_embeddings` but differ in
        /// `short_factor` / `long_factor` values — e.g.
        /// Phi-3.5-mini vs Phi-3-mini-128k, Phi-4-mini-instruct
        /// vs Phi-4-mini-reasoning. Computed with
        /// [`crate::hash_json_value`].
        pub rope_scaling_hash: Option<u64>,
    }

    /// One registration per `#[forward] fn <arch>()`. The macro
    /// emits an `inventory::submit!` block that constructs this.
    /// `try_load` function-pointer signature — extracted as a type
    /// alias so the registration struct doesn't trip clippy's
    /// `type_complexity` lint.
    pub type ArchTryLoadFn = fn(
        &mut GpuWeights,
        CUstream,
        usize, // max_model_len — runtime value (CLI --max-model-len or HF config fallback)
        u8,    // tp_rank — runtime value, must be < tp_world_size
        HfFingerprint<'_>,
    ) -> ::anyhow::Result<Option<Box<dyn FerriteWeights>>>;

    pub struct FerriteArchRegistration {
        /// The arch's identifier — `"llama"`, `"qwen2"`, … — from
        /// the carrier fn name. Used in logs.
        pub arch_name: &'static str,
        /// HF `architectures` strings this arch claims to handle.
        /// Harvested by the macro from the union of every compiled
        /// model's `config.json` `architectures: [..]` field.
        pub hf_arches: &'static [&'static str],
        /// GGUF `general.architecture` tags this arch claims to handle.
        /// Family-level — multiple forward arches can claim the same
        /// gguf tag (`"deepseek2"` covers V2, V3-LoRA, V3-flat;
        /// `"llama"` covers Llama and Mistral). The dispatcher tries
        /// each claimant in turn and the per-variant
        /// `fingerprint_matches` discriminates by bounds.
        ///
        /// Populated by the macro from the arch's `quantizations.json`
        /// `ggml.gguf_arch` field (defaults to `arch_name` when the
        /// entry is bare). Empty for arches without a `"ggml"`
        /// quantization preset.
        pub gguf_archs: &'static [&'static str],
        /// Compile-time tensor-parallel world size this registration
        /// covers. The macro emits one `FerriteArchRegistration` per
        /// `(arch, tp_world_size)` tuple — at task #7's outer-loop
        /// fanout that's `{1, 2, 4, 8}` per arch; until then every
        /// emitted registration is `tp_world_size = 1`. The top-level
        /// `try_load` matches on this so the runtime tp size picks
        /// the right pre-compiled variant set.
        pub tp_world_size: u8,
        /// Try to load the arch's compiled variants. Internally
        /// iterates per-variant fingerprint sniffs. Returns
        /// `Ok(Some(..))` on a variant hit, `Ok(None)` when the arch
        /// matched by name but no compiled variant's fingerprint
        /// sniff accepted the live `GpuWeights` (caller should fall
        /// back to a hand-written path), or `Err(..)` only on a
        /// genuine load failure (I/O, shape mismatch inside a matched
        /// variant, …).
        pub try_load: ArchTryLoadFn,
    }

    inventory::collect!(FerriteArchRegistration);

    // ── Multimodal sibling surface ────────────────────────────────
    //
    // `MultimodalForward` is a sibling trait, NOT a default-method
    // extension on `FerriteWeights`. Text-only arches don't
    // implement it. Discovery uses a sibling inventory row so the
    // text-side `FerriteArchRegistration` stays untouched and
    // text-only arches never need to know MM exists. cuda_worker
    // calls `try_load_mm` after `try_load` succeeds; an arch with
    // no MM submission yields `Ok(None)` and the worker proceeds
    // text-only. Phase B of the multimodal plan
    // (`~/.claude/plans/distributed-mapping-map.md`) lands the
    // surface; per-arch impls land in Phase D.

    /// One image's preprocessed pixel input, post-CPU-normalization.
    /// Pixels are uploaded to GPU by `vision_forward`; CPU-side
    /// `MultimodalData` (`vllm-common::ImageData`) is the boundary
    /// type the engine plumbs in, this is the interior shape the
    /// vision encoder consumes.
    #[derive(Debug)]
    pub struct PixelInput<'a> {
        /// Flat normalized pixels in CHW layout, length =
        /// `3 * height * width`. Borrowed; `vision_forward` uploads
        /// to GPU and the borrow ends when it returns.
        pub pixels: &'a [f32],
        pub height: u32,
        pub width: u32,
    }

    /// One projected image's embedding location in the token
    /// sequence, plus a slice of the encoder output that occupies
    /// it. The slice is `[token_offset .. token_offset + length]`
    /// of `vision_forward`'s returned `OwnedTensor`. Mirrors
    /// `vllm-common::PlaceholderRange` but in token-space (post-
    /// expansion); cuda_worker's splice consumes these directly.
    ///
    /// `grid_t` / `grid_h_merged` / `grid_w_merged` carry the
    /// per-image grid dimensions (post spatial-merge) so the
    /// executor can build `[3, n_tokens]` MRoPE positions for image-
    /// bearing batches on Qwen2-VL-class arches: image tokens
    /// scan in `(t, h, w)` row-major order with each row of the
    /// positions tensor holding the corresponding axis's coordinate.
    /// Caller (cuda_worker) initializes these to zero when
    /// constructing the input `placeholders`; `vision_forward` fills
    /// them on the returned `Vec<EmbedPatch>` by zipping its
    /// per-image `grid_thw` with `placeholders`. Text-only arches
    /// that never call `vision_forward` leave them at zero —
    /// `length == grid_t * grid_h_merged * grid_w_merged` is the
    /// post-merger invariant for filled patches.
    #[derive(Debug, Clone, Default)]
    pub struct EmbedPatch {
        /// Position in the input-id sequence where this image's
        /// projected embeddings start.
        pub token_offset: u32,
        /// Number of token slots this image occupies (= number of
        /// rows in the corresponding slice of the projected
        /// `OwnedTensor`).
        pub length: u32,
        /// Temporal-axis grid size (1 for still images on Qwen2-VL,
        /// >1 for video frames).
        pub grid_t: u32,
        /// Height-axis grid size after spatial merge (= raw `grid_h
        /// / spatial_merge_size`).
        pub grid_h_merged: u32,
        /// Width-axis grid size after spatial merge (= raw `grid_w
        /// / spatial_merge_size`).
        pub grid_w_merged: u32,
    }

    /// Sibling trait to [`FerriteWeights`]. Implemented ONLY by
    /// arches with a vision component — the qwen2 carrier-fn's
    /// emitted `Weights` does NOT implement it; only the Qwen2-VL
    /// variant's wrapper does. No `unimplemented!` defaults: arches
    /// that don't carry a vision encoder simply don't implement
    /// the trait, and their inventory rows don't surface here.
    ///
    /// Phase D of the multimodal plan lands the first impl
    /// (qwen2 vision encoder). Until then this trait has zero
    /// callers and `try_load_mm` always returns `Ok(None)`.
    pub trait MultimodalForward: Send + Sync {
        /// Encode one or more pixel batches and project into the
        /// language model's hidden space.
        ///
        /// Returns:
        /// - An `OwnedTensor` `[total_mm_tokens, hidden]` — the
        ///   stacked projected embeddings for every image, in the
        ///   order they appear in the input batch.
        /// - A `Vec<EmbedPatch>` of the same length as `pixel_batches`
        ///   recording where each image's slice lands in the token
        ///   sequence. `cuda_worker`'s splice consumes these to D2D-
        ///   copy each slice into the corresponding `Embed` tile rows.
        ///
        /// # Safety
        /// `device` must be the live CUDA device the encoder
        /// kernels launch on; the caller must keep `pixel_batches`
        /// alive for the duration of the call (the host-side
        /// borrow ends before any returned GPU memory is read).
        unsafe fn vision_forward(
            &self,
            pixel_batches: &[PixelInput<'_>],
            placeholders: &[EmbedPatch],
            device: &mut GpuDevice,
        ) -> (OwnedTensor, Vec<EmbedPatch>);

        /// Pure-CPU companion to [`Self::vision_forward`]. Returns one
        /// `(grid_t, grid_h_merged, grid_w_merged)` tuple per input image
        /// — the same metadata that `vision_forward` would fill on each
        /// returned [`EmbedPatch`], without launching any GPU work.
        ///
        /// The cached-prefix path (`tokens_before > 0` for an MM-bearing
        /// req: vision encoder already ran on a prior step and the
        /// projected embeds live in cached KV blocks) uses this to
        /// reconstruct the same per-image grid info that
        /// `cuda_worker::build_mrope_positions_2d` needs to compute
        /// MRoPE positions for the cached tokens. Without it, fall-back
        /// 1D positions for the trailing new tokens disagree with the
        /// 3D MRoPE positions used to encode the cached KV → attention
        /// goes haywire and the model emits `<|im_end|>` immediately.
        fn embed_patch_grids(&self, pixel_batches: &[PixelInput<'_>]) -> Vec<(u32, u32, u32)>;

        /// Per-arch CPU-side preprocessing metadata. Lets the executor
        /// read declarative flags (e.g. `mrope_positions`) the
        /// per-arch `pub const PROCESSOR: ferrite_vision::MmMetadata`
        /// declared. Default impl panics so every arch must surface its
        /// declaration; the macro-emitted `VisionWrapper<W>` impl
        /// returns the per-variant baked const.
        fn mm_metadata(&self) -> &'static ferrite_vision::MmMetadata;
    }

    /// Sibling MM-load fn. Same dispatch shape as [`ArchTryLoadFn`]
    /// — caller's `GpuWeights` already carries every tensor on disk
    /// (including `visual.*` for an MM checkpoint), so this fn
    /// extracts the vision sub-tree and returns a handle. Returns
    /// `Ok(None)` when the arch claims the HF arch string but the
    /// live checkpoint has no vision tensors (text-only checkpoint
    /// loaded through an MM-capable arch entry — falls through).
    pub type MmTryLoadFn = fn(
        &mut GpuWeights,
        CUstream,
        usize, // max_model_len
        u8,    // tp_rank
        HfFingerprint<'_>,
    ) -> ::anyhow::Result<Option<Box<dyn MultimodalForward>>>;

    /// Sibling registration row. One per arch that ships a vision
    /// encoder. `hf_arches` and `gguf_archs` follow the same rules
    /// as [`FerriteArchRegistration`] but the keys typically only
    /// list the multimodal-conditional-generation variants
    /// (e.g. `Qwen2VLForConditionalGeneration`, NOT plain
    /// `Qwen2ForCausalLM`).
    ///
    /// `mm_metadata` is the per-arch CPU-side preprocessing
    /// declaration baked from a `pub const PROCESSOR: MmMetadata` in
    /// the arch crate, threaded through `#[vision_forward(processor =
    /// path::PROCESSOR, ...)]`. ferrite stays arch-agnostic: every
    /// arch-specific knob (placeholder token id key, size policy,
    /// tokens-per-image policy, preprocess fn) is data on this row,
    /// not a switch in ferrite or the macro.
    pub struct FerriteMmRegistration {
        pub arch_name: &'static str,
        pub hf_arches: &'static [&'static str],
        pub gguf_archs: &'static [&'static str],
        pub tp_world_size: u8,
        pub try_load_mm: MmTryLoadFn,
        pub mm_metadata: ferrite_vision::MmMetadata,
    }

    inventory::collect!(FerriteMmRegistration);

    /// Walk the [`FerriteMmRegistration`] inventory and return the
    /// first registration whose `hf_arches` claims any of the
    /// supplied HF architecture strings. Returns `None` for text-
    /// only models. Used by serve at startup to pick MM metadata
    /// without knowing any arch names itself.
    ///
    /// `tp_world_size` filter is intentionally not applied here —
    /// MM metadata is identical across the tp variants of an arch
    /// (preprocessing is host-side and replicated), so the first
    /// hit is sufficient.
    pub fn resolve_mm_metadata(hf_arches: &[String]) -> Option<&'static FerriteMmRegistration> {
        inventory::iter::<FerriteMmRegistration>().find(|reg| {
            hf_arches
                .iter()
                .any(|a| reg.hf_arches.contains(&a.as_str()))
        })
    }

    /// Top-level ferrite loader. Walks every `#[forward]`-registered
    /// arch; the first whose `hf_arches` list contains `arch_hint`
    /// AND whose `tp_world_size` matches the runtime `tp_world_size`
    /// wins and attempts to load. Returns `Ok(None)` when either
    /// (a) no registered (arch, tp) pair claims the request, or
    /// (b) a pair matched but no compiled variant's fingerprint sniff
    /// accepted the live `GpuWeights`. Both cases let the caller
    /// fall back to the hand-written path without hard-failing.
    ///
    /// Until task #7's outer-loop fanout lands, every emitted
    /// registration is at `tp_world_size = 1`, so callers passing
    /// `tp_world_size > 1` always see `Ok(None)` (and fall back) —
    /// matching the current behavior, since cuda_worker already gates
    /// ferrite eligibility on `!use_tp`.
    pub fn try_load(
        gw: &mut GpuWeights,
        stream: CUstream,
        arch_hint: &str,
        tp_world_size: u8,
        tp_rank: u8,
        max_model_len: usize,
        hf: HfFingerprint<'_>,
    ) -> ::anyhow::Result<Option<Box<dyn FerriteWeights>>> {
        // Walk every registration that claims this HF arch identifier
        // for this TP world size. `Ok(Some(_))` and `Err(_)` terminate;
        // `Ok(None)` (this registration's variants all rejected the
        // live `GpuWeights`) falls through to the next claimant —
        // required when more than one impl crate registers the same
        // HF arch (e.g. `ferrite-model-deepseek-v3` LoRA-Q variants
        // alongside `ferrite-model-deepseek-v3-flat` direct-Q variants
        // for `DeepseekV3ForCausalLM` checkpoints with `q_lora_rank=null`,
        // and `ferrite-model-mistral` claiming `LlamaForCausalLM` as an
        // alias for GGUFs whose `general.architecture = "llama"` flattens
        // Llama-2 and Mistral together).
        // The inner `transpose` flips `Result<Option<W>>` →
        // `Option<Result<W>>` so `find_map` treats `Ok(None)` as
        // "keep looking" and any other shape as a hit.
        inventory::iter::<FerriteArchRegistration>()
            .filter(|reg| {
                (reg.hf_arches.contains(&arch_hint) || reg.gguf_archs.contains(&arch_hint))
                    && reg.tp_world_size == tp_world_size
            })
            .find_map(|reg| (reg.try_load)(gw, stream, max_model_len, tp_rank, hf).transpose())
            .transpose()
    }

    /// MM-handle counterpart to [`try_load`]. Walks
    /// [`FerriteMmRegistration`] rows for the same
    /// `(arch_hint, tp_world_size)` filter. Returns `Ok(None)`
    /// when no MM-capable arch claims the HF identifier — the
    /// expected case for text-only checkpoints, where cuda_worker
    /// proceeds with `embed_patches: &[]`. Phase D of the
    /// multimodal plan lands the first row.
    pub fn try_load_mm(
        gw: &mut GpuWeights,
        stream: CUstream,
        arch_hint: &str,
        tp_world_size: u8,
        tp_rank: u8,
        max_model_len: usize,
        hf: HfFingerprint<'_>,
    ) -> ::anyhow::Result<Option<Box<dyn MultimodalForward>>> {
        inventory::iter::<FerriteMmRegistration>()
            .filter(|reg| {
                (reg.hf_arches.contains(&arch_hint) || reg.gguf_archs.contains(&arch_hint))
                    && reg.tp_world_size == tp_world_size
            })
            .find_map(|reg| (reg.try_load_mm)(gw, stream, max_model_len, tp_rank, hf).transpose())
            .transpose()
    }
}

#[cfg(feature = "cuda")]
pub use dispatcher::{
    EmbedPatch, FerriteArchRegistration, FerriteMmRegistration, FerriteWeights, HfFingerprint,
    MmTryLoadFn, MultimodalForward, PersistentDecodeResources, PixelInput, resolve_mm_metadata,
    try_load, try_load_mm,
};

/// Re-export `inventory` so the `#[forward]`-macro-emitted
/// `inventory::submit!` block resolves without the consuming crate
/// having to add its own `inventory` dep.
#[cfg(feature = "cuda")]
pub use inventory;
