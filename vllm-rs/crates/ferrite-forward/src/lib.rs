// SPDX-License-Identifier: Apache-2.0
//! Consumer-facing crate for the `#[forward]` attribute macro.
//!
//! Re-exports the proc-macro and exposes runtime support types
//! the generated code depends on: most importantly [`ForwardCtx`],
//! the ambient-args bundle the emitted forward fn takes.

pub use ferrite_forward_macro::{forward, vision_forward};

#[cfg(feature = "cuda")]
pub mod attack_surface;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod backend_compat;
pub mod cpu_golden;
pub mod paged_kv_layout;
// `info` is the non-generic, hashable backbone view used by
// `vllm ferrite info`. The whole pipeline (BackboneDumpRegistration
// inventory + Instruction::normalize) is backend-agnostic — opened to
// both `cuda` and `metal` so the CLI subcommand walks compiled metal
// arches too.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod info;
// `instr` is dual-mode: the `Instruction<W>` enum + `CanonicalParams` trait +
// `WtFn`/`CosSinFn` aliases compile under either `cuda` or `metal`. The
// CUDA-only eval/run/run_backbone fns inside are individually
// `#[cfg(feature = "cuda")]`-gated.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod instr;
// Layered-load helpers are dual-mode like `layers` / `rotary`: stream-free
// helpers (Embedding, RmsNorm, LinearDense, plus the new `_concat_packed`
// variant) compile under metal too; cuda-stream-using and quant variants are
// individually `#[cfg(feature = "cuda")]`-gated inside the file.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod loaders;
#[cfg(feature = "cuda")]
pub mod tile_table;
#[cfg(feature = "cuda")]
pub mod vision_arch;

// Metal interpreter: lowering pass + worker pool. Phase 5.A lands the
// lowering data model and `From<&[Instruction<W>]>` impl.
#[cfg(feature = "metal")]
pub mod interpreter;

#[cfg(any(feature = "cuda", feature = "metal"))]
pub use info::{
    BackboneDumpRegistration, BucketDump, NormalizedField, NormalizedStep, VariantDump,
    normalize_slice,
};

// Backend-agnostic frontend types. The lift dropped `WtFn`/`CosSinFn`
// — every variant now resolves weights through the per-arch
// `WeightAccessors` impl. `Instruction` and `CanonicalParams` stay
// cross-backend because the metal interpreter still walks the same
// `&[Instruction]` slices the cuda eval body matches on.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use instr::{CanonicalParams, Instruction, WeightAccessors};
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use backend_compat::{BackendCompat, Cuda, Metal, Wgpu};
// CUDA-only runtime entry points. `WeightAccessors` is the per-arch
// tape-level weight-resolution trait emitted by the macro (cuda eval
// body and the metal worker call into it instead of unpacking
// fn-pointers from variant fields).
#[cfg(feature = "cuda")]
pub use instr::{InterpreterCtx, run, run_backbone};
// Stream-free / quant-free helpers — reachable under either backend.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use loaders::{
    load_layered_embedding, load_layered_layer_norm, load_layered_layer_norm_vision,
    load_layered_linear_dense, load_layered_linear_dense_concat_packed,
    load_layered_linear_dense_vision, load_layered_rms_norm, load_layered_rms_norm_vision,
};
// MLX-affine helpers — Metal-only. Macro-emitted code uses these
// against `mlx-community/*-4bit` checkpoints.
#[cfg(feature = "metal")]
pub use loaders::{
    load_layered_linear_affine_dequant_as_dense,
    load_layered_linear_affine_dequant_concat_as_dense, load_layered_linear_affine_quant,
};
// Stream-using and quant helpers — cuda-only.
#[cfg(feature = "cuda")]
pub use loaders::{
    load_layered_bnb4, load_layered_bnb4_concat, load_layered_embedding_sharded,
    load_layered_fp8_block_linear, load_layered_fp8_block_linear_concat, load_layered_fp8_linear,
    load_layered_fp8_linear_concat, load_layered_linear_dense_concat,
    load_layered_linear_dense_concat_sharded, load_layered_linear_dense_concat_vision,
    load_layered_linear_dense_sharded, load_layered_marlin_linear,
    load_layered_marlin_linear_concat,
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
// `BucketEntry` is a plain tuple struct — no backend deps. Lifted to
// `any(cuda, metal)` so the metal-side macro emits `FORWARD_TABLE`
// too and `vllm ferrite info`'s `dump()` walks it. The cuda runtime
// uses fields 6/7/8 (num_slots / backbone_slot / terminal_slot) to
// size the tile table and pick return slots; the metal pool uses
// `METAL_BUCKETS` instead, so those fields stay populated but
// untouched on metal.
#[cfg(any(feature = "cuda", feature = "metal"))]
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
    for e in table {
        if e.0 <= num_tokens && num_tokens < e.1 && e.2 <= sk && sk < e.3 {
            return e;
        }
    }
    &table[0]
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

#[cfg(any(feature = "cuda", feature = "metal"))]
mod ctx {
    use ferrite_cuda_core::tensor::TensorView;
    use ferrite_kernels::kv_cache::KvCachePool;

    #[cfg(feature = "cuda")]
    use super::EmbedPatch;

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
        ///   arm in `vllm-executor::ferrite_worker`); ferrite-forward
        ///   itself is shape-agnostic past the kernel boundary.
        pub positions: TensorView<'a>,
        pub slot_mapping: TensorView<'a>,
        pub cu_seqlens_q: TensorView<'a>,
        pub seqused_k: TensorView<'a>,
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
        #[cfg(feature = "cuda")]
        pub mm_embeds: Option<TensorView<'a>>,
        #[cfg(feature = "cuda")]
        pub embed_patches: &'a [EmbedPatch],
        /// Vision-tower 2D RoPE cos table, shape `[total_L, head_dim/2]`,
        /// bf16. Built host-side from `grid_thw` per vision-encoder call;
        /// the caller (`vision_forward`) uploads it and sets the field
        /// before invoking the vision interpreter. `None` for text-side
        /// forward calls — the `Instruction::VisionRope` arm panics on
        /// `expect` if reached without these set, mirroring the
        /// `tp_group` contract for `Instruction::AllReduce` at tp>1.
        #[cfg(feature = "cuda")]
        pub vision_rope_cos: Option<TensorView<'a>>,
        /// Vision-tower 2D RoPE sin table. Same shape / population /
        /// invariants as [`Self::vision_rope_cos`].
        #[cfg(feature = "cuda")]
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
        #[cfg(feature = "cuda")]
        pub pixels: Option<TensorView<'a>>,
        /// Qwen2.5-VL: cu_seqlens for the per-image **full-frame**
        /// segmentation. Populated by the vision wrapper for arches
        /// whose body calls `varlen_attention(..., cu_seqlens_full,
        /// max_seqlen_full)` at fullatt-layer indices; `None` for
        /// every text-side call and for vision arches that use a
        /// single `cu_seqlens_q` (Qwen2-VL).
        #[cfg(feature = "cuda")]
        pub vision_cu_seqlens_full: Option<TensorView<'a>>,
        /// Qwen2.5-VL: cu_seqlens for the per-window segmentation.
        /// Populated by the vision wrapper for windowed-attention
        /// layers; same `None` semantics as
        /// [`Self::vision_cu_seqlens_full`].
        #[cfg(feature = "cuda")]
        pub vision_cu_seqlens_window: Option<TensorView<'a>>,
        /// Qwen2.5-VL: max segment length under
        /// [`Self::vision_cu_seqlens_full`]. `None` when not in use.
        #[cfg(feature = "cuda")]
        pub vision_max_seqlen_full: Option<usize>,
        /// Qwen2.5-VL: max segment length under
        /// [`Self::vision_cu_seqlens_window`]. `None` when not in use.
        #[cfg(feature = "cuda")]
        pub vision_max_seqlen_window: Option<usize>,
        /// Qwen2.5-VL: per-merged-cell natural→window-grouped
        /// permutation `[L / spatial_merge_size²]` u32. Drives the
        /// entry-side `embedding_gather(x, window_index)` (and the
        /// matching `embedding_gather(cos/sin, window_index)`) so
        /// every windowed-attention layer reads contiguous segments.
        #[cfg(feature = "cuda")]
        pub vision_window_index: Option<TensorView<'a>>,
        /// Qwen2.5-VL: inverse of [`Self::vision_window_index`] —
        /// per-merged-cell window-grouped→natural permutation that
        /// undoes the entry permute on the merger output before
        /// splice into the language-model embedding stream.
        #[cfg(feature = "cuda")]
        pub vision_reverse_indices: Option<TensorView<'a>>,
        /// SigLIP-style learned positional embedding indices, shape
        /// `[num_tokens]` u32. Built host-side as `[0..num_pos,
        /// 0..num_pos, ...]` per image. Consumed by
        /// `Instruction::PosEmbed` via `kernels::embedding_gather_masked`
        /// (the same kernel `Instruction::Embed` calls). `None` for
        /// text-side forward calls and for vision arches that don't
        /// need a positional embedding (Qwen2-VL / Qwen2.5-VL use
        /// 2D RoPE via `vision_rope` instead).
        #[cfg(feature = "cuda")]
        pub vision_position_ids: Option<TensorView<'a>>,
        // The TP communicator the `Instruction::AllReduce` arm calls
        // into. `None` at tp=1 (the lowering pass emits no AllReduce
        // rows, so the field is never read). `Some(_)` only when
        // built with `--features nccl` AND the worker constructed an
        // NCCL group for this rank — see vllm-executor::ferrite_worker.
        #[cfg(feature = "nccl")]
        pub tp_group: Option<&'a std::sync::Arc<ferrite_cuda_core::NcclGroup>>,
    }
}
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use ctx::ForwardCtx;

// ── Cross-arch dispatcher ────────────────────────────────────────
//
// Every `#[forward] fn <arch>()` macro invocation auto-emits an
// `impl FerriteWeights for Weights` + an `inventory::submit!`
// registration, so `try_load` discovers every compiled arch
// without a hand-written central list. Adding a new arch touches
// only its source file plus a single `pub mod <arch>;` in the
// calling crate's lib.rs (Rust module system requirement).
//
// Same trait + registration + walk under both backends. The trait
// method signatures are identical character-for-character — the
// types they reference (`GpuDevice`, `OwnedTensor`, `ForwardCtx`)
// are cfg-mutex'd to their cuda or metal incarnation, so a single
// signature serves both. Per-arch `impl FerriteWeights` bodies
// (macro-emitted) cfg-mutex'd internally where the call shape
// genuinely differs (cuda calls cuda's `forward(weights, ctx,
// device, num_tokens)`; metal goes through `MetalWorkerPool::forward`).

#[cfg(any(feature = "cuda", feature = "metal"))]
mod dispatcher {
    use ferrite_cuda_core::CUstream;
    use ferrite_cuda_core::GpuDevice;
    use ferrite_cuda_core::OwnedTensor;
    use ferrite_cuda_core::weights::GpuWeights;

    use super::ForwardCtx;

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

        /// Per-worker arena peak in bytes (metal only).
        ///
        /// `MetalWorkerPool::for_buckets` derives the per-worker arena
        /// layout as the elementwise-max across [`METAL_BUCKETS`]'
        /// `arena_bytes` rows; the peak resident bytes per worker is
        /// the sum of that elementwise-max. The metal worker reads
        /// this to size `peak_activation_bytes` in
        /// `determine_available_memory`, replacing the 512 MiB
        /// placeholder from Step 3.B.
        ///
        /// Default returns 512 MiB so cuda-backed arches that never
        /// override this still surface a sane placeholder if the
        /// trait method is reached on a non-metal build path.
        #[cfg(feature = "metal")]
        fn metal_arena_peak_bytes(&self) -> u64 {
            512 * 1024 * 1024
        }

        /// Per-canonical metal dtype. The macro emits an override
        /// returning `<Self as CanonicalParams>::METAL_DTYPE` so the
        /// worker can route argmax / weight-loader / etc. between
        /// the f16 and bf16 paths without monomorphizing on `W`.
        /// Default `Bf16` matches the trait-level default and the
        /// modern HF checkpoint dtype.
        #[cfg(feature = "metal")]
        fn metal_dtype(&self) -> crate::interpreter::metal::MetalDtype {
            crate::interpreter::metal::MetalDtype::Bf16
        }
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
        pub rope_scaling_type: Option<&'a str>,
        /// Deterministic hash of the full `rope_scaling` JSON
        /// subobject (or `None` when the checkpoint has no
        /// rope_scaling). Discriminates checkpoints that share
        /// `rope_scaling_type` but differ in `short_factor` /
        /// `long_factor` / `original_max_position_embeddings`
        /// values — e.g. Phi-3.5-mini vs Phi-3-mini-128k,
        /// Phi-4-mini-instruct vs Phi-4-mini-reasoning. Computed
        /// with [`crate::hash_json_value`].
        ///
        /// `max_position_embeddings` is intentionally not on this
        /// fingerprint. It doesn't affect forward-fn codegen (MPE
        /// is a runtime KV-cache sizing input), and including it
        /// rejected checkpoints whose published `config.json` has
        /// a context window narrower than the upstream base
        /// (e.g. mlx-community 4bit Qwen2.5-1.5B publishes
        /// `max_position_embeddings: 32768` against the upstream
        /// `131072`). The rope-scaling-type and rope-scaling-hash
        /// disambiguation already catches every Phi-3-style
        /// long/short fork — `MPE` was redundant.
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
    // text-only arches never need to know MM exists. ferrite_worker
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
    #[cfg(feature = "cuda")]
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
    /// expansion); ferrite_worker's splice consumes these directly.
    ///
    /// `grid_t` / `grid_h_merged` / `grid_w_merged` carry the
    /// per-image grid dimensions (post spatial-merge) so the
    /// executor can build `[3, n_tokens]` MRoPE positions for image-
    /// bearing batches on Qwen2-VL-class arches: image tokens
    /// scan in `(t, h, w)` row-major order with each row of the
    /// positions tensor holding the corresponding axis's coordinate.
    /// Caller (ferrite_worker) initializes these to zero when
    /// constructing the input `placeholders`; `vision_forward` fills
    /// them on the returned `Vec<EmbedPatch>` by zipping its
    /// per-image `grid_thw` with `placeholders`. Text-only arches
    /// that never call `vision_forward` leave them at zero —
    /// `length == grid_t * grid_h_merged * grid_w_merged` is the
    /// post-merger invariant for filled patches.
    #[cfg(feature = "cuda")]
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
    #[cfg(feature = "cuda")]
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
        ///   sequence. `ferrite_worker`'s splice consumes these to D2D-
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
        /// `ferrite_worker::build_mrope_positions_2d` needs to compute
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
    #[cfg(feature = "cuda")]
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
    #[cfg(feature = "cuda")]
    pub struct FerriteMmRegistration {
        pub arch_name: &'static str,
        pub hf_arches: &'static [&'static str],
        pub gguf_archs: &'static [&'static str],
        pub tp_world_size: u8,
        pub try_load_mm: MmTryLoadFn,
        pub mm_metadata: ferrite_vision::MmMetadata,
    }

    #[cfg(feature = "cuda")]
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
    #[cfg(feature = "cuda")]
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
    /// matching the current behavior, since ferrite_worker already gates
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
    /// expected case for text-only checkpoints, where ferrite_worker
    /// proceeds with `embed_patches: &[]`. Phase D of the
    /// multimodal plan lands the first row.
    #[cfg(feature = "cuda")]
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

#[cfg(any(feature = "cuda", feature = "metal"))]
pub use dispatcher::{FerriteArchRegistration, FerriteWeights, HfFingerprint, try_load};
// Multimodal sibling surface — text-side only crates (every metal
// arch today) skip; cuda owns the vision pipeline.
#[cfg(feature = "cuda")]
pub use dispatcher::{
    EmbedPatch, FerriteMmRegistration, MmTryLoadFn, MultimodalForward, PixelInput,
    resolve_mm_metadata, try_load_mm,
};

/// Re-export `inventory` so the `#[forward]`-macro-emitted
/// `inventory::submit!` block resolves without the consuming crate
/// having to add its own `inventory` dep. Available under either
/// backend feature — the macro emits the same `inventory::submit!{
/// FerriteArchRegistration{ ... }}` shape under both.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use inventory;
