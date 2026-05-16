// SPDX-License-Identifier: Apache-2.0
//! Codegen: emit the per-(model × workload) forward fn + the
//! concrete `Weights` struct + `Weights::load` the caller uses.
//!
//! Ferrite owns weight loading. From the DSL + solver-picked Impls
//! the compiler knows every weight the forward needs — including
//! which fused Impls (e.g. `FusedQkvRopeCacheImpl`) want packed
//! weights built by concatenating multiple safetensors entries. So
//! the emitted module contains both:
//!
//! - `pub struct Weights { … }` — one field per unique `WeightAccessor`
//!   across every picked Impl in every workload bucket. Fused
//!   accessors' fields are packed `LinearLayer`s produced by
//!   streaming concat at load time.
//! - `impl Weights { pub fn load(gw, stream) -> Result<Self> }` —
//!   reads safetensors via `GpuWeights` and produces the packed
//!   struct.
//! - `pub unsafe fn forward_m_<N>(wm: &Weights, ctx, device) -> OwnedTensor`
//!   per workload bucket, plus a `forward(wm, ctx, device, num_tokens)`
//!   dispatcher.
//!
//! The caller's entire integration is two calls: `Weights::load(...)`
//! at startup and `forward(...)` per step.
//!
//! Emission-per-subgraph is delegated to
//! [`crate::impl_lib::Implementation::emit_call`]: codegen walks
//! the LOOP's waves and asks each subgraph's bound impl to emit
//! its own tokens. New kernels / new ops extend the library, not
//! this file.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use crate::classified::{OpKind, Program, WeightId};
use crate::config::ModelParams;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{ImplementationLibrary, WeightAccessor, WeightSlot};
use crate::interpreter_codegen::{ArchOpcodes, emit_bucket_static_slice, lower_bucket};
use crate::schedule::WorkloadLoops;
use crate::solver::WorkloadAssignments;

// ── Weights struct + loader emission ─────────────────────────────

/// Translate a DSL weight path (+ optional index) to the prefix the
/// safetensors file uses (without the trailing `.weight` / `.bias`).
///
/// HF decoder-only convention:
/// - `lm_head` → `lm_head`
/// - indexed weight like `self_attn.q_proj[L]` → `model.layers.<L>.self_attn.q_proj`
/// - other top-level (`embed_tokens`, `norm`, …) → `model.<dotted>`
///
/// Architectures that diverge (e.g. some models wrap lm_head in a
/// `model.` prefix) can override via a future per-arch conventions
/// mechanism; this covers Llama, Qwen2, Mistral, Gemma2.
fn safetensors_prefix(
    program: &Program,
    decoder_safetensors_prefix: Option<&str>,
    id: WeightId,
    index: Option<u64>,
) -> String {
    // DSL idents can't start with a digit, so paths like
    // `merger.mlp.0` are written `merger.mlp_0` in the body /
    // manifest; translate `_<digit>` suffixes back to `.<digit>`
    // for the on-disk safetensors key (which uses Python-attribute
    // dotted form, including numeric submodule indices).
    //
    // Vision-prelude only: if the per-arch
    // `vision_safetensors_layout.verbatim_segments` list contains a
    // raw segment, it's passed through unchanged. LLaVA-1.5's
    // `multi_modal_projector.linear_1` / `linear_2` are real Python
    // attribute names with literal underscores — the heuristic
    // would mistranslate them to `linear.1` / `linear.2`.
    let is_vision_for_verbatim = matches!(program.prelude, crate::classified::Prelude::Vision);
    let qwen_default_layout = crate::config::VisionSafetensorsLayout::qwen_default();
    let layout_for_verbatim = if is_vision_for_verbatim {
        program
            .vision_layout
            .as_ref()
            .unwrap_or(&qwen_default_layout)
    } else {
        &qwen_default_layout
    };
    let segs: Vec<String> = program
        .weights
        .path(id)
        .iter()
        .map(|seg| {
            if is_vision_for_verbatim
                && layout_for_verbatim
                    .verbatim_segments
                    .iter()
                    .any(|v| v == seg)
            {
                seg.to_string()
            } else {
                translate_digit_suffix(seg)
            }
        })
        .collect();
    let joined = segs.join(".");
    let is_vision = matches!(program.prelude, crate::classified::Prelude::Vision);
    if is_vision {
        // Resolve the per-arch vision layout (defaults to today's
        // hardcoded Qwen layout when the config omits the field —
        // that path is byte-equivalent to the previous behavior).
        let qwen_default = crate::config::VisionSafetensorsLayout::qwen_default();
        let layout = program.vision_layout.as_ref().unwrap_or(&qwen_default);
        // Subtree override: when the DSL path's first segment maps
        // to a sibling subtree on disk, the override fully replaces
        // the `<default_root>(.<layered_subpath>.{l})?` prefix and
        // the matching subtree is treated as unindexed (today no
        // subtree consumer is per-block; add a per-subtree indexed
        // flag if a future arch needs it).
        if let Some(first) = segs.first()
            && let Some(disk) = layout.subtrees.get(first)
        {
            let rest = &segs[1..];
            return if rest.is_empty() {
                disk.clone()
            } else {
                format!("{disk}.{}", rest.join("."))
            };
        }
        return match index {
            Some(l) => format!(
                "{}.{}.{}.{}",
                layout.default_root, layout.layered_subpath, l, joined
            ),
            None => format!("{}.{}", layout.default_root, joined),
        };
    }
    // Multimodal arches that nest the text decoder under
    // `language_model.<...>` (Gemma3-MM). Variant configs set
    // `decoder_safetensors_prefix: "language_model"`; we prepend it to
    // every text-decoder key (lm_head, model.layers.*, model.<...>).
    // Text-only and Qwen-style VL leave it `None` → byte-equivalent
    // `model.<...>` / `lm_head` keys.
    let key = match (index, joined.as_str()) {
        (_, "lm_head") => "lm_head".to_string(),
        // DeepSeek: the `moe` DSL name maps to `mlp` in HF safetensors.
        (Some(l), "moe") => format!("model.layers.{l}.mlp"),
        (Some(l), _) => format!("model.layers.{l}.{joined}"),
        (None, _) => format!("model.{joined}"),
    };
    match decoder_safetensors_prefix {
        Some(prefix) => format!("{prefix}.{key}"),
        None => key,
    }
}

/// Translate trailing `_<digits>` in a DSL path segment back to
/// `.<digits>` so safetensors keys like `mlp.0` round-trip through
/// the DSL's ident-only path syntax.
fn translate_digit_suffix(seg: &str) -> String {
    if let Some(idx) = seg.rfind('_')
        && idx + 1 < seg.len()
        && seg[idx + 1..].chars().all(|c| c.is_ascii_digit())
    {
        format!("{}.{}", &seg[..idx], &seg[idx + 1..])
    } else {
        seg.to_string()
    }
}

/// How the `Weights::load` method constructs a field from safetensors.
#[derive(Clone)]
enum FieldLoad {
    /// `Embedding::load(gw, prefix)`.
    Embedding(String),
    /// `RmsNorm::load(gw, prefix, eps)`. `eps` is baked in from the
    /// model config (`rms_norm_eps`).
    RmsNorm(String, f32),
    /// `LayerNorm::load(gw, prefix, eps)` — pulls `<prefix>.weight` AND
    /// optional `<prefix>.bias`. Used by `MeanSubRmsNormBiasAddImpl`
    /// (encoder models like ModernBERT, vision towers like Qwen2-VL).
    LayerNorm(String, f32),
    /// `LinearLayer::load_dense(gw, prefix)`.
    LinearDense(String),
    /// `LinearLayer::load_dense_concat(gw, &[prefix0, prefix1, ...], stream)`.
    LinearConcat(Vec<String>),
    /// `LinearLayer::load_raw(gw, key)` — reads `<key>` verbatim
    /// (no `.weight` / `.bias` suffix). Used for `nn.Parameter` weights
    /// (e.g. Gemma3 MM projector's `mm_input_projection_weight`) declared
    /// in the per-arch manifest with `kind: "raw_linear"`. Single-source
    /// only; fused-concat raw-linear isn't a real PyTorch shape and would
    /// be a manifest authoring error.
    RawLinear(String),
    /// The model has `tie_word_embeddings: true`: `lm_head` shares
    /// its weight with `embed_tokens`. No safetensors read — build
    /// the `LinearLayer` from the already-loaded embedding field
    /// whose name is carried here.
    LinearTiedToEmbedding(syn::Ident),
    /// 4-bit packed INT4 linear that feeds a Marlin GEMM. Single-
    /// source (one prefix) or fused (multiple prefixes concat along
    /// dim N → one wider `MarlinLinear`). `format` selects the
    /// on-disk convention (AWQ vs GPTQ); only the loader fn name
    /// and a couple of storage-specific args (`desc_act` for GPTQ)
    /// vary, so a single arm emits both. Emits
    /// `MarlinLinear::load_{awq,gptq}[_concat]` against the
    /// ambient `__marlin_ws` / `__device_id` bindings that
    /// [`emit_weights_struct`] plants at the top of `Weights::load`
    /// when any Marlin accessor is present.
    MarlinLinear {
        prefixes: Vec<String>,
        format: MarlinFormat,
        group_size: u32,
    },
    /// BitsAndBytes 4-bit packed linear. Single prefix
    /// (`Bnb4bitLinear::load`) or fused across several
    /// (`Bnb4bitLinear::load_concat` — packed nibbles + absmax
    /// byte-concat along the N axis). Consumes the shared
    /// `__bnb_code` + `__bnb_scratch` bindings the
    /// [`emit_weights_struct`] prelude plants when any BNB4
    /// accessor is present.
    Bnb4Linear {
        prefixes: Vec<String>,
        /// Per-shard output dims — `sum()` is the fused
        /// `out_features` the loader hands to `Bnb4bitLinear`.
        /// Single-prefix loads carry one entry.
        out_features_per_shard: Vec<u32>,
        in_features: u32,
        blocksize: u32,
    },
    /// FP8 E4M3 linear. Single prefix (`Fp8Linear::load`) or fused
    /// across several (`Fp8Linear::load_concat` — max-scale merge,
    /// per Python vLLM's `requantize_with_max_scale`). The loader
    /// reads shapes + scale layout from the on-disk tensors so it
    /// handles per-tensor dynamic, per-channel, and online-quant
    /// (BF16 checkpoint) paths from the same arm. `output_dtype`
    /// is threaded in via the `__fp8_dtype` prelude binding.
    Fp8Linear { prefixes: Vec<String> },
    /// FP8 E4M3 blockwise-quantized linear (DeepSeek-V3-style 128×128
    /// block scales). Single prefix (`Fp8BlockLinear::load`) or fused
    /// across several (`Fp8BlockLinear::load_concat` — concat FP8
    /// weight and 2-D block-scale shards along N). Shares the
    /// `__fp8_dtype` prelude binding with `FieldLoad::Fp8Linear`.
    Fp8BlockLinear { prefixes: Vec<String> },
    /// DeepSeek V2/V3 MoE layer. One field per layer index; each
    /// calls `DeepSeekV2MoELayer::load` with the per-arch constants
    /// baked in as literals.
    DeepSeekV2Moe {
        prefix: String,
        n_routed_experts: usize,
        n_shared_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        hidden_size: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        /// True when `scoring_func="sigmoid"` + `topk_method="noaux_tc"` (DeepSeek V3 / Kimi K2).
        use_sigmoid: bool,
        /// Number of expert groups for grouped top-k (V3/Kimi K2). 0 = flat top-k.
        n_expert_group: usize,
        /// Number of groups to select in the first-stage grouped top-k. 0 = disabled.
        topk_group: usize,
    },
    /// FP8 blockwise-quantized analog of `DeepSeekV2Moe` — DeepSeek-V3
    /// official 671B and Kimi K2 official checkpoints.
    DeepSeekV2Fp8BlockMoe {
        prefix: String,
        n_routed_experts: usize,
        n_shared_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        hidden_size: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        use_sigmoid: bool,
        n_expert_group: usize,
        topk_group: usize,
    },
    /// GGML/GGUF analog of `DeepSeekV2Moe` — V2-Lite, Moonlight, K2 GGUFs.
    /// Calls `DeepSeekV2GgmlMoELayer::load_gguf` with the same per-arch
    /// constants. Expert weights stay quantized end-to-end.
    DeepSeekV2GgmlMoe {
        prefix: String,
        n_routed_experts: usize,
        n_shared_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        hidden_size: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        use_sigmoid: bool,
        n_expert_group: usize,
        topk_group: usize,
    },
    /// Mixtral-style BF16 fused MoE — no shared expert. Calls
    /// `FusedMoELayer::load`. Used by Mixtral (and any future
    /// shared-expert-free MoE arch using HF's
    /// `block_sparse_moe.experts.{e}.{w1,w2,w3}` layout).
    FusedMoe {
        prefix: String,
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        hidden_size: usize,
    },
    /// Qwen-MoE-style BF16 fused MoE + shared expert. Calls
    /// `SharedFusedMoELayer::load`. Used by Qwen2-MoE / Qwen3-MoE
    /// whose checkpoints use HF's `experts.{e}.{gate,up,down}_proj`
    /// naming and ship a shared-expert SwiGLU + sigmoid gate.
    SharedFusedMoe {
        prefix: String,
        num_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        shared_expert_intermediate_size: usize,
        hidden_size: usize,
    },
}

/// Emit the `GptqLayout` token stream that selects the loader's
/// on-disk branch (native `.qweight` vs compressed-tensors
/// `.weight_packed`). Used by both the single and concat GPTQ
/// FieldLoad arms; the generated `MarlinLinear::load_gptq{,_concat}`
/// consume it directly.
fn gptq_layout_ts(layout: crate::quantization::GptqLayout) -> TokenStream {
    match layout {
        crate::quantization::GptqLayout::Qweight => {
            quote! { ::ferrite_kernels::layers_quant::GptqLayout::Qweight }
        }
        crate::quantization::GptqLayout::WeightPacked => {
            quote! { ::ferrite_kernels::layers_quant::GptqLayout::WeightPacked }
        }
    }
}

/// Per-storage-format parameters carried on a [`FieldLoad::MarlinLinear`].
/// `group_size` lives on the outer struct because both formats share
/// it; the enum captures the storage-specific bits.
#[derive(Clone, Copy, Debug)]
enum MarlinFormat {
    Awq,
    Gptq {
        /// Mirrors `quantization_config.desc_act`. `true` ⇒ loader
        /// reads `.g_idx`, argsort-permutes, and hands sort_indices
        /// to `gptq_repack_into` so same-group columns are
        /// contiguous post-repack.
        desc_act: bool,
        /// On-disk layout — `Qweight` for AutoGPTQ native
        /// (`.qweight [K/8, N]` + `.scales [num_groups, N]`) or
        /// `WeightPacked` for compressed-tensors repack
        /// (`.weight_packed [N, K/8]` + `.weight_scale [N,
        /// num_groups]`). The loader transposes CT tensors before
        /// `gptq_repack_into`, so the downstream kernel path is
        /// identical regardless of which layout this variant came
        /// from.
        layout: crate::quantization::GptqLayout,
    },
}

/// MoE config bits shared by `DeepSeekV2Moe` and `DeepSeekV2Fp8BlockMoe`
/// FieldLoad variants. Reads the per-variant config JSON once and falls
/// back to the model's `bounds` / `scalars` map when a key is absent.
struct DeepSeekMoeCfg {
    n_routed_experts: usize,
    n_shared_experts: usize,
    top_k: usize,
    moe_intermediate_size: usize,
    hidden_size: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
    use_sigmoid: bool,
    n_expert_group: usize,
    topk_group: usize,
}

fn read_deepseek_moe_cfg(model: &ModelParams) -> DeepSeekMoeCfg {
    let src = std::fs::read_to_string(&model.source_path).unwrap_or_default();
    let v: serde_json::Value = serde_json::from_str(&src).unwrap_or(serde_json::Value::Null);
    let n_routed_experts = v
        .get("n_routed_experts")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or_else(|| model.bounds.get("n_routed_experts").copied().unwrap_or(64) as usize);
    let n_shared_experts = v
        .get("n_shared_experts")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or_else(|| model.bounds.get("n_shared_experts").copied().unwrap_or(2) as usize);
    let top_k = v
        .get("num_experts_per_tok")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or_else(|| {
            model
                .bounds
                .get("num_experts_per_tok")
                .copied()
                .unwrap_or(6) as usize
        });
    let moe_intermediate_size = v
        .get("moe_intermediate_size")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or_else(|| {
            model
                .bounds
                .get("moe_intermediate_size")
                .copied()
                .unwrap_or(1536) as usize
        });
    let hidden_size = model.bounds.get("hidden_size").copied().unwrap_or(2048) as usize;
    let norm_topk_prob = v
        .get("norm_topk_prob")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    let routed_scaling_factor = v
        .get("routed_scaling_factor")
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
        .unwrap_or_else(|| {
            model
                .scalars
                .get("routed_scaling_factor")
                .copied()
                .unwrap_or(1.0) as f32
        });
    let use_sigmoid = v
        .get("scoring_func")
        .and_then(|x| x.as_str())
        .map(|s| s == "sigmoid")
        .unwrap_or(false)
        && v.get("topk_method")
            .and_then(|x| x.as_str())
            .map(|s| s == "noaux_tc")
            .unwrap_or(false);
    let n_expert_group = v
        .get("n_group")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(0);
    let topk_group = v
        .get("topk_group")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(0);
    DeepSeekMoeCfg {
        n_routed_experts,
        n_shared_experts,
        top_k,
        moe_intermediate_size,
        hidden_size,
        norm_topk_prob,
        routed_scaling_factor,
        use_sigmoid,
        n_expert_group,
        topk_group,
    }
}

/// Distill a `WeightAccessor` into its field-load plan. Uses the
/// accessor's declared `rust_type` + `source_weights` and the
/// model's config (for `rms_norm_eps` / `tie_word_embeddings`).
fn plan_field_load(
    accessor: &WeightAccessor,
    program: &Program,
    fuf: &Fuf,
    model: &ModelParams,
    manifest: &crate::weights_manifest::WeightsManifest,
) -> FieldLoad {
    let ty = accessor.rust_type.to_string().replace(' ', "");
    let is_deepseek_v2_moe = ty.ends_with("::DeepSeekV2MoELayer")
        || ty == "DeepSeekV2MoELayer"
        || ty.ends_with("layers_moe::DeepSeekV2MoELayer");
    let is_deepseek_v2_fp8_block_moe = ty.ends_with("::DeepSeekV2Fp8BlockMoELayer")
        || ty == "DeepSeekV2Fp8BlockMoELayer"
        || ty.ends_with("layers_moe::DeepSeekV2Fp8BlockMoELayer");
    let is_fused_moe = ty.ends_with("::FusedMoELayer")
        || ty == "FusedMoELayer"
        || ty.ends_with("layers_moe::FusedMoELayer");
    let is_shared_fused_moe = ty.ends_with("::SharedFusedMoELayer")
        || ty == "SharedFusedMoELayer"
        || ty.ends_with("layers_moe::SharedFusedMoELayer");
    let is_deepseek_v2_ggml_moe = ty.ends_with("::DeepSeekV2GgmlMoELayer")
        || ty == "DeepSeekV2GgmlMoELayer"
        || ty.ends_with("layers_moe::DeepSeekV2GgmlMoELayer");
    let is_embedding =
        ty.ends_with("::Embedding") || ty == "Embedding" || ty.ends_with("layers::Embedding");
    let is_rmsnorm =
        ty.ends_with("::RmsNorm") || ty == "RmsNorm" || ty.ends_with("layers::RmsNorm");
    let is_layer_norm =
        ty.ends_with("::LayerNorm") || ty == "LayerNorm" || ty.ends_with("layers::LayerNorm");
    let is_linear =
        ty.ends_with("::LinearLayer") || ty == "LinearLayer" || ty.ends_with("layers::LinearLayer");
    let is_marlin = ty.ends_with("::MarlinLinear")
        || ty == "MarlinLinear"
        || ty.ends_with("layers::MarlinLinear");
    let is_bnb4 = ty.ends_with("::Bnb4bitLinear")
        || ty == "Bnb4bitLinear"
        || ty.ends_with("layers::Bnb4bitLinear");
    // Post-Fp8AnyLinear: the macro-emitted field type is always
    // `Fp8AnyLinear` (the wrapper enum). Whether to load as `Std`
    // or `Block` is decided per-weight from the on-disk storage
    // format below.
    let is_fp8_any = ty.ends_with("::Fp8AnyLinear")
        || ty == "Fp8AnyLinear"
        || ty.ends_with("layers::Fp8AnyLinear");

    let prefixes: Vec<String> = accessor
        .source_weights
        .iter()
        .map(|(id, idx)| {
            safetensors_prefix(
                program,
                model.decoder_safetensors_prefix.as_deref(),
                *id,
                *idx,
            )
        })
        .collect();

    if is_marlin {
        // Marlin accessors are always emitted by a quant-aware impl
        // whose sources resolve to an AWQ or GPTQ storage format.
        // Group size + format must agree across every source of a
        // fused accessor (HF's fused-QKV/gate-up layers share one
        // quantization); mismatch is a data-integrity error in the
        // upstream HF repo, so we panic at macro-expansion time
        // rather than silently emit a wrong loader.
        let mut resolved: Option<(MarlinFormat, u32)> = None;
        for (wid, _idx) in &accessor.source_weights {
            let fmt = crate::quantization::storage_format_for_weight(program, fuf, *wid, model);
            let (mf, g) = match fmt {
                crate::quantization::StorageFormat::Awq { group_size: g, .. } => {
                    (MarlinFormat::Awq, g)
                }
                crate::quantization::StorageFormat::Gptq {
                    group_size: g,
                    desc_act,
                    layout,
                    ..
                } => (MarlinFormat::Gptq { desc_act, layout }, g),
                other => panic!(
                    "accessor `{}` declared `MarlinLinear` but source weight resolves to \
                     non-quantized storage ({other:?}) — solver picked a Marlin impl for a \
                     dense weight, which is a matcher bug",
                    accessor.name,
                ),
            };
            match &resolved {
                None => resolved = Some((mf, g)),
                Some((existing_mf, existing_g)) => {
                    // Format mismatch (one source Awq, another Gptq)
                    // would require two different loaders for one
                    // fused accessor — HF never mixes formats within
                    // a single MergedColumnParallelLinear.
                    let format_match = matches!(
                        (existing_mf, &mf),
                        (MarlinFormat::Awq, MarlinFormat::Awq)
                            | (MarlinFormat::Gptq { .. }, MarlinFormat::Gptq { .. })
                    );
                    if !format_match {
                        panic!(
                            "accessor `{}` fuses sources with mismatched Marlin formats \
                                 ({existing_mf:?} vs {mf:?})",
                            accessor.name,
                        );
                    }
                    if *existing_g != g {
                        panic!(
                            "accessor `{}` fuses sources with mismatched group_size \
                                 (saw {existing_g} then {g})",
                            accessor.name,
                        );
                    }
                    // desc_act and on-disk layout have to agree
                    // across sources — `.g_idx` is shared at the K
                    // axis, so fused-QKV sub-weights either all
                    // carry it or none do; a mixed AutoGPTQ /
                    // compressed-tensors fusion would be an upstream
                    // packaging error that would produce garbage
                    // after the repack.
                    if let (
                        MarlinFormat::Gptq {
                            desc_act: existing_desc_act,
                            layout: existing_layout,
                        },
                        MarlinFormat::Gptq { desc_act, layout },
                    ) = (existing_mf, &mf)
                    {
                        if *existing_desc_act != *desc_act {
                            panic!(
                                "accessor `{}` fuses GPTQ sources with mismatched desc_act \
                                     (saw {existing_desc_act} then {desc_act})",
                                accessor.name,
                            );
                        }
                        if *existing_layout != *layout {
                            panic!(
                                "accessor `{}` fuses GPTQ sources with mismatched layouts \
                                     ({existing_layout:?} vs {layout:?})",
                                accessor.name,
                            );
                        }
                    }
                }
            }
        }
        let (format, group_size) =
            resolved.expect("MarlinLinear accessor declares at least one source weight");
        return FieldLoad::MarlinLinear {
            prefixes,
            format,
            group_size,
        };
    }

    if is_bnb4 {
        // Each source weight's manifest shape evaluated against the
        // arch's bounds gives `[out_features, in_features]`. Fused
        // BNB4 concats along N, so every shard must agree on
        // `in_features`; the loader takes `out_features_per_shard`
        // + one shared `in_features`. Mismatched in_features is an
        // upstream repo bug — panic at macro-expansion time.
        let (mut out_per_shard, mut in_features, mut blocksize) =
            (Vec::<u32>::new(), None::<u32>, None::<u32>);
        for (wid, _idx) in &accessor.source_weights {
            let fmt = crate::quantization::storage_format_for_weight(program, fuf, *wid, model);
            let bs = match fmt {
                crate::quantization::StorageFormat::Bnb4 { blocksize: bs, .. } => bs,
                other => panic!(
                    "accessor `{}` declared `Bnb4bitLinear` but source weight resolves to \
                     non-BNB4 storage ({other:?}) — matcher bug",
                    accessor.name,
                ),
            };
            if let Some(existing) = blocksize
                && existing != bs
            {
                panic!(
                    "accessor `{}` fuses BNB4 sources with mismatched blocksize \
                         ({existing} then {bs})",
                    accessor.name,
                );
            }
            blocksize = Some(bs);

            let segments = program.weights.path(*wid);
            let dotted = segments.join(".");
            let shape = manifest.lookup(segments).unwrap_or_else(|| {
                panic!(
                    "accessor `{}`: weight `{dotted}` missing from weights manifest; \
                     required for BNB4 out_features/in_features evaluation",
                    accessor.name,
                )
            });
            if shape.len() != 2 {
                panic!(
                    "accessor `{}`: BNB4 source weight `{dotted}` has shape len \
                     {} (expected 2 for a matmul)",
                    accessor.name,
                    shape.len(),
                );
            }
            let in_f =
                crate::shape::eval_closed_dim(&shape[0], &model.bounds).unwrap_or_else(|| {
                    panic!(
                        "accessor `{}`: can't resolve in_features for `{dotted}`",
                        accessor.name
                    )
                }) as u32;
            let out_f =
                crate::shape::eval_closed_dim(&shape[1], &model.bounds).unwrap_or_else(|| {
                    panic!(
                        "accessor `{}`: can't resolve out_features for `{dotted}`",
                        accessor.name
                    )
                }) as u32;
            if let Some(existing) = in_features
                && existing != in_f
            {
                panic!(
                    "accessor `{}` fuses BNB4 sources with mismatched in_features \
                         ({existing} then {in_f})",
                    accessor.name,
                );
            }
            in_features = Some(in_f);
            out_per_shard.push(out_f);
        }
        return FieldLoad::Bnb4Linear {
            prefixes,
            out_features_per_shard: out_per_shard,
            in_features: in_features.expect("at least one source weight"),
            blocksize: blocksize.expect("at least one source weight"),
        };
    }

    if is_fp8_any {
        // The accessor's emitted field type is `Fp8AnyLinear`. We
        // pick `FieldLoad::Fp8Linear` (per-tensor / per-channel,
        // wraps as `Fp8AnyLinear::Std`) or `FieldLoad::Fp8BlockLinear`
        // (blockwise, wraps as `Fp8AnyLinear::Block`) by reading the
        // source weight's on-disk storage format. All source weights
        // of one fused accessor share a format — HF never mixes
        // per-tensor and blockwise scales within a single
        // MergedColumnParallelLinear — so we read the FIRST source's
        // format and assert the rest agree.
        let mut block_size: Option<bool> = None;
        for (wid, _idx) in &accessor.source_weights {
            let fmt = crate::quantization::storage_format_for_weight(program, fuf, *wid, model);
            let is_block = match fmt {
                crate::quantization::StorageFormat::Fp8 { block_size, .. } => block_size.is_some(),
                other => panic!(
                    "accessor `{}` declared `Fp8AnyLinear` but source weight resolves to \
                     non-FP8 storage ({other:?}) — matcher bug",
                    accessor.name,
                ),
            };
            match block_size {
                None => block_size = Some(is_block),
                Some(prev) => {
                    if prev != is_block {
                        panic!(
                            "accessor `{}` fuses sources with mismatched FP8 layout \
                             (one block-quant, the other per-tensor/per-channel)",
                            accessor.name,
                        );
                    }
                }
            }
        }
        return if block_size == Some(true) {
            FieldLoad::Fp8BlockLinear { prefixes }
        } else {
            FieldLoad::Fp8Linear { prefixes }
        };
    }

    if is_deepseek_v2_fp8_block_moe {
        assert_eq!(
            prefixes.len(),
            1,
            "DeepSeekV2Fp8BlockMoELayer accessor `{}` with {} sources (expected 1 per layer)",
            accessor.name,
            prefixes.len(),
        );
        let prefix = prefixes.into_iter().next().unwrap();
        let cfg = read_deepseek_moe_cfg(model);
        return FieldLoad::DeepSeekV2Fp8BlockMoe {
            prefix,
            n_routed_experts: cfg.n_routed_experts,
            n_shared_experts: cfg.n_shared_experts,
            top_k: cfg.top_k,
            moe_intermediate_size: cfg.moe_intermediate_size,
            hidden_size: cfg.hidden_size,
            norm_topk_prob: cfg.norm_topk_prob,
            routed_scaling_factor: cfg.routed_scaling_factor,
            use_sigmoid: cfg.use_sigmoid,
            n_expert_group: cfg.n_expert_group,
            topk_group: cfg.topk_group,
        };
    }

    if is_deepseek_v2_ggml_moe {
        assert_eq!(
            prefixes.len(),
            1,
            "DeepSeekV2GgmlMoELayer accessor `{}` with {} sources (expected 1 per layer)",
            accessor.name,
            prefixes.len(),
        );
        let prefix = prefixes.into_iter().next().unwrap();
        let cfg = read_deepseek_moe_cfg(model);
        return FieldLoad::DeepSeekV2GgmlMoe {
            prefix,
            n_routed_experts: cfg.n_routed_experts,
            n_shared_experts: cfg.n_shared_experts,
            top_k: cfg.top_k,
            moe_intermediate_size: cfg.moe_intermediate_size,
            hidden_size: cfg.hidden_size,
            norm_topk_prob: cfg.norm_topk_prob,
            routed_scaling_factor: cfg.routed_scaling_factor,
            use_sigmoid: cfg.use_sigmoid,
            n_expert_group: cfg.n_expert_group,
            topk_group: cfg.topk_group,
        };
    }

    if is_fused_moe {
        assert_eq!(
            prefixes.len(),
            1,
            "FusedMoELayer accessor `{}` with {} sources (expected 1 per layer)",
            accessor.name,
            prefixes.len(),
        );
        let prefix = prefixes.into_iter().next().unwrap();
        let src = std::fs::read_to_string(&model.source_path).unwrap_or_default();
        let v: serde_json::Value = serde_json::from_str(&src).unwrap_or(serde_json::Value::Null);
        let num_experts = v
            .get("num_local_experts")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                model.bounds.get("num_local_experts").copied().unwrap_or(8) as usize
            });
        let top_k = v
            .get("num_experts_per_tok")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                model
                    .bounds
                    .get("num_experts_per_tok")
                    .copied()
                    .unwrap_or(2) as usize
            });
        let intermediate_size = model
            .bounds
            .get("intermediate_size")
            .copied()
            .unwrap_or(14336) as usize;
        let hidden_size = model.bounds.get("hidden_size").copied().unwrap_or(4096) as usize;
        return FieldLoad::FusedMoe {
            prefix,
            num_experts,
            top_k,
            intermediate_size,
            hidden_size,
        };
    }

    if is_shared_fused_moe {
        assert_eq!(
            prefixes.len(),
            1,
            "SharedFusedMoELayer accessor `{}` with {} sources (expected 1 per layer)",
            accessor.name,
            prefixes.len(),
        );
        let prefix = prefixes.into_iter().next().unwrap();
        let src = std::fs::read_to_string(&model.source_path).unwrap_or_default();
        let v: serde_json::Value = serde_json::from_str(&src).unwrap_or(serde_json::Value::Null);
        let num_experts = v
            .get("num_experts")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| model.bounds.get("num_experts").copied().unwrap_or(60) as usize);
        let top_k = v
            .get("num_experts_per_tok")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                model
                    .bounds
                    .get("num_experts_per_tok")
                    .copied()
                    .unwrap_or(4) as usize
            });
        let moe_intermediate_size = v
            .get("moe_intermediate_size")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                model
                    .bounds
                    .get("moe_intermediate_size")
                    .copied()
                    .unwrap_or(1408) as usize
            });
        let shared_expert_intermediate_size = v
            .get("shared_expert_intermediate_size")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                model
                    .bounds
                    .get("shared_expert_intermediate_size")
                    .copied()
                    .unwrap_or(0) as usize
            });
        let hidden_size = model.bounds.get("hidden_size").copied().unwrap_or(2048) as usize;
        return FieldLoad::SharedFusedMoe {
            prefix,
            num_experts,
            top_k,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            hidden_size,
        };
    }

    if is_deepseek_v2_moe {
        assert_eq!(
            prefixes.len(),
            1,
            "DeepSeekV2MoELayer accessor `{}` with {} sources (expected 1 per layer)",
            accessor.name,
            prefixes.len(),
        );
        let prefix = prefixes.into_iter().next().unwrap();
        // Read MoE params from the model config JSON.
        let src = std::fs::read_to_string(&model.source_path).unwrap_or_default();
        let v: serde_json::Value = serde_json::from_str(&src).unwrap_or(serde_json::Value::Null);
        let n_routed_experts = v
            .get("n_routed_experts")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                model.bounds.get("n_routed_experts").copied().unwrap_or(64) as usize
            });
        let n_shared_experts = v
            .get("n_shared_experts")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| model.bounds.get("n_shared_experts").copied().unwrap_or(2) as usize);
        let top_k = v
            .get("num_experts_per_tok")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                model
                    .bounds
                    .get("num_experts_per_tok")
                    .copied()
                    .unwrap_or(6) as usize
            });
        let moe_intermediate_size = v
            .get("moe_intermediate_size")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                model
                    .bounds
                    .get("moe_intermediate_size")
                    .copied()
                    .unwrap_or(1536) as usize
            });
        let hidden_size = model.bounds.get("hidden_size").copied().unwrap_or(2048) as usize;
        let norm_topk_prob = v
            .get("norm_topk_prob")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let routed_scaling_factor = v
            .get("routed_scaling_factor")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .unwrap_or_else(|| {
                model
                    .scalars
                    .get("routed_scaling_factor")
                    .copied()
                    .unwrap_or(1.0) as f32
            });
        // sigmoid routing when scoring_func="sigmoid" AND topk_method="noaux_tc"
        // (DeepSeek V3 / Kimi K2). Matches Python vLLM's condition.
        let use_sigmoid = v
            .get("scoring_func")
            .and_then(|x| x.as_str())
            .map(|s| s == "sigmoid")
            .unwrap_or(false)
            && v.get("topk_method")
                .and_then(|x| x.as_str())
                .map(|s| s == "noaux_tc")
                .unwrap_or(false);
        let n_expert_group = v
            .get("n_group")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(0);
        let topk_group = v
            .get("topk_group")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(0);
        return FieldLoad::DeepSeekV2Moe {
            prefix,
            n_routed_experts,
            n_shared_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            norm_topk_prob,
            routed_scaling_factor,
            use_sigmoid,
            n_expert_group,
            topk_group,
        };
    }

    if is_embedding {
        assert_eq!(
            prefixes.len(),
            1,
            "Embedding accessor `{}` with {} sources",
            accessor.name,
            prefixes.len()
        );
        FieldLoad::Embedding(prefixes.into_iter().next().unwrap())
    } else if is_rmsnorm {
        assert_eq!(
            prefixes.len(),
            1,
            "RmsNorm accessor `{}` with {} sources",
            accessor.name,
            prefixes.len()
        );
        let eps = rms_norm_eps(model);
        FieldLoad::RmsNorm(prefixes.into_iter().next().unwrap(), eps)
    } else if is_layer_norm {
        assert_eq!(
            prefixes.len(),
            1,
            "LayerNorm accessor `{}` with {} sources",
            accessor.name,
            prefixes.len()
        );
        // Same eps source as RmsNorm — `rms_norm_eps` already accepts
        // the `layer_norm_eps` JSON key (CommandR convention) as a
        // fallback. ModernBERT writes `norm_eps` / `layer_norm_eps`,
        // both already in the fallback chain. Vision configs (Qwen2-VL
        // / Qwen2.5-VL / SigLIP) carry `vision_norm_eps` as a scalar;
        // `rms_norm_eps`'s JSON-only fallback chain misses that, so
        // vision-prefixed accessors override eps via the
        // `vision_norm_eps` scalar lookup once routed to a vision
        // loader at codegen time.
        let eps = rms_norm_eps(model);
        FieldLoad::LayerNorm(prefixes.into_iter().next().unwrap(), eps)
    } else if is_linear {
        // Tied-embedding special case: if this accessor is the
        // `lm_head` and the model's config.json has
        // `tie_word_embeddings: true`, there's no lm_head weight in
        // safetensors — its buffer is shared with `embed_tokens`.
        // HF convention: every decoder-only model that ties them
        // calls the sharing field `embed_tokens`; the macro looks
        // up that field by name.
        if accessor.name == "lm_head"
            && prefixes.len() == 1
            && (prefixes[0] == "lm_head" || prefixes[0].ends_with(".lm_head"))
            && tie_word_embeddings(model)
        {
            return FieldLoad::LinearTiedToEmbedding(syn::Ident::new(
                "embed_tokens",
                proc_macro2::Span::call_site(),
            ));
        }
        if prefixes.len() == 1 {
            // `kind: "raw_linear"` opt-in (per the per-arch
            // weights manifest): the underlying tensor is an
            // `nn.Parameter`, not an `nn.Linear`. Looked up off
            // the DSL-side path of the source weight (same
            // convention as `manifest.lookup`). Fused-concat
            // RawLinear is not a real PyTorch shape — only the
            // single-source branch routes here.
            let segments = program.weights.path(accessor.source_weights[0].0);
            if matches!(
                manifest.kind(segments),
                crate::weights_manifest::ManifestEntryKind::RawLinear
            ) {
                return FieldLoad::RawLinear(prefixes.into_iter().next().unwrap());
            }
            FieldLoad::LinearDense(prefixes.into_iter().next().unwrap())
        } else {
            FieldLoad::LinearConcat(prefixes)
        }
    } else {
        // Unknown weight type — fall back to raw Embedding-shaped
        // load. New weight types (LayerNorm with bias, etc.) land
        // as new arms here alongside their Impl's `rust_type`.
        assert_eq!(
            prefixes.len(),
            1,
            "unhandled weight type `{ty}` for accessor `{}`",
            accessor.name,
        );
        FieldLoad::Embedding(prefixes.into_iter().next().unwrap())
    }
}

fn rms_norm_eps(model: &ModelParams) -> f32 {
    // HF configs carry `rms_norm_eps` (RMSNorm convention used by
    // Llama/Qwen2/Mistral/etc.) or `layer_norm_eps` (CohereLayerNorm
    // convention used by CommandR). Both name the same numerical
    // role — the eps inside the row-normalization kernel — so the
    // RmsNorm-typed weight loader accepts either. Reads the JSON
    // directly because the bounds map captures only integers.
    let fallback: f32 = 1e-5;
    let Ok(s) = std::fs::read_to_string(&model.source_path) else {
        return fallback;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) else {
        return fallback;
    };
    let read = |key| v.get(key).and_then(|x| x.as_f64()).map(|x| x as f32);
    read("rms_norm_eps")
        .or_else(|| read("layer_norm_eps"))
        .or_else(|| read("vision_norm_eps"))
        .unwrap_or(fallback)
}

fn tie_word_embeddings(model: &ModelParams) -> bool {
    // `ModelParams::tie_word_embeddings` is populated by the config
    // loader from the HF JSON; read directly rather than re-parsing
    // the file here.
    model.tie_word_embeddings
}

/// Aggregate every unique WeightAccessor across every workload
/// bucket's SFUF. Errors on name collisions with conflicting
/// `rust_type`s.
fn collect_accessors(
    program: &Program,
    fuf: &Fuf,
    sfufs: &WorkloadAssignments,
    lib: &ImplementationLibrary,
    _model_for_trace: &ModelParams,
) -> Result<Vec<WeightAccessor>, TokenStream> {
    // name → (first-seen accessor, rust_type string for collision check).
    let mut by_name: BTreeMap<String, (WeightAccessor, String)> = BTreeMap::new();
    let mut conflicts: Vec<String> = Vec::new();

    for sfuf in sfufs.per_workload.values() {
        for sg in sfuf.subgraphs() {
            let imp_id = sfuf
                .impl_of(sg)
                .expect("solver committed an impl for every subgraph");
            let claimed = sfuf.tiles_in_subgraph(sg);
            let imp = lib.get(imp_id);
            for acc in imp.required_weights(&claimed, fuf, program) {
                if std::env::var("FERRITE_GGUF_BUILD_TRACE").is_ok() {
                    let storages: Vec<_> = acc
                        .source_weights
                        .iter()
                        .map(|(wid, _)| {
                            crate::quantization::storage_format_for_weight(
                                program,
                                fuf,
                                *wid,
                                _model_for_trace,
                            )
                        })
                        .collect();
                    eprintln!(
                        "[ggml-build] collect_accessors model={} impl={} acc={} sources={} storages={:?}",
                        _model_for_trace.source_stem,
                        imp.name(),
                        acc.name,
                        acc.source_weights.len(),
                        storages,
                    );
                }
                let key = acc.name.to_string();
                let ty_str = acc.rust_type.to_string();
                by_name
                    .entry(key.clone())
                    .and_modify(|(_, existing_ty)| {
                        if *existing_ty != ty_str {
                            conflicts.push(format!(
                                "Weights field `{key}` declared with \
                                 conflicting types: `{existing_ty}` vs `{ty_str}`"
                            ));
                        }
                    })
                    .or_insert((acc.clone(), ty_str));
            }
        }
    }

    if !conflicts.is_empty() {
        let msg = conflicts.join("\n");
        return Err(quote! { compile_error!(#msg); });
    }
    Ok(by_name.into_values().map(|(a, _)| a).collect())
}

/// Emit a `fingerprint_matches(gw)` method body — the per-variant
/// check the arch-level dispatcher uses to auto-detect which
/// compiled model a runtime `GpuWeights` corresponds to. All values
/// are baked at macro-expansion time from `model.bounds` +
/// `model.quantization`; the runtime cost is a handful of
/// `gw.contains` / `gw.tensor_info` lookups.
///
/// Checks:
/// 1. **Embedding shape** matches `(vocab_size, hidden_size)` from
///    config.json. Rules out arches with a different width or vocab.
/// 2. **Last-layer tensor present** (`model.layers.{N-1}.self_attn.q_proj.<suffix>`)
///    where `suffix` is `qweight` for AWQ/GPTQ, `weight` for dense.
/// 3. **Next-layer tensor absent** (same name with layer `N`). Rules
///    out larger compiled variants with the same suffix.
/// 4. **Opposite-suffix tensor absent**. Rules out the other
///    dense-vs-quant twin (GPTQ vs AWQ share the qweight suffix —
///    they're distinguished by check #5).
/// 5. **Quant-format shape check**. When the compiled variant is
///    AWQ or GPTQ, the `qweight` shape discriminates:
///    `[K, N/8]` is AWQ, `[K/8, N]` is GPTQ. For the q_proj
///    specifically both dims are `hidden_size`, so checking
///    `shape[0] == hidden_size` (AWQ) vs `shape[0] == hidden_size/8`
///    (GPTQ) is enough. Without this check a GPTQ model would
///    fingerprint-match an AWQ variant of the same arch+width and
///    the emitted `load_awq` loader would panic in
///    `awq_to_marlin_zero_points`.
fn emit_fingerprint_check(
    model: &ModelParams,
    manifest: &crate::weights_manifest::WeightsManifest,
    tp_world_size: u8,
) -> TokenStream {
    let num_hidden_layers = *model
        .bounds
        .get("num_hidden_layers")
        .unwrap_or_else(|| panic!("model `{}` missing `num_hidden_layers`", model.source_stem));
    let hidden_size = *model
        .bounds
        .get("hidden_size")
        .unwrap_or_else(|| panic!("model `{}` missing `hidden_size`", model.source_stem));
    let vocab_size = *model
        .bounds
        .get("vocab_size")
        .unwrap_or_else(|| panic!("model `{}` missing `vocab_size`", model.source_stem));

    // Pick the on-disk tensor suffix per compiled variant's
    // `quantization_config`. AutoGPTQ + AWQ both ship `.qweight`;
    // compressed-tensors INT4 ships `.weight_packed` with the axes
    // transposed. Dense ships `.weight`. bitsandbytes ships U8
    // `.weight` alongside a sibling `.weight.absmax` that's unique
    // to its storage layout — use that as the positive sniff so
    // it's disjoint from dense bf16 `.weight`. The `opposite_suffix`
    // is the negative check: if a compiled dense variant sees
    // `.qweight`, that's a quant model in disguise and the
    // fingerprint should miss.
    let (suffix, opposite_suffix) = match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(crate::quantization::QuantMethod::Gptq {
            layout: crate::quantization::GptqLayout::WeightPacked,
            ..
        }) => ("weight_packed", "weight"),
        Some(crate::quantization::QuantMethod::Bnb4 { .. }) => ("weight.absmax", "qweight"),
        Some(crate::quantization::QuantMethod::Fp8 { .. }) => ("weight_scale", "qweight"),
        // GGUF ships the same `.weight` suffix as dense (the loader
        // stores the quantized tensor under the HF-style name) — the
        // disambiguator vs Dense is `gw.is_gguf()` (added below in
        // qweight_shape_gate) plus the absence of fp8/bnb4 markers
        // at `.weight_scale` / `.weight.absmax`.
        Some(crate::quantization::QuantMethod::Ggml) => ("weight", "qweight"),
        Some(_) => ("qweight", "weight"),
        None => ("weight", "qweight"),
    };

    let last_layer = num_hidden_layers.saturating_sub(1);
    // Per-arch decoder root for fingerprint-tensor names: `model` for
    // text-only and Qwen-style VL, `<prefix>.model` for arches whose
    // variant config sets `decoder_safetensors_prefix` (Gemma3-MM nests
    // text decoder weights under `language_model.<...>`).
    let dec_root: String = match model.decoder_safetensors_prefix.as_deref() {
        Some(prefix) => format!("{prefix}.model"),
        None => "model".to_string(),
    };
    // Pick a layered tensor that ACTUALLY EXISTS ON DISK to use as the
    // fingerprint sniff. Packed parents come first (Phi-3 ships
    // `self_attn.qkv_proj.weight` on disk; ModernBERT ships
    // `attn.Wqkv.weight`; the per-slice virtual entries get carved at
    // load time, AFTER fingerprint matching). Then MLA archs (DeepSeek
    // V3) which use `q_a_proj` instead of `q_proj`. Then plain
    // `self_attn.q_proj` (llama-style decoder fleet) and finally
    // `attn.q_proj` (encoder-style without `self_` prefix). Without
    // this, the fingerprint misses and ferrite returns Ok(None) even
    // when it has a compiled variant for this arch.
    let fp_leaf_owned: String = if let Some(parent) = manifest
        .packed_splits
        .iter()
        .find(|(_, children)| {
            children
                .iter()
                .any(|c| c == "self_attn.q_proj" || c == "attn.q_proj")
        })
        .map(|(k, _)| k.clone())
    {
        parent
    } else if manifest.entries.contains_key("self_attn.q_a_proj") {
        "self_attn.q_a_proj".to_string()
    } else if manifest.entries.contains_key("self_attn.q_proj") {
        "self_attn.q_proj".to_string()
    } else if manifest.entries.contains_key("attn.q_proj") {
        "attn.q_proj".to_string()
    } else {
        "self_attn.q_proj".to_string()
    };
    let fp_leaf: &str = fp_leaf_owned.as_str();
    let last_tensor = format!("{dec_root}.layers.{last_layer}.{fp_leaf}.{suffix}");
    let one_past_tensor = format!("{dec_root}.layers.{num_hidden_layers}.{fp_leaf}.{suffix}");
    let opposite_tensor = format!("{dec_root}.layers.0.{fp_leaf}.{opposite_suffix}");
    // BNB4 checkpoints ship the U8-packed nibbles at `.weight`
    // (same suffix as dense bf16 weights) with a sibling
    // `.weight.absmax` that's unique to bitsandbytes. Dense + AWQ
    // + GPTQ + CT variants must reject when absmax is present; the
    // BNB4 variant itself uses `.weight.absmax` as the positive
    // sniff and doesn't need the rejection.
    let bnb4_exclusion = matches!(
        model.quantization.as_ref().map(|qc| &qc.method),
        Some(crate::quantization::QuantMethod::Bnb4 { .. })
    );
    let bnb4_marker_tensor = format!("{dec_root}.layers.0.{fp_leaf}.weight.absmax");
    let bnb4_marker_tensor = bnb4_marker_tensor.as_str();
    // FP8 checkpoints ship `.weight` (FP8E4M3 bytes — same suffix
    // as dense bf16) alongside a sibling `.weight_scale`. Dense /
    // AWQ / native-GPTQ / BNB4 variants must reject when
    // `.weight_scale` is present; the FP8 variant itself keys off
    // this tensor in its suffix-based checks and doesn't need the
    // rejection. Compressed-tensors INT4 (`GptqLayout::WeightPacked`)
    // ALSO ships `.weight_scale` (as the group-scale tensor) — its
    // own qweight-shape gate on `.weight_packed` already makes it
    // disjoint from FP8, so skip the fp8 exclusion for it to avoid
    // false-rejecting CT-INT4 checkpoints.
    let fp8_exclusion = matches!(
        model.quantization.as_ref().map(|qc| &qc.method),
        Some(crate::quantization::QuantMethod::Fp8 { .. })
            | Some(crate::quantization::QuantMethod::Gptq {
                layout: crate::quantization::GptqLayout::WeightPacked,
                ..
            })
    );
    let fp8_marker_tensor = format!("{dec_root}.layers.0.{fp_leaf}.weight_scale");
    let fp8_marker_tensor = fp8_marker_tensor.as_str();

    let hidden_lit = proc_macro2::Literal::usize_unsuffixed(hidden_size as usize);
    // GGUF's on-disk loader (`GgufGpuWeights::load`) pre-shards
    // `ShardDim0` tensors — including `embed_tokens` — at file-read
    // time (`ferrite-kernels/src/ggml.rs::gguf_shard_kind_for_hf_name`).
    // At tp>1 the per-rank embed shape is `[vocab_size / tp, hidden]`,
    // so the fingerprint's vocab literal must match the sharded dim-0.
    // Safetensors variants keep the full tensor in memory (sharding
    // happens inside the codegen-emitted `_sharded` load helpers AFTER
    // `fingerprint_matches` runs), so their vocab literal stays whole.
    let vocab_for_fp = match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(crate::quantization::QuantMethod::Ggml) if tp_world_size > 1 => {
            vocab_size / (tp_world_size as u64)
        }
        _ => vocab_size,
    };
    let vocab_lit = proc_macro2::Literal::usize_unsuffixed(vocab_for_fp as usize);

    // HF-config disambiguator: variants of the same arch that share
    // on-disk tensor shapes (Phi-3-mini-4k vs Phi-3.5-mini-128k) only
    // differ in `max_position_embeddings` and `rope_scaling.type`.
    // Bake the manifest's values here and reject at fingerprint time
    // when the caller-supplied `HfFingerprint` contradicts them.
    // `None` in the runtime view is permissive (caller didn't supply
    // the hint); a `Some(x)` that disagrees with the manifest's
    // compile-time literal is a hard reject.
    let max_pos_lit = model
        .bounds
        .get("max_position_embeddings")
        .map(|&v| proc_macro2::Literal::u64_unsuffixed(v));
    let max_pos_check: TokenStream = match max_pos_lit {
        Some(lit) => quote! {
            if let Some(mp) = hf.max_position_embeddings
                && mp != #lit
            {
                return false;
            }
        },
        None => quote! {},
    };
    let rope_scaling_expected: Option<&'static str> = match &model.rope_scaling {
        Some(crate::config::RopeScaling::Llama3 { .. }) => Some("llama3"),
        Some(crate::config::RopeScaling::LongRope { .. }) => Some("longrope"),
        Some(crate::config::RopeScaling::Yarn { .. }) => Some("yarn"),
        // Qwen2-VL / Qwen2.5-VL: `extract_rope_scaling` doesn't fold
        // `"mrope"` into a `RopeScaling` variant (it doesn't drive the
        // text-side `RotaryCache` — `mrope_section` lives on
        // `CanonicalParams::MROPE_SECTION` instead), but the fingerprint
        // still has to expect `Some("mrope")` from the live HF config
        // or the variant will reject its own checkpoint.
        None if model.mrope_section.is_some() => Some("mrope"),
        None => None,
    };
    let rope_scaling_check: TokenStream = match rope_scaling_expected {
        Some(kind) => quote! {
            // Manifest declares a non-trivial scaling; reject a
            // checkpoint whose HF config has a different (or absent)
            // rope_scaling.type.
            match hf.rope_scaling_type {
                Some(t) if t == #kind => {}
                None => {} // permissive when caller didn't supply it
                Some(_) => return false,
            }
        },
        None => quote! {
            // Manifest has no scaling; reject a checkpoint whose HF
            // config declares one (longrope / llama3). The
            // alphabetically-earlier variant's fingerprint would
            // otherwise win for a scaled checkpoint and bake the
            // wrong RoPE into its Weights.
            if hf.rope_scaling_type.is_some() {
                return false;
            }
        },
    };

    // Content-hash discriminator: reject checkpoints whose
    // `rope_scaling` JSON (factor vectors included) doesn't
    // bit-identically match the manifest's. Discriminates
    // Phi-3.5-mini vs Phi-3-mini-128k, Phi-4-mini-instruct vs
    // Phi-4-mini-reasoning, etc. — same type + max_pos, different
    // short/long_factor values. Permissive when the executor
    // didn't compute the hash (`None`), strict otherwise.
    let rope_scaling_hash_check: TokenStream = match model.rope_scaling_hash {
        Some(hash) => {
            let lit = proc_macro2::Literal::u64_unsuffixed(hash);
            quote! {
                if let Some(h) = hf.rope_scaling_hash
                    && h != #lit
                {
                    return false;
                }
            }
        }
        None => quote! {},
    };

    // Per-format qweight-shape gate. AWQ/GPTQ/CT all pack 4-bit
    // weights but into different axis orders — this gate is the
    // only way to distinguish compiled variants of the same
    // arch+size that differ only in quantization method/layout.
    //
    //   AWQ            `.qweight`       [K, N/8]     → shape[0] == hidden
    //   GPTQ  native   `.qweight`       [K/8, N]     → shape[0] == hidden/8
    //   GPTQ  CT       `.weight_packed` [N, K/8]     → shape[0] == hidden (== N for q_proj, N==K)
    //
    // For q_proj specifically `N == K == hidden_size`, so the AWQ
    // and CT gates coincide on `shape[0] == hidden_size`. The
    // `suffix` picked above (`qweight` vs `weight_packed`) is what
    // makes them disjoint — a CT model doesn't ship `.qweight`.
    let qweight_shape_gate: TokenStream = match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(crate::quantization::QuantMethod::Awq { .. }) => {
            let k_lit = hidden_lit.clone();
            quote! {
                match gw.tensor_info(#last_tensor) {
                    Some((shape, _))
                        if shape.len() == 2 && shape[0] == #k_lit => {}
                    _ => return false,
                }
            }
        }
        Some(crate::quantization::QuantMethod::Gptq {
            layout: crate::quantization::GptqLayout::Qweight,
            ..
        }) => {
            let k_packed = proc_macro2::Literal::usize_unsuffixed(hidden_size as usize / 8);
            quote! {
                match gw.tensor_info(#last_tensor) {
                    Some((shape, _))
                        if shape.len() == 2 && shape[0] == #k_packed => {}
                    _ => return false,
                }
            }
        }
        Some(crate::quantization::QuantMethod::Gptq {
            layout: crate::quantization::GptqLayout::WeightPacked,
            ..
        }) => {
            // compressed-tensors `.weight_packed` is [N, K/8]. For
            // q_proj N == hidden_size, so shape[0] == hidden.
            let n_lit = hidden_lit.clone();
            quote! {
                match gw.tensor_info(#last_tensor) {
                    Some((shape, _))
                        if shape.len() == 2 && shape[0] == #n_lit => {}
                    _ => return false,
                }
            }
        }
        Some(crate::quantization::QuantMethod::Bnb4 { .. }) => {
            // bitsandbytes fingerprint sniffs the `.weight.absmax`
            // sibling itself — presence is sufficient to distinguish
            // from dense bf16 `.weight`. No shape check: absmax
            // length varies with blocksize (64 default) and is
            // per-model, not worth baking into the compile-time
            // fingerprint.
            quote! {}
        }
        Some(crate::quantization::QuantMethod::Fp8 { .. }) => {
            // FP8 fingerprint sniffs `.weight_scale` on the first
            // q_proj — presence + shape distinguishes FP8 from every
            // other variant. The `bnb4_exclusion` block below also
            // rejects when `.weight.absmax` is present so FP8
            // doesn't fingerprint-match a BNB4 checkpoint of the
            // same arch+size.
            quote! {}
        }
        Some(crate::quantization::QuantMethod::Ggml) | None => quote! {},
    };

    // Backing-store reject: every non-Ggml variant must reject a
    // GGUF-backed `GpuWeights`, and the Ggml variant must require
    // one. Without this the dense variant's shape-only fingerprint
    // matches a GGUF (since `gw.contains` checks the quantized map)
    // and the dense load body's fused-QKV path fires on Ggml
    // weights — panicking at runtime when a Cutlass instr calls
    // `dense_weight()` on a `LinearLayer::GgmlConcat`.
    let is_gguf_gate: TokenStream = match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(crate::quantization::QuantMethod::Ggml) => quote! {
            if !gw.is_gguf() {
                return false;
            }
        },
        _ => quote! {
            if gw.is_gguf() {
                return false;
            }
        },
    };

    // GPTQ-Qweight desc_act disambiguation. With overlay fan-out we
    // synthesize BOTH `gptq-sym` (desc_act=false) and
    // `gptq-sym-desc_act` (desc_act=true) variants per dense base —
    // both have the same `.qweight [K/8, N]` shape gate above. The
    // `.g_idx` tensor is what the runtime actually needs to
    // disambiguate: AutoGPTQ ships it iff desc_act=true (it encodes
    // the activation-order permutation the loader passes to
    // `gptq_repack_into`). desc_act=false repos either omit `.g_idx`
    // or carry it inert; we treat presence as the signal that
    // selects the desc_act=true variant. Without this disambiguation
    // the alphabetically-earlier `gptq-sym` would win for a
    // desc_act=true repo, threading `desc_act=false` through
    // `MarlinFormat::Gptq` and producing garbage weights.
    let g_idx_disambiguation: TokenStream = match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(crate::quantization::QuantMethod::Gptq {
            desc_act,
            layout: crate::quantization::GptqLayout::Qweight,
            ..
        }) => {
            let g_idx_tensor_owned = format!("{dec_root}.layers.0.self_attn.q_proj.g_idx");
            let g_idx_tensor = g_idx_tensor_owned.as_str();
            if *desc_act {
                quote! {
                    if !gw.contains(#g_idx_tensor) {
                        return false;
                    }
                }
            } else {
                quote! {
                    if gw.contains(#g_idx_tensor) {
                        return false;
                    }
                }
            }
        }
        _ => quote! {},
    };

    // FP8 `activation_scheme` disambiguation. Overlay fan-out may
    // synthesize both `fp8-dynamic-per-tensor` and
    // `fp8-static-per-tensor` variants for a single dense base; they
    // share the same `.weight_scale` suffix gate above. The on-disk
    // signal that actually distinguishes them is the per-projection
    // `.input_scale` tensor — neuralmagic / RedHatAI static FP8
    // checkpoints carry it (it's the pre-calibrated per-tensor
    // activation scale), dynamic checkpoints omit it. Without this
    // disambiguation the alphabetically-earlier dynamic variant
    // would win for a static checkpoint and `Fp8Linear::forward`
    // would fall into the per-token dynamic-quant path instead of
    // the pre-calibrated static path.
    let input_scale_disambiguation: TokenStream =
        match model.quantization.as_ref().map(|qc| &qc.method) {
            Some(crate::quantization::QuantMethod::Fp8 {
                scheme: crate::quantization::Fp8ActivationScheme::Static,
                block_size: None,
            }) => {
                let input_scale_tensor_owned =
                    format!("{dec_root}.layers.0.self_attn.q_proj.input_scale");
                let input_scale_tensor = input_scale_tensor_owned.as_str();
                quote! {
                    if !gw.contains(#input_scale_tensor) {
                        return false;
                    }
                }
            }
            Some(crate::quantization::QuantMethod::Fp8 {
                scheme: crate::quantization::Fp8ActivationScheme::Dynamic,
                block_size: None,
            }) => {
                let input_scale_tensor_owned =
                    format!("{dec_root}.layers.0.self_attn.q_proj.input_scale");
                let input_scale_tensor = input_scale_tensor_owned.as_str();
                quote! {
                    if gw.contains(#input_scale_tensor) {
                        return false;
                    }
                }
            }
            _ => quote! {},
        };

    // FP8 blockwise disambiguation. Overlay fan-out may synthesize
    // both `fp8-*-per-tensor` and `fp8-block-*` variants for a single
    // dense base; they share the `.weight_scale` positive sniff. The
    // on-disk signal that distinguishes them is the scale tensor's
    // shape — per-tensor ships a scalar `[1]` or per-channel `[N, 1]`,
    // blockwise ships a 2-D `[ceil(N/bn), ceil(K/bk)]` with bk > 1.
    // Some blockwise repos name the tensor `.weight_scale_inv`
    // (DeepSeek-V3, Qwen3-MoE); its mere presence is a sufficient
    // block signal too. Without this, the alphabetically-earlier
    // per-tensor variant would win for a block checkpoint and
    // `Fp8Linear::load` would fail at runtime reading a 2-D scale
    // expecting a scalar.
    let block_disambiguation: TokenStream = match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(crate::quantization::QuantMethod::Fp8 {
            block_size: Some(_),
            ..
        }) => {
            // MLA archs ship `q_a_proj` instead of `q_proj` — match
            // the leaf the fingerprint already chose above so V3 / K2
            // FP8-block fixtures aren't silently rejected.
            let inv_tensor = format!("{dec_root}.layers.0.{fp_leaf}.weight_scale_inv");
            let scale_tensor = format!("{dec_root}.layers.0.{fp_leaf}.weight_scale");
            let inv_tensor = inv_tensor.as_str();
            let scale_tensor = scale_tensor.as_str();
            quote! {
                // Block variant: accept if `.weight_scale_inv` exists,
                // or if `.weight_scale` is 2-D with more than one
                // column (the per-tensor and per-channel layouts have
                // `shape.len() == 1` or `shape[1] == 1` respectively).
                let __fp8_is_block = gw.contains(#inv_tensor)
                    || matches!(
                        gw.tensor_info(#scale_tensor),
                        Some((shape, _)) if shape.len() == 2 && shape[1] > 1,
                    );
                if !__fp8_is_block {
                    return false;
                }
            }
        }
        Some(crate::quantization::QuantMethod::Fp8 {
            block_size: None, ..
        }) => {
            let inv_tensor = format!("{dec_root}.layers.0.{fp_leaf}.weight_scale_inv");
            let scale_tensor = format!("{dec_root}.layers.0.{fp_leaf}.weight_scale");
            let inv_tensor = inv_tensor.as_str();
            let scale_tensor = scale_tensor.as_str();
            quote! {
                // Per-tensor variant: reject block checkpoints.
                if gw.contains(#inv_tensor) {
                    return false;
                }
                if let Some((shape, _)) = gw.tensor_info(#scale_tensor)
                    && shape.len() == 2
                    && shape[1] > 1
                {
                    return false;
                }
            }
        }
        _ => quote! {},
    };

    // The embedding tensor's on-disk path varies per arch — llama uses
    // `model.embed_tokens.weight`, ModernBERT uses
    // `model.embeddings.tok_embeddings.weight`. The manifest entry whose
    // shape is `[vocab_size, hidden_size]` is the embedding table; use
    // its key (with `model.` prefix + `.weight` suffix) as the
    // fingerprint sniff. Fall back to the llama-style path when no
    // entry matches, preserving the previous behavior for any arch
    // whose manifest predates this generalization.
    let embed_path: String = manifest
        .entries
        .iter()
        .find(|(_, shape)| {
            shape.len() == 2
                && matches!(&shape[0], crate::shape::Dim::Bound(s) if s == "vocab_size")
                && matches!(&shape[1], crate::shape::Dim::Bound(s) if s == "hidden_size")
        })
        .map(|(k, _)| format!("{dec_root}.{k}.weight"))
        .unwrap_or_else(|| format!("{dec_root}.embed_tokens.weight"));
    let embed_path_lit = proc_macro2::Literal::string(embed_path.as_str());

    quote! {
        /// Per-variant compile-time fingerprint check. See
        /// macro's `emit_fingerprint_check` for the rules.
        #[cfg(feature = "cuda")]
        pub fn fingerprint_matches(
            gw: &::ferrite_cuda_core::weights::GpuWeights,
            hf: ::ferrite_forward::HfFingerprint<'_>,
        ) -> bool {
            match gw.tensor_shape_any(#embed_path_lit) {
                Some(ref shape)
                    if shape.len() >= 2
                        && shape[0] == #vocab_lit
                        && shape[1] == #hidden_lit => {}
                _ => return false,
            }
            if !gw.contains(#last_tensor) {
                return false;
            }
            if gw.contains(#one_past_tensor) {
                return false;
            }
            if gw.contains(#opposite_tensor) {
                return false;
            }
            // BNB4 marker exclusion — reject dense/AWQ/GPTQ/CT
            // fingerprints when the checkpoint ships BNB4's
            // `.weight.absmax` sibling. The BNB4 variant itself
            // keys off this tensor in its suffix-based checks
            // above, so this exclusion runs only for non-BNB4
            // variants.
            if !#bnb4_exclusion && gw.contains(#bnb4_marker_tensor) {
                return false;
            }
            // FP8 marker exclusion — reject dense/AWQ/GPTQ/CT/BNB4
            // fingerprints when the checkpoint ships FP8's
            // `.weight_scale` sibling. The FP8 variant itself keys
            // off `.weight_scale` as its positive suffix, so this
            // exclusion runs only for non-FP8 variants.
            if !#fp8_exclusion && gw.contains(#fp8_marker_tensor) {
                return false;
            }
            #qweight_shape_gate
            #is_gguf_gate
            #g_idx_disambiguation
            #input_scale_disambiguation
            #block_disambiguation
            #max_pos_check
            #rope_scaling_check
            #rope_scaling_hash_check
            true
        }
    }
}

/// `emit_weights_struct` has two modes. `Canonical` defines its own
/// `pub struct Weights { ... }`; `Shim { canonical }` aliases the
/// struct to the canonical sibling module's Weights (for cross-
/// variant forward-fn dedup) while still emitting this variant's
/// own `load` + `fingerprint_matches` bodies.
pub(crate) enum WeightsEmitMode<'a> {
    Canonical,
    Shim { canonical: &'a Ident },
}

/// Emit the `Weights` struct definition (or alias) + its `load` +
/// `fingerprint_matches` free fns.
#[allow(clippy::too_many_arguments)]
fn emit_weights_struct(
    program: &Program,
    fuf: &Fuf,
    sfufs: &WorkloadAssignments,
    lib: &ImplementationLibrary,
    model: &ModelParams,
    manifest: &crate::weights_manifest::WeightsManifest,
    mode: WeightsEmitMode<'_>,
    tp_world_size: u8,
    emit_fingerprint: bool,
) -> TokenStream {
    let accessors = match collect_accessors(program, fuf, sfufs, lib, model) {
        Ok(a) => a,
        Err(err) => return err,
    };

    // Storage-format guard: a given accessor's `rust_type` must be
    // compatible with every one of its source weights' storage
    // formats. The allowed pairs today:
    //   `LinearLayer`   ↔ `Dense` | `Ggml`
    //   `Embedding`     ↔ `Dense`
    //   `RmsNorm`       ↔ `Dense`
    //   `MarlinLinear`  ↔ `Awq { .. }` | `Gptq { .. }`
    //   `Bnb4bitLinear` ↔ `Bnb4 { .. }`
    //   `Fp8Linear`     ↔ `Fp8 { .. }`
    //
    // GGUF rides on the `LinearLayer` accessor type because the
    // runtime enum already has a `Ggml(Box<GgmlLinear>)` arm that
    // dispatches at forward time — so codegen produces the same
    // accessor field type for Dense and Ggml; the FieldLoad arm
    // picks `take_quantized_linear` vs `take` based on storage.
    //
    // Any other pair means the solver picked an impl whose
    // declared accessor type doesn't match the bits on disk — a
    // matcher bug. Fail at macro-expansion time so new quant
    // formats can't slip through without a matching FieldLoad arm.
    for a in &accessors {
        let ty = a.rust_type.to_string().replace(' ', "");
        let accessor_is_marlin = ty.ends_with("::MarlinLinear")
            || ty == "MarlinLinear"
            || ty.ends_with("layers::MarlinLinear");
        let accessor_is_bnb4 = ty.ends_with("::Bnb4bitLinear")
            || ty == "Bnb4bitLinear"
            || ty.ends_with("layers::Bnb4bitLinear");
        // FP8 accessors come through `fp8_accessor_type_for`, which
        // post-Fp8AnyLinear-unblocker always returns `Fp8AnyLinear`.
        // The storage-format guard accepts that type for both
        // per-tensor / per-channel (`block_size: None`) and
        // blockwise (`block_size: Some(_)`) FP8 storage — the
        // Fp8AnyLinear enum dispatches at runtime on the loaded
        // variant.
        let accessor_is_fp8_any = ty.ends_with("::Fp8AnyLinear")
            || ty == "Fp8AnyLinear"
            || ty.ends_with("layers::Fp8AnyLinear")
            || ty.ends_with("::DeepSeekV2Fp8BlockMoELayer")
            || ty == "DeepSeekV2Fp8BlockMoELayer"
            || ty.ends_with("layers_moe::DeepSeekV2Fp8BlockMoELayer");
        for (wid, _idx) in &a.source_weights {
            let fmt = crate::quantization::storage_format_for_weight(program, fuf, *wid, model);
            let ok = matches!(
                (
                    &fmt,
                    accessor_is_marlin,
                    accessor_is_bnb4,
                    accessor_is_fp8_any
                ),
                (
                    crate::quantization::StorageFormat::Dense,
                    false,
                    false,
                    false
                ) | (
                    crate::quantization::StorageFormat::Awq { .. },
                    true,
                    false,
                    false
                ) | (
                    crate::quantization::StorageFormat::Gptq { .. },
                    true,
                    false,
                    false
                ) | (
                    crate::quantization::StorageFormat::Bnb4 { .. },
                    false,
                    true,
                    false
                ) | (
                    crate::quantization::StorageFormat::Fp8 { .. },
                    false,
                    false,
                    true
                ) | (
                    crate::quantization::StorageFormat::Ggml,
                    false,
                    false,
                    false
                ),
            );
            if !ok {
                let dotted = program.weights.path(*wid).join(".");
                let msg = format!(
                    "model `{stem}`: weight `{dotted}` has storage ({fmt:?}) that \
                     doesn't match accessor `{name}` (type `{ty}`). Add a quant-aware \
                     Impl + FieldLoad arm for this pair.",
                    stem = model.source_stem,
                    name = a.name,
                );
                return quote! { compile_error!(#msg); };
            }
        }
    }

    // Group accessors by base for the Vec<T> compression. Layered
    // groups collapse to one `pub <base>: Vec<T>` field + one
    // `(0..N).map(|layer| …).collect::<Result<Vec<_>>>()?` Vec-build
    // in `load_with`; unindexed groups keep the per-accessor field
    // and per-FieldLoad let. The same group list also drives
    // `emit_weights_accessor_methods`, so all three sites stay in
    // sync — adding a new accessor only changes the inputs here.
    let groups = group_accessors_by_base(&accessors);

    let fields: Vec<TokenStream> = groups
        .iter()
        .flat_map(|g| {
            let ty = &g.rust_type;
            match g.kind {
                AccessorGroupKind::Unindexed => {
                    let name = syn::Ident::new(&g.base, proc_macro2::Span::call_site());
                    vec![quote! { pub #name: #ty, }]
                }
                AccessorGroupKind::LayeredContiguous => {
                    let name = syn::Ident::new(&g.base, proc_macro2::Span::call_site());
                    vec![quote! { pub #name: ::std::vec::Vec<#ty>, }]
                }
                AccessorGroupKind::LayeredSparse => g
                    .entries
                    .iter()
                    .map(|(_, acc)| {
                        let n = &acc.name;
                        quote! { pub #n: #ty, }
                    })
                    .collect(),
            }
        })
        .collect();

    // Emit each group as its own let-binding in the load body.
    // Layered groups become `let <base>: Vec<T> = (0..N).map(…)
    // .collect()?;`. Unindexed groups keep the existing per-FieldLoad
    // let (computed by `plan_field_load` from the single accessor).
    // Order matters: a tied `lm_head` reads `embed_tokens.weight`,
    // and `groups` iterates BTreeMap-sorted by base, which puts
    // `embed_tokens` before `lm_head` alphabetically.
    //
    // Scan plans for the prelude-needed flags (any_marlin / any_bnb4
    // / any_fp8) over the FULL accessor list — keeps the prelude
    // logic identical to the pre-grouping version. The prelude
    // bindings (`__marlin_ws`, `__bnb_code`, `__fp8_dtype`, …) are
    // captured by the layered closures via lexical scope.
    let plans: Vec<FieldLoad> = accessors
        .iter()
        .map(|a| plan_field_load(a, program, fuf, model, manifest))
        .collect();
    // Per-arch decoder root for embed_tokens probes etc. `model` for
    // text-only and Qwen-style VL, `<prefix>.model` for arches whose
    // variant config sets `decoder_safetensors_prefix` (Gemma3-MM nests
    // text decoder weights under `language_model.<...>`).
    let dec_root_for_emit: String = match model.decoder_safetensors_prefix.as_deref() {
        Some(prefix) => format!("{prefix}.model"),
        None => "model".to_string(),
    };
    let embed_tokens_weight_path: String = format!("{dec_root_for_emit}.embed_tokens.weight");
    let any_marlin = plans
        .iter()
        .any(|p| matches!(p, FieldLoad::MarlinLinear { .. }));
    let any_bnb4 = plans
        .iter()
        .any(|p| matches!(p, FieldLoad::Bnb4Linear { .. }));
    let any_fp8 = plans.iter().any(|p| {
        matches!(
            p,
            FieldLoad::Fp8Linear { .. } | FieldLoad::Fp8BlockLinear { .. }
        )
    });
    // `max(out_features * in_features)` across every BNB4 accessor
    // — sizes the per-model shared dequant scratch buffer. Zero
    // when the model has no BNB4 accessors (the prelude block is
    // then elided entirely).
    let bnb4_max_elements: usize = plans
        .iter()
        .filter_map(|p| match p {
            FieldLoad::Bnb4Linear {
                out_features_per_shard,
                in_features,
                ..
            } => {
                let total_out: usize = out_features_per_shard.iter().map(|o| *o as usize).sum();
                Some(total_out * (*in_features as usize))
            }
            _ => None,
        })
        .max()
        .unwrap_or(0);
    // Map each accessor to its FieldLoad once so the unindexed-let
    // arm and the layered Vec-build arm can both look it up by
    // accessor identity. (`accessors` is sorted, plans was built in
    // the same order, so a name → plan lookup is fine.)
    let plan_by_name: std::collections::BTreeMap<String, &FieldLoad> = accessors
        .iter()
        .zip(plans.iter())
        .map(|(a, p)| (a.name.to_string(), p))
        .collect();

    let is_vision = matches!(program.prelude, crate::classified::Prelude::Vision);
    let lets: Vec<TokenStream> = groups
        .iter()
        .map(|g| emit_group_let(g, &plan_by_name, model, tp_world_size, is_vision))
        .collect();

    // Shared-per-model Marlin prelude: one workspace allocation
    // (GpuTensor is `Copy` — each MarlinLinear captures the same
    // buffer by value), one device-id query. Only planted when at
    // least one accessor resolves to a Marlin FieldLoad (AWQ or
    // GPTQ); dense models skip it so their `Weights::load` body is
    // byte-for-byte identical to before quantization landed.
    let marlin_prelude: TokenStream = if any_marlin {
        quote! {
            let __device = unsafe { ::ferrite_cuda_core::driver::current_device()? };
            let __device_id: i32 = __device as i32;
            let __num_sm = unsafe { ::ferrite_cuda_core::driver::device_get_num_sm(__device)? };
            let __marlin_ws =
                ::ferrite_kernels::layers_quant::alloc_marlin_workspace(__num_sm, stream)?;
        }
    } else {
        quote! {}
    };

    // Shared-per-model BNB4 prelude: upload the NF4/FP4 LUT once
    // (16-entry f32 table) and allocate a single dequant scratch
    // buffer sized to `max(out*in)` across every BNB4 accessor.
    // Every `Bnb4bitLinear` on this device captures both tensors by
    // value (`GpuTensor: Copy`). Elided when the model has no BNB4.
    //
    // Compute dtype for the shared dequant scratch comes from
    // `embed_tokens.weight` — guaranteed present on every arch and
    // always stored in the model's compute dtype (never BNB-packed,
    // per bitsandbytes' default `llm_int8_skip_modules`). Reading
    // it off disk keeps bf16-compute and fp16-compute checkpoints
    // both correct without a config scrape.
    let bnb4_prelude: TokenStream = if any_bnb4 {
        let max_elements = proc_macro2::Literal::usize_unsuffixed(bnb4_max_elements);
        let code_expr = match model.quantization.as_ref().map(|qc| &qc.method) {
            Some(crate::quantization::QuantMethod::Bnb4 {
                quant_type: crate::quantization::BnbQuantType::NF4,
                ..
            }) => quote! { ::ferrite_kernels::layers_quant::NF4_CODE },
            Some(crate::quantization::QuantMethod::Bnb4 {
                quant_type: crate::quantization::BnbQuantType::FP4,
                ..
            }) => quote! { ::ferrite_kernels::layers_quant::FP4_CODE },
            _ => unreachable!("any_bnb4 implies QuantMethod::Bnb4"),
        };
        quote! {
            let __bnb_code =
                ::ferrite_kernels::layers_quant::upload_bnb_code(&#code_expr, stream)?;
            let __bnb_dtype = gw
                .tensor_info(#embed_tokens_weight_path)
                .map(|(_, dt)| dt)
                .unwrap_or(::ferrite_cuda_core::dtype::DType::BF16);
            let __bnb_scratch = ::ferrite_kernels::layers_quant::alloc_bnb_dequant_scratch(
                #max_elements,
                __bnb_dtype,
                stream,
            )?;
        }
    } else {
        quote! {}
    };

    // Shared-per-model FP8 prelude: read the compute dtype from
    // `embed_tokens.weight` (always present, always in compute dtype
    // — never FP8-quantized) and hand it to every `Fp8Linear::load`.
    // Matches the pattern used by the BNB4 prelude for its dequant
    // scratch allocation. Elided when no FP8 accessors are present.
    let fp8_prelude: TokenStream = if any_fp8 {
        quote! {
            let __fp8_dtype = gw
                .tensor_info(#embed_tokens_weight_path)
                .map(|(_, dt)| dt)
                .unwrap_or(::ferrite_cuda_core::dtype::DType::BF16);
        }
    } else {
        quote! {}
    };

    // Manifest-driven packed-tensor splits: archs whose checkpoints
    // ship fused qkv / gate_up under a single on-disk name declare
    // `__packed_splits__` in their `weights.json` (e.g. Phi-3 family).
    // For each (packed_prefix → [target …]) entry, we emit one call
    // per transformer layer that synthesizes the per-slice virtual
    // entries before any `FieldLoad`. Row counts come from looking up
    // each target in the manifest and evaluating the first dim against
    // `model.bounds`. The helper is a no-op when the packed parent
    // isn't present, so models without packed checkpoints are unaffected
    // even if they share the manifest (none do today).
    let is_vision = matches!(program.prelude, crate::classified::Prelude::Vision);
    let packed_splits_prelude: TokenStream = if manifest.packed_splits.is_empty() {
        quote! {}
    } else {
        // Vision configs carry `vision_depth` instead of
        // `num_hidden_layers`; the per-block prefix is
        // `visual.blocks.<L>.` instead of `model.layers.<L>.`.
        let layer_count_key = if is_vision {
            "vision_depth"
        } else {
            "num_hidden_layers"
        };
        let block_prefix_template = if is_vision {
            "visual.blocks"
        } else {
            "model.layers"
        };
        let num_hidden_layers = *model.bounds.get(layer_count_key).unwrap_or_else(|| {
            panic!(
                "model `{}` has `__packed_splits__` but no `{}` bound",
                model.source_stem, layer_count_key,
            )
        }) as usize;
        let mut calls: Vec<TokenStream> = Vec::new();
        for (packed_prefix, targets) in &manifest.packed_splits {
            let mut sized: Vec<(String, u64)> = Vec::with_capacity(targets.len());
            for target in targets {
                let shape = manifest.entries.get(target).unwrap_or_else(|| panic!(
                    "model `{}`: __packed_splits__ target `{target}` not declared in manifest entries",
                    model.source_stem,
                ));
                // Manifest convention is `[in_features, out_features]`
                // (Gemm expects `K = x.last == w.first`), so the
                // on-disk safetensors row count — what
                // `synthesize_packed_row_split_sizes` needs — is
                // `shape.last()`, the out dim. For a 1-D tensor (norm
                // weight), there's no "in" vs "out"; we still take the
                // sole dim, though packed-split targets are always 2-D
                // projection matrices in practice.
                let rows_dim = shape.last().unwrap_or_else(|| {
                    panic!(
                        "model `{}`: __packed_splits__ target `{target}` has empty shape",
                        model.source_stem,
                    )
                });
                let rows =
                    crate::shape::eval_closed_dim(rows_dim, &model.bounds).unwrap_or_else(|| {
                        panic!(
                            "model `{}`: __packed_splits__ target `{target}` out-dim `{:?}` \
                         did not resolve against bounds",
                            model.source_stem, rows_dim,
                        )
                    });
                // The helper takes the leaf suffix under the shared
                // grandparent (e.g. "q_proj" under "self_attn"); strip
                // the parent prefix if the target shares one with the
                // packed prefix (the common case), else pass the full
                // dotted suffix — `synthesize_packed_row_split_sizes`
                // joins grandparent + suffix either way.
                let packed_parent = packed_prefix.rsplit_once('.').map(|(p, _)| p).unwrap_or("");
                let suffix = target
                    .strip_prefix(&format!("{packed_parent}."))
                    .unwrap_or(target)
                    .to_string();
                sized.push((suffix, rows));
            }
            let pairs: Vec<TokenStream> = sized
                .iter()
                .map(|(suffix, rows)| {
                    let rows_lit = proc_macro2::Literal::usize_unsuffixed(*rows as usize);
                    quote! { (#suffix, #rows_lit) }
                })
                .collect();
            let tp_world_lit = proc_macro2::Literal::u8_unsuffixed(tp_world_size);
            calls.push(quote! {
                for __l in 0..#num_hidden_layers {
                    let __pp = ::std::format!(
                        concat!(#block_prefix_template, ".{}.{}"), __l, #packed_prefix,
                    );
                    // `_tp` variant handles the GGUF quantized parent at
                    // tp > 1 — fused `attn_qkv` / `ffn_up` are
                    // replicated on every rank by the GGUF loader's
                    // shard-kind rule table (the fused name isn't in
                    // there), so the packed-splits prelude carves
                    // per-rank views directly. At tp == 1 and for
                    // safetensors parents, delegates to the unsharded
                    // path below.
                    gw.synthesize_packed_row_split_sizes_tp(
                        &__pp,
                        &[ #(#pairs),* ],
                        tp_rank as usize,
                        #tp_world_lit as usize,
                    )?;
                }
            });
        }
        quote! {
            #(#calls)*
        }
    };

    // `Self { a, b, c }` shorthand — fields are the just-bound
    // locals. Unindexed and Vec-compressed groups contribute one
    // ident (the base name); sparse-layered groups contribute one
    // ident per per-layer accessor (matching the per-layer fields
    // in the struct + the per-layer let bindings in the body).
    let field_shorthand: Vec<syn::Ident> = groups
        .iter()
        .flat_map(|g| match g.kind {
            AccessorGroupKind::Unindexed | AccessorGroupKind::LayeredContiguous => {
                vec![syn::Ident::new(&g.base, proc_macro2::Span::call_site())]
            }
            AccessorGroupKind::LayeredSparse => {
                g.entries.iter().map(|(_, acc)| acc.name.clone()).collect()
            }
        })
        .collect();
    // Vision encoders skip the fingerprint emission entirely — they
    // route through the hand-written `FerriteMmRegistration` instead
    // of the inventory-based arch dispatcher, so `fingerprint_matches`
    // is unreachable. Skipping also avoids the `num_hidden_layers` /
    // `hidden_size` / `vocab_size` panics in `emit_fingerprint_check`
    // — vision configs (`vision_*` + `d_model` only) lack those keys.
    let fingerprint_method = if emit_fingerprint {
        emit_fingerprint_check(model, manifest, tp_world_size)
    } else {
        TokenStream::new()
    };

    // Detect whether this arch uses `rotary_local` (dual-rotary,
    // e.g. Gemma3). If so, emit a `rotary_local: RotaryCache` field
    // on Weights + its construction in `load`. The global rotary
    // stays on ForwardCtx; only the local one lives here.
    let uses_rotary_local = fuf.nodes.iter().any(|n| {
        n.inputs.iter().any(|i| {
            matches!(
                i,
                crate::fuf::FufInput::Extern {
                    kind: crate::classified::ExternKind::RotaryLocal,
                    ..
                }
            )
        })
    });

    // Compute dtype the rotary cache must match the model's compute dtype:
    // the rope kernel dispatches on Q/K activation dtype but reinterprets
    // cos/sin bytes through that same plan, so a BF16 cos/sin against F16
    // activations reads the table through the wrong exponent width
    // (5 vs 8 bits).
    //
    // Read at RUNTIME from `embed_tokens.weight`'s on-disk dtype — that
    // tensor is always present and always in the model's compute dtype
    // (never quantized). Baking from the manifest's `torch_dtype` is
    // unsafe: AWQ checkpoints frequently override the base model's dtype
    // (Qwen2.5 base ships as bf16; the `*-Instruct-AWQ` variants ship as
    // f16), and the per-AWQ-variant `quant_config.json` does not change
    // the manifest's compile-time `torch_dtype` literal. Mismatch produces
    // grammatical-but-incoherent output (rope rotations applied through
    // the wrong exponent layout). The manifest's `torch_dtype` survives
    // only as the fallback when the embed tensor isn't visible in the
    // weights table — same pattern used by the BNB4 / FP8 preludes above.
    let rope_dtype_fallback: TokenStream = match model.torch_dtype.as_deref() {
        Some("float16" | "fp16" | "f16" | "half") => {
            quote! { ::ferrite_cuda_core::dtype::DType::F16 }
        }
        _ => quote! { ::ferrite_cuda_core::dtype::DType::BF16 },
    };
    let rotary_prelude: TokenStream = quote! {
        let __rope_dtype = gw
            .tensor_info(#embed_tokens_weight_path)
            .map(|(_, dt)| dt)
            .unwrap_or(#rope_dtype_fallback);
    };
    let rope_dtype: TokenStream = quote! { __rope_dtype };

    let rotary_local_field: TokenStream = if uses_rotary_local {
        quote! { pub rotary_local: ::ferrite_kernels::rotary::RotaryCache, }
    } else {
        quote! {}
    };

    let rotary_local_load: TokenStream = if uses_rotary_local {
        let head_dim = *model
            .bounds
            .get("head_dim")
            .expect("model must have head_dim for RotaryLocal") as usize;
        let max_pos = *model
            .bounds
            .get("max_position_embeddings")
            .expect("model must have max_position_embeddings for RotaryLocal")
            as usize;
        let local_theta = model
            .scalars
            .get("rope_local_base_freq")
            .copied()
            .or_else(|| model.bounds.get("rope_local_base_freq").map(|&v| v as f64))
            .or_else(|| model.scalars.get("rope_theta").copied())
            .or_else(|| model.bounds.get("rope_theta").map(|&v| v as f64))
            .expect("model must have rope_local_base_freq for RotaryLocal");
        quote! {
            let rotary_local = unsafe {
                ::ferrite_kernels::rotary::RotaryCache::new_from_stream(
                    #head_dim,
                    #max_pos,
                    #local_theta,
                    None,
                    #rope_dtype,
                    stream,
                )
            }?;
        }
    } else {
        quote! {}
    };

    let rotary_local_init: TokenStream = if uses_rotary_local {
        quote! { rotary_local, }
    } else {
        quote! {}
    };

    // Primary `rotary: RotaryCache` field. Ferrite owns rotary end-
    // to-end: `ForwardCtx` carries no rotary, the emitted forward
    // reads `wm.rotary`, and this block picks the right
    // `RotaryCache` constructor at macro-expansion time from the
    // manifest. Four cases cross-join `partial_rotary_factor` (None
    // ⇒ full head_dim, Some(f) ⇒ rotary_dim = f * head_dim) with
    // `rope_scaling` (None / Llama3 / LongRope).
    let uses_rotary = fuf.nodes.iter().any(|n| {
        n.inputs.iter().any(|i| {
            matches!(
                i,
                crate::fuf::FufInput::Extern {
                    kind: crate::classified::ExternKind::Rotary,
                    ..
                }
            )
        })
    });

    let rotary_field: TokenStream = if uses_rotary {
        quote! { pub rotary: ::ferrite_kernels::rotary::RotaryCache, }
    } else {
        quote! {}
    };

    let rotary_load: TokenStream = if uses_rotary {
        let head_dim = *model
            .bounds
            .get("head_dim")
            .expect("model must have head_dim for Rotary") as usize;
        let max_pos = *model
            .bounds
            .get("max_position_embeddings")
            .expect("model must have max_position_embeddings for Rotary")
            as usize;
        // HF's implicit default when `rope_theta` is omitted (e.g.
        // `llama-2-70b.json`). Matches transformers' LlamaConfig.
        let rope_theta = model
            .scalars
            .get("rope_theta")
            .copied()
            .or_else(|| model.bounds.get("rope_theta").map(|&v| v as f64))
            .unwrap_or(10000.0);
        // `partial_rotary_factor == 1.0` is the full-rotary identity;
        // route it through `new_from_stream` instead of
        // `new_partial_from_stream` with `rotary_dim == head_dim`
        // so models that spell this out (Phi-4-reasoning) don't
        // exercise the partial-rotary kernel path unnecessarily.
        let partial = model
            .scalars
            .get("partial_rotary_factor")
            .copied()
            .filter(|&f| (f - 1.0).abs() > 1e-9);
        let rotary_dim_lit: Option<TokenStream> = partial.map(|f| {
            let rd = f * head_dim as f64;
            assert!(
                (rd - rd.round()).abs() < 1e-9,
                "partial_rotary_factor {f} * head_dim {head_dim} = {rd} is not integer",
            );
            let rd = rd.round() as usize;
            quote! { #rd }
        });
        // For YaRN (DeepSeek V2 MLA), the rope portion uses `qk_rope_head_dim`
        // rather than the full `head_dim`. Pull it from bounds if present.
        let yarn_rope_head_dim: Option<usize> =
            model.bounds.get("qk_rope_head_dim").map(|&v| v as usize);
        let scaling = model.rope_scaling.clone();
        // For MLA models (DeepSeek V2/V3 flat), the rope portion uses
        // `qk_rope_head_dim` rather than the full `head_dim`. YaRN already
        // uses `yarn_rope_head_dim`; apply the same override to standard RoPE.
        let rope_cache_head_dim = yarn_rope_head_dim.unwrap_or(head_dim);
        let rope_cache_head_dim_lit = proc_macro2::Literal::usize_unsuffixed(rope_cache_head_dim);

        let body = match (rotary_dim_lit, scaling) {
            (None, None) => quote! {
                ::ferrite_kernels::rotary::RotaryCache::new_from_stream(
                    #rope_cache_head_dim_lit,
                    #max_pos,
                    #rope_theta,
                    None,
                    #rope_dtype,
                    stream,
                )
            },
            (
                None,
                Some(crate::config::RopeScaling::Llama3 {
                    factor,
                    low_freq_factor,
                    high_freq_factor,
                    original_max_position_embeddings,
                }),
            ) => {
                let orig = original_max_position_embeddings as usize;
                quote! {
                    ::ferrite_kernels::rotary::RotaryCache::new_from_stream(
                        #head_dim,
                        #max_pos,
                        #rope_theta,
                        Some(&::ferrite_kernels::rotary::Llama3RopeScaling {
                            factor: #factor,
                            low_freq_factor: #low_freq_factor,
                            high_freq_factor: #high_freq_factor,
                            original_max_position_embeddings: #orig,
                        }),
                        #rope_dtype,
                        stream,
                    )
                }
            }
            (
                None,
                Some(crate::config::RopeScaling::LongRope {
                    short_factor,
                    long_factor,
                    original_max_position_embeddings,
                    short_mscale,
                    long_mscale,
                }),
            ) => {
                let orig = original_max_position_embeddings as usize;
                quote! {
                    ::ferrite_kernels::rotary::RotaryCache::new_longrope_from_stream(
                        #head_dim,
                        #max_pos,
                        max_model_len,
                        #rope_theta,
                        &::ferrite_kernels::rotary::LongRopeScaling {
                            short_factor: vec![ #(#short_factor),* ],
                            long_factor: vec![ #(#long_factor),* ],
                            original_max_position_embeddings: #orig,
                            short_mscale: #short_mscale,
                            long_mscale: #long_mscale,
                        },
                        #rope_dtype,
                        stream,
                    )
                }
            }
            (
                _,
                Some(crate::config::RopeScaling::Yarn {
                    factor,
                    beta_fast,
                    beta_slow,
                    mscale,
                    mscale_all_dim,
                    original_max_position_embeddings,
                }),
            ) => {
                let orig = original_max_position_embeddings as usize;
                let rope_hd = yarn_rope_head_dim.unwrap_or(head_dim);
                quote! {
                    ::ferrite_kernels::rotary::RotaryCache::new_yarn_from_stream(
                        #rope_hd,
                        #max_pos,
                        #rope_theta,
                        &::ferrite_kernels::rotary::YarnRopeScaling {
                            factor: #factor,
                            beta_fast: #beta_fast,
                            beta_slow: #beta_slow,
                            mscale: #mscale,
                            mscale_all_dim: #mscale_all_dim,
                            original_max_position_embeddings: #orig,
                        },
                        #rope_dtype,
                        stream,
                    )
                }
            }
            (Some(rotary_dim), None) => quote! {
                ::ferrite_kernels::rotary::RotaryCache::new_partial_from_stream(
                    #head_dim,
                    #rotary_dim,
                    #max_pos,
                    #rope_theta,
                    None,
                    #rope_dtype,
                    stream,
                )
            },
            (
                Some(rotary_dim),
                Some(crate::config::RopeScaling::Llama3 {
                    factor,
                    low_freq_factor,
                    high_freq_factor,
                    original_max_position_embeddings,
                }),
            ) => {
                let orig = original_max_position_embeddings as usize;
                quote! {
                    ::ferrite_kernels::rotary::RotaryCache::new_partial_from_stream(
                        #head_dim,
                        #rotary_dim,
                        #max_pos,
                        #rope_theta,
                        Some(&::ferrite_kernels::rotary::Llama3RopeScaling {
                            factor: #factor,
                            low_freq_factor: #low_freq_factor,
                            high_freq_factor: #high_freq_factor,
                            original_max_position_embeddings: #orig,
                        }),
                        #rope_dtype,
                        stream,
                    )
                }
            }
            (
                Some(rotary_dim),
                Some(crate::config::RopeScaling::LongRope {
                    short_factor,
                    long_factor,
                    original_max_position_embeddings,
                    short_mscale,
                    long_mscale,
                }),
            ) => {
                let orig = original_max_position_embeddings as usize;
                quote! {
                    ::ferrite_kernels::rotary::RotaryCache::new_partial_longrope_from_stream(
                        #head_dim,
                        #rotary_dim,
                        #max_pos,
                        max_model_len,
                        #rope_theta,
                        &::ferrite_kernels::rotary::LongRopeScaling {
                            short_factor: vec![ #(#short_factor),* ],
                            long_factor: vec![ #(#long_factor),* ],
                            original_max_position_embeddings: #orig,
                            short_mscale: #short_mscale,
                            long_mscale: #long_mscale,
                        },
                        #rope_dtype,
                        stream,
                    )
                }
            }
        };
        quote! {
            let rotary = unsafe { #body }?;
        }
    } else {
        quote! {}
    };

    let rotary_init: TokenStream = if uses_rotary {
        quote! { rotary, }
    } else {
        quote! {}
    };

    // Per-arch accessor methods on `Weights`. Group accessor field
    // names by stripping any trailing `_<digits>` suffix; for each
    // base, emit `pub fn <base>(&self, layer: u32) -> &<Ty>` that
    // matches on the layer index. The host-interpreter's per-arch
    // enum variants carry `weight_fn: fn(&Weights, u32) -> &Ty`
    // pointing at one of these methods, so per-claim weight selection
    // is a const fn-pointer field rather than a baked-in `wm.<field>`
    // path. Non-layered accessors (no `_<digits>` suffix) get the
    // same signature for uniformity; their body is `&self.<field>`
    // and ignores the layer arg.
    let accessor_methods = emit_weights_accessor_methods(&accessors);

    // Rotary cos_sin accessors for the host-interpreter path. Both
    // `wm.rotary` and `wm.rotary_local` are conditionally-emitted
    // fields — the interpreter arm body is a single token stream
    // shared across every claim of an Impl on this arch, so it
    // can't directly write `wm.rotary_local.cos_sin_cache` (Llama
    // would fail to type-check). Per-claim selection rides on a
    // `cos_sin_fn: for<'a> fn(&'a Weights, u32) -> ferrite_cuda_core::tensor::GpuTensor`
    // OpInstance field; `fan_out` resolves it to one of these
    // accessor names. Llama's interpreter never sees `Weights::rotary_local_cos_sin`
    // because `RotaryLocal` extern doesn't appear in its FUF.
    let rotary_cos_sin_methods: TokenStream = match &mode {
        WeightsEmitMode::Canonical => {
            let main = if uses_rotary {
                quote! {
                    #[cfg(feature = "cuda")]
                    #[inline]
                    #[allow(dead_code)]
                    pub fn rotary_cos_sin(&self, _layer: u32)
                        -> ::ferrite_cuda_core::tensor::GpuTensor
                    {
                        self.rotary.cos_sin_cache
                    }
                }
            } else {
                quote! {}
            };
            let local = if uses_rotary_local {
                quote! {
                    #[cfg(feature = "cuda")]
                    #[inline]
                    #[allow(dead_code)]
                    pub fn rotary_local_cos_sin(&self, _layer: u32)
                        -> ::ferrite_cuda_core::tensor::GpuTensor
                    {
                        self.rotary_local.cos_sin_cache
                    }
                }
            } else {
                quote! {}
            };
            if uses_rotary || uses_rotary_local {
                quote! {
                    #[cfg(feature = "cuda")]
                    impl Weights {
                        #main
                        #local
                    }
                }
            } else {
                quote! {}
            }
        }
        // Shims share the canonical's Weights via type alias, so
        // they inherit these methods automatically.
        WeightsEmitMode::Shim { .. } => quote! {},
    };

    // Struct definition vs type alias per emit mode.
    let weights_def: TokenStream = match &mode {
        WeightsEmitMode::Canonical => quote! {
            /// Every weight the forward needs, packed for the
            /// solver-picked Impls. Construct via `load`.
            #[cfg(feature = "cuda")]
            pub struct Weights {
                #(#fields)*
                #rotary_field
                #rotary_local_field
            }
        },
        WeightsEmitMode::Shim { canonical } => quote! {
            /// Shim — shares canonical sibling's `Weights`.
            #[cfg(feature = "cuda")]
            pub type Weights = super::#canonical::Weights;
        },
    };
    let weights_ctor: TokenStream = match &mode {
        WeightsEmitMode::Canonical => quote! { Weights },
        WeightsEmitMode::Shim { canonical } => quote! { super::#canonical::Weights },
    };

    let marlin_fmt = marlin_format_literal(model);

    // Canonical emits the full `load_with(marlin_storage)` body +
    // a thin `load()` wrapper that passes this variant's Marlin
    // format literal. Shim variants skip `load_with` entirely —
    // they just thread their own MarlinFormat into the canonical
    // sibling's `load_with`. rustc doesn't re-monomorphize the
    // shim's one-line delegation body, so the expensive load
    // compile work (N_layers × N_accessors lines) runs ONCE per
    // equivalence class.
    match &mode {
        WeightsEmitMode::Canonical => quote! {
            #weights_def

            #accessor_methods

            #rotary_cos_sin_methods

            #fingerprint_method

            /// Canonical load body. `marlin_storage` lets AWQ/GPTQ/CT
            /// variants share one compiled copy; ignored elsewhere.
            /// `tp_rank` is the runtime rank-id (0..tp_world_size);
            /// the bake `tp_world_size` literal lives on `<W as
            /// CanonicalParams>::…` divisor constants and on the
            /// per-(model, tp) emitted `_sharded` loader call sites
            /// (task #5).
            #[cfg(feature = "cuda")]
            #[allow(clippy::too_many_lines, clippy::not_unsafe_ptr_arg_deref, unused_variables)]
            pub fn load_with(
                gw: &mut ::ferrite_cuda_core::weights::GpuWeights,
                stream: ::ferrite_cuda_core::CUstream,
                max_model_len: usize,
                marlin_storage: ::ferrite_kernels::layers_quant::MarlinFormat,
                tp_rank: u8,
            ) -> ::anyhow::Result<Weights> {
                #packed_splits_prelude
                #marlin_prelude
                #bnb4_prelude
                #fp8_prelude
                #rotary_prelude
                #(#lets)*
                #rotary_load
                #rotary_local_load
                Ok(#weights_ctor {
                    #(#field_shorthand,)*
                    #rotary_init
                    #rotary_local_init
                })
            }

            /// Variant entry — threads this variant's MarlinFormat.
            #[cfg(feature = "cuda")]
            #[inline]
            #[allow(clippy::not_unsafe_ptr_arg_deref)]
            pub fn load(
                gw: &mut ::ferrite_cuda_core::weights::GpuWeights,
                stream: ::ferrite_cuda_core::CUstream,
                max_model_len: usize,
                tp_rank: u8,
            ) -> ::anyhow::Result<Weights> {
                load_with(gw, stream, max_model_len, #marlin_fmt, tp_rank)
            }
        },
        WeightsEmitMode::Shim { canonical } => quote! {
            #weights_def

            #fingerprint_method

            /// Shim — delegates to canonical's `load_with`.
            #[cfg(feature = "cuda")]
            #[inline]
            #[allow(clippy::not_unsafe_ptr_arg_deref)]
            pub fn load(
                gw: &mut ::ferrite_cuda_core::weights::GpuWeights,
                stream: ::ferrite_cuda_core::CUstream,
                max_model_len: usize,
                tp_rank: u8,
            ) -> ::anyhow::Result<Weights> {
                super::#canonical::load_with(gw, stream, max_model_len, #marlin_fmt, tp_rank)
            }
        },
    }
}

/// Split a Weights field name into `(base, layer)` where `layer`
/// is `Some(n)` if the name ends in `_<digits>` (e.g.
/// `input_layernorm_3` → `("input_layernorm", Some(3))`) and
/// `None` for un-suffixed names like `embed_tokens` or `lm_head`.
///
/// The trailing-digits rule is the convention `weight_field_name`
/// has used since the start: `(stem)_(layer_index)` for per-layer
/// accessors, bare `stem` for arch-wide ones. New accessors must
/// follow the same rule or the per-arch accessor method codegen
/// will silently group them as non-layered. (Mixed bases —
/// e.g. one `input_layernorm` and one `input_layernorm_0` under
/// the same name root — panic at codegen time.)
pub(crate) fn split_base_layer(field_name: &str) -> (String, Option<u64>) {
    if let Some((base, suffix)) = field_name.rsplit_once('_')
        && !suffix.is_empty()
        && suffix.chars().all(|c| c.is_ascii_digit())
        && let Ok(n) = suffix.parse::<u64>()
    {
        return (base.to_string(), Some(n));
    }
    (field_name.to_string(), None)
}

/// One row in the per-arch `Weights` shape. The field/struct emit,
/// the `impl Weights` accessor methods, and the `load_with` body
/// all walk the same group list so they stay in sync.
pub(crate) struct AccessorGroup<'a> {
    /// Field name on `Weights` — the accessor's stem with the
    /// `_<layer>` suffix stripped. Stays a plain `String` because
    /// callers also need it as a runtime format-string fragment.
    pub base: String,
    /// Shared element type. For Vec-compressed layered groups this
    /// is the `T` in `Vec<T>`; for unindexed and sparse-layered
    /// groups it's the field type as-is.
    pub rust_type: TokenStream,
    /// What shape this group lowers to.
    pub kind: AccessorGroupKind,
    /// `entries[i].0`: layer index. For `LayeredContiguous`,
    /// entries are sorted by layer and occupy `0..entries.len()`
    /// contiguously (the Vec-build relies on this). For
    /// `Unindexed`, a single entry with `None`. For `LayeredSparse`,
    /// entries are sorted by layer but may have gaps or start at
    /// `layer > 0` — the per-layer fallback handles either case.
    pub entries: Vec<(Option<u64>, &'a WeightAccessor)>,
}

/// How an [`AccessorGroup`] is lowered. The Vec-compressed path is
/// only safe when the layered group fills `Vec[0..N]` contiguously;
/// real-world archs with conditional-per-layer accessors (e.g.
/// DeepSeek-V2's `moe` is layers 1..N — layer 0 is dense FFN) take
/// the legacy per-layer-fields fallback.
pub(crate) enum AccessorGroupKind {
    /// Single field, no layer arg. Field type is `T`. Accessor
    /// method ignores its `layer` arg and returns `&self.<base>`.
    Unindexed,
    /// Layered family that occupies `Vec[0..N]` contiguously.
    /// Field type is `Vec<T>`. Accessor method is
    /// `&self.<base>[layer as usize]`. Load body emits one
    /// `(0..N).map(|layer| …).collect()`.
    LayeredContiguous,
    /// Layered family with gaps or non-zero start (e.g. layers
    /// 1..N only). Per-layer fields `pub <base>_<L>: T` are
    /// emitted; accessor method is a `match layer { L => &self.<base>_<L>, … }`.
    /// Load body emits one `let <base>_<L> = …;` per entry.
    LayeredSparse,
}

/// Group `accessors` by their `(base, layer)` split.
///
/// Outcomes:
/// - Single unindexed accessor → `AccessorGroup { kind: Unindexed }`.
/// - Layered family covering `0..N` contiguously →
///   `AccessorGroup { kind: LayeredContiguous }` (Vec compression).
/// - Layered family with gaps or non-zero start →
///   `AccessorGroup { kind: LayeredSparse }` (per-layer fallback).
///
/// Panics on:
/// - a base mixing layered and unindexed entries;
/// - a base whose entries declare conflicting `rust_type`s.
pub(crate) fn group_accessors_by_base(accessors: &[WeightAccessor]) -> Vec<AccessorGroup<'_>> {
    use std::collections::BTreeMap;

    struct Bucket<'a> {
        rust_type: TokenStream,
        rust_type_str: String,
        layered_entries: BTreeMap<u64, &'a WeightAccessor>,
        unindexed: Option<&'a WeightAccessor>,
    }

    let mut by_base: BTreeMap<String, Bucket<'_>> = BTreeMap::new();
    for acc in accessors {
        let full = acc.name.to_string();
        let (base, layer) = split_base_layer(&full);
        let ty_str = acc.rust_type.to_string();
        let entry = by_base.entry(base.clone()).or_insert_with(|| Bucket {
            rust_type: acc.rust_type.clone(),
            rust_type_str: ty_str.clone(),
            layered_entries: BTreeMap::new(),
            unindexed: None,
        });
        if entry.rust_type_str != ty_str {
            panic!(
                "Weights accessor base `{base}` has mismatched types across layers: \
                 `{}` vs `{ty_str}`",
                entry.rust_type_str,
            );
        }
        match layer {
            Some(n) => {
                if entry.layered_entries.insert(n, acc).is_some() {
                    panic!("Weights accessor base `{base}` has duplicate layer index {n}",);
                }
            }
            None => {
                if entry.unindexed.is_some() {
                    panic!("Weights accessor base `{base}` has multiple unindexed entries",);
                }
                entry.unindexed = Some(acc);
            }
        }
    }

    by_base
        .into_iter()
        .map(|(base, bucket)| {
            match (bucket.unindexed, bucket.layered_entries.is_empty()) {
                (Some(acc), true) => AccessorGroup {
                    base,
                    rust_type: bucket.rust_type,
                    kind: AccessorGroupKind::Unindexed,
                    entries: vec![(None, acc)],
                },
                (None, false) => {
                    let layers: Vec<u64> = bucket.layered_entries.keys().copied().collect();
                    let starts_at_zero = layers.first().copied() == Some(0);
                    let contiguous = layers.iter().enumerate().all(|(i, l)| *l == i as u64);
                    let kind = if starts_at_zero && contiguous {
                        AccessorGroupKind::LayeredContiguous
                    } else {
                        // Real-world examples: DeepSeek MoE (layers
                        // 1..N), per-window-size attention overrides,
                        // any future per-layer-conditional accessor.
                        // Keep the legacy `match layer { … }` shape so
                        // these compile without forcing a Vec
                        // representation that doesn't fit.
                        AccessorGroupKind::LayeredSparse
                    };
                    let entries: Vec<(Option<u64>, &WeightAccessor)> = layers
                        .iter()
                        .map(|l| (Some(*l), bucket.layered_entries[l]))
                        .collect();
                    AccessorGroup {
                        base,
                        rust_type: bucket.rust_type,
                        kind,
                        entries,
                    }
                }
                _ => panic!("Weights accessor base `{base}` mixes layered and non-layered fields",),
            }
        })
        .collect()
}

/// Convert an L=0-baked safetensors prefix (e.g.
/// `"model.layers.0.input_layernorm"` or
/// `"visual.blocks.0.attn.q"`) into a TokenStream that evaluates
/// to a runtime `String` for the layer in scope. The emitted tokens
/// reference a local `layer: u32` binding the caller plants in scope
/// (the closure arg of the Vec-build).
///
/// `vision_layered_root_with_zero` is the per-arch vision-side
/// `<default_root>.<layered_subpath>.0.` prefix (e.g.
/// `"visual.blocks.0."` for Qwen or
/// `"vision_tower.vision_model.encoder.layers.0."` for Gemma3-MM).
/// `None` for decoder bodies — the function only knows about
/// `model.layers.0.`.
fn layer_templated_prefix_expr(
    layer0_prefix: &str,
    vision_layered_root_with_zero: Option<&str>,
    decoder_layered_root_with_zero: Option<&str>,
) -> TokenStream {
    if let Some(tail) = layer0_prefix.strip_prefix("model.layers.0.") {
        // Route through `ferrite_forward::layer_weight_path(layer,
        // suffix)` instead of inlining `format!()`. The post-macro
        // expansion of `format!("model.layers.{}.X", layer)` is a
        // 5-line `::alloc::__export::must_use({
        //     ::alloc::fmt::format(format_args!(...))
        // })` block; the helper fn collapses every call site to
        // one line of expanded source. Fires per-layer per-accessor
        // per-canonical — thousands of times on llama.
        return quote! { ::ferrite_forward::layer_weight_path(layer, #tail) };
    }
    if layer0_prefix == "model.layers.0" {
        return quote! { ::std::format!("model.layers.{}", layer) };
    }
    if let Some(zero_prefix) = vision_layered_root_with_zero
        && let Some(tail) = layer0_prefix.strip_prefix(zero_prefix)
    {
        // Shave the trailing `.0.` from the zero_prefix to recover
        // the bare root (`visual.blocks` /
        // `vision_tower.vision_model.encoder.layers`) for the
        // runtime templater.
        let root = zero_prefix
            .strip_suffix(".0.")
            .or_else(|| zero_prefix.strip_suffix(".0"))
            .unwrap_or(zero_prefix);
        return quote! { ::ferrite_forward::vision_block_weight_path(#root, layer, #tail) };
    }
    if let Some(zero_prefix) = decoder_layered_root_with_zero
        && let Some(tail) = layer0_prefix.strip_prefix(zero_prefix)
    {
        // MM-decoder-prefixed root (e.g.
        // `language_model.model.layers`). Same shape as vision —
        // strip the trailing `.0.` to recover the bare root and emit
        // a `layer_weight_path_with_root` call.
        let root = zero_prefix
            .strip_suffix(".0.")
            .or_else(|| zero_prefix.strip_suffix(".0"))
            .unwrap_or(zero_prefix);
        return quote! { ::ferrite_forward::layer_weight_path_with_root(#root, layer, #tail) };
    }
    panic!(
        "layered accessor's L=0 prefix `{layer0_prefix}` doesn't \
         start with `model.layers.0` or the per-arch vision layered \
         root — codegen invariant violated"
    );
}

/// Strip the `model.layers.0.` prefix to recover the per-layer
/// suffix string the `load_layered_*` helpers in
/// [`ferrite_forward::loaders`] take. E.g. `"model.layers.0.input_layernorm"`
/// → `"input_layernorm"`. Panics on prefixes that don't start with
/// `model.layers.0.` — every layered accessor's L=0 prefix carries
/// that prefix by construction (see [`safetensors_prefix`]); a
/// mismatch surfaces a `default_required_weights` bug instead of
/// silently emitting a malformed helper call.
fn layered_suffix<'a>(
    layer0_prefix: &'a str,
    vision_layered_root_with_zero: Option<&str>,
    decoder_layered_root_with_zero: Option<&str>,
) -> &'a str {
    if let Some(s) = layer0_prefix.strip_prefix("model.layers.0.") {
        return s;
    }
    if let Some(zero_prefix) = vision_layered_root_with_zero
        && let Some(s) = layer0_prefix.strip_prefix(zero_prefix)
    {
        return s;
    }
    if let Some(zero_prefix) = decoder_layered_root_with_zero
        && let Some(s) = layer0_prefix.strip_prefix(zero_prefix)
    {
        return s;
    }
    panic!(
        "layered accessor's L=0 prefix `{layer0_prefix}` doesn't \
         start with `model.layers.0.` or the per-arch vision/decoder \
         layered root — codegen invariant violated"
    )
}

/// Emit the unindexed accessor's let-binding (kept as the
/// existing per-FieldLoad shape — the prefix is a baked `&str`
/// literal, so no runtime `format!` machinery is needed). The
/// returned tokens include the trailing `;` and the leading
/// `let #name = …`.
fn emit_unindexed_let(name: &syn::Ident, plan: &FieldLoad, tp_world_size: u8) -> TokenStream {
    use crate::tp_lowering::{ShardKind, shard_kind_for_dotted_prefix};
    let tp_world_lit = proc_macro2::Literal::u8_unsuffixed(tp_world_size);
    let sharded = tp_world_size > 1;
    match plan {
        FieldLoad::Embedding(prefix) => {
            // Embed paths route to vocab-parallel `_sharded` at tp>1
            // (matches Python vLLM `VocabParallelEmbedding`). Non-
            // embed names that happen to deserialize through the
            // `Embedding` FieldLoad arm fall back to `Replicate`.
            let kind = shard_kind_for_dotted_prefix(prefix);
            if sharded && kind == ShardKind::ShardDim0 {
                quote! {
                    let #name = ::ferrite_kernels::layers::Embedding::load_sharded(
                        gw, #prefix, tp_rank as usize, #tp_world_lit as usize,
                    )?;
                }
            } else {
                quote! {
                    let #name = ::ferrite_kernels::layers::Embedding::load(gw, #prefix)?;
                }
            }
        }
        FieldLoad::RmsNorm(prefix, eps) => quote! {
            let #name = ::ferrite_kernels::layers::RmsNorm::load(gw, #prefix, #eps)?;
        },
        FieldLoad::LayerNorm(prefix, eps) => quote! {
            let #name = ::ferrite_kernels::layers::LayerNorm::load(gw, #prefix, #eps)?;
        },
        FieldLoad::LinearDense(prefix) => {
            // Per-Linear shard kind: q/k/v/gate/up/lm_head/embed →
            // ShardDim0 (column / vocab parallel); o/down → ShardDim1
            // (row parallel); else Replicate. lm_head's shard is what
            // makes Python's tied-weights case self-consistent — since
            // both embed_tokens and lm_head end up dim-0-sharded, the
            // tied `Linear::new(embed.weight, None)` in
            // `LinearTiedToEmbedding` below sees an already-sharded
            // weight without further work.
            let kind = shard_kind_for_dotted_prefix(prefix);
            match (sharded, kind) {
                (true, ShardKind::ShardDim0) => quote! {
                    let #name = ::ferrite_kernels::layers::LinearLayer::load_dense_sharded(
                        gw, #prefix, 0usize, tp_rank as usize, #tp_world_lit as usize,
                    )?;
                },
                (true, ShardKind::ShardDim1) => quote! {
                    let #name = ::ferrite_kernels::layers::LinearLayer::load_dense_sharded(
                        gw, #prefix, 1usize, tp_rank as usize, #tp_world_lit as usize,
                    )?;
                },
                _ => quote! {
                    // `load_dense_or_ggml`: try `take_quantized_linear`
                    // first (for `StorageFormat::Ggml` weights), fall
                    // back to dense safetensors path. Transparent on
                    // every existing safetensors model since the
                    // GGUF map is empty there.
                    let #name = ::ferrite_kernels::layers::LinearLayer::load_dense_or_ggml(gw, #prefix)?;
                },
            }
        }
        FieldLoad::LinearConcat(prefixes) => {
            // Fused QKV / gate_up are always column-parallel — no
            // row-parallel concat exists in any current arch. The
            // sharded helper packs each source's per-rank slice into
            // one contiguous buffer; biases follow column-parallel
            // rule (sliced along dim 0 too).
            if sharded {
                quote! {
                    let #name = ::ferrite_kernels::layers::LinearLayer::load_dense_concat_sharded(
                        gw,
                        &[ #(#prefixes),* ],
                        stream,
                        tp_rank as usize,
                        #tp_world_lit as usize,
                    )?;
                }
            } else {
                quote! {
                    // `load_dense_concat_or_ggml`: tries GGUF byte-pack
                    // first (when every prefix has a quantized linear),
                    // falls back to the existing safetensors concat
                    // path. Transparent on safetensors models.
                    let #name = ::ferrite_kernels::layers::LinearLayer::load_dense_concat_or_ggml(
                        gw,
                        &[ #(#prefixes),* ],
                        stream,
                    )?;
                }
            }
        }
        FieldLoad::RawLinear(key) => quote! {
            // Raw nn.Parameter load — `<key>` is the verbatim
            // safetensors key (no `.weight` / `.bias` suffix).
            // No bias, no TP shard (raw projector weights are
            // global, not per-block / per-rank). The `LinearLayer`
            // wrapper carries the tensor through to the body's
            // gemm op via `dense_weight()`.
            let #name = ::ferrite_kernels::layers::LinearLayer::load_raw(gw, #key)?;
        },
        FieldLoad::LinearTiedToEmbedding(embed_ident) => quote! {
            // Tied embedding: lm_head reuses the
            // `#embed_ident` field's weight tensor. Shape
            // [vocab_size, hidden_size] works for both
            // Embedding (gather rows) and LinearLayer
            // (matmul against hidden_size). No bias.
            let #name = ::ferrite_kernels::layers::LinearLayer::Dense(
                ::ferrite_kernels::layers::Linear::new(
                    #embed_ident.weight,
                    None,
                )
            );
        },
        FieldLoad::MarlinLinear { prefixes, .. } => {
            // Every Marlin accessor emits the SAME call shape
            // regardless of AWQ/GPTQ/CT: the runtime
            // `MarlinFormat` discriminator is threaded in from
            // `load_with`'s `marlin_storage` param. That's what
            // lets cross-variant load-body dedup collapse
            // AWQ/GPTQ/CT variants of the same (arch, size) to
            // one canonical `load_with` body.
            if prefixes.len() == 1 {
                let prefix = &prefixes[0];
                quote! {
                    let #name = ::ferrite_kernels::layers::MarlinLinear::load(
                        gw,
                        #prefix,
                        marlin_storage,
                        __marlin_ws,
                        __device_id,
                    )?;
                }
            } else {
                quote! {
                    let #name = ::ferrite_kernels::layers::MarlinLinear::load_concat(
                        gw,
                        &[ #(#prefixes),* ],
                        marlin_storage,
                        __marlin_ws,
                        __device_id,
                    )?;
                }
            }
        }
        FieldLoad::Fp8Linear { prefixes } => {
            if prefixes.len() == 1 {
                let prefix = &prefixes[0];
                quote! {
                    let #name = ::ferrite_kernels::layers::Fp8AnyLinear::Std(
                        ::ferrite_kernels::layers::Fp8Linear::load(
                            gw,
                            #prefix,
                            __fp8_dtype,
                        )?
                    );
                }
            } else {
                quote! {
                    let #name = ::ferrite_kernels::layers::Fp8AnyLinear::Std(
                        ::ferrite_kernels::layers::Fp8Linear::load_concat(
                            gw,
                            &[ #(#prefixes),* ],
                            __fp8_dtype,
                        )?
                    );
                }
            }
        }
        FieldLoad::Fp8BlockLinear { prefixes } => {
            if prefixes.len() == 1 {
                let prefix = &prefixes[0];
                quote! {
                    let #name = ::ferrite_kernels::layers::Fp8AnyLinear::Block(
                        ::ferrite_kernels::layers::Fp8BlockLinear::load(
                            gw,
                            #prefix,
                            __fp8_dtype,
                        )?
                    );
                }
            } else {
                quote! {
                    let #name = ::ferrite_kernels::layers::Fp8AnyLinear::Block(
                        ::ferrite_kernels::layers::Fp8BlockLinear::load_concat(
                            gw,
                            &[ #(#prefixes),* ],
                            __fp8_dtype,
                        )?
                    );
                }
            }
        }
        FieldLoad::Bnb4Linear {
            prefixes,
            out_features_per_shard,
            in_features,
            blocksize,
        } => {
            let in_features = *in_features as usize;
            let blocksize = *blocksize as usize;
            let outs: Vec<proc_macro2::Literal> = out_features_per_shard
                .iter()
                .map(|o| proc_macro2::Literal::usize_unsuffixed(*o as usize))
                .collect();
            if prefixes.len() == 1 {
                let prefix = &prefixes[0];
                let out = &outs[0];
                quote! {
                    let #name = ::ferrite_kernels::layers::Bnb4bitLinear::load(
                        gw,
                        #prefix,
                        __bnb_code,
                        __bnb_scratch,
                        #out,
                        #in_features,
                        #blocksize,
                    )?;
                }
            } else {
                quote! {
                    let #name = ::ferrite_kernels::layers::Bnb4bitLinear::load_concat(
                        gw,
                        &[ #(#prefixes),* ],
                        __bnb_code,
                        __bnb_scratch,
                        &[ #(#outs),* ],
                        #in_features,
                        #blocksize,
                    )?;
                }
            }
        }
        FieldLoad::DeepSeekV2Moe {
            prefix,
            n_routed_experts,
            n_shared_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            norm_topk_prob,
            routed_scaling_factor,
            use_sigmoid,
            n_expert_group,
            topk_group,
        } => {
            let n_routed_experts = *n_routed_experts;
            let n_shared_experts = *n_shared_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let hidden_size = *hidden_size;
            let norm_topk_prob = *norm_topk_prob;
            let routed_scaling_factor = *routed_scaling_factor;
            let use_sigmoid = *use_sigmoid;
            let n_expert_group = *n_expert_group;
            let topk_group = *topk_group;
            quote! {
                let #name = ::ferrite_kernels::layers_moe::DeepSeekV2MoELayer::load(
                    gw,
                    #prefix,
                    #n_routed_experts,
                    #n_shared_experts,
                    #top_k,
                    #moe_intermediate_size,
                    #hidden_size,
                    #norm_topk_prob,
                    #routed_scaling_factor,
                    #use_sigmoid,
                    #n_expert_group,
                    #topk_group,
                    stream,
                )?;
            }
        }
        FieldLoad::DeepSeekV2Fp8BlockMoe {
            prefix,
            n_routed_experts,
            n_shared_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            norm_topk_prob,
            routed_scaling_factor,
            use_sigmoid,
            n_expert_group,
            topk_group,
        } => {
            let n_routed_experts = *n_routed_experts;
            let n_shared_experts = *n_shared_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let hidden_size = *hidden_size;
            let norm_topk_prob = *norm_topk_prob;
            let routed_scaling_factor = *routed_scaling_factor;
            let use_sigmoid = *use_sigmoid;
            let n_expert_group = *n_expert_group;
            let topk_group = *topk_group;
            quote! {
                let #name = ::ferrite_kernels::layers_moe::DeepSeekV2Fp8BlockMoELayer::load(
                    gw,
                    #prefix,
                    #n_routed_experts,
                    #n_shared_experts,
                    #top_k,
                    #moe_intermediate_size,
                    #hidden_size,
                    #norm_topk_prob,
                    #routed_scaling_factor,
                    #use_sigmoid,
                    #n_expert_group,
                    #topk_group,
                    __fp8_dtype,
                    stream,
                )?;
            }
        }
        FieldLoad::DeepSeekV2GgmlMoe {
            prefix,
            n_routed_experts,
            n_shared_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            norm_topk_prob,
            routed_scaling_factor,
            use_sigmoid,
            n_expert_group,
            topk_group,
        } => {
            let n_routed_experts = *n_routed_experts;
            let n_shared_experts = *n_shared_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let hidden_size = *hidden_size;
            let norm_topk_prob = *norm_topk_prob;
            let routed_scaling_factor = *routed_scaling_factor;
            let use_sigmoid = *use_sigmoid;
            let n_expert_group = *n_expert_group;
            let topk_group = *topk_group;
            quote! {
                let #name = ::ferrite_kernels::layers_moe::DeepSeekV2GgmlMoELayer::load_gguf(
                    gw,
                    #prefix,
                    #n_routed_experts,
                    #n_shared_experts,
                    #top_k,
                    #moe_intermediate_size,
                    #hidden_size,
                    #norm_topk_prob,
                    #routed_scaling_factor,
                    #use_sigmoid,
                    #n_expert_group,
                    #topk_group,
                    stream,
                )?;
            }
        }
        FieldLoad::FusedMoe {
            prefix,
            num_experts,
            top_k,
            intermediate_size,
            hidden_size,
        } => {
            let num_experts = *num_experts;
            let top_k = *top_k;
            let intermediate_size = *intermediate_size;
            let hidden_size = *hidden_size;
            quote! {
                let #name = ::ferrite_kernels::layers_moe::FusedMoELayer::load(
                    gw,
                    #prefix,
                    #num_experts,
                    #top_k,
                    #intermediate_size,
                    #hidden_size,
                    stream,
                )?;
            }
        }
        FieldLoad::SharedFusedMoe {
            prefix,
            num_experts,
            top_k,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            hidden_size,
        } => {
            let num_experts = *num_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let shared_expert_intermediate_size = *shared_expert_intermediate_size;
            let hidden_size = *hidden_size;
            quote! {
                let #name = ::ferrite_kernels::layers_moe::SharedFusedMoELayer::load(
                    gw,
                    #prefix,
                    #num_experts,
                    #top_k,
                    #moe_intermediate_size,
                    #shared_expert_intermediate_size,
                    #hidden_size,
                    stream,
                )?;
            }
        }
    }
}

/// Emit the let-binding(s) for one [`AccessorGroup`].
///
/// - `Unindexed` → one `let <base> = <load_call>;` (delegates to
///   [`emit_unindexed_let`]).
/// - `LayeredContiguous` → one `let <base>: Vec<T> = (0..N).map(|layer|
///   { … }).collect()?;`. The per-iteration body comes from
///   [`emit_layered_load_body`], which rewrites the layer-0-baked
///   prefix(es) into runtime `format!()` calls.
/// - `LayeredSparse` → one `let <base>_<L> = …;` per entry — same
///   shape as the legacy pre-Vec-compression let chain. Used when
///   the layered family has gaps or doesn't start at layer 0
///   (e.g. DeepSeek MoE on layers 1..N), since `Vec[layer as usize]`
///   would be off-by-one without an offset.
fn emit_group_let(
    group: &AccessorGroup<'_>,
    plans: &std::collections::BTreeMap<String, &FieldLoad>,
    model: &ModelParams,
    tp_world_size: u8,
    is_vision: bool,
) -> TokenStream {
    match group.kind {
        AccessorGroupKind::Unindexed => {
            let base_ident = syn::Ident::new(&group.base, proc_macro2::Span::call_site());
            let acc = group.entries[0].1;
            let plan = plans
                .get(&acc.name.to_string())
                .copied()
                .expect("unindexed accessor missing from plan map");
            emit_unindexed_let(&base_ident, plan, tp_world_size)
        }
        AccessorGroupKind::LayeredSparse => {
            // Each entry keeps its per-layer field name and gets
            // its own per-FieldLoad let — same as the pre-grouping
            // codegen. Order follows `entries`'s sort by layer
            // index.
            let lets: Vec<TokenStream> = group
                .entries
                .iter()
                .map(|(_, acc)| {
                    let plan = plans
                        .get(&acc.name.to_string())
                        .copied()
                        .expect("sparse-layered accessor missing from plan map");
                    emit_unindexed_let(&acc.name, plan, tp_world_size)
                })
                .collect();
            quote! { #(#lets)* }
        }
        AccessorGroupKind::LayeredContiguous => {
            let base_ident = syn::Ident::new(&group.base, proc_macro2::Span::call_site());
            // Plan from the first entry (layer 0). All entries share
            // this plan modulo prefix; emit_layered_load_body
            // delegates to a `::ferrite_forward::load_layered_*`
            // helper that owns the (0..N).map().collect() loop. The
            // call site collapses to one line of expanded source.
            let l0 = group.entries[0].1;
            let plan = plans
                .get(&l0.name.to_string())
                .copied()
                .expect("layered group's layer-0 accessor missing from plan map");
            // `n_layers` may be less than `num_hidden_layers` — a
            // contiguous-from-zero group can be partial (e.g.
            // DeepSeek's `mlp_down_proj` is layer 0 only; layers 1..N
            // use `moe` instead). The helper is parameterized by
            // `entries.len()`, and the static-slice rows only ever
            // index 0..n_layers, so partial coverage is fine.
            let n_layers = group.entries.len() as u32;
            // Vision-side layered root: `<default_root>.<layered_subpath>`
            // baked from `vision_safetensors_layout` in the per-arch config.
            // Examples: `visual.blocks` (Qwen), `vision_tower.vision_model
            // .encoder.layers` (Gemma3-MM). `None` for decoder bodies.
            let vision_root_owned: Option<String> = if is_vision {
                let layout = model
                    .vision_layout
                    .clone()
                    .unwrap_or_else(crate::config::VisionSafetensorsLayout::qwen_default);
                Some(format!(
                    "{}.{}",
                    layout.default_root, layout.layered_subpath
                ))
            } else {
                None
            };
            // Decoder-side layered root: `model.layers` (text-only and
            // Qwen-style VL) or `<prefix>.model.layers` for arches whose
            // variant config sets `decoder_safetensors_prefix`
            // (Gemma3-MM nests the text decoder under `language_model.<...>`).
            let decoder_root_owned: String = match model.decoder_safetensors_prefix.as_deref() {
                Some(prefix) => format!("{prefix}.model.layers"),
                None => "model.layers".to_string(),
            };
            let call = emit_layered_load_body(
                plan,
                n_layers,
                tp_world_size,
                is_vision,
                vision_root_owned.as_deref(),
                &decoder_root_owned,
            );
            quote! {
                let #base_ident = #call;
            }
        }
    }
}

/// Emit the full `Result<Vec<T>>` expression for a layered
/// accessor's load. Delegates to a `::ferrite_forward::load_layered_*`
/// helper that owns the `(0..N).map(|layer| Type::load(…)).collect()`
/// loop, replacing the previous emit-the-closure-body shape. Each
/// call site collapses from ~4 lines (typed Vec annotation, range,
/// closure, collect) to one.
///
/// `LinearTiedToEmbedding` is unreachable here — tied embedding
/// only attaches to the `lm_head` accessor, which is unindexed.
/// `DeepSeekV2Moe` falls through to a layered-MoE helper still
/// emitted inline (only DeepSeek arches use it; not worth a helper
/// crossing the ferrite-forward / ferrite-kernels seam).
fn emit_layered_load_body(
    plan: &FieldLoad,
    n_layers: u32,
    tp_world_size: u8,
    is_vision: bool,
    vision_layered_root: Option<&str>,
    decoder_layered_root: &str,
) -> TokenStream {
    use crate::tp_lowering::{ShardKind, shard_kind_for_dotted_prefix};
    let n_lit = proc_macro2::Literal::u32_unsuffixed(n_layers);
    let tp_world_lit = proc_macro2::Literal::u8_unsuffixed(tp_world_size);
    let sharded = tp_world_size > 1;
    // Build the `<root>.0.` form for `layered_suffix` /
    // `layer_templated_prefix_expr` consumers. `None` for decoder bodies.
    let vision_zero_prefix: Option<String> = vision_layered_root.map(|r| format!("{r}.0."));
    let vision_zero_prefix_ref: Option<&str> = vision_zero_prefix.as_deref();
    let vision_root_lit_opt: Option<TokenStream> = vision_layered_root.map(|root| {
        let lit = syn::LitStr::new(root, proc_macro2::Span::call_site());
        quote! { #lit }
    });
    // Decoder layered root literal for the `*_with_root`-aware
    // load_layered_* helpers. `"model.layers"` for text-only and
    // Qwen-style VL; `"language_model.model.layers"` (or wherever
    // `decoder_safetensors_prefix` points) for Gemma3-MM-style arches
    // where HF nests the text decoder.
    let dec_root_lit: TokenStream = {
        let lit = syn::LitStr::new(decoder_layered_root, proc_macro2::Span::call_site());
        quote! { #lit }
    };
    // `<decoder_root>.0.` form for `layered_suffix` /
    // `layer_templated_prefix_expr` consumers under decoder bodies
    // whose variant overrides the default `model.layers.<L>.<suffix>`
    // template (Gemma3-MM nests under `language_model.<...>`). Empty
    // when the root is the canonical `model.layers` (no override).
    let decoder_zero_prefix: Option<String> = if decoder_layered_root != "model.layers" {
        Some(format!("{decoder_layered_root}.0."))
    } else {
        None
    };
    let decoder_zero_prefix_ref: Option<&str> = decoder_zero_prefix.as_deref();
    match plan {
        FieldLoad::Embedding(prefix) => {
            let suffix = layered_suffix(prefix, vision_zero_prefix_ref, decoder_zero_prefix_ref);
            // Embedding is vocab-parallel at tp>1 (matches Python
            // VocabParallelEmbedding). Layered embed accessors don't
            // exist in any current arch but the helper is here for
            // codegen uniformity; non-`embed_tokens` last segments
            // fall back to the unsharded path.
            let kind = shard_kind_for_dotted_prefix(prefix);
            if sharded && kind == ShardKind::ShardDim0 {
                quote! {
                    ::ferrite_forward::load_layered_embedding_sharded(
                        gw, #n_lit, #dec_root_lit, #suffix, tp_rank as usize, #tp_world_lit as usize,
                    )?
                }
            } else {
                quote! {
                    ::ferrite_forward::load_layered_embedding(gw, #n_lit, #dec_root_lit, #suffix)?
                }
            }
        }
        FieldLoad::RmsNorm(prefix, eps) => {
            let suffix = layered_suffix(prefix, vision_zero_prefix_ref, decoder_zero_prefix_ref);
            if is_vision {
                let root = vision_root_lit_opt
                    .clone()
                    .expect("is_vision=true requires vision_layered_root");
                quote! {
                    ::ferrite_forward::load_layered_rms_norm_vision(gw, #n_lit, #root, #suffix, #eps)?
                }
            } else {
                quote! {
                    ::ferrite_forward::load_layered_rms_norm(gw, #n_lit, #dec_root_lit, #suffix, #eps)?
                }
            }
        }
        FieldLoad::LayerNorm(prefix, eps) => {
            let suffix = layered_suffix(prefix, vision_zero_prefix_ref, decoder_zero_prefix_ref);
            if is_vision {
                let root = vision_root_lit_opt
                    .clone()
                    .expect("is_vision=true requires vision_layered_root");
                quote! {
                    ::ferrite_forward::load_layered_layer_norm_vision(gw, #n_lit, #root, #suffix, #eps)?
                }
            } else {
                quote! {
                    ::ferrite_forward::load_layered_layer_norm(gw, #n_lit, #dec_root_lit, #suffix, #eps)?
                }
            }
        }
        FieldLoad::LinearDense(prefix) => {
            let suffix = layered_suffix(prefix, vision_zero_prefix_ref, decoder_zero_prefix_ref);
            let kind = shard_kind_for_dotted_prefix(prefix);
            match (sharded, kind, is_vision) {
                (true, ShardKind::ShardDim0, _) => quote! {
                    ::ferrite_forward::load_layered_linear_dense_sharded(
                        gw, #n_lit, #dec_root_lit, #suffix, 0usize, tp_rank as usize, #tp_world_lit as usize,
                    )?
                },
                (true, ShardKind::ShardDim1, _) => quote! {
                    ::ferrite_forward::load_layered_linear_dense_sharded(
                        gw, #n_lit, #dec_root_lit, #suffix, 1usize, tp_rank as usize, #tp_world_lit as usize,
                    )?
                },
                (_, _, true) => {
                    let root = vision_root_lit_opt
                        .clone()
                        .expect("is_vision=true requires vision_layered_root");
                    quote! {
                        ::ferrite_forward::load_layered_linear_dense_vision(gw, #n_lit, #root, #suffix)?
                    }
                }
                (_, _, false) => quote! {
                    ::ferrite_forward::load_layered_linear_dense(gw, #n_lit, #dec_root_lit, #suffix)?
                },
            }
        }
        FieldLoad::LinearConcat(prefixes) => {
            let suffixes: Vec<&str> = prefixes
                .iter()
                .map(|p| layered_suffix(p, vision_zero_prefix_ref, decoder_zero_prefix_ref))
                .collect();
            // Always column-parallel — no row-parallel concat exists.
            if sharded {
                quote! {
                    ::ferrite_forward::load_layered_linear_dense_concat_sharded(
                        gw,
                        #n_lit,
                        #dec_root_lit,
                        &[ #(#suffixes),* ],
                        stream,
                        tp_rank as usize,
                        #tp_world_lit as usize,
                    )?
                }
            } else if is_vision {
                // Vision-prelude per-block prefix is `<root>.<L>.`,
                // baked from `vision_safetensors_layout` in the per-
                // arch config (e.g. `visual.blocks` for Qwen,
                // `vision_tower.vision_model.encoder.layers` for
                // Gemma3-MM).
                let root = vision_root_lit_opt
                    .clone()
                    .expect("is_vision=true requires vision_layered_root");
                quote! {
                    ::ferrite_forward::load_layered_linear_dense_concat_vision(
                        gw,
                        #n_lit,
                        #root,
                        &[ #(#suffixes),* ],
                        stream,
                    )?
                }
            } else {
                quote! {
                    ::ferrite_forward::load_layered_linear_dense_concat(
                        gw,
                        #n_lit,
                        #dec_root_lit,
                        &[ #(#suffixes),* ],
                        stream,
                    )?
                }
            }
        }
        FieldLoad::LinearTiedToEmbedding(_) => panic!(
            "LinearTiedToEmbedding is only ever used for the unindexed \
             `lm_head` accessor — should never appear in a layered group"
        ),
        FieldLoad::RawLinear(_) => panic!(
            "RawLinear (nn.Parameter) is global by construction — \
             should never appear in a layered group"
        ),
        FieldLoad::MarlinLinear { prefixes, .. } => {
            if prefixes.len() == 1 {
                let suffix = layered_suffix(
                    &prefixes[0],
                    vision_zero_prefix_ref,
                    decoder_zero_prefix_ref,
                );
                quote! {
                    ::ferrite_forward::load_layered_marlin_linear(
                        gw, #n_lit, #dec_root_lit, #suffix, marlin_storage, __marlin_ws, __device_id,
                    )?
                }
            } else {
                let suffixes: Vec<&str> = prefixes
                    .iter()
                    .map(|p| layered_suffix(p, vision_zero_prefix_ref, decoder_zero_prefix_ref))
                    .collect();
                quote! {
                    ::ferrite_forward::load_layered_marlin_linear_concat(
                        gw, #n_lit, #dec_root_lit,
                        &[ #(#suffixes),* ],
                        marlin_storage, __marlin_ws, __device_id,
                    )?
                }
            }
        }
        FieldLoad::Fp8Linear { prefixes } => {
            if prefixes.len() == 1 {
                let suffix = layered_suffix(
                    &prefixes[0],
                    vision_zero_prefix_ref,
                    decoder_zero_prefix_ref,
                );
                quote! {
                    ::ferrite_forward::load_layered_fp8_linear(
                        gw, #n_lit, #dec_root_lit, #suffix, __fp8_dtype,
                    )?
                }
            } else {
                let suffixes: Vec<&str> = prefixes
                    .iter()
                    .map(|p| layered_suffix(p, vision_zero_prefix_ref, decoder_zero_prefix_ref))
                    .collect();
                quote! {
                    ::ferrite_forward::load_layered_fp8_linear_concat(
                        gw, #n_lit, #dec_root_lit,
                        &[ #(#suffixes),* ],
                        __fp8_dtype,
                    )?
                }
            }
        }
        FieldLoad::Fp8BlockLinear { prefixes } => {
            if prefixes.len() == 1 {
                let suffix = layered_suffix(
                    &prefixes[0],
                    vision_zero_prefix_ref,
                    decoder_zero_prefix_ref,
                );
                quote! {
                    ::ferrite_forward::load_layered_fp8_block_linear(
                        gw, #n_lit, #dec_root_lit, #suffix, __fp8_dtype,
                    )?
                }
            } else {
                let suffixes: Vec<&str> = prefixes
                    .iter()
                    .map(|p| layered_suffix(p, vision_zero_prefix_ref, decoder_zero_prefix_ref))
                    .collect();
                quote! {
                    ::ferrite_forward::load_layered_fp8_block_linear_concat(
                        gw, #n_lit, #dec_root_lit,
                        &[ #(#suffixes),* ],
                        __fp8_dtype,
                    )?
                }
            }
        }
        FieldLoad::Bnb4Linear {
            prefixes,
            out_features_per_shard,
            in_features,
            blocksize,
        } => {
            let in_features = *in_features as usize;
            let blocksize = *blocksize as usize;
            let outs: Vec<proc_macro2::Literal> = out_features_per_shard
                .iter()
                .map(|o| proc_macro2::Literal::usize_unsuffixed(*o as usize))
                .collect();
            if prefixes.len() == 1 {
                let suffix = layered_suffix(
                    &prefixes[0],
                    vision_zero_prefix_ref,
                    decoder_zero_prefix_ref,
                );
                let out = &outs[0];
                quote! {
                    ::ferrite_forward::load_layered_bnb4(
                        gw, #n_lit, #dec_root_lit, #suffix,
                        __bnb_code, __bnb_scratch,
                        #out, #in_features, #blocksize,
                    )?
                }
            } else {
                let suffixes: Vec<&str> = prefixes
                    .iter()
                    .map(|p| layered_suffix(p, vision_zero_prefix_ref, decoder_zero_prefix_ref))
                    .collect();
                quote! {
                    ::ferrite_forward::load_layered_bnb4_concat(
                        gw, #n_lit, #dec_root_lit,
                        &[ #(#suffixes),* ],
                        __bnb_code, __bnb_scratch,
                        &[ #(#outs),* ],
                        #in_features, #blocksize,
                    )?
                }
            }
        }
        FieldLoad::DeepSeekV2Moe {
            prefix,
            n_routed_experts,
            n_shared_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            norm_topk_prob,
            routed_scaling_factor,
            use_sigmoid,
            n_expert_group,
            topk_group,
        } => {
            let p = layer_templated_prefix_expr(
                prefix,
                vision_zero_prefix_ref,
                decoder_zero_prefix_ref,
            );
            let n_routed_experts = *n_routed_experts;
            let n_shared_experts = *n_shared_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let hidden_size = *hidden_size;
            let norm_topk_prob = *norm_topk_prob;
            let routed_scaling_factor = *routed_scaling_factor;
            let use_sigmoid = *use_sigmoid;
            let n_expert_group = *n_expert_group;
            let topk_group = *topk_group;
            quote! {
                (0u32..#n_lit)
                    .map(|layer: u32| -> ::anyhow::Result<_> {
                        ::ferrite_kernels::layers_moe::DeepSeekV2MoELayer::load(
                            gw,
                            &#p,
                            #n_routed_experts,
                            #n_shared_experts,
                            #top_k,
                            #moe_intermediate_size,
                            #hidden_size,
                            #norm_topk_prob,
                            #routed_scaling_factor,
                            #use_sigmoid,
                            #n_expert_group,
                            #topk_group,
                            stream,
                        )
                    })
                    .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
            }
        }
        FieldLoad::DeepSeekV2Fp8BlockMoe {
            prefix,
            n_routed_experts,
            n_shared_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            norm_topk_prob,
            routed_scaling_factor,
            use_sigmoid,
            n_expert_group,
            topk_group,
        } => {
            let p = layer_templated_prefix_expr(
                prefix,
                vision_zero_prefix_ref,
                decoder_zero_prefix_ref,
            );
            let n_routed_experts = *n_routed_experts;
            let n_shared_experts = *n_shared_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let hidden_size = *hidden_size;
            let norm_topk_prob = *norm_topk_prob;
            let routed_scaling_factor = *routed_scaling_factor;
            let use_sigmoid = *use_sigmoid;
            let n_expert_group = *n_expert_group;
            let topk_group = *topk_group;
            quote! {
                (0u32..#n_lit)
                    .map(|layer: u32| -> ::anyhow::Result<_> {
                        ::ferrite_kernels::layers_moe::DeepSeekV2Fp8BlockMoELayer::load(
                            gw,
                            &#p,
                            #n_routed_experts,
                            #n_shared_experts,
                            #top_k,
                            #moe_intermediate_size,
                            #hidden_size,
                            #norm_topk_prob,
                            #routed_scaling_factor,
                            #use_sigmoid,
                            #n_expert_group,
                            #topk_group,
                            __fp8_dtype,
                            stream,
                        )
                    })
                    .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
            }
        }
        FieldLoad::DeepSeekV2GgmlMoe {
            prefix,
            n_routed_experts,
            n_shared_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            norm_topk_prob,
            routed_scaling_factor,
            use_sigmoid,
            n_expert_group,
            topk_group,
        } => {
            let p = layer_templated_prefix_expr(
                prefix,
                vision_zero_prefix_ref,
                decoder_zero_prefix_ref,
            );
            let n_routed_experts = *n_routed_experts;
            let n_shared_experts = *n_shared_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let hidden_size = *hidden_size;
            let norm_topk_prob = *norm_topk_prob;
            let routed_scaling_factor = *routed_scaling_factor;
            let use_sigmoid = *use_sigmoid;
            let n_expert_group = *n_expert_group;
            let topk_group = *topk_group;
            quote! {
                (0u32..#n_lit)
                    .map(|layer: u32| -> ::anyhow::Result<_> {
                        ::ferrite_kernels::layers_moe::DeepSeekV2GgmlMoELayer::load_gguf(
                            gw,
                            &#p,
                            #n_routed_experts,
                            #n_shared_experts,
                            #top_k,
                            #moe_intermediate_size,
                            #hidden_size,
                            #norm_topk_prob,
                            #routed_scaling_factor,
                            #use_sigmoid,
                            #n_expert_group,
                            #topk_group,
                            stream,
                        )
                    })
                    .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
            }
        }
        FieldLoad::FusedMoe {
            prefix,
            num_experts,
            top_k,
            intermediate_size,
            hidden_size,
        } => {
            let p = layer_templated_prefix_expr(
                prefix,
                vision_zero_prefix_ref,
                decoder_zero_prefix_ref,
            );
            let num_experts = *num_experts;
            let top_k = *top_k;
            let intermediate_size = *intermediate_size;
            let hidden_size = *hidden_size;
            quote! {
                (0u32..#n_lit)
                    .map(|layer: u32| -> ::anyhow::Result<_> {
                        ::ferrite_kernels::layers_moe::FusedMoELayer::load(
                            gw,
                            &#p,
                            #num_experts,
                            #top_k,
                            #intermediate_size,
                            #hidden_size,
                            stream,
                        )
                    })
                    .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
            }
        }
        FieldLoad::SharedFusedMoe {
            prefix,
            num_experts,
            top_k,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            hidden_size,
        } => {
            let p = layer_templated_prefix_expr(
                prefix,
                vision_zero_prefix_ref,
                decoder_zero_prefix_ref,
            );
            let num_experts = *num_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let shared_expert_intermediate_size = *shared_expert_intermediate_size;
            let hidden_size = *hidden_size;
            quote! {
                (0u32..#n_lit)
                    .map(|layer: u32| -> ::anyhow::Result<_> {
                        ::ferrite_kernels::layers_moe::SharedFusedMoELayer::load(
                            gw,
                            &#p,
                            #num_experts,
                            #top_k,
                            #moe_intermediate_size,
                            #shared_expert_intermediate_size,
                            #hidden_size,
                            stream,
                        )
                    })
                    .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
            }
        }
    }
}

/// Emit one accessor method per base name on the per-arch `Weights`
/// struct. Layered bases (`input_layernorm_0`, `input_layernorm_1`,
/// …) — backed by a single `Vec<T>` field — produce
/// `pub fn input_layernorm(&self, layer: u32) -> &RmsNorm { &self.input_layernorm[layer as usize] }`.
/// Non-layered bases get the same signature for caller uniformity;
/// the body returns `&self.<base>` and ignores the layer arg.
///
/// Compresses what used to be a 40-arm `match layer` per accessor
/// into a single slice index — the 578-line `impl Weights { … }`
/// block on commandr collapses to ~50 lines, and llama's 28k-line
/// equivalent collapses by the same proportion.
///
/// Returns an `impl Weights { ... }` block. Empty (zero accessors)
/// is fine — the impl block is then elided entirely.
fn emit_weights_accessor_methods(accessors: &[WeightAccessor]) -> TokenStream {
    let groups = group_accessors_by_base(accessors);
    if groups.is_empty() {
        return TokenStream::new();
    }
    // Per-method `#[cfg]` / `#[inline]` / `#[allow(dead_code)]` are
    // redundant — the impl block carries the cfg, the methods are
    // trivial enough that LLVM inlines them in release without the
    // hint, and the unused-warning is suppressed at the impl level.
    // Dropping per-method attrs collapses each accessor from ~7
    // lines to ~3.
    let methods: Vec<TokenStream> = groups
        .iter()
        .map(|g| {
            let base_ident = syn::Ident::new(&g.base, proc_macro2::Span::call_site());
            let ty = &g.rust_type;
            match g.kind {
                AccessorGroupKind::LayeredContiguous => quote! {
                    pub fn #base_ident(&self, layer: u32) -> &#ty {
                        unsafe { self.#base_ident.get_unchecked(layer as usize) }
                    }
                },
                AccessorGroupKind::Unindexed => quote! {
                    pub fn #base_ident(&self, _: u32) -> &#ty { &self.#base_ident }
                },
                AccessorGroupKind::LayeredSparse => {
                    // Per-layer fields → match arms. Codegen-issued
                    // static rows only ever pass layers that exist
                    // (guaranteed by fan_out's per-claim layer
                    // plumbing), so `unreachable_unchecked()` on non-
                    // emitted layers is safe.
                    let arms: Vec<TokenStream> = g
                        .entries
                        .iter()
                        .map(|(layer, acc)| {
                            let layer_lit = proc_macro2::Literal::u32_unsuffixed(
                                layer.expect("LayeredSparse entry has Some(layer)") as u32,
                            );
                            let fname = &acc.name;
                            quote! { #layer_lit => &self.#fname, }
                        })
                        .collect();
                    quote! {
                        pub fn #base_ident(&self, layer: u32) -> &#ty {
                            match layer {
                                #(#arms)*
                                _ => unsafe { ::core::hint::unreachable_unchecked() },
                            }
                        }
                    }
                }
            }
        })
        .collect();
    quote! {
        #[cfg(feature = "cuda")]
        #[allow(dead_code)]
        impl Weights {
            #(#methods)*
        }
    }
}

/// Emit `impl ::ferrite_forward::WeightAccessors for Weights { ... }`
/// keyed on `(bucket, op_idx)` per the typed-fanout design.
///
/// The variant of an `Instruction` at position `op_idx` in `bucket`
/// fixes the *kind* of weight needed; the per-arch impl resolves
/// `(bucket, op_idx)` → which named field on `Weights`. We walk
/// every bucket × position, look up the recorded `weight_slots`,
/// and emit one match arm per (bucket, op_idx, kind) triple.
///
/// Sprint 1 scope: `rms_norm_at` only. Sibling methods (`linear_at`,
/// `embedding_at`, `cos_sin_at`, MoE getters) land per sprint as
/// the matching `Instruction<W>` variants migrate off
/// `WtFn`/`CosSinFn`.
fn emit_weight_accessors_impl(
    canonical_lowered: &BTreeMap<crate::solver::WorkloadPoint, (CanonicalLowered, u32, u32, u32)>,
) -> TokenStream {
    use crate::impl_lib::WeightKind;

    // Walk every BUCKET × position in declaration order. The bucket
    // index in FORWARD_TABLE matches the (m, sk) row order we emit;
    // we use a synthetic "bucket id" of `2 * row_idx + slice_idx`
    // where `slice_idx` is 0 for backbone, 1 for lm_head — matching
    // `BUCKET_BACKBONE` / `BUCKET_LM_HEAD` for the simple
    // single-bucket case.
    //
    // For now (Sprint 1 + simple model lowering): every bucket row
    // shares the same canonical's instruction stream, so the match
    // table only needs entries for `(BUCKET_BACKBONE, op_idx)` and
    // `(BUCKET_LM_HEAD, op_idx)` from any one bucket row. We use
    // the first canonical entry's lowered slices as the source.
    //
    // When per-bucket variation lands (different Impls per workload
    // point), this becomes per-bucket-row and the bucket_id encoding
    // expands; the runtime caller of `rms_norm_at` will pass the
    // matching encoded id from the FORWARD_TABLE row it dispatched
    // through.
    // For each kind, collect `(bucket, op_idx, slot) => self.<base>(layer)`
    // match arms across every canonical's backbone and lm_head. `slot` is
    // the per-(bucket, op_idx, kind) ordinal, derived by walking
    // `weight_slots` in declaration order and counting prior occurrences
    // of the same kind at the same op.
    use std::collections::HashMap;
    let mut by_kind: HashMap<&'static str, Vec<TokenStream>> = HashMap::new();

    // Walk EVERY canonical lowered entry. Each distinct bucket
    // (workload point) gets a pair of bucket ids: 2*ci (backbone)
    // and 2*ci+1 (lm_head). FORWARD_TABLE rows pass these ids into
    // run/run_backbone, which forward them to run_slice → eval →
    // the per-arch WeightAccessors match arms.
    let emit_for = |bucket_id: u32,
                    weight_slots: &[Vec<WeightSlot>],
                    by_kind: &mut HashMap<&'static str, Vec<TokenStream>>| {
        let bucket_lit = proc_macro2::Literal::u32_unsuffixed(bucket_id);
        for (op_idx, slots) in weight_slots.iter().enumerate() {
            let op_lit = proc_macro2::Literal::u32_unsuffixed(op_idx as u32);
            let mut counts: HashMap<&'static str, u32> = HashMap::new();
            for slot in slots {
                let key = match slot.kind {
                    WeightKind::RmsNorm => "rms_norm_at",
                    WeightKind::Embedding => "embedding_at",
                    WeightKind::Linear => "linear_at",
                    WeightKind::LayerNorm => "layer_norm_at",
                    WeightKind::Marlin => "marlin_at",
                    WeightKind::Bnb4 => "bnb4_at",
                    WeightKind::Fp8 => "fp8_at",
                    WeightKind::DeepSeekMoe => "deepseek_moe_at",
                    WeightKind::DeepSeekMoeFp8 => "deepseek_moe_fp8_at",
                    WeightKind::DeepSeekMoeGgml => "deepseek_moe_ggml_at",
                    WeightKind::FusedMoe => "fused_moe_at",
                    WeightKind::SharedFusedMoe => "shared_fused_moe_at",
                    WeightKind::CosSin => "cos_sin_at",
                };
                let n = counts.entry(key).or_insert(0);
                let slot_lit = proc_macro2::Literal::u32_unsuffixed(*n);
                *n += 1;
                let base = &slot.base;
                // CosSin pulls from a `RotaryCache` field on the per-arch
                // `Weights` struct (`wm.rotary` or `wm.rotary_local`),
                // not from a `fn <base>(layer) -> &T` getter — rotary is
                // shared across layers, and the cache itself owns a
                // single `cos_sin_cache: GpuTensor`. The trait method
                // returns `GpuTensor` by value, so `.clone()` produces
                // a cheap handle copy.
                let arm_body = match slot.kind {
                    WeightKind::CosSin => quote! { self.#base.cos_sin_cache.clone() },
                    _ => quote! { self.#base(layer) },
                };
                by_kind.entry(key).or_default().push(quote! {
                    (#bucket_lit, #op_lit, #slot_lit) => #arm_body,
                });
            }
        }
    };
    for (ci, (_wp, (cl, _, _, _))) in canonical_lowered.iter().enumerate() {
        let bb_id = (ci as u32) * 2;
        let lm_id = bb_id + 1;
        emit_for(bb_id, &cl.backbone.weight_slots, &mut by_kind);
        emit_for(lm_id, &cl.lm_head.weight_slots, &mut by_kind);
    }

    let method_emit = |method: &str, ret_ty: TokenStream| -> TokenStream {
        let method_id = syn::Ident::new(method, proc_macro2::Span::call_site());
        let arms = by_kind.get(method).cloned().unwrap_or_default();
        if arms.is_empty() {
            // Default trait body already returns `unreachable!()`. No
            // override needed when the arch never consumes this kind.
            return quote! {};
        }
        quote! {
            fn #method_id(
                &self,
                bucket: u32,
                op_idx: u32,
                slot: u32,
                layer: u32,
            ) -> #ret_ty {
                // `layer` is unused for `cos_sin_at` (rotary is layer-
                // independent); referenced explicitly here so the
                // generated body type-checks identically across every
                // method.
                let _ = layer;
                match (bucket, op_idx, slot) {
                    #(#arms)*
                    _ => unreachable!(
                        "WeightAccessors::{}: no match for (bucket={}, op_idx={}, slot={})",
                        stringify!(#method_id), bucket, op_idx, slot,
                    ),
                }
            }
        }
    };

    let rms_norm = method_emit(
        "rms_norm_at",
        quote! { &::ferrite_kernels::layers::RmsNorm },
    );
    let embedding = method_emit(
        "embedding_at",
        quote! { &::ferrite_kernels::layers::Embedding },
    );
    let linear = method_emit(
        "linear_at",
        quote! { &::ferrite_kernels::layers::LinearLayer },
    );
    let layer_norm = method_emit(
        "layer_norm_at",
        quote! { &::ferrite_kernels::layers::LayerNorm },
    );
    let marlin = method_emit(
        "marlin_at",
        quote! { &::ferrite_kernels::layers::MarlinLinear },
    );
    let bnb4 = method_emit(
        "bnb4_at",
        quote! { &::ferrite_kernels::layers::Bnb4bitLinear },
    );
    let fp8 = method_emit(
        "fp8_at",
        quote! { &::ferrite_kernels::layers::Fp8AnyLinear },
    );
    let dsmoe = method_emit(
        "deepseek_moe_at",
        quote! { &::ferrite_kernels::layers_moe::DeepSeekV2MoELayer },
    );
    let dsmoe_fp8 = method_emit(
        "deepseek_moe_fp8_at",
        quote! { &::ferrite_kernels::layers_moe::DeepSeekV2Fp8BlockMoELayer },
    );
    let dsmoe_ggml = method_emit(
        "deepseek_moe_ggml_at",
        quote! { &::ferrite_kernels::layers_moe::DeepSeekV2GgmlMoELayer },
    );
    let fused_moe = method_emit(
        "fused_moe_at",
        quote! { &::ferrite_kernels::layers_moe::FusedMoELayer },
    );
    let shared_moe = method_emit(
        "shared_fused_moe_at",
        quote! { &::ferrite_kernels::layers_moe::SharedFusedMoELayer },
    );
    let cos_sin = method_emit(
        "cos_sin_at",
        quote! { ::ferrite_cuda_core::tensor::GpuTensor },
    );

    quote! {
        #[cfg(feature = "cuda")]
        impl ::ferrite_forward::WeightAccessors for Weights {
            #rms_norm
            #embedding
            #linear
            #layer_norm
            #marlin
            #bnb4
            #fp8
            #dsmoe
            #dsmoe_fp8
            #dsmoe_ggml
            #fused_moe
            #shared_moe
            #cos_sin
        }
    }
}

/// Emit the `MarlinFormat` const this variant's `load` passes into
/// the (canonical or shared) `load_with` body. For non-Marlin
/// variants (dense / BNB4 / FP8 later) the value is still a valid
/// `MarlinFormat` — the `load_with` body just doesn't reference
/// `marlin_storage`, so the const is never read.
fn marlin_format_literal(model: &ModelParams) -> TokenStream {
    use crate::quantization::QuantMethod;
    match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(QuantMethod::Awq { group_size, .. }) => {
            let gs = proc_macro2::Literal::u32_unsuffixed(*group_size);
            quote! {
                ::ferrite_kernels::layers_quant::MarlinFormat::Awq { group_size: #gs }
            }
        }
        Some(QuantMethod::Gptq {
            group_size,
            desc_act,
            layout,
            ..
        }) => {
            let gs = proc_macro2::Literal::u32_unsuffixed(*group_size);
            let da = *desc_act;
            let layout_ts = gptq_layout_ts(*layout);
            quote! {
                ::ferrite_kernels::layers_quant::MarlinFormat::Gptq {
                    group_size: #gs,
                    desc_act: #da,
                    layout: #layout_ts,
                }
            }
        }
        // Dense / BNB4 / other: emit a placeholder; `load_with` in
        // those equivalence classes never reads the param. Keeping
        // a non-unit value here means the emitted const is well-
        // typed regardless of arch / quant family.
        _ => quote! {
            ::ferrite_kernels::layers_quant::MarlinFormat::Awq { group_size: 128 }
        },
    }
}

// ── Forward fn emission ──────────────────────────────────────────
//
// The macro lowers each canonical bucket's solved FUF into a flat
// instruction list (a `&[Op]` static slice), emits one per-arch
// `Op` enum + one per-arch `__interpret` helper, then emits one
// thin per-bucket fn per (forward, backbone) × workload-point that:
// 1. allocates the runtime tile table,
// 2. runs the alias prelude (zero-copy `View` aliases the lowering
//    surfaced via `output_alias`),
// 3. calls `__interpret(&FORWARD_M_<N>, &mut __tiles, …)`,
// 4. takes ownership of the slot the lowering tagged as final.
//
// Buckets in the same SFUF equivalence class as a canonical share
// the canonical's body via thin `#[inline(always)]` wrappers — same
// dedup the previous codegen path used.

/// Ident for a per-workload-bucket forward fn. Name is
/// `<prefix>_<m>` when `sk_bucket == 0` (legacy 1-D sweep) and
/// `<prefix>_<m>_sk_<sk>` otherwise. Preserves the pre-sk naming
/// for models that don't opt into an sk axis.
fn bucket_fn_ident(prefix: &str, wp: crate::solver::WorkloadPoint) -> proc_macro2::Ident {
    if wp.sk_bucket == 0 {
        format_ident!("{}_{}", prefix, wp.num_tokens)
    } else {
        format_ident!("{}_{}_sk_{}", prefix, wp.num_tokens, wp.sk_bucket)
    }
}

/// Ident for a per-workload-bucket `static <PREFIX>_<m>: &[Op]`.
/// Mirrors [`bucket_fn_ident`] in shape but uses upper-case so the
/// emitted module reads naturally — `FORWARD_M_64` /
/// `BACKBONE_M_64_SK_2048`.
fn bucket_static_ident(prefix: &str, wp: crate::solver::WorkloadPoint) -> proc_macro2::Ident {
    if wp.sk_bucket == 0 {
        format_ident!("{}_{}", prefix, wp.num_tokens)
    } else {
        format_ident!("{}_{}_SK_{}", prefix, wp.num_tokens, wp.sk_bucket)
    }
}

/// Per-canonical-bucket lowering products. The `backbone` slice
/// carries everything the forward pass needs except the terminal
/// `lm_head` gemm; the `lm_head` slice carries that single row.
/// `forward` runs both, `forward_backbone` runs only the backbone
/// and DtoD-copies the backbone-output slot.
struct CanonicalLowered {
    backbone: crate::interpreter_codegen::LoweredBucket,
    lm_head: crate::interpreter_codegen::LoweredBucket,
}

/// Walk the FUF from the end, skipping nodes appended by lowering
/// passes that aren't the body's terminal output:
///
/// - `OpKind::MmEmbedSplice` — `tp_lowering::insert_mm_splices`
///   pushes one per image-bearing batch; semantically adjacent to
///   the Embed, lives at array tail for `push`-based insertion.
/// - `OpKind::LoadPixels` — `vision_lowering::materialize_pixels`
///   pushes one to materialize the `pixels` extern as a tile;
///   semantically the FIRST op (everything reads from it), but
///   lives at array tail for the same `push`-based reason.
///
/// The "last node" for backbone-output / terminal-subgraph
/// identification must be the body's actual terminal — the lm_head
/// Gemm (tp=1) or its AllGather wrapper (tp>1) for decoders, the
/// merger MLP's bias_add for vision encoders.
fn last_non_splice_node(fuf: &Fuf) -> Option<&crate::fuf::FufNode> {
    fuf.nodes.iter().rev().find(|n| {
        !matches!(
            n.op,
            crate::classified::OpKind::MmEmbedSplice | crate::classified::OpKind::LoadPixels
        )
    })
}

/// How the FUF's terminal node maps onto the backbone/lm_head split.
///
/// - [`BackboneLayout::Decoder`] — the FUF ends in `gemm(<tile>,
///   lm_head)` (or that gemm followed by an `AllGather` at tp>1).
///   The backbone is everything except the terminal subgraph; the
///   `(TileId, u8)` it carries is the lm_head Gemm's hidden-state
///   input — the slot `forward_backbone` returns and the slot the
///   lm_head slice reads.
/// - [`BackboneLayout::Encoder`] — the FUF's terminal is NOT
///   `gemm(_, lm_head)`. Covers text-side encoders (ModernBERT)
///   AND every `#[vision_forward]` body (Qwen2-VL / Qwen2.5-VL /
///   SigLIP / …) since vision bodies don't have a trainable
///   lm_head. The whole pipeline is the backbone, the lm_head slice
///   is empty, and `forward` returns the FUF's last-node output
///   directly.
#[derive(Clone, Copy, Debug)]
enum BackboneLayout {
    Decoder { backbone_out: (TileId, u8) },
    Encoder,
}

/// Classify the FUF's terminal as decoder vs encoder. A decoder
/// terminal is `gemm(<tile>, <lm_head_weight>)`, optionally followed
/// by an `AllGather` (inserted by tp>1 lowering on the vocab-parallel
/// lm_head Gemm). Walks past trailing `MmEmbedSplice` / `LoadPixels`
/// nodes via [`last_non_splice_node`] — those are appended by
/// lowering passes but aren't the body's actual terminal.
fn backbone_layout(fuf: &Fuf, program: &Program) -> BackboneLayout {
    const LM_HEAD_PREFIX: &str = "lm_head";

    let last_node = last_non_splice_node(fuf).expect("FUF must be non-empty to emit a forward fn");
    let lm_head_node = if last_node.op == crate::classified::OpKind::AllGather {
        // tp>1: walk past the AllGather to the underlying Gemm.
        // AllGather has exactly one tile input by construction.
        match last_node.inputs.first() {
            Some(FufInput::Tile { id, .. }) => fuf.get(*id),
            _ => return BackboneLayout::Encoder,
        }
    } else {
        last_node
    };

    // Decoder shape: `gemm(<tile>, <lm_head weight>)`.
    if lm_head_node.op != crate::classified::OpKind::Gemm {
        return BackboneLayout::Encoder;
    }
    let weight_is_lm_head = matches!(
        lm_head_node.inputs.get(1),
        Some(FufInput::Weight { id, .. })
            if program
                .weights
                .path(*id)
                .first()
                .map(|s| s.as_str())
                == Some(LM_HEAD_PREFIX)
    );
    if !weight_is_lm_head {
        return BackboneLayout::Encoder;
    }
    match lm_head_node.inputs.first() {
        Some(FufInput::Tile { id, slot }) => BackboneLayout::Decoder {
            backbone_out: (*id, *slot),
        },
        _ => BackboneLayout::Encoder,
    }
}

/// Build the workload-point bounds map `lower_bucket` and Impls
/// consume — model.bounds + the workload-specific `num_tokens` and
/// `sk_bucket` overrides. Mirrors what `solve_workloads` does before
/// each per-point solve.
fn bounds_for_wp(
    model: &ModelParams,
    wp: crate::solver::WorkloadPoint,
    tp_world_size: u8,
) -> BTreeMap<String, u64> {
    let mut bounds = model.bounds.clone();
    bounds.insert("num_tokens".to_string(), wp.num_tokens);
    bounds.insert("sk_bucket".to_string(), wp.sk_bucket);
    // At tp>1, runtime weights are per-rank shards; the codegen
    // `gemm_nk_from_fuf` evaluates the FUF's symbolic `Shape`
    // against this bounds map and bakes the resulting (n, k) into
    // the emitted `Instruction` for `assert_weight_shape` to check
    // at runtime. Per-rank weights ⇒ per-rank bounds, or the
    // assertion (added in 812452cff) panics with a sharded/unsharded
    // K mismatch on the first row-parallel gemm.
    if tp_world_size > 1 {
        let tp = tp_world_size as u64;
        for k in [
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
        ] {
            if let Some(v) = bounds.get_mut(k) {
                *v = (*v / tp).max(1);
            }
        }
    }
    bounds
}

/// Render the alias-prelude statements for one [`LoweredBucket`].
/// Each `(dst, src)` becomes `__tiles[dst as usize] =
/// Some(TileEntry::View { ref_slot: src });`. Aliases are dropped
/// in the lowering when `dst == src`, so callers don't need a
/// guard.
fn emit_alias_prelude(aliases: &[(u32, u32)]) -> Vec<TokenStream> {
    // Each row: `__tiles[dst] = Some(view(src));` — 1 line.
    // `view()` is a tiny helper on `ferrite_forward` that wraps
    // `TileEntry::View { ref_slot }`; using it instead of inlining
    // the struct literal keeps prettyplease from breaking each
    // alias onto three lines, which matters because there are
    // hundreds of aliases per bucket on the deeper models.
    aliases
        .iter()
        .map(|(dst, src)| {
            let dst = proc_macro2::Literal::u32_unsuffixed(*dst);
            let src = proc_macro2::Literal::u32_unsuffixed(*src);
            quote! { __tiles[#dst as usize] = Some(::ferrite_forward::view(#src)); }
        })
        .collect()
}

/// Render the comma-separated impl-name list (`"rmsnorm_ref,
/// fused_qkv_rope_cache, …"`) the per-bucket fn passes to
/// `tracing::debug!`. `filter_terminal` drops the terminal subgraph
/// for the backbone fn's listing.
fn impl_names_for(
    loop_ir: &crate::schedule::Loop,
    lib: &ImplementationLibrary,
    skip_terminal: Option<crate::solver::SubgraphId>,
) -> String {
    loop_ir
        .waves
        .iter()
        .flat_map(|w| w.subgraphs.iter())
        .filter(|(sg, _)| Some(*sg) != skip_terminal)
        .map(|(_, imp_id)| lib.get(*imp_id).name())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Emit the full per-model module body: Weights struct + loader,
/// one forward fn per workload bucket, and a dispatching wrapper.
///
/// When `canonical_override` is `Some(ident)`, this variant is a
/// shim for that canonical sibling — emit `pub type Weights =
/// super::<ident>::Weights;` instead of a fresh struct, emit the
/// variant-specific `load` + `fingerprint_matches` bodies
/// (loaders differ per quant preset, fingerprints differ per
/// tensor-suffix gate), and `pub use` the canonical's forward +
/// forward_backbone + per-bucket forward_m_<N> fns. rustc doesn't
/// re-monomorphize `pub use` re-exports, so the canonical fn body
/// is optimized ONCE regardless of how many variants share it.
#[allow(clippy::too_many_arguments)]
/// Emit `impl ::ferrite_forward::CanonicalParams for Weights { … }` —
/// per-canonical model constants the universal `Instruction::eval`
/// reads as `W::HEAD_DIM`, `W::INTERMEDIATE_SIZE`, etc. Default
/// 0 / 0.0 / -1 for fields the canonical doesn't carry.
///
/// `tp_world_size` shards the column-parallel dims (`num_q_heads`,
/// `num_kv_heads`, `intermediate_size`) by `tp_world_size` — each
/// rank owns `1/tp` of those dims. The sharded values flow into the
/// emitted `Instruction::eval` body as `<W as CanonicalParams>::…`
/// constants, so kernel launches at tp>1 see the per-rank sizes
/// automatically. At `tp_world_size = 1` (every emission until task
/// #7's outer-loop fanout lands) sharding is identity — output is
/// byte-identical to single-rank builds.
///
/// Caller is responsible for ensuring `tp_world_size` evenly divides
/// every column-parallel dim (KV replication when
/// `num_kv_heads < tp_size` is task #6's loader-sharding work);
/// `compile()` skips a `(variant, tp)` tuple when divisibility fails.
fn emit_canonical_params_impl(model: &ModelParams, tp_world_size: u8) -> TokenStream {
    let tp = tp_world_size as u32;
    let tp_us = tp_world_size as usize;
    let head_dim = *model.bounds.get("head_dim").unwrap_or(&0) as u32;
    let num_q_heads = (*model.bounds.get("num_attention_heads").unwrap_or(&0) as u32) / tp;
    let num_kv_heads = (*model.bounds.get("num_key_value_heads").unwrap_or(&0) as u32) / tp;
    // For text bodies this comes straight off `intermediate_size`. For
    // vision-only bodies (like Qwen2.5-VL) the bound is absent, but
    // `vision_intermediate_size_padded` carries the SwiGLU intermediate
    // width that `silu_and_mul_fused`'s split-point reads out of
    // `W::INTERMEDIATE_SIZE` at runtime — so fall back to that. Vision
    // is replicated per-rank, so don't divide by tp.
    let intermediate_size = if let Some(v) = model.bounds.get("intermediate_size") {
        (*v as usize) / tp_us
    } else if let Some(v) = model.bounds.get("vision_intermediate_size_padded") {
        *v as usize
    } else {
        0
    };
    let q_size = (num_q_heads as usize) * (head_dim as usize);
    let kv_size = (num_kv_heads as usize) * (head_dim as usize);
    let kv_lora_rank = *model.bounds.get("kv_lora_rank").unwrap_or(&0) as usize;
    let qk_nope_head_dim = *model.bounds.get("qk_nope_head_dim").unwrap_or(&0) as usize;
    let qk_rope_head_dim = *model.bounds.get("qk_rope_head_dim").unwrap_or(&0) as usize;
    let v_head_dim = *model.bounds.get("v_head_dim").unwrap_or(&0) as usize;
    let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;

    // Vision-tower constants. Set in `#[vision_forward]` configs via
    // `vision_num_heads` / `vision_head_dim` bounds; absent in text
    // configs so the defaults (0 / 0.0) match the trait defaults
    // declared in `ferrite_forward::CanonicalParams`. Q_SIZE is the
    // rank-2 last-dim (`H*D`) the FUF carries; VISION_ATTN_SCALE is
    // `1/sqrt(head_dim)`.
    let vision_num_heads = *model.bounds.get("vision_num_heads").unwrap_or(&0) as u32;
    let vision_head_dim = *model.bounds.get("vision_head_dim").unwrap_or(&0) as u32;
    let vision_q_size = (vision_num_heads as usize) * (vision_head_dim as usize);
    let vision_attn_scale: f32 = if vision_head_dim > 0 {
        1.0_f32 / (vision_head_dim as f32).sqrt()
    } else {
        0.0
    };

    // SigLIP-style patch grid side (square); zero for arches that
    // don't carry an image-tower patch grid. `Instruction::AvgPool2d`
    // reads this to fold flat-row index → (row, col) on the encoder's
    // pooling pass.
    let vision_patch_grid_side = *model.bounds.get("vision_patch_grid_side").unwrap_or(&0) as u32;
    let vision_pool_kernel = *model.bounds.get("vision_pool_kernel").unwrap_or(&0) as u32;

    // attention_multiplier (Granite override) → query_pre_attn_scalar
    // (Gemma2) → 1/sqrt(head_dim). Default 0.0 if no attention path.
    let attn_scale: f32 = if let Some(s) = model.scalars.get("attention_multiplier") {
        *s as f32
    } else if let Some(q) = model.scalars.get("query_pre_attn_scalar") {
        (*q as f32).powf(-0.5)
    } else if head_dim > 0 {
        1.0_f32 / (head_dim as f32).sqrt()
    } else {
        0.0
    };
    let attn_softcap: f32 = model
        .scalars
        .get("attn_logit_softcapping")
        .copied()
        .unwrap_or(0.0) as f32;
    let sliding_window: i32 = model
        .bounds
        .get("sliding_window")
        .map(|w| *w as i32)
        .unwrap_or(-1);
    let final_logit_softcapping: f32 = model
        .scalars
        .get("final_logit_softcapping")
        .copied()
        .unwrap_or(0.0) as f32;
    // MLA scale: 1/sqrt(qk_head_dim) with YaRN mscale_all_dim
    // correction. Mirrors MlaAttentionImpl's previous bake.
    let mla_attn_scale: f32 = if qk_head_dim > 0 {
        let base = 1.0_f32 / (qk_head_dim as f32).sqrt();
        match &model.rope_scaling {
            Some(crate::config::RopeScaling::Yarn {
                factor,
                mscale_all_dim,
                ..
            }) if *mscale_all_dim != 0.0 => {
                let mm = if *factor <= 1.0 {
                    1.0_f64
                } else {
                    0.1 * mscale_all_dim * factor.ln() + 1.0
                };
                base * (mm * mm) as f32
            }
            _ => base,
        }
    } else {
        0.0
    };

    let head_dim_lit = proc_macro2::Literal::u32_unsuffixed(head_dim);
    let num_q_heads_lit = proc_macro2::Literal::u32_unsuffixed(num_q_heads);
    let num_kv_heads_lit = proc_macro2::Literal::u32_unsuffixed(num_kv_heads);
    let q_size_lit = proc_macro2::Literal::usize_unsuffixed(q_size);
    let kv_size_lit = proc_macro2::Literal::usize_unsuffixed(kv_size);
    let intermediate_size_lit = proc_macro2::Literal::usize_unsuffixed(intermediate_size);
    let kv_lora_rank_lit = proc_macro2::Literal::usize_unsuffixed(kv_lora_rank);
    let qk_nope_head_dim_lit = proc_macro2::Literal::usize_unsuffixed(qk_nope_head_dim);
    let qk_rope_head_dim_lit = proc_macro2::Literal::usize_unsuffixed(qk_rope_head_dim);
    let v_head_dim_lit = proc_macro2::Literal::usize_unsuffixed(v_head_dim);
    let qk_head_dim_lit = proc_macro2::Literal::usize_unsuffixed(qk_head_dim);
    let attn_scale_lit = proc_macro2::Literal::f32_unsuffixed(attn_scale);
    let attn_softcap_lit = proc_macro2::Literal::f32_unsuffixed(attn_softcap);
    let sliding_window_lit = proc_macro2::Literal::i32_unsuffixed(sliding_window);
    let final_logit_softcapping_lit = proc_macro2::Literal::f32_unsuffixed(final_logit_softcapping);
    let mla_attn_scale_lit = proc_macro2::Literal::f32_unsuffixed(mla_attn_scale);
    let vision_num_heads_lit = proc_macro2::Literal::u32_unsuffixed(vision_num_heads);
    let vision_head_dim_lit = proc_macro2::Literal::u32_unsuffixed(vision_head_dim);
    let vision_q_size_lit = proc_macro2::Literal::usize_unsuffixed(vision_q_size);
    let vision_attn_scale_lit = proc_macro2::Literal::f32_unsuffixed(vision_attn_scale);
    let vision_patch_grid_side_lit = proc_macro2::Literal::u32_unsuffixed(vision_patch_grid_side);
    let vision_pool_kernel_lit = proc_macro2::Literal::u32_unsuffixed(vision_pool_kernel);

    // MRoPE section override. `Some([t, h, w])` only when the
    // config carries `rope_scaling.mrope_section` (Qwen2-VL /
    // Qwen2.5-VL); every text-only arch keeps the default `None`
    // and the rope kernel takes the legacy 1D-positions fast path.
    // Sum-equals-`head_dim/2` is checked here (panic at expansion
    // time, not at runtime) — text decode of a misconfigured
    // multimodal variant never compiles past this guard.
    let mrope_section_tokens = match model.mrope_section {
        Some([t, h, w]) => {
            let pair_count = head_dim / 2;
            if t + h + w != pair_count {
                let msg = format!(
                    "model `{}`: rope_scaling.mrope_section [{t}, {h}, {w}] sums to {} \
                     but head_dim/2 = {pair_count}. Fix the config so the sum matches.",
                    model.source_stem,
                    t + h + w,
                );
                return quote! { compile_error!(#msg); };
            }
            let t_lit = proc_macro2::Literal::u32_unsuffixed(t);
            let h_lit = proc_macro2::Literal::u32_unsuffixed(h);
            let w_lit = proc_macro2::Literal::u32_unsuffixed(w);
            quote! {
                const MROPE_SECTION: ::core::option::Option<[u32; 3]> =
                    ::core::option::Option::Some([#t_lit, #h_lit, #w_lit]);
            }
        }
        None => quote! {},
    };

    quote! {
        #[cfg(feature = "cuda")]
        impl ::ferrite_forward::CanonicalParams for Weights {
            const HEAD_DIM: u32 = #head_dim_lit;
            const NUM_Q_HEADS: u32 = #num_q_heads_lit;
            const NUM_KV_HEADS: u32 = #num_kv_heads_lit;
            const Q_SIZE: usize = #q_size_lit;
            const KV_SIZE: usize = #kv_size_lit;
            const INTERMEDIATE_SIZE: usize = #intermediate_size_lit;
            const ATTN_SCALE: f32 = #attn_scale_lit;
            const ATTN_SOFTCAP: f32 = #attn_softcap_lit;
            const SLIDING_WINDOW: i32 = #sliding_window_lit;
            const KV_LORA_RANK: usize = #kv_lora_rank_lit;
            const QK_NOPE_HEAD_DIM: usize = #qk_nope_head_dim_lit;
            const QK_ROPE_HEAD_DIM: usize = #qk_rope_head_dim_lit;
            const V_HEAD_DIM: usize = #v_head_dim_lit;
            const FINAL_LOGIT_SOFTCAPPING: f32 = #final_logit_softcapping_lit;
            const QK_HEAD_DIM: usize = #qk_head_dim_lit;
            const MLA_ATTN_SCALE: f32 = #mla_attn_scale_lit;
            const VISION_NUM_HEADS: u32 = #vision_num_heads_lit;
            const VISION_HEAD_DIM: u32 = #vision_head_dim_lit;
            const VISION_Q_SIZE: usize = #vision_q_size_lit;
            const VISION_ATTN_SCALE: f32 = #vision_attn_scale_lit;
            const VISION_PATCH_GRID_SIDE: u32 = #vision_patch_grid_side_lit;
            const VISION_POOL_KERNEL: u32 = #vision_pool_kernel_lit;
            #mrope_section_tokens
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Walk every canonical in `canonical_lowered` and run the Tape-
/// level claim for each. The winning [`crate::tape_claim::TapeClaimer`]
/// emits its own compile-time artifacts (`.cu` files for
/// [`crate::tape::tk_mega::TkMegaTapeClaimer`]; nothing for
/// [`crate::tape::host_interp::HostInterpreterTapeClaimer`]). The
/// returned tokenstream is the concatenation of every claimer's
/// Rust-side declarations; the map surfaces per-bucket forward-fn
/// idents so [`emit_model`]'s dispatch table can point at them.
///
/// Ineligible canonicals (e.g. all-`Tk*` but model dims fail the
/// TK constraints) fall back to the host interpreter silently —
/// host always matches, so `pick()` can't return `None`.
/// Returns `(rust_decls, forward_fn_by_canonical, ms_forward_fn_by_canonical)`.
/// `ms_forward_fn_by_canonical` maps decode-only M=1 canonicals that have a
/// successfully-emitted `_ms` variant to their `forward_mega_ms_<canonical>`
/// fn ident. Consumed by the caller to build `MEGA_FORWARD_TABLE_MULTI_STEP`.
fn emit_mega_artifacts_inline(
    model: &ModelParams,
    canonical_lowered: &BTreeMap<crate::solver::WorkloadPoint, (CanonicalLowered, u32, u32, u32)>,
    arch_opcodes: &ArchOpcodes,
    accessor_type_by_base: &BTreeMap<String, String>,
    tp_world_size: u8,
    target_profile: &crate::target::TargetProfile,
) -> (
    TokenStream,
    BTreeMap<crate::solver::WorkloadPoint, Ident>,
    BTreeMap<crate::solver::WorkloadPoint, Ident>,
    BTreeMap<crate::solver::WorkloadPoint, Ident>,
) {
    use crate::tape_claim::{TapeEmitCtx, starter_tape_library};
    let shapes = arch_opcodes.shapes_by_name();
    let eps = rms_norm_eps(model);
    let library = starter_tape_library();

    let ctx = TapeEmitCtx {
        shapes: &shapes,
        rms_norm_eps: eps,
        profile: target_profile,
        tp_world_size,
        bounds: &model.bounds,
        scalars: &model.scalars,
        accessor_type_by_base,
    };

    type WpIdentMap = BTreeMap<crate::solver::WorkloadPoint, Ident>;
    let mut rust_decls = TokenStream::new();
    let mut canonical_forward_fn: WpIdentMap = BTreeMap::new();
    let mut canonical_ms_forward_fn: WpIdentMap = BTreeMap::new();
    let mut canonical_pd_start_fn: WpIdentMap = BTreeMap::new();

    for (wp, (lowered, _num_slots, _backbone_slot, terminal_slot)) in canonical_lowered {
        let canonical_name = format!(
            "{}_m_{}_sk_{}",
            model
                .source_stem
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .collect::<String>(),
            wp.num_tokens,
            wp.sk_bucket
        );

        let backbone = &lowered.backbone.instances;
        let lm_head = &lowered.lm_head.instances;

        let Some((idx, info)) = library.pick(backbone, lm_head, &ctx) else {
            panic!(
                "tape claim: no executor matched canonical `{canonical_name}`; \
                 check that the library contains HostInterpreterTapeClaimer"
            );
        };

        let claimer = library.claimer(idx);
        let emission = claimer.emit(
            &canonical_name,
            *wp,
            backbone,
            lm_head,
            *terminal_slot,
            &ctx,
            info.as_ref(),
        );

        if let Some(path) = &emission.cu_path {
            eprintln!(
                "tape claim: `{canonical_name}` claimed by `{}`, wrote {} ({} backbone ops, {} lm_head ops)",
                claimer.name(),
                path.display(),
                backbone.len(),
                lm_head.len()
            );
        }

        rust_decls.extend(emission.rust_decls);
        if let Some(fn_ident) = emission.forward_fn {
            canonical_forward_fn.insert(*wp, fn_ident);
        }
        if let Some(ms_fn_ident) = emission.ms_launch_fn {
            canonical_ms_forward_fn.insert(*wp, ms_fn_ident);
        }
        if let Some(pd_fn_ident) = emission.persistent_decode_launch_fn {
            canonical_pd_start_fn.insert(*wp, pd_fn_ident);
        }
    }

    (rust_decls, canonical_forward_fn, canonical_ms_forward_fn, canonical_pd_start_fn)
}

/// Render one `*const u16` expression per catalog-ordered accessor,
/// mapping each accessor's Rust return type (looked up in
/// `accessor_type_by_base`) to the appropriate path through the
/// per-arch `Weights` struct. Used by [`emit_mega_artifacts_inline`]
/// to populate the `match w_idx { 0 => …, 1 => …, … }` arm in the
/// emitted `forward_mega_<canonical>` body.
///
/// The return type → pointer-path mapping:
///
/// | Rust return type (trimmed)              | Pointer path |
/// | --------------------------------------- | ------------ |
/// | `LinearLayer` / `…::LinearLayer`        | `wm.<base>(layer).dense_weight().as_ptr::<u16>()` |
/// | `Embedding` / `…::Embedding`            | `wm.<base>(layer).weight.as_ptr::<u16>()` |
/// | `RmsNorm` / `…::RmsNorm`                | `wm.<base>(layer).weight.as_ptr::<u16>()` |
///
/// Special-cased synthesized accessors (not in the collected-accessor
/// map; emitted by `rotary_cos_sin_methods` / `rotary_local_cos_sin`):
///
/// | Accessor base          | Pointer path |
/// | ---------------------- | ------------ |
/// | `rotary_cos_sin`       | `wm.rotary_cos_sin(layer).as_ptr::<u16>()` |
/// | `rotary_local_cos_sin` | `wm.rotary_local_cos_sin(layer).as_ptr::<u16>()` |
///
/// An accessor whose type isn't recognized produces a
/// `compile_error!` expression in its slot — this surfaces the
/// accessor name + type so the reader can either add a new arm here
/// or mark the variant `#error` at `canonical_mega_meta` time.
pub(crate) fn build_mega_accessor_ptr_exprs(
    accessors: &[String],
    accessor_type_by_base: &BTreeMap<String, String>,
    canonical_name: &str,
) -> Vec<TokenStream> {
    accessors
        .iter()
        .map(|base| {
            let base_ident = syn::Ident::new(base, proc_macro2::Span::call_site());
            // Synthesized rotary accessors live off of
            // `emit_weights_struct::rotary_cos_sin_methods` and return
            // `GpuTensor` by value.
            if base == "rotary_cos_sin" || base == "rotary_local_cos_sin" {
                return quote! {
                    wm.#base_ident(layer).as_ptr::<u16>()
                };
            }
            let ty = match accessor_type_by_base.get(base) {
                Some(t) => t,
                None => {
                    let err = format!(
                        "ferrite-forward mega: canonical `{canonical_name}` accessor \
                         `{base}` not found in collected Weights accessor map — \
                         cannot synthesize pointer-extraction path"
                    );
                    return quote! { compile_error!(#err) };
                }
            };
            // Trim whitespace the tokenstream stringifier adds around
            // `::` so the type tail matches cleanly.
            let normalized = ty.replace(' ', "");
            if normalized.ends_with("LinearLayer") {
                quote! { wm.#base_ident(layer).dense_weight().as_ptr::<u16>() }
            } else if normalized.ends_with("Embedding") {
                quote! { wm.#base_ident(layer).weight.as_ptr::<u16>() }
            } else if normalized.ends_with("RmsNorm") {
                quote! { wm.#base_ident(layer).weight.as_ptr::<u16>() }
            } else {
                let err = format!(
                    "ferrite-forward mega: canonical `{canonical_name}` accessor \
                     `{base}` has unsupported Rust return type `{ty}` — only \
                     `LinearLayer`, `Embedding`, `RmsNorm`, and the synthesized \
                     `rotary*_cos_sin` accessors are wired for bf16 mega \
                     pointer extraction. Add an arm in \
                     `build_mega_accessor_ptr_exprs` or mark the variant \
                     ineligible earlier in `canonical_mega_meta`."
                );
                quote! { compile_error!(#err) }
            }
        })
        .collect()
}

/// Build the per-base accessor type map consumed by
/// [`emit_mega_artifacts_inline`] — one entry per distinct accessor
/// stem with its Rust return type stringified (e.g.
/// `"q_proj" → "LinearLayer"`, `"embed_tokens" → "Embedding"`).
/// Layered bases collapse their per-layer `WeightAccessor` entries
/// into one map entry (every layer agrees on the `rust_type`;
/// `collect_accessors`'s conflict check guarantees it).
fn mega_accessor_type_map(
    accessors: &[crate::impl_lib::WeightAccessor],
) -> BTreeMap<String, String> {
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    for a in accessors {
        let full = a.name.to_string();
        let (base, _layer) = split_base_layer(&full);
        let ty_str = a.rust_type.to_string();
        out.entry(base).or_insert(ty_str);
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn emit_model(
    program: &Program,
    model: &ModelParams,
    fuf: &Fuf,
    sfufs: &WorkloadAssignments,
    loops: &WorkloadLoops,
    sfufs_decode: &WorkloadAssignments,
    loops_decode: &WorkloadLoops,
    lib: &ImplementationLibrary,
    manifest: &crate::weights_manifest::WeightsManifest,
    canonical_override: Option<&Ident>,
    tp_world_size: u8,
    emit_fingerprint: bool,
    target_profile: &crate::target::TargetProfile,
) -> TokenStream {
    // Vision encoders have no terminal `gemm(<tile>, lm_head)`; the
    // entire FUF is the backbone. The `BackboneLayout::Encoder` arm
    // (which already covers text-side encoders like ModernBERT)
    // handles this naturally because `backbone_layout` reports
    // Encoder when the terminal isn't `gemm(_, lm_head)`. No
    // separate vision flag needed.
    if let Some(canonical) = canonical_override {
        return emit_shim_model(
            program,
            fuf,
            sfufs,
            lib,
            model,
            manifest,
            canonical,
            tp_world_size,
            emit_fingerprint,
        );
    }
    let weights = emit_weights_struct(
        program,
        fuf,
        sfufs,
        lib,
        model,
        manifest,
        WeightsEmitMode::Canonical,
        tp_world_size,
        emit_fingerprint,
    );

    // Group workload points by SFUF signature (sorted subgraph → impl).
    // Buckets with identical impl picks produce byte-identical fn
    // bodies, so we lower the canonical ONCE and emit duplicates as
    // thin `#[inline(always)]` shims that delegate to the canonical
    // fn. Public API (every `forward_m_<M>[_sk_<SK>]` /
    // `forward_backbone_m_<M>[_sk_<SK>]` name a user might take a
    // fn-pointer to) is preserved. Dedup runs over `(num_tokens,
    // sk_bucket)` 2-D points so models with `sk_buckets` declared get
    // the same compile-time win.
    let bucket_points: Vec<crate::solver::WorkloadPoint> =
        sfufs.per_workload.keys().copied().collect();
    let mut sfuf_to_canonical: HashMap<Vec<(u32, u32)>, crate::solver::WorkloadPoint> =
        HashMap::new();
    let mut bucket_canonical: Vec<crate::solver::WorkloadPoint> =
        Vec::with_capacity(bucket_points.len());
    for wp in &bucket_points {
        let sfuf = &sfufs.per_workload[wp];
        let mut sig: Vec<(u32, u32)> = sfuf.impls.iter().map(|(sg, imp)| (sg.0, imp.0)).collect();
        sig.sort();
        let canonical = *sfuf_to_canonical.entry(sig).or_insert(*wp);
        bucket_canonical.push(canonical);
    }

    // Lower every canonical bucket once, backbone-shaped: skip the
    // terminal subgraph (the `gemm(<final_norm>, lm_head)` row) and
    // emit it separately as a tiny LM_HEAD slice. forward and
    // forward_backbone share the backbone slice; forward additionally
    // runs LM_HEAD; forward_backbone DtoD-copies the backbone-output
    // slot. No more pair of near-identical full slices per bucket.
    let mut arch_opcodes = ArchOpcodes::new();
    let mut canonical_lowered: BTreeMap<
        crate::solver::WorkloadPoint,
        (
            CanonicalLowered,
            /* num_slots */ u32,
            /* backbone_slot */ u32,
            /* terminal_slot */ u32,
        ),
    > = BTreeMap::new();
    // `last_node_id` must be the lm_head Gemm (tp=1) or the post-lm_head
    // AllGather (tp>1), not one of the post-Embed `MmEmbedSplice` nodes
    // that `tp_lowering::insert_mm_splices` appends. Those sit at fuf-
    // array-tail but semantically belong near the Embed; walking past
    // them with `last_non_splice_node` recovers the real terminal.
    let last_node_id = last_non_splice_node(fuf)
        .expect("non-empty FUF expected")
        .id;
    let layout = backbone_layout(fuf, program);
    // For encoder layouts (text-side encoders like ModernBERT AND
    // every `#[vision_forward]` body) the FUF's terminal IS the
    // backbone output; decoder layouts carry the lm_head Gemm's
    // hidden-state input as the backbone output.
    let backbone_out = match layout {
        BackboneLayout::Decoder { backbone_out } => backbone_out,
        BackboneLayout::Encoder => (last_node_id, 0),
    };
    for (i, wp) in bucket_points.iter().enumerate() {
        if bucket_canonical[i] != *wp {
            continue;
        }
        let sfuf = &sfufs.per_workload[wp];
        let loop_ir = loops
            .per_workload
            .get(wp)
            .expect("schedule populated every key");
        let bounds = bounds_for_wp(model, *wp, tp_world_size);
        // Decoder mode skips the terminal subgraph in the backbone
        // lowering and emits it as a separate LM_HEAD slice. Encoder
        // mode lowers the entire pipeline as the backbone (no split).
        let skip_subgraph = match layout {
            BackboneLayout::Decoder { .. } => Some(
                sfuf.subgraph_of(last_node_id)
                    .expect("terminal tile must be in a subgraph"),
            ),
            BackboneLayout::Encoder => None,
        };

        // Protect the backbone output (`take_owned` reads it). For
        // decoder we additionally protect the terminal slot so the
        // backbone's drop pass leaves it free for lm_head to write.
        let mut protected_bb: HashSet<(TileId, u8)> = HashSet::new();
        protected_bb.insert(backbone_out);
        if matches!(layout, BackboneLayout::Decoder { .. }) {
            protected_bb.insert((last_node_id, 0));
        }

        // Per-bucket colored slot map. Computed once and shared
        // between backbone lowering and the lm_head fan_out so they
        // agree on slot indices.
        let slots = crate::interpreter_codegen::colored_slot_map(
            fuf,
            sfuf,
            loop_ir,
            lib,
            None,
            &protected_bb,
        );
        let backbone_slot = slots.of(backbone_out.0, backbone_out.1);
        // In encoder mode `take_owned` of the backbone-output slot is
        // also the terminal — the same slot index plays both roles.
        let terminal_slot = match layout {
            BackboneLayout::Decoder { .. } => slots.of(last_node_id, 0),
            BackboneLayout::Encoder => backbone_slot,
        };
        let num_slots = slots.total();

        // Decoder: skip the terminal subgraph so it emits as a
        // separate lm_head slice. Encoder (text encoders OR vision
        // bodies): pass `None` so the whole FUF lowers as backbone.
        let lowered_bb = lower_bucket(
            fuf,
            sfuf,
            loop_ir,
            program,
            model,
            lib,
            &bounds,
            skip_subgraph,
            &protected_bb,
            &mut arch_opcodes,
            backbone_out,
            &slots,
        );

        // LM_HEAD — only emitted in decoder mode. One row, computed
        // by directly invoking the terminal subgraph's `fan_out`
        // against the same slot map. Encoder mode (text encoders OR
        // vision bodies) emits an empty slice — the whole pipeline
        // already ran in the backbone.
        let lowered_lm = match layout {
            BackboneLayout::Decoder { .. } => {
                let terminal_sg =
                    skip_subgraph.expect("decoder layout always has a terminal subgraph");
                let term_imp_id = sfuf
                    .impl_of(terminal_sg)
                    .expect("terminal subgraph has an Impl assignment");
                let term_imp = lib.get(term_imp_id);
                let term_claimed = sfuf.tiles_in_subgraph(terminal_sg);
                let term_match = crate::impl_lib::MatchInfo {
                    claimed_tiles: term_claimed.clone(),
                    boundary_inputs: crate::interpreter_codegen::collect_boundary_inputs(
                        fuf,
                        &term_claimed,
                    ),
                    boundary_outputs: term_claimed,
                };
                let term_emits = term_imp
                    .fan_out(&term_match, fuf, program, &bounds, &slots)
                    .expect("terminal subgraph's Impl must implement fan_out");
                let term_accs = term_imp.required_weights(&term_match.claimed_tiles, fuf, program);
                let term_slots = crate::interpreter_codegen::weight_accessors_to_slots(&term_accs);
                let term_weight_slots: Vec<Vec<WeightSlot>> =
                    term_emits.iter().map(|_| term_slots.clone()).collect();
                // Eval body lives in `ferrite_forward::Instruction::eval`
                // — `arch_opcodes` keeps the shape registration for
                // `emit_bucket_static_slice`'s shape-checking pass.
                arch_opcodes.register(term_imp.opcode_shape());
                crate::interpreter_codegen::LoweredBucket {
                    instances: term_emits,
                    weight_slots: term_weight_slots,
                    num_slots,
                    final_slot: terminal_slot,
                }
            }
            BackboneLayout::Encoder => crate::interpreter_codegen::LoweredBucket {
                instances: Vec::new(),
                weight_slots: Vec::new(),
                num_slots,
                final_slot: terminal_slot,
            },
        };

        canonical_lowered.insert(
            *wp,
            (
                CanonicalLowered {
                    backbone: lowered_bb,
                    lm_head: lowered_lm,
                },
                num_slots,
                backbone_slot,
                terminal_slot,
            ),
        );
    }

    // Layer-template detection: collapse the contiguous repeating
    // sub-sequence of the slice (the per-layer transformer body)
    // into one `Instruction::Loop(N, body_len)` row + one
    // iteration's body. Fused boundary effects (e.g., FusedAddRmsNorm
    // absorbing layer L's final add into layer L+1's first norm)
    // leave layer 0 / the last layer structurally distinct, so the
    // detection picks the largest CONTIGUOUS run that genuinely
    // repeats — middle layers — and keeps the boundary residues as
    // straight-line code in prelude/suffix.
    for (cl, _, _, _) in canonical_lowered.values_mut() {
        crate::interpreter_codegen::apply_loop_compression(
            &arch_opcodes,
            &mut cl.backbone,
            "layer",
        );
        crate::interpreter_codegen::apply_loop_compression(&arch_opcodes, &mut cl.lm_head, "layer");
    }

    // ── Decode-role canonical_lowered ────────────────────────────
    //
    // Mirrors the block above but uses `sfufs_decode`/`loops_decode`
    // (the role=Decode solve). Drives `MEGA_FORWARD_TABLE_DECODE` so
    // the mega path dispatches TK paged-decode kernels at M>=2 when
    // all q_lens==1. Same `arch_opcodes` and layout, different
    // impl assignments per workload point.
    let bucket_points_decode: Vec<crate::solver::WorkloadPoint> =
        sfufs_decode.per_workload.keys().copied().collect();
    let mut sfuf_to_canonical_decode: HashMap<Vec<(u32, u32)>, crate::solver::WorkloadPoint> =
        HashMap::new();
    let mut bucket_canonical_decode: Vec<crate::solver::WorkloadPoint> =
        Vec::with_capacity(bucket_points_decode.len());
    for wp in &bucket_points_decode {
        let sfuf = &sfufs_decode.per_workload[wp];
        let mut sig: Vec<(u32, u32)> = sfuf.impls.iter().map(|(sg, imp)| (sg.0, imp.0)).collect();
        sig.sort();
        let canonical = *sfuf_to_canonical_decode.entry(sig).or_insert(*wp);
        bucket_canonical_decode.push(canonical);
    }
    // Build a decode-canonical vector parallel to `bucket_points` (the
    // REGULAR solve's ordering) so MEGA_FORWARD_TABLE_DECODE[i] aligns
    // with FORWARD_TABLE[i]. `bucket_canonical_decode` above uses the
    // DECODE solve's HashMap iteration order (non-deterministic); using
    // it directly in the table would look up decode fns by REGULAR
    // canonicals, which differ at M>1 where decode picks all-TK while
    // regular picks non-TK attention.
    //
    // For each regular bucket point, find its decode-solve canonical by
    // recomputing the sig from `sfufs_decode`. This maps "what regular
    // solve bucket i covers" → "what decode canonical serves it".
    let bucket_decode_canonical_for_table: Vec<crate::solver::WorkloadPoint> =
        bucket_points.iter().map(|wp| {
            if let Some(sfuf_dec) = sfufs_decode.per_workload.get(wp) {
                let mut sig: Vec<(u32, u32)> = sfuf_dec.impls.iter()
                    .map(|(sg, imp)| (sg.0, imp.0))
                    .collect();
                sig.sort();
                sfuf_to_canonical_decode.get(&sig).copied().unwrap_or(*wp)
            } else {
                *wp
            }
        }).collect();

    let mut canonical_lowered_decode: BTreeMap<
        crate::solver::WorkloadPoint,
        (CanonicalLowered, u32, u32, u32),
    > = BTreeMap::new();
    for (i, wp) in bucket_points_decode.iter().enumerate() {
        if bucket_canonical_decode[i] != *wp {
            continue;
        }
        let sfuf = &sfufs_decode.per_workload[wp];
        let loop_ir = loops_decode
            .per_workload
            .get(wp)
            .expect("decode schedule populated every key");
        let bounds = bounds_for_wp(model, *wp, tp_world_size);
        let skip_subgraph = match layout {
            BackboneLayout::Decoder { .. } => Some(
                sfuf.subgraph_of(last_node_id)
                    .expect("terminal tile must be in a subgraph"),
            ),
            BackboneLayout::Encoder => None,
        };
        let mut protected_bb: HashSet<(TileId, u8)> = HashSet::new();
        protected_bb.insert(backbone_out);
        if matches!(layout, BackboneLayout::Decoder { .. }) {
            protected_bb.insert((last_node_id, 0));
        }
        let slots = crate::interpreter_codegen::colored_slot_map(
            fuf,
            sfuf,
            loop_ir,
            lib,
            None,
            &protected_bb,
        );
        let backbone_slot = slots.of(backbone_out.0, backbone_out.1);
        let terminal_slot = match layout {
            BackboneLayout::Decoder { .. } => slots.of(last_node_id, 0),
            BackboneLayout::Encoder => backbone_slot,
        };
        let num_slots = slots.total();
        let lowered_bb = lower_bucket(
            fuf,
            sfuf,
            loop_ir,
            program,
            model,
            lib,
            &bounds,
            skip_subgraph,
            &protected_bb,
            &mut arch_opcodes,
            backbone_out,
            &slots,
        );
        let lowered_lm = match layout {
            BackboneLayout::Decoder { .. } => {
                let terminal_sg =
                    skip_subgraph.expect("decoder layout always has a terminal subgraph");
                let term_imp_id = sfuf
                    .impl_of(terminal_sg)
                    .expect("terminal subgraph has an Impl assignment");
                let term_imp = lib.get(term_imp_id);
                let term_claimed = sfuf.tiles_in_subgraph(terminal_sg);
                let term_match = crate::impl_lib::MatchInfo {
                    claimed_tiles: term_claimed.clone(),
                    boundary_inputs: crate::interpreter_codegen::collect_boundary_inputs(
                        fuf,
                        &term_claimed,
                    ),
                    boundary_outputs: term_claimed,
                };
                let term_emits = term_imp
                    .fan_out(&term_match, fuf, program, &bounds, &slots)
                    .expect("terminal subgraph Impl must implement fan_out");
                arch_opcodes.register(term_imp.opcode_shape());
                crate::interpreter_codegen::LoweredBucket {
                    instances: term_emits,
                    num_slots,
                    final_slot: terminal_slot,
                }
            }
            BackboneLayout::Encoder => crate::interpreter_codegen::LoweredBucket {
                instances: Vec::new(),
                num_slots,
                final_slot: terminal_slot,
            },
        };
        canonical_lowered_decode.insert(
            *wp,
            (
                CanonicalLowered {
                    backbone: lowered_bb,
                    lm_head: lowered_lm,
                },
                num_slots,
                backbone_slot,
                terminal_slot,
            ),
        );
    }
    for (cl, _, _, _) in canonical_lowered_decode.values_mut() {
        crate::interpreter_codegen::apply_loop_compression(
            &arch_opcodes,
            &mut cl.backbone,
            "layer",
        );
        crate::interpreter_codegen::apply_loop_compression(
            &arch_opcodes,
            &mut cl.lm_head,
            "layer",
        );
    }

    // Device-interpreter megakernel codegen — writes per-variant
    // `.cu` files into the cudaforge cache (gated on
    // FERRITE_MEGA=1 so default builds don't emit). Files are
    // picked up by ferrite-cuda-builder's build.rs and
    // nvcc-compiled into libmegakernels.a.
    //
    // Per FERRITE_TK_PLAN.md: emits straight-line C++ over TK 2.0
    // primitives with four warp-role walker bodies. Same
    // instruction set as the host interpreter, just inlined into
    // one kernel.
    // Build mega from the decode-role canonical_lowered. The mega
    // kernels are decode-only by design (TK attention_partial does not
    // support prefill causal masking); the decode-role solve picks
    // TkAttentionViaCache / AttentionViaCache at M>=2 while the
    // role-agnostic (existing) FORWARD_TABLE still uses prefill impls
    // there. MEGA_FORWARD_TABLE_DECODE is indexed by the same bucket
    // idx as FORWARD_TABLE so `find_bucket_idx` works for both.
    let (
        mega_rust_decls,
        mega_forward_fn_by_canonical_decode,
        mega_ms_forward_fn_by_canonical,
        mega_persistent_decode_start_fn_by_canonical,
    ) = if std::env::var_os("FERRITE_MEGA").is_some() {
        let accessor_type_map =
            match collect_accessors(program, fuf, sfufs_decode, lib, model) {
                Ok(accs) => mega_accessor_type_map(&accs),
                Err(_) => BTreeMap::new(),
            };
        emit_mega_artifacts_inline(
            model,
            &canonical_lowered_decode,
            &arch_opcodes,
            &accessor_type_map,
            tp_world_size,
            target_profile,
        )
    } else {
        (
            TokenStream::new(),
            BTreeMap::<crate::solver::WorkloadPoint, Ident>::new(),
            BTreeMap::<crate::solver::WorkloadPoint, Ident>::new(),
            BTreeMap::<crate::solver::WorkloadPoint, Ident>::new(),
        )
    };
    // Keep backward-compat alias so code below that was using the old
    // name still compiles; it now refers to the decode mega map.
    let mega_forward_fn_by_canonical: &BTreeMap<crate::solver::WorkloadPoint, Ident> =
        &mega_forward_fn_by_canonical_decode;

    // Per-canonical CanonicalParams impl + Instruction type alias.
    // The alias keeps every static-slice row on a single line of
    // expanded source (without it prettyplease wraps
    // `::ferrite_forward::Instruction::<Weights>::Variant(…)` over
    // 3-4 lines per row).
    let canonical_params_impl = emit_canonical_params_impl(model, tp_world_size);
    let weight_accessors_impl = emit_weight_accessors_impl(&canonical_lowered);
    // Per-canonical: alias the generic `Instruction<Weights>` for
    // the slice element type AND glob-import the variant
    // constructors so each static-slice row reads `Embed(...)` /
    // `RmsNorm(...)` instead of
    // `::ferrite_forward::Instruction::<Weights>::Embed(...)`
    // (which prettyplease wraps over 3-4 lines per row).
    let instruction_alias = quote! {
        #[cfg(feature = "cuda")]
        #[allow(non_camel_case_types, dead_code)]
        type __I = ::ferrite_forward::Instruction;
        #[cfg(feature = "cuda")]
        use ::ferrite_forward::Instruction::*;
    };
    let shapes_by_name = arch_opcodes.shapes_by_name();

    // Static slices: BACKBONE_M_<wp> + LM_HEAD_M_<wp> per CANONICAL
    // bucket only. Non-canonical buckets share their canonical
    // sibling's slices via the FORWARD_TABLE entries below.
    let mut static_slices: Vec<TokenStream> = Vec::new();
    for (i, wp) in bucket_points.iter().enumerate() {
        if bucket_canonical[i] != *wp {
            continue;
        }
        let (lowered, _, _, _) = &canonical_lowered[wp];
        let backbone_static_ident = bucket_static_ident("BACKBONE_M", *wp);
        let lm_head_static_ident = bucket_static_ident("LM_HEAD_M", *wp);
        static_slices.push(emit_bucket_static_slice(
            &backbone_static_ident,
            &shapes_by_name,
            &lowered.backbone.instances,
        ));
        static_slices.push(emit_bucket_static_slice(
            &lm_head_static_ident,
            &shapes_by_name,
            &lowered.lm_head.instances,
        ));
    }

    // `sk_axis_active`: true when the model declared `sk_buckets`;
    // otherwise every wp has sk_bucket==0 and the table emits
    // `[0, u64::MAX)` for sk on every entry.
    let sk_axis_active = sfufs.per_workload.keys().any(|wp| wp.sk_bucket != 0);
    let num_tokens_points: Vec<u64> = sfufs.num_tokens_points();
    let mut sk_by_m: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for wp in sfufs.per_workload.keys() {
        sk_by_m.entry(wp.num_tokens).or_default().push(wp.sk_bucket);
    }
    for v in sk_by_m.values_mut() {
        v.sort();
        v.dedup();
    }

    // (m_min, m_max_excl, sk_min, sk_max_excl) for each workload.
    // M=1 is special-cased: the solver may pick M=1-only kernels
    // (cutlass_gemv) that fail at M>1, so the next bucket starts at
    // M=2 even if the configured points list `[1, 8, ...]`.
    let m_max_excl_for = |m_idx: usize| -> u64 {
        if m_idx + 1 == num_tokens_points.len() {
            u64::MAX
        } else {
            num_tokens_points[m_idx + 1]
        }
    };
    let m_min_for = |m_idx: usize, m: u64| -> u64 {
        if m_idx > 0 && num_tokens_points[0] == 1 && num_tokens_points[m_idx - 1] == 1 {
            2
        } else {
            m
        }
    };
    let m_idx_of: HashMap<u64, usize> = num_tokens_points
        .iter()
        .enumerate()
        .map(|(i, &m)| (m, i))
        .collect();
    // Bucket-id assignment for the per-arch `WeightAccessors` impl:
    // canonical_lowered.iter() ordering pairs each canonical with
    // bucket ids `(2*ci, 2*ci+1)` for backbone and lm_head. This
    // mapping must match the iteration order in
    // `emit_weight_accessors_impl` so the match-arm keys line up.
    let canonical_to_bucket_id: HashMap<_, u32> = canonical_lowered
        .iter()
        .enumerate()
        .map(|(ci, (wp, _))| (*wp, (ci as u32) * 2))
        .collect();

    let mut bucket_table_entries: Vec<TokenStream> = Vec::new();
    for (i, wp) in bucket_points.iter().enumerate() {
        let canonical = bucket_canonical[i];
        let bb_static = bucket_static_ident("BACKBONE_M", canonical);
        let lm_static = bucket_static_ident("LM_HEAD_M", canonical);
        let bb_bucket_id = *canonical_to_bucket_id
            .get(&canonical)
            .expect("canonical_to_bucket_id missing entry for canonical workload point");
        let bb_bucket_lit = proc_macro2::Literal::u32_unsuffixed(bb_bucket_id);
        let lm_bucket_lit = proc_macro2::Literal::u32_unsuffixed(bb_bucket_id + 1);
        // Per-bucket slot metadata. The colored slot map is built
        // per workload point (the solver may pick Impls that need
        // different intermediate-tile counts per bucket — e.g.
        // CutlassGemmAdd fuses the residual into the GEMM at
        // prefill, freeing a slot vs the decode-bucket's separate
        // Add). The non-canonical buckets share their canonical's
        // slot metadata since they share its static slices.
        let (_, num_slots_b, backbone_slot_b, terminal_slot_b) = &canonical_lowered[&canonical];
        let num_slots_lit = proc_macro2::Literal::u32_unsuffixed(*num_slots_b);
        let backbone_slot_lit = proc_macro2::Literal::u32_unsuffixed(*backbone_slot_b);
        let terminal_slot_lit = proc_macro2::Literal::u32_unsuffixed(*terminal_slot_b);
        let m_idx = m_idx_of[&wp.num_tokens];
        let m_min = if wp.num_tokens == 1 {
            1
        } else {
            m_min_for(m_idx, wp.num_tokens)
        };
        let m_max_excl = if wp.num_tokens == 1 {
            2
        } else {
            m_max_excl_for(m_idx)
        };
        let (sk_min, sk_max_excl) = if sk_axis_active {
            let sk_buckets = &sk_by_m[&wp.num_tokens];
            let j = sk_buckets
                .iter()
                .position(|&sk| sk == wp.sk_bucket)
                .unwrap();
            // The first sk bucket per m group must accept sk < its
            // own configured value — old codegen routed this via a
            // `_ =>` fallback arm onto the smallest sk fn. So emit
            // sk_min=0 for j==0 instead of wp.sk_bucket; without
            // this, find_bucket misses on small max_seqlen_k (the
            // prefill case for short prompts) and the fallback to
            // table[0] silently picks an m=1 row.
            let sk_min = if j == 0 { 0 } else { wp.sk_bucket };
            let sk_max_excl = if j + 1 == sk_buckets.len() {
                u64::MAX
            } else {
                sk_buckets[j + 1]
            };
            (sk_min, sk_max_excl)
        } else {
            (0u64, u64::MAX)
        };
        let m_min_lit = proc_macro2::Literal::u64_unsuffixed(m_min);
        let m_max_lit = if m_max_excl == u64::MAX {
            quote! { u64::MAX }
        } else {
            let v = proc_macro2::Literal::u64_unsuffixed(m_max_excl);
            quote! { #v }
        };
        let sk_min_lit = proc_macro2::Literal::u64_unsuffixed(sk_min);
        let sk_max_lit = if sk_max_excl == u64::MAX {
            quote! { u64::MAX }
        } else {
            let v = proc_macro2::Literal::u64_unsuffixed(sk_max_excl);
            quote! { #v }
        };
        bucket_table_entries.push(quote! {
            __B(
                #m_min_lit, #m_max_lit, #sk_min_lit, #sk_max_lit,
                #bb_static, #lm_static,
                #num_slots_lit, #backbone_slot_lit, #terminal_slot_lit,
                #bb_bucket_lit, #lm_bucket_lit,
            ),
        });
    }

    // FORWARD_TABLE — one row per bucket. `__B` aliases the
    // `BucketEntry` tuple-struct constructor so each row stays on a
    // single line in expanded source.
    let forward_table = quote! {
        #[cfg(feature = "cuda")]
        static FORWARD_TABLE: &[::ferrite_forward::BucketEntry<__I>] = {
            use ::ferrite_forward::BucketEntry as __B;
            &[
                #(#bucket_table_entries)*
            ]
        };
    };

    // Encoder layouts have no lm_head split — `forward_backbone` is
    // semantically identical to `forward`. Decoder layouts return the
    // pre-lm_head activation via a DtoD memcpy of the protected
    // backbone slot. Both shapes are dispatched via the same
    // `FORWARD_TABLE` row.
    let forward_backbone_fn = match layout {
        BackboneLayout::Decoder { .. } => quote! {
            /// Backbone-only dispatch (no lm_head). Returns a fresh
            /// OwnedTensor (memcpy of the backbone tile).
            #[cfg(feature = "cuda")]
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn forward_backbone(
                wm: &Weights,
                ctx: &::ferrite_forward::ForwardCtx,
                device: &mut ::ferrite_cuda_core::device::GpuDevice,
                num_tokens: u64,
            ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
                let e = ::ferrite_forward::find_bucket(
                    FORWARD_TABLE, num_tokens, ctx.max_seqlen_k as u64,
                );
                unsafe {
                    ::ferrite_forward::run_backbone(e.4, e.9, wm, ctx, device, e.6, e.7)
                }
            }
        },
        BackboneLayout::Encoder => quote! {
            /// Backbone-only dispatch — for encoder architectures the
            /// whole pipeline IS the backbone, so this delegates to
            /// `forward` (no lm_head, no DtoD memcpy).
            #[cfg(feature = "cuda")]
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn forward_backbone(
                wm: &Weights,
                ctx: &::ferrite_forward::ForwardCtx,
                device: &mut ::ferrite_cuda_core::device::GpuDevice,
                num_tokens: u64,
            ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
                unsafe { forward(wm, ctx, device, num_tokens) }
            }
        },
    };

    // Phase 3f-2l-ii: MEGA_FORWARD_TABLE — parallel to FORWARD_TABLE,
    // indexed by the same bucket row position. Each row is
    // `Option<unsafe fn(&Weights, &ForwardCtx, &mut GpuDevice)
    //                                   -> OwnedTensor>`:
    // `Some(forward_mega_<canonical>)` when the bucket's canonical
    // has a linkable mega launch symbol AND its accessor set mapped
    // cleanly to pointer-extraction paths, `None` otherwise (the
    // canonical fell back to `#error` on at least one op OR one of
    // its accessors has no registered bf16-ptr path). Emitted only
    // when the build-time `FERRITE_MEGA=1` gate fired — otherwise
    // the table is empty and `forward()` never consults it.
    //
    // Runtime dispatch: `forward()` looks up the bucket index once
    // and consults both tables at that index. Host-interpreter
    // fallback lives on every bucket (`FORWARD_TABLE` is always
    // populated), so a `None` row here silently routes through the
    // host path.
    let mega_forward_fn_ty = quote! {
        unsafe fn(
            &Weights,
            &::ferrite_forward::ForwardCtx,
            &mut ::ferrite_cuda_core::device::GpuDevice,
        ) -> ::ferrite_cuda_core::alloc::OwnedTensor
    };
    // MEGA_FORWARD_TABLE_DECODE — parallel to FORWARD_TABLE, indexed
    // by the same bucket row position, built from the decode-role
    // solve. Mega kernels are decode-only (TK attention_partial has
    // no prefill causal mask); runtime dispatch gates on
    // `ctx.max_seqlen_q == 1` before consulting this table.
    // Each row stores (expected_num_tokens, fn) so the dispatch can
    // guard on exact num_tokens match. A bucket covers a RANGE of M
    // values but a MEGA kernel is compiled for exactly one NUM_TOKENS.
    // Without the guard, num_tokens=2 would call the M=8 kernel and
    // access memory for 6 phantom sequences → CUDA_ERROR_ILLEGAL_ADDRESS.
    // DEBUG: print mega_forward_fn_by_canonical keys
    let mega_table_emission: TokenStream = if mega_forward_fn_by_canonical.is_empty() {
        quote! {
            #[cfg(feature = "cuda")]
            #[allow(dead_code)]
            static MEGA_FORWARD_TABLE_DECODE: &[
                ::core::option::Option<(u64, #mega_forward_fn_ty)>
            ] = &[];
        }
    } else {
        // Build a fallback map: for each num_tokens (M), pick ANY canonical
        // we successfully emitted for that M. Used when a bucket's primary
        // canonical points to a workload-point that didn't produce a mega
        // fn (e.g. its impl signature collided with another canonical's
        // and lost the canonical race, or its variant was rejected by
        // `canonical_mega_meta`).
        // Each table entry stores (compiled_M, fn_ident). Dispatch site
        // guards on num_tokens == compiled_M, so prefill at num_tokens=8
        // dispatches to the m=8 kernel and decode at num_tokens=1 dispatches
        // to m=1. Without this fallback, MEGA_FORWARD_TABLE_DECODE[bucket_idx]
        // = None for any bucket whose canonical didn't emit, and mega never
        // dispatches for that bucket.
        let by_m: BTreeMap<u64, &Ident> = mega_forward_fn_by_canonical
            .iter()
            .map(|(wp, ident)| (wp.num_tokens, ident))
            .collect();
        let rows: Vec<TokenStream> = bucket_decode_canonical_for_table
            .iter()
            .map(|decode_canonical| {
                // Direct hit on the canonical?
                if let Some(ident) = mega_forward_fn_by_canonical.get(decode_canonical) {
                    let expected_m: u64 = decode_canonical.num_tokens as u64;
                    return quote! { ::core::option::Option::Some((#expected_m, #ident)), };
                }
                // Same-M fallback: prefer a kernel emitted for the bucket's
                // canonical num_tokens. Lets buckets sharing the canonical-
                // assignment race still resolve to a working fn at the same M.
                if let Some(ident) = by_m.get(&decode_canonical.num_tokens) {
                    let expected_m: u64 = decode_canonical.num_tokens as u64;
                    return quote! { ::core::option::Option::Some((#expected_m, #ident)), };
                }
                // No same-M fn emitted. Pick the SMALLEST emitted M ≥ this
                // bucket's canonical num_tokens (closest-larger). The bucket
                // covers a range; the dispatch site's `num_tokens ==
                // expected_m` guard will only fire when actual num_tokens
                // matches the chosen M, so picking a larger M for the table
                // entry is safe — it just means the exact-match check happens
                // later. Without this, prefill buckets with no m=2/m=4 kernel
                // would go to None and prefill mega never fires.
                let pick = by_m
                    .iter()
                    .find(|(m, _)| **m >= decode_canonical.num_tokens)
                    .map(|(m, ident)| (*m, *ident));
                match pick {
                    Some((m, ident)) => {
                        quote! { ::core::option::Option::Some((#m, #ident)), }
                    }
                    None => quote! { ::core::option::Option::None, },
                }
            })
            .collect();
        quote! {
            #[cfg(feature = "cuda")]
            #[allow(dead_code)]
            static MEGA_FORWARD_TABLE_DECODE: &[
                ::core::option::Option<(u64, #mega_forward_fn_ty)>
            ] = &[
                #(#rows)*
            ];
        }
    };

    // MEGA_FORWARD_TABLE_MULTI_STEP — parallel to MEGA_FORWARD_TABLE_DECODE
    // but entries point at `forward_mega_ms_<canonical>` fns that call
    // `launch_multi_step` and return a zero-element placeholder tensor.
    // Only M=1 decode canonicals that successfully emitted a `_ms` CU
    // (logits_slot found + TkFusedAddRmsNormGemm in lm_head) get Some(_).
    // Runtime dispatch: `ctx.multi_step.is_some()` gates on this table
    // BEFORE the single-step table so multi-step wins when staged.
    let mega_ms_table_emission: TokenStream = if mega_ms_forward_fn_by_canonical.is_empty() {
        quote! {
            #[cfg(feature = "cuda")]
            #[allow(dead_code)]
            static MEGA_FORWARD_TABLE_MULTI_STEP: &[
                ::core::option::Option<(u64, #mega_forward_fn_ty)>
            ] = &[];
        }
    } else {
        let rows: Vec<TokenStream> = bucket_decode_canonical_for_table
            .iter()
            .map(|decode_canonical| {
                match mega_ms_forward_fn_by_canonical.get(decode_canonical) {
                    Some(ident) => {
                        let expected_m: u64 = decode_canonical.num_tokens as u64;
                        quote! { ::core::option::Option::Some((#expected_m, #ident)), }
                    }
                    None => quote! { ::core::option::Option::None, },
                }
            })
            .collect();
        quote! {
            #[cfg(feature = "cuda")]
            #[allow(dead_code)]
            static MEGA_FORWARD_TABLE_MULTI_STEP: &[
                ::core::option::Option<(u64, #mega_forward_fn_ty)>
            ] = &[
                #(#rows)*
            ];
        }
    };

    // MEGA_PERSISTENT_DECODE_TABLE — parallel to MEGA_FORWARD_TABLE_MULTI_STEP
    // but for the persistent-decode path. Entries are
    // `start_persistent_decode_*` fns that launch the cooperative persistent
    // kernel and return `PersistentDecodeResources`.
    // fn type differs from mega_forward_fn_ty: extra `protocol` arg, different return.
    let mega_persistent_decode_fn_ty = quote! {
        unsafe fn(
            &Weights,
            &::ferrite_forward::ForwardCtx,
            &mut ::ferrite_cuda_core::device::GpuDevice,
            *mut ::std::ffi::c_void,
        ) -> ::core::result::Result<::ferrite_forward::PersistentDecodeResources, i32>
    };
    let mega_persistent_decode_table_emission: TokenStream = if mega_persistent_decode_start_fn_by_canonical.is_empty() {
        quote! {
            #[cfg(feature = "cuda")]
            #[allow(dead_code)]
            static MEGA_PERSISTENT_DECODE_TABLE: &[
                ::core::option::Option<(u64, #mega_persistent_decode_fn_ty)>
            ] = &[];
        }
    } else {
        let rows: Vec<TokenStream> = bucket_decode_canonical_for_table
            .iter()
            .map(|decode_canonical| {
                match mega_persistent_decode_start_fn_by_canonical.get(decode_canonical) {
                    Some(ident) => {
                        let expected_m: u64 = decode_canonical.num_tokens as u64;
                        quote! { ::core::option::Option::Some((#expected_m, #ident)), }
                    }
                    None => quote! { ::core::option::Option::None, },
                }
            })
            .collect();
        quote! {
            #[cfg(feature = "cuda")]
            #[allow(dead_code)]
            static MEGA_PERSISTENT_DECODE_TABLE: &[
                ::core::option::Option<(u64, #mega_persistent_decode_fn_ty)>
            ] = &[
                #(#rows)*
            ];
        }
    };

    quote! {
        #weights

        #canonical_params_impl

        #weight_accessors_impl

        #instruction_alias

        #(#static_slices)*

        // Mega (ferrite-TK) per-variant extern decls + LAUNCH_FN_<VARIANT>
        // constants + per-canonical `forward_mega_<canonical>` fns —
        // one pair per canonical bucket whose `.cu` codegen'd without
        // bailing to `#error` AND whose accessor set resolves to
        // known bf16 pointer-extraction paths. Empty token stream when
        // `FERRITE_MEGA=1` isn't set at macro-expansion time.
        #mega_rust_decls

        #forward_table

        // Parallel to FORWARD_TABLE: per-bucket Option<forward_mega_*>.
        // Populated only under build-time `FERRITE_MEGA=1`; otherwise
        // an empty slice. Runtime dispatch inspects this alongside
        // `::ferrite_forward::mega_enabled()` to choose the mega path.
        #mega_table_emission

        // Parallel to MEGA_FORWARD_TABLE_DECODE: M=1-only multi-step
        // cooperative kernels. Entries point at `forward_mega_ms_*` fns
        // that call `launch_multi_step` and return a zero-element tensor.
        // Only populated when FERRITE_MEGA=1 at build time AND the canonical
        // has a logits_slot + TkFusedAddRmsNormGemm lm_head.
        #mega_ms_table_emission

        // Parallel to MEGA_FORWARD_TABLE_DECODE: M=1-only persistent-decode
        // start fns. Entries point at `start_persistent_decode_*` fns that
        // launch the persistent cooperative kernel and return PersistentDecodeResources.
        // Only populated when FERRITE_MEGA=1 at build time AND the canonical
        // has a logits_slot + TkFusedAddRmsNormGemm lm_head.
        #mega_persistent_decode_table_emission

        /// Dispatch on (num_tokens, sk_bucket) → bucket entry, then
        /// run the universal interpreter (or the mega kernel when
        /// `FERRITE_MEGA=1` is set at runtime AND the bucket has a
        /// compiled mega forward fn).
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward(
            wm: &Weights,
            ctx: &::ferrite_forward::ForwardCtx,
            device: &mut ::ferrite_cuda_core::device::GpuDevice,
            num_tokens: u64,
        ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
            let idx = ::ferrite_forward::find_bucket_idx(
                FORWARD_TABLE, num_tokens, ctx.max_seqlen_k as u64,
            );
            // Persistent-decode dispatch: M=1, persistent_decode_session staged by the executor.
            // On first call (session.resources is None): launch persistent kernel.
            // On subsequent calls: submit step via protocol buffer and poll output.
            // Takes priority over multi-step and single-step mega.
            // persistent_decode_session is a raw *mut so we can mutate through &ForwardCtx.
            if ::ferrite_forward::mega_enabled()
                && !ctx.persistent_decode_session.is_null()
                && ctx.persistent_decode_step.is_some()
                && ctx.max_seqlen_q == 1
            {
                if let ::core::option::Option::Some((expected_m, pd_fn)) =
                    MEGA_PERSISTENT_DECODE_TABLE.get(idx).copied().flatten()
                {
                    if num_tokens == expected_m {
                        // SAFETY: caller (CudaWorker) owns the PersistentDecodeSession and ensures it
                        // lives for the duration of this call. No aliasing — only one
                        // forward() call runs at a time on the worker thread.
                        let session = unsafe { &mut *ctx.persistent_decode_session };
                        let step = ctx.persistent_decode_step.unwrap();
                        // Start the persistent kernel on the first call.
                        #[cfg(feature = "cuda")]
                        if session.resources.is_none() {
                            if ::ferrite_forward::trace_enabled() {
                                ::std::eprintln!(
                                    "ferrite-forward pd: starting kernel bucket_idx={} num_tokens={} sk={}",
                                    idx, num_tokens, ctx.max_seqlen_k,
                                );
                            }
                            match unsafe { pd_fn(wm, ctx, device, session.protocol_ptr()) } {
                                ::core::result::Result::Ok(resources) => {
                                    session.resources = ::core::option::Option::Some(resources);
                                }
                                ::core::result::Result::Err(rc) => {
                                    ::std::eprintln!(
                                        "ferrite-forward pd: start failed rc={rc}, falling through"
                                    );
                                    // Fall through to single-step mega / host interp.
                                }
                            }
                        }
                        // Execute the step via the pinned protocol buffer.
                        #[cfg(feature = "cuda")]
                        if session.resources.is_some() {
                            if ::ferrite_forward::trace_enabled() {
                                ::std::eprintln!(
                                    "ferrite-forward pd: step pos={} sl={}",
                                    step.position, step.seq_len,
                                );
                            }
                            unsafe {
                                session.write_step_input(
                                    step.input_id,
                                    step.position,
                                    step.seq_len,
                                    step.slot_mapping,
                                    step.block_table_stride,
                                    &step.block_ids[..step.num_block_ids],
                                );
                                session.signal_cpu_step();
                                session.poll_output_token();
                            }
                            // Return zero-element placeholder; executor reads
                            // session.last_output_token directly.
                            return device.caching.alloc_tensor(
                                &[0usize],
                                ::ferrite_cuda_core::dtype::DType::BF16,
                            );
                        }
                    }
                }
            }
            // Multi-step cooperative dispatch: M=1, multi_step ctx staged by
            // the executor. Takes priority over single-step mega so N decode
            // steps fuse into one cooperative kernel launch when possible.
            if ::ferrite_forward::mega_enabled()
                && ctx.multi_step.is_some()
                && ctx.max_seqlen_q == 1
            {
                if let ::core::option::Option::Some((expected_m, ms_fn)) =
                    MEGA_FORWARD_TABLE_MULTI_STEP.get(idx).copied().flatten()
                {
                    if num_tokens == expected_m {
                        if ::ferrite_forward::trace_enabled() {
                            ::std::eprintln!(
                                "ferrite-forward mega-ms: dispatch bucket_idx={} num_tokens={} sk={}",
                                idx,
                                num_tokens,
                                ctx.max_seqlen_k,
                            );
                        }
                        return unsafe { ms_fn(wm, ctx, device) };
                    }
                }
            }
            // Single-step mega dispatch: decode-only (all q_lens == 1). The
            // MEGA_FORWARD_TABLE_DECODE was built from the decode-role
            // solve (TK paged-cache attention at M>=2, etc.). Prefill
            // or mixed batches (max_seqlen_q > 1) fall through to the
            // host interpreter path — mega kernels have no
            // new-token causal mask for prefill.
            // Mega: fires for both prefill AND decode (no max_seqlen_q gate).
            // Guard on num_tokens == expected_m so we only call kernels
            // compiled for exactly this token count.
            if ::ferrite_forward::mega_enabled() {
                if let ::core::option::Option::Some((expected_m, mega_fn)) =
                    MEGA_FORWARD_TABLE_DECODE.get(idx).copied().flatten()
                {
                    if num_tokens == expected_m {
                        if ::ferrite_forward::trace_enabled() {
                            ::std::eprintln!(
                                "ferrite-forward mega: dispatch bucket_idx={} num_tokens={} sk={} max_seqlen_q={}",
                                idx,
                                num_tokens,
                                ctx.max_seqlen_k,
                                ctx.max_seqlen_q,
                            );
                        }
                        return unsafe { mega_fn(wm, ctx, device) };
                    }
                }
            }
            let e = &FORWARD_TABLE[idx];
            unsafe {
                ::ferrite_forward::run(e.4, e.9, e.5, e.10, wm, ctx, device, e.6, e.8)
            }
        }

        #forward_backbone_fn

        /// Start a persistent-decode kernel for M=1 greedy decode.
        /// Looks up `MEGA_PERSISTENT_DECODE_TABLE[bucket_idx]`; returns
        /// `None` when no persistent-decode kernel was compiled for this
        /// (num_tokens, sk_bucket) combination.
        #[cfg(feature = "cuda")]
        #[allow(dead_code)]
        pub unsafe fn start_persistent_decode(
            wm: &Weights,
            ctx: &::ferrite_forward::ForwardCtx,
            device: &mut ::ferrite_cuda_core::device::GpuDevice,
            protocol: *mut ::std::ffi::c_void,
            num_tokens: u64,
            max_seqlen_k: u64,
        ) -> ::core::option::Option<
            ::core::result::Result<::ferrite_forward::PersistentDecodeResources, i32>,
        > {
            let idx = ::ferrite_forward::find_bucket_idx(
                FORWARD_TABLE, num_tokens, max_seqlen_k,
            );
            if let ::core::option::Option::Some((expected_m, pd_fn)) =
                MEGA_PERSISTENT_DECODE_TABLE.get(idx).copied().flatten()
            {
                if num_tokens == expected_m {
                    return ::core::option::Option::Some(unsafe {
                        pd_fn(wm, ctx, device, protocol)
                    });
                }
            }
            ::core::option::Option::None
        }

        /// Walk `FORWARD_TABLE` and return one [`BucketDump`] per
        /// row, with backbone + lm_head normalized for non-generic
        /// inspection (no `&Weights`, no GPU). Used by
        /// `vllm ferrite info` via the inventory registry.
        #[cfg(feature = "cuda")]
        pub fn dump() -> ::std::vec::Vec<::ferrite_forward::BucketDump> {
            FORWARD_TABLE
                .iter()
                .map(|e| ::ferrite_forward::BucketDump {
                    m_min: e.0,
                    m_max_excl: e.1,
                    sk_min: e.2,
                    sk_max_excl: e.3,
                    backbone: ::ferrite_forward::normalize_slice(e.4),
                    lm_head: ::ferrite_forward::normalize_slice(e.5),
                })
                .collect()
        }
    }
}

/// Unused — consumers used to reach this by name from tests.
/// Retained as a no-op type anchor while the trait-based shim is
/// being deleted from downstream call sites.
#[allow(dead_code)]
fn _unused(_: OpKind) {}

/// Emit a shim variant module: one whose forward-fn bodies are
/// byte-identical to a canonical sibling's. Instead of re-emitting
/// the bodies (which rustc would LLVM-optimize independently per
/// variant, compounding release-build time multiplicatively), we:
///
/// - `pub type Weights = super::<canonical>::Weights;` — share the
///   same struct layout; variants in the same equivalence class
///   end up wrapping the same concrete type at the arch-dispatcher
///   level, which is fine for `enum Outer { V1(T), V2(T) }`.
/// - `pub fn fingerprint_matches` — VARIANT-specific. The
///   tensor-suffix gate (e.g. dense `.weight` vs AWQ `.qweight` vs
///   CT `.weight_packed` vs BNB4 `.weight.absmax`) differs per
///   variant, so each ships its own sniff.
/// - `pub fn load` — VARIANT-specific. The loader calls
///   `MarlinLinear::load_awq` vs `load_gptq` vs
///   `Bnb4bitLinear::load` etc. depending on the variant's
///   `quantization_config`, but constructs the same canonical
///   `Weights` struct at the end (same accessor shapes across the
///   equivalence class).
/// - `pub use super::<canonical>::{forward, forward_backbone,
///   forward_m_<N>, forward_backbone_m_<N>, ...};` — no fn-body
///   re-emit. rustc doesn't re-monomorphize `pub use` paths, so
///   the canonical's release-optimized forward is called directly
///   through this module without additional LLVM work.
#[allow(clippy::too_many_arguments)]
fn emit_shim_model(
    program: &Program,
    fuf: &Fuf,
    sfufs: &WorkloadAssignments,
    lib: &ImplementationLibrary,
    model: &ModelParams,
    manifest: &crate::weights_manifest::WeightsManifest,
    canonical: &Ident,
    tp_world_size: u8,
    emit_fingerprint: bool,
) -> TokenStream {
    let weights = emit_weights_struct(
        program,
        fuf,
        sfufs,
        lib,
        model,
        manifest,
        WeightsEmitMode::Shim { canonical },
        tp_world_size,
        emit_fingerprint,
    );

    // Per-bucket fn surfaces are gone — dispatch lives on the
    // canonical's `FORWARD_TABLE` + `find_bucket`. Re-export the
    // arch-level dispatchers only.
    let _ = sfufs;
    quote! {
        #weights

        #[cfg(feature = "cuda")]
        pub use super::#canonical::{dump, forward, forward_backbone};
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impl_lib::WeightAccessor;
    use quote::format_ident;

    /// Helper: build a minimal `ModelParams` from raw bound values
    /// for canonical-params shard tests. Only the fields
    /// `emit_canonical_params_impl` actually reads are populated;
    /// everything else gets a sensible default.
    fn shard_test_model(
        num_attention_heads: u64,
        num_key_value_heads: u64,
        head_dim: u64,
        intermediate_size: u64,
    ) -> crate::config::ModelParams {
        use std::collections::BTreeMap;
        use std::path::PathBuf;

        let mut bounds: BTreeMap<String, u64> = BTreeMap::new();
        bounds.insert("num_attention_heads".into(), num_attention_heads);
        bounds.insert("num_key_value_heads".into(), num_key_value_heads);
        bounds.insert("head_dim".into(), head_dim);
        bounds.insert("intermediate_size".into(), intermediate_size);
        crate::config::ModelParams {
            name: "shard_test".into(),
            source_stem: "shard-test".into(),
            source_path: PathBuf::new(),
            bounds,
            scalars: BTreeMap::new(),
            quantization: None,
            tie_word_embeddings: false,
            architectures: Vec::new(),
            extra_tracked_paths: Vec::new(),
            rope_scaling: None,
            rope_scaling_hash: None,
            mrope_section: None,
            vision_layout: None,
            vision_d_model_fingerprint: None,
            vision_patch_embed_flatten: None,
            vision_class_embedding_fold: None,
            decoder_safetensors_prefix: None,
            torch_dtype: None,
        }
    }

    /// `emit_canonical_params_impl` at `tp_world_size = 1` is
    /// identity — bounds flow through unchanged. Pinning this so
    /// task #7's outer-loop fanout cannot accidentally regress the
    /// single-rank build (which is every per-arch-crate build under
    /// `--features cuda` until activation lands).
    #[test]
    fn canonical_params_at_tp_eq_1_is_identity() {
        // Llama-2-7B-ish numbers: 32 q heads, 32 kv heads,
        // head_dim=128, intermediate=11008.
        let m = shard_test_model(32, 32, 128, 11008);
        let ts = emit_canonical_params_impl(&m, 1).to_string();
        // Q size = 32 * 128 = 4096; KV size = 32 * 128 = 4096.
        assert!(
            ts.contains("NUM_Q_HEADS : u32 = 32"),
            "tp=1 must keep NUM_Q_HEADS = 32; got {ts}"
        );
        assert!(
            ts.contains("NUM_KV_HEADS : u32 = 32"),
            "tp=1 must keep NUM_KV_HEADS = 32; got {ts}"
        );
        assert!(
            ts.contains("INTERMEDIATE_SIZE : usize = 11008"),
            "tp=1 must keep INTERMEDIATE_SIZE = 11008; got {ts}"
        );
        assert!(
            ts.contains("Q_SIZE : usize = 4096"),
            "tp=1 must keep Q_SIZE = 4096; got {ts}"
        );
    }

    /// `emit_canonical_params_impl` at `tp_world_size = 2` shards
    /// every column-parallel dim by 2. The kernel-launch sizes that
    /// flow through `<W as CanonicalParams>::…` constants in the
    /// emitted `Instruction::eval` body MUST be the per-rank values
    /// — nothing else in the compiled body is tp-aware.
    #[test]
    fn canonical_params_at_tp_eq_2_shards_column_parallel_dims() {
        let m = shard_test_model(32, 32, 128, 11008);
        let ts = emit_canonical_params_impl(&m, 2).to_string();
        assert!(
            ts.contains("NUM_Q_HEADS : u32 = 16"),
            "tp=2 must shard NUM_Q_HEADS to 16; got {ts}"
        );
        assert!(
            ts.contains("NUM_KV_HEADS : u32 = 16"),
            "tp=2 must shard NUM_KV_HEADS to 16; got {ts}"
        );
        assert!(
            ts.contains("INTERMEDIATE_SIZE : usize = 5504"),
            "tp=2 must shard INTERMEDIATE_SIZE to 5504; got {ts}"
        );
        // Q_SIZE = (32/2) * 128 = 2048
        assert!(
            ts.contains("Q_SIZE : usize = 2048"),
            "tp=2 must compute Q_SIZE from sharded heads to 2048; got {ts}"
        );
        assert!(
            ts.contains("KV_SIZE : usize = 2048"),
            "tp=2 must compute KV_SIZE from sharded heads to 2048; got {ts}"
        );
        // HEAD_DIM is per-head and never sharded.
        assert!(
            ts.contains("HEAD_DIM : u32 = 128"),
            "tp=2 must keep HEAD_DIM = 128; got {ts}"
        );
    }

    /// `emit_canonical_params_impl` at `tp_world_size = 8` shards
    /// every column-parallel dim by 8. Pins the floor-divide
    /// behavior on a value that's the upper bound of the compile-
    /// time set — the compile() outer-loop fanout in task #7 stops
    /// at tp=8 by default, so this is the largest case that ever
    /// reaches emit.
    #[test]
    fn canonical_params_at_tp_eq_8_shards_column_parallel_dims() {
        // Llama-3-8B-ish numbers: 32 q heads, 8 kv heads,
        // head_dim=128, intermediate=14336.
        let m = shard_test_model(32, 8, 128, 14336);
        let ts = emit_canonical_params_impl(&m, 8).to_string();
        // 32 / 8 = 4
        assert!(
            ts.contains("NUM_Q_HEADS : u32 = 4"),
            "tp=8 must shard NUM_Q_HEADS to 4; got {ts}"
        );
        // 8 / 8 = 1
        assert!(
            ts.contains("NUM_KV_HEADS : u32 = 1"),
            "tp=8 must shard NUM_KV_HEADS to 1; got {ts}"
        );
        // 14336 / 8 = 1792
        assert!(
            ts.contains("INTERMEDIATE_SIZE : usize = 1792"),
            "tp=8 must shard INTERMEDIATE_SIZE to 1792; got {ts}"
        );
    }

    #[test]
    fn split_base_layer_recognizes_layered_and_unlayered_names() {
        // Layered: trailing _<digits> peels off as the layer index.
        assert_eq!(
            split_base_layer("input_layernorm_3"),
            ("input_layernorm".to_string(), Some(3))
        );
        assert_eq!(
            split_base_layer("self_attn_q_proj_31"),
            ("self_attn_q_proj".to_string(), Some(31))
        );
        // Non-layered: no trailing _<digits>.
        assert_eq!(
            split_base_layer("embed_tokens"),
            ("embed_tokens".to_string(), None)
        );
        assert_eq!(split_base_layer("lm_head"), ("lm_head".to_string(), None));
        // Trailing-non-digit suffix isn't a layer — the whole name
        // stays as the base.
        assert_eq!(
            split_base_layer("rotary_local"),
            ("rotary_local".to_string(), None)
        );
        // Empty trailing chunk after `_` is not a layer either.
        assert_eq!(
            split_base_layer("trailing_"),
            ("trailing_".to_string(), None)
        );
    }

    #[test]
    fn accessor_methods_compress_layered_fields_to_slice_index() {
        // Three layers' worth of `input_layernorm_<n>` fields collapse
        // into ONE `pub fn input_layernorm(&self, layer: u32) ->
        // &RmsNorm` whose body is a single slice index — no per-layer
        // arms, no 40-arm match. This is the load-bearing invariant
        // for the impl Weights compression: 40 arms × 5+ accessors
        // worth of Rust tokens disappear.
        let ty: TokenStream = quote! { ::ferrite_kernels::layers::RmsNorm };
        let accessors = (0u64..3)
            .map(|n| WeightAccessor {
                name: format_ident!("input_layernorm_{}", n),
                rust_type: ty.clone(),
                source_weights: vec![],
            })
            .collect::<Vec<_>>();
        let ts = emit_weights_accessor_methods(&accessors).to_string();
        assert!(ts.contains("impl Weights"));
        assert!(ts.contains("fn input_layernorm"));
        assert!(ts.contains("layer : u32"));
        // Single slice index — no match arm bloat. The body is
        // `unsafe { self.input_layernorm.get_unchecked(layer as usize) }`.
        assert!(ts.contains("self . input_layernorm . get_unchecked"));
        assert!(ts.contains("layer as usize"));
        // No 40-arm match anymore — the per-layer arms are gone.
        assert!(!ts.contains("match layer"));
        // No per-layer field references — the data lives in a single
        // `Vec<T>` field with the base name.
        assert!(!ts.contains("input_layernorm_0"));
        assert!(!ts.contains("input_layernorm_1"));
        assert!(!ts.contains("input_layernorm_2"));
        // No format-args bloat.
        assert!(!ts.contains("out of range"));
        assert!(!ts.contains("panic"));
    }

    #[test]
    fn accessor_methods_emit_unit_arm_for_unlayered_fields() {
        // `embed_tokens` has no trailing layer index; the method
        // ignores its layer arg and returns the field directly. Same
        // shape as before the Vec compression — unindexed accessors
        // never had a match.
        let accessors = vec![WeightAccessor {
            name: format_ident!("embed_tokens"),
            rust_type: quote! { ::ferrite_kernels::layers::Embedding },
            source_weights: vec![],
        }];
        let ts = emit_weights_accessor_methods(&accessors).to_string();
        assert!(ts.contains("fn embed_tokens"));
        // Unindexed accessor's layer arg is `_: u32` — the underscore
        // prefix on `_layer` was unnecessary chars, dropped along
        // with per-method `#[cfg]` / `#[inline]` / `#[allow]` attrs.
        assert!(ts.contains("_ : u32"));
        assert!(ts.contains("& self . embed_tokens"));
        // No `match` block for non-layered accessors — the body is
        // a direct field reference, branchless. No slice index either.
        assert!(!ts.contains("match layer"));
        assert!(!ts.contains("get_unchecked"));
    }

    #[test]
    #[should_panic(expected = "mixes layered and non-layered")]
    fn accessor_methods_panic_on_mixed_layered_and_unlayered() {
        // Pathological: an accessor base with both an indexed and
        // an un-indexed field. Codegen must refuse — there's no
        // sensible single method body for the mix, and silently
        // picking one would mask a solver / Impl bug.
        let ty: TokenStream = quote! { ::ferrite_kernels::layers::RmsNorm };
        let accessors = vec![
            WeightAccessor {
                name: format_ident!("norm"),
                rust_type: ty.clone(),
                source_weights: vec![],
            },
            WeightAccessor {
                name: format_ident!("norm_0"),
                rust_type: ty.clone(),
                source_weights: vec![],
            },
        ];
        let _ = emit_weights_accessor_methods(&accessors);
    }

    #[test]
    #[should_panic(expected = "mismatched types")]
    fn accessor_methods_panic_on_type_disagreement_within_a_base() {
        // Two layers under the same base claim different `rust_type`s
        // — codegen invariant violation, not a recoverable case.
        let accessors = vec![
            WeightAccessor {
                name: format_ident!("input_layernorm_0"),
                rust_type: quote! { ::ferrite_kernels::layers::RmsNorm },
                source_weights: vec![],
            },
            WeightAccessor {
                name: format_ident!("input_layernorm_1"),
                rust_type: quote! { ::ferrite_kernels::layers::LinearLayer },
                source_weights: vec![],
            },
        ];
        let _ = emit_weights_accessor_methods(&accessors);
    }

    #[test]
    fn accessor_methods_empty_input_yields_empty_tokens() {
        // No accessors → no impl block. (Empty `impl Weights {}`
        // would be valid Rust but pointless; the codegen elides it.)
        let ts = emit_weights_accessor_methods(&[]).to_string();
        assert!(ts.is_empty());
    }

    // ── group_accessors_by_base / Vec compression invariants ────────

    fn ln_acc(layer: Option<u64>) -> WeightAccessor {
        let name = match layer {
            Some(n) => format_ident!("input_layernorm_{}", n),
            None => format_ident!("input_layernorm"),
        };
        WeightAccessor {
            name,
            rust_type: quote! { ::ferrite_kernels::layers::RmsNorm },
            source_weights: vec![],
        }
    }

    #[test]
    fn group_accessors_layered_collapses_to_one_group() {
        let accs: Vec<_> = (0u64..4).map(|n| ln_acc(Some(n))).collect();
        let groups = group_accessors_by_base(&accs);
        assert_eq!(groups.len(), 1, "all 4 layers share one base group");
        let g = &groups[0];
        assert_eq!(g.base, "input_layernorm");
        assert!(
            matches!(g.kind, AccessorGroupKind::LayeredContiguous),
            "all entries are indexed and start at layer 0 → LayeredContiguous"
        );
        assert_eq!(g.entries.len(), 4);
        for (i, (layer_opt, _acc)) in g.entries.iter().enumerate() {
            assert_eq!(*layer_opt, Some(i as u64), "contiguous layers 0..N");
        }
    }

    #[test]
    fn group_accessors_unindexed_is_one_entry_with_none() {
        let accs = vec![WeightAccessor {
            name: format_ident!("embed_tokens"),
            rust_type: quote! { ::ferrite_kernels::layers::Embedding },
            source_weights: vec![],
        }];
        let groups = group_accessors_by_base(&accs);
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert_eq!(g.base, "embed_tokens");
        assert!(matches!(g.kind, AccessorGroupKind::Unindexed));
        assert_eq!(g.entries.len(), 1);
        assert_eq!(g.entries[0].0, None);
    }

    #[test]
    fn group_accessors_orders_groups_alphabetically() {
        // Order matters: a tied `lm_head` references `embed_tokens`,
        // so `embed_tokens`'s let-binding must precede `lm_head`'s in
        // the load body. Alphabetical ordering preserves that without
        // a special sort.
        let accs = vec![
            WeightAccessor {
                name: format_ident!("lm_head"),
                rust_type: quote! { ::ferrite_kernels::layers::LinearLayer },
                source_weights: vec![],
            },
            WeightAccessor {
                name: format_ident!("embed_tokens"),
                rust_type: quote! { ::ferrite_kernels::layers::Embedding },
                source_weights: vec![],
            },
            ln_acc(Some(0)),
            ln_acc(Some(1)),
        ];
        let groups = group_accessors_by_base(&accs);
        let bases: Vec<_> = groups.iter().map(|g| g.base.as_str()).collect();
        assert_eq!(bases, vec!["embed_tokens", "input_layernorm", "lm_head"]);
    }

    #[test]
    #[should_panic(expected = "mixes layered and non-layered")]
    fn group_accessors_panics_on_mixed_layered_unindexed() {
        let accs = vec![ln_acc(None), ln_acc(Some(0))];
        let _ = group_accessors_by_base(&accs);
    }

    #[test]
    #[should_panic(expected = "mismatched types")]
    fn group_accessors_panics_on_type_disagreement() {
        let accs = vec![
            WeightAccessor {
                name: format_ident!("input_layernorm_0"),
                rust_type: quote! { ::ferrite_kernels::layers::RmsNorm },
                source_weights: vec![],
            },
            WeightAccessor {
                name: format_ident!("input_layernorm_1"),
                rust_type: quote! { ::ferrite_kernels::layers::LinearLayer },
                source_weights: vec![],
            },
        ];
        let _ = group_accessors_by_base(&accs);
    }

    #[test]
    fn group_accessors_sparse_layered_falls_back_to_per_layer_fields() {
        // Layered group with a gap (layers 0 and 2, missing 1) →
        // LayeredSparse, not a Vec. Real-world archs hit sparse
        // layouts (DeepSeek MoE on layers 1..N, future per-window-
        // size attention overrides), so codegen must accept them
        // and emit per-layer fields + match-arm accessors.
        let accs = vec![ln_acc(Some(0)), ln_acc(Some(2))];
        let groups = group_accessors_by_base(&accs);
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert!(matches!(g.kind, AccessorGroupKind::LayeredSparse));
        // Entries preserve the layer indices the group originally
        // had — no padding/None at the gap.
        assert_eq!(
            g.entries.iter().map(|(l, _)| *l).collect::<Vec<_>>(),
            vec![Some(0u64), Some(2u64)]
        );
    }

    #[test]
    fn group_accessors_layered_starting_above_zero_is_sparse() {
        // DeepSeek-V2's `moe` accessor: layer 0 is dense FFN, layers
        // 1..N are MoE — `moe` only exists for 1..N. The Vec-compressed
        // path would be off-by-one (Vec[layer as usize] reads layer
        // L from index L instead of L-1), so non-zero-starting groups
        // also take the sparse fallback.
        let accs = vec![ln_acc(Some(1)), ln_acc(Some(2))];
        let groups = group_accessors_by_base(&accs);
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert!(matches!(g.kind, AccessorGroupKind::LayeredSparse));
    }

    #[test]
    fn accessor_methods_emit_match_for_sparse_layered_groups() {
        // The sparse path emits per-layer fields and a match-arm
        // accessor that's identical in shape to the legacy
        // pre-Vec-compression emit. Codegen-issued static rows only
        // pass layers that exist, so the catch-all is
        // `unreachable_unchecked`.
        let accs = vec![
            WeightAccessor {
                name: format_ident!("moe_1"),
                rust_type: quote! { ::ferrite_kernels::layers_moe::DeepSeekV2MoELayer },
                source_weights: vec![],
            },
            WeightAccessor {
                name: format_ident!("moe_2"),
                rust_type: quote! { ::ferrite_kernels::layers_moe::DeepSeekV2MoELayer },
                source_weights: vec![],
            },
        ];
        let ts = emit_weights_accessor_methods(&accs).to_string();
        assert!(ts.contains("fn moe"));
        assert!(ts.contains("match layer"));
        // `Literal::u32_unsuffixed` emits the bare integer; the
        // arm body resolves to it via the inferred match-arm type.
        assert!(ts.contains("1 => & self . moe_1"));
        assert!(ts.contains("2 => & self . moe_2"));
        assert!(ts.contains("unreachable_unchecked"));
        // Sparse groups don't take the Vec path.
        assert!(!ts.contains("get_unchecked"));
    }

    // ── Vec compression: layered prefix templating ──────────────────

    #[test]
    fn layer_template_replaces_layers_dot_zero_with_runtime_format() {
        let ts =
            layer_templated_prefix_expr("model.layers.0.input_layernorm", None, None).to_string();
        // Routes through the `layer_weight_path` helper so the
        // emitted load_with body has one fn-call per accessor per
        // layer instead of the 5-line `format!()` macro expansion.
        assert!(
            ts.contains("layer_weight_path"),
            "expected layer_weight_path call, got: {ts}"
        );
        assert!(
            ts.contains("\"input_layernorm\""),
            "expected suffix string literal, got: {ts}"
        );
    }

    #[test]
    #[should_panic(expected = "doesn't start with `model.layers.0`")]
    fn layer_template_rejects_unindexed_prefix() {
        // `lm_head` and `model.embed_tokens` are unindexed prefixes;
        // they must never reach the layered template helper. A panic
        // here surfaces a `default_required_weights` bug instead of
        // silently emitting a malformed Vec build.
        let _ = layer_templated_prefix_expr("model.embed_tokens", None, None);
    }

    #[test]
    fn emit_layered_load_body_uses_layer_template_for_rmsnorm() {
        let plan = FieldLoad::RmsNorm("model.layers.0.input_layernorm".to_string(), 1e-5);
        let ts = emit_layered_load_body(&plan, 32, 1, false, None, "model.layers").to_string();
        // Delegates to the load_layered_rms_norm helper in
        // ferrite-forward. The closure / collect / Vec annotation
        // the loop used to emit per accessor are now owned by the
        // helper — call sites collapse to one line.
        assert!(
            ts.contains("load_layered_rms_norm"),
            "expected helper call, got: {ts}"
        );
        // Threads (gw, n_layers, suffix, eps) — the suffix is the
        // post-`model.layers.0.` tail, n_layers is the count.
        assert!(ts.contains("\"input_layernorm\""));
        assert!(ts.contains("32"));
        // and bakes the static eps literal.
        assert!(ts.contains("0.00001"));
        // No closure / collect machinery on the call site.
        assert!(!ts.contains(". collect"));
        assert!(!ts.contains("| layer :"));
    }

    #[test]
    fn emit_layered_load_body_binds_locals_for_concat_prefixes() {
        // The `_concat` helpers take `&[&str]` of suffixes; the
        // emitter strips each `model.layers.0.` prefix down to the
        // tail and bakes them as a static `&[…]`. The per-iteration
        // String binds + as_str refs the previous shape needed are
        // now owned by `load_layered_linear_dense_concat`.
        let plan = FieldLoad::LinearConcat(vec![
            "model.layers.0.self_attn.q_proj".to_string(),
            "model.layers.0.self_attn.k_proj".to_string(),
            "model.layers.0.self_attn.v_proj".to_string(),
        ]);
        let ts = emit_layered_load_body(&plan, 32, 1, false, None, "model.layers").to_string();
        assert!(
            ts.contains("load_layered_linear_dense_concat"),
            "expected helper call, got: {ts}"
        );
        assert!(ts.contains("\"self_attn.q_proj\""));
        assert!(ts.contains("\"self_attn.k_proj\""));
        assert!(ts.contains("\"self_attn.v_proj\""));
        // No per-iteration String bind / .as_str() / closure tokens.
        assert!(!ts.contains("__p_0"));
        assert!(!ts.contains(". as_str ()"));
        assert!(!ts.contains("| layer :"));
    }

    /// At `tp_world_size = 1` every emitted load call must be
    /// byte-identical to the pre-task-#5 build — the existing tests
    /// above pin one direction (helper presence + suffix bake), this
    /// one pins the negative: the `_sharded` variant must NOT appear.
    /// Defends against a future regression that drops the `tp == 1`
    /// short-circuit and silently routes the dense build through
    /// `_sharded` with `world = 1`.
    #[test]
    fn emit_layered_load_body_at_tp_eq_1_emits_no_sharded_call() {
        let plans: &[FieldLoad] = &[
            FieldLoad::LinearDense("model.layers.0.self_attn.q_proj".to_string()),
            FieldLoad::LinearDense("model.layers.0.self_attn.o_proj".to_string()),
            FieldLoad::LinearDense("model.layers.0.input_layernorm".to_string()),
            FieldLoad::LinearConcat(vec![
                "model.layers.0.self_attn.q_proj".to_string(),
                "model.layers.0.self_attn.k_proj".to_string(),
                "model.layers.0.self_attn.v_proj".to_string(),
            ]),
            FieldLoad::Embedding("model.layers.0.embed_tokens".to_string()),
        ];
        for plan in plans {
            let ts = emit_layered_load_body(plan, 32, 1, false, None, "model.layers").to_string();
            assert!(
                !ts.contains("_sharded"),
                "tp=1 must never emit a `_sharded` helper call (got: {ts})",
            );
            assert!(
                !ts.contains("tp_rank"),
                "tp=1 must not reference `tp_rank` (got: {ts})",
            );
        }
    }

    /// At tp>1, layered `LinearDense` accessors route to the sharded
    /// helper with the right `dim` baked in: column-parallel
    /// (q/k/v/gate/up) → `dim = 0`; row-parallel (o/down) → `dim = 1`.
    /// The shard kind comes from the prefix's last segment via
    /// `tp_lowering::shard_kind_for_dotted_prefix`. A regression that
    /// flipped the dim or routed q_proj as row-parallel would mismatch
    /// the FUF's sharded `<W>::*` constants → kernel shape error at
    /// the first per-rank gemm; this test catches that at codegen time.
    #[test]
    fn emit_layered_load_body_at_tp_gt_1_dispatches_by_shard_kind() {
        // Column-parallel (ShardDim0): q_proj. Expect `dim = 0` lit.
        let q = FieldLoad::LinearDense("model.layers.0.self_attn.q_proj".to_string());
        let ts = emit_layered_load_body(&q, 32, 2, false, None, "model.layers").to_string();
        assert!(
            ts.contains("load_layered_linear_dense_sharded"),
            "tp=2 q_proj must route to sharded helper (got: {ts})",
        );
        assert!(
            ts.contains("0usize"),
            "q_proj must be dim=0 column-parallel (got: {ts})",
        );
        assert!(ts.contains("tp_rank"));

        // Row-parallel (ShardDim1): o_proj. Expect `dim = 1` lit.
        let o = FieldLoad::LinearDense("model.layers.0.self_attn.o_proj".to_string());
        let ts = emit_layered_load_body(&o, 32, 2, false, None, "model.layers").to_string();
        assert!(
            ts.contains("load_layered_linear_dense_sharded"),
            "tp=2 o_proj must route to sharded helper (got: {ts})",
        );
        assert!(
            ts.contains("1usize"),
            "o_proj must be dim=1 row-parallel (got: {ts})",
        );

        // Replicate (norm, etc.): NO sharded helper, even at tp>1.
        let n = FieldLoad::LinearDense("model.layers.0.input_layernorm".to_string());
        let ts = emit_layered_load_body(&n, 32, 2, false, None, "model.layers").to_string();
        assert!(
            !ts.contains("_sharded"),
            "Replicate path must not route to sharded helper at tp>1 (got: {ts})",
        );

        // Concat (always column-parallel) → `_concat_sharded`. Bakes
        // `world` literal but no `dim` arg (concat-sharded is always
        // dim=0 internally).
        let c = FieldLoad::LinearConcat(vec![
            "model.layers.0.mlp.gate_proj".to_string(),
            "model.layers.0.mlp.up_proj".to_string(),
        ]);
        let ts = emit_layered_load_body(&c, 32, 4, false, None, "model.layers").to_string();
        assert!(
            ts.contains("load_layered_linear_dense_concat_sharded"),
            "tp=4 gate_up concat must route to concat_sharded (got: {ts})",
        );
        // The macro bakes `tp_world_size` as an unsuffixed integer
        // literal cast to `usize` (e.g. `4 as usize`). Asserting on
        // the bare `4 as usize` token sequence — the unsuffixed form
        // is `proc_macro2::Literal::u8_unsuffixed`'s contract.
        assert!(
            ts.contains("4 as usize"),
            "expected `4 as usize` for baked tp_world_size literal (got: {ts})",
        );
    }
}

#[cfg(test)]
mod fingerprint_tests {
    use super::*;
    use std::path::PathBuf;

    /// Per-arch config dir, in ff-interpreter's per-crate layout
    /// (`crates/ferrite-model-<arch>/configs/`). Pass the arch slug
    /// using hyphens, e.g. `"deepseek-v3"`, `"qwen3"`.
    fn arch_configs(arch: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(format!("ferrite-model-{arch}"))
            .join("configs")
    }

    /// MLA archs (DeepSeek V3 / Kimi K2) ship `q_a_proj` rather than
    /// `q_proj`, so the FP8-block disambiguation tensor names must
    /// follow the same `fp_leaf` selection the rest of the
    /// fingerprint already uses. Regression for the bug where the
    /// V3 FP8-block fingerprint hardcoded `q_proj.weight_scale_inv`
    /// and silently rejected every V3 FP8-block checkpoint.
    #[test]
    fn fp8_block_disambiguation_uses_q_a_proj_for_mla_archs() {
        let dir = arch_configs("deepseek-v3");
        let configs = crate::config::load_dir(&dir).expect("load deepseek-v3 configs");
        let manifest = crate::weights_manifest::load_or_empty(&dir)
            .expect("load deepseek-v3 weights manifest");
        // Pick a V3 variant with FP8-block quantization (block_size: Some).
        let model = configs
            .iter()
            .find(|c| {
                matches!(
                    c.quantization.as_ref().map(|qc| &qc.method),
                    Some(crate::quantization::QuantMethod::Fp8 {
                        block_size: Some(_),
                        ..
                    })
                )
            })
            .expect("at least one V3 FP8-block variant");
        let ts = emit_fingerprint_check(model, &manifest, 1).to_string();
        assert!(
            ts.contains("q_a_proj.weight_scale_inv"),
            "MLA arch FP8-block fingerprint should sniff q_a_proj.weight_scale_inv, got:\n{ts}",
        );
        assert!(
            !ts.contains("q_proj.weight_scale_inv"),
            "MLA arch FP8-block fingerprint must not reference q_proj.weight_scale_inv \
             (V3/K2 ship q_a_proj on disk; q_proj presence would silently reject every \
             real checkpoint), got:\n{ts}",
        );
        assert!(
            ts.contains("q_a_proj.weight_scale"),
            "MLA arch FP8-block fingerprint should sniff q_a_proj.weight_scale, got:\n{ts}",
        );
    }

    /// Flat-Q MLA arches (Moonlight / K2 direct-q_proj) ship `q_proj`
    /// rather than `q_a_proj`, so the FP8-block fingerprint must use
    /// `q_proj.weight_scale_inv` even though the arch is otherwise
    /// structurally MLA (kv_a_proj_with_mqa, kv_b_proj, etc.).
    #[test]
    fn fp8_block_disambiguation_uses_q_proj_for_flat_q_mla_archs() {
        let dir = arch_configs("deepseek-v3-flat");
        let configs = crate::config::load_dir(&dir).expect("load deepseek-v3-flat configs");
        let manifest = crate::weights_manifest::load_or_empty(&dir)
            .expect("load deepseek-v3-flat weights manifest");
        let model = configs
            .iter()
            .find(|c| {
                matches!(
                    c.quantization.as_ref().map(|qc| &qc.method),
                    Some(crate::quantization::QuantMethod::Fp8 {
                        block_size: Some(_),
                        ..
                    })
                )
            })
            .expect("at least one deepseek-v3-flat FP8-block variant");
        let ts = emit_fingerprint_check(model, &manifest, 1).to_string();
        assert!(
            ts.contains("q_proj.weight_scale_inv"),
            "flat-Q MLA FP8-block fingerprint should use q_proj.weight_scale_inv \
             (no q_a_proj on disk for q_lora_rank=null checkpoints), got:\n{ts}",
        );
        assert!(
            !ts.contains("q_a_proj.weight_scale_inv"),
            "flat-Q MLA FP8-block fingerprint must not reference q_a_proj \
             (Moonlight ships q_proj, not q_a_proj), got:\n{ts}",
        );
    }

    /// Non-MLA arches (Llama / Qwen / etc.) keep the historical
    /// `q_proj` leaf — the fp_leaf selection is purely opt-in for
    /// archs whose manifest declares `q_a_proj`.
    #[test]
    fn fp8_block_disambiguation_uses_q_proj_for_non_mla_archs() {
        let dir = arch_configs("qwen3");
        let configs = crate::config::load_dir(&dir).expect("load qwen3 configs");
        let manifest =
            crate::weights_manifest::load_or_empty(&dir).expect("load qwen3 weights manifest");
        let model = configs
            .iter()
            .find(|c| {
                matches!(
                    c.quantization.as_ref().map(|qc| &qc.method),
                    Some(crate::quantization::QuantMethod::Fp8 {
                        block_size: Some(_),
                        ..
                    })
                )
            })
            .expect("at least one Qwen3 FP8-block variant");
        let ts = emit_fingerprint_check(model, &manifest, 1).to_string();
        assert!(
            ts.contains("q_proj.weight_scale_inv"),
            "non-MLA FP8-block fingerprint should still use q_proj, got:\n{ts}",
        );
    }

    /// Build a minimal `Program` whose `WeightTable` carries one
    /// entry — `lm_head` — at WeightId(0). Other tests can intern
    /// additional weights to push `lm_head` off slot 0; this helper
    /// keeps the encoder/decoder layout test focused on what matters
    /// (the dotted weight name carried at the FUF terminal).
    fn layout_test_program(weight_paths: &[&[&str]]) -> crate::classified::Program {
        let mut weights = crate::classified::WeightTable::default();
        for path in weight_paths {
            let segments: Vec<String> = path.iter().map(|s| (*s).to_string()).collect();
            let _ = weights.intern_str(segments);
        }
        crate::classified::Program {
            statements: Vec::new(),
            locals: Default::default(),
            weights,
            reshape_targets: Default::default(),
            prelude: crate::classified::Prelude::Decoder,
            vision_layout: None,
            decoder_safetensors_prefix: None,
        }
    }

    fn shape_2d() -> Vec<crate::shape::Dim> {
        use crate::shape::Dim;
        vec![Dim::Lit(4), Dim::Lit(16)]
    }

    /// Decoder layout: FUF ends in `gemm(<tile>, lm_head)`.
    /// `backbone_layout` reports `Decoder` and the carried
    /// `(TileId, slot)` is the lm_head Gemm's hidden-state input.
    #[test]
    fn backbone_layout_recognizes_decoder_terminator() {
        use crate::classified::{ExternKind, OpKind, WeightId};
        use crate::fuf::{Fuf, FufInput, FufNode, TileId};
        use crate::quantization::StorageFormat;

        let program = layout_test_program(&[&["lm_head", "weight"]]);
        let fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![shape_2d()],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Gemm,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                        FufInput::Weight {
                            id: WeightId(0),
                            index: None,
                            storage: StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![shape_2d()],
                },
            ],

            barrier_meta: Default::default(),
        };
        match backbone_layout(&fuf, &program) {
            BackboneLayout::Decoder { backbone_out } => {
                assert_eq!(
                    backbone_out,
                    (TileId(0), 0),
                    "decoder backbone-out must be the lm_head Gemm's first tile input",
                );
            }
            BackboneLayout::Encoder => panic!("expected Decoder layout, got Encoder"),
        }
    }

    /// Encoder layout: FUF ends in something other than
    /// `gemm(_, lm_head)` (here a plain `Add` — same shape ModernBERT
    /// produces at the encoder output). `backbone_layout` returns
    /// `Encoder`; the FUF's terminal is itself the backbone output.
    #[test]
    fn backbone_layout_recognizes_encoder_terminator() {
        use crate::classified::{ExternKind, OpKind};
        use crate::fuf::{Fuf, FufInput, FufNode, TileId};

        let program = layout_test_program(&[]);
        let fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![shape_2d()],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Add,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                    ],
                    outputs: vec![shape_2d()],
                },
            ],

            barrier_meta: Default::default(),
        };
        assert!(
            matches!(backbone_layout(&fuf, &program), BackboneLayout::Encoder),
            "FUF terminating in Add must classify as Encoder",
        );
    }

    /// Decoder terminator with a tp>1 AllGather appended after the
    /// lm_head Gemm. `backbone_layout` walks past the AllGather to
    /// find the underlying Gemm and reports `Decoder` with the right
    /// hidden-state input.
    #[test]
    fn backbone_layout_walks_past_allgather_to_decoder_gemm() {
        use crate::classified::{ExternKind, OpKind, WeightId};
        use crate::fuf::{Fuf, FufInput, FufNode, TileId};
        use crate::quantization::StorageFormat;

        let program = layout_test_program(&[&["lm_head", "weight"]]);
        let fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![shape_2d()],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Gemm,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                        FufInput::Weight {
                            id: WeightId(0),
                            index: None,
                            storage: StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![shape_2d()],
                },
                FufNode {
                    id: TileId(2),
                    op: OpKind::AllGather,
                    inputs: vec![FufInput::Tile {
                        id: TileId(1),
                        slot: 0,
                    }],
                    outputs: vec![shape_2d()],
                },
            ],

            barrier_meta: Default::default(),
        };
        match backbone_layout(&fuf, &program) {
            BackboneLayout::Decoder { backbone_out } => assert_eq!(backbone_out, (TileId(0), 0)),
            BackboneLayout::Encoder => panic!("expected Decoder past AllGather"),
        }
    }

    /// Terminal Gemm whose second input is some non-`lm_head` weight
    /// (e.g. a plain `down_proj`) is NOT a decoder lm_head row —
    /// classify it as Encoder so the lowering doesn't try to split a
    /// nonexistent lm_head off.
    #[test]
    fn backbone_layout_non_lm_head_gemm_is_encoder() {
        use crate::classified::{ExternKind, OpKind, WeightId};
        use crate::fuf::{Fuf, FufInput, FufNode, TileId};
        use crate::quantization::StorageFormat;

        let program = layout_test_program(&[&["mlp", "down_proj", "weight"]]);
        let fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![shape_2d()],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Gemm,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                        FufInput::Weight {
                            id: WeightId(0),
                            index: None,
                            storage: StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![shape_2d()],
                },
            ],

            barrier_meta: Default::default(),
        };
        assert!(
            matches!(backbone_layout(&fuf, &program), BackboneLayout::Encoder),
            "Gemm whose 2nd input is not `lm_head` must classify as Encoder",
        );
    }
}
