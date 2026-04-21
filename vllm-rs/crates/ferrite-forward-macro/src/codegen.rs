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

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use crate::classified::{OpKind, Program, WeightId};
use crate::config::ModelParams;
use crate::emit::{EmitCtx, EmitMode, LocalMap};
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{ImplId, ImplementationLibrary, WeightAccessor};
use crate::schedule::{Loop, WorkloadLoops};
use crate::solver::{Assignment, SubgraphId, WorkloadAssignments};

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
fn safetensors_prefix(program: &Program, id: WeightId, index: Option<u64>) -> String {
    let joined = program.weights.path(id).join(".");
    match (index, joined.as_str()) {
        (_, "lm_head") => "lm_head".to_string(),
        (Some(l), _) => format!("model.layers.{l}.{joined}"),
        (None, _) => format!("model.{joined}"),
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
    /// `CohereLayerNorm::load(gw, prefix, eps)`. Same shape as
    /// `RmsNorm` — single weight tensor + scalar eps — but the eps
    /// source is the `layer_norm_eps` config field rather than
    /// `rms_norm_eps`. Used for arches whose pre-attention norm is
    /// a full LayerNorm with weight only (Cohere's CommandR family).
    CohereLayerNorm(String, f32),
    /// `LinearLayer::load_dense(gw, prefix)`.
    LinearDense(String),
    /// `LinearLayer::load_dense_concat(gw, &[prefix0, prefix1, ...], stream)`.
    LinearConcat(Vec<String>),
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
    let is_embedding =
        ty.ends_with("::Embedding") || ty == "Embedding" || ty.ends_with("layers::Embedding");
    let is_rmsnorm =
        ty.ends_with("::RmsNorm") || ty == "RmsNorm" || ty.ends_with("layers::RmsNorm");
    let is_cohere_layer_norm = ty.ends_with("::CohereLayerNorm")
        || ty == "CohereLayerNorm"
        || ty.ends_with("layers::CohereLayerNorm");
    let is_linear =
        ty.ends_with("::LinearLayer") || ty == "LinearLayer" || ty.ends_with("layers::LinearLayer");
    let is_marlin = ty.ends_with("::MarlinLinear")
        || ty == "MarlinLinear"
        || ty.ends_with("layers::MarlinLinear");
    let is_bnb4 = ty.ends_with("::Bnb4bitLinear")
        || ty == "Bnb4bitLinear"
        || ty.ends_with("layers::Bnb4bitLinear");
    let is_fp8 =
        ty.ends_with("::Fp8Linear") || ty == "Fp8Linear" || ty.ends_with("layers::Fp8Linear");
    let is_fp8_block = ty.ends_with("::Fp8BlockLinear")
        || ty == "Fp8BlockLinear"
        || ty.ends_with("layers::Fp8BlockLinear");

    let prefixes: Vec<String> = accessor
        .source_weights
        .iter()
        .map(|(id, idx)| safetensors_prefix(program, *id, *idx))
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

    if is_fp8 {
        // FP8 per-tensor / per-channel. Sources must all resolve to
        // `StorageFormat::Fp8 { block_size: None, .. }`. Shapes +
        // scale layout are sniffed by the loader at runtime
        // (per-tensor scalar vs per-channel [N,1] vs online BF16→FP8
        // quant), so no macro-time dim math needed.
        for (wid, _idx) in &accessor.source_weights {
            let fmt = crate::quantization::storage_format_for_weight(program, fuf, *wid, model);
            if !matches!(
                fmt,
                crate::quantization::StorageFormat::Fp8 {
                    block_size: None,
                    ..
                }
            ) {
                panic!(
                    "accessor `{}` declared `Fp8Linear` but source weight resolves to \
                     non-per-tensor-FP8 storage ({fmt:?}) — matcher bug",
                    accessor.name,
                );
            }
        }
        return FieldLoad::Fp8Linear { prefixes };
    }

    if is_fp8_block {
        // FP8 blockwise (DeepSeek-V3 128×128). Sources must all
        // resolve to `StorageFormat::Fp8 { block_size: Some(_), .. }`.
        // `Fp8BlockLinear::load` derives the per-shard block size
        // from the ratio of weight shape to scale shape at runtime;
        // no macro-time dim math needed.
        for (wid, _idx) in &accessor.source_weights {
            let fmt = crate::quantization::storage_format_for_weight(program, fuf, *wid, model);
            if !matches!(
                fmt,
                crate::quantization::StorageFormat::Fp8 {
                    block_size: Some(_),
                    ..
                }
            ) {
                panic!(
                    "accessor `{}` declared `Fp8BlockLinear` but source weight resolves to \
                     non-block-FP8 storage ({fmt:?}) — matcher bug",
                    accessor.name,
                );
            }
        }
        return FieldLoad::Fp8BlockLinear { prefixes };
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
    } else if is_cohere_layer_norm {
        assert_eq!(
            prefixes.len(),
            1,
            "CohereLayerNorm accessor `{}` with {} sources",
            accessor.name,
            prefixes.len()
        );
        let eps = layer_norm_eps(model);
        FieldLoad::CohereLayerNorm(prefixes.into_iter().next().unwrap(), eps)
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
            && prefixes[0] == "lm_head"
            && tie_word_embeddings(model)
        {
            return FieldLoad::LinearTiedToEmbedding(syn::Ident::new(
                "embed_tokens",
                proc_macro2::Span::call_site(),
            ));
        }
        if prefixes.len() == 1 {
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
    // HF configs carry `rms_norm_eps` as a top-level float. The
    // config loader today captures only integer bounds; read the
    // JSON directly for this float. Fall back to the HF default if
    // absent (Llama/Qwen2/Mistral all set it explicitly).
    let fallback: f32 = 1e-5;
    let Ok(s) = std::fs::read_to_string(&model.source_path) else {
        return fallback;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) else {
        return fallback;
    };
    v.get("rms_norm_eps")
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
        .unwrap_or(fallback)
}

fn layer_norm_eps(model: &ModelParams) -> f32 {
    // Cohere/CommandR config carries `layer_norm_eps` (full LayerNorm
    // epsilon — distinct field from `rms_norm_eps`). Same JSON-read
    // shape as `rms_norm_eps` since the integer-only bounds map
    // doesn't capture floats.
    let fallback: f32 = 1e-5;
    let Ok(s) = std::fs::read_to_string(&model.source_path) else {
        return fallback;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) else {
        return fallback;
    };
    v.get("layer_norm_eps")
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
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
        Some(_) => ("qweight", "weight"),
        None => ("weight", "qweight"),
    };

    let last_layer = num_hidden_layers.saturating_sub(1);
    // Packed-source archs (Phi-3 family) ship `self_attn.qkv_proj.weight`
    // on disk instead of the per-slice `self_attn.q_proj.weight` that
    // the default fingerprint looks for. Detect via manifest's
    // `__packed_splits__` section: if the arch declares
    // `self_attn.qkv_proj` as a packed parent, sniff THAT tensor
    // instead — without this, the fingerprint misses and ferrite
    // falls back to the hand-written path even when it has a
    // compiled variant for this arch.
    let fp_leaf: &str = if manifest.packed_splits.contains_key("self_attn.qkv_proj") {
        "self_attn.qkv_proj"
    } else {
        "self_attn.q_proj"
    };
    let last_tensor = format!("model.layers.{last_layer}.{fp_leaf}.{suffix}");
    let one_past_tensor = format!("model.layers.{num_hidden_layers}.{fp_leaf}.{suffix}");
    let opposite_tensor = format!("model.layers.0.{fp_leaf}.{opposite_suffix}");
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
    let bnb4_marker_tensor = "model.layers.0.self_attn.q_proj.weight.absmax";
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
    let fp8_marker_tensor = "model.layers.0.self_attn.q_proj.weight_scale";

    let hidden_lit = proc_macro2::Literal::usize_unsuffixed(hidden_size as usize);
    let vocab_lit = proc_macro2::Literal::usize_unsuffixed(vocab_size as usize);

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
        None => quote! {},
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
            let g_idx_tensor = "model.layers.0.self_attn.q_proj.g_idx";
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
                let input_scale_tensor = "model.layers.0.self_attn.q_proj.input_scale";
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
                let input_scale_tensor = "model.layers.0.self_attn.q_proj.input_scale";
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
            let inv_tensor = "model.layers.0.self_attn.q_proj.weight_scale_inv";
            let scale_tensor = "model.layers.0.self_attn.q_proj.weight_scale";
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
            let inv_tensor = "model.layers.0.self_attn.q_proj.weight_scale_inv";
            let scale_tensor = "model.layers.0.self_attn.q_proj.weight_scale";
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

    quote! {
        /// Return `true` iff the tensors in `gw` match this
        /// variant's compile-time fingerprint. See
        /// `emit_fingerprint_check` in the macro for the rules.
        /// Emitted as a free fn (not `Weights::fingerprint_matches`
        /// method) so shim variants can alias `Weights` to a
        /// canonical sibling while still carrying a variant-
        /// specific fingerprint check.
        #[cfg(feature = "cuda")]
        pub fn fingerprint_matches(
            gw: &::ferrite_cuda_core::weights::GpuWeights,
            hf: ::ferrite_forward::HfFingerprint<'_>,
        ) -> bool {
            match gw.tensor_info("model.embed_tokens.weight") {
                Some((shape, _))
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
fn emit_weights_struct(
    program: &Program,
    fuf: &Fuf,
    sfufs: &WorkloadAssignments,
    lib: &ImplementationLibrary,
    model: &ModelParams,
    manifest: &crate::weights_manifest::WeightsManifest,
    mode: WeightsEmitMode<'_>,
) -> (TokenStream, crate::emit::WeightLayout) {
    let accessors = match collect_accessors(program, fuf, sfufs, lib) {
        Ok(a) => a,
        Err(err) => return (err, crate::emit::WeightLayout::new()),
    };

    // Storage-format guard: a given accessor's `rust_type` must be
    // compatible with every one of its source weights' storage
    // formats. The allowed pairs today:
    //   `LinearLayer`   ↔ `Dense`
    //   `Embedding`     ↔ `Dense`
    //   `RmsNorm`       ↔ `Dense`
    //   `MarlinLinear`  ↔ `Awq { .. }` | `Gptq { .. }`
    //   `Bnb4bitLinear` ↔ `Bnb4 { .. }`
    //   `Fp8Linear`     ↔ `Fp8 { .. }`
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
        let accessor_is_fp8 =
            ty.ends_with("::Fp8Linear") || ty == "Fp8Linear" || ty.ends_with("layers::Fp8Linear");
        let accessor_is_fp8_block = ty.ends_with("::Fp8BlockLinear")
            || ty == "Fp8BlockLinear"
            || ty.ends_with("layers::Fp8BlockLinear");
        for (wid, _idx) in &a.source_weights {
            let fmt = crate::quantization::storage_format_for_weight(program, fuf, *wid, model);
            let ok = matches!(
                (
                    &fmt,
                    accessor_is_marlin,
                    accessor_is_bnb4,
                    accessor_is_fp8,
                    accessor_is_fp8_block,
                ),
                (
                    crate::quantization::StorageFormat::Dense,
                    false,
                    false,
                    false,
                    false,
                ) | (
                    crate::quantization::StorageFormat::Awq { .. },
                    true,
                    false,
                    false,
                    false,
                ) | (
                    crate::quantization::StorageFormat::Gptq { .. },
                    true,
                    false,
                    false,
                    false,
                ) | (
                    crate::quantization::StorageFormat::Bnb4 { .. },
                    false,
                    true,
                    false,
                    false,
                ) | (
                    crate::quantization::StorageFormat::Fp8 {
                        block_size: None,
                        ..
                    },
                    false,
                    false,
                    true,
                    false,
                ) | (
                    crate::quantization::StorageFormat::Fp8 {
                        block_size: Some(_),
                        ..
                    },
                    false,
                    false,
                    false,
                    true,
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
                return (
                    quote! { compile_error!(#msg); },
                    crate::emit::WeightLayout::new(),
                );
            }
        }
    }

    // Detect array-collapsible families. Two accessors are in the
    // same family iff they share a `family.stem`, their `rust_type`
    // stringifies the same, AND their `FieldLoad` plan shares a
    // std::mem::discriminant. A family qualifies for array collapse
    // only when its members' `family.idx` values form exactly
    // `0..N` for some `N` ≥ 1 — any gap or duplicate falls back to
    // flat fields. `plans` is computed a few lines below, so we do
    // detection in two stages: first partition by stem+type, then
    // (after `plans`) filter on discriminant uniformity + index
    // coverage.
    let plans_for_family: Vec<FieldLoad> = accessors
        .iter()
        .map(|a| plan_field_load(a, program, fuf, model, manifest))
        .collect();

    // `family_of[i] = Some(group_idx)` when accessor i participates
    // in a qualifying family; None means "stays a flat field".
    let mut family_groups: Vec<WeightFamilyGroup> = Vec::new();
    let mut family_of: Vec<Option<usize>> = vec![None; accessors.len()];
    {
        // Group accessor indices by their declared family stem.
        let mut by_stem: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, a) in accessors.iter().enumerate() {
            if let Some((stem, _)) = &a.family {
                by_stem.entry(stem.to_string()).or_default().push(i);
            }
        }
        for (stem_str, indices) in by_stem {
            if indices.len() < 2 {
                continue;
            }
            let first = &accessors[indices[0]];
            let first_ty = first.rust_type.to_string();
            let first_disc = std::mem::discriminant(&plans_for_family[indices[0]]);
            let type_ok = indices
                .iter()
                .all(|&i| accessors[i].rust_type.to_string() == first_ty);
            let disc_ok = indices
                .iter()
                .all(|&i| std::mem::discriminant(&plans_for_family[i]) == first_disc);
            if !type_ok || !disc_ok {
                continue;
            }
            // Collect (idx, accessor_pos) pairs, sort by idx,
            // require the idx sequence == 0..len.
            let mut pairs: Vec<(u64, usize)> = indices
                .iter()
                .map(|&i| (accessors[i].family.as_ref().unwrap().1, i))
                .collect();
            pairs.sort_by_key(|(idx, _)| *idx);
            let consecutive = pairs
                .iter()
                .enumerate()
                .all(|(k, (idx, _))| *idx as usize == k);
            if !consecutive {
                continue;
            }
            let group_idx = family_groups.len();
            let members_accessor_positions: Vec<usize> =
                pairs.iter().map(|(_, pos)| *pos).collect();
            for &pos in &members_accessor_positions {
                family_of[pos] = Some(group_idx);
            }
            let stem_ident = first.family.as_ref().unwrap().0.clone();
            family_groups.push(WeightFamilyGroup {
                stem: stem_ident,
                rust_type: first.rust_type.clone(),
                len: members_accessor_positions.len(),
                member_accessor_positions: members_accessor_positions,
                _stem_str: stem_str,
            });
        }
    }

    // Build the read-rewrite layout: family members map to
    // `#stem[#idx usize]`; orphans are absent (caller falls back
    // to the flat ident).
    let mut weight_layout = crate::emit::WeightLayout::new();
    for group in &family_groups {
        let stem = &group.stem;
        for (idx_in_group, &acc_pos) in group.member_accessor_positions.iter().enumerate() {
            let flat_name = accessors[acc_pos].name.to_string();
            let idx_lit = proc_macro2::Literal::usize_unsuffixed(idx_in_group);
            weight_layout.insert_array_access(&flat_name, quote! { #stem[#idx_lit] });
            weight_layout.insert_family_stem(&flat_name, stem.clone());
        }
    }

    // Emit struct fields: one `pub #stem: [#ty; N]` per family
    // group, one `pub #name: #ty` per orphan. Orphans keep their
    // ordering relative to `accessors`; families emit once at
    // their first member's position so deterministic struct order
    // is preserved across rebuilds.
    let mut emitted_family: Vec<bool> = vec![false; family_groups.len()];
    let mut fields_vec: Vec<TokenStream> = Vec::with_capacity(accessors.len());
    for (i, a) in accessors.iter().enumerate() {
        match family_of[i] {
            Some(g) => {
                if !emitted_family[g] {
                    emitted_family[g] = true;
                    let group = &family_groups[g];
                    let stem = &group.stem;
                    let ty = &group.rust_type;
                    let n = group.len;
                    let n_lit = proc_macro2::Literal::usize_unsuffixed(n);
                    fields_vec.push(quote! { pub #stem: [#ty; #n_lit], });
                }
            }
            None => {
                let name = &a.name;
                let ty = &a.rust_type;
                fields_vec.push(quote! { pub #name: #ty, });
            }
        }
    }
    let fields = fields_vec.iter();

    // Emit each field as its own let-binding in the load body.
    // This lets later loaders reference earlier ones (e.g. a tied
    // `lm_head` reads `embed_tokens.weight`). Field order inside
    // Self { .. } is irrelevant to Rust; let-binding order is what
    // matters. `accessors` iterates BTreeMap-sorted — which puts
    // `embed_tokens` before `lm_head` alphabetically, so the tied
    // case works without a special sort.
    let plans = plans_for_family;
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
    let lets: Vec<TokenStream> = accessors
        .iter()
        .zip(plans.iter())
        .map(|(a, plan)| {
            let name = &a.name;
            match plan {
                FieldLoad::Embedding(prefix) => quote! {
                    let #name = ::ferrite_kernels::layers::Embedding::load(gw, #prefix)?;
                },
                FieldLoad::RmsNorm(prefix, eps) => quote! {
                    let #name = ::ferrite_kernels::layers::RmsNorm::load(gw, #prefix, #eps)?;
                },
                FieldLoad::CohereLayerNorm(prefix, eps) => quote! {
                    let #name = ::ferrite_kernels::layers::CohereLayerNorm::load(
                        gw, #prefix, #eps,
                    )?;
                },
                FieldLoad::LinearDense(prefix) => quote! {
                    let #name = ::ferrite_kernels::layers::LinearLayer::load_dense(gw, #prefix)?;
                },
                FieldLoad::LinearConcat(prefixes) => quote! {
                    let #name = ::ferrite_kernels::layers::LinearLayer::load_dense_concat(
                        gw,
                        &[ #(#prefixes),* ],
                        stream,
                    )?;
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
                    // Every Marlin accessor emits the SAME call
                    // shape regardless of AWQ/GPTQ/CT: the runtime
                    // `MarlinFormat` discriminator is threaded in
                    // from `load_with`'s `marlin_storage` param.
                    // That's what lets cross-variant load-body
                    // dedup collapse AWQ/GPTQ/CT variants of the
                    // same (arch, size) to one canonical `load_with`
                    // body — only the `marlin_storage` const each
                    // variant's one-line `load` passes differs.
                    let single = prefixes.len() == 1;
                    if single {
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
                            let #name = ::ferrite_kernels::layers::Fp8Linear::load(
                                gw,
                                #prefix,
                                __fp8_dtype,
                            )?;
                        }
                    } else {
                        quote! {
                            let #name = ::ferrite_kernels::layers::Fp8Linear::load_concat(
                                gw,
                                &[ #(#prefixes),* ],
                                __fp8_dtype,
                            )?;
                        }
                    }
                }
                FieldLoad::Fp8BlockLinear { prefixes } => {
                    if prefixes.len() == 1 {
                        let prefix = &prefixes[0];
                        quote! {
                            let #name = ::ferrite_kernels::layers::Fp8BlockLinear::load(
                                gw,
                                #prefix,
                                __fp8_dtype,
                            )?;
                        }
                    } else {
                        quote! {
                            let #name = ::ferrite_kernels::layers::Fp8BlockLinear::load_concat(
                                gw,
                                &[ #(#prefixes),* ],
                                __fp8_dtype,
                            )?;
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
            }
        })
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
                .tensor_info("model.embed_tokens.weight")
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
                .tensor_info("model.embed_tokens.weight")
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
    let packed_splits_prelude: TokenStream = if manifest.packed_splits.is_empty() {
        quote! {}
    } else {
        let num_hidden_layers = *model.bounds.get("num_hidden_layers").unwrap_or_else(|| {
            panic!(
                "model `{}` has `__packed_splits__` but no `num_hidden_layers` bound",
                model.source_stem,
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
            calls.push(quote! {
                for __l in 0..#num_hidden_layers {
                    let __pp = ::std::format!("model.layers.{}.{}", __l, #packed_prefix);
                    gw.synthesize_packed_row_split_sizes(&__pp, &[ #(#pairs),* ])?;
                }
            });
        }
        quote! {
            #(#calls)*
        }
    };

    // Array-assembly prelude: for each family group, consume the
    // per-layer `let #member_flat_name = ...` bindings produced by
    // `lets` into a `let #stem = [#m0, #m1, …, #m_{N-1}];` array
    // binding. Placed between `lets` and the `Self { .. }`
    // constructor so the constructor can reference `#stem` via
    // shorthand without also carrying the flat names.
    let family_array_assemblies: Vec<TokenStream> = family_groups
        .iter()
        .map(|group| {
            let stem = &group.stem;
            let member_idents: Vec<&syn::Ident> = group
                .member_accessor_positions
                .iter()
                .map(|&i| &accessors[i].name)
                .collect();
            quote! { let #stem = [ #(#member_idents),* ]; }
        })
        .collect();

    // `Self { … }` shorthand — one entry per array family (using
    // the family stem) plus one per orphan (using the flat name),
    // emitted in `accessors` order so the struct definition and
    // the constructor stay in lockstep.
    let mut seen_family: Vec<bool> = vec![false; family_groups.len()];
    let field_shorthand: Vec<syn::Ident> = accessors
        .iter()
        .enumerate()
        .filter_map(|(i, a)| match family_of[i] {
            Some(g) if !seen_family[g] => {
                seen_family[g] = true;
                Some(family_groups[g].stem.clone())
            }
            Some(_) => None,
            None => Some(a.name.clone()),
        })
        .collect();
    let fingerprint_method = emit_fingerprint_check(model, manifest);

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
                    ::ferrite_cuda_core::dtype::DType::BF16,
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
        let scaling = model.rope_scaling.clone();
        let body = match (rotary_dim_lit, scaling) {
            (None, None) => quote! {
                ::ferrite_kernels::rotary::RotaryCache::new_from_stream(
                    #head_dim,
                    #max_pos,
                    #rope_theta,
                    None,
                    ::ferrite_cuda_core::dtype::DType::BF16,
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
                        ::ferrite_cuda_core::dtype::DType::BF16,
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
                        ::ferrite_cuda_core::dtype::DType::BF16,
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
                    ::ferrite_cuda_core::dtype::DType::BF16,
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
                        ::ferrite_cuda_core::dtype::DType::BF16,
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
                        ::ferrite_cuda_core::dtype::DType::BF16,
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

    // Struct definition vs type alias per emit mode.
    let weights_def: TokenStream = match &mode {
        WeightsEmitMode::Canonical => quote! {
            /// Every weight the emitted forward needs, already
            /// packed exactly how the solver-picked Impls want to
            /// see it. Construct via the sibling free `load` fn.
            #[cfg(feature = "cuda")]
            pub struct Weights {
                #(#fields)*
                #rotary_field
                #rotary_local_field
            }
        },
        WeightsEmitMode::Shim { canonical } => quote! {
            /// This variant's emitted forward + load bodies are
            /// byte-identical to the canonical sibling's (same
            /// solver-picked `Impl` set → same emit, and Marlin
            /// quant format threads through at runtime via
            /// `load_with`'s `marlin_storage` param). We share
            /// the canonical's `Weights` via type alias; per-
            /// variant state is just `load` (a one-liner
            /// calling `canonical::load_with(MY_MARLIN_FORMAT)`)
            /// and `fingerprint_matches`.
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
    let tokens = match &mode {
        WeightsEmitMode::Canonical => quote! {
            #weights_def

            #fingerprint_method

            /// Read every field from an open `GpuWeights` (a
            /// safetensors view). The canonical per-equivalence-
            /// class load body, parameterized on
            /// `marlin_storage` so AWQ/GPTQ/CT variants share
            /// one compiled copy. Non-Marlin equivalence classes
            /// ignore the param; it's still threaded for uniform
            /// signature across `load_with` across archs.
            #[cfg(feature = "cuda")]
            #[allow(clippy::too_many_lines, clippy::not_unsafe_ptr_arg_deref, unused_variables)]
            pub fn load_with(
                gw: &mut ::ferrite_cuda_core::weights::GpuWeights,
                stream: ::ferrite_cuda_core::CUstream,
                max_model_len: usize,
                marlin_storage: ::ferrite_kernels::layers_quant::MarlinFormat,
            ) -> ::anyhow::Result<Weights> {
                #packed_splits_prelude
                #marlin_prelude
                #bnb4_prelude
                #fp8_prelude
                #(#lets)*
                #(#family_array_assemblies)*
                #rotary_load
                #rotary_local_load
                Ok(#weights_ctor {
                    #(#field_shorthand,)*
                    #rotary_init
                    #rotary_local_init
                })
            }

            /// Variant-facing entry point. Threads this compiled
            /// variant's `MarlinFormat` into the shared
            /// `load_with` body. rustc inlines this wrapper
            /// trivially; no codegen overhead.
            #[cfg(feature = "cuda")]
            #[inline]
            #[allow(clippy::not_unsafe_ptr_arg_deref)]
            pub fn load(
                gw: &mut ::ferrite_cuda_core::weights::GpuWeights,
                stream: ::ferrite_cuda_core::CUstream,
                max_model_len: usize,
            ) -> ::anyhow::Result<Weights> {
                load_with(gw, stream, max_model_len, #marlin_fmt)
            }
        },
        WeightsEmitMode::Shim { canonical } => quote! {
            #weights_def

            #fingerprint_method

            /// Shim loader — single-line call-through to the
            /// canonical sibling's `load_with` with this
            /// variant's `MarlinFormat` threaded in. No body
            /// emit; rustc compiles the canonical's `load_with`
            /// once and every shim in the equivalence class
            /// shares it.
            #[cfg(feature = "cuda")]
            #[inline]
            #[allow(clippy::not_unsafe_ptr_arg_deref)]
            pub fn load(
                gw: &mut ::ferrite_cuda_core::weights::GpuWeights,
                stream: ::ferrite_cuda_core::CUstream,
                max_model_len: usize,
            ) -> ::anyhow::Result<Weights> {
                super::#canonical::load_with(gw, stream, max_model_len, #marlin_fmt)
            }
        },
    };
    (tokens, weight_layout)
}

/// A set of accessors whose flat fields `<stem>_<0..N>` collapse
/// to one `pub #stem: [#ty; N]` array field on the emitted
/// `Weights` struct. Detected by
/// [`emit_weights_struct`]'s family-partitioning pass using each
/// accessor's `family` tag plus type/`FieldLoad`-discriminant
/// uniformity; read sites consult the accompanying
/// [`WeightLayout`](crate::emit::WeightLayout) to rewrite
/// `wm.<flat_name>` into `wm.<stem>[<idx>]`.
struct WeightFamilyGroup {
    stem: syn::Ident,
    rust_type: TokenStream,
    len: usize,
    /// Positions into the shared `accessors` Vec, sorted by
    /// `family.idx` so array assembly order is deterministic and
    /// `layout[member_i] = #stem[#i]` matches.
    member_accessor_positions: Vec<usize>,
    /// Kept for debug-print/diag purposes only; the stem ident is
    /// the authoritative form used in emitted code.
    #[allow(dead_code)]
    _stem_str: String,
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

/// For each `OwnedTensor` tile-local that the forward fn binds, the
/// subgraph after which it can safely be dropped (its last
/// cross-subgraph use, with alias chains followed back to the
/// underlying owner). Produced by `compute_drops_after`, consumed by
/// the per-bucket emitters to inject `drop(t_X_Y);` in the right spot.
type DropPlan = HashMap<SubgraphId, Vec<(TileId, u8)>>;

/// Compute when each owning `OwnedTensor` tile-local can be dropped.
///
/// Walks every subgraph's `output_alias` declaration to build the
/// alias map: `(tile, slot) → Some(upstream)` (alias) or
/// `(tile, slot) → None` (owner). Outputs absent from the map are
/// treated as untracked (e.g. paged-cache views) and ignored.
///
/// For each cross-subgraph tile-input, resolves the consumed tile
/// through the alias chain to its underlying owner, and records the
/// latest subgraph that touches it. Returns a map keyed by that
/// subgraph: after emitting it, drop the listed tile-locals.
///
/// `skip_subgraph` is the terminal subgraph excluded from the
/// backbone forward (`forward_backbone`); its consumptions are
/// ignored so we don't keep tile-locals alive past the backbone's
/// real last use. `protected` are owners that must NEVER be dropped
/// (the function's return tile, and for backbone, the backbone-output
/// tile that the caller clones out).
fn compute_drops_after(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    lib: &ImplementationLibrary,
    skip_subgraph: Option<SubgraphId>,
    protected: &HashSet<(TileId, u8)>,
) -> DropPlan {
    // Per-subgraph topological order (subgraphs within the same wave
    // are unordered relative to each other in the LOOP, but for
    // single-wave-per-subgraph graphs that doesn't matter; for the
    // general case we treat their order in `wave.subgraphs` as
    // authoritative).
    let mut order: HashMap<SubgraphId, usize> = HashMap::new();
    let mut next = 0;
    for wave in &loop_ir.waves {
        for (sg, _) in &wave.subgraphs {
            order.insert(*sg, next);
            next += 1;
        }
    }

    // alias: (tile, slot) → Some(upstream) means this output is a
    // TensorView aliasing upstream's OwnedTensor; None means this
    // output IS the owner. Outputs absent are untracked.
    //
    // consumed: upstream (tile, slot)s that some impl moves into its
    // own output binding — the upstream local is no longer accessible
    // after that subgraph, so we must never emit a `drop()` for it.
    let mut alias: HashMap<(TileId, u8), Option<(TileId, u8)>> = HashMap::new();
    let mut consumed: HashSet<(TileId, u8)> = HashSet::new();
    for wave in &loop_ir.waves {
        for (sg, imp_id) in &wave.subgraphs {
            if Some(*sg) == skip_subgraph {
                continue;
            }
            let claimed = sfuf.tiles_in_subgraph(*sg);
            let imp = lib.get(*imp_id);
            for (k, v) in imp.output_alias(&claimed, fuf) {
                alias.insert(k, v);
            }
            for upstream in imp.consumes_input_tiles(&claimed, fuf) {
                consumed.insert(upstream);
            }
        }
    }

    // Resolve a (tile, slot) to its underlying owner, or `None` if
    // untracked / aliases extern memory. Cycle-safe via a small set.
    let resolve = |start: (TileId, u8)| -> Option<(TileId, u8)> {
        let mut cur = start;
        let mut seen = HashSet::new();
        loop {
            if !seen.insert(cur) {
                return None;
            }
            match alias.get(&cur) {
                Some(None) => return Some(cur),
                Some(Some(up)) => cur = *up,
                None => return None,
            }
        }
    };

    // For each owner, the latest subgraph that touches it (directly
    // or via an alias).
    let mut last_use: HashMap<(TileId, u8), SubgraphId> = HashMap::new();
    for wave in &loop_ir.waves {
        for (sg, _) in &wave.subgraphs {
            if Some(*sg) == skip_subgraph {
                continue;
            }
            let claimed: HashSet<TileId> = sfuf.tiles_in_subgraph(*sg).into_iter().collect();
            for tile in &claimed {
                for input in &fuf.get(*tile).inputs {
                    if let FufInput::Tile { id, slot } = input {
                        // Intra-subgraph consumption is invisible at
                        // codegen — handled inside emit_call.
                        if claimed.contains(id) {
                            continue;
                        }
                        if let Some(owner) = resolve((*id, *slot)) {
                            let new_pos = order[sg];
                            let keep = match last_use.get(&owner) {
                                Some(prev) => order[prev] < new_pos,
                                None => true,
                            };
                            if keep {
                                last_use.insert(owner, *sg);
                            }
                        }
                    }
                }
            }
        }
    }

    let mut plan: DropPlan = HashMap::new();
    for (owner, sg) in last_use {
        if protected.contains(&owner) {
            continue;
        }
        // If some impl moves this owner into its own output, the
        // owner local is gone after that subgraph — don't drop it.
        if consumed.contains(&owner) {
            continue;
        }
        plan.entry(sg).or_default().push(owner);
    }
    // Determinism — owners within a subgraph drop in stable order.
    for v in plan.values_mut() {
        v.sort();
    }
    plan
}

/// Allocate the stable local-binding ident per tile-output slot
/// used by every per-bucket emission.
fn build_local_map(fuf: &Fuf) -> LocalMap {
    let mut locals: LocalMap = HashMap::new();
    for node in &fuf.nodes {
        for slot in 0..node.outputs.len().max(1) as u8 {
            locals.insert((node.id, slot), format_ident!("t_{}_{}", node.id.0, slot));
        }
    }
    locals
}

/// Per-model library of deduplicated kernel-call fragments.
///
/// Each entry is a private `unsafe fn __frag_<N>(...)` emitted
/// inside the model's module. The forward body (and the backbone
/// body) replaces its inline kernel calls with one-line
/// `let (t_a, t_b) = unsafe { __frag_N(tile_inputs, weight_refs,
/// wm, ctx, device) };` dispatches against this library. Two
/// subgraphs whose abstract-mode emit_call output stringifies
/// identically share a fragment — the critical win is layer-level
/// sharing inside a single model (one rmsnorm body emitted once,
/// called N times for an N-layer transformer).
///
/// Fragment fn body = `emit_call` output in `EmitMode::Abstract`
/// mode, where tile/weight references are replaced by fn-param
/// idents (`input_<i>`, `w_<i>`). Extern references like
/// `wm.rotary_local` / `ctx.input_ids` stay verbatim and are
/// resolved via the `wm` / `ctx` params the fragment takes;
/// they're consistent across call sites within a model, so they
/// don't defeat dedup.
#[derive(Default)]
struct FragmentLibrary {
    /// Stringified abstract-body signature → fragment index.
    by_sig: HashMap<String, usize>,
    /// Fragment fn tokens, in insertion order. Index matches the
    /// N in `__frag_N`.
    fns: Vec<TokenStream>,
}

/// Diagnostic counters for dedup effectiveness. Reset after each
/// per-model emit_model print so the output shows per-model stats.
static CALL_SITE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static INLINE_FALLBACK_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

impl FragmentLibrary {
    fn into_fns(self) -> Vec<TokenStream> {
        self.fns
    }
}

/// Per-workload stencil analysis — the class-structure view of the
/// SFUF that the Rust codegen pivot (STENCIL_IR_V2_DESIGN.md §8.1)
/// consumes.
///
/// One bundle per workload bucket. Carries enough to rewrite
/// per-subgraph emission into per-class emission:
/// - `class_of`: SubgraphId → class_idx, so `emit_subgraph` can look
///   up its class in O(1).
/// - `class_impl_id`: class_idx → `Some(impl_id)` when every member
///   of the class picked the same impl (homogeneous — the codegen
///   pivot collapses to one fragment + loop); `None` when impls
///   differ (heterogeneous — Qwen3 A.2 case, falls back to
///   per-subgraph emission).
/// - `class_members`: class_idx → `Vec<SubgraphId>` in period order,
///   source of the loop bound and per-iteration state (layer index,
///   kv_cache slot).
///
/// Computed but not yet consumed — plumbing first (task 1 of
/// STENCIL_IR_V2_DESIGN.md §9 step 6). The downstream emission
/// rewrites (tasks 2-4) read this bundle instead of stringifying
/// abstract bodies for dedup.
#[derive(Debug, Clone)]
struct StencilBundle {
    /// SubgraphId → class_idx. Every subgraph the assignment knows
    /// about is present. Built from `periodicity::group_regions` +
    /// `region_formation::form_regions`.
    class_of: HashMap<SubgraphId, usize>,
    /// class_idx → homogeneous impl (or `None` if heterogeneous).
    /// Parallel to `class_members`.
    class_impl_id: Vec<Option<ImplId>>,
    /// class_idx → subgraphs in the class, in Region-id order
    /// (matches the order periodicity hashing groups them). The
    /// length of this vec is the `repeat` axis bound for the class.
    class_members: Vec<Vec<SubgraphId>>,
}

impl StencilBundle {
    fn compute(fuf: &Fuf, sfuf: &Assignment) -> Self {
        let st = crate::subtile::subtile(fuf);
        let fr = crate::region_formation::form_regions(&st, sfuf);
        let classes = crate::periodicity::group_regions(&fr.graph);
        let report = crate::class_impl::check_class_impls(&classes, &fr.region_subgraphs, sfuf);

        let mut class_of: HashMap<SubgraphId, usize> = HashMap::new();
        let mut class_members: Vec<Vec<SubgraphId>> = Vec::with_capacity(classes.len());
        for (idx, class) in classes.iter().enumerate() {
            let mut members: Vec<SubgraphId> = Vec::with_capacity(class.members.len());
            for &rid in &class.members {
                let sg = fr.region_subgraphs[rid as usize];
                class_of.insert(sg, idx);
                members.push(sg);
            }
            class_members.push(members);
        }

        let class_impl_id: Vec<Option<ImplId>> = report
            .per_class
            .iter()
            .map(|picks| match picks.len() {
                1 => Some(picks[0]),
                _ => None,
            })
            .collect();

        let bundle = StencilBundle {
            class_of,
            class_impl_id,
            class_members,
        };
        bundle.refine_by_edge_pattern(fuf, sfuf)
    }

    /// Iteratively split classes whose members observe different
    /// Δrepeat patterns to a shared class — the precise condition
    /// that drives `non_uniform_pairs` in the edge summary.
    ///
    /// Targeted refinement: the region-level 1-hop neighbour hash
    /// in `group_regions` over-collapses some patterns (e.g. two
    /// residual Add tiles per transformer layer, or two per-layer
    /// RmsNorms, hashing to one class with period `2 × num_layers`).
    /// That over-collapse shows up as at least one class-pair
    /// (C, P) with multiple distinct Δrepeat values. For each such
    /// pair, C's members (or P's) must be distinguishable by Δ,
    /// otherwise the pair couldn't carry multiple Δs.
    ///
    /// Per-member signature: for every non-uniform pair touching
    /// the member's class, the sorted multiset of
    /// `(role [0=consumer-of-pair, 1=producer-of-pair], Δrepeat)`.
    /// Uniform pairs are excluded from the signature — splitting on
    /// those would fracture legitimate residual-stream classes
    /// whose only asymmetry is "iter 0 reads pre-loop, iter ≥1
    /// reads a carry", which the emitter's LoopCarry + PreLoop
    /// provenance already handles.
    ///
    /// Partition-refinement: each iteration splits a class by
    /// signature (monotone); stop when no split happens.
    /// Re-computation of edges + pair uniformity after each split
    /// is necessary because a class split may turn a previously-
    /// non-uniform pair uniform (removing it from the signature)
    /// or reveal fresh non-uniformity as producer classes
    /// redistribute.
    #[allow(clippy::type_complexity)]
    fn refine_by_edge_pattern(mut self, fuf: &Fuf, sfuf: &Assignment) -> Self {
        loop {
            let edges = self.class_edges(fuf, sfuf);

            // Collect distinct Δs per (consumer_class, producer_class)
            // pair — the raw data the emitter's `uniform_pairs` check
            // runs on. Pairs with |Δ set| ≥ 2 are the refinement
            // offenders.
            let mut pair_deltas: BTreeMap<(usize, usize), BTreeSet<i64>> = BTreeMap::new();
            for e in &edges {
                pair_deltas
                    .entry((e.consumer_class, e.producer_class))
                    .or_default()
                    .insert(e.delta_repeat());
            }
            let non_uniform_pairs: BTreeSet<(usize, usize)> = pair_deltas
                .iter()
                .filter(|(_, deltas)| deltas.len() >= 2)
                .map(|(k, _)| *k)
                .collect();
            if non_uniform_pairs.is_empty() {
                break self;
            }

            // Per-member signature: for each non-uniform pair the
            // member participates in, `(pair_index, role, edge_count)`.
            // Role 0 = member is the consumer side of the pair, 1 =
            // producer side. We count edges per (pair, role) rather
            // than carrying Δrepeat directly: Δ-based signatures
            // cascade-atomise when a producer class's split is what
            // will eventually make the consumer's Δs uniform (e.g.
            // qwen3's period-56 class 6 feeds period-28 class 7 with
            // one edge per consumer at unique Δs — splitting by Δ on
            // the consumer side over-splits it into 28 groups; the
            // right fix is to split the producer first by "has edge
            // / no edge" to the consumer, then the consumer's Δs
            // naturally become uniform on re-compute).
            //
            // The edge-count signature still discriminates cleanly
            // for the motivating case: for producer class P whose
            // members' out-edges to consumer C are interleaved, half
            // have count=1 and half have count=0 → 2 sub-classes.
            // Once P splits, the next iteration re-evaluates and
            // typically converges fast.
            let pair_index: BTreeMap<(usize, usize), usize> = non_uniform_pairs
                .iter()
                .enumerate()
                .map(|(i, k)| (*k, i))
                .collect();

            // Per-member-index-within-class. `iter_of` recomputes
            // from `class_members` order, which is what we set
            // below, so it's the right invariant for this
            // refinement pass.
            let iter_of: HashMap<SubgraphId, usize> = self
                .class_of
                .keys()
                .map(|&sg| (sg, self.iter_of(sg)))
                .collect();

            // Precompute period-ratio for each non-uniform pair
            // where p_period % c_period == 0: this splits
            // downsampling-style producer classes by
            // (producer_iter % ratio). For qwen3's 56→28 pair it
            // yields 2 sub-classes of 28 each, matching the
            // within-period position (e.g. Q-norm vs K-norm).
            let pair_ratio: BTreeMap<usize, usize> = non_uniform_pairs
                .iter()
                .enumerate()
                .filter_map(|(i, &(c, p))| {
                    let cp = self.class_members[c].len();
                    let pp = self.class_members[p].len();
                    if cp > 0 && pp > 0 && pp.is_multiple_of(cp) && pp / cp >= 2 {
                        Some((i, pp / cp))
                    } else {
                        None
                    }
                })
                .collect();

            // Signature value per (member, pair, role) is a (count,
            // mod_group) tuple. `mod_group = 0` for roles where the
            // ratio doesn't apply (consumer side, or non-integer-
            // ratio pair). Producer side with an applicable ratio
            // gets `member_iter % ratio`, so downsampling patterns
            // split by within-period position even when every
            // producer has the same edge count.
            let mut sigs: HashMap<SubgraphId, BTreeMap<(usize, u8), (u32, u32)>> = HashMap::new();
            for e in &edges {
                let pair = (e.consumer_class, e.producer_class);
                let Some(&pi) = pair_index.get(&pair) else {
                    continue;
                };
                // Consumer side — no mod split.
                {
                    let entry = sigs
                        .entry(e.consumer_sg)
                        .or_default()
                        .entry((pi, 0))
                        .or_insert((0, 0));
                    entry.0 += 1;
                }
                // Producer side — add mod_group when applicable.
                let mod_group = pair_ratio
                    .get(&pi)
                    .map(|&r| (iter_of[&e.producer_sg] % r) as u32)
                    .unwrap_or(0);
                {
                    let entry = sigs
                        .entry(e.producer_sg)
                        .or_default()
                        .entry((pi, 1))
                        .or_insert((0, mod_group));
                    entry.0 += 1;
                    // If two edges disagree on mod_group (shouldn't
                    // happen — mod_group depends only on the
                    // producer's own iter), keep the first.
                }
            }

            // Group each class's members by signature. More than one
            // group = split.
            let mut new_members: Vec<Vec<SubgraphId>> = Vec::new();
            let mut any_split = false;
            for members in &self.class_members {
                let mut groups: BTreeMap<Vec<((usize, u8), (u32, u32))>, Vec<SubgraphId>> =
                    BTreeMap::new();
                for &sg in members {
                    let sig: Vec<((usize, u8), (u32, u32))> = sigs
                        .get(&sg)
                        .map(|m| m.iter().map(|(k, v)| (*k, *v)).collect())
                        .unwrap_or_default();
                    groups.entry(sig).or_default().push(sg);
                }
                if groups.len() > 1 {
                    any_split = true;
                }
                for (_sig, g) in groups {
                    new_members.push(g);
                }
            }
            if !any_split {
                break self;
            }

            // Sort new classes by the minimum SubgraphId in each for
            // deterministic iteration.
            new_members.sort_by_key(|m| m.iter().copied().min().unwrap_or(SubgraphId(u32::MAX)));

            // Rebuild class_of + class_impl_id.
            let mut new_class_of: HashMap<SubgraphId, usize> = HashMap::new();
            let mut new_class_impl_id: Vec<Option<ImplId>> = Vec::with_capacity(new_members.len());
            for (idx, members) in new_members.iter().enumerate() {
                let mut impls: Vec<ImplId> =
                    members.iter().filter_map(|&sg| sfuf.impl_of(sg)).collect();
                impls.sort();
                impls.dedup();
                new_class_impl_id.push(if impls.len() == 1 {
                    Some(impls[0])
                } else {
                    None
                });
                for &sg in members {
                    new_class_of.insert(sg, idx);
                }
            }
            self = StencilBundle {
                class_of: new_class_of,
                class_impl_id: new_class_impl_id,
                class_members: new_members,
            };
        }
    }

    /// Count of classes whose members share one impl. Diagnostic
    /// only — useful to confirm the bundle matches the lib.rs
    /// stencil print.
    fn homogeneous_count(&self) -> usize {
        self.class_impl_id.iter().filter(|p| p.is_some()).count()
    }

    /// Iteration index of `sg` within its own class — the position
    /// in `class_members[class_of[sg]]`. Matches the `__repeat`
    /// value that the collapsed-mode emitter will bind for this
    /// subgraph's iteration.
    fn iter_of(&self, sg: SubgraphId) -> usize {
        let class = *self
            .class_of
            .get(&sg)
            .expect("every subgraph is in some class");
        self.class_members[class]
            .iter()
            .position(|&s| s == sg)
            .expect("class_members is a permutation of subgraphs")
    }

    /// Tile-input edges between subgraphs, classified by
    /// (consumer_class, producer_class, Δrepeat). The emitter uses
    /// this to (a) topo-sort classes within one iteration
    /// (Δrepeat = 0 edges), (b) materialise loop-carried state for
    /// intra-class-pair edges with Δrepeat ≠ 0, (c) hoist pre-loop
    /// inputs when the producer class is period-1 and its only
    /// member lives outside the loop.
    ///
    /// Walks every claimed tile's inputs — O(total tile-input
    /// count). Returns a flat edge list so the caller can bucket
    /// any way it likes; higher-level summaries (per class-pair
    /// Δrepeat table, non-uniform pair detection) layer on top.
    fn class_edges(&self, fuf: &Fuf, sfuf: &Assignment) -> Vec<ClassEdge> {
        let mut edges: Vec<ClassEdge> = Vec::new();
        for (&consumer_sg, &consumer_class) in &self.class_of {
            let consumer_iter = self.iter_of(consumer_sg);
            for consumer_tile in sfuf.tiles_in_subgraph(consumer_sg) {
                for input in &fuf.get(consumer_tile).inputs {
                    let FufInput::Tile {
                        id: producer_tile, ..
                    } = input
                    else {
                        continue;
                    };
                    let Some(producer_sg) = sfuf.subgraph_of(*producer_tile) else {
                        continue;
                    };
                    if producer_sg == consumer_sg {
                        continue;
                    }
                    let Some(&producer_class) = self.class_of.get(&producer_sg) else {
                        continue;
                    };
                    let producer_iter = self.iter_of(producer_sg);
                    edges.push(ClassEdge {
                        consumer_class,
                        consumer_iter,
                        producer_class,
                        producer_iter,
                        consumer_sg,
                        producer_sg,
                    });
                }
            }
        }
        edges
    }
}

/// One tile-input edge between two subgraphs in distinct classes,
/// tagged with each side's iteration index. `Δrepeat` (the number
/// of outer-loop iterations separating consumer from producer) is
/// `consumer_iter - producer_iter`; `= 0` for intra-iteration edges
/// (topo within one loop body), `> 0` for loop-carried edges
/// (previous iteration's result), `< 0` would indicate a
/// back-edge that the loop can't respect without restructuring —
/// should never occur for a well-formed forward pass.
#[derive(Debug, Clone, Copy)]
struct ClassEdge {
    consumer_class: usize,
    consumer_iter: usize,
    producer_class: usize,
    producer_iter: usize,
    consumer_sg: SubgraphId,
    producer_sg: SubgraphId,
}

impl ClassEdge {
    fn delta_repeat(&self) -> i64 {
        self.consumer_iter as i64 - self.producer_iter as i64
    }
}

/// Diagnostic summary of the class-edge structure for one SFUF:
/// how many distinct Δrepeat values appear, histogram of edge
/// counts per Δrepeat, and count of class pairs with non-uniform
/// Δrepeat (shouldn't happen for well-formed forwards). Format
/// intended to land next to the `stencil ·` line in build output.
pub(crate) fn summarize_class_edges(fuf: &Fuf, sfuf: &Assignment) -> String {
    let bundle = StencilBundle::compute(fuf, sfuf);
    let edges = bundle.class_edges(fuf, sfuf);

    // Histogram of Δrepeat.
    let mut histo: BTreeMap<i64, usize> = BTreeMap::new();
    for e in &edges {
        *histo.entry(e.delta_repeat()).or_insert(0) += 1;
    }

    // Non-uniform class pairs: a pair (consumer_class,
    // producer_class) is uniform iff every member pair with an
    // edge shares one Δrepeat. Non-uniform means the collapse is
    // lossy for that pair — codegen has to fall back to unrolled
    // emission for it (or refuse the collapse for that edge).
    let mut pair_deltas: BTreeMap<(usize, usize), BTreeSet<i64>> = BTreeMap::new();
    for e in &edges {
        pair_deltas
            .entry((e.consumer_class, e.producer_class))
            .or_default()
            .insert(e.delta_repeat());
    }
    let non_uniform_pairs = pair_deltas.values().filter(|set| set.len() > 1).count();

    let histo_str: String = histo
        .iter()
        .map(|(d, n)| format!("Δ{d}={n}"))
        .collect::<Vec<_>>()
        .join(" ");

    // When any negative Δ shows up, it means a consumer at iter i
    // reads from a producer at iter > i — either the class member
    // ordering is inconsistent between two classes (grouping bug)
    // or there's a non-affine pattern in the forward. Either way,
    // surface the offending pair(s) so we can inspect.
    let suspicious: Vec<((usize, usize), Vec<i64>)> = pair_deltas
        .iter()
        .filter(|(_, ds)| ds.iter().any(|d| *d < 0))
        .map(|(pair, ds)| (*pair, ds.iter().copied().collect()))
        .collect();
    let suspicious_str: String = if suspicious.is_empty() {
        String::new()
    } else {
        let list: Vec<String> = suspicious
            .iter()
            .map(|((c, p), ds)| {
                let c_period = bundle.class_members[*c].len();
                let p_period = bundle.class_members[*p].len();
                format!("({c}[{c_period}]<-{p}[{p_period}]: {ds:?})")
            })
            .collect();
        format!(" neg=[{}]", list.join(", "))
    };
    format!(
        "edges={total} pairs={pairs} {histo} non_uniform_pairs={non_uniform_pairs}{suspicious_str}",
        total = edges.len(),
        pairs = pair_deltas.len(),
        histo = histo_str,
    )
}

/// Class-level emission schedule derived from a `StencilBundle`.
///
/// The collapsed emitter (STENCIL_IR_V2_DESIGN.md §13 6.2.b.5) walks
/// this instead of the wave schedule. Per §13:
///
/// - `pre_loop`: period-1 classes whose member produces data the
///   periodic group consumes. Emitted ahead of the `for __repeat`
///   loop. Typically the embed / initial-residual path.
/// - `periodic`: classes with period ≥ 2, in intra-iter topological
///   order (Δrepeat=0 edges only). Emitted inside one
///   `for __repeat in 0..max_period` loop body with per-class
///   `(offset, period)` guards when periods differ.
/// - `post_loop`: period-1 classes downstream of the periodic group.
///   Emitted after the loop. Typically the final rmsnorm + lm_head.
/// - `carried`: edges with Δrepeat ≥ 1 — the loop-carried state
///   (residual stream, kv cache rewinds). Each entry becomes a
///   `let mut` declared before the loop, read at the top of the
///   body, written at the bottom.
///
/// Analysis-only. No emission, no token output. Consumed by the
/// collapsed emitter in a follow-up commit.
#[derive(Debug, Clone)]
struct ClassSchedule {
    pre_loop: Vec<usize>,
    periodic: Vec<usize>,
    post_loop: Vec<usize>,
    carried: Vec<ClassEdge>,
    /// Max period across `periodic` classes — the loop bound.
    /// `0` when `periodic` is empty (degenerate forward, no repeat).
    max_period: usize,
    /// `true` iff every class in `periodic` shares `max_period`.
    /// Informational; the p1 path handles `false` via offsets.
    uniform_period: bool,
    /// `true` iff every edge that crosses classes is uniform per
    /// class pair (matches `summarize_class_edges` non_uniform_pairs
    /// == 0). The collapsed emitter refuses non-uniform pairs today.
    uniform_pairs: bool,
    /// `true` iff every periodic class picked one impl across its
    /// members — i.e. `class_impl_id[c].is_some()` for every c in
    /// `periodic`. Heterogeneous periodic classes (Qwen3 §A.2) need
    /// the impl-table dispatch deferred to 6.2.b.5e.
    homogeneous_periodic: bool,
    /// Per-class offset in the global `__repeat` axis (`0..max_period`).
    /// For periodic class C with `period = class_members[C].len()`:
    /// `offset(C) + period(C) ≤ max_period`; C's iter `i` runs at
    /// global `__repeat = offset(C) + i`. For non-periodic classes
    /// (pre_loop / post_loop) the entry is `0` and unused.
    ///
    /// Assigned via `max_period - period(C)` under the assumption that
    /// short periodic classes start late and end aligned with
    /// `max_period` — matches llama/mistral's observed Δ=-1 boundary
    /// pattern (`(9[39]<-4[40]: [-1])`). The validator
    /// [`StencilBundle::validate_class_offsets`] checks every periodic
    /// class-pair edge's Δ against these offsets; failures bubble up
    /// as an emitter refusal rather than silently miscompiling.
    class_offsets: Vec<usize>,
    /// `true` iff every periodic class-pair edge's observed Δ is
    /// consistent with `class_offsets` under either intra-iter
    /// (`offset_diff = -Δ`) or carry-1 (`offset_diff = -Δ + 1`)
    /// interpretation. When `false`, the collapsed emitter refuses —
    /// offsets derived from period alone can't explain the edge
    /// structure (likely a class that starts early instead of late,
    /// which p1's simple offset rule doesn't model).
    offsets_consistent: bool,
}

impl StencilBundle {
    /// Compute the class-level emission schedule.
    ///
    /// Steps:
    /// 1. Build an intra-iter class DAG from Δrepeat=0 edges.
    /// 2. Kahn-topo-sort. Stable by class index on ties.
    /// 3. Split period-1 classes at the boundary of the periodic
    ///    group: a period-1 class appearing before any periodic
    ///    member in topo order is `pre_loop`; after, `post_loop`.
    /// 4. Collect Δrepeat ≥ 1 edges as `carried`.
    /// 5. Flag period uniformity + pair uniformity + homogeneity.
    /// 6. Re-route aliased-emittable period-1 classes into `periodic`
    ///    with a natural offset derived from their class_edges
    ///    (STENCIL_IR_V2_DESIGN.md §13 "alternative viable
    ///    follow-up"). The peel 5j performs on llama-fleet produces
    ///    period-1 FusedAddRmsNorm instances whose deps otherwise
    ///    trip `try_emit_collapsed_bucket`'s forbidden-partition
    ///    gate; `emit_aliased_class_inline` already handles
    ///    short-aliased (period=1, offset=fixed) emission.
    fn schedule(&self, fuf: &Fuf, sfuf: &Assignment, lib: &ImplementationLibrary) -> ClassSchedule {
        let n = self.class_members.len();
        let edges = self.class_edges(fuf, sfuf);

        // Intra-iter adjacency (Δrepeat=0). `pred_count[c]` is how
        // many distinct producer classes c depends on at Δ=0.
        let mut preds: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        let mut succs: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        for e in &edges {
            if e.delta_repeat() == 0 && e.consumer_class != e.producer_class {
                preds[e.consumer_class].insert(e.producer_class);
                succs[e.producer_class].insert(e.consumer_class);
            }
        }

        // Kahn's algorithm. Ties broken by class index for
        // determinism across rebuilds.
        let mut in_deg: Vec<usize> = preds.iter().map(|p| p.len()).collect();
        let mut ready: BTreeSet<usize> = (0..n).filter(|&c| in_deg[c] == 0).collect();
        let mut topo: Vec<usize> = Vec::with_capacity(n);
        while let Some(&c) = ready.iter().next() {
            ready.remove(&c);
            topo.push(c);
            for &s in &succs[c] {
                in_deg[s] -= 1;
                if in_deg[s] == 0 {
                    ready.insert(s);
                }
            }
        }
        // If topo is short, the Δ=0 subgraph has a cycle — can't
        // legally collapse. Fall through with whatever we got; the
        // caller's `uniform_pairs` check surfaces the problem.
        if topo.len() != n {
            for c in 0..n {
                if !topo.contains(&c) {
                    topo.push(c);
                }
            }
        }

        // Partition: first find the topo index of the earliest and
        // latest periodic class. Period-1 classes topo-before the
        // earliest periodic go pre-loop; topo-after the latest go
        // post-loop; in-between period-1 classes also go pre-loop
        // (they don't depend on loop output, just on pre-loop or
        // other period-1 classes, and emitting them pre-loop keeps
        // the loop body minimal).
        let periods: Vec<usize> = self.class_members.iter().map(|m| m.len()).collect();
        let first_periodic_topo = topo.iter().position(|&c| periods[c] >= 2);
        let last_periodic_topo = topo.iter().rposition(|&c| periods[c] >= 2);

        let mut pre_loop: Vec<usize> = Vec::new();
        let mut periodic: Vec<usize> = Vec::new();
        let mut post_loop: Vec<usize> = Vec::new();
        for (pos, &c) in topo.iter().enumerate() {
            if periods[c] >= 2 {
                periodic.push(c);
                continue;
            }
            match (first_periodic_topo, last_periodic_topo) {
                (Some(first), Some(last)) => {
                    if pos < first {
                        pre_loop.push(c);
                    } else if pos > last {
                        post_loop.push(c);
                    } else {
                        // Period-1 class interleaved with periodic
                        // group. Hoist to pre_loop — its outputs
                        // feed inside the loop as constants, and it
                        // doesn't itself need to run per-iteration.
                        pre_loop.push(c);
                    }
                }
                // No periodic classes at all — everything is pre-loop.
                _ => pre_loop.push(c),
            }
        }

        let carried: Vec<ClassEdge> = edges
            .iter()
            .filter(|e| e.delta_repeat() >= 1)
            .copied()
            .collect();

        let max_period = periodic.iter().map(|&c| periods[c]).max().unwrap_or(0);
        let uniform_period = periodic.iter().all(|&c| periods[c] == max_period);

        // Pair-uniformity: each (consumer_class, producer_class)
        // pair emits exactly one Δrepeat across its edges.
        let mut pair_deltas: BTreeMap<(usize, usize), BTreeSet<i64>> = BTreeMap::new();
        for e in &edges {
            pair_deltas
                .entry((e.consumer_class, e.producer_class))
                .or_default()
                .insert(e.delta_repeat());
        }
        let uniform_pairs = pair_deltas.values().all(|set| set.len() == 1);

        let homogeneous_periodic = periodic.iter().all(|&c| self.class_impl_id[c].is_some());

        // Offset assignment — one slot per class; only the periodic
        // entries carry signal. `max_period - period(C)` places short
        // classes late in the global __repeat window so their final
        // iter aligns with max_period's final iter (matches llama's
        // Δ=-1 boundary-class shape).
        let mut periodic_set: BTreeSet<usize> = periodic.iter().copied().collect();
        let mut class_offsets: Vec<usize> = vec![0; n];
        for &c in &periodic {
            class_offsets[c] = max_period.saturating_sub(periods[c]);
        }

        // Re-route aliased-emittable period-1 classes from pre/post_loop
        // into `periodic` with a natural offset derived from edges
        // (STENCIL_IR_V2_DESIGN.md §13). `emit_aliased_class_inline`
        // handles short-aliased (period=1, offset=fixed) emission via
        // `if __repeat >= offset && __repeat < offset+1` guards. This
        // lifts the llama-fleet refusal where 5j peels a
        // FusedAddRmsNorm instance into a period-1 class whose dep
        // crosses into `periodic` (the forbidden-partition case).
        //
        // Offset derivation: for each class_edges entry between C'
        // (period-1) and P (periodic), solve `offset(C') = offset(P) +
        // iter_of_P_member_on_that_edge` under intra-iter semantics.
        // C' has a single member at iter 0, so Δ = consumer_iter -
        // producer_iter collapses to ± the periodic-side iter index.
        // If multiple edges agree on one offset in [0, max_period),
        // accept and move C' into `periodic`. Inconsistent or absent
        // → leave C' in its current partition (today's behaviour).
        if max_period > 0 {
            let mut rerouted: BTreeSet<usize> = BTreeSet::new();
            for &c in pre_loop.iter().chain(post_loop.iter()) {
                if periods[c] != 1 {
                    continue;
                }
                let Some(imp_id) = self.class_impl_id[c] else {
                    continue;
                };
                let Some(&rep_sg) = self.class_members[c].first() else {
                    continue;
                };
                let imp = lib.get(imp_id);
                let claimed = sfuf.tiles_in_subgraph(rep_sg);
                if !is_aliased_emittable(imp, &claimed, fuf) {
                    continue;
                }
                // Collect offset candidates from class_edges. For each
                // edge between C' (period-1) and a periodic neighbour
                // N at N-iter i, N's member fires at
                // __repeat = offset(N) + i:
                // - C' as PRODUCER: C' must have fired by that
                //   __repeat (intra-iter or carry). Tightest: fire at
                //   __repeat = min over producer-side edges — later
                //   consumer iters carry forward.
                // - C' as CONSUMER: C' reads the neighbour's output at
                //   that __repeat. Tightest: fire at
                //   __repeat = max over consumer-side edges — earlier
                //   producer iters must have already fired (they have).
                // A mixed-direction class takes max(max_consumer, min_producer);
                // if max_consumer > min_producer, infeasible.
                let mut min_producer_edge: Option<usize> = None;
                let mut max_consumer_edge: Option<usize> = None;
                let mut infeasible = false;
                for e in &edges {
                    let (other_class, other_iter, cp_is_consumer) = if e.consumer_class == c
                        && periodic_set.contains(&e.producer_class)
                    {
                        (e.producer_class, e.producer_iter, true)
                    } else if e.producer_class == c && periodic_set.contains(&e.consumer_class) {
                        (e.consumer_class, e.consumer_iter, false)
                    } else {
                        continue;
                    };
                    let off_other = class_offsets[other_class];
                    let off_candidate = off_other + other_iter;
                    if off_candidate >= max_period {
                        infeasible = true;
                        break;
                    }
                    if cp_is_consumer {
                        max_consumer_edge =
                            Some(max_consumer_edge.map_or(off_candidate, |m| m.max(off_candidate)));
                    } else {
                        min_producer_edge =
                            Some(min_producer_edge.map_or(off_candidate, |m| m.min(off_candidate)));
                    }
                }
                if infeasible {
                    continue;
                }
                let offset = match (max_consumer_edge, min_producer_edge) {
                    (None, None) => continue,
                    (Some(cmax), None) => cmax,
                    (None, Some(pmin)) => pmin,
                    (Some(cmax), Some(pmin)) => {
                        if cmax > pmin {
                            continue;
                        }
                        cmax.max(pmin)
                    }
                };
                class_offsets[c] = offset;
                rerouted.insert(c);
            }
            if !rerouted.is_empty() {
                pre_loop.retain(|c| !rerouted.contains(c));
                post_loop.retain(|c| !rerouted.contains(c));
                for &c in &rerouted {
                    periodic.push(c);
                    periodic_set.insert(c);
                }
                // Re-sort `periodic` by offset then class-index to
                // keep emission order deterministic and roughly
                // topological in the __repeat axis.
                periodic.sort_by_key(|&c| (class_offsets[c], c));
            }
        }
        let periodic_set = periodic_set;

        // Validate: each edge between two periodic classes is
        // explainable as intra-iter (offset_diff = -Δ) or carry-1
        // (offset_diff = -Δ + 1) under the assigned offsets. Edges
        // to/from non-periodic classes are ignored (post-loop reads
        // the periodic class's final-iter output via __carry / __last
        // separately; pre-loop doesn't participate in offset
        // propagation).
        let mut offsets_consistent = true;
        for e in &edges {
            if !periodic_set.contains(&e.consumer_class)
                || !periodic_set.contains(&e.producer_class)
            {
                continue;
            }
            if e.consumer_class == e.producer_class {
                continue;
            }
            let diff =
                class_offsets[e.consumer_class] as i64 - class_offsets[e.producer_class] as i64;
            let d = e.delta_repeat();
            let intra_ok = diff == -d;
            let carry_ok = diff == -d + 1;
            if !intra_ok && !carry_ok {
                offsets_consistent = false;
                break;
            }
        }

        ClassSchedule {
            pre_loop,
            periodic,
            post_loop,
            carried,
            max_period,
            uniform_period,
            uniform_pairs,
            homogeneous_periodic,
            class_offsets,
            offsets_consistent,
        }
    }

    /// Classify the boundary tile inputs of every periodic class's
    /// representative subgraph by provenance — where each input
    /// comes from in the *collapsed* graph model, which the emitter
    /// needs to wire the right token expression to each fragment
    /// call's input slot.
    ///
    /// Three categories per (class, boundary_tile_input):
    ///
    /// - `IntraIter { producer_class }`: producer is a periodic
    ///   class in the same iteration (Δrepeat = 0). Emitter reads
    ///   the iter-local binding for `producer_class`.
    /// - `LoopCarry { producer_class, pre_loop_init_sg }`: producer
    ///   is periodic with Δrepeat = 1 (or more — rejected for now).
    ///   At iter 0 the value comes from `pre_loop_init_sg`'s output
    ///   (a pre-loop class); at iter I > 0 from the previous
    ///   iteration's `producer_class` output. Emitter hoists a
    ///   `let mut` initialised from the pre-loop sg, passes `&mut`
    ///   into the call, writes the producer's new output after the
    ///   producer's call in the same iteration.
    /// - `PreLoop { producer_sg }`: producer is a period-1 class;
    ///   its OwnedTensor lives in the outer scope. Emitter reads
    ///   the today-style `locals[&(id, slot)]` binding.
    ///
    /// Returns `None` when the model's shape violates a precondition
    /// the emitter can't yet handle (non-uniform producer-class
    /// pairing, Δrepeat > 1, producer class not in any partition).
    /// The caller (`emit_forward_collapsed_bucket`) treats `None` as
    /// "fall through to unimplemented!()" with an explanatory panic.
    fn class_input_provenance(
        &self,
        sched: &ClassSchedule,
        fuf: &Fuf,
        sfuf: &Assignment,
    ) -> Option<Vec<ClassInputs>> {
        let mut out: Vec<ClassInputs> = Vec::with_capacity(sched.periodic.len());
        let pre_loop_set: BTreeSet<usize> = sched.pre_loop.iter().copied().collect();
        let periodic_set: BTreeSet<usize> = sched.periodic.iter().copied().collect();

        for &consumer_class in &sched.periodic {
            let rep_sg = *self.class_members[consumer_class].first()?;
            let rep_claimed = sfuf.tiles_in_subgraph(rep_sg);
            let claimed_set: BTreeSet<TileId> = rep_claimed.iter().copied().collect();

            // Walk rep's boundary tile inputs in the same order
            // `emit_subgraph` uses so slot indices line up with the
            // fragment's `input_i` params.
            let mut seen: BTreeSet<(TileId, u8)> = BTreeSet::new();
            let mut slots: Vec<InputSlot> = Vec::new();
            for &t in &rep_claimed {
                let node = fuf.get(t);
                for input in &node.inputs {
                    let FufInput::Tile { id, slot } = input else {
                        continue;
                    };
                    if claimed_set.contains(id) {
                        continue;
                    }
                    if !seen.insert((*id, *slot)) {
                        continue;
                    }
                    let producer_sg = sfuf.subgraph_of(*id)?;
                    let producer_class = *self.class_of.get(&producer_sg)?;
                    let producer_iter = self.iter_of(producer_sg);

                    let origin = if pre_loop_set.contains(&producer_class) {
                        // Pre-loop (period-1) producer. Consumer's
                        // iter-0 input reads this local directly. We
                        // model this as a LoopCarry iff the same
                        // consumer slot is also fed by a periodic
                        // producer at iter 1+ (residual stream); as
                        // a plain PreLoop otherwise (constant ref).
                        //
                        // Detect the paired periodic producer by
                        // looking at iter 1 of the consumer's class.
                        paired_periodic_for_slot(
                            self,
                            sfuf,
                            fuf,
                            consumer_class,
                            &rep_claimed,
                            *id,
                            *slot,
                        )
                        .map(|(pc, prod_pos, _prod_slot)| InputOrigin::LoopCarry {
                            producer_class: pc,
                            producer_pos: prod_pos,
                            pre_loop_init_sg: Some(producer_sg),
                        })
                        .unwrap_or(InputOrigin::PreLoop { producer_sg })
                    } else if periodic_set.contains(&producer_class) {
                        // Periodic producer under shifted-offset semantics
                        // (p1). For consumer's rep (iter 0, global
                        // __repeat = offset(C)), producer's iter at that
                        // __repeat is `offset(C) - offset(P)` (intra-iter)
                        // or one less (carry — producer ran at __repeat
                        // = offset(C) - 1). Pick the branch that matches
                        // the observed iter_of(producer_sg); reject if
                        // neither. A shifted-carry has no natural
                        // pre-loop init because consumer starts strictly
                        // after producer (`offset(C) > offset(P)`); the
                        // LoopCarry variant with `pre_loop_init_sg =
                        // None` plumbs that case to the emitter, which
                        // then hoists an `Option<OwnedTensor>` carry var
                        // init None rather than a straight `OwnedTensor`.
                        let c_off = sched.class_offsets[consumer_class] as i64;
                        let p_off = sched.class_offsets[producer_class] as i64;
                        let intra_expected = c_off - p_off;
                        let carry_expected = intra_expected - 1;
                        let observed = producer_iter as i64;
                        // Position of the producing tile (`*id`) in
                        // producer_sg's claim. Shared across both
                        // IntraIter and LoopCarry branches so the
                        // emitter can name per-export idents.
                        let prod_claim = sfuf.tiles_in_subgraph(producer_sg);
                        let prod_pos = prod_claim.iter().position(|&x| x == *id)? as u8;
                        if observed == intra_expected && intra_expected >= 0 {
                            InputOrigin::IntraIter {
                                producer_class,
                                producer_pos: prod_pos,
                            }
                        } else if observed == carry_expected && carry_expected >= 0 {
                            InputOrigin::LoopCarry {
                                producer_class,
                                producer_pos: prod_pos,
                                pre_loop_init_sg: None,
                            }
                        } else {
                            return None;
                        }
                    } else {
                        // Post-loop producer feeding a periodic
                        // consumer — cyclic, reject.
                        return None;
                    };
                    slots.push(InputSlot {
                        producer_tile: *id,
                        producer_slot: *slot,
                        origin,
                    });
                }
            }

            out.push(ClassInputs {
                consumer_class,
                rep_sg,
                slots,
            });
        }
        Some(out)
    }
}

/// Find the periodic producer class paired with a pre-loop producer
/// for the same consumer-slot — i.e. the class that feeds the same
/// boundary input at iter ≥ 1. This is how the residual stream's
/// "iter-0 from embed, iter ≥ 1 from residual_add" pattern is
/// detected and collapsed to one `let mut` carry variable.
///
/// Walks the consumer's iter-1 subgraph's claimed tiles, matches by
/// *positional* boundary-input index: the iter-1 sg's Nth boundary
/// tile input, if it points into a periodic class with Δrepeat=1,
/// is the carry producer.
fn paired_periodic_for_slot(
    bundle: &StencilBundle,
    sfuf: &Assignment,
    fuf: &Fuf,
    consumer_class: usize,
    rep_claimed: &[TileId],
    target_id: TileId,
    target_slot: u8,
) -> Option<(usize, u8, u8)> {
    let members = bundle.class_members.get(consumer_class)?;
    if members.len() < 2 {
        return None;
    }
    // Position of (target_id, target_slot) in the rep's boundary
    // inputs (same dedup order `class_input_provenance` uses above).
    let mut target_pos: Option<usize> = None;
    {
        let claimed_set: BTreeSet<TileId> = rep_claimed.iter().copied().collect();
        let mut seen: BTreeSet<(TileId, u8)> = BTreeSet::new();
        let mut i = 0usize;
        'outer: for &t in rep_claimed {
            for input in &fuf.get(t).inputs {
                let FufInput::Tile { id, slot } = input else {
                    continue;
                };
                if claimed_set.contains(id) {
                    continue;
                }
                if !seen.insert((*id, *slot)) {
                    continue;
                }
                if *id == target_id && *slot == target_slot {
                    target_pos = Some(i);
                    break 'outer;
                }
                i += 1;
            }
        }
    }
    let target_pos = target_pos?;

    // Walk iter-1's boundary inputs, find the one at the same
    // positional index, look up its producer class.
    let iter1_sg = members[1];
    let iter1_claimed = sfuf.tiles_in_subgraph(iter1_sg);
    let claimed_set: BTreeSet<TileId> = iter1_claimed.iter().copied().collect();
    let mut seen: BTreeSet<(TileId, u8)> = BTreeSet::new();
    let mut i = 0usize;
    for &t in &iter1_claimed {
        for input in &fuf.get(t).inputs {
            let FufInput::Tile { id, slot } = input else {
                continue;
            };
            if claimed_set.contains(id) {
                continue;
            }
            if !seen.insert((*id, *slot)) {
                continue;
            }
            if i == target_pos {
                let producer_sg = sfuf.subgraph_of(*id)?;
                let pc = *bundle.class_of.get(&producer_sg)?;
                // Must be a periodic class (period ≥ 2).
                if bundle.class_members[pc].len() < 2 {
                    return None;
                }
                // Position of `*id` in producer_sg's claim.
                let prod_claim = sfuf.tiles_in_subgraph(producer_sg);
                let prod_pos = prod_claim.iter().position(|&x| x == *id)?;
                return Some((pc, prod_pos as u8, *slot));
            }
            i += 1;
        }
    }
    None
}

/// Boundary inputs of one periodic class's representative, tagged
/// by provenance. See `StencilBundle::class_input_provenance`.
#[derive(Debug, Clone)]
struct ClassInputs {
    consumer_class: usize,
    rep_sg: SubgraphId,
    slots: Vec<InputSlot>,
}

#[derive(Debug, Clone)]
struct InputSlot {
    producer_tile: TileId,
    producer_slot: u8,
    origin: InputOrigin,
}

#[derive(Debug, Clone)]
enum InputOrigin {
    /// Producer is a periodic class in the same iteration. Emitter
    /// reads the iter-local binding for `producer_class`.
    ///
    /// `producer_pos` is the position of the producing tile in the
    /// producer class rep's claim (sorted TileId order). Multi-tile
    /// claims (e.g. `FusedAddRmsNormImpl` claims `(Add, RmsNorm)` →
    /// pos 0 and 1) can export distinct outputs per position.
    IntraIter {
        producer_class: usize,
        producer_pos: u8,
    },
    /// Producer is periodic with a carry-1 relationship. Consumer
    /// at global __repeat = R reads producer's output at __repeat =
    /// R - 1. Two shapes:
    /// - `Some(sg)`: iter-0 initial value comes from
    ///   `pre_loop_init_sg`'s OwnedTensor (the residual-stream shape
    ///   — consumer is at offset 0 and its iter-0 reads pre-loop).
    /// - `None`: shifted-carry (p1) — consumer is at offset > 0 and
    ///   its iter-0 reads the previous __repeat's producer output
    ///   which already ran within the loop. No pre-loop init; the
    ///   emitter hoists an `Option<OwnedTensor>` carry variable
    ///   initialized `None` and populated before consumer reads.
    ///
    /// `producer_pos` is the position (within the producer class
    /// rep's claim) of the tile whose output is being carried. For
    /// the `Some` case, the iter-1 producer's position — detected
    /// by the positional boundary-input match in
    /// `paired_periodic_for_slot`.
    LoopCarry {
        producer_class: usize,
        producer_pos: u8,
        pre_loop_init_sg: Option<SubgraphId>,
    },
    /// Producer is a period-1 class; its OwnedTensor lives in the
    /// outer (bucket-fn) scope. Emitter reads the today-style
    /// `locals[&(id, slot)]` binding.
    PreLoop { producer_sg: SubgraphId },
}

/// Decide whether a subgraph's emit can be pulled out into a
/// reusable fragment fn.
///
/// Inlineable-only cases (returns `false`):
/// - `output_alias` reports any output that borrows from an
///   upstream tile — the fragment would need to return a
///   `TensorView<'a>` tied to a caller-owned OwnedTensor, and
///   lifting that through a fn boundary adds lifetime noise with
///   no win.
/// - `consumes_input_tiles` is non-empty — the fragment would
///   need to take the upstream tile by value, changing ownership
///   in the call site in ways the drop-pass already assumes. Easier
///   to leave these inline.
fn can_fragmentize(
    imp: &dyn crate::impl_lib::Implementation,
    claimed: &[TileId],
    fuf: &Fuf,
) -> bool {
    // Output aliasing stays inline. An aliasing output is a
    // `TensorView<'a>` that borrows an upstream OwnedTensor;
    // returning that through a fn boundary means the fragment fn
    // declares a lifetime tied to a consumed input, and the outer
    // bucket body must preserve that borrow across downstream
    // subgraph calls. Tractable but deferred.
    if !imp
        .output_alias(claimed, fuf)
        .iter()
        .all(|(_, src)| src.is_none())
    {
        return false;
    }
    // Multi-output tiles (rope_append producing q/k/v; FusedQkvRope*
    // family) stay inline on the unrolled path — its single
    // `__out_<pos>_<slot>` return ident only carries slot 0, and
    // downstream consumers of slots >0 would have no local binding.
    // The collapsed emitter handles multi-output via its own
    // fragment builder (`emit_fragment_call_expr`) with per-slot
    // idents + tuple return, so it calls `can_fragmentize_collapsed`
    // below instead of this function.
    for &t in claimed {
        if fuf.get(t).outputs.len() > 1 {
            return false;
        }
    }
    // Multi-tile subgraphs (fused kernels claiming > 1 tile) need
    // the fragment to emit bindings for every internal tile's
    // output, then return the subgraph's last tile's output.
    // Supported, but check claimed.len() is at least 1.
    //
    // Note: `consumes_input_tiles()` is NOT a bar to fragmentizing.
    // Consumed boundary inputs are lifted to fragment fn params
    // typed `OwnedTensor` (by value) vs `TensorView<'_>` (by view)
    // — see `emit_subgraph`.
    !claimed.is_empty()
}

/// Collapsed-mode fragmentization gate. Same as [`can_fragmentize`]
/// without the multi-output bail — the collapsed path's
/// `emit_fragment_call_expr` returns a tuple over referenced output
/// slots and the call site destructures per-slot.
fn can_fragmentize_collapsed(
    imp: &dyn crate::impl_lib::Implementation,
    claimed: &[TileId],
    fuf: &Fuf,
) -> bool {
    if !imp
        .output_alias(claimed, fuf)
        .iter()
        .all(|(_, src)| src.is_none())
    {
        return false;
    }
    !claimed.is_empty()
}

/// Collapsed-mode aliased-inline gate (§14.2). True when every
/// `output_alias` entry points at a BOUNDARY input tile (not another
/// claimed tile) and the impl doesn't consume any input. Such impls
/// (`FusedAddRmsNormImpl`, `FusedAddRmsNormWithOffsetImpl`, `AddRefImpl`)
/// can't be lifted through a fragment fn — their outputs are
/// `TensorView<'_>` aliases that would need lifetime-tied fn signatures
/// — so the collapsed emitter inlines their emit_call body directly
/// inside the loop, with a custom local map redirecting the rep's
/// boundary tile idents to the collapsed-path per-export idents (carry
/// vars / intra-iter class outputs).
fn is_aliased_emittable(
    imp: &dyn crate::impl_lib::Implementation,
    claimed: &[TileId],
    fuf: &Fuf,
) -> bool {
    if claimed.is_empty() {
        return false;
    }
    if !imp.consumes_input_tiles(claimed, fuf).is_empty() {
        return false;
    }
    let aliases = imp.output_alias(claimed, fuf);
    if aliases.is_empty() {
        return false;
    }
    let claimed_set: HashSet<TileId> = claimed.iter().copied().collect();
    aliases.iter().all(|(_, src)| match src {
        Some((src_tile, _)) => !claimed_set.contains(src_tile),
        None => false,
    })
}

/// Emit one subgraph. Uses the fragment library when the subgraph's
/// impl is "clean" (no aliased / consumed outputs); falls back to
/// inline emission otherwise.
#[allow(clippy::too_many_arguments)]
fn emit_subgraph(
    fuf: &Fuf,
    sfuf: &Assignment,
    program: &Program,
    model: &ModelParams,
    lib: &ImplementationLibrary,
    sg: SubgraphId,
    imp_id: ImplId,
    locals: &LocalMap,
    library: &mut FragmentLibrary,
    _stencil: &StencilBundle,
    weight_layout: &crate::emit::WeightLayout,
) -> TokenStream {
    let imp = lib.get(imp_id);
    let claimed = sfuf.tiles_in_subgraph(sg);

    if !can_fragmentize(imp, &claimed, fuf) {
        INLINE_FALLBACK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let ctx = EmitCtx {
            fuf,
            program,
            model,
            claimed_tiles: &claimed,
            locals,
            mode: EmitMode::Concrete,
            weight_layout: Some(weight_layout),
            repeat_var: None,
        };
        return imp.emit_call(&ctx);
    }
    CALL_SITE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Collect boundary tile inputs (inputs whose producer is not
    // in claimed_tiles), in the order they appear across claimed
    // tiles' input lists. Dedupe — the same boundary output can
    // feed multiple claimed tiles but only one fn param.
    //
    // Split into VIEWED (kernel only reads) vs CONSUMED (impl's
    // `consumes_input_tiles()` moves the upstream owner into its
    // own output — in-place kernels like add_rmsnorm). Viewed
    // become `TensorView<'_>` fn params; consumed become
    // `OwnedTensor` fn params (by value).
    let consumed_set: HashSet<(TileId, u8)> = imp
        .consumes_input_tiles(&claimed, fuf)
        .into_iter()
        .collect();
    let claimed_set: HashSet<TileId> = claimed.iter().copied().collect();
    let mut tile_params_ordered: Vec<(TileId, u8)> = Vec::new();
    let mut consumed_params_ordered: Vec<(TileId, u8)> = Vec::new();
    let mut seen_boundary: HashSet<(TileId, u8)> = HashSet::new();
    for &t in &claimed {
        let node = fuf.get(t);
        for input in &node.inputs {
            if let FufInput::Tile { id, slot } = input
                && !claimed_set.contains(id)
                && seen_boundary.insert((*id, *slot))
            {
                if consumed_set.contains(&(*id, *slot)) {
                    consumed_params_ordered.push((*id, *slot));
                } else {
                    tile_params_ordered.push((*id, *slot));
                }
            }
        }
    }

    // Weight accessors as declared by the impl. Each becomes one
    // fragment param of the accessor's `rust_type`.
    let accessors: Vec<WeightAccessor> = imp.required_weights(&claimed, fuf, program);

    // Build the abstract EmitCtx: tile/weight params replace
    // concrete `locals[..]` / `wm.<field>` references.
    let mut tile_params: HashMap<(TileId, u8), syn::Ident> = HashMap::new();
    let mut tile_param_idents: Vec<syn::Ident> = Vec::with_capacity(tile_params_ordered.len());
    for (i, &(id, slot)) in tile_params_ordered.iter().enumerate() {
        let p = format_ident!("input_{}", i);
        tile_params.insert((id, slot), p.clone());
        tile_param_idents.push(p);
    }
    let mut consumed_params: HashMap<(TileId, u8), syn::Ident> = HashMap::new();
    let mut consumed_param_idents: Vec<syn::Ident> =
        Vec::with_capacity(consumed_params_ordered.len());
    for (i, &(id, slot)) in consumed_params_ordered.iter().enumerate() {
        let p = format_ident!("consumed_{}", i);
        consumed_params.insert((id, slot), p.clone());
        consumed_param_idents.push(p);
    }
    let mut weight_params: HashMap<(WeightId, Option<u64>), syn::Ident> = HashMap::new();
    let mut weight_params_by_name: HashMap<String, syn::Ident> = HashMap::new();
    let mut weight_param_idents: Vec<syn::Ident> = Vec::with_capacity(accessors.len());
    for (i, acc) in accessors.iter().enumerate() {
        let p = format_ident!("w_{}", i);
        for &(wid, idx) in &acc.source_weights {
            weight_params.insert((wid, idx), p.clone());
        }
        weight_params_by_name.insert(acc.name.to_string(), p.clone());
        weight_param_idents.push(p);
    }

    let abstract_ctx = EmitCtx {
        fuf,
        program,
        model,
        claimed_tiles: &claimed,
        locals,
        mode: EmitMode::Abstract {
            tile_params,
            consumed_params,
            weight_params,
            weight_params_by_name,
        },
        // Abstract mode short-circuits weight reads through the
        // fragment's `w_N` params before the layout is ever
        // consulted, so the layout is functionally unused here.
        // Still thread it for symmetry with Concrete and to keep
        // future fragment-local rewrites straightforward.
        weight_layout: Some(weight_layout),
        // Fragment bodies are per-class, shape-stable; baking the
        // concrete layer literal here would defeat the whole
        // point, so every `ctx.layer_expr(_)` in the abstract
        // body goes through the loop variable whenever this ctx
        // is used for class-loop emission. 6.2.a keeps the old
        // unrolled path hot, so None here matches today's
        // behavior — 6.2.b wires a real Some(repeat_var).
        repeat_var: None,
    };
    let abstract_body = imp.emit_call(&abstract_ctx);

    // Fragment return value: the LAST claimed tile's slot-0
    // output. `output_ident` in Abstract mode returns
    // `__out_<claimed_pos>_<slot>`, so the return ident is
    // `__out_<claimed.len()-1>_0`. Multi-output subgraphs are
    // filtered out by `can_fragmentize`; fused multi-tile subgraphs
    // return the last tile's output (matches how the solver
    // committed the subgraph — topological final).
    let out_tile = *claimed.last().expect("non-empty claimed");
    let out_slot: u8 = 0;
    let out_claimed_pos = claimed.len() - 1;

    // Signature. The abstract body string captures the kernel call
    // shape; adding input/weight/output counts distinguishes
    // fragments that happen to stringify the same but differ in
    // param arity (defensive).
    let weight_types_str: String = accessors
        .iter()
        .map(|a| a.rust_type.to_string())
        .collect::<Vec<_>>()
        .join("|");
    let sig = format!(
        "ti={}|ci={}|wt={}|body={}",
        tile_params_ordered.len(),
        consumed_params_ordered.len(),
        weight_types_str,
        abstract_body,
    );

    // Intern the fragment. On miss, build and push the fn tokens.
    let frag_idx = if let Some(&idx) = library.by_sig.get(&sig) {
        idx
    } else {
        let idx = library.fns.len();
        let frag_name = format_ident!("__frag_{}", idx);

        let tile_param_decls: Vec<TokenStream> = tile_param_idents
            .iter()
            .map(|p| {
                quote! {
                    #p: ::ferrite_cuda_core::TensorView<'_>
                }
            })
            .collect();
        let consumed_param_decls: Vec<TokenStream> = consumed_param_idents
            .iter()
            .map(|p| {
                quote! {
                    #p: ::ferrite_cuda_core::alloc::OwnedTensor
                }
            })
            .collect();
        let weight_param_decls: Vec<TokenStream> = accessors
            .iter()
            .zip(weight_param_idents.iter())
            .map(|(acc, p)| {
                let ty = &acc.rust_type;
                quote! { #p: & #ty }
            })
            .collect();

        let return_ident = format_ident!("__out_{}_{}", out_claimed_pos, out_slot);
        let fn_tokens = quote! {
            #[cfg(feature = "cuda")]
            #[allow(clippy::too_many_arguments, unused_mut, unused_variables, non_snake_case)]
            unsafe fn #frag_name(
                #(#tile_param_decls,)*
                #(#consumed_param_decls,)*
                #(#weight_param_decls,)*
                wm: &Weights,
                ctx: &::ferrite_forward::ForwardCtx,
                device: &mut ::ferrite_cuda_core::device::GpuDevice,
            ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
                #abstract_body
                #return_ident
            }
        };
        library.by_sig.insert(sig, idx);
        library.fns.push(fn_tokens);
        idx
    };

    // Call site. Viewed tile args are wrapped `(*local).as_view()`
    // (TensorView, Copy); consumed tile args are passed by move
    // (the upstream local becomes inaccessible, which the drop
    // pass already accounts for via `consumes_input_tiles`).
    // Weight args go through the accessor's field name on `wm` —
    // for fused accessors that's the fused field, for singletons
    // it's the per-weight field.
    let frag_name = format_ident!("__frag_{}", frag_idx);
    let input_args: Vec<TokenStream> = tile_params_ordered
        .iter()
        .map(|&(id, slot)| {
            let local = &locals[&(id, slot)];
            quote! { (*#local).as_view() }
        })
        .collect();
    let consumed_args: Vec<TokenStream> = consumed_params_ordered
        .iter()
        .map(|&(id, slot)| {
            let local = &locals[&(id, slot)];
            quote! { #local }
        })
        .collect();
    let weight_args: Vec<TokenStream> = accessors
        .iter()
        .map(|acc| {
            let access = weight_layout.access_tokens(&acc.name);
            quote! { &wm.#access }
        })
        .collect();

    let out_local = locals
        .get(&(out_tile, out_slot))
        .cloned()
        .expect("output local for claimed tile must exist");

    quote! {
        let #out_local = unsafe {
            #frag_name(
                #(#input_args,)*
                #(#consumed_args,)*
                #(#weight_args,)*
                wm,
                ctx,
                device,
            )
        };
    }
}

/// Emit the wave walk for one bucket, optionally skipping a single
/// subgraph (the terminal lm_head) and with a caller-chosen
/// `protected` set for drop analysis. Shared by:
/// - the `forward_m_<N>` full-body emission (skip=None, protect
///   last tile),
/// - the `__body_m_<N>` helper emission (skip=terminal, protect
///   backbone tile).
#[allow(clippy::too_many_arguments)]
fn emit_wave_walk(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    program: &Program,
    model: &ModelParams,
    lib: &ImplementationLibrary,
    locals: &LocalMap,
    skip_subgraph: Option<SubgraphId>,
    protected: &HashSet<(TileId, u8)>,
    library: &mut FragmentLibrary,
    stencil: &StencilBundle,
    weight_layout: &crate::emit::WeightLayout,
) -> Vec<TokenStream> {
    let drops = compute_drops_after(fuf, sfuf, loop_ir, lib, skip_subgraph, protected);
    let mut body: Vec<TokenStream> = Vec::new();
    for wave in &loop_ir.waves {
        for (sg, imp_id) in &wave.subgraphs {
            if Some(*sg) == skip_subgraph {
                continue;
            }
            body.push(emit_subgraph(
                fuf,
                sfuf,
                program,
                model,
                lib,
                *sg,
                *imp_id,
                locals,
                library,
                stencil,
                weight_layout,
            ));
            if let Some(owners) = drops.get(sg) {
                for (t, s) in owners {
                    let ident = &locals[&(*t, *s)];
                    body.push(quote! { drop(#ident); });
                }
            }
        }
    }
    body
}

/// Emit a backbone-only per-bucket forward that runs every subgraph
/// EXCEPT the terminal one (assumed to be the `logits = gemm(normed,
/// lm_head)` call at the end of every causal-LM DSL body). Returns a
/// fresh-allocated clone of the subgraph output that would have been
/// the lm_head's input — typically the final rmsnorm's output.
///
/// Used by pipeline-parallelism intermediate ranks, which consume
/// backbone hidden-states from one rank and hand them to the next
/// without ever running lm_head.
///
/// The emitted fn's signature mirrors `forward_m_<N>` exactly except
/// for the name and semantic return value.
/// `true` when `FERRITE_STENCIL_CODEGEN=1` is set in the build env.
///
/// Toggle for the 6.2.b.5 collapsed-mode emitter. Off by default; on
/// routes `emit_forward_for_bucket` through `emit_forward_collapsed_bucket`.
/// The collapsed emitter is the load-bearing piece of STENCIL_IR_V2_DESIGN.md
/// §9 step 6 — one fragment per class, per-iteration dispatch via
/// `for __repeat in 0..max_period { … }` with guards for period mismatch.
fn stencil_codegen_enabled() -> bool {
    std::env::var("FERRITE_STENCIL_CODEGEN").ok().as_deref() == Some("1")
}

#[allow(clippy::too_many_arguments)]
fn emit_forward_for_bucket(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    program: &Program,
    model: &ModelParams,
    lib: &ImplementationLibrary,
    wp: crate::solver::WorkloadPoint,
    library: &mut FragmentLibrary,
    weight_layout: &crate::emit::WeightLayout,
) -> TokenStream {
    if stencil_codegen_enabled() {
        return emit_forward_collapsed_bucket(
            fuf,
            sfuf,
            loop_ir,
            program,
            model,
            lib,
            wp,
            library,
            weight_layout,
        );
    }
    let locals = build_local_map(fuf);

    // Forward returns the last tile's slot-0 output — its owner must
    // not be dropped before the function returns.
    let mut protected: HashSet<(TileId, u8)> = HashSet::new();
    if let Some(last) = fuf.nodes.last() {
        protected.insert((last.id, 0));
    }
    let stencil = StencilBundle::compute(fuf, sfuf);
    let body = emit_wave_walk(
        fuf,
        sfuf,
        loop_ir,
        program,
        model,
        lib,
        &locals,
        None,
        &protected,
        library,
        &stencil,
        weight_layout,
    );

    // The forward's return value: the last tile's output.
    let last_output = fuf
        .nodes
        .last()
        .map(|n| {
            let id = locals[&(n.id, 0)].clone();
            quote! { #id }
        })
        .unwrap_or_else(|| quote! { unreachable!("empty FUF") });

    let fn_name = bucket_fn_ident("forward_m", wp);

    let impl_names: String = loop_ir
        .waves
        .iter()
        .flat_map(|w| w.subgraphs.iter())
        .map(|(_, imp_id)| lib.get(*imp_id).name())
        .collect::<Vec<_>>()
        .join(", ");
    let impl_names_lit = proc_macro2::Literal::string(&impl_names);

    quote! {
        /// Forward pass for this model × workload bucket. Walks
        /// the solver-picked kernels in wavefront order.
        ///
        /// # Safety
        /// All tensors in `ctx` must be valid GPU memory; `device`
        /// must be the live CUDA device.
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments, unused_mut, unused_variables)]
        pub unsafe fn #fn_name(
            wm: &Weights,
            ctx: &::ferrite_forward::ForwardCtx,
            device: &mut ::ferrite_cuda_core::device::GpuDevice,
        ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
            ::tracing::debug!(
                m = ctx.input_ids.shape()[0],
                sk = ctx.max_seqlen_k,
                impls = #impl_names_lit,
                "ferrite forward"
            );
            #(#body)*
            #last_output
        }
    }
}

/// STENCIL_IR_V2_DESIGN.md §9 step 6.2.b.5 — class-driven forward emission.
///
/// Entry point when `FERRITE_STENCIL_CODEGEN=1`. Walks the StencilBundle's
/// classes and emits:
/// - pre-loop: period-1 classes topo-before any periodic class (emitted
///   via the existing `emit_subgraph` in concrete mode, binding to
///   `locals[&(tile, slot)]`);
/// - loop-carry hoist: `let mut __carry_P = <pre_loop_init_sg local>;`
///   per distinct LoopCarry producer class P;
/// - dedicated-last hoist: `let mut __last_C: Option<OwnedTensor> = None;`
///   per periodic class C consumed post-loop but not already a carry
///   producer;
/// - loop: `for __repeat in 0..max_period { … }` — each periodic class
///   emits a `let __cC_out = unsafe { __frag_N(...) };` binding whose
///   inputs resolve via the class's `ClassInputs` provenance (IntraIter
///   → `__cPC_out`; LoopCarry → `__carry_P`; PreLoop → `locals[…]`) and
///   whose weight args thread `__repeat` through `access_tokens_with_repeat`.
///   The fragment body itself uses `repeat_var = Some(repeat)` so
///   `ctx.layer_expr` inside it resolves to the fragment's `repeat: usize`
///   param — passed `__repeat` at the call site. At the bottom of the
///   body, each carry producer class writes `__carry_P = __cP_out;` and
///   each dedicated-last class writes `__last_C = Some(__cC_out);`.
/// - post-loop: periodic classes consumed post-loop bind their local
///   `t_<tile>_<slot>` idents to the final-iter values (`__carry_P` by
///   move for carry producers, `__last_C.take().unwrap()` for dedicated
///   lasts); then each post-loop class emits via `emit_subgraph` normally.
///
/// Refusal: any precondition failure returns a per-variant
/// `unimplemented!(<reason>)` body. Per `feedback_stencil_is_the_model`,
/// we never silently fall back to the unrolled path.
#[allow(clippy::too_many_arguments)]
fn emit_forward_collapsed_bucket(
    fuf: &Fuf,
    sfuf: &Assignment,
    _loop_ir: &Loop,
    program: &Program,
    model: &ModelParams,
    lib: &ImplementationLibrary,
    wp: crate::solver::WorkloadPoint,
    library: &mut FragmentLibrary,
    weight_layout: &crate::emit::WeightLayout,
) -> TokenStream {
    let stencil = StencilBundle::compute(fuf, sfuf);
    let sched = stencil.schedule(fuf, sfuf, lib);
    let provenance = stencil.class_input_provenance(&sched, fuf, sfuf);

    match try_emit_collapsed_bucket(
        &stencil,
        &sched,
        provenance.as_deref(),
        fuf,
        sfuf,
        program,
        model,
        lib,
        wp,
        library,
        weight_layout,
    ) {
        Ok(tokens) => tokens,
        Err(reason) => emit_collapsed_refusal(wp, &stencil, &sched, provenance.as_deref(), &reason),
    }
}

/// Build the `unimplemented!(<reason>)` body returned when
/// `try_emit_collapsed_bucket` rejects a variant. The reason string is
/// baked into the emitted `unimplemented!` so any runtime dispatch
/// surfaces the precise precondition that tripped.
fn emit_collapsed_refusal(
    wp: crate::solver::WorkloadPoint,
    stencil: &StencilBundle,
    sched: &ClassSchedule,
    provenance: Option<&[ClassInputs]>,
    reason: &str,
) -> TokenStream {
    let provenance_ok = provenance.is_some();
    let msg = format!(
        "FERRITE_STENCIL_CODEGEN collapsed-mode emitter refused: {reason} \
         (classes={cls}, homogeneous={hom}, pre={pre} periodic={per} post={post} \
         max_period={mp} uniform_period={up} offsets_consistent={oc} \
         uniform_pairs={upa} homogeneous_periodic={hp} carried_edges={ce} \
         provenance_ok={pov}); see STENCIL_IR_V2_DESIGN.md §13 6.2.b.5",
        cls = stencil.class_members.len(),
        hom = stencil.homogeneous_count(),
        pre = sched.pre_loop.len(),
        per = sched.periodic.len(),
        post = sched.post_loop.len(),
        mp = sched.max_period,
        up = sched.uniform_period,
        oc = sched.offsets_consistent,
        upa = sched.uniform_pairs,
        hp = sched.homogeneous_periodic,
        ce = sched.carried.len(),
        pov = provenance_ok,
    );
    let fn_name = bucket_fn_ident("forward_m", wp);
    quote! {
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments, unused_mut, unused_variables)]
        pub unsafe fn #fn_name(
            wm: &Weights,
            ctx: &::ferrite_forward::ForwardCtx,
            device: &mut ::ferrite_cuda_core::device::GpuDevice,
        ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
            let _ = (wm, ctx, device);
            unimplemented!(#msg)
        }
    }
}

fn carry_var_ident(producer_class: usize, pos: u8, slot: u8) -> syn::Ident {
    format_ident!("__carry_c{}_p{}_s{}", producer_class, pos, slot)
}

fn last_var_ident(class: usize, pos: u8, slot: u8) -> syn::Ident {
    format_ident!("__last_c{}_p{}_s{}", class, pos, slot)
}

fn class_out_ident(class: usize, pos: u8, slot: u8) -> syn::Ident {
    format_ident!("__c{}_out_{}_{}", class, pos, slot)
}

/// Owned `(pos, slot)` exports of class `c`'s rep. Each entry is a
/// `(claimed-position, output-slot)` pair whose impl marks the output
/// as an OwnedTensor (`output_alias` entry with `src=None`).
///
/// Walks ALL claimed tiles, not just the last — multi-tile fragmentized
/// classes only export the last tile's outputs today (intermediate
/// tiles are fragment-internal), but restricting to `claim.len() - 1`
/// is done here so a multi-pos claim whose intermediate outputs are
/// also `src=None` doesn't overstate what the fragment can return.
///
/// Untracked slots (e.g. FusedQkvRopeCacheImpl's K/V paged-cache views
/// — absent from `output_alias`) and aliased slots (src=Some) are both
/// excluded; the 5i alias-emittable path tracks those separately.
fn class_owned_exports(
    stencil: &StencilBundle,
    sfuf: &Assignment,
    fuf: &Fuf,
    lib: &ImplementationLibrary,
    c: usize,
) -> BTreeSet<(u8, u8)> {
    let Some(&rep) = stencil.class_members[c].first() else {
        return BTreeSet::new();
    };
    let Some(imp_id) = stencil.class_impl_id[c] else {
        return BTreeSet::new();
    };
    let imp = lib.get(imp_id);
    let claimed = sfuf.tiles_in_subgraph(rep);
    let Some(&last_tile) = claimed.last() else {
        return BTreeSet::new();
    };
    let last_pos = (claimed.len() - 1) as u8;
    imp.output_alias(&claimed, fuf)
        .into_iter()
        .filter_map(|((t, s), src)| {
            if t == last_tile && src.is_none() {
                Some((last_pos, s))
            } else {
                None
            }
        })
        .collect()
}

/// Core collapsed-emit body. Returns `Err(reason)` when the variant
/// violates a precondition; on success, returns the full
/// `pub unsafe fn forward_m_<N>(…) -> OwnedTensor { … }` token stream.
#[allow(clippy::too_many_arguments)]
fn try_emit_collapsed_bucket(
    stencil: &StencilBundle,
    sched: &ClassSchedule,
    provenance: Option<&[ClassInputs]>,
    fuf: &Fuf,
    sfuf: &Assignment,
    program: &Program,
    model: &ModelParams,
    lib: &ImplementationLibrary,
    wp: crate::solver::WorkloadPoint,
    library: &mut FragmentLibrary,
    weight_layout: &crate::emit::WeightLayout,
) -> Result<TokenStream, String> {
    // ── preconditions ────────────────────────────────────────────
    if !sched.offsets_consistent {
        return Err("offsets_consistent=false (periodic class-pair edges not \
             explainable under period-derived offsets)"
            .into());
    }
    if !sched.uniform_pairs {
        return Err("uniform_pairs=false (a class pair has non-uniform Δrepeat edges)".into());
    }
    if !sched.homogeneous_periodic {
        return Err("homogeneous_periodic=false (a periodic class has heterogeneous impls)".into());
    }
    let Some(provenance) = provenance else {
        return Err("provenance=None (boundary inputs cross partitions unexpectedly)".into());
    };

    // Every periodic class's rep must be fragmentizable + have no
    // consumed input tiles — OR qualify for the aliased-inline path
    // (`is_aliased_emittable`, §14.2). Multi-output tiles (rope_append
    // producing q/k/v; fused QKV-rope family) are OK on the fragment
    // path: the fragment returns a tuple over the slots actually
    // referenced by downstream consumers, per-export idents let call
    // sites bind each. Aliased-emittable classes (FusedAddRmsNormImpl
    // family) refuse fragmentization because their outputs are views
    // that would need lifetime-tied fn signatures; the emitter inlines
    // their emit_call instead and skips end-of-iter carry assignment
    // for alias-through-carry outputs (see `alias_through_carries`).
    let mut aliased_classes: BTreeSet<usize> = BTreeSet::new();
    for &c in &sched.periodic {
        let rep = *stencil
            .class_members
            .get(c)
            .and_then(|m| m.first())
            .ok_or_else(|| format!("periodic class {c} has no members"))?;
        let imp_id = stencil
            .class_impl_id
            .get(c)
            .copied()
            .flatten()
            .ok_or_else(|| format!("class {c} has no homogeneous impl"))?;
        let imp = lib.get(imp_id);
        let claimed = sfuf.tiles_in_subgraph(rep);
        let fragmentizable = imp.consumes_input_tiles(&claimed, fuf).is_empty()
            && can_fragmentize_collapsed(imp, &claimed, fuf);
        if fragmentizable {
            continue;
        }
        if is_aliased_emittable(imp, &claimed, fuf) {
            aliased_classes.insert(c);
            continue;
        }
        if !imp.consumes_input_tiles(&claimed, fuf).is_empty() {
            return Err(format!("class {c} rep impl consumes input tiles"));
        }
        return Err(format!("class {c} rep not fragmentizable"));
    }

    // Pre/post loop classes must be period-1 (analysis already does
    // this but assert defensively). An aliased-emittable pre/post-loop
    // class is emitted via `emit_subgraph`'s Concrete inline fallback,
    // which resolves boundary idents through the default
    // `t_<tile>_<slot>` LocalMap. That only works if the producer
    // subgraph has already been emitted in the enclosing scope:
    // pre_loop classes need upstream producers in `pre_loop` (emitted
    // before them); post_loop classes can also read from `periodic`
    // (promoted to outer-scope lets by the per-export post-ref
    // bindings) or `pre_loop` (emitted long before the loop). Refuse
    // when a dependency crosses into the forbidden partition.
    let pre_loop_set: BTreeSet<usize> = sched.pre_loop.iter().copied().collect();
    let post_loop_set: BTreeSet<usize> = sched.post_loop.iter().copied().collect();
    let periodic_set_pre: BTreeSet<usize> = sched.periodic.iter().copied().collect();
    for &c in sched.pre_loop.iter().chain(sched.post_loop.iter()) {
        if stencil.class_members[c].len() != 1 {
            return Err(format!("pre/post_loop class {c} has period > 1"));
        }
        let sg = stencil.class_members[c][0];
        let imp_id = sfuf
            .impl_of(sg)
            .ok_or_else(|| format!("pre/post class {c} sg has no impl"))?;
        let imp = lib.get(imp_id);
        let claimed = sfuf.tiles_in_subgraph(sg);
        if !is_aliased_emittable(imp, &claimed, fuf)
            || can_fragmentize_collapsed(imp, &claimed, fuf)
        {
            continue;
        }
        let is_pre = pre_loop_set.contains(&c);
        let claimed_set: HashSet<TileId> = claimed.iter().copied().collect();
        for &t in &claimed {
            for input in &fuf.get(t).inputs {
                if let FufInput::Tile { id, .. } = input
                    && !claimed_set.contains(id)
                {
                    let Some(producer_sg) = sfuf.subgraph_of(*id) else {
                        continue;
                    };
                    let Some(&producer_class) = stencil.class_of.get(&producer_sg) else {
                        continue;
                    };
                    let ok = if is_pre {
                        pre_loop_set.contains(&producer_class)
                    } else {
                        // post_loop can pull from pre_loop (outer lets
                        // established long before) or periodic (post-
                        // ref bindings land the final-iter value into
                        // the enclosing scope). Another post_loop
                        // class is OK if already emitted in schedule
                        // order; accept conservatively.
                        pre_loop_set.contains(&producer_class)
                            || periodic_set_pre.contains(&producer_class)
                            || post_loop_set.contains(&producer_class)
                    };
                    if !ok {
                        return Err(format!(
                            "pre/post_loop class {c} aliased impl depends on class {producer_class} in a forbidden partition"
                        ));
                    }
                }
            }
        }
    }

    // ── identify loop-carry producer (class, pos, slot) exports ──
    //
    // Per (producer_class P, producer_pos POS, producer_slot S), the
    // init source is the UNION of what consumers declare: any consumer
    // with `Some(sg)` pins the init to that sg (residual-stream shape
    // — consumer at offset 0 reads pre-loop at iter 0); consumers with
    // `None` are shifted-carry (consumer at offset > 0 reads prev
    // __repeat's value which has already been written inside the
    // loop). When all consumers are `None`, the init is `None` and the
    // emitter hoists an `Option<OwnedTensor>` carry var. When mixed,
    // the `Some` wins — shifted consumers don't read the init.
    //
    // Key by (class, pos, slot) so multi-tile claims (e.g.
    // FusedAddRmsNorm claims (Add, RmsNorm)) can carry distinct
    // outputs per position. Today's residual stream is pos=0 slot=0
    // on a single-tile single-output class; the extra `pos` is what
    // 5i needs to disentangle aliased multi-tile impls.
    let mut carry_inits: BTreeMap<(usize, u8, u8), Option<SubgraphId>> = BTreeMap::new();
    for ci in provenance {
        for s in &ci.slots {
            if let InputOrigin::LoopCarry {
                producer_class,
                producer_pos,
                pre_loop_init_sg,
            } = s.origin
            {
                let key = (producer_class, producer_pos, s.producer_slot);
                let entry = carry_inits.entry(key).or_insert(None);
                match (&*entry, pre_loop_init_sg) {
                    (Some(a), Some(b)) if *a != b => {
                        return Err(format!(
                            "carry producer class {producer_class} pos {producer_pos} slot {} has conflicting pre-loop inits",
                            s.producer_slot,
                        ));
                    }
                    (None, Some(_)) => *entry = pre_loop_init_sg,
                    _ => {}
                }
            }
        }
    }
    let carry_producers: BTreeSet<(usize, u8, u8)> = carry_inits.keys().copied().collect();

    // ── find post-loop consumers of periodic classes ─────────────
    // For each (periodic producer class P, pos POS, slot S) referenced
    // by a post-loop subgraph's boundary input, record the producing
    // (tile, slot) that the post-loop side refers to. One tile-id per
    // (class, pos, slot) — positional uniqueness across the class's
    // iterations is an invariant.
    let periodic_set: BTreeSet<usize> = sched.periodic.iter().copied().collect();
    let mut post_refs: BTreeMap<(usize, u8, u8), (TileId, u8)> = BTreeMap::new();
    for &c in &sched.post_loop {
        let sg = stencil.class_members[c][0];
        let claimed = sfuf.tiles_in_subgraph(sg);
        let claimed_set: HashSet<TileId> = claimed.iter().copied().collect();
        for &t in &claimed {
            for input in &fuf.get(t).inputs {
                if let FufInput::Tile { id, slot } = input {
                    if claimed_set.contains(id) {
                        continue;
                    }
                    let Some(producer_sg) = sfuf.subgraph_of(*id) else {
                        continue;
                    };
                    let Some(&pc) = stencil.class_of.get(&producer_sg) else {
                        continue;
                    };
                    if !periodic_set.contains(&pc) {
                        continue;
                    }
                    // Position of `*id` within producer_sg's claim.
                    let prod_claim = sfuf.tiles_in_subgraph(producer_sg);
                    let Some(prod_pos) = prod_claim.iter().position(|&x| x == *id) else {
                        continue;
                    };
                    let key = (pc, prod_pos as u8, *slot);
                    if let Some(prev) = post_refs.get(&key) {
                        if *prev != (*id, *slot) {
                            return Err(format!(
                                "periodic class {pc} pos {prod_pos} slot {slot} referenced by post-loop from multiple tiles"
                            ));
                        }
                    } else {
                        post_refs.insert(key, (*id, *slot));
                    }
                }
            }
        }
    }
    let post_loop_exports: BTreeSet<(usize, u8, u8)> = post_refs.keys().copied().collect();
    let dedicated_last: BTreeSet<(usize, u8, u8)> = post_loop_exports
        .difference(&carry_producers)
        .copied()
        .collect();

    // Final tile: must live in a pre_loop or post_loop class. A
    // periodic-class final tile would need the emitter to expose the
    // last iter's output as the fn return, which 5e doesn't model.
    let last_tile = fuf.nodes.last().ok_or_else(|| "empty FUF".to_string())?.id;
    let last_sg = sfuf
        .subgraph_of(last_tile)
        .ok_or_else(|| "last tile has no subgraph".to_string())?;
    let last_class = *stencil
        .class_of
        .get(&last_sg)
        .ok_or_else(|| "last subgraph has no class".to_string())?;
    if periodic_set.contains(&last_class) {
        return Err(format!(
            "final tile lives in periodic class {last_class} — needs post-loop path"
        ));
    }

    // ── per-class output-export inventory ────────────────────────
    //
    // For each periodic class, compute:
    //   - `referenced_exports[c]` = set of `(pos, slot)` exports of
    //     class c's rep read by any downstream consumer: periodic
    //     IntraIter / LoopCarry, or post-loop. These are what the
    //     fragment must return.
    //   - `owned_exports[c]` = `(pos, slot)` pairs whose impl tracks
    //     them as OwnedTensor (src=None in `output_alias`). Untracked
    //     slots (e.g. FusedQkvRope*'s K/V paged-cache views) can't be
    //     returned through the fragment fn boundary, so a referenced
    //     export that isn't owned is a refusal (for the fragmentized
    //     path — the 5i alias-emittable path lifts this gate).
    let n_classes = stencil.class_members.len();
    let mut referenced_exports: Vec<BTreeSet<(u8, u8)>> = vec![BTreeSet::new(); n_classes];
    for ci in provenance {
        for s in &ci.slots {
            match &s.origin {
                InputOrigin::IntraIter {
                    producer_class,
                    producer_pos,
                }
                | InputOrigin::LoopCarry {
                    producer_class,
                    producer_pos,
                    ..
                } => {
                    referenced_exports[*producer_class].insert((*producer_pos, s.producer_slot));
                }
                InputOrigin::PreLoop { .. } => {}
            }
        }
    }
    for &(pc, pos, slot) in post_refs.keys() {
        referenced_exports[pc].insert((pos, slot));
    }

    for &c in &sched.periodic {
        if aliased_classes.contains(&c) {
            // Aliased-emittable classes expose every `output_alias`
            // entry (src=None OR src=Some) as `__cC_out_<pos>_<slot>`
            // bindings. A referenced export is legal iff some alias
            // entry targets the same (pos, slot).
            let rep = stencil.class_members[c][0];
            let claimed = sfuf.tiles_in_subgraph(rep);
            let imp_id =
                stencil.class_impl_id[c].ok_or_else(|| format!("class {c} has no impl"))?;
            let imp = lib.get(imp_id);
            let aliases = imp.output_alias(&claimed, fuf);
            let exported: BTreeSet<(u8, u8)> = aliases
                .iter()
                .filter_map(|((t, s), _)| {
                    claimed.iter().position(|x| x == t).map(|p| (p as u8, *s))
                })
                .collect();
            for &(pos, slot) in &referenced_exports[c] {
                if !exported.contains(&(pos, slot)) {
                    return Err(format!(
                        "aliased class {c} pos {pos} slot {slot} referenced but not listed in output_alias"
                    ));
                }
            }
            continue;
        }
        let owned = class_owned_exports(stencil, sfuf, fuf, lib, c);
        for &(pos, slot) in &referenced_exports[c] {
            if !owned.contains(&(pos, slot)) {
                return Err(format!(
                    "class {c} pos {pos} slot {slot} referenced downstream but not owned (output_alias untracked or aliased) — 5g scope refuses"
                ));
            }
        }
    }

    // ── alias-through-carries ────────────────────────────────────
    // For each aliased class c, walk `output_alias` entries that
    // target a boundary input tile. If that boundary's provenance is
    // a LoopCarry whose producer is in `carry_producers`, the inline
    // kernel mutates the carry buffer in place — the end-of-iter
    // `__carry = __cC_out_P_S` update is both redundant (the buffer
    // already holds the next iter's value) and a type error
    // (`__cC_out_P_S: TensorView<'_>` vs `__carry: OwnedTensor`).
    // Track these for the carry-update loop to skip.
    let mut alias_through_carries: BTreeSet<(usize, u8, u8)> = BTreeSet::new();
    for &c in &aliased_classes {
        let rep = stencil.class_members[c][0];
        let claimed = sfuf.tiles_in_subgraph(rep);
        let claimed_set: HashSet<TileId> = claimed.iter().copied().collect();
        let imp_id = stencil.class_impl_id[c].ok_or_else(|| format!("class {c} has no impl"))?;
        let imp = lib.get(imp_id);
        let aliases = imp.output_alias(&claimed, fuf);
        // Rebuild the rep's boundary-input ordering to match provenance.
        let mut tile_params_ordered: Vec<(TileId, u8)> = Vec::new();
        let mut seen: HashSet<(TileId, u8)> = HashSet::new();
        for &t in &claimed {
            for input in &fuf.get(t).inputs {
                if let FufInput::Tile { id, slot } = input
                    && !claimed_set.contains(id)
                    && seen.insert((*id, *slot))
                {
                    tile_params_ordered.push((*id, *slot));
                }
            }
        }
        let ci = provenance
            .iter()
            .find(|ci| ci.consumer_class == c)
            .ok_or_else(|| format!("aliased class {c} missing provenance entry"))?;
        for ((_out_tile, _out_slot), src) in aliases.iter() {
            let Some((src_tile, src_slot)) = src else {
                continue;
            };
            let Some(bidx) = tile_params_ordered
                .iter()
                .position(|(t, s)| t == src_tile && s == src_slot)
            else {
                continue;
            };
            let s = &ci.slots[bidx];
            if let InputOrigin::LoopCarry {
                producer_class,
                producer_pos,
                ..
            } = &s.origin
            {
                let key = (*producer_class, *producer_pos, s.producer_slot);
                if carry_producers.contains(&key) {
                    alias_through_carries.insert(key);
                }
            }
        }
    }

    // Aliased classes don't currently support post-loop consumption:
    // their exports bind as `TensorView<'_>` which doesn't fit the
    // `Option<OwnedTensor>` dedicated-last pattern.
    for &(c, pos, slot) in &dedicated_last {
        if aliased_classes.contains(&c) {
            return Err(format!(
                "aliased class {c} pos {pos} slot {slot} referenced post-loop — unsupported"
            ));
        }
    }

    // Aliased exports only need `__cC_out_<pos>_<slot>` bindings for
    // IntraIter or post-loop consumers. LoopCarry-only exports feed
    // through `__carry_cC_p<pos>_s<slot>` (which the in-place kernel
    // mutates via `alias_through_carries`), so we skip any per-iter
    // __cC_out for them.
    let mut aliased_intraiter_exports: BTreeSet<(usize, u8, u8)> = BTreeSet::new();
    for ci in provenance {
        for s in &ci.slots {
            if let InputOrigin::IntraIter {
                producer_class,
                producer_pos,
            } = &s.origin
            {
                aliased_intraiter_exports.insert((*producer_class, *producer_pos, s.producer_slot));
            }
        }
    }
    for &(pc, pos, slot) in post_refs.keys() {
        aliased_intraiter_exports.insert((pc, pos, slot));
    }

    // Per-class ordered returned exports list (sorted by (pos, slot)),
    // used to pick tuple-vs-scalar fragment return and to name the
    // per-export call-site bindings. Classes with no referenced
    // exports still need a fragment call for side-effects (e.g. cache
    // writes); emit them with unit return.
    let class_returned_exports: Vec<Vec<(u8, u8)>> = (0..n_classes)
        .map(|c| referenced_exports[c].iter().copied().collect())
        .collect();

    // ── emission ────────────────────────────────────────────────
    let locals = build_local_map(fuf);
    let mut body: Vec<TokenStream> = Vec::new();

    // Pre-loop classes via today's emit_subgraph. Binds
    // `let t_<tile>_<slot> = …;` into the enclosing scope.
    for &c in &sched.pre_loop {
        let sg = stencil.class_members[c][0];
        let imp_id = sfuf
            .impl_of(sg)
            .ok_or_else(|| format!("pre_loop sg {sg:?} has no impl"))?;
        body.push(emit_subgraph(
            fuf,
            sfuf,
            program,
            model,
            lib,
            sg,
            imp_id,
            &locals,
            library,
            stencil,
            weight_layout,
        ));
    }

    // Carry hoists. A carry with pre-loop init is an OwnedTensor
    // (today's residual-stream shape, byte-identical to 6.2.b.5e).
    // A carry without init (shifted-carry — all consumers are short
    // and start after the producer) is `Option<OwnedTensor>` init
    // `None`; the producer populates it before any consumer reads.
    //
    // Keyed per (class, slot) so multi-output producers can carry a
    // subset of their slots without forcing the others into the carry
    // set. The init sg's output slot for slot-keyed carries matches
    // the producer class's slot (the pre-loop init tile uses its slot
    // 0 today; multi-output inits with slot > 0 are a future concern).
    let carries_with_init: BTreeSet<(usize, u8, u8)> = carry_inits
        .iter()
        .filter_map(|(&k, init)| init.is_some().then_some(k))
        .collect();
    for &(pc, pos, slot) in &carry_producers {
        let carry = carry_var_ident(pc, pos, slot);
        match carry_inits[&(pc, pos, slot)] {
            Some(init_sg) => {
                let init_tiles = sfuf.tiles_in_subgraph(init_sg);
                let init_tile = *init_tiles
                    .last()
                    .ok_or_else(|| format!("carry init sg {init_sg:?} has no tiles"))?;
                let init_local = locals
                    .get(&(init_tile, 0))
                    .cloned()
                    .ok_or_else(|| "carry init local missing".to_string())?;
                body.push(quote! {
                    let mut #carry: ::ferrite_cuda_core::alloc::OwnedTensor = #init_local;
                });
            }
            None => {
                body.push(quote! {
                    let mut #carry: ::std::option::Option<
                        ::ferrite_cuda_core::alloc::OwnedTensor,
                    > = None;
                });
            }
        }
    }

    // Dedicated-last hoists: Option<OwnedTensor>, filled each iter.
    for &(c, pos, slot) in &dedicated_last {
        let last = last_var_ident(c, pos, slot);
        body.push(quote! {
            let mut #last: ::std::option::Option<::ferrite_cuda_core::alloc::OwnedTensor> = None;
        });
    }

    // ── loop body ───────────────────────────────────────────────
    let max_period = sched.max_period;
    let prov_by_class: HashMap<usize, &ClassInputs> = provenance
        .iter()
        .map(|ci| (ci.consumer_class, ci))
        .collect();

    // Per-class shape: "full" = offset 0 + period == max_period (runs
    // every iter; straight OwnedTensor binding, today's form).
    // "short" = offset > 0 OR period < max_period (guarded call with
    // Option<OwnedTensor> hoisted above the loop so cross-iter reads
    // resolve to the most-recent Some). Full emission is byte-
    // identical to 6.2.b.5e for uniform variants (commandr-style).
    let short_classes: BTreeSet<usize> = sched
        .periodic
        .iter()
        .copied()
        .filter(|&c| sched.class_offsets[c] != 0 || stencil.class_members[c].len() != max_period)
        .collect();

    // Short-class Option hoists live OUTSIDE the loop so consumers in
    // later iters still see the Some assigned in earlier iters (for
    // carry-target moves + post-loop reads). Re-assigned on each
    // guarded firing; `.take()` at carry/last update sites. One hoist
    // per (class, referenced_export) — multi-export short classes get
    // one Option per (pos, slot). Full aliased classes emit the
    // `__cC_out_P_S: TensorView = ...` binding directly per iter (no
    // hoist). Short aliased classes hoist `Option<GpuTensor>` (the
    // raw descriptor the inline kernel aliases into) — only for
    // IntraIter/post-loop consumers, since LoopCarry-only exports
    // flow through the carry var and need no __cC_out.
    for &c in &short_classes {
        let is_aliased = aliased_classes.contains(&c);
        for &(pos, slot) in &class_returned_exports[c] {
            if is_aliased {
                if !aliased_intraiter_exports.contains(&(c, pos, slot)) {
                    continue;
                }
                let out = class_out_ident(c, pos, slot);
                body.push(quote! {
                    let mut #out: ::std::option::Option<::ferrite_cuda_core::GpuTensor> = None;
                });
            } else {
                let out = class_out_ident(c, pos, slot);
                body.push(quote! {
                    let mut #out: ::std::option::Option<::ferrite_cuda_core::alloc::OwnedTensor> = None;
                });
            }
        }
    }

    let mut loop_body: Vec<TokenStream> = Vec::new();
    for &c in &sched.periodic {
        let ci = prov_by_class
            .get(&c)
            .copied()
            .ok_or_else(|| format!("class {c} missing provenance entry"))?;
        if aliased_classes.contains(&c) {
            let is_short = short_classes.contains(&c);
            let tokens = emit_aliased_class_inline(
                c,
                ci,
                stencil,
                fuf,
                sfuf,
                program,
                model,
                lib,
                &locals,
                weight_layout,
                &carry_producers,
                &carries_with_init,
                is_short,
                sched.class_offsets[c],
                stencil.class_members[c].len(),
                &aliased_intraiter_exports,
                &short_classes,
                &aliased_classes,
            )?;
            loop_body.push(tokens);
            continue;
        }
        let offset = sched.class_offsets[c];
        let period = stencil.class_members[c].len();
        let is_short = short_classes.contains(&c);
        let returned = &class_returned_exports[c];
        let fragment_expr = emit_fragment_call_expr(
            c,
            ci,
            offset,
            returned,
            stencil,
            fuf,
            sfuf,
            program,
            model,
            lib,
            &locals,
            library,
            weight_layout,
            &short_classes,
            &carry_producers,
            &carries_with_init,
            &class_returned_exports,
        )?;
        let call = match (is_short, returned.len()) {
            // Full class, zero referenced exports: fragment returns `()`.
            (false, 0) => quote! { #fragment_expr; },
            // Full class, single referenced export: `let __cC_out_P_S = <expr>;`.
            (false, 1) => {
                let (pos, slot) = returned[0];
                let out = class_out_ident(c, pos, slot);
                quote! { let #out = #fragment_expr; }
            }
            // Full class, multiple referenced exports: tuple destructure.
            (false, _) => {
                let outs: Vec<syn::Ident> = returned
                    .iter()
                    .map(|&(pos, slot)| class_out_ident(c, pos, slot))
                    .collect();
                quote! { let ( #( #outs ),* ) = #fragment_expr; }
            }
            // Short class, zero referenced exports: guarded statement.
            (true, 0) => {
                let offset_lit = proc_macro2::Literal::usize_unsuffixed(offset);
                let period_lit = proc_macro2::Literal::usize_unsuffixed(period);
                quote! {
                    if __repeat >= #offset_lit && __repeat < #offset_lit + #period_lit {
                        #fragment_expr;
                    }
                }
            }
            // Short class, single export: assign the one Option inside
            // the guard.
            (true, 1) => {
                let (pos, slot) = returned[0];
                let out = class_out_ident(c, pos, slot);
                let offset_lit = proc_macro2::Literal::usize_unsuffixed(offset);
                let period_lit = proc_macro2::Literal::usize_unsuffixed(period);
                quote! {
                    if __repeat >= #offset_lit && __repeat < #offset_lit + #period_lit {
                        #out = ::std::option::Option::Some(#fragment_expr);
                    }
                }
            }
            // Short class, multiple exports: destructure to temp
            // locals then assign each Option slot.
            (true, _) => {
                let tmps: Vec<syn::Ident> = (0..returned.len())
                    .map(|i| format_ident!("__tmp_{}_{}", c, i))
                    .collect();
                let outs: Vec<syn::Ident> = returned
                    .iter()
                    .map(|&(pos, slot)| class_out_ident(c, pos, slot))
                    .collect();
                let assigns = tmps.iter().zip(outs.iter()).map(|(t, o)| {
                    quote! { #o = ::std::option::Option::Some(#t); }
                });
                let offset_lit = proc_macro2::Literal::usize_unsuffixed(offset);
                let period_lit = proc_macro2::Literal::usize_unsuffixed(period);
                quote! {
                    if __repeat >= #offset_lit && __repeat < #offset_lit + #period_lit {
                        let ( #( #tmps ),* ) = #fragment_expr;
                        #( #assigns )*
                    }
                }
            }
        };
        loop_body.push(call);
    }

    // End-of-iter updates: carries + dedicated lasts. Four shapes
    // from the cross of (producer full vs short) × (carry has init vs
    // not). The assignment moves/unwraps accordingly; guards on short
    // producers keep the update limited to iters where the producer
    // ran, while init-less carries wrap the assigned value in `Some`
    // so their `Option<OwnedTensor>` type stays consistent.
    //
    // Keyed per (producer_class, producer_pos, producer_slot) — each
    // carried export reads its own `__cC_out_P_S` binding.
    for &(pc, pos, slot) in &carry_producers {
        // Alias-through-carry: the in-place kernel for this carry's
        // producing class already mutated the carry's backing buffer
        // during its emit_call, so the end-of-iter reassignment is
        // both redundant and a type mismatch (`__cC_out_P_S:
        // TensorView` vs `__carry: OwnedTensor`).
        if alias_through_carries.contains(&(pc, pos, slot)) {
            continue;
        }
        let carry = carry_var_ident(pc, pos, slot);
        let c_out = class_out_ident(pc, pos, slot);
        let has_init = carries_with_init.contains(&(pc, pos, slot));
        let short = short_classes.contains(&pc);
        let rhs = match (short, has_init) {
            (false, true) => quote! { #c_out },
            (false, false) => quote! { ::std::option::Option::Some(#c_out) },
            (true, true) => quote! { #c_out.take().expect("short-class carry producer") },
            (true, false) => quote! { #c_out.take() },
        };
        let update = quote! { #carry = #rhs; };
        if short {
            let offset = sched.class_offsets[pc];
            let period = stencil.class_members[pc].len();
            let off_lit = proc_macro2::Literal::usize_unsuffixed(offset);
            let per_lit = proc_macro2::Literal::usize_unsuffixed(period);
            loop_body.push(quote! {
                if __repeat >= #off_lit && __repeat < #off_lit + #per_lit {
                    #update
                }
            });
        } else {
            loop_body.push(update);
        }
    }
    for &(c, pos, slot) in &dedicated_last {
        let last = last_var_ident(c, pos, slot);
        let c_out = class_out_ident(c, pos, slot);
        if short_classes.contains(&c) {
            let offset = sched.class_offsets[c];
            let period = stencil.class_members[c].len();
            let off_lit = proc_macro2::Literal::usize_unsuffixed(offset);
            let per_lit = proc_macro2::Literal::usize_unsuffixed(period);
            loop_body.push(quote! {
                if __repeat >= #off_lit && __repeat < #off_lit + #per_lit {
                    #last = ::std::option::Option::Some(
                        #c_out.take().expect("short-class dedicated-last"),
                    );
                }
            });
        } else {
            loop_body.push(quote! { #last = ::std::option::Option::Some(#c_out); });
        }
    }

    let max_period_lit = proc_macro2::Literal::usize_unsuffixed(max_period);
    body.push(quote! {
        for __repeat in 0usize..#max_period_lit {
            #(#loop_body)*
        }
    });

    // Post-loop: bind each referenced periodic (class, slot)'s
    // final-iter value to the local ident the post-loop sg expects.
    // Moves out of __carry_cC_sS (single-use) or .take() from
    // __last_cC_sS.
    for (&(pc, ppos, pslot), &(tile, slot)) in &post_refs {
        let local = locals
            .get(&(tile, slot))
            .cloned()
            .ok_or_else(|| format!("post_ref tile {tile:?} slot {slot} missing local"))?;
        if carry_producers.contains(&(pc, ppos, pslot)) {
            let carry = carry_var_ident(pc, ppos, pslot);
            let rhs = if carries_with_init.contains(&(pc, ppos, pslot)) {
                quote! { #carry }
            } else {
                quote! { #carry.take().expect("init-less carry final") }
            };
            body.push(quote! {
                let #local: ::ferrite_cuda_core::alloc::OwnedTensor = #rhs;
            });
        } else {
            let last = last_var_ident(pc, ppos, pslot);
            body.push(quote! {
                let #local: ::ferrite_cuda_core::alloc::OwnedTensor =
                    #last.take().expect("periodic class last output");
            });
        }
    }

    for &c in &sched.post_loop {
        let sg = stencil.class_members[c][0];
        let imp_id = sfuf
            .impl_of(sg)
            .ok_or_else(|| format!("post_loop sg {sg:?} has no impl"))?;
        body.push(emit_subgraph(
            fuf,
            sfuf,
            program,
            model,
            lib,
            sg,
            imp_id,
            &locals,
            library,
            stencil,
            weight_layout,
        ));
    }

    let last_output = {
        let id = locals
            .get(&(last_tile, 0))
            .cloned()
            .ok_or_else(|| "last tile local missing".to_string())?;
        quote! { #id }
    };

    let fn_name = bucket_fn_ident("forward_m", wp);
    Ok(quote! {
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments, unused_mut, unused_variables)]
        pub unsafe fn #fn_name(
            wm: &Weights,
            ctx: &::ferrite_forward::ForwardCtx,
            device: &mut ::ferrite_cuda_core::device::GpuDevice,
        ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
            #(#body)*
            #last_output
        }
    })
}

/// Emit the bare `unsafe { __frag_N(…inputs…, repeat, wm, ctx, device) }`
/// call expression for one periodic class. The caller wraps this in
/// `let __cC_out = …` (full class — runs every iter) or
/// `if guard { __cC_out = Some(…); }` (short class — p1 offset/period).
///
/// Interns the fragment body in abstract mode with `repeat_var =
/// Some(repeat)` + an extra `repeat: usize` param. At the call site
/// `repeat` is bound to `__repeat - offset(consumer_class)` — the
/// class-local iter — so the fragment's `ctx.layer_expr` + weight
/// access read the right per-class element without baking the offset
/// into the fragment body. Offset 0 emits just `__repeat` verbatim
/// (byte-identical to 6.2.b.5e for uniform-period variants).
///
/// Input args come from the class's `ClassInputs` provenance:
/// - `IntraIter` → `(*__cPC_out).as_view()` (full producer) or
///   `(*__cPC_out.as_ref().unwrap()).as_view()` (short producer).
/// - `LoopCarry` → `(*__carry_P).as_view()` (producer class is in
///   `carry_producers`).
/// - `PreLoop` → `(*locals[&(tile, slot)]).as_view()`.
#[allow(clippy::too_many_arguments)]
fn emit_fragment_call_expr(
    consumer_class: usize,
    class_inputs: &ClassInputs,
    consumer_offset: usize,
    returned_exports: &[(u8, u8)],
    stencil: &StencilBundle,
    fuf: &Fuf,
    sfuf: &Assignment,
    program: &Program,
    model: &ModelParams,
    lib: &ImplementationLibrary,
    locals: &LocalMap,
    library: &mut FragmentLibrary,
    weight_layout: &crate::emit::WeightLayout,
    short_classes: &BTreeSet<usize>,
    carry_producers: &BTreeSet<(usize, u8, u8)>,
    carries_with_init: &BTreeSet<(usize, u8, u8)>,
    class_returned_exports: &[Vec<(u8, u8)>],
) -> Result<TokenStream, String> {
    let rep_sg = class_inputs.rep_sg;
    let imp_id = stencil.class_impl_id[consumer_class]
        .ok_or_else(|| format!("class {consumer_class} not homogeneous"))?;
    let imp = lib.get(imp_id);
    let claimed = sfuf.tiles_in_subgraph(rep_sg);
    let claimed_set: HashSet<TileId> = claimed.iter().copied().collect();

    // Boundary tile-input ordering must match `class_input_provenance`'s
    // dedup order so slot N in `class_inputs.slots` aligns with the
    // Nth `input_i` fragment param.
    let mut tile_params_ordered: Vec<(TileId, u8)> = Vec::new();
    let mut seen: HashSet<(TileId, u8)> = HashSet::new();
    for &t in &claimed {
        for input in &fuf.get(t).inputs {
            if let FufInput::Tile { id, slot } = input
                && !claimed_set.contains(id)
                && seen.insert((*id, *slot))
            {
                tile_params_ordered.push((*id, *slot));
            }
        }
    }
    if tile_params_ordered.len() != class_inputs.slots.len() {
        return Err(format!(
            "class {consumer_class}: boundary input count mismatch (emit={}, provenance={})",
            tile_params_ordered.len(),
            class_inputs.slots.len()
        ));
    }

    let accessors: Vec<WeightAccessor> = imp.required_weights(&claimed, fuf, program);

    // Build the abstract EmitCtx with `repeat_var = Some(repeat)` so
    // every `ctx.layer_expr` inside the fragment body refers to the
    // fragment's `repeat: usize` param (passed `__repeat` at the call
    // site). Tile/weight references go through the fragment fn-params
    // as usual.
    let mut tile_params: HashMap<(TileId, u8), syn::Ident> = HashMap::new();
    let mut tile_param_idents: Vec<syn::Ident> = Vec::with_capacity(tile_params_ordered.len());
    for (i, &(id, slot)) in tile_params_ordered.iter().enumerate() {
        let p = format_ident!("input_{}", i);
        tile_params.insert((id, slot), p.clone());
        tile_param_idents.push(p);
    }
    let mut weight_params: HashMap<(WeightId, Option<u64>), syn::Ident> = HashMap::new();
    let mut weight_params_by_name: HashMap<String, syn::Ident> = HashMap::new();
    let mut weight_param_idents: Vec<syn::Ident> = Vec::with_capacity(accessors.len());
    for (i, acc) in accessors.iter().enumerate() {
        let p = format_ident!("w_{}", i);
        for &(wid, idx) in &acc.source_weights {
            weight_params.insert((wid, idx), p.clone());
        }
        weight_params_by_name.insert(acc.name.to_string(), p.clone());
        weight_param_idents.push(p);
    }

    let repeat_ident = format_ident!("repeat");
    let abstract_ctx = EmitCtx {
        fuf,
        program,
        model,
        claimed_tiles: &claimed,
        locals,
        mode: EmitMode::Abstract {
            tile_params,
            consumed_params: HashMap::new(),
            weight_params,
            weight_params_by_name,
        },
        weight_layout: Some(weight_layout),
        repeat_var: Some(quote! { #repeat_ident }),
    };
    let abstract_body = imp.emit_call(&abstract_ctx);

    // Return shape follows `returned_exports`:
    //   - 0 entries: `()` — fragment is called for side effects only.
    //   - 1 entry : single `OwnedTensor` (same signature as 5f when
    //     pos = claim.len() - 1 and slot = 0).
    //   - ≥2 entries: tuple of `OwnedTensor` in (pos, slot) order.
    // The abstract body already binds `__out_<pos>_<slot>` for every
    // output of every claimed tile via `ctx.output_ident`, so the
    // fragment body just references the chosen exports' idents.
    let return_idents: Vec<syn::Ident> = returned_exports
        .iter()
        .map(|&(pos, slot)| format_ident!("__out_{}_{}", pos, slot))
        .collect();

    // Sig includes `collapsed` tag + extra repeat param + returned
    // export set so collapsed fragments never collide with the
    // unrolled path's interned fns or across classes with different
    // return shapes even if their abstract bodies stringify the same.
    let weight_types_str: String = accessors
        .iter()
        .map(|a| a.rust_type.to_string())
        .collect::<Vec<_>>()
        .join("|");
    let returned_str: String = returned_exports
        .iter()
        .map(|(p, s)| format!("{p}:{s}"))
        .collect::<Vec<_>>()
        .join(",");
    let sig = format!(
        "collapsed|ti={}|ci=0|wt={}|ret={}|body={}",
        tile_params_ordered.len(),
        weight_types_str,
        returned_str,
        abstract_body,
    );

    let frag_idx = if let Some(&idx) = library.by_sig.get(&sig) {
        idx
    } else {
        let idx = library.fns.len();
        let frag_name = format_ident!("__frag_{}", idx);

        let tile_param_decls: Vec<TokenStream> = tile_param_idents
            .iter()
            .map(|p| quote! { #p: ::ferrite_cuda_core::TensorView<'_> })
            .collect();
        let weight_param_decls: Vec<TokenStream> = accessors
            .iter()
            .zip(weight_param_idents.iter())
            .map(|(acc, p)| {
                let ty = &acc.rust_type;
                quote! { #p: & #ty }
            })
            .collect();

        let (return_ty, return_tail) = match return_idents.as_slice() {
            [] => (quote! { () }, quote! { () }),
            [one] => (
                quote! { ::ferrite_cuda_core::alloc::OwnedTensor },
                quote! { #one },
            ),
            many => {
                let tys: Vec<TokenStream> = many
                    .iter()
                    .map(|_| quote! { ::ferrite_cuda_core::alloc::OwnedTensor })
                    .collect();
                (quote! { ( #( #tys ),* ) }, quote! { ( #( #many ),* ) })
            }
        };

        let fn_tokens = quote! {
            #[cfg(feature = "cuda")]
            #[allow(clippy::too_many_arguments, unused_mut, unused_variables, non_snake_case)]
            unsafe fn #frag_name(
                #(#tile_param_decls,)*
                #(#weight_param_decls,)*
                #repeat_ident: usize,
                wm: &Weights,
                ctx: &::ferrite_forward::ForwardCtx,
                device: &mut ::ferrite_cuda_core::device::GpuDevice,
            ) -> #return_ty {
                #abstract_body
                #return_tail
            }
        };
        library.by_sig.insert(sig, idx);
        library.fns.push(fn_tokens);
        idx
    };
    CALL_SITE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Call site: assemble input args per provenance slot.
    let mut input_args: Vec<TokenStream> = Vec::with_capacity(class_inputs.slots.len());
    for (i, slot) in class_inputs.slots.iter().enumerate() {
        // Positional alignment: slot i of provenance == (id, slot) of
        // tile_params_ordered[i]. Assert to catch any divergence.
        if (slot.producer_tile, slot.producer_slot) != tile_params_ordered[i] {
            return Err(format!(
                "class {consumer_class}: provenance slot {i} mismatch (prov=({:?},{}), emit=({:?},{}))",
                slot.producer_tile,
                slot.producer_slot,
                tile_params_ordered[i].0,
                tile_params_ordered[i].1,
            ));
        }
        let arg = match &slot.origin {
            InputOrigin::IntraIter {
                producer_class,
                producer_pos,
            } => {
                // Verify the producer class actually returns this
                // export (collected into class_returned_exports).
                let export = (*producer_pos, slot.producer_slot);
                if !class_returned_exports[*producer_class].contains(&export) {
                    return Err(format!(
                        "class {consumer_class}: IntraIter producer class {producer_class} does not return export (pos {}, slot {})",
                        producer_pos, slot.producer_slot,
                    ));
                }
                let c_out = class_out_ident(*producer_class, *producer_pos, slot.producer_slot);
                if short_classes.contains(producer_class) {
                    quote! { (*#c_out.as_ref().expect("short-class producer")).as_view() }
                } else {
                    quote! { (*#c_out).as_view() }
                }
            }
            InputOrigin::LoopCarry {
                producer_class,
                producer_pos,
                ..
            } => {
                let key = (*producer_class, *producer_pos, slot.producer_slot);
                if !carry_producers.contains(&key) {
                    return Err(format!(
                        "class {consumer_class}: carry refers to class {producer_class} pos {} slot {} missing from carry set",
                        producer_pos, slot.producer_slot,
                    ));
                }
                let carry = carry_var_ident(*producer_class, *producer_pos, slot.producer_slot);
                if carries_with_init.contains(&key) {
                    quote! { (*#carry).as_view() }
                } else {
                    quote! { (*#carry.as_ref().expect("init-less carry read")).as_view() }
                }
            }
            InputOrigin::PreLoop { producer_sg } => {
                let init_tiles = sfuf.tiles_in_subgraph(*producer_sg);
                let init_tile = *init_tiles
                    .last()
                    .ok_or_else(|| format!("PreLoop producer sg {producer_sg:?} has no tiles"))?;
                let local = locals
                    .get(&(init_tile, slot.producer_slot))
                    .cloned()
                    .ok_or_else(|| {
                        format!(
                            "PreLoop producer local for ({init_tile:?}, {}) missing",
                            slot.producer_slot
                        )
                    })?;
                quote! { (*#local).as_view() }
            }
        };
        input_args.push(arg);
    }

    // Class-local repeat expression: `__repeat - offset`, or just
    // `__repeat` when offset is 0 (keeps the uniform-period emission
    // byte-identical to 6.2.b.5e).
    let repeat_tokens: TokenStream = if consumer_offset == 0 {
        quote! { __repeat }
    } else {
        let off_lit = proc_macro2::Literal::usize_unsuffixed(consumer_offset);
        quote! { (__repeat - #off_lit) }
    };

    let weight_args: Vec<TokenStream> = accessors
        .iter()
        .map(|acc| {
            let access = weight_layout.access_tokens_with_repeat(&acc.name, Some(&repeat_tokens));
            quote! { &wm.#access }
        })
        .collect();

    let frag_name = format_ident!("__frag_{}", frag_idx);
    Ok(quote! {
        unsafe {
            #frag_name(
                #(#input_args,)*
                #(#weight_args,)*
                #repeat_tokens,
                wm,
                ctx,
                device,
            )
        }
    })
}

/// Aliased-inline emission for one periodic class (§14.2).
///
/// Aliased-emittable impls (`FusedAddRmsNormImpl` family) mutate their
/// inputs in place and expose outputs as `TensorView<'_>` aliases — a
/// shape fragment fns can't return without lifetime parameters. The
/// collapsed emitter inlines the impl's `emit_call` directly inside
/// the loop body, with a custom local map that redirects the rep's
/// boundary tile idents to the collapsed-path per-export idents
/// (carry vars / intra-iter class outputs / pre-loop locals).
///
/// The inline body typically expands to:
///
/// ```ignore
/// unsafe { fused_add_rms_norm_inplace(*<delta_ident>, *<residual_ident>, ...); }
/// let __cC_out_<rms_pos>_0 = unsafe { (*<delta_ident>).as_view() };
/// let __cC_out_<add_pos>_0 = unsafe { (*<residual_ident>).as_view() };
/// ```
///
/// Callers must have populated `carry_producers` / `carries_with_init`
/// before invoking this helper — a LoopCarry-origin boundary whose
/// producer key is absent from `carry_producers` is a refusal.
#[allow(clippy::too_many_arguments)]
fn emit_aliased_class_inline(
    c: usize,
    class_inputs: &ClassInputs,
    stencil: &StencilBundle,
    fuf: &Fuf,
    sfuf: &Assignment,
    program: &Program,
    model: &ModelParams,
    lib: &ImplementationLibrary,
    locals: &LocalMap,
    weight_layout: &crate::emit::WeightLayout,
    carry_producers: &BTreeSet<(usize, u8, u8)>,
    carries_with_init: &BTreeSet<(usize, u8, u8)>,
    is_short: bool,
    offset: usize,
    period: usize,
    intraiter_refs: &BTreeSet<(usize, u8, u8)>,
    short_classes: &BTreeSet<usize>,
    _aliased_classes: &BTreeSet<usize>,
) -> Result<TokenStream, String> {
    let rep_sg = class_inputs.rep_sg;
    let imp_id =
        stencil.class_impl_id[c].ok_or_else(|| format!("aliased class {c} not homogeneous"))?;
    let imp = lib.get(imp_id);
    let claimed = sfuf.tiles_in_subgraph(rep_sg);
    let claimed_set: HashSet<TileId> = claimed.iter().copied().collect();

    // Boundary tile-input ordering must mirror what
    // `class_input_provenance` produced so provenance.slots[i] aligns
    // with tile_params_ordered[i].
    let mut tile_params_ordered: Vec<(TileId, u8)> = Vec::new();
    let mut seen: HashSet<(TileId, u8)> = HashSet::new();
    for &t in &claimed {
        for input in &fuf.get(t).inputs {
            if let FufInput::Tile { id, slot } = input
                && !claimed_set.contains(id)
                && seen.insert((*id, *slot))
            {
                tile_params_ordered.push((*id, *slot));
            }
        }
    }
    if tile_params_ordered.len() != class_inputs.slots.len() {
        return Err(format!(
            "aliased class {c}: boundary input count mismatch (emit={}, provenance={})",
            tile_params_ordered.len(),
            class_inputs.slots.len()
        ));
    }

    // Build the inline local map: overridden per-boundary idents plus
    // per-claimed-tile output idents (`__cC_out_<pos>_<slot>`).
    // Some origin kinds need a prelude unwrap before emit_call can
    // consume the boundary via `*#ident` (Option<OwnedTensor> carries
    // from period-mismatched consumers; Option<GpuTensor> from a
    // short-aliased IntraIter producer). The prelude binds a
    // `TensorView<'_>` ident that derefs to `GpuTensor` uniformly.
    let mut inline_locals: LocalMap = locals.clone();
    let mut prelude: Vec<TokenStream> = Vec::new();
    for (i, slot) in class_inputs.slots.iter().enumerate() {
        let (btile, bslot) = tile_params_ordered[i];
        let repl = match &slot.origin {
            InputOrigin::IntraIter {
                producer_class,
                producer_pos,
            } => {
                let c_out = class_out_ident(*producer_class, *producer_pos, slot.producer_slot);
                if short_classes.contains(producer_class) {
                    // Short producer: `__cN_out: Option<_>`. Unwrap
                    // into a TensorView so emit_call's `*#ident`
                    // pattern works uniformly. `.as_view()` is
                    // available on both `&OwnedTensor` (via Deref)
                    // and `&GpuTensor`.
                    let view = format_ident!(
                        "__intra_view_c{}_p{}_s{}",
                        producer_class,
                        producer_pos,
                        slot.producer_slot
                    );
                    let expect_msg = proc_macro2::Literal::string(&format!(
                        "aliased inline: short intra-iter producer c{} p{} s{} None",
                        producer_class, producer_pos, slot.producer_slot
                    ));
                    prelude.push(quote! {
                        let #view = unsafe {
                            #c_out
                                .as_ref()
                                .expect(#expect_msg)
                                .as_view()
                        };
                    });
                    view
                } else {
                    c_out
                }
            }
            InputOrigin::LoopCarry {
                producer_class,
                producer_pos,
                ..
            } => {
                let key = (*producer_class, *producer_pos, slot.producer_slot);
                if !carry_producers.contains(&key) {
                    return Err(format!(
                        "aliased class {c}: carry producer {key:?} missing from carry set"
                    ));
                }
                let carry = carry_var_ident(*producer_class, *producer_pos, slot.producer_slot);
                if carries_with_init.contains(&key) {
                    carry
                } else {
                    // `__carry: Option<OwnedTensor>`. Unwrap into a
                    // TensorView bound locally so emit_call's
                    // `*#ident` and `(*#ident).as_view()` patterns
                    // work uniformly. The `unsafe` is required by
                    // `GpuTensor::as_view`.
                    let view = format_ident!(
                        "__carry_view_c{}_p{}_s{}",
                        producer_class,
                        producer_pos,
                        slot.producer_slot
                    );
                    prelude.push(quote! {
                        let #view = unsafe {
                            #carry
                                .as_ref()
                                .expect("aliased inline: init-less carry read before producer fired")
                                .as_view()
                        };
                    });
                    view
                }
            }
            InputOrigin::PreLoop { .. } => locals
                .get(&(slot.producer_tile, slot.producer_slot))
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "aliased class {c}: PreLoop local for ({:?}, {}) missing",
                        slot.producer_tile, slot.producer_slot
                    )
                })?,
        };
        inline_locals.insert((btile, bslot), repl);
    }

    // Override each claimed tile's output slots to the right ident.
    // Full class: map directly to `__cC_out_<pos>_<slot>` so emit_call's
    // `let #out = ...` binds the per-iter TensorView that downstream
    // IntraIter consumers read in the same iter body. Short class:
    // map to per-iter temp idents `__alias_tmp_cC_p<P>_s<S>`; after
    // the body we assign each IntraIter/post-loop-referenced export
    // into the hoisted `Option<GpuTensor>` so reads outside the guard
    // still resolve (existing short-class reader uses
    // `.as_ref().expect(...).as_view()`, which works identically for
    // `Option<GpuTensor>` and `Option<OwnedTensor>`).
    let output_ident = |pos: u8, slot: u8| -> syn::Ident {
        if is_short {
            format_ident!("__alias_tmp_c{}_p{}_s{}", c, pos, slot)
        } else {
            class_out_ident(c, pos, slot)
        }
    };
    for (pos, &t) in claimed.iter().enumerate() {
        let node = fuf.get(t);
        let nslots = node.outputs.len().max(1) as u8;
        for s in 0..nslots {
            inline_locals.insert((t, s), output_ident(pos as u8, s));
        }
    }

    let repeat_tokens = quote! { __repeat };
    let ctx = EmitCtx {
        fuf,
        program,
        model,
        claimed_tiles: &claimed,
        locals: &inline_locals,
        mode: EmitMode::Concrete,
        weight_layout: Some(weight_layout),
        repeat_var: Some(repeat_tokens),
    };
    let body = imp.emit_call(&ctx);
    if !is_short {
        return Ok(quote! {
            #( #prelude )*
            #body
        });
    }

    // Short aliased: wrap in `if __repeat >= offset && __repeat < offset + period { .. }`
    // and assign each intraiter-referenced export's temp TensorView
    // into the hoisted `Option<GpuTensor>` (via `.as_raw()`).
    let off_lit = proc_macro2::Literal::usize_unsuffixed(offset);
    let per_lit = proc_macro2::Literal::usize_unsuffixed(period);
    let mut assigns: Vec<TokenStream> = Vec::new();
    for (pos, _t) in claimed.iter().enumerate() {
        let pos_u8 = pos as u8;
        // Every claimed tile has at least slot 0. Intraiter-referenced
        // exports always target slot indices that appear in the impl's
        // output_alias list; since aliased impls used today only
        // expose slot 0 per claimed tile, iterating slot 0 covers
        // every referenced export without needing an output_alias
        // walk here. (If a future aliased impl exposes slot > 0, the
        // referenced-export set will name that slot and the loop
        // below will miss it — add an output_alias walk then.)
        for s in 0u8..1 {
            if !intraiter_refs.contains(&(c, pos_u8, s)) {
                continue;
            }
            let tmp = output_ident(pos_u8, s);
            let out = class_out_ident(c, pos_u8, s);
            assigns.push(quote! {
                #out = ::std::option::Option::Some(#tmp.as_raw());
            });
        }
    }
    Ok(quote! {
        if __repeat >= #off_lit && __repeat < #off_lit + #per_lit {
            #( #prelude )*
            #body
            #( #assigns )*
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn emit_forward_backbone_for_bucket(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    program: &Program,
    model: &ModelParams,
    lib: &ImplementationLibrary,
    wp: crate::solver::WorkloadPoint,
    library: &mut FragmentLibrary,
    weight_layout: &crate::emit::WeightLayout,
) -> TokenStream {
    let locals = build_local_map(fuf);

    let Some(last_node) = fuf.nodes.last() else {
        // Empty FUF: degenerate, emit a stub that panics.
        let fn_name = bucket_fn_ident("forward_backbone_m", wp);
        return quote! {
            #[cfg(feature = "cuda")]
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn #fn_name(
                wm: &Weights,
                ctx: &::ferrite_forward::ForwardCtx,
                device: &mut ::ferrite_cuda_core::device::GpuDevice,
            ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
                unreachable!("forward_backbone: empty FUF")
            }
        };
    };
    let terminal_sg = sfuf
        .subgraph_of(last_node.id)
        .expect("terminal tile must be in a subgraph");

    let backbone_out: (crate::fuf::TileId, u8) = match last_node.inputs.first() {
        Some(FufInput::Tile { id, slot }) => (*id, *slot),
        _ => panic!(
            "forward_backbone: terminal tile's first input is not a Tile \
             (DSL must end in `gemm(<tile>, lm_head)`)"
        ),
    };
    let backbone_ident = locals[&backbone_out].clone();

    let mut protected: HashSet<(TileId, u8)> = HashSet::new();
    protected.insert(backbone_out);
    let stencil = StencilBundle::compute(fuf, sfuf);
    let body = emit_wave_walk(
        fuf,
        sfuf,
        loop_ir,
        program,
        model,
        lib,
        &locals,
        Some(terminal_sg),
        &protected,
        library,
        &stencil,
        weight_layout,
    );

    let fn_name = bucket_fn_ident("forward_backbone_m", wp);

    let bb_impl_names: String = loop_ir
        .waves
        .iter()
        .flat_map(|w| w.subgraphs.iter())
        .filter(|(sg, _)| Some(*sg) != Some(terminal_sg))
        .map(|(_, imp_id)| lib.get(*imp_id).name())
        .collect::<Vec<_>>()
        .join(", ");
    let bb_impl_names_lit = proc_macro2::Literal::string(&bb_impl_names);

    quote! {
        /// Backbone-only forward (no lm_head). Returns the output
        /// that would have been the final gemm's input — a freshly-
        /// allocated `OwnedTensor` so the caller owns the buffer
        /// independent of any in-fn alias.
        ///
        /// # Safety
        /// All tensors in `ctx` must be valid GPU memory; `device`
        /// must be the live CUDA device.
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments, unused_mut, unused_variables)]
        pub unsafe fn #fn_name(
            wm: &Weights,
            ctx: &::ferrite_forward::ForwardCtx,
            device: &mut ::ferrite_cuda_core::device::GpuDevice,
        ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
            ::tracing::debug!(
                m = ctx.input_ids.shape()[0],
                sk = ctx.max_seqlen_k,
                impls = #bb_impl_names_lit,
                "ferrite forward_backbone"
            );
            #(#body)*
            let __bb_view = unsafe { (*#backbone_ident).as_view() };
            let __bb_shape_u32: &[u32] = __bb_view.shape();
            let __bb_shape: ::std::vec::Vec<usize> =
                __bb_shape_u32.iter().map(|&d| d as usize).collect();
            let __bb_out = device
                .caching
                .alloc_tensor(&__bb_shape, __bb_view.dtype());
            ::ferrite_cuda_core::driver::memcpy_dtod_async(
                __bb_out.raw_ptr(),
                __bb_view.raw_ptr() as *const u8,
                __bb_view.size_bytes(),
                device.compute_stream,
            )
            .expect("forward_backbone: DtoD memcpy of output");
            __bb_out
        }
    }
}

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
pub fn emit_model(
    program: &Program,
    model: &ModelParams,
    fuf: &Fuf,
    sfufs: &WorkloadAssignments,
    loops: &WorkloadLoops,
    lib: &ImplementationLibrary,
    manifest: &crate::weights_manifest::WeightsManifest,
    canonical_override: Option<&Ident>,
) -> TokenStream {
    if let Some(canonical) = canonical_override {
        return emit_shim_model(program, fuf, sfufs, lib, model, manifest, canonical);
    }
    let (weights, weight_layout) = emit_weights_struct(
        program,
        fuf,
        sfufs,
        lib,
        model,
        manifest,
        WeightsEmitMode::Canonical,
    );

    // Group workload points by SFUF signature (sorted subgraph → impl).
    // Buckets with identical impl picks produce byte-identical fn
    // bodies, so we emit the full body ONCE at the canonical point and
    // emit the duplicates as thin `#[inline(always)]` shims that
    // delegate to the canonical fn. Public API (every
    // `forward_m_<M>[_sk_<SK>]` / `forward_backbone_m_<M>[_sk_<SK>]`
    // name a user might take a fn-pointer to) is preserved. Measured:
    // most models collapse 5 buckets → 2 unique SFUFs, cutting the
    // `quote!` work and the rustc-visible emitted body volume roughly
    // in half on those models. Extended to 2-D here: dedup runs over
    // `(num_tokens, sk_bucket)` points too, so models with `sk_buckets`
    // declared get the same compile-time win.
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

    // Per-model fragment library. `emit_forward_for_bucket` and
    // `emit_forward_backbone_for_bucket` both feed into it; the
    // resulting unique fn bodies get spliced into the emitted
    // module alongside the forward dispatchers.
    let mut fragment_library = FragmentLibrary::default();

    let mut bucket_fns: Vec<TokenStream> = Vec::with_capacity(bucket_points.len());
    let mut backbone_fns: Vec<TokenStream> = Vec::with_capacity(bucket_points.len());
    for (i, wp) in bucket_points.iter().enumerate() {
        let sfuf = &sfufs.per_workload[wp];
        let canonical = bucket_canonical[i];
        if canonical == *wp {
            let loop_ir = loops
                .per_workload
                .get(wp)
                .expect("schedule populated every key");
            bucket_fns.push(emit_forward_for_bucket(
                fuf,
                sfuf,
                loop_ir,
                program,
                model,
                lib,
                *wp,
                &mut fragment_library,
                &weight_layout,
            ));
            backbone_fns.push(emit_forward_backbone_for_bucket(
                fuf,
                sfuf,
                loop_ir,
                program,
                model,
                lib,
                *wp,
                &mut fragment_library,
                &weight_layout,
            ));
        } else {
            let fwd_name = bucket_fn_ident("forward_m", *wp);
            let fwd_target = bucket_fn_ident("forward_m", canonical);
            bucket_fns.push(quote! {
                #[cfg(feature = "cuda")]
                #[allow(clippy::too_many_arguments)]
                #[inline(always)]
                pub unsafe fn #fwd_name(
                    wm: &Weights,
                    ctx: &::ferrite_forward::ForwardCtx,
                    device: &mut ::ferrite_cuda_core::device::GpuDevice,
                ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
                    unsafe { #fwd_target(wm, ctx, device) }
                }
            });
            let bb_name = bucket_fn_ident("forward_backbone_m", *wp);
            let bb_target = bucket_fn_ident("forward_backbone_m", canonical);
            backbone_fns.push(quote! {
                #[cfg(feature = "cuda")]
                #[allow(clippy::too_many_arguments)]
                #[inline(always)]
                pub unsafe fn #bb_name(
                    wm: &Weights,
                    ctx: &::ferrite_forward::ForwardCtx,
                    device: &mut ::ferrite_cuda_core::device::GpuDevice,
                ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
                    unsafe { #bb_target(wm, ctx, device) }
                }
            });
        }
    }

    // `sk_axis_active` is true when the model declared `sk_buckets`;
    // otherwise all workload points have `sk_bucket == 0` and we
    // emit the pre-sk 1-D dispatch verbatim (no nested match, no
    // runtime `ctx.max_seqlen_k` lookup).
    let sk_axis_active = sfufs.per_workload.keys().any(|wp| wp.sk_bucket != 0);

    let num_tokens_points: Vec<u64> = sfufs.num_tokens_points();

    // Per-num_tokens set of sk buckets (sorted). Used to build both
    // the per-m inner match (sk → bucket fn) and the outer match
    // arm ranges.
    let mut sk_by_m: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for wp in sfufs.per_workload.keys() {
        sk_by_m.entry(wp.num_tokens).or_default().push(wp.sk_bucket);
    }
    for v in sk_by_m.values_mut() {
        v.sort();
        v.dedup();
    }

    // Build one outer match arm per compiled num_tokens. For
    // `sk_axis_active == false`, the arm is `lo..=hi => unsafe {
    // forward_m_<m>(...) }`. For `sk_axis_active == true`, the arm
    // is `lo..=hi => match sk_bucket { lo..=hi => forward_m_<m>_sk_<sk>(...), ... }`.
    let build_match_arms = |prefix: &str| -> (Vec<TokenStream>, Option<TokenStream>) {
        let arms: Vec<TokenStream> = num_tokens_points
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                let lo = proc_macro2::Literal::u64_unsuffixed(m);
                let range_tokens = if i + 1 == num_tokens_points.len() {
                    quote! { #lo.. }
                } else {
                    let next = num_tokens_points[i + 1];
                    let hi = proc_macro2::Literal::u64_unsuffixed(next - 1);
                    quote! { #lo..=#hi }
                };
                if sk_axis_active {
                    let sk_buckets = &sk_by_m[&m];
                    let sk_arms: Vec<TokenStream> = sk_buckets
                        .iter()
                        .enumerate()
                        .map(|(j, &sk)| {
                            let wp = crate::solver::WorkloadPoint {
                                num_tokens: m,
                                sk_bucket: sk,
                            };
                            let fn_name = bucket_fn_ident(prefix, wp);
                            let sk_lo = proc_macro2::Literal::u64_unsuffixed(sk);
                            if j + 1 == sk_buckets.len() {
                                quote! { #sk_lo.. => unsafe { #fn_name(wm, ctx, device) }, }
                            } else {
                                let next_sk = sk_buckets[j + 1];
                                let sk_hi = proc_macro2::Literal::u64_unsuffixed(next_sk - 1);
                                quote! { #sk_lo..=#sk_hi => unsafe { #fn_name(wm, ctx, device) }, }
                            }
                        })
                        .collect();
                    let fallback_wp = crate::solver::WorkloadPoint {
                        num_tokens: m,
                        sk_bucket: sk_buckets[0],
                    };
                    let fallback_name = bucket_fn_ident(prefix, fallback_wp);
                    quote! {
                        #range_tokens => {
                            let sk_bucket_runtime = ctx.max_seqlen_k as u64;
                            match sk_bucket_runtime {
                                #(#sk_arms)*
                                _ => unsafe { #fallback_name(wm, ctx, device) },
                            }
                        },
                    }
                } else {
                    let wp = crate::solver::WorkloadPoint::num_tokens_only(m);
                    let fn_name = bucket_fn_ident(prefix, wp);
                    quote! { #range_tokens => unsafe { #fn_name(wm, ctx, device) }, }
                }
            })
            .collect();
        let fallback_arm = num_tokens_points.first().map(|&m| {
            if sk_axis_active {
                let sk_buckets = &sk_by_m[&m];
                let wp = crate::solver::WorkloadPoint {
                    num_tokens: m,
                    sk_bucket: sk_buckets[0],
                };
                let fn_name = bucket_fn_ident(prefix, wp);
                quote! { _ => unsafe { #fn_name(wm, ctx, device) }, }
            } else {
                let wp = crate::solver::WorkloadPoint::num_tokens_only(m);
                let fn_name = bucket_fn_ident(prefix, wp);
                quote! { _ => unsafe { #fn_name(wm, ctx, device) }, }
            }
        });
        (arms, fallback_arm)
    };

    let (match_arms, fallback_arm) = build_match_arms("forward_m");
    let (backbone_match_arms, backbone_fallback_arm) = build_match_arms("forward_backbone_m");

    // Per-model fragment library: each unique kernel-call body
    // emitted once as `__frag_<N>`. Forward/backbone bodies are a
    // sequence of one-line `let t_X = unsafe { __frag_N(...) };`
    // dispatches against this library, so multi-layer transformers
    // stop emitting the same rmsnorm / gemm / silu etc body once
    // per layer.
    let fragment_fns = fragment_library.into_fns();
    eprintln!(
        "    codegen-profile: {} fragments emitted, {} call sites, {} inline-fallback sites",
        fragment_fns.len(),
        CALL_SITE_COUNT.load(std::sync::atomic::Ordering::Relaxed),
        INLINE_FALLBACK_COUNT.load(std::sync::atomic::Ordering::Relaxed),
    );
    CALL_SITE_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    INLINE_FALLBACK_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);

    quote! {
        #weights

        #(#fragment_fns)*

        #(#bucket_fns)*
        #(#backbone_fns)*

        /// Dispatch on `(num_tokens, sk_bucket)`. Outer match is on
        /// `num_tokens`; inner match (when the model's `#[forward]`
        /// declared `sk_buckets`) picks the kernel specialized for
        /// the current KV-cache span. Runtime points that don't
        /// exactly equal a compiled point get the specialization for
        /// the largest compiled bucket ≤ runtime — correct (kernels
        /// work at any value) if potentially suboptimal for cost.
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward(
            wm: &Weights,
            ctx: &::ferrite_forward::ForwardCtx,
            device: &mut ::ferrite_cuda_core::device::GpuDevice,
            num_tokens: u64,
        ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
            match num_tokens {
                #(#match_arms)*
                #fallback_arm
            }
        }

        /// Backbone-only dispatch (no lm_head). Same 2-axis
        /// dispatch structure as [`forward`].
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward_backbone(
            wm: &Weights,
            ctx: &::ferrite_forward::ForwardCtx,
            device: &mut ::ferrite_cuda_core::device::GpuDevice,
            num_tokens: u64,
        ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
            match num_tokens {
                #(#backbone_match_arms)*
                #backbone_fallback_arm
            }
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
fn emit_shim_model(
    program: &Program,
    fuf: &Fuf,
    sfufs: &WorkloadAssignments,
    lib: &ImplementationLibrary,
    model: &ModelParams,
    manifest: &crate::weights_manifest::WeightsManifest,
    canonical: &Ident,
) -> TokenStream {
    let (weights, _weight_layout) = emit_weights_struct(
        program,
        fuf,
        sfufs,
        lib,
        model,
        manifest,
        WeightsEmitMode::Shim { canonical },
    );

    // Per-bucket fn names the canonical emits. We re-export each
    // by name so downstream code that takes a fn pointer to
    // `<shim>::forward_m_64` resolves through to the canonical's
    // compiled body without any additional fn-pointer indirection.
    let mut bucket_names: Vec<Ident> = Vec::new();
    for wp in sfufs.per_workload.keys() {
        bucket_names.push(bucket_fn_ident("forward_m", *wp));
        bucket_names.push(bucket_fn_ident("forward_backbone_m", *wp));
    }
    // Deterministic order for build reproducibility.
    bucket_names.sort_by_key(|a| a.to_string());
    bucket_names.dedup_by(|a, b| a == b);

    quote! {
        #weights

        // Top-level dispatchers + per-bucket fns all live on the
        // canonical sibling; re-export by name so `<shim>::forward`
        // and `<shim>::forward_m_64` both resolve transparently to
        // the canonical's compiled body.
        #[cfg(feature = "cuda")]
        pub use super::#canonical::{forward, forward_backbone};
        #[cfg(feature = "cuda")]
        pub use super::#canonical::{#(#bucket_names),*};
    }
}
