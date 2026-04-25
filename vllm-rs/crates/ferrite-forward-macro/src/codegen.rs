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
use crate::impl_lib::{ImplementationLibrary, WeightAccessor};
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
fn safetensors_prefix(program: &Program, id: WeightId, index: Option<u64>) -> String {
    let joined = program.weights.path(id).join(".");
    match (index, joined.as_str()) {
        (_, "lm_head") => "lm_head".to_string(),
        // DeepSeek: the `moe` DSL name maps to `mlp` in HF safetensors.
        (Some(l), "moe") => format!("model.layers.{l}.mlp"),
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
    let is_deepseek_v2_moe = ty.ends_with("::DeepSeekV2MoELayer")
        || ty == "DeepSeekV2MoELayer"
        || ty.ends_with("layers_moe::DeepSeekV2MoELayer");
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
    // MLA archs (DeepSeek V3) have `q_a_proj` instead of `q_proj`
    // on disk — use that as the fingerprint leaf for these archs.
    let fp_leaf: &str = if manifest.packed_splits.contains_key("self_attn.qkv_proj") {
        "self_attn.qkv_proj"
    } else if manifest.entries.contains_key("self_attn.q_a_proj") {
        "self_attn.q_a_proj"
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
    let bnb4_marker_tensor = format!("model.layers.0.{fp_leaf}.weight.absmax");
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
    let fp8_marker_tensor = format!("model.layers.0.{fp_leaf}.weight_scale");
    let fp8_marker_tensor = fp8_marker_tensor.as_str();

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
        Some(crate::config::RopeScaling::Yarn { .. }) => Some("yarn"),
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
) -> TokenStream {
    let accessors = match collect_accessors(program, fuf, sfufs, lib) {
        Ok(a) => a,
        Err(err) => return err,
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
        // FP8 accessors come through `fp8_accessor_type_for`, which
        // post-Fp8AnyLinear-unblocker always returns `Fp8AnyLinear`.
        // The storage-format guard accepts that type for both
        // per-tensor / per-channel (`block_size: None`) and
        // blockwise (`block_size: Some(_)`) FP8 storage — the
        // Fp8AnyLinear enum dispatches at runtime on the loaded
        // variant.
        let accessor_is_fp8_any = ty.ends_with("::Fp8AnyLinear")
            || ty == "Fp8AnyLinear"
            || ty.ends_with("layers::Fp8AnyLinear");
        for (wid, _idx) in &a.source_weights {
            let fmt = crate::quantization::storage_format_for_weight(program, fuf, *wid, model);
            let ok = matches!(
                (&fmt, accessor_is_marlin, accessor_is_bnb4, accessor_is_fp8_any),
                (crate::quantization::StorageFormat::Dense, false, false, false)
                    | (crate::quantization::StorageFormat::Awq { .. }, true, false, false)
                    | (crate::quantization::StorageFormat::Gptq { .. }, true, false, false)
                    | (crate::quantization::StorageFormat::Bnb4 { .. }, false, true, false)
                    | (crate::quantization::StorageFormat::Fp8 { .. }, false, false, true),
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

    let fields = accessors.iter().map(|a| {
        let name = &a.name;
        let ty = &a.rust_type;
        quote! { pub #name: #ty, }
    });

    // Emit each field as its own let-binding in the load body.
    // This lets later loaders reference earlier ones (e.g. a tied
    // `lm_head` reads `embed_tokens.weight`). Field order inside
    // Self { .. } is irrelevant to Rust; let-binding order is what
    // matters. `accessors` iterates BTreeMap-sorted — which puts
    // `embed_tokens` before `lm_head` alphabetically, so the tied
    // case works without a special sort.
    let plans: Vec<FieldLoad> = accessors
        .iter()
        .map(|a| plan_field_load(a, program, fuf, model, manifest))
        .collect();
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

    // `Self { a, b, c }` shorthand — fields are the just-bound
    // locals, in the same order we declared the struct fields.
    let field_shorthand: Vec<&syn::Ident> = accessors.iter().map(|a| &a.name).collect();
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
        // For YaRN (DeepSeek V2 MLA), the rope portion uses `qk_rope_head_dim`
        // rather than the full `head_dim`. Pull it from bounds if present.
        let yarn_rope_head_dim: Option<usize> =
            model.bounds.get("qk_rope_head_dim").map(|&v| v as usize);
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
    match &mode {
        WeightsEmitMode::Canonical => quote! {
            #weights_def

            #accessor_methods

            #rotary_cos_sin_methods

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

/// Emit one accessor method per base name on the per-arch `Weights`
/// struct. Layered bases (`input_layernorm_0`, `input_layernorm_1`,
/// …) collapse into one `pub fn input_layernorm(&self, layer: u32) ->
/// &RmsNorm` whose body matches on `layer` and returns `&self.<field>`
/// for the corresponding generated field. Non-layered bases get the
/// same signature for caller uniformity; the body returns
/// `&self.<field>` and ignores the layer arg.
///
/// Returns an `impl Weights { ... }` block. Empty (zero accessors)
/// is fine — the impl block is then empty.
fn emit_weights_accessor_methods(accessors: &[WeightAccessor]) -> TokenStream {
    use std::collections::BTreeMap;

    struct Spec {
        rust_type: TokenStream,
        rust_type_str: String,
        by_layer: BTreeMap<u64, syn::Ident>,
        nonlayered: Option<syn::Ident>,
    }

    let mut by_base: BTreeMap<String, Spec> = BTreeMap::new();
    for acc in accessors {
        let full = acc.name.to_string();
        let (base, layer) = split_base_layer(&full);
        let ty_str = acc.rust_type.to_string();
        let entry = by_base.entry(base.clone()).or_insert_with(|| Spec {
            rust_type: acc.rust_type.clone(),
            rust_type_str: ty_str.clone(),
            by_layer: BTreeMap::new(),
            nonlayered: None,
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
                entry.by_layer.insert(n, acc.name.clone());
            }
            None => {
                entry.nonlayered = Some(acc.name.clone());
            }
        }
    }

    let methods: Vec<TokenStream> = by_base
        .iter()
        .map(|(base, spec)| {
            let base_ident = syn::Ident::new(base, proc_macro2::Span::call_site());
            let ty = &spec.rust_type;
            match (&spec.nonlayered, spec.by_layer.is_empty()) {
                (Some(field), true) => quote! {
                    #[cfg(feature = "cuda")]
                    #[inline]
                    #[allow(dead_code)]
                    pub fn #base_ident(&self, _layer: u32) -> &#ty {
                        &self.#field
                    }
                },
                (None, false) => {
                    let arms: Vec<TokenStream> = spec
                        .by_layer
                        .iter()
                        .map(|(layer, fname)| {
                            let lit = proc_macro2::Literal::u32_unsuffixed(*layer as u32);
                            quote! { #lit => &self.#fname, }
                        })
                        .collect();
                    quote! {
                        #[cfg(feature = "cuda")]
                        #[inline]
                        #[allow(dead_code)]
                        pub fn #base_ident(&self, layer: u32) -> &#ty {
                            match layer {
                                #(#arms)*
                                // Codegen guarantees `layer` is one of the
                                // emitted indices — every static-slice row
                                // carries a literal `layer:` field minted
                                // from the same per-arch fan_out pass.
                                _ => unsafe { ::core::hint::unreachable_unchecked() },
                            }
                        }
                    }
                }
                _ => panic!("Weights accessor base `{base}` mixes layered and non-layered fields"),
            }
        })
        .collect();

    if methods.is_empty() {
        return quote! {};
    }

    quote! {
        #[cfg(feature = "cuda")]
        impl Weights {
            #(#methods)*
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

/// Resolve the `(TileId, u8)` whose `OwnedTensor` is the backbone's
/// "return value" — the input the terminal `gemm(<tile>, lm_head)`
/// would have read. Panics if the FUF doesn't end in a tile-input
/// terminal, since `forward_backbone` has no defined behavior for
/// architectures whose terminal is anything other than `gemm(...,
/// lm_head)`.
fn backbone_output_for(fuf: &Fuf) -> (TileId, u8) {
    let last_node = fuf
        .nodes
        .last()
        .expect("FUF must be non-empty to emit a forward fn");
    match last_node.inputs.first() {
        Some(FufInput::Tile { id, slot }) => (*id, *slot),
        _ => panic!(
            "forward_backbone: terminal tile's first input is not a Tile \
             (DSL must end in `gemm(<tile>, lm_head)`)"
        ),
    }
}

/// Build the workload-point bounds map `lower_bucket` and Impls
/// consume — model.bounds + the workload-specific `num_tokens` and
/// `sk_bucket` overrides. Mirrors what `solve_workloads` does before
/// each per-point solve.
fn bounds_for_wp(model: &ModelParams, wp: crate::solver::WorkloadPoint) -> BTreeMap<String, u64> {
    let mut bounds = model.bounds.clone();
    bounds.insert("num_tokens".to_string(), wp.num_tokens);
    bounds.insert("sk_bucket".to_string(), wp.sk_bucket);
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
    let weights = emit_weights_struct(
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
    let last_node_id = fuf.nodes.last().expect("non-empty FUF expected").id;
    let backbone_out = backbone_output_for(fuf);
    for (i, wp) in bucket_points.iter().enumerate() {
        if bucket_canonical[i] != *wp {
            continue;
        }
        let sfuf = &sfufs.per_workload[wp];
        let loop_ir = loops
            .per_workload
            .get(wp)
            .expect("schedule populated every key");
        let bounds = bounds_for_wp(model, *wp);
        let terminal_sg = sfuf
            .subgraph_of(last_node_id)
            .expect("terminal tile must be in a subgraph");

        // Backbone — protect backbone_out (carries through to lm_head
        // OR the DtoD copy) AND the terminal slot (so backbone's drop
        // pass leaves it for lm_head to write).
        let mut protected_bb: HashSet<(TileId, u8)> = HashSet::new();
        protected_bb.insert(backbone_out);
        protected_bb.insert((last_node_id, 0));

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
        let terminal_slot = slots.of(last_node_id, 0);
        let num_slots = slots.total();

        let lowered_bb = lower_bucket(
            fuf,
            sfuf,
            loop_ir,
            program,
            model,
            lib,
            &bounds,
            Some(terminal_sg),
            &protected_bb,
            &mut arch_opcodes,
            backbone_out,
            &slots,
        );

        // LM_HEAD — one row, computed by directly invoking the
        // terminal subgraph's `fan_out` against the same slot map.
        // No aliases, no drops, no recursion — terminal is the last
        // subgraph in topological order.
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
        arch_opcodes.register(term_imp.opcode_shape(), term_imp.interpreter_arm(model));
        let lowered_lm = crate::interpreter_codegen::LoweredBucket {
            instances: term_emits,
            num_slots,
            final_slot: terminal_slot,
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

    // Arch-wide constant extraction: any field on a variant whose
    // value is byte-identical across every instance (across every
    // canonical bucket's forward + backbone slice) gets dropped from
    // the variant + the rows, and rebound to its constant value at
    // the top of the variant's match-arm body. This is what stops
    // commandr's QkvRopeCache rows from carrying `interleaved: true`,
    // `biased: false`, `cos_sin_fn: Weights::rotary_cos_sin`, and
    // (for Impls picked for a single accessor) `weight_fn:
    // Weights::self_attn_qkv` on every single row.
    {
        let mut refs: Vec<&mut crate::interpreter_codegen::LoweredBucket> = Vec::new();
        for (cl, _, _, _) in canonical_lowered.values_mut() {
            refs.push(&mut cl.backbone);
            refs.push(&mut cl.lm_head);
        }
        crate::interpreter_codegen::extract_arch_wide_constants(&mut arch_opcodes, &mut refs);
    }

    // Layer-template detection: collapse the contiguous repeating
    // sub-sequence of the slice (the per-layer transformer body)
    // into one `Op::Loop(N, body_len)` row + one iteration's body.
    // Fused boundary effects (e.g., FusedAddRmsNorm absorbing layer
    // L's final add into layer L+1's first norm) leave layer 0 / the
    // last layer structurally distinct, so the detection picks the
    // largest CONTIGUOUS run that genuinely repeats — middle layers
    // — and keeps the boundary residues as straight-line code in
    // prelude/suffix.
    for (cl, _, _, _) in canonical_lowered.values_mut() {
        crate::interpreter_codegen::apply_loop_compression(
            &arch_opcodes,
            &mut cl.backbone,
            "layer",
        );
        crate::interpreter_codegen::apply_loop_compression(&arch_opcodes, &mut cl.lm_head, "layer");
    }

    // One per-arch opcode enum + one per-arch interpreter helper
    // for the module. Both private to the module.
    let enum_ident = format_ident!("Op");
    let helper_ident = format_ident!("__interpret");
    let arch_enum_ts = arch_opcodes.emit_enum(&enum_ident);
    let arch_interpreter_ts = arch_opcodes.emit_interpreter(&helper_ident, &enum_ident);
    let shapes_by_name = arch_opcodes.shapes_by_name();

    // Per-bucket statics + per-bucket fns. Canonical buckets get a
    // freshly-emitted body; non-canonical buckets get a thin shim
    // delegating to the canonical's forward fn.
    let mut static_slices: Vec<TokenStream> = Vec::new();
    let mut bucket_fns: Vec<TokenStream> = Vec::with_capacity(bucket_points.len());
    let mut backbone_fns: Vec<TokenStream> = Vec::with_capacity(bucket_points.len());
    for (i, wp) in bucket_points.iter().enumerate() {
        let canonical = bucket_canonical[i];
        if canonical == *wp {
            let (lowered, num_slots_val, backbone_slot_val, terminal_slot_val) =
                &canonical_lowered[wp];

            let backbone_static_ident = bucket_static_ident("BACKBONE_M", *wp);
            let lm_head_static_ident = bucket_static_ident("LM_HEAD_M", *wp);
            static_slices.push(emit_bucket_static_slice(
                &backbone_static_ident,
                &enum_ident,
                &shapes_by_name,
                &lowered.backbone.instances,
            ));
            static_slices.push(emit_bucket_static_slice(
                &lm_head_static_ident,
                &enum_ident,
                &shapes_by_name,
                &lowered.lm_head.instances,
            ));

            let num_slots = proc_macro2::Literal::u32_unsuffixed(*num_slots_val);
            let bb_final_slot = proc_macro2::Literal::u32_unsuffixed(*backbone_slot_val);
            let fwd_final_slot = proc_macro2::Literal::u32_unsuffixed(*terminal_slot_val);

            // forward = backbone + one lm_head step.
            let fwd_fn_name = bucket_fn_ident("forward_m", *wp);
            bucket_fns.push(quote! {
                #[cfg(feature = "cuda")]
                #[allow(clippy::too_many_arguments, unused_mut, unused_variables)]
                pub unsafe fn #fwd_fn_name(
                    wm: &Weights,
                    ctx: &::ferrite_forward::ForwardCtx,
                    device: &mut ::ferrite_cuda_core::device::GpuDevice,
                ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
                    let mut __tiles: ::std::vec::Vec<Option<::ferrite_forward::TileEntry>> =
                        (0u32..#num_slots).map(|_| None).collect();
                    unsafe {
                        #helper_ident(#backbone_static_ident, &mut __tiles, wm, ctx, device);
                        #helper_ident(#lm_head_static_ident, &mut __tiles, wm, ctx, device);
                    }
                    ::ferrite_forward::take_owned(&mut __tiles, #fwd_final_slot)
                }
            });

            // forward_backbone = backbone, then DtoD-copy the
            // backbone-output slot into a freshly-allocated
            // OwnedTensor so the caller owns the buffer.
            let bb_fn_name = bucket_fn_ident("forward_backbone_m", *wp);
            backbone_fns.push(quote! {
                #[cfg(feature = "cuda")]
                #[allow(clippy::too_many_arguments, unused_mut, unused_variables)]
                pub unsafe fn #bb_fn_name(
                    wm: &Weights,
                    ctx: &::ferrite_forward::ForwardCtx,
                    device: &mut ::ferrite_cuda_core::device::GpuDevice,
                ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
                    let mut __tiles: ::std::vec::Vec<Option<::ferrite_forward::TileEntry>> =
                        (0u32..#num_slots).map(|_| None).collect();
                    unsafe {
                        #helper_ident(#backbone_static_ident, &mut __tiles, wm, ctx, device);
                    }
                    let __bb_view = unsafe {
                        ::ferrite_forward::tile_ref(&__tiles, #bb_final_slot).as_view(&__tiles)
                    };
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
            });
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
                // M=1 must be an exclusive range: the solver may pick
                // M=1-only kernels (cutlass_gemv) that fail at M>1.
                // Start the *next* bucket at 2 so M=2..next routes there.
                let lo = if i > 0 && num_tokens_points[0] == 1 && num_tokens_points[i - 1] == 1 {
                    proc_macro2::Literal::u64_unsuffixed(2)
                } else {
                    proc_macro2::Literal::u64_unsuffixed(m)
                };
                let range_tokens = if m == 1 {
                    let one = proc_macro2::Literal::u64_unsuffixed(1);
                    quote! { #one..=#one }
                } else if i + 1 == num_tokens_points.len() {
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

    quote! {
        #weights

        #arch_enum_ts

        #arch_interpreter_ts

        #(#static_slices)*

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
    let weights = emit_weights_struct(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impl_lib::WeightAccessor;
    use quote::format_ident;

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
    fn accessor_methods_collapse_per_layer_fields_into_one_method() {
        // Three layers' worth of `input_layernorm_<n>` fields should
        // produce one `pub fn input_layernorm(&self, layer: u32) ->
        // &RmsNorm` with three match arms.
        let ty: TokenStream = quote! { ::ferrite_kernels::layers::RmsNorm };
        let accessors = (0u64..3)
            .map(|n| WeightAccessor {
                name: format_ident!("input_layernorm_{}", n),
                rust_type: ty.clone(),
                source_weights: vec![],
            })
            .collect::<Vec<_>>();
        let ts = emit_weights_accessor_methods(&accessors).to_string();
        // One impl, one fn, three arms (0/1/2 → &self.input_layernorm_<n>).
        assert!(ts.contains("impl Weights"));
        assert!(ts.contains("fn input_layernorm"));
        assert!(ts.contains("layer : u32"));
        assert!(ts.contains("& self . input_layernorm_0"));
        assert!(ts.contains("& self . input_layernorm_1"));
        assert!(ts.contains("& self . input_layernorm_2"));
        // The catch-all is `unreachable_unchecked()` — codegen
        // guarantees `layer` is one of the registered indices, so no
        // runtime panic, no format-args bloat in the expanded crate.
        assert!(ts.contains("unreachable_unchecked"));
        assert!(!ts.contains("out of range"));
        assert!(!ts.contains("panic"));
    }

    #[test]
    fn accessor_methods_emit_unit_arm_for_unlayered_fields() {
        // `embed_tokens` has no trailing layer index; the method
        // ignores its layer arg and returns the field directly.
        let accessors = vec![WeightAccessor {
            name: format_ident!("embed_tokens"),
            rust_type: quote! { ::ferrite_kernels::layers::Embedding },
            source_weights: vec![],
        }];
        let ts = emit_weights_accessor_methods(&accessors).to_string();
        assert!(ts.contains("fn embed_tokens"));
        assert!(ts.contains("_layer : u32"));
        assert!(ts.contains("& self . embed_tokens"));
        // No `match` block for non-layered accessors — the body is
        // a direct field reference, branchless.
        assert!(!ts.contains("match layer"));
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
                rust_type: quote! { ::ferrite_kernels::layers::CohereLayerNorm },
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
}
