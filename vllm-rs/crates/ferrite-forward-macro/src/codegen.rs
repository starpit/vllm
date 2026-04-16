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

use crate::classified::{OpKind, Program, WeightId};
use crate::config::ModelParams;
use crate::emit::{EmitCtx, LocalMap};
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{ImplementationLibrary, WeightAccessor};
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
    let path = program.weights.path(id);
    let dotted: Vec<String> = path.iter().map(|s| s.to_string()).collect();
    let joined = dotted.join(".");
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
    /// `LinearLayer::load_dense(gw, prefix)`.
    LinearDense(String),
    /// `LinearLayer::load_dense_concat(gw, &[prefix0, prefix1, ...], stream)`.
    LinearConcat(Vec<String>),
    /// The model has `tie_word_embeddings: true`: `lm_head` shares
    /// its weight with `embed_tokens`. No safetensors read — build
    /// the `LinearLayer` from the already-loaded embedding field
    /// whose name is carried here.
    LinearTiedToEmbedding(syn::Ident),
    /// AWQ-packed INT4 linear. `group_size` is read from the
    /// model's `quantization_config.group_size`. Emits
    /// `MarlinLinear::load_awq(gw, prefix, group_size, workspace,
    /// device_id)` against the ambient `__marlin_ws` / `__device_id`
    /// bindings that [`emit_weights_struct`] plants at the top of
    /// `Weights::load` when any AWQ accessor is present.
    AwqLinear { prefix: String, group_size: u32 },
    /// AWQ fused linear (QKV, gate/up) — AWQ qweight/scales/qzeros
    /// all concat along dim N, so the fused accessor collapses to
    /// one `MarlinLinear`. Emits `MarlinLinear::load_awq_concat`.
    AwqLinearConcat {
        prefixes: Vec<String>,
        group_size: u32,
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
) -> FieldLoad {
    let ty = accessor.rust_type.to_string().replace(' ', "");
    let is_embedding =
        ty.ends_with("::Embedding") || ty == "Embedding" || ty.ends_with("layers::Embedding");
    let is_rmsnorm =
        ty.ends_with("::RmsNorm") || ty == "RmsNorm" || ty.ends_with("layers::RmsNorm");
    let is_linear =
        ty.ends_with("::LinearLayer") || ty == "LinearLayer" || ty.ends_with("layers::LinearLayer");
    let is_marlin = ty.ends_with("::MarlinLinear")
        || ty == "MarlinLinear"
        || ty.ends_with("layers::MarlinLinear");

    let prefixes: Vec<String> = accessor
        .source_weights
        .iter()
        .map(|(id, idx)| safetensors_prefix(program, *id, *idx))
        .collect();

    if is_marlin {
        // AWQ accessors are always emitted by a quant-aware impl
        // whose sources are `StorageFormat::Awq { group_size, .. }`.
        // The `group_size` must agree across every source of a fused
        // accessor (HF's fused-QKV/gate-up layers share one
        // group_size); mismatch is a data-integrity error in the
        // upstream HF repo and we panic at macro-expansion time
        // rather than silently emit a wrong loader.
        let mut group_size: Option<u32> = None;
        for (wid, _idx) in &accessor.source_weights {
            let fmt = crate::quantization::storage_format_for_weight(program, fuf, *wid, model);
            match fmt {
                crate::quantization::StorageFormat::Awq { group_size: g, .. } => match group_size {
                    None => group_size = Some(g),
                    Some(existing) if existing == g => {}
                    Some(existing) => panic!(
                        "accessor `{}` fuses sources with mismatched AWQ group_size \
                             (saw {existing} then {g})",
                        accessor.name,
                    ),
                },
                other => panic!(
                    "accessor `{}` declared `MarlinLinear` but source weight resolves to \
                     non-Awq storage ({other:?}) — solver picked a Marlin impl for a dense \
                     weight, which is a matcher bug",
                    accessor.name,
                ),
            }
        }
        let group_size =
            group_size.expect("MarlinLinear accessor declares at least one source weight");
        if prefixes.len() == 1 {
            return FieldLoad::AwqLinear {
                prefix: prefixes.into_iter().next().unwrap(),
                group_size,
            };
        } else {
            return FieldLoad::AwqLinearConcat {
                prefixes,
                group_size,
            };
        }
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

    for sfuf in sfufs.per_num_tokens.values() {
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

/// Emit the `Weights` struct definition + its `load` method.
fn emit_weights_struct(
    program: &Program,
    fuf: &Fuf,
    sfufs: &WorkloadAssignments,
    lib: &ImplementationLibrary,
    model: &ModelParams,
) -> TokenStream {
    let accessors = match collect_accessors(program, fuf, sfufs, lib) {
        Ok(a) => a,
        Err(err) => return err,
    };

    // Storage-format guard: a given accessor's `rust_type` must be
    // compatible with every one of its source weights' storage
    // formats. The allowed pairs today:
    //   `LinearLayer`  ↔ `Dense`
    //   `Embedding`    ↔ `Dense`
    //   `RmsNorm`      ↔ `Dense`
    //   `MarlinLinear` ↔ `Awq { .. }`
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
        for (wid, _idx) in &a.source_weights {
            let fmt = crate::quantization::storage_format_for_weight(program, fuf, *wid, model);
            let ok = matches!(
                (&fmt, accessor_is_marlin),
                (crate::quantization::StorageFormat::Dense, false)
                    | (crate::quantization::StorageFormat::Awq { .. }, true),
            );
            if !ok {
                let path = program.weights.path(*wid);
                let dotted = path
                    .iter()
                    .map(syn::Ident::to_string)
                    .collect::<Vec<_>>()
                    .join(".");
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
        .map(|a| plan_field_load(a, program, fuf, model))
        .collect();
    let any_awq = plans.iter().any(|p| {
        matches!(
            p,
            FieldLoad::AwqLinear { .. } | FieldLoad::AwqLinearConcat { .. }
        )
    });
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
                FieldLoad::AwqLinear { prefix, group_size } => {
                    let group_size = *group_size as usize;
                    quote! {
                        let #name = ::ferrite_kernels::layers::MarlinLinear::load_awq(
                            gw,
                            #prefix,
                            #group_size,
                            __marlin_ws,
                            __device_id,
                        )?;
                    }
                }
                FieldLoad::AwqLinearConcat {
                    prefixes,
                    group_size,
                } => {
                    let group_size = *group_size as usize;
                    quote! {
                        let #name = ::ferrite_kernels::layers::MarlinLinear::load_awq_concat(
                            gw,
                            &[ #(#prefixes),* ],
                            #group_size,
                            __marlin_ws,
                            __device_id,
                        )?;
                    }
                }
            }
        })
        .collect();

    // Shared-per-model Marlin prelude: one workspace allocation
    // (GpuTensor is `Copy` — each MarlinLinear captures the same
    // buffer by value), one device-id query. Only planted when at
    // least one accessor resolves to an AWQ FieldLoad; dense models
    // skip it so their `Weights::load` body is byte-for-byte
    // identical to before this commit.
    let awq_prelude: TokenStream = if any_awq {
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

    // `Self { a, b, c }` shorthand — fields are the just-bound
    // locals, in the same order we declared the struct fields.
    let field_shorthand: Vec<&syn::Ident> = accessors.iter().map(|a| &a.name).collect();

    quote! {
        /// Every weight the emitted forward needs, already packed
        /// exactly how the solver-picked Impls want to see it.
        ///
        /// Construct via [`Self::load`]. Hand the resulting struct
        /// to [`forward`] by reference.
        #[cfg(feature = "cuda")]
        pub struct Weights {
            #(#fields)*
        }

        #[cfg(feature = "cuda")]
        impl Weights {
            /// Read every field from an open `GpuWeights` (a
            /// safetensors view). Fused accessors stream their
            /// source weights directly into one packed GPU buffer
            /// without intermediate allocation.
            #[allow(clippy::too_many_lines, unused_variables)]
            pub fn load(
                gw: &mut ::ferrite_cuda_core::weights::GpuWeights,
                stream: ::ferrite_cuda_core::CUstream,
            ) -> ::anyhow::Result<Self> {
                #awq_prelude
                #(#lets)*
                Ok(Self {
                    #(#field_shorthand),*
                })
            }
        }
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

/// Emit one per-workload-bucket forward fn.
fn emit_forward_for_bucket(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    program: &Program,
    model: &ModelParams,
    lib: &ImplementationLibrary,
    num_tokens: u64,
) -> TokenStream {
    // Allocate a stable local-binding ident per tile-output slot.
    let mut locals: LocalMap = HashMap::new();
    for node in &fuf.nodes {
        for slot in 0..node.outputs.len().max(1) as u8 {
            locals.insert((node.id, slot), format_ident!("t_{}_{}", node.id.0, slot));
        }
    }

    // Forward returns the last tile's slot-0 output — its owner must
    // not be dropped before the function returns.
    let mut protected: HashSet<(TileId, u8)> = HashSet::new();
    if let Some(last) = fuf.nodes.last() {
        protected.insert((last.id, 0));
    }
    let drops = compute_drops_after(fuf, sfuf, loop_ir, lib, None, &protected);

    // Walk waves in order; each subgraph's bound impl emits its own
    // kernel invocation. Codegen stays mechanical — no per-op match
    // lives here. After each subgraph, drop any tile-local owners
    // whose last cross-subgraph use was here.
    let mut body: Vec<TokenStream> = Vec::new();
    for wave in &loop_ir.waves {
        for (sg, imp_id) in &wave.subgraphs {
            let claimed = sfuf.tiles_in_subgraph(*sg);
            let ctx = EmitCtx {
                fuf,
                program,
                model,
                claimed_tiles: &claimed,
                locals: &locals,
            };
            body.push(lib.get(*imp_id).emit_call(&ctx));
            if let Some(owners) = drops.get(sg) {
                for (t, s) in owners {
                    let ident = &locals[&(*t, *s)];
                    body.push(quote! { drop(#ident); });
                }
            }
        }
    }

    // The forward's return value: the last tile's output.
    let last_output = fuf
        .nodes
        .last()
        .map(|n| {
            let id = locals[&(n.id, 0)].clone();
            quote! { #id }
        })
        .unwrap_or_else(|| quote! { unreachable!("empty FUF") });

    let fn_name = format_ident!("forward_m_{}", num_tokens);
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
            #(#body)*
            #last_output
        }
    }
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
fn emit_forward_backbone_for_bucket(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    program: &Program,
    model: &ModelParams,
    lib: &ImplementationLibrary,
    num_tokens: u64,
) -> TokenStream {
    let mut locals: LocalMap = HashMap::new();
    for node in &fuf.nodes {
        for slot in 0..node.outputs.len().max(1) as u8 {
            locals.insert((node.id, slot), format_ident!("t_{}_{}", node.id.0, slot));
        }
    }

    // The terminal tile — the DSL's last op, expected to be the
    // lm_head gemm. Skip the subgraph that claims it.
    let Some(last_node) = fuf.nodes.last() else {
        // Empty FUF: degenerate, emit a stub that panics.
        let fn_name = format_ident!("forward_backbone_m_{}", num_tokens);
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

    // The backbone output is whatever tile feeds the terminal tile's
    // first input slot — for a DSL that ends in `gemm(normed, lm_head)`
    // that's the final `normed` (post-final-rmsnorm) tile.
    let backbone_out: (crate::fuf::TileId, u8) = match last_node.inputs.first() {
        Some(FufInput::Tile { id, slot }) => (*id, *slot),
        _ => panic!(
            "forward_backbone: terminal tile's first input is not a Tile \
             (DSL must end in `gemm(<tile>, lm_head)`)"
        ),
    };
    let backbone_ident = locals[&backbone_out].clone();

    // Backbone returns a clone of `backbone_out`; its underlying
    // owner (resolved via the alias chain in `compute_drops_after`)
    // must survive until after that clone runs at end of fn.
    let mut protected: HashSet<(TileId, u8)> = HashSet::new();
    protected.insert(backbone_out);
    let drops = compute_drops_after(fuf, sfuf, loop_ir, lib, Some(terminal_sg), &protected);

    let mut body: Vec<TokenStream> = Vec::new();
    for wave in &loop_ir.waves {
        for (sg, imp_id) in &wave.subgraphs {
            if *sg == terminal_sg {
                continue;
            }
            let claimed = sfuf.tiles_in_subgraph(*sg);
            let ctx = EmitCtx {
                fuf,
                program,
                model,
                claimed_tiles: &claimed,
                locals: &locals,
            };
            body.push(lib.get(*imp_id).emit_call(&ctx));
            if let Some(owners) = drops.get(sg) {
                for (t, s) in owners {
                    let ident = &locals[&(*t, *s)];
                    body.push(quote! { drop(#ident); });
                }
            }
        }
    }

    let fn_name = format_ident!("forward_backbone_m_{}", num_tokens);
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
            #(#body)*
            // Clone the backbone tile's output. It may be bound as
            // either an `OwnedTensor` (singleton impl output) or a
            // `TensorView` alias on an upstream buffer (fused-impl
            // output). `(*_).as_view()` works uniformly — both Deref
            // to `GpuTensor`, and `GpuTensor::as_view` yields a
            // fresh `TensorView`.
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

/// Emit the full per-model module body: Weights struct + loader,
/// one forward fn per workload bucket, and a dispatching wrapper.
pub fn emit_model(
    program: &Program,
    model: &ModelParams,
    fuf: &Fuf,
    sfufs: &WorkloadAssignments,
    loops: &WorkloadLoops,
    lib: &ImplementationLibrary,
) -> TokenStream {
    let weights = emit_weights_struct(program, fuf, sfufs, lib, model);

    let bucket_fns: Vec<TokenStream> = sfufs
        .per_num_tokens
        .iter()
        .map(|(m, sfuf)| {
            let loop_ir = loops
                .per_num_tokens
                .get(m)
                .expect("schedule populated every key");
            emit_forward_for_bucket(fuf, sfuf, loop_ir, program, model, lib, *m)
        })
        .collect();
    let backbone_fns: Vec<TokenStream> = sfufs
        .per_num_tokens
        .iter()
        .map(|(m, sfuf)| {
            let loop_ir = loops
                .per_num_tokens
                .get(m)
                .expect("schedule populated every key");
            emit_forward_backbone_for_bucket(fuf, sfuf, loop_ir, program, model, lib, *m)
        })
        .collect();

    // match num_tokens dispatch. Each compiled bucket `m_k` covers
    // the inclusive range `[m_k, m_{k+1} - 1]`; the last bucket
    // covers `m_last..=u64::MAX`. Runtime num_tokens that don't
    // exactly equal a compiled point get the specialization for
    // the largest compiled bucket ≤ num_tokens — correct
    // (kernels work at any M) if suboptimal for cost. The user
    // compiles more buckets if they want tighter cost fits.
    let bucket_points: Vec<u64> = sfufs.per_num_tokens.keys().copied().collect();
    let match_arms: Vec<TokenStream> = bucket_points
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            let fn_name = format_ident!("forward_m_{}", m);
            let lo = proc_macro2::Literal::u64_unsuffixed(m);
            if i + 1 == bucket_points.len() {
                // last bucket: cover m..=u64::MAX
                quote! { #lo.. => unsafe { #fn_name(wm, ctx, device) }, }
            } else {
                let next = bucket_points[i + 1];
                let hi = proc_macro2::Literal::u64_unsuffixed(next - 1);
                quote! { #lo..=#hi => unsafe { #fn_name(wm, ctx, device) }, }
            }
        })
        .collect();
    let backbone_match_arms: Vec<TokenStream> = bucket_points
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            let fn_name = format_ident!("forward_backbone_m_{}", m);
            let lo = proc_macro2::Literal::u64_unsuffixed(m);
            if i + 1 == bucket_points.len() {
                quote! { #lo.. => unsafe { #fn_name(wm, ctx, device) }, }
            } else {
                let next = bucket_points[i + 1];
                let hi = proc_macro2::Literal::u64_unsuffixed(next - 1);
                quote! { #lo..=#hi => unsafe { #fn_name(wm, ctx, device) }, }
            }
        })
        .collect();
    // Below the smallest compiled bucket (e.g. num_tokens=0 if
    // someone somehow passes it): fall through to the smallest
    // bucket. Realistically unreachable.
    let fallback_arm = bucket_points.first().map(|&m| {
        let fn_name = format_ident!("forward_m_{}", m);
        quote! { _ => unsafe { #fn_name(wm, ctx, device) }, }
    });
    let backbone_fallback_arm = bucket_points.first().map(|&m| {
        let fn_name = format_ident!("forward_backbone_m_{}", m);
        quote! { _ => unsafe { #fn_name(wm, ctx, device) }, }
    });

    quote! {
        #weights

        #(#bucket_fns)*
        #(#backbone_fns)*

        /// Dispatch on `num_tokens`. Each compiled bucket covers
        /// an inclusive range starting at its compiled point; the
        /// largest bucket covers everything above.
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

        /// Backbone-only dispatch (no lm_head). See
        /// [`forward_backbone_m_*`] for what each bucket skips and
        /// the shape of the returned tensor (per the DSL, a
        /// freshly-allocated `[num_tokens, hidden_size]` `OwnedTensor`).
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
