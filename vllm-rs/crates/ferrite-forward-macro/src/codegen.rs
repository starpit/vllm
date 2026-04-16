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
use crate::impl_lib::{DevicePhase, ImplId, ImplementationLibrary, WeightAccessor};
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
}

/// Distill a `WeightAccessor` into its field-load plan. Uses the
/// accessor's declared `rust_type` + `source_weights` and the
/// model's config (for `rms_norm_eps` / `tie_word_embeddings`).
fn plan_field_load(accessor: &WeightAccessor, program: &Program, model: &ModelParams) -> FieldLoad {
    let ty = accessor.rust_type.to_string().replace(' ', "");
    let is_embedding =
        ty.ends_with("::Embedding") || ty == "Embedding" || ty.ends_with("layers::Embedding");
    let is_rmsnorm =
        ty.ends_with("::RmsNorm") || ty == "RmsNorm" || ty.ends_with("layers::RmsNorm");
    let is_linear =
        ty.ends_with("::LinearLayer") || ty == "LinearLayer" || ty.ends_with("layers::LinearLayer");

    let prefixes: Vec<String> = accessor
        .source_weights
        .iter()
        .map(|(id, idx)| safetensors_prefix(program, *id, *idx))
        .collect();

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
    // HF convention: if set, lm_head reuses embed_tokens.weight; no
    // separate `lm_head.weight` tensor in safetensors.
    let Ok(s) = std::fs::read_to_string(&model.source_path) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) else {
        return false;
    };
    v.get("tie_word_embeddings")
        .and_then(|x| x.as_bool())
        .unwrap_or(false)
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
    let lets: Vec<TokenStream> = accessors
        .iter()
        .map(|a| {
            let name = &a.name;
            let plan = plan_field_load(a, program, model);
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
            }
        })
        .collect();

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

/// Context for megakernel emission — bundles the many parameters
/// needed by `try_emit_megakernel`.
struct MegakernelEmitCtx<'a> {
    sfuf: &'a Assignment,
    fuf: &'a Fuf,
    program: &'a Program,
    model: &'a ModelParams,
    lib: &'a ImplementationLibrary,
    locals: &'a LocalMap,
    drops: &'a DropPlan,
    num_tokens: u64,
}

/// Try to emit a megakernel launch for a group of ≥2 DC subgraphs.
///
/// Calls `device_phase` on each subgraph's impl. If all return `Some`,
/// generates the `.cu` source (written to the megakernel cache), emits
/// an `extern "C"` FFI declaration and the Rust call site.
///
/// Returns `Some(tokens)` on success, `None` if any impl can't provide
/// a device phase (caller should fall back to individual `emit_call`).
fn try_emit_megakernel(
    dc_group: &[(&SubgraphId, &ImplId)],
    mctx: &MegakernelEmitCtx,
    mega_idx: &mut usize,
) -> Option<Vec<TokenStream>> {
    if dc_group.len() < 2 {
        return None;
    }

    // Collect DevicePhases + Rust param expressions for each subgraph.
    let mut phases: Vec<DevicePhase> = Vec::new();
    let mut rust_preambles: Vec<Vec<proc_macro2::TokenStream>> = Vec::new();
    let mut rust_values: Vec<Vec<proc_macro2::TokenStream>> = Vec::new();

    for (idx, (sg, imp_id)) in dc_group.iter().enumerate() {
        let claimed = mctx.sfuf.tiles_in_subgraph(**sg);
        let ctx = EmitCtx {
            fuf: mctx.fuf,
            program: mctx.program,
            model: mctx.model,
            claimed_tiles: &claimed,
            locals: mctx.locals,
            num_tokens: Some(mctx.num_tokens),
        };
        let imp = mctx.lib.get(**imp_id);
        let (phase, preamble, values) = imp.device_phase(idx, &ctx)?;
        phases.push(phase);
        rust_preambles.push(preamble);
        rust_values.push(values);
    }

    // Generate the .cu source and write to megakernel cache.
    let wave_label = format!("m{}_w{}", mctx.num_tokens, *mega_idx);
    *mega_idx += 1;
    let generated = generate_megakernel_labeled(&wave_label, &phases);

    // Write .cu to cache directory (best-effort; build.rs picks it up).
    // Mirrors the path used by ferrite-cuda-builder/build.rs:
    // ~/.cache/cudaforge/megakernels/ (Linux/Mac).
    if let Ok(home) = std::env::var("HOME") {
        let mega_dir = std::path::PathBuf::from(home).join(".cache/cudaforge/megakernels");
        let _ = std::fs::create_dir_all(&mega_dir);
        let cu_path = mega_dir.join(format!("{}.cu", generated.launch_fn_name));
        let _ = std::fs::write(&cu_path, &generated.cuda_source);
    }

    // Emit Rust code: extern "C" declaration + param setup + call.
    let launch_fn = format_ident!("{}", generated.launch_fn_name);
    let mut tokens = Vec::new();

    // Build the extern "C" parameter list.
    let extern_params: Vec<TokenStream> = generated
        .flat_params
        .iter()
        .map(|(c_type, name)| {
            let name_ident = format_ident!("{}", name);
            let ty = c_type_to_rust(c_type);
            quote! { #name_ident: #ty }
        })
        .collect();

    tokens.push(quote! {
        unsafe extern "C" {
            fn #launch_fn(
                #(#extern_params,)*
                __grid_x: i32,
                __block_x: i32,
                __smem_bytes: usize,
                __stream: u64,
            ) -> i32;
        }
    });

    // Preamble statements go in the OUTER scope so output ident
    // bindings (e.g. `let t_3_0 = device.caching.alloc_tensor(...)`)
    // survive past the megakernel launch and are visible to
    // downstream code that consumes those tiles.
    let mut preamble_stmts: Vec<TokenStream> = Vec::new();
    for preamble in &rust_preambles {
        for stmt in preamble {
            preamble_stmts.push(stmt.clone());
        }
    }
    tokens.push(quote! { #(#preamble_stmts)* });

    // Param value assignments + launch call in a contained block
    // (the param idents like p0_out are only needed for the FFI call).
    let mut param_stmts: Vec<TokenStream> = Vec::new();
    for (phase, values) in phases.iter().zip(rust_values.iter()) {
        for ((c_type, name), val) in phase.flat_params.iter().zip(values.iter()) {
            let name_ident = format_ident!("{}", name);
            let ty = c_type_to_rust(c_type);
            param_stmts.push(quote! {
                let #name_ident: #ty = #val;
            });
        }
    }

    let param_idents: Vec<proc_macro2::Ident> = generated
        .flat_params
        .iter()
        .map(|(_, name)| format_ident!("{}", name))
        .collect();

    tokens.push(quote! {
        {
            #(#param_stmts)*
            let __stream_raw = device.compute_stream as u64;
            let __grid_x = device.num_sm as i32;
            let __block_x = 256i32;
            let __smem_bytes = 2048usize;
            let __ret = unsafe {
                #launch_fn(
                    #(#param_idents,)*
                    __grid_x,
                    __block_x,
                    __smem_bytes,
                    __stream_raw,
                )
            };
            assert_eq!(__ret, 0, "megakernel launch failed");
        }
    });

    // Emit drops for each subgraph in the group.
    for (sg, _) in dc_group {
        if let Some(owners) = mctx.drops.get(sg) {
            for (t, s) in owners {
                let ident = &mctx.locals[&(*t, *s)];
                tokens.push(quote! { drop(#ident); });
            }
        }
    }

    Some(tokens)
}

/// Map a C type string to a Rust FFI type.
fn c_type_to_rust(c_type: &str) -> TokenStream {
    match c_type {
        "void*" => quote! { *mut ::core::ffi::c_void },
        "const void*" => quote! { *const ::core::ffi::c_void },
        "int" => quote! { i32 },
        "float" => quote! { f32 },
        "double" => quote! { f64 },
        "size_t" | "uint64_t" => quote! { u64 },
        "int64_t" => quote! { i64 },
        _ => {
            let ty_ident = format_ident!("{}", c_type);
            quote! { #ty_ident }
        }
    }
}

/// Like `generate_megakernel` but with a string label instead of
/// numeric wave_idx, for disambiguation across workload buckets.
fn generate_megakernel_labeled(
    label: &str,
    phases: &[DevicePhase],
) -> crate::cuda_codegen::GeneratedMegakernel {
    assert!(
        phases.len() >= 2,
        "megakernel requires at least 2 phases (got {})",
        phases.len()
    );

    use std::fmt::Write;

    let launch_fn_name = format!("megakernel_{label}_launch");
    let params_struct_name = format!("Megakernel_{label}_Params");
    let kernel_name = format!("megakernel_{label}");

    let mut src = String::new();
    writeln!(src, "// Auto-generated megakernel for {label}").unwrap();
    writeln!(
        src,
        "// DO NOT EDIT — regenerate via the forward! proc macro."
    )
    .unwrap();
    writeln!(src).unwrap();
    writeln!(src, "#include <cuda_bf16.h>").unwrap();
    writeln!(src, "#include <cooperative_groups.h>").unwrap();
    writeln!(src, "#include \"megakernel_ops.cuh\"").unwrap();
    writeln!(src).unwrap();

    // Emit per-phase preambles (CUTLASS includes, type aliases, etc.)
    for phase in phases {
        for line in &phase.preamble {
            writeln!(src, "{line}").unwrap();
        }
    }
    if phases.iter().any(|p| !p.preamble.is_empty()) {
        writeln!(src).unwrap();
    }

    let mut all_flat: Vec<(String, String)> = Vec::new();
    for phase in phases {
        all_flat.extend(phase.flat_params.iter().cloned());
    }

    writeln!(src, "struct {params_struct_name} {{").unwrap();
    for (i, phase) in phases.iter().enumerate() {
        writeln!(src, "    // Phase {i}").unwrap();
        for field in &phase.internal_fields {
            writeln!(src, "    {field};").unwrap();
        }
    }
    writeln!(src, "}};").unwrap();
    writeln!(src).unwrap();

    writeln!(
        src,
        "extern \"C\" __global__ void {kernel_name}({params_struct_name} p) {{"
    )
    .unwrap();
    writeln!(src, "    namespace cg = cooperative_groups;").unwrap();
    writeln!(src, "    extern __shared__ char smem[];").unwrap();
    writeln!(src).unwrap();

    // Destructure params struct into local variables so kernel_body
    // lines can reference bare names (p0_out, p1_input, etc.).
    for phase in phases {
        for field in &phase.internal_fields {
            let name = field.split_whitespace().last().unwrap_or("");
            let name = name.trim_start_matches('*');
            writeln!(src, "    auto {name} = p.{name};").unwrap();
        }
    }
    writeln!(src).unwrap();

    for (i, phase) in phases.iter().enumerate() {
        if i > 0 {
            writeln!(src, "    cg::this_grid().sync();").unwrap();
            writeln!(src).unwrap();
        }
        writeln!(src, "    // Phase {i}").unwrap();
        for line in &phase.kernel_body {
            writeln!(src, "    {line}").unwrap();
        }
        writeln!(src).unwrap();
    }

    writeln!(src, "}}").unwrap();
    writeln!(src).unwrap();

    writeln!(src, "extern \"C\" int {launch_fn_name}(").unwrap();
    for (c_type, name) in &all_flat {
        writeln!(src, "    {c_type} {name},").unwrap();
    }
    writeln!(src, "    int __grid_x, int __block_x,").unwrap();
    writeln!(src, "    size_t __smem_bytes,").unwrap();
    writeln!(src, "    uint64_t __stream)").unwrap();
    writeln!(src, "{{").unwrap();
    writeln!(src, "    {params_struct_name} params;").unwrap();
    for phase in phases {
        for line in &phase.params_build {
            writeln!(src, "    {line}").unwrap();
        }
    }
    writeln!(src).unwrap();
    writeln!(src, "    dim3 grid(__grid_x);").unwrap();
    writeln!(src, "    dim3 block(__block_x);").unwrap();
    writeln!(src, "    void* args[] = {{ &params }};").unwrap();
    writeln!(src, "    return cudaLaunchCooperativeKernel(").unwrap();
    writeln!(src, "        (void*){kernel_name},").unwrap();
    writeln!(
        src,
        "        grid, block, args, __smem_bytes, (cudaStream_t)__stream);"
    )
    .unwrap();
    writeln!(src, "}}").unwrap();

    crate::cuda_codegen::GeneratedMegakernel {
        cuda_source: src,
        launch_fn_name,
        flat_params: all_flat,
    }
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

    // Walk waves in order. Megakernel waves (is_megakernel=true)
    // attempt to emit a single cooperative kernel launch; non-mega
    // waves emit individual `emit_call` per subgraph.
    let mctx = MegakernelEmitCtx {
        sfuf,
        fuf,
        program,
        model,
        lib,
        locals: &locals,
        drops: &drops,
        num_tokens,
    };
    let mut body: Vec<TokenStream> = Vec::new();
    let mut mega_idx = 0usize;
    for wave in &loop_ir.waves {
        if wave.is_megakernel {
            let dc_group: Vec<(&SubgraphId, &ImplId)> =
                wave.subgraphs.iter().map(|(sg, id)| (sg, id)).collect();
            if let Some(mega_tokens) = try_emit_megakernel(&dc_group, &mctx, &mut mega_idx) {
                body.extend(mega_tokens);
                continue;
            }
        }
        for (sg, imp_id) in &wave.subgraphs {
            let claimed = sfuf.tiles_in_subgraph(*sg);
            let ctx = EmitCtx {
                fuf,
                program,
                model,
                claimed_tiles: &claimed,
                locals: &locals,
                num_tokens: None,
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

    let mctx = MegakernelEmitCtx {
        sfuf,
        fuf,
        program,
        model,
        lib,
        locals: &locals,
        drops: &drops,
        num_tokens,
    };
    let mut body: Vec<TokenStream> = Vec::new();
    let mut mega_idx = 0usize;
    for wave in &loop_ir.waves {
        if wave.is_megakernel {
            let dc_group: Vec<(&SubgraphId, &ImplId)> = wave
                .subgraphs
                .iter()
                .filter(|(sg, _)| *sg != terminal_sg)
                .map(|(sg, id)| (sg, id))
                .collect();
            if dc_group.len() >= 2
                && let Some(mega_tokens) = try_emit_megakernel(&dc_group, &mctx, &mut mega_idx)
            {
                body.extend(mega_tokens);
                continue;
            }
        }
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
                num_tokens: None,
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
