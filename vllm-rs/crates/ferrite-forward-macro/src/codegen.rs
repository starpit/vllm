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
use crate::target::TargetProfile;

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
    target: &'a TargetProfile,
    lib: &'a ImplementationLibrary,
    locals: &'a LocalMap,
    drops: &'a DropPlan,
    num_tokens: u64,
}

/// Try to emit a TK megakernel for the entire forward pass (SM90+).
///
/// Generates a single `.cu` file using the KVM runtime with vendored
/// ThunderKittens ops. Returns the complete forward function body
/// as a TokenStream, or `None` if TK emission isn't applicable.
#[allow(clippy::too_many_arguments)]
fn try_emit_tk_forward(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    program: &Program,
    model: &ModelParams,
    target: &TargetProfile,
    lib: &ImplementationLibrary,
    _locals: &LocalMap,
    num_tokens: u64,
) -> Option<TokenStream> {
    use crate::cuda_codegen::{TkModelDims, generate_tk_megakernel};
    // KVM instruction generation is now inline — no longer uses instruction.rs primitives.

    // Extract model dimensions from ModelParams bounds.
    let get = |key: &str| -> Option<u32> {
        model.bounds.get(key).map(|v| *v as u32)
    };
    let num_layers = get("num_hidden_layers")?;
    let hidden_dim = get("hidden_size")?;
    let intermediate_dim = get("intermediate_size")?;
    let num_attention_heads = get("num_attention_heads")?;
    let num_kv_heads = get("num_key_value_heads").unwrap_or(num_attention_heads);
    let head_dim = get("head_dim").unwrap_or(hidden_dim / num_attention_heads);
    let vocab_size = get("vocab_size")?;

    // The vendored attention_partial.cu only supports head_dim=64 and GQA_RATIO=4.
    // Skip TK for incompatible models — they'll use the BSP megakernel instead.
    let gqa_ratio = num_attention_heads / num_kv_heads;
    if head_dim != 64 || gqa_ratio != 4 {
        return None;
    }

    let dims = TkModelDims {
        num_layers,
        hidden_dim,
        intermediate_dim,
        head_dim,
        num_attention_heads,
        num_kv_heads,
        kv_block_size: 16,
        matvec_block_size: 16,
        vocab_size,
        sm_count: target.num_sms,
    };

    let model_name = model.source_stem.replace(['-', '.'], "_");
    let generated = generate_tk_megakernel(&model_name, &dims);

    // Write .cu to cache directory.
    if let Ok(home) = std::env::var("HOME") {
        let mega_dir = std::path::PathBuf::from(home).join(".cache/cudaforge/megakernels");
        let _ = std::fs::create_dir_all(&mega_dir);
        let cu_path = mega_dir.join(format!("{}.cu", generated.launch_fn_name));
        let _ = std::fs::write(&cu_path, &generated.cuda_source);
    }

    // Build KVM-format instruction schedule.
    //
    // The KVM controller expects composite fused opcodes (1-7) matching the
    // ThunderKittens ops. Each instruction is 32 ints (INSTRUCTION_WIDTH=32).
    // Instructions are organized as [num_sms][max_instructions_per_sm][32].
    // The controller loops `rows()` times; opcode 0 = NoOp (skip).
    //
    // We generate instructions matching the Python scheduler in
    // Megakernels/demos/latency/scheduler.py.
    const KVM_INST_WIDTH: usize = 32;
    let sm_count = target.num_sms as usize;
    let block_size = 16usize; // matvec_block_size for all ops

    // Helper: serialize an instruction into a 32-word array (opcode + fields, zero-padded).
    fn serialize_kvm_inst(words: &[i32]) -> [i32; 32] {
        let mut out = [0i32; 32];
        for (i, &w) in words.iter().enumerate() {
            out[i] = w;
        }
        out
    }

    // Per-layer instruction generation, then round-robin assignment to SMs.
    let mut all_instructions: Vec<[i32; 32]> = Vec::new();

    for layer in 0..num_layers as i32 {
        // ── Op 1: RMS_QKV_MatVecRopeAppend ──
        // Distribute qkv_outdim / block_size blocks proportionally across SMs.
        let qkv_outdim = (num_attention_heads + 2 * num_kv_heads) * head_dim;
        let num_qkv_blocks = qkv_outdim as usize / block_size;
        let blocks_per_sm = num_qkv_blocks as f64 / sm_count as f64;
        for sm_idx in 0..sm_count {
            let start = (sm_idx as f64 * blocks_per_sm).round() as i32;
            let end = ((sm_idx + 1) as f64 * blocks_per_sm).round() as i32;
            all_instructions.push(serialize_kvm_inst(&[1, layer, start, end]));
        }

        // ── Op 2: PartialAttention ──
        // For skip_attn_reduction mode: 1 partial per kv_head.
        let num_partials = 1i32;
        for kv_head_idx in 0..num_kv_heads as i32 {
            for partial_idx in 0..num_partials {
                all_instructions.push(serialize_kvm_inst(&[
                    2, layer, kv_head_idx, num_partials, partial_idx,
                ]));
            }
        }

        // ── Op 4: O_ProjResidual ──
        // One instruction per output block (hidden_dim / block_size blocks).
        let num_o_blocks = hidden_dim as usize / block_size;
        for o_block_idx in 0..num_o_blocks as i32 {
            all_instructions.push(serialize_kvm_inst(&[
                4, layer, o_block_idx, o_block_idx + 1, 0,
            ]));
        }

        // ── Op 5: LayerNormDoubleMatVecSiLU (upgate) ──
        // Distribute intermediate_dim / block_size blocks round-robin across SMs.
        let num_up_blocks = intermediate_dim as usize / block_size;
        for sm_idx in 0..sm_count {
            let mut block_idxs: Vec<i32> = Vec::new();
            let mut idx = sm_idx;
            while idx < num_up_blocks {
                block_idxs.push(idx as i32);
                idx += sm_count;
            }
            if !block_idxs.is_empty() {
                // Serialization: opcode, layer_idx, len(block_idxs), block_idxs...
                let mut words = vec![5i32, layer, block_idxs.len() as i32];
                words.extend_from_slice(&block_idxs);
                all_instructions.push(serialize_kvm_inst(&words));
            }
        }

        // ── Op 6: DownProjResidual ──
        // num_col_splits = intermediate_dim / hidden_dim; distribute jobs across SMs.
        let num_down_blocks = hidden_dim as usize / block_size;
        let num_col_splits = intermediate_dim as usize / hidden_dim as usize;
        let mut jobs: Vec<(usize, usize)> = Vec::new();
        for col_idx in 0..num_col_splits {
            for down_block_idx in 0..num_down_blocks {
                jobs.push((col_idx, down_block_idx));
            }
        }
        let mut num_assigned = 0usize;
        for sm_idx in 0..sm_count {
            let jobs_left = jobs.len() - num_assigned;
            let sms_left = sm_count - sm_idx;
            let jobs_per_sm = jobs_left as f64 / sms_left as f64;
            let jobs_for_this_sm = jobs_per_sm.round() as usize;
            if jobs_for_this_sm == 0 { continue; }
            let raw_sliced = &jobs[num_assigned..num_assigned + jobs_for_this_sm];
            // Only take jobs with same col_idx as first job in slice.
            let col_idx = raw_sliced[0].0;
            let sliced: Vec<_> = raw_sliced.iter()
                .take_while(|j| j.0 == col_idx)
                .collect();
            let start_block = sliced[0].1 as i32;
            let end_block = start_block + sliced.len() as i32;
            all_instructions.push(serialize_kvm_inst(&[
                6, layer, start_block, end_block, col_idx as i32,
            ]));
            num_assigned += sliced.len();
        }
    }

    // ── Op 7: RMS_LM_Head ──
    let num_logit_blocks = vocab_size as usize / block_size;
    let blocks_per_sm_lm = num_logit_blocks as f64 / sm_count as f64;
    for sm_idx in 0..sm_count {
        let start = (sm_idx as f64 * blocks_per_sm_lm).round() as i32;
        let end = ((sm_idx + 1) as f64 * blocks_per_sm_lm).round() as i32;
        all_instructions.push(serialize_kvm_inst(&[7, start, end]));
    }

    // Round-robin assign to SMs.
    let mut per_sm: Vec<Vec<[i32; 32]>> = vec![Vec::new(); sm_count];
    for (i, inst) in all_instructions.iter().enumerate() {
        per_sm[i % sm_count].push(*inst);
    }

    // Pad all SM queues to the same length with NoOp (opcode 0).
    let max_per_sm = per_sm.iter().map(|v| v.len()).max().unwrap_or(0);
    for queue in &mut per_sm {
        while queue.len() < max_per_sm {
            queue.push([0i32; 32]);
        }
    }

    // Serialize instruction tensor: [num_sms][max_per_sm][32]
    let total_words = sm_count * max_per_sm * KVM_INST_WIDTH;
    let inst_data: Vec<u32> = {
        let mut data = vec![0u32; total_words];
        for (sm, queue) in per_sm.iter().enumerate() {
            for (i, inst) in queue.iter().enumerate() {
                let offset = (sm * max_per_sm + i) * KVM_INST_WIDTH;
                for (j, &w) in inst.iter().enumerate() {
                    data[offset + j] = w as u32;
                }
            }
        }
        data
    };

    let num_barriers = num_layers as usize * 10 * (num_attention_heads + 2 * num_kv_heads) as usize;
    let inst_data_tokens: Vec<proc_macro2::TokenStream> = inst_data
        .iter()
        .map(|w| quote! { #w })
        .collect();
    let total_words_lit = total_words;
    let num_barriers_lit = num_barriers;
    let max_per_sm_lit = max_per_sm;

    // ── Classify weight accessors by TK role ──
    //
    // Walk ALL subgraphs (not just megakernel) to discover every weight
    // accessor. Classify each by examining the source weight paths in
    // the Program's weight table.
    let all_wave_sgs: Vec<(crate::solver::SubgraphId, crate::impl_lib::ImplId)> = loop_ir
        .waves
        .iter()
        .flat_map(|w| w.subgraphs.iter().cloned())
        .collect();

    let mut qkv_fields: Vec<(usize, syn::Ident)> = Vec::new();
    let mut o_fields: Vec<(usize, syn::Ident)> = Vec::new();
    let mut gate_up_fields: Vec<(usize, syn::Ident)> = Vec::new();
    let mut down_fields: Vec<(usize, syn::Ident)> = Vec::new();
    let mut attn_norm_fields: Vec<(usize, syn::Ident)> = Vec::new();
    let mut mlp_norm_fields: Vec<(usize, syn::Ident)> = Vec::new();
    let mut embed_field: Option<syn::Ident> = None;
    let mut lm_head_field: Option<syn::Ident> = None;
    let mut lm_head_norm_field: Option<syn::Ident> = None;
    let mut seen_names: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (sg_id, imp_id) in &all_wave_sgs {
        let tiles = sfuf.tiles_in_subgraph(*sg_id);
        let imp = lib.get(*imp_id);
        for wa in imp.required_weights(&tiles, fuf, program) {
            let name_str = wa.name.to_string();
            if !seen_names.insert(name_str) {
                continue;
            }

            // Classify by source weight paths.
            let mut role = "";
            let mut layer_idx: Option<usize> = None;

            for (wid, _) in &wa.source_weights {
                let path = program.weights.path(*wid);
                let parts: Vec<String> = path.iter().map(|s| s.to_string()).collect();
                let joined = parts.join(".");

                // Extract layer index from "layers.N" pattern.
                for (i, seg) in parts.iter().enumerate() {
                    if i > 0 && parts[i - 1] == "layers"
                        && let Ok(idx) = seg.parse::<usize>()
                    {
                        layer_idx = Some(idx);
                    }
                }

                if joined.contains("q_proj")
                    || joined.contains("k_proj")
                    || joined.contains("v_proj")
                {
                    role = "qkv";
                } else if joined.contains("o_proj") {
                    role = "o";
                } else if joined.contains("gate_proj") || joined.contains("up_proj") {
                    role = "gate_up";
                } else if joined.contains("down_proj") {
                    role = "down";
                } else if joined.contains("input_layernorm") {
                    role = "attn_norm";
                } else if joined.contains("post_attention_layernorm") {
                    role = "mlp_norm";
                } else if !joined.contains("layers") {
                    if joined.contains("embed") {
                        role = "embed";
                    } else if joined.contains("lm_head") {
                        role = "lm_head";
                    } else if joined.ends_with("norm") {
                        role = "lm_head_norm";
                    }
                }
            }

            match role {
                "qkv" => qkv_fields.push((layer_idx.unwrap_or(0), wa.name.clone())),
                "o" => o_fields.push((layer_idx.unwrap_or(0), wa.name.clone())),
                "gate_up" => gate_up_fields.push((layer_idx.unwrap_or(0), wa.name.clone())),
                "down" => down_fields.push((layer_idx.unwrap_or(0), wa.name.clone())),
                "attn_norm" => attn_norm_fields.push((layer_idx.unwrap_or(0), wa.name.clone())),
                "mlp_norm" => mlp_norm_fields.push((layer_idx.unwrap_or(0), wa.name.clone())),
                "embed" => embed_field = Some(wa.name.clone()),
                "lm_head" => lm_head_field = Some(wa.name.clone()),
                "lm_head_norm" => lm_head_norm_field = Some(wa.name.clone()),
                _ => {}
            }
        }
    }

    // Sort per-layer fields by layer index.
    qkv_fields.sort_by_key(|(l, _)| *l);
    o_fields.sort_by_key(|(l, _)| *l);
    gate_up_fields.sort_by_key(|(l, _)| *l);
    down_fields.sort_by_key(|(l, _)| *l);
    attn_norm_fields.sort_by_key(|(l, _)| *l);
    mlp_norm_fields.sort_by_key(|(l, _)| *l);

    // If we can't find all required weight roles, fall back to BSP.
    let embed_ident = embed_field?;
    let lm_head_ident = lm_head_field?;
    let lm_head_norm_ident = lm_head_norm_field?;

    if qkv_fields.len() != num_layers as usize
        || o_fields.len() != num_layers as usize
        || down_fields.len() != num_layers as usize
        || attn_norm_fields.len() != num_layers as usize
        || mlp_norm_fields.len() != num_layers as usize
    {
        return None; // incomplete weight set for TK
    }

    // ── Build per-layer D2D copy statements (compile-time unrolled) ──

    let num_layers_usize = num_layers as usize;
    let hidden_usize = hidden_dim as usize;
    let intermediate_usize = intermediate_dim as usize;
    let head_dim_usize = head_dim as usize;
    let num_heads_usize = num_attention_heads as usize;
    let num_kv_heads_usize = num_kv_heads as usize;
    let vocab_usize = vocab_size as usize;
    let num_sms_usize = target.num_sms as usize;
    let qkv_out_dim = (num_attention_heads + 2 * num_kv_heads) * head_dim;
    let qkv_out_usize = qkv_out_dim as usize;

    // QKV weight stacking: each layer's fused QKV → contiguous buffer.
    let qkv_copy_stmts: Vec<TokenStream> = qkv_fields.iter().enumerate().map(|(i, (_, field))| {
        quote! {
            ::ferrite_cuda_core::driver::memcpy_dtod_async(
                __tk_qkv_buf.add(#i * __qkv_per_layer),
                wm.#field.dense_weight().raw_ptr(),
                __qkv_per_layer,
                __stream,
            ).expect("qkv stack");
        }
    }).collect();

    // O-proj weight stacking.
    let o_copy_stmts: Vec<TokenStream> = o_fields.iter().enumerate().map(|(i, (_, field))| {
        quote! {
            ::ferrite_cuda_core::driver::memcpy_dtod_async(
                __tk_o_buf.add(#i * __o_per_layer),
                wm.#field.dense_weight().raw_ptr(),
                __o_per_layer,
                __stream,
            ).expect("o stack");
        }
    }).collect();

    // Gate + Up: fused gate_up accessor has [gate|up] concatenated.
    // Split into separate gate and up buffers.
    let gate_up_copy_stmts: Vec<TokenStream> = if gate_up_fields.len() == num_layers as usize {
        gate_up_fields.iter().enumerate().map(|(i, (_, field))| {
            quote! {
                {
                    let __fused_ptr = wm.#field.dense_weight().raw_ptr();
                    // Gate = first intermediate_dim rows.
                    ::ferrite_cuda_core::driver::memcpy_dtod_async(
                        __tk_gate_buf.add(#i * __gate_per_layer),
                        __fused_ptr,
                        __gate_per_layer,
                        __stream,
                    ).expect("gate stack");
                    // Up = second intermediate_dim rows.
                    ::ferrite_cuda_core::driver::memcpy_dtod_async(
                        __tk_up_buf.add(#i * __gate_per_layer),
                        __fused_ptr.add(__gate_per_layer),
                        __gate_per_layer,
                        __stream,
                    ).expect("up stack");
                }
            }
        }).collect()
    } else {
        return None; // need fused gate+up for every layer
    };

    // Down-proj weight stacking.
    let down_copy_stmts: Vec<TokenStream> = down_fields.iter().enumerate().map(|(i, (_, field))| {
        quote! {
            ::ferrite_cuda_core::driver::memcpy_dtod_async(
                __tk_down_buf.add(#i * __down_per_layer),
                wm.#field.dense_weight().raw_ptr(),
                __down_per_layer,
                __stream,
            ).expect("down stack");
        }
    }).collect();

    // Attn norm weight stacking.
    let attn_norm_copy_stmts: Vec<TokenStream> = attn_norm_fields.iter().enumerate().map(|(i, (_, field))| {
        quote! {
            ::ferrite_cuda_core::driver::memcpy_dtod_async(
                __tk_attn_norm_buf.add(#i * __norm_per_layer),
                wm.#field.weight.raw_ptr(),
                __norm_per_layer,
                __stream,
            ).expect("attn_norm stack");
        }
    }).collect();

    // MLP norm weight stacking.
    let mlp_norm_copy_stmts: Vec<TokenStream> = mlp_norm_fields.iter().enumerate().map(|(i, (_, field))| {
        quote! {
            ::ferrite_cuda_core::driver::memcpy_dtod_async(
                __tk_mlp_norm_buf.add(#i * __norm_per_layer),
                wm.#field.weight.raw_ptr(),
                __norm_per_layer,
                __stream,
            ).expect("mlp_norm stack");
        }
    }).collect();

    // First attn norm accessor — used to read rms_norm_eps.
    let first_attn_norm = &attn_norm_fields[0].1;

    // ── Emit Rust code ──

    let launch_fn = format_ident!("{}", generated.launch_fn_name);
    let fn_name = format_ident!("forward_m_{}", num_tokens);

    let extern_params: Vec<TokenStream> = generated
        .flat_params
        .iter()
        .map(|(c_type, name)| {
            let name_ident = format_ident!("{}", name);
            let ty = c_type_to_rust(c_type);
            quote! { #name_ident: #ty }
        })
        .collect();

    let tokens = quote! {
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn #fn_name(
            wm: &Weights,
            ctx: &::ferrite_forward::ForwardCtx,
            device: &mut ::ferrite_cuda_core::device::GpuDevice,
        ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
            use ::std::sync::atomic::Ordering;

            // One-time weight stacking statics (persistent GPU allocs).
            static __TK_INIT: ::std::sync::Once = ::std::sync::Once::new();
            static __TK_QKV: ::std::sync::atomic::AtomicU64 = ::std::sync::atomic::AtomicU64::new(0);
            static __TK_O: ::std::sync::atomic::AtomicU64 = ::std::sync::atomic::AtomicU64::new(0);
            static __TK_GATE: ::std::sync::atomic::AtomicU64 = ::std::sync::atomic::AtomicU64::new(0);
            static __TK_UP: ::std::sync::atomic::AtomicU64 = ::std::sync::atomic::AtomicU64::new(0);
            static __TK_DOWN: ::std::sync::atomic::AtomicU64 = ::std::sync::atomic::AtomicU64::new(0);
            static __TK_ATTN_NORM: ::std::sync::atomic::AtomicU64 = ::std::sync::atomic::AtomicU64::new(0);
            static __TK_MLP_NORM: ::std::sync::atomic::AtomicU64 = ::std::sync::atomic::AtomicU64::new(0);

            unsafe extern "C" {
                fn #launch_fn(#(#extern_params,)* __stream: u64) -> i32;
            }

            let __stream = device.compute_stream;
            let __num_layers: usize = #num_layers_usize;
            let __hidden: usize = #hidden_usize;
            let __intermediate: usize = #intermediate_usize;
            let __head_dim: usize = #head_dim_usize;
            let __num_heads: usize = #num_heads_usize;
            let __num_kv_heads: usize = #num_kv_heads_usize;
            let __qkv_out: usize = #qkv_out_usize;
            let __vocab: usize = #vocab_usize;
            let __num_sms: usize = #num_sms_usize;

            // ── One-time weight stacking (persistent GPU allocs) ──
            __TK_INIT.call_once(|| {
                ::tracing::info!("TK megakernel: initializing weight stacking for {} layers, hidden={}, sms={}",
                    __num_layers, __hidden, __num_sms);
                // QKV: [num_layers, qkv_out, hidden] bf16
                let __qkv_per_layer = __qkv_out * __hidden * 2;
                let __qkv_total = __num_layers * __qkv_per_layer;
                let __tk_qkv_buf = ::ferrite_cuda_core::driver::mem_alloc(__qkv_total)
                    .expect("tk qkv alloc");
                #(#qkv_copy_stmts)*
                __TK_QKV.store(__tk_qkv_buf as u64, Ordering::Relaxed);

                // O: [num_layers, hidden, hidden] bf16
                let __o_per_layer = __hidden * __hidden * 2;
                let __o_total = __num_layers * __o_per_layer;
                let __tk_o_buf = ::ferrite_cuda_core::driver::mem_alloc(__o_total)
                    .expect("tk o alloc");
                #(#o_copy_stmts)*
                __TK_O.store(__tk_o_buf as u64, Ordering::Relaxed);

                // Gate + Up: split from fused [gate|up]
                let __gate_per_layer = __intermediate * __hidden * 2;
                let __gate_total = __num_layers * __gate_per_layer;
                let __tk_gate_buf = ::ferrite_cuda_core::driver::mem_alloc(__gate_total)
                    .expect("tk gate alloc");
                let __tk_up_buf = ::ferrite_cuda_core::driver::mem_alloc(__gate_total)
                    .expect("tk up alloc");
                #(#gate_up_copy_stmts)*
                __TK_GATE.store(__tk_gate_buf as u64, Ordering::Relaxed);
                __TK_UP.store(__tk_up_buf as u64, Ordering::Relaxed);

                // Down: [num_layers, hidden, intermediate] bf16
                let __down_per_layer = __hidden * __intermediate * 2;
                let __down_total = __num_layers * __down_per_layer;
                let __tk_down_buf = ::ferrite_cuda_core::driver::mem_alloc(__down_total)
                    .expect("tk down alloc");
                #(#down_copy_stmts)*
                __TK_DOWN.store(__tk_down_buf as u64, Ordering::Relaxed);

                // Attn norm: [num_layers, hidden] bf16
                let __norm_per_layer = __hidden * 2;
                let __norm_total = __num_layers * __norm_per_layer;
                let __tk_attn_norm_buf = ::ferrite_cuda_core::driver::mem_alloc(__norm_total)
                    .expect("tk attn_norm alloc");
                #(#attn_norm_copy_stmts)*
                __TK_ATTN_NORM.store(__tk_attn_norm_buf as u64, Ordering::Relaxed);

                // MLP norm: [num_layers, hidden] bf16
                let __tk_mlp_norm_buf = ::ferrite_cuda_core::driver::mem_alloc(__norm_total)
                    .expect("tk mlp_norm alloc");
                #(#mlp_norm_copy_stmts)*
                __TK_MLP_NORM.store(__tk_mlp_norm_buf as u64, Ordering::Relaxed);

                // Sync stacking copies.
                ::ferrite_cuda_core::driver::stream_synchronize(__stream)
                    .expect("tk weight stack sync");
            });

            let __qkv_ptr = __TK_QKV.load(Ordering::Relaxed) as u64;
            let __o_ptr = __TK_O.load(Ordering::Relaxed) as u64;
            let __gate_ptr = __TK_GATE.load(Ordering::Relaxed) as u64;
            let __up_ptr = __TK_UP.load(Ordering::Relaxed) as u64;
            let __down_ptr = __TK_DOWN.load(Ordering::Relaxed) as u64;
            let __attn_norm_ptr = __TK_ATTN_NORM.load(Ordering::Relaxed) as u64;
            let __mlp_norm_ptr = __TK_MLP_NORM.load(Ordering::Relaxed) as u64;
            let __lm_head_norm_ptr = wm.#lm_head_norm_ident.weight.raw_ptr() as u64;
            let __lm_head_ptr = wm.#lm_head_ident.dense_weight().raw_ptr() as u64;

            // ── Upload instruction tensor to device ──
            let __max_inst_per_sm: usize = #max_per_sm_lit;
            static INST_DATA: &[u32] = &[#(#inst_data_tokens),*];
            let __inst_bytes = #total_words_lit * 4;
            let __inst_buf = device.caching.alloc(__inst_bytes);
            ::ferrite_cuda_core::driver::memcpy_htod_async(
                __inst_buf, INST_DATA.as_ptr() as *const u8, __inst_bytes,
                __stream,
            ).expect("instruction upload failed");

            // ── Allocate and zero barrier counters ──
            let __bar_bytes = #num_barriers_lit * 4;
            let __bar_buf = device.caching.alloc(__bar_bytes);
            ::ferrite_cuda_core::driver::memset_d8(
                __bar_buf, 0, __bar_bytes, __stream,
            ).expect("barrier memset failed");

            // ── Timing buffer (required by KVM: [num_sms][max_inst_per_sm][128]) ──
            let __timing_bytes = __num_sms * __max_inst_per_sm * 128 * 4;
            let __timing_buf = device.caching.alloc(__timing_bytes);
            ::ferrite_cuda_core::driver::memset_d8(
                __timing_buf, 0, __timing_bytes, __stream,
            ).expect("timing memset failed");

            // ── Embedding gather: input_ids → hidden_states ──
            let __hidden_states = ::ferrite_kernels::kernels::embedding_gather(
                wm.#embed_ident.weight,
                *ctx.input_ids,
                &mut device.caching,
                __stream,
            );

            // ── Activation buffers ──
            let __q_post_rope = device.caching.alloc_tensor(
                &[1, __num_heads * __head_dim],
                ::ferrite_cuda_core::DType::BF16,
            );
            let __attn_out = device.caching.alloc_tensor(
                &[1, __hidden],
                ::ferrite_cuda_core::DType::BF16,
            );
            let __attn_lse_rows = ((__num_sms + 15) / 16) * 16;
            let __attn_lse = device.caching.alloc_tensor(
                &[__num_heads, __attn_lse_rows],
                ::ferrite_cuda_core::DType::F32,
            );
            let __attn_out_int = device.caching.alloc_tensor(
                &[__num_heads, __num_sms, __head_dim],
                ::ferrite_cuda_core::DType::F32,
            );
            let __silu_out = device.caching.alloc_tensor(
                &[1, __intermediate],
                ::ferrite_cuda_core::DType::BF16,
            );
            let __logits = device.caching.alloc_tensor(
                &[1, __vocab],
                ::ferrite_cuda_core::DType::BF16,
            );

            // ── KV cache: stack per-layer caches into contiguous buffers ──
            let __kv_num_blocks = (*ctx.kv_cache.k_cache(0)).dim(0);
            let __kv_block_size = (*ctx.kv_cache.k_cache(0)).dim(1);
            let __k_per_layer = __kv_num_blocks * __kv_block_size * __num_kv_heads * __head_dim * 2;
            let __k_total = __num_layers * __k_per_layer;
            let __k_stacked = device.caching.alloc(__k_total);
            let __v_stacked = device.caching.alloc(__k_total);
            for __layer in 0..__num_layers {
                ::ferrite_cuda_core::driver::memcpy_dtod_async(
                    __k_stacked.add(__layer * __k_per_layer),
                    (*ctx.kv_cache.k_cache(__layer)).raw_ptr(),
                    __k_per_layer,
                    __stream,
                ).expect("k cache stack");
                ::ferrite_cuda_core::driver::memcpy_dtod_async(
                    __v_stacked.add(__layer * __k_per_layer),
                    (*ctx.kv_cache.v_cache(__layer)).raw_ptr(),
                    __k_per_layer,
                    __stream,
                ).expect("v cache stack");
            }

            // ── RoPE tables (separate cos, sin as f32) ──
            let __rope_cos_ptr = ctx.rotary.cos_cache.raw_ptr() as u64;
            let __rope_sin_ptr = ctx.rotary.sin_cache.raw_ptr() as u64;
            let __rope_rows = ctx.rotary.cos_cache.dim(0) as i32;

            // ── Scalars ──
            let __pos_id: u32 = {
                let mut __val = 0u32;
                ::ferrite_cuda_core::driver::memcpy_dtoh_async(
                    &mut __val as *mut u32 as *mut u8,
                    (*ctx.positions).raw_ptr(),
                    4,
                    __stream,
                ).expect("pos read");
                ::ferrite_cuda_core::driver::stream_synchronize(__stream)
                    .expect("pos sync");
                __val
            };
            let __attn_scale: f32 = 1.0 / (__head_dim as f32).sqrt();
            let __rms_norm_eps: f32 = wm.#first_attn_norm.eps;

            // ── Launch TK megakernel ──
            let __stream_u64 = __stream as u64;
            let __ret = #launch_fn(
                // VM state
                __bar_buf as u64,                              // bar_ptr
                __num_layers as i32,                           // bar_depth
                (__num_heads + 2 * __num_kv_heads) as i32,     // bar_rows
                __inst_buf as u64,                             // instructions_ptr
                __num_sms as i32,                              // instructions_depth
                __max_inst_per_sm as i32,                      // instructions_rows
                __timing_buf as u64,                           // timings_ptr
                // Weights (stacked across layers)
                __qkv_ptr,                                     // qkv_weights_ptr
                __num_layers as i32,                           // qkv_weights_depth
                (__qkv_out as i32),                            // qkv_weights_rows
                __o_ptr,                                       // o_weights_ptr
                __num_layers as i32,                           // o_weights_depth
                __hidden as i32,                               // o_weights_rows
                __up_ptr,                                      // up_weights_ptr
                __num_layers as i32,                           // up_weights_depth
                __intermediate as i32,                         // up_weights_rows
                __gate_ptr,                                    // gate_weights_ptr
                __num_layers as i32,                           // gate_weights_depth
                __intermediate as i32,                         // gate_weights_rows
                __lm_head_ptr,                                 // lm_head_weights_ptr
                1i32,                                          // lm_head_weights_depth
                (__vocab as i32),                               // lm_head_weights_rows
                __down_ptr,                                    // down_weights_ptr
                __num_layers as i32,                           // down_weights_depth
                __hidden as i32,                               // down_weights_rows
                // Norm weights (stacked across layers)
                __attn_norm_ptr,                               // attn_norm_weights_ptr
                __num_layers as i32,                           // attn_norm_weights_rows
                __mlp_norm_ptr,                                // mlp_norm_weights_ptr
                __num_layers as i32,                           // mlp_norm_weights_rows
                __lm_head_norm_ptr,                            // lm_head_norm_weights_ptr
                1i32,                                          // lm_head_norm_weights_rows
                // KV cache (stacked)
                __k_stacked as u64,                            // k_cache_ptr
                (__num_kv_heads as i32),                       // k_cache_batch
                (__kv_num_blocks as i32),                      // k_cache_depth
                (__kv_block_size as i32),                      // k_cache_rows
                __v_stacked as u64,                            // v_cache_ptr
                (__num_kv_heads as i32),                       // v_cache_batch
                (__kv_num_blocks as i32),                      // v_cache_depth
                (__kv_block_size as i32),                      // v_cache_rows
                // RoPE
                __rope_cos_ptr,                                // rope_cos_ptr
                __rope_sin_ptr,                                // rope_sin_ptr
                __rope_rows,                                   // rope_rows
                // Activation buffers
                __hidden_states.raw_ptr() as u64,     // hidden_states_ptr
                __q_post_rope.raw_ptr() as u64,       // q_post_rope_ptr
                __attn_out.raw_ptr() as u64,          // attn_out_ptr
                __attn_lse.raw_ptr() as u64,          // attn_lse_ptr
                __attn_lse_rows as i32,                        // attn_lse_rows
                __attn_out_int.raw_ptr() as u64,      // attn_out_intermediates_ptr
                __num_sms as i32,                              // attn_out_intermediates_rows
                __silu_out.raw_ptr() as u64,          // silu_out_ptr
                __logits.raw_ptr() as u64,            // logits_ptr
                __vocab as i32,                                // logits_cols
                // Scalars
                __pos_id,                                      // pos_id
                __attn_scale,                                  // attn_scale
                __rms_norm_eps,                                // rms_norm_eps
                1i32,                                          // skip_attn_reduction (=true, 1 partition)
                // Stream
                __stream_u64,
            );
            assert_eq!(__ret, 0, "TK megakernel launch failed (CUDA error {})", __ret);

            // ── Copy KV cache back from stacked buffer ──
            // The TK kernel writes new K/V tokens during rms_qkv_rope_append.
            // Copy updated caches back to the per-layer KvCachePool tensors.
            for __layer in 0..__num_layers {
                ::ferrite_cuda_core::driver::memcpy_dtod_async(
                    (*ctx.kv_cache.k_cache(__layer)).raw_ptr(),
                    __k_stacked.add(__layer * __k_per_layer),
                    __k_per_layer,
                    __stream,
                ).expect("k cache writeback");
                ::ferrite_cuda_core::driver::memcpy_dtod_async(
                    (*ctx.kv_cache.v_cache(__layer)).raw_ptr(),
                    __v_stacked.add(__layer * __k_per_layer),
                    __k_per_layer,
                    __stream,
                ).expect("v cache writeback");
            }

            // Return logits.
            __logits
        }
    };

    Some(tokens)
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
    if dc_group.is_empty() {
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
        "unsigned int" => quote! { u32 },
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
        !phases.is_empty(),
        "megakernel requires at least 1 phase (got {})",
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
    writeln!(src, "#include <cstdint>").unwrap();
    writeln!(src, "#include <cuda_runtime.h>").unwrap();
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

    let num_barriers = if phases.len() > 1 {
        phases.len() - 1
    } else {
        0
    };

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
    writeln!(src, "    void* args[] = {{ &params }};").unwrap();
    writeln!(src, "    return cudaLaunchCooperativeKernel(").unwrap();
    writeln!(src, "        (void*){kernel_name},").unwrap();
    writeln!(
        src,
        "        dim3(__grid_x), dim3(__block_x), args, __smem_bytes, (cudaStream_t)__stream);"
    )
    .unwrap();
    writeln!(src, "}}").unwrap();

    crate::cuda_codegen::GeneratedMegakernel {
        cuda_source: src,
        launch_fn_name,
        flat_params: all_flat,
        num_barriers,
    }
}

/// Emit one per-workload-bucket forward fn.
#[allow(clippy::too_many_arguments)]
fn emit_forward_for_bucket(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    program: &Program,
    model: &ModelParams,
    target: &TargetProfile,
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

    // SM90+ (Hopper/Blackwell): try TK megakernel path.
    // This replaces the entire wave-by-wave BSP emission with a
    // single kernel using the KVM runtime (warp-specialized, TMA
    // pipelined, per-SM static instruction queues).
    if target.compute_capability >= 90
        && let Some(tk_tokens) = try_emit_tk_forward(
            fuf, sfuf, loop_ir, program, model, target, lib, &locals, num_tokens,
        )
    {
        return tk_tokens;
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
        target,
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
#[allow(clippy::too_many_arguments)]
fn emit_forward_backbone_for_bucket(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    program: &Program,
    model: &ModelParams,
    target: &TargetProfile,
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
        target,
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
    target: &TargetProfile,
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
            emit_forward_for_bucket(fuf, sfuf, loop_ir, program, model, target, lib, *m)
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
            emit_forward_backbone_for_bucket(fuf, sfuf, loop_ir, program, model, target, lib, *m)
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
