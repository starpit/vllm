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

use std::collections::{BTreeMap, HashMap};

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::classified::{OpKind, Program, WeightId};
use crate::config::ModelParams;
use crate::emit::{EmitCtx, LocalMap};
use crate::fuf::Fuf;
use crate::impl_lib::{ImplementationLibrary, WeightAccessor};
use crate::schedule::{Loop, WorkloadLoops};
use crate::solver::{Assignment, WorkloadAssignments};

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
}

/// Distill a `WeightAccessor` into its field-load plan. Uses the
/// accessor's declared `rust_type` + `source_weights` and the
/// model's bounds (for `rms_norm_eps`).
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

    let loaders = accessors.iter().map(|a| {
        let name = &a.name;
        let plan = plan_field_load(a, program, model);
        match plan {
            FieldLoad::Embedding(prefix) => quote! {
                #name: ::ferrite_kernels::layers::Embedding::load(gw, #prefix)?,
            },
            FieldLoad::RmsNorm(prefix, eps) => quote! {
                #name: ::ferrite_kernels::layers::RmsNorm::load(gw, #prefix, #eps)?,
            },
            FieldLoad::LinearDense(prefix) => quote! {
                #name: ::ferrite_kernels::layers::LinearLayer::load_dense(gw, #prefix)?,
            },
            FieldLoad::LinearConcat(prefixes) => quote! {
                #name: ::ferrite_kernels::layers::LinearLayer::load_dense_concat(
                    gw,
                    &[ #(#prefixes),* ],
                    stream,
                )?,
            },
        }
    });

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
            pub fn load(
                gw: &mut ::ferrite_cuda_core::weights::GpuWeights,
                stream: ::ferrite_cuda_core::CUstream,
            ) -> ::anyhow::Result<Self> {
                Ok(Self {
                    #(#loaders)*
                })
            }
        }
    }
}

// ── Forward fn emission ──────────────────────────────────────────

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

    // Walk waves in order; each subgraph's bound impl emits its own
    // kernel invocation. Codegen stays mechanical — no per-op match
    // lives here.
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

    // match num_tokens dispatch. Coalesce adjacent buckets with
    // the same forward fn in a future pass; for now one arm per
    // bucket.
    let match_arms: Vec<TokenStream> = sfufs
        .per_num_tokens
        .keys()
        .map(|m| {
            let lit = proc_macro2::Literal::u64_unsuffixed(*m);
            let fn_name = format_ident!("forward_m_{}", m);
            quote! { #lit => unsafe { #fn_name(wm, ctx, device) }, }
        })
        .collect();

    quote! {
        #weights

        #(#bucket_fns)*

        /// Dispatch on `num_tokens`. Panics if the runtime
        /// num_tokens isn't one of the compiled buckets.
        /// Future: coalesce into inclusive ranges.
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
                other => panic!("no compiled bucket for num_tokens={other}"),
            }
        }
    }
}

/// Unused — consumers used to reach this by name from tests.
/// Retained as a no-op type anchor while the trait-based shim is
/// being deleted from downstream call sites.
#[allow(dead_code)]
fn _unused(_: OpKind) {}
