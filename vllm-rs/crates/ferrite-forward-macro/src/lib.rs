// SPDX-License-Identifier: Apache-2.0
//! `#[forward]` attribute macro — compile a DSL forward pass into
//! specialized Rust + CUDA per model architecture.
//!
//! This crate is the proc-macro shell. Every compiler pass (parse,
//! classify, shape-infer, CFG, unroll, solve, schedule) lives in
//! internal modules here so they can be unit-tested in isolation.
//! The attribute macro drives them end-to-end at macro expansion
//! time: it reads the configs + target profile, runs the whole
//! pipeline for every (model × workload-point), and emits the
//! generated code.
//!
//! Until codegen (PLAN task #5) lands, the emitted code is a
//! placeholder `pub mod <model>` per model containing constants
//! derived from the real pipeline (tile count, wave count,
//! predicted cost per workload). Integration tests assert those
//! constants, proving the pipeline actually executes at compile
//! time — the failure mode `feedback_integration_test_per_phase`
//! memorializes.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Ident, ItemFn, LitInt, LitStr, Token, parse_macro_input};

mod ast;
mod cfg;
mod classified;
mod classify;
mod codegen;
mod concurrency;
mod config;
mod cost;
mod emit;
mod fuf;
mod impl_lib;
mod parse;
mod schedule;
mod shape;
mod solver;
mod target;
mod weight_conventions;

// ── Attribute argument parsing ────────────────────────────────────

struct ForwardArgs {
    /// Path to the directory of config.json files for this
    /// architecture, relative to CARGO_MANIFEST_DIR of the invoking
    /// crate.
    models_dir: LitStr,
    /// Path to the target profile JSON, relative to
    /// CARGO_MANIFEST_DIR of the invoking crate.
    target: LitStr,
    /// Discrete `num_tokens` points to solve at. Non-empty.
    workloads: Vec<u64>,
    /// Span used for error reporting when a required arg is
    /// missing.
    span: Span,
}

impl Parse for ForwardArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let span = input.span();
        let mut models_dir: Option<LitStr> = None;
        let mut target: Option<LitStr> = None;
        let mut workloads: Option<Vec<u64>> = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "models_dir" => models_dir = Some(input.parse()?),
                "target" => target = Some(input.parse()?),
                "workloads" => {
                    let list;
                    syn::bracketed!(list in input);
                    let mut pts = Vec::new();
                    while !list.is_empty() {
                        let n: LitInt = list.parse()?;
                        pts.push(n.base10_parse::<u64>()?);
                        if !list.is_empty() {
                            list.parse::<Token![,]>()?;
                        }
                    }
                    workloads = Some(pts);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown #[forward] argument: `{other}`"),
                    ));
                }
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        let models_dir =
            models_dir.ok_or_else(|| syn::Error::new(span, "#[forward] missing `models_dir`"))?;
        let target = target.ok_or_else(|| syn::Error::new(span, "#[forward] missing `target`"))?;
        let workloads =
            workloads.ok_or_else(|| syn::Error::new(span, "#[forward] missing `workloads`"))?;
        if workloads.is_empty() {
            return Err(syn::Error::new(span, "#[forward] `workloads` is empty"));
        }

        Ok(Self {
            models_dir,
            target,
            workloads,
            span,
        })
    }
}

// ── Macro entry point ─────────────────────────────────────────────

#[proc_macro_attribute]
pub fn forward(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as ForwardArgs);
    let carrier = parse_macro_input!(item as ItemFn);

    match compile(&args, &carrier) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn compile(args: &ForwardArgs, carrier: &ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    // CARGO_MANIFEST_DIR at macro-expansion time is the invoking
    // crate's root. All configured paths resolve against it.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").map_err(|_| {
        syn::Error::new(
            args.span,
            "CARGO_MANIFEST_DIR not set — cannot resolve models_dir/target paths",
        )
    })?;
    let base = std::path::PathBuf::from(manifest_dir);
    let models_dir = base.join(args.models_dir.value());
    let target_path = base.join(args.target.value());

    // ── Front end: parse + classify + shape-infer ─────────────────
    let ast = parse::parse_block(&carrier.block)
        .map_err(|e| syn::Error::new(args.span, format!("parse: {e}")))?;
    let classified = classify::classify(&ast)
        .map_err(|e| syn::Error::new(args.span, format!("classify: {e}")))?;
    let inferred = shape::infer(&classified)
        .map_err(|e| syn::Error::new(args.span, format!("shape infer: {e}")))?;

    // ── Load configs + target + library ────────────────────────────
    let models = config::load_dir(&models_dir).map_err(|e| {
        syn::Error::new(
            args.models_dir.span(),
            format!("models_dir `{}`: {e}", models_dir.display()),
        )
    })?;
    if models.is_empty() {
        return Err(syn::Error::new(
            args.models_dir.span(),
            format!("no *.json configs in {}", models_dir.display()),
        ));
    }
    let target_profile = target::load_file(&target_path).map_err(|e| {
        syn::Error::new(
            args.target.span(),
            format!("target `{}`: {e}", target_path.display()),
        )
    })?;

    let library = impl_lib::starter_library();

    // Stable rebuild-on-JSON-change: emit `const _: &str =
    // include_str!("<abs path>");` for every file the macro read.
    // Rustc treats `include_str!` paths as source-dependency inputs
    // and cargo rebuilds the caller when any of them change. Works
    // on stable; no build.rs or nightly feature required.
    let mut tracked: Vec<proc_macro2::TokenStream> = Vec::new();
    for m in &models {
        let p = m.source_path.to_string_lossy().into_owned();
        let lit = syn::LitStr::new(&p, proc_macro2::Span::call_site());
        tracked.push(quote! { const _: &str = include_str!(#lit); });
    }
    {
        let p = target_path.to_string_lossy().into_owned();
        let lit = syn::LitStr::new(&p, proc_macro2::Span::call_site());
        tracked.push(quote! { const _: &str = include_str!(#lit); });
    }

    // ── Per-model × per-workload pipeline ─────────────────────────
    let mut per_model_ts: Vec<proc_macro2::TokenStream> = Vec::new();
    // Collected per-model identifying shape for the arch-level
    // Weights enum's `load` dispatch. Each entry = (model_ident,
    // bounds-tuple literals in the fixed order below).
    let mut arch_dispatch_arms: Vec<(Ident, Vec<u64>)> = Vec::new();

    for model in &models {
        let model_cfg = cfg::build_cfg(&classified, model)
            .map_err(|e| syn::Error::new(args.span, format!("cfg [{}]: {e}", model.source_stem)))?;
        let model_fuf = fuf::unroll(&model_cfg, &inferred).map_err(|e| {
            syn::Error::new(args.span, format!("unroll [{}]: {e}", model.source_stem))
        })?;

        let mut sfufs = solver::solve(
            &model_fuf,
            &library,
            &target_profile,
            &inferred,
            &model.bounds,
            &args.workloads,
        )
        .map_err(|e| syn::Error::new(args.span, format!("solve [{}]: {e}", model.source_stem)))?;

        let loops = schedule::schedule_workloads(&model_fuf, &sfufs);

        // Post-scheduler: recompute each bucket's `predicted_us`
        // with per-wave contention applied via ConcurrencyModel.
        // On the current all-HostCallback serial-chain library this
        // doesn't change the number, but the machinery stays ready
        // for DeviceCallable + megakernel waves.
        cost::refresh_predicted_us(
            &model_fuf,
            &mut sfufs,
            &loops,
            &library,
            &target_profile,
            &model.bounds,
        );

        let stub_items = emit_model_stub_items(&model_fuf, &sfufs, &loops);
        let codegen_items =
            codegen::emit_model(&classified, model, &model_fuf, &sfufs, &loops, &library);
        let model_mod = &model.name;
        per_model_ts.push(quote! {
            pub mod #model_mod {
                #stub_items
                #codegen_items
            }
        });

        arch_dispatch_arms.push((model.name.clone(), collect_dispatch_bounds(model)));
    }

    let arch_dispatch_ts = emit_arch_dispatcher(&arch_dispatch_arms);

    // Group every model's emitted module under one `pub mod <arch>`
    // matching the carrier fn's name. Callers access as
    // `llama::llama_3_2_1b::NUM_TILES` etc. — the architecture
    // identifier scopes the models it owns.
    let arch_mod = &carrier.sig.ident;

    Ok(quote! {
        pub mod #arch_mod {
            // Rebuild-on-change for every JSON the macro read.
            #(#tracked)*

            #(#per_model_ts)*

            #arch_dispatch_ts
        }
    })
}

/// The identifying HF-config fields the arch-level `Weights::load`
/// matches on, in a fixed order. Any two compiled models that agree
/// on all seven values would collide; bump this if you add an arch
/// where that happens.
const DISPATCH_FIELDS: &[&str] = &[
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
];

fn collect_dispatch_bounds(model: &config::ModelParams) -> Vec<u64> {
    DISPATCH_FIELDS
        .iter()
        .map(|k| {
            *model.bounds.get(*k).unwrap_or_else(|| {
                panic!(
                    "model `{}` is missing required bound `{k}`; add it to \
                     config.json or to weight_conventions::derive_implicit_bounds",
                    model.source_stem,
                )
            })
        })
        .collect()
}

/// Arch-level dispatcher: an enum over every compiled variant plus
/// a `load` that fingerprints the runtime config and a `forward`
/// that delegates to the matched variant's per-model forward.
fn emit_arch_dispatcher(arms: &[(Ident, Vec<u64>)]) -> proc_macro2::TokenStream {
    if arms.is_empty() {
        return quote! {};
    }

    // Variant ident = PascalCase of the model ident (e.g.
    // `llama_3_2_1b` → `Llama_3_2_1b`). Keep the underscores — they
    // carry meaning (dotted-version components) and collapsing them
    // would create ambiguity between e.g. `llama32` and `llama_3_2`.
    let variants: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|(model_ident, _)| {
            let variant_ident = pascal_case(model_ident);
            quote! { #variant_ident(#model_ident::Weights) }
        })
        .collect();

    let load_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|(model_ident, bounds)| {
            let variant_ident = pascal_case(model_ident);
            let bound_lits: Vec<proc_macro2::Literal> = bounds
                .iter()
                .map(|b| proc_macro2::Literal::u64_unsuffixed(*b))
                .collect();
            quote! {
                (#(#bound_lits),*) => Ok(Self::#variant_ident(
                    #model_ident::Weights::load(gw, stream)?,
                )),
            }
        })
        .collect();

    let forward_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|(model_ident, _)| {
            let variant_ident = pascal_case(model_ident);
            quote! {
                Weights::#variant_ident(w) => unsafe {
                    #model_ident::forward(w, ctx, device, num_tokens)
                },
            }
        })
        .collect();

    let fields: Vec<proc_macro2::TokenStream> = DISPATCH_FIELDS
        .iter()
        .map(|k| {
            let id = Ident::new(k, Span::call_site());
            quote! { #id: u64 }
        })
        .collect();
    let field_names: Vec<Ident> = DISPATCH_FIELDS
        .iter()
        .map(|k| Ident::new(k, Span::call_site()))
        .collect();

    quote! {
        /// One variant per compiled model config. Holds that
        /// model's specialized `Weights`.
        #[cfg(feature = "cuda")]
        pub enum Weights {
            #(#variants),*
        }

        #[cfg(feature = "cuda")]
        impl Weights {
            /// Auto-detect the compiled variant from the runtime
            /// HF-config fields, load weights (streaming concat for
            /// fused accessors), and return the enum-wrapped result.
            ///
            /// Errors if no compiled variant's fingerprint matches.
            #[allow(clippy::too_many_arguments)]
            pub fn load(
                gw: &mut ::ferrite_cuda_core::weights::GpuWeights,
                stream: ::ferrite_cuda_core::CUstream,
                #(#fields),*
            ) -> ::anyhow::Result<Self> {
                match (#(#field_names),*) {
                    #(#load_arms)*
                    shape => ::anyhow::bail!(
                        "no compiled variant matches config fingerprint {:?}",
                        shape,
                    ),
                }
            }
        }

        /// Dispatching forward. Matches the `Weights` variant and
        /// calls the per-model specialized `forward`.
        ///
        /// # Safety
        /// All tensors in `ctx` must be valid GPU memory; `device`
        /// must be the live CUDA device.
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward(
            w: &Weights,
            ctx: &::ferrite_forward::ForwardCtx,
            device: &mut ::ferrite_cuda_core::device::GpuDevice,
            num_tokens: u64,
        ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
            match w {
                #(#forward_arms)*
            }
        }
    }
}

/// PascalCase a snake_case ident while preserving underscores
/// between segments (they're version separators in our model names).
fn pascal_case(ident: &Ident) -> Ident {
    let s = ident.to_string();
    let mut out = String::with_capacity(s.len());
    let mut capitalize_next = true;
    for c in s.chars() {
        if c == '_' {
            out.push('_');
            capitalize_next = true;
        } else if capitalize_next {
            out.extend(c.to_uppercase());
            capitalize_next = false;
        } else {
            out.push(c);
        }
    }
    Ident::new(&out, Span::call_site())
}

/// Emit pipeline-observation constants (NUM_TILES /
/// NUM_SUBGRAPHS / NUM_WAVES / PREDICTED_US per workload) as
/// items inside the per-model module. Useful for integration
/// tests that observe the pipeline ran. Lives alongside the
/// codegen-emitted forward fns in the same module.
fn emit_model_stub_items(
    fuf: &fuf::Fuf,
    sfufs: &solver::WorkloadAssignments,
    loops: &schedule::WorkloadLoops,
) -> proc_macro2::TokenStream {
    let num_tiles = fuf.len();
    let mut workload_ts: Vec<proc_macro2::TokenStream> = Vec::new();
    for (m, sfuf) in &sfufs.per_num_tokens {
        let loop_ir = loops
            .per_num_tokens
            .get(m)
            .expect("schedule_workloads populates every key");
        let wl_mod = Ident::new(&format!("m_{m}"), Span::call_site());
        let num_subgraphs = sfuf.num_subgraphs();
        let num_waves = loop_ir.num_waves();
        let predicted_us = sfuf.predicted_us;
        workload_ts.push(quote! {
            pub mod #wl_mod {
                pub const NUM_SUBGRAPHS: usize = #num_subgraphs;
                pub const NUM_WAVES: usize = #num_waves;
                pub const PREDICTED_US: f64 = #predicted_us;
            }
        });
    }
    quote! {
        pub const NUM_TILES: usize = #num_tiles;
        #(#workload_ts)*
    }
}
