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
mod class_impl;
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
mod periodicity;
mod periodicity_plan;
mod quantization;
mod region_formation;
mod schedule;
mod shape;
mod solver;
mod subtile;
mod target;
mod viz_dump;
mod weights_manifest;

// ── Attribute argument parsing ────────────────────────────────────

struct ForwardArgs {
    /// Path to the target profile JSON, relative to
    /// CARGO_MANIFEST_DIR of the invoking crate.
    target: LitStr,
    /// Discrete `num_tokens` points to solve at. Non-empty.
    workloads: Vec<u64>,
    /// Discrete `sk_bucket` (KV-cache span in tokens) points to solve
    /// at. Optional — when absent, the solver sweeps only the
    /// `num_tokens` axis with `sk_bucket = 0` (the sentinel "sk
    /// axis unused"). Declare this for models where attention
    /// dispatch wants to pick different kernels at different KV
    /// spans — e.g. FlashInfer decode wins on long sk, FA2 wins at
    /// small prefill.
    sk_buckets: Vec<u64>,
    /// Span used for error reporting when a required arg is
    /// missing.
    span: Span,
}

impl Parse for ForwardArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let span = input.span();
        let mut target: Option<LitStr> = None;
        let mut workloads: Option<Vec<u64>> = None;
        let mut sk_buckets: Option<Vec<u64>> = None;

        fn parse_u64_list(input: ParseStream) -> syn::Result<Vec<u64>> {
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
            Ok(pts)
        }

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "target" => target = Some(input.parse()?),
                "workloads" => workloads = Some(parse_u64_list(input)?),
                "sk_buckets" => sk_buckets = Some(parse_u64_list(input)?),
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

        let target = target.ok_or_else(|| syn::Error::new(span, "#[forward] missing `target`"))?;
        let workloads =
            workloads.ok_or_else(|| syn::Error::new(span, "#[forward] missing `workloads`"))?;
        if workloads.is_empty() {
            return Err(syn::Error::new(span, "#[forward] `workloads` is empty"));
        }
        // Default: 1-D sweep. `sk_bucket = 0` is the sentinel value
        // that non-sk-constrained impls (Any / NumTokensRange) accept
        // unconditionally; FI impls with a real sk range won't match,
        // so they can't be picked unless the model declares a real
        // sk_buckets list.
        let sk_buckets = sk_buckets.unwrap_or_default();

        Ok(Self {
            target,
            workloads,
            sk_buckets,
            span,
        })
    }
}

/// Discover `model_architectures/<arch>` for the given arch
/// identifier by walking up from `start` (the invoking crate's
/// manifest dir) looking for a parent that contains a
/// `model_architectures` directory with a `<arch>` child.
///
/// This is how `#[forward] fn llama() { ... }` knows to read
/// `model_architectures/llama/*.json` without the user spelling
/// out `models_dir`. Walk-up stops at the first match, or returns
/// an error naming every directory it checked.
/// Format a microsecond value adaptively for human scanning:
/// `<1000µs` as `Nµs`, `<100ms` as `N.Xms`, else `Nms`.
/// Cross-variant forward-fn dedup. Returns a map `variant_idx →
/// canonical_module_ident`, where `canonical_module_ident` is the
/// module name of the variant chosen to carry the full emitted
/// forward-fn bodies for its equivalence class. Variants whose
/// canonical is themselves emit full bodies; others emit shims.
///
/// The key — what makes two variants' forward fn bodies
/// byte-identical — is:
/// 1. The arch's DSL (always shared within a `#[forward]` call).
/// 2. The variant's integer `bounds` (baked as literals in
///    `ctx.bound(...)` and the unroll trip counts).
/// 3. The variant's float `scalars` (baked as literals in
///    `attention_scale_for` / similar).
/// 4. The SFUF per workload point (which `Impl` runs at each
///    subgraph → whose `emit_call` output lands in the body).
///
/// Among AWQ / GPTQ / CT variants of the same dense base, items
/// 1-3 are identical and item 4 collapses because they all resolve
/// to `Marlin*Impl`. Dense + BNB4 stay separate because their
/// `Impl` picks differ from Marlin's and from each other's.
///
/// Canonical selection: the variant with the earliest
/// `source_stem` in the equivalence class wins. Deterministic
/// across macro re-expansions so cargo's incremental cache stays
/// stable.
fn compute_canonical_variants(
    solved: &[impl HasSolvedSig],
) -> std::collections::HashMap<usize, Ident> {
    let mut by_sig: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, sm) in solved.iter().enumerate() {
        by_sig.entry(sm.dedup_signature()).or_default().push(i);
    }

    let mut out: std::collections::HashMap<usize, Ident> = std::collections::HashMap::new();
    for members in by_sig.values() {
        // Pick the member with the alphabetically-earliest stem as
        // canonical. All other members point to it.
        let mut ordered = members.clone();
        ordered.sort_by_key(|&i| solved[i].source_stem().to_string());
        let canonical_idx = ordered[0];
        let canonical_ident = Ident::new(
            solved[canonical_idx].model_name(),
            proc_macro2::Span::call_site(),
        );
        for &idx in &ordered {
            out.insert(idx, canonical_ident.clone());
        }
    }
    out
}

/// Trait over the per-variant fields `compute_canonical_variants`
/// needs; implemented inline on the `SolvedModel` wrapper inside
/// `forward_impl`. Keeps the helper callable without plumbing the
/// concrete `SolvedModel` type through.
trait HasSolvedSig {
    fn dedup_signature(&self) -> String;
    fn source_stem(&self) -> &str;
    fn model_name(&self) -> &str;
}

fn fmt_us(us: f64) -> String {
    if us < 1000.0 {
        format!("{us:.0}µs")
    } else if us < 100_000.0 {
        format!("{:.1}ms", us / 1000.0)
    } else {
        format!("{:.0}ms", us / 1000.0)
    }
}

fn discover_models_dir(start: &std::path::Path, arch: &str) -> Result<std::path::PathBuf, String> {
    let mut checked: Vec<std::path::PathBuf> = Vec::new();
    let mut cur: Option<&std::path::Path> = Some(start);
    while let Some(dir) = cur {
        let candidate = dir.join("model_architectures").join(arch);
        if candidate.is_dir() {
            return Ok(candidate);
        }
        checked.push(candidate);
        cur = dir.parent();
    }
    Err(format!(
        "no `model_architectures/{arch}` directory found walking up from {}. \
         Searched: {}",
        start.display(),
        checked
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", "),
    ))
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
    // crate's root. Target path resolves against it; the arch
    // `model_architectures/<arch>` directory is discovered by
    // walking up from here, using the carrier's fn name as <arch>.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").map_err(|_| {
        syn::Error::new(
            args.span,
            "CARGO_MANIFEST_DIR not set — cannot resolve paths",
        )
    })?;
    let base = std::path::PathBuf::from(manifest_dir);
    let arch_name = carrier.sig.ident.to_string();
    let models_dir = discover_models_dir(&base, &arch_name)
        .map_err(|e| syn::Error::new(carrier.sig.ident.span(), e))?;
    let target_path = base.join(args.target.value());

    // ── Front end: parse + classify ───────────────────────────────
    let ast = parse::parse_block(&carrier.block)
        .map_err(|e| syn::Error::new(args.span, format!("parse: {e}")))?;
    let mut classified = classify::classify(&ast)
        .map_err(|e| syn::Error::new(args.span, format!("classify: {e}")))?;

    // ── Load configs + manifest + target ──────────────────────────
    // Shape inference needs both the arch's `weights.json` manifest
    // (for declared weight shapes) and one model's bounds (for
    // numerical-equivalence anchoring). The prober's cross-size
    // validation guarantees every model's bounds resolve the
    // manifest's formulas consistently, so any one model's bounds
    // suffice — we use the first (alphabetical) model.
    let models = config::load_dir(&models_dir).map_err(|e| {
        syn::Error::new(
            carrier.sig.ident.span(),
            format!("models_dir `{}`: {e}", models_dir.display()),
        )
    })?;
    if models.is_empty() {
        return Err(syn::Error::new(
            carrier.sig.ident.span(),
            format!("no *.json configs in {}", models_dir.display()),
        ));
    }
    let manifest = weights_manifest::load_or_empty(&models_dir).map_err(|e| {
        syn::Error::new(
            carrier.sig.ident.span(),
            format!("weights.json in {}: {e}", models_dir.display()),
        )
    })?;

    // Shape inference may flag reshape-recoverable mismatches (e.g.
    // per-head QK-norm in Qwen3/Gemma3). Catch those, synthesize the
    // Reshape stmts into `classified`, and retry — downstream passes
    // (CFG, FUF, solver, codegen) see a program with explicit reshape
    // tiles. Bounded to one recovery pass: a clean hint set resolves
    // on the second pass. Anything that doesn't is a compiler bug
    // we'd rather surface than loop on.
    let infer_bounds = &models[0].bounds;
    let inferred = match shape::infer(&classified, &manifest, infer_bounds) {
        Ok(inf) => inf,
        Err(shape::ShapeError::ReshapeRecovery { hints }) => {
            shape::apply_reshape_hints(&mut classified, &hints);
            shape::infer(&classified, &manifest, infer_bounds).map_err(|e| {
                syn::Error::new(
                    args.span,
                    format!("shape infer (after reshape recovery): {e}"),
                )
            })?
        }
        Err(e) => {
            return Err(syn::Error::new(args.span, format!("shape infer: {e}")));
        }
    };

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
    let mut tracked_paths: std::collections::BTreeSet<std::path::PathBuf> =
        std::collections::BTreeSet::new();
    for m in &models {
        tracked_paths.insert(m.source_path.clone());
        for extra in &m.extra_tracked_paths {
            tracked_paths.insert(extra.clone());
        }
    }
    // Arch-level metadata files: the weights manifest and the
    // quantization preset declaration. They influence compiled
    // output (manifest shapes, synthesized variant set) without
    // being per-variant source paths.
    let quantizations_path = models_dir.join("quantizations.json");
    if quantizations_path.exists() {
        tracked_paths.insert(quantizations_path);
    }
    // De-dupe before emitting; multiple variants share preset /
    // override paths and the target profile belongs to every
    // variant.
    tracked_paths.insert(target_path.clone());
    for p in &tracked_paths {
        let s = p.to_string_lossy().into_owned();
        let lit = syn::LitStr::new(&s, proc_macro2::Span::call_site());
        tracked.push(quote! { const _: &str = include_str!(#lit); });
    }

    // ── Per-model × per-workload pipeline ─────────────────────────
    //
    // Two passes: solve every variant first (gather fuf + sfufs + loops
    // per variant), then group variants whose emitted forward-fn
    // bodies would be byte-identical and emit each group's canonical
    // variant in full. Non-canonical variants emit `pub use`
    // re-exports of the canonical's forward fns + their own variant-
    // specific `Weights` type alias / `load` / `fingerprint_matches`.
    //
    // The dedup key captures every input that flows into the emitted
    // forward fn body: model bounds (baked as int literals in emit),
    // scalars (baked as float literals), and the SFUF-per-workload-
    // point (which Impl runs at each subgraph → which `emit_call`
    // output ends up in the body). Two variants that share this
    // tuple compile to byte-identical forward fns — across AWQ /
    // GPTQ / CT of the same (arch, size), for instance, because
    // they all resolve to the same `Marlin*Impl` family and the
    // quant knobs (`desc_act`, `sym`, etc.) only change the
    // per-variant `load`, never the forward.
    struct SolvedModel<'a> {
        model: &'a config::ModelParams,
        fuf: fuf::Fuf,
        sfufs: solver::WorkloadAssignments,
        loops: schedule::WorkloadLoops,
        stub_items: proc_macro2::TokenStream,
    }

    impl HasSolvedSig for SolvedModel<'_> {
        fn dedup_signature(&self) -> String {
            let mut parts: Vec<String> = Vec::new();
            // Bounds get baked as integer literals in the emitted
            // forward body (`ctx.bound("hidden_size")` expands to the
            // concrete number at macro expansion). Variants with
            // differing bounds produce different literal output.
            for (k, v) in &self.model.bounds {
                parts.push(format!("b:{k}={v}"));
            }
            // Scalars get baked as float literals (attention_scale_for,
            // softcap). Same reasoning.
            for (k, v) in &self.model.scalars {
                parts.push(format!("s:{k}={v}"));
            }
            // `tie_word_embeddings` routes lm_head through
            // `FieldLoad::LinearTiedToEmbedding` (no safetensors
            // read) vs `FieldLoad::LinearDense` — two different
            // bodies, so variants with different tie settings
            // can't share a compiled `load_with`.
            parts.push(format!("t:{}", self.model.tie_word_embeddings));
            // `rope_scaling` (short/long_factor, type, orig_max) is
            // baked into `RotaryCache::new_*` as literal arguments in
            // `load_with`. Two variants with identical bounds +
            // scalars but different `rope_scaling` (Phi-4-mini-instruct
            // vs Phi-4-mini-reasoning: all-1.0 short_factor vs the
            // non-trivial vector) MUST NOT share a canonical — the
            // shim would bake the canonical's rotary for both.
            parts.push(format!("r:{}", self.model.rope_scaling_hash.unwrap_or(0)));
            // SFUF per (num_tokens, sk_bucket) point: which Impl runs
            // at each subgraph. Identical SFUFs → each impl's
            // `emit_call` produces identical output at identical
            // positions in the body.
            let mut wps: Vec<_> = self.sfufs.per_workload.iter().collect();
            wps.sort_by_key(|(wp, _)| (wp.num_tokens, wp.sk_bucket));
            for (wp, sf) in wps {
                let mut impls: Vec<(u32, u32)> =
                    sf.impls.iter().map(|(sg, i)| (sg.0, i.0)).collect();
                impls.sort();
                parts.push(format!("w:{}-{}-{:?}", wp.num_tokens, wp.sk_bucket, impls));
            }
            parts.join("|")
        }
        fn source_stem(&self) -> &str {
            &self.model.source_stem
        }
        fn model_name(&self) -> &str {
            &self.model.name
        }
    }

    let mut solved: Vec<SolvedModel<'_>> = Vec::with_capacity(models.len());

    for model in &models {
        let model_cfg = cfg::build_cfg(&classified, model)
            .map_err(|e| syn::Error::new(args.span, format!("cfg [{}]: {e}", model.source_stem)))?;
        let mut model_fuf = fuf::unroll(&model_cfg, &inferred).map_err(|e| {
            syn::Error::new(args.span, format!("unroll [{}]: {e}", model.source_stem))
        })?;
        model_fuf.annotate_storage_formats(&classified, model);

        let t_solve = std::time::Instant::now();
        let mut sfufs = solver::solve(
            &model_fuf,
            &library,
            &target_profile,
            &inferred,
            &model.bounds,
            &args.workloads,
            &args.sk_buckets,
        )
        .map_err(|e| syn::Error::new(args.span, format!("solve [{}]: {e}", model.source_stem)))?;
        let d_solve = t_solve.elapsed();

        let loops = schedule::schedule_workloads(&model_fuf, &sfufs);
        cost::refresh_predicted_us(
            &model_fuf,
            &mut sfufs,
            &loops,
            &library,
            &target_profile,
            &model.bounds,
        );

        let max_waves = loops
            .per_workload
            .values()
            .map(|l| l.num_waves())
            .max()
            .unwrap_or(0);
        let sk_axis_active = sfufs.per_workload.keys().any(|wp| wp.sk_bucket != 0);
        let per_m: String = sfufs
            .per_workload
            .iter()
            .map(|(wp, a)| {
                if sk_axis_active {
                    format!(
                        " M={}sk={}→{}",
                        wp.num_tokens,
                        wp.sk_bucket,
                        fmt_us(a.predicted_us)
                    )
                } else {
                    format!(" M={}→{}", wp.num_tokens, fmt_us(a.predicted_us))
                }
            })
            .collect();
        eprintln!(
            "  ferrite · {variant:<30} · {tiles:>4} tiles · {waves:>3} waves · {solve_ms:>3} ms ·{per_m}",
            variant = model.source_stem,
            tiles = model_fuf.len(),
            waves = max_waves,
            solve_ms = d_solve.as_millis(),
        );

        // Stencil pipeline instrumentation (STENCIL_IR_V2_DESIGN.md steps 2-4).
        // Runs the sub-tile → region-formation → periodicity pipeline on
        // every SFUF at the first workload point for diagnostic printing.
        // Zero effect on emitted code — pure measurement of where the
        // collapse factor lands on real transformer graphs.
        if let Some((_wp, sfuf)) = sfufs.per_workload.iter().next() {
            let st = subtile::subtile(&model_fuf);
            let fr = region_formation::form_regions(&st, sfuf);
            let rg = &fr.graph;
            let classes = periodicity::group_regions(rg);
            let plan = periodicity_plan::plan_collapse(rg, &classes);
            let sum = periodicity::summarize(&classes);
            let impl_report = class_impl::check_class_impls(&classes, &fr.region_subgraphs, sfuf);
            let factor = if plan.classes.is_empty() {
                0.0
            } else {
                rg.regions.len() as f64 / plan.classes.len() as f64
            };
            let consistent = impl_report.inconsistent_classes.is_empty();
            eprintln!(
                "    stencil · {regions:>4} regions → {classes:>3} classes ({factor:.1}× collapse) · control {ctrl_orig:>4} → {ctrl_new:>3} · max_period {max_period:>3} · class_impls_consistent={consistent}",
                regions = rg.regions.len(),
                classes = plan.classes.len(),
                factor = factor,
                ctrl_orig = rg.control.len(),
                ctrl_new = plan.control.len(),
                max_period = sum.max_period,
            );
            if !consistent {
                for (class_idx, picks) in &impl_report.inconsistent_classes {
                    eprintln!("      class {class_idx} has heterogeneous impls: {picks:?}",);
                }
            }
            // Δrepeat structure — verify the loop-carried deps
            // look affine (one delta per class pair) and count
            // intra- vs cross-iteration edges. Feeds the
            // collapsed-mode emitter's loop-carry decision.
            let deps = codegen::summarize_class_edges(&model_fuf, sfuf);
            eprintln!("    stencil-deps · {deps}");
        }

        let stub_items = emit_model_stub_items(&model_fuf, &sfufs, &loops);

        solved.push(SolvedModel {
            model,
            fuf: model_fuf,
            sfufs,
            loops,
            stub_items,
        });
    }

    // Cross-variant forward-fn dedup. Key each variant by its
    // (bounds, scalars, SFUF-per-workload-point) tuple and pick the
    // earliest-by-source_stem variant in each equivalence class as
    // the canonical. Non-canonical variants emit thin shims that
    // `pub use` the canonical's forward fns (one-line re-exports —
    // rustc doesn't re-monomorphize them, so LLVM optimization work
    // scales with `#distinct equivalence classes`, not
    // `#variants`).
    let canonical_for: std::collections::HashMap<usize, Ident> =
        compute_canonical_variants(&solved);

    // Env-gated viz JSON dump. No-op unless `FERRITE_VIZ_OUT` is set;
    // schema lives in `viz_dump`. Placed after `canonical_for` so the
    // dump can dedupe alias variants down to a `{name, canonical}`
    // stub (saves ~10× on Llama).
    {
        let mut hf: Vec<String> = models
            .iter()
            .flat_map(|m| m.architectures.iter().cloned())
            .collect();
        hf.sort();
        hf.dedup();
        let canonical_names: Vec<String> = (0..solved.len())
            .map(|i| canonical_for[&i].to_string())
            .collect();
        viz_dump::dump_now(
            &arch_name,
            &hf,
            &classified,
            &library,
            &carrier.block,
            solved.iter().map(|s| s.model),
            solved.iter().map(|s| &s.fuf),
            solved.iter().map(|s| &s.sfufs),
            solved.iter().map(|s| &s.loops),
            &canonical_names,
        );
    }

    let mut per_model_ts: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut arch_dispatch_arms: Vec<(Ident, Vec<u64>)> = Vec::new();

    for (idx, sm) in solved.iter().enumerate() {
        let model_mod = Ident::new(&sm.model.name, Span::call_site());
        let canonical_ident = &canonical_for[&idx];
        let is_canonical = *canonical_ident == model_mod;
        let canonical_override = if is_canonical {
            None
        } else {
            Some(canonical_ident.clone())
        };

        let codegen_items = codegen::emit_model(
            &classified,
            sm.model,
            &sm.fuf,
            &sm.sfufs,
            &sm.loops,
            &library,
            &manifest,
            canonical_override.as_ref(),
        );
        let stub_items = &sm.stub_items;
        per_model_ts.push(quote! {
            pub mod #model_mod {
                #stub_items
                #codegen_items
            }
        });

        arch_dispatch_arms.push((model_mod, collect_dispatch_bounds(sm.model)));
    }

    // Union of HF `architectures: [..]` strings across every compiled
    // model — the set of `arch_hint` values `ferrite_forward::try_load`
    // will route to this arch. Deduped + sorted for determinism.
    let mut hf_arches: Vec<String> = models
        .iter()
        .flat_map(|m| m.architectures.iter().cloned())
        .collect();
    hf_arches.sort();
    hf_arches.dedup();

    let arch_ident = Ident::new(&arch_name, carrier.sig.ident.span());
    let arch_dispatch_ts = emit_arch_dispatcher(&arch_ident, &hf_arches, &arch_dispatch_arms);

    // Emit items INLINE at the carrier's scope (no wrapping mod).
    // The carrier fn itself is consumed — it was only a host for
    // the DSL body + the arch ident. The file-module that contains
    // the #[forward] invocation becomes the public entry point:
    // if `ferrite-models/src/llama.rs` contains
    // `#[forward] fn llama() { ... }`, the caller accesses
    // `ferrite_models::llama::Weights` directly (no
    // `::arch::` / `::llama::` / etc.).
    Ok(quote! {
        // Rebuild-on-change for every JSON the macro read.
        #(#tracked)*

        #(#per_model_ts)*

        #arch_dispatch_ts
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
                     config.json or to config::derive_implicit_bounds",
                    model.source_stem,
                )
            })
        })
        .collect()
}

/// Arch-level dispatcher: an enum over every compiled variant plus
/// a `load` that auto-detects the right variant by walking each
/// variant's compile-emitted `fingerprint_matches(gw)` until one
/// claims the runtime `GpuWeights`. Accessor methods
/// (`num_hidden_layers`, …) delegate to per-variant constants baked
/// from each model's config.json.
///
/// Also emits the auto-registration with
/// [`ferrite_forward::try_load`]: an `impl FerriteWeights for Weights`
/// that routes the trait methods to the just-emitted accessors +
/// `forward` / `forward_backbone`, and an `inventory::submit!` block
/// carrying the union of HF `architectures` strings this arch
/// claims. The top-level loader walks the inventory at runtime; no
/// hand-written per-arch entry anywhere in the caller's codebase.
fn emit_arch_dispatcher(
    arch_ident: &Ident,
    hf_arches: &[String],
    arms: &[(Ident, Vec<u64>)],
) -> proc_macro2::TokenStream {
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

    // Auto-detect `load` body: try each variant's
    // `fingerprint_matches` in declaration order; first match wins.
    // Variants that share a fingerprint (e.g. two Llama configs that
    // agree on every bound AND quant flag — today only rope params
    // would differ) resolve to the earliest declared — a known
    // limitation; authors with that collision should drop one of
    // the colliding configs from `model_architectures/`.
    let try_fingerprint_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|(model_ident, _)| {
            let variant_ident = pascal_case(model_ident);
            quote! {
                if #model_ident::fingerprint_matches(gw, hf) {
                    return Ok(Some(Self::#variant_ident(
                        #model_ident::load(gw, stream, max_model_len)?,
                    )));
                }
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
    let forward_backbone_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|(model_ident, _)| {
            let variant_ident = pascal_case(model_ident);
            quote! {
                Weights::#variant_ident(w) => unsafe {
                    #model_ident::forward_backbone(w, ctx, device, num_tokens)
                },
            }
        })
        .collect();

    // Accessor methods on Weights — each returns a per-variant
    // constant from the matched model's bounds. Consumers (e.g.
    // vllm-executor's CudaModel enum) delegate their own accessor
    // arms to these, replacing N duplicated `m.model.layers[0].foo`
    // walks with a single method call.
    let accessor_methods: Vec<proc_macro2::TokenStream> = DISPATCH_FIELDS
        .iter()
        .map(|field| {
            let method_name = Ident::new(field, Span::call_site());
            let arms_ts: Vec<proc_macro2::TokenStream> = arms
                .iter()
                .map(|(model_ident, bounds)| {
                    let variant_ident = pascal_case(model_ident);
                    let idx = DISPATCH_FIELDS
                        .iter()
                        .position(|f| f == field)
                        .expect("field in DISPATCH_FIELDS");
                    let val = proc_macro2::Literal::u64_unsuffixed(bounds[idx]);
                    quote! { Weights::#variant_ident(_) => #val, }
                })
                .collect();
            quote! {
                #[doc = concat!(
                    "The matched variant's `", stringify!(#method_name),
                    "` — from the model's config.json at macro-expansion time."
                )]
                pub fn #method_name(&self) -> u64 {
                    match self {
                        #(#arms_ts)*
                    }
                }
            }
        })
        .collect();

    // Literals the auto-emitted `FerriteWeights` impl + inventory
    // registration reference.
    let arch_name_lit = proc_macro2::Literal::string(&arch_ident.to_string());
    let hf_arch_lits: Vec<proc_macro2::Literal> = hf_arches
        .iter()
        .map(|s| proc_macro2::Literal::string(s))
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
            #(#accessor_methods)*

            /// Auto-detect the compiled variant by sniffing the
            /// runtime `GpuWeights` against each variant's compile-
            /// baked fingerprint (embedding shape + last-layer
            /// tensor presence + quant suffix), then load.
            ///
            /// Returns `Ok(Some(..))` on a variant hit, `Ok(None)`
            /// when no compiled variant's fingerprint accepted the
            /// live `GpuWeights` (caller falls back to a hand-written
            /// path), or `Err(..)` only when a matched variant's
            /// `Weights::load` itself failed (I/O, shape mismatch
            /// inside a loader). A fingerprint miss is not an error —
            /// ferrite's job is to cover the storage formats it
            /// compiled for, not every format on disk.
            pub fn load(
                gw: &mut ::ferrite_cuda_core::weights::GpuWeights,
                stream: ::ferrite_cuda_core::CUstream,
                max_model_len: usize,
                hf: ::ferrite_forward::HfFingerprint<'_>,
            ) -> ::anyhow::Result<Option<Self>> {
                #(#try_fingerprint_arms)*
                Ok(None)
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

        /// Dispatching backbone-only forward (no lm_head). Returns
        /// `[num_tokens, hidden_size]` as an independently-owned
        /// `OwnedTensor`. For pipeline-parallel intermediate ranks
        /// that hand hidden states to the next rank.
        ///
        /// # Safety
        /// All tensors in `ctx` must be valid GPU memory; `device`
        /// must be the live CUDA device.
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward_backbone(
            w: &Weights,
            ctx: &::ferrite_forward::ForwardCtx,
            device: &mut ::ferrite_cuda_core::device::GpuDevice,
            num_tokens: u64,
        ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
            match w {
                #(#forward_backbone_arms)*
            }
        }

        // ── Auto-registration with ferrite_forward::try_load ──────
        //
        // `FerriteWeights` impl routes trait methods to the just-
        // emitted accessors + `forward` / `forward_backbone`.
        // `inventory::submit!` adds this arch to the global registry.
        // No hand-written central list anywhere.

        #[cfg(feature = "cuda")]
        impl ::ferrite_forward::FerriteWeights for Weights {
            fn arch_name(&self) -> &'static str { #arch_name_lit }
            fn num_hidden_layers(&self) -> u64 { self.num_hidden_layers() }
            fn hidden_size(&self) -> u64 { self.hidden_size() }
            fn intermediate_size(&self) -> u64 { self.intermediate_size() }
            fn num_attention_heads(&self) -> u64 { self.num_attention_heads() }
            fn num_key_value_heads(&self) -> u64 { self.num_key_value_heads() }
            fn head_dim(&self) -> u64 { self.head_dim() }
            fn vocab_size(&self) -> u64 { self.vocab_size() }

            unsafe fn forward(
                &self,
                ctx: &::ferrite_forward::ForwardCtx,
                device: &mut ::ferrite_cuda_core::device::GpuDevice,
                num_tokens: u64,
            ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
                unsafe { forward(self, ctx, device, num_tokens) }
            }

            unsafe fn forward_backbone(
                &self,
                ctx: &::ferrite_forward::ForwardCtx,
                device: &mut ::ferrite_cuda_core::device::GpuDevice,
                num_tokens: u64,
            ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
                unsafe { forward_backbone(self, ctx, device, num_tokens) }
            }
        }

        #[cfg(feature = "cuda")]
        ::ferrite_forward::inventory::submit! {
            ::ferrite_forward::FerriteArchRegistration {
                arch_name: #arch_name_lit,
                hf_arches: &[#(#hf_arch_lits),*],
                try_load: |gw, stream, max_model_len, hf| {
                    Weights::load(gw, stream, max_model_len, hf).map(|opt| {
                        opt.map(|w| ::std::boxed::Box::new(w)
                            as ::std::boxed::Box<dyn ::ferrite_forward::FerriteWeights>)
                    })
                },
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
    for (wp, sfuf) in &sfufs.per_workload {
        let loop_ir = loops
            .per_workload
            .get(wp)
            .expect("schedule_workloads populates every key");
        // Module name: `m_{m}` in the legacy 1-D sweep (sk_bucket=0),
        // `m_{m}_sk_{sk}` in a 2-D sweep. Integration tests can look
        // up either by name.
        let wl_mod = if wp.sk_bucket == 0 {
            Ident::new(&format!("m_{}", wp.num_tokens), Span::call_site())
        } else {
            Ident::new(
                &format!("m_{}_sk_{}", wp.num_tokens, wp.sk_bucket),
                Span::call_site(),
            )
        };
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
