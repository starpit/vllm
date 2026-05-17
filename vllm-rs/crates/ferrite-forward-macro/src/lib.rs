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

use std::path::Path;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Ident, ItemFn, LitInt, Token, parse_macro_input};

mod ast;
use ferrite_fusion_synth::atom;
use ferrite_fusion_synth::atom_lib;
mod cfg;
mod classified;
mod classify;
mod codegen;
mod concurrency;
mod config;
use ferrite_fusion_synth::fuse_pass;
mod cost;
mod emit;
mod fuf;
mod impl_lib;
mod interpreter_codegen;
#[cfg(feature = "metal")]
mod metal;
#[cfg(feature = "metal")]
mod metal_bridge;
mod parse;
mod quantization;
mod schedule;
mod shape;
mod solver;
mod solver_metal_tests;
mod target;
mod tp_lowering;
mod vision_lowering;
mod viz_dump;
mod weights_manifest;

mod vision_glue;

// ── Attribute argument parsing ────────────────────────────────────

struct ForwardArgs {
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
    /// Path to the per-arch CPU pixel-pack fn. Required for
    /// `#[vision_forward]`, ignored by `#[forward]`. The fn signature
    /// must match `fn(&VisionConfig, &[f32], u32, u32) -> (Vec<u16>,
    /// (u32, u32, u32))`. The macro-emitted `VisionArchWeights` impl
    /// forwards its `pixel_pack` associated fn to this path.
    pixel_pack: Option<syn::Path>,
    /// Path to a `pub const PROCESSOR: ferrite_vision::MmMetadata` in
    /// the per-arch crate declaring CPU-side host preprocessing
    /// metadata (placeholder token id key, size policy, tokens-per-
    /// image policy, preprocess fn). Required for `#[vision_forward]`,
    /// ignored by `#[forward]`. Baked into every emitted
    /// `FerriteMmRegistration` row. ferrite stays arch-agnostic — every
    /// arch-specific knob is data on the const, not a switch in ferrite.
    processor: Option<syn::Path>,
    /// Span used for error reporting when a required arg is
    /// missing.
    span: Span,
}

impl Parse for ForwardArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let span = input.span();
        let mut workloads: Option<Vec<u64>> = None;
        let mut sk_buckets: Option<Vec<u64>> = None;
        let mut pixel_pack: Option<syn::Path> = None;
        let mut processor: Option<syn::Path> = None;

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
                "workloads" => workloads = Some(parse_u64_list(input)?),
                "sk_buckets" => sk_buckets = Some(parse_u64_list(input)?),
                "pixel_pack" => pixel_pack = Some(input.parse::<syn::Path>()?),
                "processor" => processor = Some(input.parse::<syn::Path>()?),
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
            workloads,
            sk_buckets,
            pixel_pack,
            processor,
            span,
        })
    }
}

/// Discover the configs directory for `arch`. Standard layout:
/// the invoking crate IS `ferrite-model-<arch>`, so configs live
/// at `<MANIFEST_DIR>/configs/`. Fallback for #[forward] usages
/// outside a per-arch crate (e.g. integration tests in
/// `ferrite-forward/tests/`): walk up to the workspace root and
/// look in `crates/ferrite-model-<arch>/configs/`.
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

/// One stable string per QuantMethod *FieldLoad arm* the codegen
/// will emit. AWQ/GPTQ/CT collapse to `q:awq` / `q:gptq` because
/// they all share the `MarlinLinear` arm (the runtime
/// `marlin_storage` discriminator covers their on-disk split).
/// FP8 splits on `block_size`: per-tensor / per-channel goes to
/// `Fp8Linear` (1D scale), blockwise goes to `Fp8BlockLinear`
/// (2D scale) — different `load_with` bodies, so they cannot
/// share a canonical.
///
/// Threaded into `dedup_signature` so equivalence-class hashing
/// keeps FP8 block / std variants in separate canonicals. Without
/// this discriminator, qwen3's `fp8-block-128x128` and
/// `fp8-dynamic-per-tensor` hash identical (Impl picks match,
/// bounds match), `block` wins canonical alphabetically, the
/// `dynamic` shim calls `Fp8BlockLinear::load` on a 1D scale, and
/// the kernel panics during graph capture with `FP8 block scale
/// must be 2D, got 1D`.
fn dedup_quant_sig(method: Option<&crate::quantization::QuantMethod>) -> String {
    match method {
        None => "q:dense".to_string(),
        Some(crate::quantization::QuantMethod::Awq { .. }) => "q:awq".to_string(),
        Some(crate::quantization::QuantMethod::Gptq { .. }) => "q:gptq".to_string(),
        Some(crate::quantization::QuantMethod::Bnb4 { .. }) => "q:bnb4".to_string(),
        Some(crate::quantization::QuantMethod::Fp8 { block_size, .. }) => {
            if block_size.is_some() {
                "q:fp8-block".to_string()
            } else {
                "q:fp8-std".to_string()
            }
        }
        Some(crate::quantization::QuantMethod::Ggml) => "q:ggml".to_string(),
        Some(crate::quantization::QuantMethod::Affine { bits, group_size }) => {
            format!("q:affine-b{bits}-g{group_size}")
        }
    }
}

/// Truthy when `FERRITE_DEBUG` is set to anything non-empty other
/// than "0". Gates the noisier solver-internals output (per-phase
/// timings, per-model solve time) at proc-macro time.
///
/// Caveat: cargo doesn't track plain env vars across proc-macro
/// invocations (the rebuild-tracking variant `tracked_env::var` is
/// nightly-only), so flipping `FERRITE_DEBUG` between cached builds
/// won't re-run the macro on its own — touch a model source file or
/// run `cargo clean -p ferrite-model-<arch>` to force re-expansion.
pub(crate) fn ferrite_debug() -> bool {
    std::env::var("FERRITE_DEBUG")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
}

/// Stable per-tp-world-size discriminator threaded into
/// `dedup_signature` so the canonical-equivalence-class hash splits
/// the (variant × tp) fanout — a model compiled at tp=1 cannot share
/// a canonical with the same model compiled at tp=2 because the
/// emitted body sees different sharded `INTERMEDIATE_SIZE` /
/// `NUM_ATTENTION_HEADS` / `NUM_KEY_VALUE_HEADS` constants on
/// `<W as CanonicalParams>` and (for ShardDim1 weights) inserts
/// `Instruction::AllReduce` rows the tp=1 body lacks.
///
/// Compile-time set: `{1, 2, 4, 8}` (+16 behind the future
/// `tp-frontier` feature for NVL72-class models). Past 16 is rare
/// enough to keep behind a cargo feature gate.
fn dedup_tp_sig(tp_world_size: u8) -> String {
    format!("tp:{tp_world_size}")
}

/// Adaptive µs-to-string formatter for build-log scoring lines.
/// Used by the per-M score emission landed in the activation phase
/// (`889c44b2f`); kept as `#[allow(dead_code)]` for the foundation
/// commits that introduce it before the consumer lands during
/// the rebase replay.
#[allow(dead_code)]
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
    // Per-arch crates own their JSONs under `configs/` next to
    // `Cargo.toml` — fast path for the standard layout.
    let local = start.join("configs");
    if local.is_dir() {
        return Ok(local);
    }
    // Fallback for #[forward] invocations that live OUTSIDE a
    // `ferrite-model-<arch>` crate (e.g. integration tests in
    // `ferrite-forward/tests/`). Walk up to the workspace root and
    // look in `crates/ferrite-model-<arch>/configs/`. The `_` → `-`
    // conversion mirrors probe-weights' arch-name handling.
    let crate_dir_name = format!("ferrite-model-{}", arch.replace('_', "-"));
    let mut cur: Option<&std::path::Path> = Some(start);
    while let Some(d) = cur {
        let candidate = d.join("crates").join(&crate_dir_name).join("configs");
        if candidate.is_dir() {
            return Ok(candidate);
        }
        cur = d.parent();
    }
    Err(format!(
        "no `configs/` at {} and no `crates/{}/configs/` walking up to the workspace root",
        local.display(),
        crate_dir_name,
    ))
}

// ── Macro entry point ─────────────────────────────────────────────

#[proc_macro_attribute]
pub fn forward(args: TokenStream, item: TokenStream) -> TokenStream {
    // FERRITE_DUMP_SYNTH=<path> writes the MVP-synthesized pre-attn
    // chunk kernel for Llama-3.2-3B-4bit shape to the given path, then
    // proceeds with normal macro expansion. Lets us run `xcrun metal`
    // on the generated source without standing up the full solver
    // integration. Phase-2.5 verification hook — drops out once the
    // full fuse pass + lowering integration lands.
    if let Ok(path) = std::env::var("FERRITE_DUMP_SYNTH") {
        let kernel = fuse_pass::dump_llama_3_2_3b_4bit_pre_attn();
        let _ = std::fs::write(&path, &kernel.source);
    }

    let args = parse_macro_input!(args as ForwardArgs);
    let carrier = parse_macro_input!(item as ItemFn);

    match compile_common(&args, &carrier, CompileMode::DECODER) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Vision-encoder sibling of [`forward`]. Compiles a
/// `#[vision_forward]` body into the same FUF / solver / codegen
/// pipeline as the decoder macro, with three local overrides:
///   - the classifier sees the **vision prelude** (`pixels`,
///     `cu_seqlens`, `cos`, `sin`, `grid_thw`, `max_seqlen`) instead
///     of the decoder prelude.
///   - the per-(model, tp) fanout is pinned to `tp_world_size = 1`
///     because vision encoders run replicated in v1 (no AllReduce /
///     AllGather lowering).
///   - the multimodal post-Embed splice pass is skipped — it
///     belongs on the decoder side, not the encoder side.
///
/// Workload bucketing reuses the existing `workloads = [...]`
/// attribute slot; the decode-iter / sk-bucket axis names from
/// `#[forward]` map cleanly onto vision's batched-total flat L
/// (per the G.3 handoff resolution).
#[proc_macro_attribute]
pub fn vision_forward(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as ForwardArgs);
    let carrier = parse_macro_input!(item as ItemFn);

    match compile_common(&args, &carrier, CompileMode::VISION) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Per-macro overrides on the shared compile pipeline. Selected
/// once at the proc-macro entry; threaded through the prelude /
/// fanout / lowering passes so the body of [`compile_common`]
/// stays almost-uniform across the decoder and vision variants.
#[derive(Clone, Copy)]
struct CompileMode {
    /// Selects the DSL extern set used by [`classify::classify_with`].
    prelude: classified::Prelude,
    /// True for `#[forward]` (decoder), false for `#[vision_forward]`.
    /// Gates the row-parallel AllReduce + lm_head AllGather lowering
    /// passes: vision is replicated in v1 so they're skipped.
    apply_tp_lowering: bool,
    /// True for `#[forward]`, false for `#[vision_forward]`. Gates
    /// the post-Embed multimodal splice pass — that splice belongs
    /// on the decoder's text-side hidden states, not the encoder's
    /// patch hidden states. Only consulted under `cuda` (the splice
    /// pass is cuda-specific); declared cuda-only so non-cuda builds
    /// don't carry a dead field.
    #[cfg(feature = "cuda")]
    apply_mm_splice: bool,
    /// True for `#[forward]` (which fans out over `{1, 2, 4, 8}` at
    /// nccl-enabled), false for `#[vision_forward]` (always tp=1).
    enable_tp_fanout: bool,
    /// True for `#[forward]`, false for `#[vision_forward]`. Gates
    /// emission of the arch-level dispatcher (`enum Weights`,
    /// `FerriteArchRegistration` inventory submission, per-variant
    /// HF-bounds accessors). Vision encoders reach their compiled
    /// `Weights` via the hand-written `FerriteMmRegistration` in
    /// each VL crate's `vision.rs`; HF `architectures` strings like
    /// `Qwen2VLForConditionalGeneration` are claimed by the text-side
    /// `qwen2` arch, not the vision encoder. Skipping here also
    /// avoids the `collect_dispatch_bounds` panic — vision configs
    /// don't carry `num_hidden_layers` / `hidden_size` /
    /// `num_attention_heads` / `vocab_size`.
    emit_arch_dispatch: bool,
}

impl CompileMode {
    const DECODER: Self = Self {
        prelude: classified::Prelude::Decoder,
        apply_tp_lowering: true,
        #[cfg(feature = "cuda")]
        apply_mm_splice: true,
        enable_tp_fanout: true,
        emit_arch_dispatch: true,
    };
    const VISION: Self = Self {
        prelude: classified::Prelude::Vision,
        apply_tp_lowering: false,
        #[cfg(feature = "cuda")]
        apply_mm_splice: false,
        enable_tp_fanout: false,
        emit_arch_dispatch: false,
    };
}

fn compile_common(
    args: &ForwardArgs,
    carrier: &ItemFn,
    mode: CompileMode,
) -> syn::Result<proc_macro2::TokenStream> {
    // CARGO_MANIFEST_DIR at macro-expansion time is the invoking
    // crate's root. Target path resolves against it; the configs
    // directory is `<MANIFEST_DIR>/configs/` for per-arch crates,
    // with a workspace-walk fallback for tests outside model crates.
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

    // `pixel_pack = path::to::fn` is OPTIONAL under VISION mode.
    // When unset, the trait's default `pixel_pack` (which delegates
    // to `VisionConfig::patches_from_normalized_chw`, the spatial-
    // merge order Qwen2-VL / Qwen2.5-VL / any arch with the same
    // `patch_size · spatial_merge_size` convention share) is used.
    // Override only for arches with different patch ordering
    // (SigLIP raster, etc.). Decoder mode ignores the arg if set.

    // ── Front end: parse + classify ───────────────────────────────
    let ast = parse::parse_block(&carrier.block)
        .map_err(|e| syn::Error::new(args.span, format!("parse: {e}")))?;
    let mut classified = classify::classify_with(&ast, mode.prelude)
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
        // FERRITE_MODELS filter that excludes every model in this
        // arch is OK — emit an empty crate so cross-arch builds
        // (e.g. `FERRITE_MODELS=llama-3.2-3b cargo build -p
        // ferrite-models`) succeed for arches that don't match.
        // load_dir already warned to stderr with did-you-mean.
        if std::env::var_os("FERRITE_MODELS").is_some() {
            return Ok(quote! {});
        }
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

    // Vision-prelude programs route safetensors prefixes through
    // the per-arch layout. Pull from the representative model;
    // every variant of one vision arch shares the layout (only
    // `d_model` differs across variants — the layout itself is
    // arch-uniform). Decoder programs leave it `None`.
    if matches!(mode.prelude, classified::Prelude::Vision) {
        classified.vision_layout = models[0].vision_layout.clone();
    }
    // Decoder-side safetensors-prefix override (e.g. Gemma3-MM nests
    // the text decoder under `language_model.<...>`). Pulled from the
    // representative model — variants of one decoder arch share the
    // disk layout. Decoder programs that don't set the JSON field
    // (text-only, Qwen-style VL) leave it `None`.
    if matches!(mode.prelude, classified::Prelude::Decoder) {
        classified.decoder_safetensors_prefix = models[0].decoder_safetensors_prefix.clone();
    }

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

    // Backend selection via feature flags (compile-time, not runtime)
    #[cfg(feature = "cuda")]
    let target_profile = {
        let target_def = ferrite_cuda_targets::detect()
            .map_err(|e| syn::Error::new(carrier.sig.ident.span(), e))?;
        target::from_profile_def(target_def)
    };

    #[cfg(feature = "metal")]
    let target_profile = {
        use ferrite_metal_kernels::device::detect_device;
        let metal_device = detect_device().ok_or_else(|| {
            syn::Error::new(
                carrier.sig.ident.span(),
                "No Metal device detected. Metal backend requires macOS with Apple Silicon.",
            )
        })?;
        target::from_metal_profile(&metal_device.profile)
    };

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    compile_error!("ferrite-forward-macro requires either 'cuda' or 'metal' feature");

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
    // override paths. Target profile is no longer a separate file —
    // it's compiled into ferrite-cuda-targets, so cargo's normal
    // dep edge invalidates this crate when the profile changes.
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
        /// Tensor-parallel world size this model was solved at. The
        /// (variant × tp) fanout constructs one SolvedModel per
        /// (model, tp_world_size) pair. At `tp_world_size = 1` (every
        /// emission when `CARGO_FEATURE_NCCL` is unset) sharding is
        /// identity. Threaded into `dedup_signature` so the
        /// canonical-equivalence-class hash keeps each tp on its own
        /// canonical, and into the `tp_lowering::insert_all_reduces`
        /// call so the FUF receives a row-parallel AllReduce only
        /// when it should.
        tp_world_size: u8,
        /// Per-(model, tp) Rust-ident form of the emitted module. At
        /// tp=1 this is `model.name` verbatim (preserves the
        /// `ferrite_models::<arch>::<model>::Weights` path callers
        /// already use); at tp>1 it gets a `_tp{N}` suffix.
        mod_name: String,
        /// Per-(model, tp) HF-form stem used as the alphabetical
        /// canonical-selection key inside `compute_canonical_variants`.
        /// At tp=1 = `model.source_stem`; at tp>1 it's
        /// `format!("{}_tp{}", model.source_stem, tp_world_size)` so
        /// every member of a tp-N equivalence class shares the suffix
        /// and within-class ordering is preserved.
        canon_stem: String,
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
            parts.push(dedup_quant_sig(
                self.model.quantization.as_ref().map(|qc| &qc.method),
            ));
            // Tensor-parallel canonicalization axis. The
            // SolvedModel.tp_world_size field is the per-(model, tp)
            // discriminator; until task #7's outer-loop fanout lands
            // every SolvedModel has `tp_world_size = 1`, so every
            // dedup string still ends in `tp:1`. When the fanout
            // turns on, two SolvedModels of the same model at tp=1
            // vs tp=2 hash to different signatures and pick separate
            // canonicals — pinned by `tp_world_sizes_pick_separate_
            // canonicals`.
            parts.push(dedup_tp_sig(self.tp_world_size));
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
            &self.canon_stem
        }
        fn model_name(&self) -> &str {
            &self.mod_name
        }
    }

    // Compile-time tp set per project_tp_design_notes. The per-arch
    // crate's `nccl` cargo feature transitively enables
    // `ferrite-forward-macro/nccl`, recompiling THIS proc-macro with
    // its own `nccl` feature on. `cfg!(feature = "nccl")` then reads
    // `true` at expand time and the macro fans out every variant
    // over `{1, 2, 4, 8}`. Without it, only tp=1 emits —
    // byte-identical to the pre-fanout build. (Cargo caches the
    // proc-macro per-feature-set, so consumers without nccl still
    // get the fast tp=1-only macro.)
    let nccl_enabled = cfg!(feature = "nccl");
    // Vision macros opt out of the tp fanout entirely — the encoder
    // is replicated in v1 (no AllReduce / AllGather lowering, no
    // sharded weight surface). Decoder macros keep the existing
    // {1, 2, 4, 8} set under nccl-enabled, [1] otherwise.
    let tp_set: &[u8] = if mode.enable_tp_fanout && nccl_enabled {
        &[1, 2, 4, 8]
    } else {
        &[1]
    };

    let mut solved: Vec<SolvedModel<'_>> = Vec::with_capacity(models.len() * tp_set.len());

    for &tp_world_size in tp_set {
        for model in &models {
            // Skip (variant, tp) tuples whose column-parallel dims don't
            // divide evenly. KV replication when `num_kv_heads < tp_size`
            // is task #6's loader-sharding work — until then, an
            // indivisible KV head count drops the (variant, tp) tuple
            // from emission rather than baking a `NUM_KV_HEADS = 0`
            // canonical that would silently fail at runtime. Hits e.g.
            // SmolLM-135M (3 KV heads) at tp ∈ {2, 4, 8}, Llama 3.2-1B
            // (8 KV heads) at tp=16 (not in the default set).
            if tp_world_size > 1 {
                let na = *model.bounds.get("num_attention_heads").unwrap_or(&1);
                let nkv = *model.bounds.get("num_key_value_heads").unwrap_or(&1);
                let inter = *model.bounds.get("intermediate_size").unwrap_or(&1);
                let tp = tp_world_size as u64;
                if na % tp != 0 || nkv % tp != 0 || inter % tp != 0 {
                    eprintln!(
                        "  ferrite · {variant:<30} · skip tp={tp_world_size} (heads={na}/{nkv}, inter={inter} not divisible)",
                        variant = model.source_stem,
                    );
                    continue;
                }
            }

            // Per-(model, tp) ident form. tp=1 keeps the existing names
            // verbatim so callers' `ferrite_models::<arch>::<model>::Weights`
            // paths stay valid; tp>1 gets a `_tp{N}` suffix.
            let mod_name = if tp_world_size == 1 {
                model.name.clone()
            } else {
                format!("{}_tp{}", model.name, tp_world_size)
            };
            let canon_stem = if tp_world_size == 1 {
                model.source_stem.clone()
            } else {
                format!("{}_tp{}", model.source_stem, tp_world_size)
            };

            let model_cfg = cfg::build_cfg(&classified, model).map_err(|e| {
                syn::Error::new(args.span, format!("cfg [{}]: {e}", model.source_stem))
            })?;
            let mut model_fuf = fuf::unroll(&model_cfg, &inferred).map_err(|e| {
                syn::Error::new(args.span, format!("unroll [{}]: {e}", model.source_stem))
            })?;
            model_fuf.annotate_storage_formats(&classified, model);
            if std::env::var("FERRITE_GGUF_BUILD_TRACE").is_ok() {
                eprintln!("[ggml-build] === BEGIN model={} ===", model.source_stem);
            }
            // Tensor-parallel lowering pass. At tp=1 (every existing
            // SolvedModel until task #7's canonical fanout lands) this is
            // a strict no-op — the FUF flowing into the solver is
            // byte-identical to single-rank builds. Vision encoders
            // skip the pass entirely (no AllReduce/AllGather; replicated
            // in v1 per the G.3 handoff resolution).
            if mode.apply_tp_lowering {
                tp_lowering::insert_all_reduces(&mut model_fuf, &classified, tp_world_size);
                tp_lowering::insert_lm_head_allgather(&mut model_fuf, &classified, tp_world_size);
            }
            // Multimodal post-Embed splice. Unconditional at every tp
            // (including tp=1) — runtime no-op for text-only batches.
            // Must run AFTER `insert_all_reduces` so at tp>1 the
            // splice sits on the reduced embedding (not each rank's
            // partial masked-gather, which the pre-refactor inline
            // splice inside `Instruction::Embed::eval` mistakenly
            // overwrote). Vision encoders skip — splice belongs on
            // the decoder side, not the encoder side.
            // MmEmbedSplice's only matcher is the CUDA-only
            // `MmEmbedSpliceImpl` (D2D-copy via cuMemcpyDtoDAsync). Under
            // `--features metal` the impl pool can't claim the synthesized
            // splice node, so the solver explodes with "no Impl matched
            // tile … op MmEmbedSplice". Keep the splice insertion CUDA-
            // only until a Metal MmEmbedSplice lands. Text-only models
            // are unaffected at the runtime level — this splice is a
            // no-op there in both backends. Multimodal Metal will need
            // a Metal `MmEmbedSpliceImpl` and to flip this back on.
            #[cfg(feature = "cuda")]
            if mode.apply_mm_splice {
                tp_lowering::insert_mm_splices(&mut model_fuf, &classified);
            }
            // Vision-prelude `pixels` extern → tile materialization
            // (G.5.e.1). Synthesizes a single `OpKind::LoadPixels`
            // node and rewrites every downstream `FufInput::Extern`
            // referencing pixels to read its slot 0. No-op when the
            // body has no pixels reference. Vision-only — decoder
            // bodies have no `Pixels` extern (the prelude split
            // makes the two extern sets disjoint).
            if mode.prelude == classified::Prelude::Vision {
                vision_lowering::materialize_pixels(&mut model_fuf);
            }

            // At tp>1, the runtime weight tensors are per-rank shards
            // (column-parallel q/k/v/gate/up halve dim 0; row-parallel
            // o/down halve dim 1). The codegen-baked weight shapes
            // (`assert_weight_shape` checks them at every Gemm-class
            // eval) must match those per-rank shapes, so the solver
            // and `gemm_nk_from_fuf` (which evaluates symbolic
            // `Shape` against bounds) need a sharded view of
            // `num_attention_heads / num_key_value_heads /
            // intermediate_size`. tp=1 keeps the unsharded bounds
            // verbatim — byte-identical to the pre-fanout build.
            let solve_bounds = if tp_world_size > 1 {
                let mut b = model.bounds.clone();
                let tp = tp_world_size as u64;
                for k in [
                    "num_attention_heads",
                    "num_key_value_heads",
                    "intermediate_size",
                ] {
                    if let Some(v) = b.get_mut(k) {
                        *v = (*v / tp).max(1);
                    }
                }
                b
            } else {
                model.bounds.clone()
            };
            let t_solve = std::time::Instant::now();
            let mut sfufs = solver::solve_with_arch_filter(
                &model_fuf,
                &library,
                &target_profile,
                Some((&classified, model)),
                &inferred,
                &solve_bounds,
                &args.workloads,
                &args.sk_buckets,
            )
            .map_err(|e| {
                syn::Error::new(args.span, format!("solve [{}]: {e}", model.source_stem))
            })?;
            let d_solve = t_solve.elapsed();

            let loops = schedule::schedule_workloads(&model_fuf, &sfufs);
            cost::refresh_predicted_us(
                &model_fuf,
                &mut sfufs,
                &loops,
                &library,
                &target_profile,
                &solve_bounds,
            );

            let max_waves = loops
                .per_workload
                .values()
                .map(|l| l.num_waves())
                .max()
                .unwrap_or(0);
            // Kernel-class summary lifted from HEAD (`9189c3147`).
            // Classification must be TOTAL: any impl name that doesn't
            // map to a known class fails the build, prompting us to
            // add the kernel to the explicit table.
            //
            // Three semantic axes:
            //   - attention backend: fa2 / fi / mla
            //   - GEMM backend (pure or fused-with-GEMM): cublas /
            //     cutlass / marlin. fp8 folds into cutlass (uses
            //     `cutlass_scaled_mm_with_bias`); bnb4 folds into
            //     cublas (dequant + cuBLAS matmul).
            //   - non-gemm: kernels that are neither attention nor
            //     GEMM-bearing — element-wise, reshapes, residual
            //     adds, standalone norms. Surfaced because their
            //     existence is usually a "why didn't we fuse this?"
            //     signal.
            const CLASS_LABELS: [&str; 8] = [
                "fa2", "fi", "mla", "cublas", "cutlass", "marlin", "non-gemm", "comm",
            ];
            // Names that are non-gemm despite a `fused_` prefix
            // (norm-side fusions with no matmul).
            const NON_GEMM_NAMES: &[&str] = &[
                "embed_ref",
                "rmsnorm_ref",
                "add_ref",
                "reshape_ref",
                "rope_append_ref",
                "scalar_mul_inplace",
                "scalar_offset_rms_norm",
                "tanh_softcap_inplace",
                "softcap",
                "nosoftcap",
                "deepseek_moe_ref",
                "deepseek_moe_fp8_block",
                "deepseek_moe_ggml",
                "fused_moe_ref",
                "shared_fused_moe_ref",
                // Metal MoE Impls. Same "host-callback dispatch
                // wrapper, internal compute steps already classified
                // (Gemm via metal_gemm_, gather_qmv via
                // metal_affine_qmm_)" shape as the cuda *_ref
                // siblings — bucket them under non-gemm for
                // accounting.
                #[cfg(feature = "metal")]
                "metal_fused_moe",
                #[cfg(feature = "metal")]
                "metal_shared_fused_moe",
                "fused_add_rms_norm",
                "fused_add_rms_norm_with_offset",
                "mean_sub_rms_norm",
                "mean_sub_rms_norm_bias_add",
                // Vision-side unary elementwise ops (G.4). Shape-
                // preserving, no matmul — same class as the text-side
                // `scalar_mul_inplace` / `tanh_softcap_inplace` lines.
                "quick_gelu_inplace",
                "gelu_erf_inplace",
                "gelu_tanh_inplace",
                // Metal kernels (Phase 5.F: the proc-macro now runs
                // under `--features metal`, so the classifier sees
                // these names alongside the CUDA ones). Hand-rolled
                // norm / elementwise / fused-MLP / RoPE — same shape
                // class as the CUDA `*_ref` siblings, just emitting
                // MSL instead of CUDA. `metal_attention_*` and
                // `metal_gemm_*` get their own prefix arms below
                // (fa2 / cutlass-equivalent). Gated on `metal` so the
                // CUDA build doesn't carry dead names in its classifier.
                #[cfg(feature = "metal")]
                "metal_add_f16",
                #[cfg(feature = "metal")]
                "metal_add_bf16",
                #[cfg(feature = "metal")]
                "metal_embed_f16",
                #[cfg(feature = "metal")]
                "metal_embed_bf16",
                #[cfg(feature = "metal")]
                "metal_affine_embed_f16",
                #[cfg(feature = "metal")]
                "metal_affine_embed_bf16",
                #[cfg(feature = "metal")]
                "metal_reshape",
                #[cfg(feature = "metal")]
                "metal_bias_add_f16",
                #[cfg(feature = "metal")]
                "metal_bias_add_bf16",
                #[cfg(feature = "metal")]
                "metal_rmsnorm_f16",
                #[cfg(feature = "metal")]
                "metal_rmsnorm_bf16",
                #[cfg(feature = "metal")]
                "metal_fused_add_rmsnorm_f16",
                #[cfg(feature = "metal")]
                "metal_fused_add_rmsnorm_bf16",
                #[cfg(feature = "metal")]
                "metal_fused_gate_up_silu_mul_f16",
                #[cfg(feature = "metal")]
                "metal_fused_gate_up_silu_mul_bf16",
                #[cfg(feature = "metal")]
                "metal_fused_gate_up_gelu_mul_f16",
                #[cfg(feature = "metal")]
                "metal_fused_gate_up_gelu_mul_bf16",
                #[cfg(feature = "metal")]
                "metal_rope_append_f16",
                #[cfg(feature = "metal")]
                "metal_rope_append_bf16",
                // CommandR and other models use the interleaved rope
                // variant; same shape class as the regular rope_append.
                #[cfg(feature = "metal")]
                "metal_rope_append_interleaved_f16",
                #[cfg(feature = "metal")]
                "metal_rope_append_interleaved_bf16",
                #[cfg(feature = "metal")]
                "metal_fatrelu_f16",
                // Metal counterparts of the CUDA `scalar_mul_inplace`
                // and `tanh_softcap_inplace` non-gemm in-place
                // mutators. Same kernel class — bandwidth-bound
                // elementwise unary.
                #[cfg(feature = "metal")]
                "metal_scalar_mul_f16",
                #[cfg(feature = "metal")]
                "metal_scalar_mul_bf16",
                #[cfg(feature = "metal")]
                "metal_tanh_softcap_f16",
                #[cfg(feature = "metal")]
                "metal_tanh_softcap_bf16",
                // Vision-prelude pixels materialization (G.5.e.1).
                // Synthesized by `vision_lowering::materialize_pixels`;
                // emits a single D2D copy that wraps `ctx.fwd.pixels`
                // into a tile-table OwnedTensor. Not a compute kernel.
                "load_pixels",
                // Vision-side varlen attention + vision rope.
                // Shape-preserving non-gemm primitives.
                "varlen_attention",
                "vision_rope",
                // Row-permutation gather (G.6.4). Used by Qwen2.5-VL's
                // window-attention dispatch — same class as the other
                // memory-bound vision primitives.
                "embedding_gather",
                // 2-D average pool over the patch grid (G.7(b)). Used
                // by Gemma3-MM's SigLIP→text projector to reduce the
                // 64×64 patch grid down to 16×16 = 256 tokens. Memory-
                // bound with one thread per output cell; non-gemm.
                "avg_pool_2d",
                // Vision learned positional embedding lookup (G.7(c.1)).
                // Reuses the decoder's `embedding_gather_masked` kernel;
                // same memory-bound class as `embed_ref`.
                "pos_embed_ref",
            ];
            let mut classes_used = [false; 8];
            let mut unknown_names: std::collections::BTreeSet<&'static str> =
                std::collections::BTreeSet::new();
            for assignment in sfufs.per_workload.values() {
                for impl_id in assignment.impls.values() {
                    let name = library.get(*impl_id).name();
                    // Most kernel-name prefixes here are CUDA-specific
                    // (flashinfer/mla/cutlass/marlin/fp8/bnb4/ggml/cublas-via-fused_/
                    // NCCL collectives + the multimodal D2D splice).
                    // Gating each behind `cfg!(feature = "cuda")` keeps
                    // the metal-only build's classifier from carrying
                    // dead arms and prevents a hypothetical
                    // metal-emitted impl that happens to start with
                    // `cutlass` etc. from being silently mis-classed.
                    let bucket = if cfg!(feature = "cuda") && name.starts_with("flashinfer") {
                        Some(1) // fi
                    } else if name.starts_with("mla_") {
                        // MLA singletons (`mla_split_ref`, `mla_attention_ref`)
                        // and `DeepSeekMoeRefImpl`-family are registered under
                        // both backends — runtime support diverges, but the
                        // classifier just buckets by name for the build-time
                        // mix line.
                        Some(2) // mla
                    } else if name.starts_with("attention_")
                        || name.starts_with("sliding_attention_")
                        || name.starts_with("fa2_")
                        || name == "encoder_attention"
                        || (cfg!(feature = "metal") && name.starts_with("metal_attention_"))
                        || (cfg!(feature = "metal") && name.starts_with("metal_sliding_attention_"))
                    {
                        Some(0) // fa2
                    } else if cfg!(feature = "cuda") && name.starts_with("marlin") {
                        Some(5) // marlin
                    } else if cfg!(feature = "cuda") && name.starts_with("fp8") {
                        Some(4) // cutlass (fp8 uses cutlass_scaled_mm)
                    } else if cfg!(feature = "cuda")
                        && (name.starts_with("bnb4") || name.starts_with("ggml"))
                    {
                        // cublas: bnb4 dequant + cuBLAS matmul; ggml
                        // dequant_mul_mat_vec at decode + cuBLAS at prefill.
                        Some(3)
                    } else if (cfg!(feature = "cuda") && name.starts_with("cutlass"))
                        || (cfg!(feature = "metal") && name.starts_with("metal_gemm_"))
                        || (cfg!(feature = "metal") && name.starts_with("metal_affine_qmm_"))
                        || (cfg!(feature = "metal") && name.starts_with("metal_synth_"))
                    {
                        // Metal GEMM is currently routed through MPS
                        // matmul2d (see ferrite-metal-kernels::gemm);
                        // metal int4 GEMM routes through the
                        // qmv/qmm_t kernels (see ferrite-metal-kernels::
                        // quantized). Both treated as cutlass-equivalent
                        // for class accounting — same "specialized
                        // matmul tile" shape from the cost-model's
                        // perspective.
                        Some(4) // cutlass
                    } else if NON_GEMM_NAMES.contains(&name) {
                        Some(6) // non-gemm
                    } else if cfg!(feature = "cuda")
                        && (name == "all_reduce" || name == "all_gather")
                    {
                        // Tensor-parallel collectives inserted by
                        // `tp_lowering` at tp>1 (AllReduce after
                        // row-parallel gemms + vocab-parallel embed;
                        // AllGather after lm_head). Maps to NCCL —
                        // semantically distinct from compute kernels.
                        Some(7) // comm
                    } else if cfg!(feature = "cuda") && name == "mm_embed_splice" {
                        // Multimodal post-Embed D2D splice inserted by
                        // `tp_lowering::insert_mm_splices`. Not a
                        // compute kernel — runs a sequence of
                        // memcpy_dtod_async calls per image placeholder.
                        // Bucketed alongside the comm kernels since
                        // they share the "not a GEMM / not a normal
                        // per-token kernel" shape.
                        Some(7) // comm
                    } else if name.starts_with("fused_") || name == "gemm_ref" {
                        // `fused_gemm_bias` (qwen2 K/V) and the gemma
                        // fusion families (`fused_add_rms_norm`,
                        // `fused_add_rms_norm_with_offset`,
                        // `scalar_offset_rms_norm`) live in both backends
                        // now. cuda routes through cuBLAS gemm_bias; metal
                        // routes through its own GEMM path. Same
                        // build-time class for accounting.
                        Some(3) // cublas / cublas-equivalent
                    } else {
                        None
                    };
                    match bucket {
                        Some(b) => classes_used[b] = true,
                        None => {
                            unknown_names.insert(name);
                        }
                    }
                }
            }
            if !unknown_names.is_empty() {
                return Err(syn::Error::new(
                    args.span,
                    format!(
                        "[{}] kernel-class summary: no class assigned for impl name(s): {}. \
                         Add a class (or extend an existing prefix) in lib.rs.",
                        model.source_stem,
                        unknown_names.iter().copied().collect::<Vec<_>>().join(", "),
                    ),
                ));
            }
            let kernel_mix: String = classes_used
                .iter()
                .zip(CLASS_LABELS.iter())
                .filter(|(seen, _)| **seen)
                .map(|(_, label)| format!(" {label}"))
                .collect();
            // Per-M scoring: gated on FERRITE_DEBUG so the default
            // build log stays terse (one line per (variant, tp)).
            let per_m_part: String = if ferrite_debug() {
                let sk_axis_active = sfufs.per_workload.keys().any(|wp| wp.sk_bucket != 0);
                sfufs
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
                    .collect()
            } else {
                String::new()
            };
            let solve_ms_part = if ferrite_debug() {
                format!(" · {:>3} ms", d_solve.as_millis())
            } else {
                String::new()
            };
            eprintln!(
                "  ferrite · {variant:<30} · {tiles:>4} tiles · {waves:>3} waves{solve_ms_part} · tp={tp:<1} ·{kernel_mix}{per_m_part}",
                variant = canon_stem,
                tp = tp_world_size,
                tiles = model_fuf.len(),
                waves = max_waves,
            );

            let stub_items = emit_model_stub_items(&model_fuf, &sfufs, &loops);

            solved.push(SolvedModel {
                model,
                fuf: model_fuf,
                sfufs,
                loops,
                stub_items,
                tp_world_size,
                mod_name,
                canon_stem,
            });
        }
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
    let mut arch_dispatch_arms: Vec<DispatchArm> = Vec::new();

    for (idx, sm) in solved.iter().enumerate() {
        let model_mod = Ident::new(&sm.mod_name, Span::call_site());
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
            sm.tp_world_size,
            mode.emit_arch_dispatch,
        );
        let stub_items = &sm.stub_items;
        // Vision arch glue: per-variant `VisionArchWeights` impl,
        // `try_load_mm` with d_model fingerprint, inventory submits
        // for tp ∈ {1,2,4,8}. Decoder mode emits empty TokenStream.
        // `pixel_pack` is None for arches that use the trait's
        // default (delegating to `VisionConfig::patches_from_normalized_chw`).
        let vision_glue = if matches!(mode.prelude, classified::Prelude::Vision) {
            let processor = args.processor.as_ref().ok_or_else(|| {
                syn::Error::new(
                    args.span,
                    "#[vision_forward] missing required `processor = path::PROCESSOR` arg \
                     (path to a `pub const PROCESSOR: ferrite_vision::MmMetadata`)",
                )
            })?;
            vision_glue::emit_per_variant(
                sm.model,
                &arch_name,
                args.pixel_pack.as_ref(),
                &manifest.pad_to_mult8,
                processor,
            )
        } else {
            proc_macro2::TokenStream::new()
        };
        per_model_ts.push(quote! {
            pub mod #model_mod {
                #stub_items
                #codegen_items
                #vision_glue
            }
        });

        // Decoder fan-in to the arch-level dispatcher. Vision
        // encoders skip — see `CompileMode::emit_arch_dispatch`.
        // `collect_dispatch_bounds` panics on configs lacking the
        // decoder-only `DISPATCH_FIELDS` (vision configs carry
        // `vision_*` keys instead), so the call itself is gated.
        if mode.emit_arch_dispatch {
            arch_dispatch_arms.push(DispatchArm {
                model_ident: model_mod,
                source_stem: sm.model.source_stem.clone(),
                bounds: collect_dispatch_bounds(sm.model),
                tp_world_size: sm.tp_world_size,
            });
        }
    }

    // Union of HF `architectures: [..]` strings across every compiled
    // model — the set of safetensors `arch_hint` values
    // `ferrite_forward::try_load` will route to this arch. Deduped +
    // sorted for determinism.
    //
    // GGUF dispatch is on a SEPARATE field (`gguf_archs` below).
    // GGUFs report family-level tags (`"deepseek2"` covers V2 + V3-LoRA +
    // V3-flat; `"llama"` covers Llama + Mistral); the dispatcher tries
    // every claimant of that tag and `fingerprint_matches` discriminates.
    let mut hf_arches: Vec<String> = models
        .iter()
        .flat_map(|m| m.architectures.iter().cloned())
        .collect();
    hf_arches.sort();
    hf_arches.dedup();

    // Vision encoders intentionally skip the arch-dispatcher emission
    // (no `enum Weights`, no `FerriteArchRegistration` inventory). The
    // hand-written `FerriteMmRegistration` in each VL crate's
    // `vision.rs` claims its HF arch string and constructs the
    // multimodal forward over the macro-emitted per-variant
    // `Weights` types directly. `arch_dispatch_arms` is empty under
    // VISION mode, so this is also covered by `emit_arch_dispatcher`'s
    // empty-arms early-return — but the explicit skip here makes the
    // intent visible at the call site.
    let arch_dispatch_ts = if mode.emit_arch_dispatch {
        let arch_ident = Ident::new(&arch_name, carrier.sig.ident.span());
        emit_arch_dispatcher(
            &arch_ident,
            &hf_arches,
            &arch_dispatch_arms,
            &models_dir,
            carrier.sig.ident.span(),
        )?
    } else {
        proc_macro2::TokenStream::new()
    };

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

/// One row of the arch-dispatch table — a single (variant, tp)
/// tuple. The compile-loop builds one of these per (model, tp) pair
/// the outer-loop fanout produced; the dispatcher uses them to build
/// the unified `Weights` enum, the per-tp `inventory::submit!` blocks,
/// and the per-variant `BackboneDumpRegistration` rows for
/// `vllm ferrite info`.
struct DispatchArm {
    model_ident: Ident,
    /// Source-config stem (e.g. `"qwen2.5-3b"`). Same for every tp
    /// of the same model — used as the variant_stem label in
    /// `BackboneDumpRegistration::dump_all` so `vllm ferrite info`
    /// can print human-readable model names regardless of the
    /// `_tp{N}`-suffixed module path.
    source_stem: String,
    /// Values aligned with [`DISPATCH_FIELDS`] — read by the
    /// per-arch accessor methods on the dispatcher's `Weights`
    /// enum. Same for every tp of the same model (the FerriteWeights
    /// trait surface is the un-sharded model config — sharded values
    /// flow through `<W as CanonicalParams>` instead).
    bounds: Vec<u64>,
    tp_world_size: u8,
}

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
    arms: &[DispatchArm],
    models_dir: &Path,
    error_span: Span,
) -> syn::Result<proc_macro2::TokenStream> {
    if arms.is_empty() {
        return Ok(quote! {});
    }

    // Variant ident = PascalCase of the model ident (e.g.
    // `llama_3_2_1b` → `Llama_3_2_1b`, `llama_3_2_1b_tp2` →
    // `Llama_3_2_1b_Tp2`). Keep the underscores — they carry meaning
    // (dotted-version components, tp suffix) and collapsing them
    // would create ambiguity between e.g. `llama32` and `llama_3_2`.
    let variants: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! { #variant_ident(#model_ident::Weights) }
        })
        .collect();

    // Distinct tp values across all arms, in ascending order. One
    // `inventory::submit!` per value; one match arm in `Weights::load`
    // per value. At nccl-disabled this is just `[1]` and the dispatcher
    // is byte-identical to the pre-fanout build.
    let mut tp_values: Vec<u8> = arms.iter().map(|a| a.tp_world_size).collect();
    tp_values.sort_unstable();
    tp_values.dedup();

    // Per-tp `Weights::load` match arms. For each tp value, walk only
    // the arms whose `tp_world_size` matches — the closure inside
    // `inventory::submit!` passes its own tp constant, so the runtime
    // never iterates wrong-tp variants. Variants that share a
    // fingerprint within a tp group resolve to the earliest declared.
    let load_match_arms: Vec<proc_macro2::TokenStream> = tp_values
        .iter()
        .map(|tp| {
            let tp_lit = proc_macro2::Literal::u8_unsuffixed(*tp);
            let arms_for_tp: Vec<proc_macro2::TokenStream> = arms
                .iter()
                .filter(|a| a.tp_world_size == *tp)
                .map(|a| {
                    let variant_ident = pascal_case(&a.model_ident);
                    let model_ident = &a.model_ident;
                    quote! {
                        if #model_ident::fingerprint_matches(gw, hf) {
                            return Ok(Some(Self::#variant_ident(
                                #model_ident::load(gw, stream, max_model_len, tp_rank)?,
                            )));
                        }
                    }
                })
                .collect();
            quote! {
                #tp_lit => {
                    #(#arms_for_tp)*
                }
            }
        })
        .collect();

    let forward_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(w) => unsafe {
                    #model_ident::forward(w, ctx, device, num_tokens)
                },
            }
        })
        .collect();
    let forward_backbone_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(w) => unsafe {
                    #model_ident::forward_backbone(w, ctx, device, num_tokens)
                },
            }
        })
        .collect();

    // Per-variant dispatch arms for `forward_with_metal_followup`.
    let forward_with_followup_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(w) => unsafe {
                    #model_ident::forward_with_metal_followup(w, ctx, device, num_tokens, followup)
                },
            }
        })
        .collect();

    // Per-variant `METAL_ARENA_PEAK_BYTES` reads. Each canonical mod
    // emits this const from the macro's per-canonical metal_emission;
    // shim variants re-export the canonical's. The trait impl below
    // dispatches on `Weights` variant and returns the matched module's
    // const so `determine_available_memory` reads the right per-arch
    // peak rather than a 512 MiB placeholder.
    let metal_arena_peak_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(_) => #model_ident::METAL_ARENA_PEAK_BYTES,
            }
        })
        .collect();

    // Accessor methods on Weights — each returns a per-variant
    // constant from the matched model's bounds. Consumers (e.g.
    // vllm-executor's CudaModel enum) delegate their own accessor
    // arms to these, replacing N duplicated `m.model.layers[0].foo`
    // walks with a single method call. Trait surface is un-sharded
    // — kernel launches read sharded values via `<W as
    // CanonicalParams>::…` instead, so every (variant, tp) pair of
    // the same model returns the same num_attention_heads / etc.
    let accessor_methods: Vec<proc_macro2::TokenStream> = DISPATCH_FIELDS
        .iter()
        .map(|field| {
            let method_name = Ident::new(field, Span::call_site());
            let arms_ts: Vec<proc_macro2::TokenStream> = arms
                .iter()
                .map(|a| {
                    let variant_ident = pascal_case(&a.model_ident);
                    let idx = DISPATCH_FIELDS
                        .iter()
                        .position(|f| f == field)
                        .expect("field in DISPATCH_FIELDS");
                    let val = proc_macro2::Literal::u64_unsuffixed(a.bounds[idx]);
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

    // GGUF spec (per-arch). Loaded once: drives both
    // (a) the `gguf_archs` field of `FerriteArchRegistration` (every arch
    //     with a `"ggml"` entry claims its gguf tag for dispatch), and
    // (b) the optional `ferrite_gguf::register!` emission below (only
    //     the canonical owner per gguf_arch tag — `register_spec=false`
    //     opts out for non-canonical claimants like Mistral or
    //     deepseek-v3-flat).
    let gguf_spec = config::load_gguf_spec(models_dir).map_err(|e| {
        syn::Error::new(
            error_span,
            format!(
                "quantizations.json gguf spec in {}: {e}",
                models_dir.display()
            ),
        )
    })?;
    let gguf_arch_lits: Vec<proc_macro2::Literal> = match &gguf_spec {
        Some(spec) => vec![proc_macro2::Literal::string(
            spec.gguf_arch
                .clone()
                .unwrap_or_else(|| arch_ident.to_string())
                .as_str(),
        )],
        None => vec![],
    };

    // Per-variant `(stem, dump_fn)` rows for the backbone-dump
    // registry. Each variant's `mod <model_ident>` emits a
    // `pub fn dump() -> Vec<BucketDump>`; here we name them so a
    // single `BackboneDumpRegistration` per arch can iterate them.
    // Iterates over EVERY (model, tp) tuple — `vllm ferrite info`
    // shows each tp variant separately (the per-tp module's `dump()`
    // reflects the sharded canonical's bucket layout).
    let dump_rows: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let model_ident = &a.model_ident;
            let stem_lit = proc_macro2::Literal::string(&a.source_stem);
            let tp_lit = proc_macro2::Literal::u8_unsuffixed(a.tp_world_size);
            quote! {
                ::ferrite_forward::VariantDump {
                    variant_stem: #stem_lit,
                    tp_world_size: #tp_lit,
                    buckets: #model_ident::dump(),
                }
            }
        })
        .collect();

    // Per-tp inventory submission. Each closure passes its own tp
    // constant into `Weights::load(...)`; the matching arm there
    // walks only the tp-N variants. The submit! block is a static
    // initializer — N submissions at compile time, walked once at
    // runtime by `ferrite_forward::try_load(arch_hint, tp_world_size)`.
    let inventory_submits: Vec<proc_macro2::TokenStream> = tp_values
        .iter()
        .map(|tp| {
            let tp_lit = proc_macro2::Literal::u8_unsuffixed(*tp);
            quote! {
                #[cfg(any(feature = "cuda", feature = "metal"))]
                ::ferrite_forward::inventory::submit! {
                    ::ferrite_forward::FerriteArchRegistration {
                        arch_name: #arch_name_lit,
                        hf_arches: &[#(#hf_arch_lits),*],
                        gguf_archs: &[#(#gguf_arch_lits),*],
                        tp_world_size: #tp_lit,
                        try_load: |gw, stream, max_model_len, tp_rank, hf| {
                            Weights::load(
                                gw, stream, max_model_len, hf, #tp_lit, tp_rank,
                            ).map(|opt| {
                                opt.map(|w| ::std::boxed::Box::new(w)
                                    as ::std::boxed::Box<dyn ::ferrite_forward::FerriteWeights>)
                            })
                        },
                    }
                }
            }
        })
        .collect();

    // GGUF spec registration. Conditional on (a) the arch having a
    // `"ggml"` entry in `quantizations.json` AND (b) `register_spec`
    // being true (default). Non-canonical owners of a gguf tag set
    // `register_spec: false` so only one crate per gguf_arch supplies
    // the inventory-side spec data — find_spec stays deterministic.
    // The arch is still routed for that tag via the `gguf_archs`
    // field of its `FerriteArchRegistration` above.
    let gguf_register_emit: proc_macro2::TokenStream = match gguf_spec {
        None => quote! {},
        Some(ref spec) if !spec.register_spec => quote! {},
        Some(spec) => {
            let gguf_arch_lit = proc_macro2::Literal::string(
                &spec.gguf_arch.unwrap_or_else(|| arch_ident.to_string()),
            );
            let qk = spec.qk_permute;
            let rope = spec.llama3_rope_scaling_inference;
            let nwo = proc_macro2::Literal::f32_suffixed(spec.norm_weight_offset);
            let renames: Vec<proc_macro2::TokenStream> = spec
                .tensor_renames
                .iter()
                .map(|(g, h)| {
                    let g = proc_macro2::Literal::string(g);
                    let h = proc_macro2::Literal::string(h);
                    quote! { (#g, #h) }
                })
                .collect();
            let m_u32: Vec<proc_macro2::TokenStream> = spec
                .metadata_u32
                .iter()
                .map(|(g, e)| {
                    let g = proc_macro2::Literal::string(g);
                    let e = proc_macro2::Literal::string(e);
                    quote! { (#g, #e) }
                })
                .collect();
            let m_f32: Vec<proc_macro2::TokenStream> = spec
                .metadata_f32
                .iter()
                .map(|(g, e)| {
                    let g = proc_macro2::Literal::string(g);
                    let e = proc_macro2::Literal::string(e);
                    quote! { (#g, #e) }
                })
                .collect();
            let d_u32: Vec<proc_macro2::TokenStream> = spec
                .metadata_defaults_u32
                .iter()
                .map(|(k, v)| {
                    let k = proc_macro2::Literal::string(k);
                    let v = proc_macro2::Literal::u32_suffixed(*v);
                    quote! { (#k, #v) }
                })
                .collect();
            let d_f32: Vec<proc_macro2::TokenStream> = spec
                .metadata_defaults_f32
                .iter()
                .map(|(k, v)| {
                    let k = proc_macro2::Literal::string(k);
                    let v = proc_macro2::Literal::f32_suffixed(*v);
                    quote! { (#k, #v) }
                })
                .collect();
            quote! {
                #[cfg(feature = "cuda")]
                ::ferrite_gguf::register! {
                    gguf_arch = #gguf_arch_lit,
                    qk_permute = #qk,
                    tensor_renames = [ #(#renames),* ],
                    metadata_u32 = [ #(#m_u32),* ],
                    metadata_f32 = [ #(#m_f32),* ],
                    metadata_defaults_u32 = [ #(#d_u32),* ],
                    metadata_defaults_f32 = [ #(#d_f32),* ],
                    llama3_rope_scaling_inference = #rope,
                    norm_weight_offset = #nwo,
                }
            }
        }
    };

    Ok(quote! {
        /// One variant per compiled model config. Holds that
        /// model's specialized `Weights`. Same shape under both
        /// backends — the variant's per-canonical `Weights` struct
        /// itself is cfg-mutex'd internally (cuda fields gated
        /// `cfg(feature = "cuda")`, metal fields gated
        /// `cfg(feature = "metal")`).
        #[cfg(any(feature = "cuda", feature = "metal"))]
        pub enum Weights {
            #(#variants),*
        }

        #[cfg(any(feature = "cuda", feature = "metal"))]
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
                tp_world_size: u8,
                tp_rank: u8,
            ) -> ::anyhow::Result<Option<Self>> {
                match tp_world_size {
                    #(#load_match_arms)*
                    _ => {}
                }
                Ok(None)
            }
        }

        /// Dispatching forward. Matches the `Weights` variant and
        /// calls the per-model specialized `forward`. Same body and
        /// signature under both backends — the cfg-mutex'd
        /// `GpuDevice` and `OwnedTensor` re-exports resolve to the
        /// matching backend's struct, and per-canonical `forward`
        /// fns now exist in both `cfg(cuda)` and `cfg(metal)` arms.
        ///
        /// # Safety
        /// All tensors in `ctx` must be valid GPU memory; `device`
        /// must be the live backend device.
        #[cfg(any(feature = "cuda", feature = "metal"))]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward(
            w: &Weights,
            ctx: &::ferrite_forward::ForwardCtx,
            device: &mut ::ferrite_cuda_core::GpuDevice,
            num_tokens: u64,
        ) -> ::ferrite_cuda_core::OwnedTensor {
            match w {
                #(#forward_arms)*
            }
        }

        /// Same as [`forward`] but with an MTL4 encoder-tail hook.
        /// See `MetalForwardFollowup` for semantics.
        ///
        /// # Safety
        /// Same as [`forward`].
        #[cfg(feature = "metal")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward_with_metal_followup(
            w: &Weights,
            ctx: &::ferrite_forward::ForwardCtx,
            device: &mut ::ferrite_cuda_core::GpuDevice,
            num_tokens: u64,
            followup: ::core::option::Option<::ferrite_forward::MetalForwardFollowup<'_>>,
        ) -> ::ferrite_cuda_core::OwnedTensor {
            match w {
                #(#forward_with_followup_arms)*
            }
        }

        /// Dispatching backbone-only forward (no lm_head). Returns
        /// `[num_tokens, hidden_size]` as an independently-owned
        /// `OwnedTensor`. For pipeline-parallel intermediate ranks
        /// that hand hidden states to the next rank — cuda-only
        /// today; metal has no PP fanout, so the per-canonical
        /// `forward_backbone` is cfg(cuda)-gated and this dispatcher
        /// matches.
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

        #[cfg(any(feature = "cuda", feature = "metal"))]
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
                device: &mut ::ferrite_cuda_core::GpuDevice,
                num_tokens: u64,
            ) -> ::ferrite_cuda_core::OwnedTensor {
                unsafe { forward(self, ctx, device, num_tokens) }
            }

            unsafe fn forward_backbone(
                &self,
                ctx: &::ferrite_forward::ForwardCtx,
                device: &mut ::ferrite_cuda_core::GpuDevice,
                num_tokens: u64,
            ) -> ::ferrite_cuda_core::OwnedTensor {
                #[cfg(feature = "cuda")]
                { unsafe { forward_backbone(self, ctx, device, num_tokens) } }
                #[cfg(feature = "metal")]
                {
                    let _ = (ctx, device, num_tokens);
                    unimplemented!(
                        "metal forward_backbone — pipeline-parallel intermediate \
                         ranks aren't supported on metal yet (no PP fanout)"
                    )
                }
            }

            #[cfg(feature = "metal")]
            fn metal_arena_peak_bytes(&self) -> u64 {
                match self {
                    #(#metal_arena_peak_arms)*
                }
            }

            #[cfg(feature = "metal")]
            unsafe fn forward_with_metal_followup(
                &self,
                ctx: &::ferrite_forward::ForwardCtx,
                device: &mut ::ferrite_cuda_core::GpuDevice,
                num_tokens: u64,
                followup: ::core::option::Option<::ferrite_forward::MetalForwardFollowup<'_>>,
            ) -> ::ferrite_cuda_core::OwnedTensor {
                unsafe { forward_with_metal_followup(self, ctx, device, num_tokens, followup) }
            }
        }

        // One `FerriteArchRegistration` per distinct tp value across
        // the compiled (variant, tp) tuples. Each closure passes its
        // own tp constant into `Weights::load(...)` so the matching
        // arm there walks only the tp-N variants. At nccl-disabled
        // this expands to a single submission with `tp_world_size:
        // 1u8` — byte-identical to the pre-fanout build.
        #(#inventory_submits)*

        // One `BackboneDumpRegistration` per arch, fanning out over
        // every (model, tp) variant via `dump_rows`. Independent of
        // the per-tp `FerriteArchRegistration` above — `vllm ferrite
        // info` walks this registry separately, with no runtime GPU
        // or weight loading. Gated on either backend feature so the
        // metal CLI sees compiled-in metal arches too.
        #[cfg(any(feature = "cuda", feature = "metal"))]
        ::ferrite_forward::inventory::submit! {
            ::ferrite_forward::BackboneDumpRegistration {
                arch_name: #arch_name_lit,
                dump_all: || vec![ #(#dump_rows),* ],
            }
        }

        // GGUF arch registration: empty when the arch's
        // `quantizations.json` has no `"ggml"` entry; otherwise one
        // `ferrite_gguf::register! { ... }` block. All per-arch GGUF
        // data flows through `quantizations.json`.
        #gguf_register_emit
    })
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

/// Emit `pub const NUM_TILES: usize = …` per canonical. The only
/// observability surface that catches a real regression — if the
/// FUF lowering or fusion logic changes the tile count, the test
/// `assert_eq!(llama_3_2_1b::NUM_TILES, 243)` fires.
///
/// Per-bucket `m_X[_sk_Y]::{NUM_SUBGRAPHS, NUM_WAVES, PREDICTED_US}`
/// modules used to live here too — they were brittle (PREDICTED_US
/// drifts every time `target_profiles/*.csv` is regenerated; the
/// other two were probed without an assertion). The test invariant
/// they really cared about (prefill cost ≫ decode cost) lives in
/// `solver::tests` now, calling `solve()` directly. ~9k lines off
/// cargo expand workspace-wide.
fn emit_model_stub_items(
    fuf: &fuf::Fuf,
    _sfufs: &solver::WorkloadAssignments,
    _loops: &schedule::WorkloadLoops,
) -> proc_macro2::TokenStream {
    let num_tiles = fuf.len();
    quote! {
        pub const NUM_TILES: usize = #num_tiles;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::{BnbQuantType, Fp8ActivationScheme, GptqLayout, QuantMethod};

    /// `dedup_quant_sig` must keep FP8 block and per-tensor / per-
    /// channel variants on separate canonicals — the load_with body
    /// emits `Fp8BlockLinear::load` for one and `Fp8Linear::load` for
    /// the other, and the runtime kernels can't share a path
    /// (`Fp8BlockLinear::load` panics on a 1D scale tensor).
    #[test]
    fn dedup_quant_sig_separates_fp8_block_from_fp8_std() {
        let block = QuantMethod::Fp8 {
            scheme: Fp8ActivationScheme::Dynamic,
            block_size: Some([128, 128]),
        };
        let std_dyn = QuantMethod::Fp8 {
            scheme: Fp8ActivationScheme::Dynamic,
            block_size: None,
        };
        let std_static = QuantMethod::Fp8 {
            scheme: Fp8ActivationScheme::Static,
            block_size: None,
        };
        // Block vs std must differ — that's the load-arm split.
        assert_ne!(
            dedup_quant_sig(Some(&block)),
            dedup_quant_sig(Some(&std_dyn))
        );
        // Within `std`, dynamic vs static can share a canonical:
        // both go through `Fp8Linear::load`, the activation_scheme
        // is a runtime Impl detail (Fp8GemmImpl handles both).
        assert_eq!(
            dedup_quant_sig(Some(&std_dyn)),
            dedup_quant_sig(Some(&std_static))
        );
    }

    /// AWQ and GPTQ all collapse to one Marlin discriminator —
    /// AWQ/GPTQ/CT variants of the same dense base SHOULD share
    /// a canonical (the runtime `marlin_storage` param threads in
    /// the on-disk format). Compressed-tensors INT4 routes through
    /// `QuantMethod::Gptq`, so it lands on `q:gptq` alongside
    /// AutoGPTQ.
    #[test]
    fn dedup_quant_sig_collapses_awq_and_gptq_internally() {
        let awq_a = QuantMethod::Awq {
            bits: 4,
            group_size: 128,
            zero_point: true,
            version: crate::quantization::AwqVersion::Gemm,
        };
        let awq_b = QuantMethod::Awq {
            bits: 4,
            group_size: 64,
            zero_point: true,
            version: crate::quantization::AwqVersion::Gemm,
        };
        // Same arm, regardless of group_size.
        assert_eq!(dedup_quant_sig(Some(&awq_a)), dedup_quant_sig(Some(&awq_b)));
        let gptq = QuantMethod::Gptq {
            bits: 4,
            group_size: 128,
            desc_act: true,
            sym: true,
            layout: GptqLayout::Qweight,
        };
        // AWQ and GPTQ must NOT collapse — different MarlinFormat
        // discriminator threaded at load time, but more importantly
        // their `marlin_storage` literal differs in the prelude.
        assert_ne!(dedup_quant_sig(Some(&awq_a)), dedup_quant_sig(Some(&gptq)));
    }

    /// `dedup_tp_sig` must give different strings for different
    /// tp_world_size values so the (variant × tp) fanout's canonical
    /// hash separates them. Mirrors
    /// `dedup_quant_sig_separates_fp8_block_from_fp8_std`'s shape.
    #[test]
    fn dedup_tp_sig_separates_each_compile_time_tp() {
        // Compile-time set per project_tp_design_notes is {1,2,4,8}.
        let sigs: Vec<String> = [1u8, 2, 4, 8].iter().map(|t| dedup_tp_sig(*t)).collect();
        for i in 0..sigs.len() {
            for j in (i + 1)..sigs.len() {
                assert_ne!(
                    sigs[i],
                    sigs[j],
                    "tp={} and tp={} must hash to different signatures",
                    [1, 2, 4, 8][i],
                    [1, 2, 4, 8][j],
                );
            }
        }
    }

    /// `dedup_tp_sig(n)` is deterministic — same input → same output
    /// across calls. Cargo's incremental cache hashes the dedup
    /// signature, so non-determinism would force spurious rebuilds.
    #[test]
    fn dedup_tp_sig_is_deterministic() {
        for tp in [1u8, 2, 4, 8, 16] {
            assert_eq!(dedup_tp_sig(tp), dedup_tp_sig(tp));
        }
    }

    /// `dedup_tp_sig` format is `tp:<n>` — pinning this so the
    /// signature stays human-readable in cargo expand and golden
    /// diffs, matching the `q:`/`b:`/`r:`/`w:` prefixes already
    /// used by the other dedup parts.
    #[test]
    fn dedup_tp_sig_format() {
        assert_eq!(dedup_tp_sig(1), "tp:1");
        assert_eq!(dedup_tp_sig(8), "tp:8");
    }

    /// BNB4 has its own FieldLoad arm (`Bnb4bitLinear::load`),
    /// distinct from FP8 / Marlin / Dense.
    #[test]
    fn dedup_quant_sig_keeps_bnb4_separate() {
        let bnb = QuantMethod::Bnb4 {
            quant_type: BnbQuantType::NF4,
            blocksize: 64,
        };
        let dense = None;
        assert_ne!(dedup_quant_sig(Some(&bnb)), dedup_quant_sig(dense));
    }

    /// `compute_canonical_variants` end-to-end on the (variant × tp)
    /// fanout: two SolvedModels of the same variant compiled at
    /// different `tp_world_size` values must NOT collapse to one
    /// canonical. The dedup string differs only in the `tp:N` part,
    /// which is enough to keep them in separate equivalence classes.
    /// This is the regression for task #7's outer-loop fanout — a
    /// future change that drops `dedup_tp_sig(self.tp_world_size)`
    /// from the dedup string would fail this test.
    #[test]
    fn tp_world_sizes_pick_separate_canonicals() {
        struct Fake {
            sig: String,
            stem: String,
            name: String,
        }
        impl HasSolvedSig for Fake {
            fn dedup_signature(&self) -> String {
                self.sig.clone()
            }
            fn source_stem(&self) -> &str {
                &self.stem
            }
            fn model_name(&self) -> &str {
                &self.name
            }
        }
        // Same variant compiled at tp=1, 2, 4, 8 — every other
        // dedup part identical, only `tp:N` differs.
        let solved = vec![
            Fake {
                sig: format!("x|{}", dedup_tp_sig(1)),
                stem: "command-r-1l".into(),
                name: "command_r_1l".into(),
            },
            Fake {
                sig: format!("x|{}", dedup_tp_sig(2)),
                stem: "command-r-1l_tp2".into(),
                name: "command_r_1l_tp2".into(),
            },
            Fake {
                sig: format!("x|{}", dedup_tp_sig(4)),
                stem: "command-r-1l_tp4".into(),
                name: "command_r_1l_tp4".into(),
            },
            Fake {
                sig: format!("x|{}", dedup_tp_sig(8)),
                stem: "command-r-1l_tp8".into(),
                name: "command_r_1l_tp8".into(),
            },
        ];
        let map = compute_canonical_variants(&solved);
        // Each tp picks its own variant as canonical — no collapse.
        assert_eq!(map[&0].to_string(), "command_r_1l");
        assert_eq!(map[&1].to_string(), "command_r_1l_tp2");
        assert_eq!(map[&2].to_string(), "command_r_1l_tp4");
        assert_eq!(map[&3].to_string(), "command_r_1l_tp8");
    }

    /// `compute_canonical_variants` end-to-end: with the q-sig
    /// discriminator in `dedup_signature`, FP8 block stays its own
    /// canonical while dynamic+static fold together (alphabetically
    /// `dynamic` wins). This is the regression test for the
    /// qwen3-fp8-dynamic graph-capture panic.
    #[test]
    fn fp8_block_and_std_pick_separate_canonicals() {
        struct Fake {
            sig: String,
            stem: String,
            name: String,
        }
        impl HasSolvedSig for Fake {
            fn dedup_signature(&self) -> String {
                self.sig.clone()
            }
            fn source_stem(&self) -> &str {
                &self.stem
            }
            fn model_name(&self) -> &str {
                &self.name
            }
        }
        // All four share every other dedup key (bounds, scalars,
        // rope, SFUF) and differ only in the q-sig — exactly the
        // qwen3 case the bug surfaced on.
        let solved = vec![
            Fake {
                sig: "x|q:fp8-block".into(),
                stem: "qwen3-0.6b-fp8-block-128x128".into(),
                name: "qwen3_0_6b_fp8_block_128x128".into(),
            },
            Fake {
                sig: "x|q:fp8-std".into(),
                stem: "qwen3-0.6b-fp8-dynamic-per-tensor".into(),
                name: "qwen3_0_6b_fp8_dynamic_per_tensor".into(),
            },
            Fake {
                sig: "x|q:fp8-std".into(),
                stem: "qwen3-0.6b-fp8-static-per-tensor".into(),
                name: "qwen3_0_6b_fp8_static_per_tensor".into(),
            },
        ];
        let map = compute_canonical_variants(&solved);
        // Block stays its own canonical.
        assert_eq!(map[&0].to_string(), "qwen3_0_6b_fp8_block_128x128");
        // Dynamic + static collapse to dynamic (alphabetically
        // earliest stem in the {dynamic, static} pair).
        assert_eq!(map[&1].to_string(), "qwen3_0_6b_fp8_dynamic_per_tensor");
        assert_eq!(map[&2].to_string(), "qwen3_0_6b_fp8_dynamic_per_tensor");
    }
}
