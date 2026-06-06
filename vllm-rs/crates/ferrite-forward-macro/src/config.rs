// SPDX-License-Identifier: Apache-2.0
//! Phase 3: read `config.json` files from an `crates/ferrite-model-<arch>/configs/`
//! directory into a `Vec<ModelParams>`.
//!
//! Each `ModelParams` has:
//!   - a `name` (file stem, normalized to a Rust identifier) used
//!     by downstream codegen to name the emitted specialization;
//!   - a `bounds` map of every top-level integer field in the JSON.
//!
//! The bounds map is populated purely from the JSON — no
//! per-architecture knowledge lives here. Downstream passes
//! (shape inference, CFG, codegen) decide which of those bounds
//! they care about by name (`num_hidden_layers`, `hidden_size`,
//! ...). That keeps this pass generic across architectures.

// This module is Phase 3's deliverable. Consumers land in Phase 4
// (shape inference resolves symbolic shapes using these bounds);
// removed then.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::quantization::QuantizationConfig;

/// One model's parameters, loaded from one `config.json`.
#[derive(Clone, Debug)]
pub struct ModelParams {
    /// File stem, normalized to a valid Rust identifier. Stored as
    /// `String` (not `Ident`) so `ModelParams` is `Send + Sync` —
    /// the macro drive parallelizes per-model work and `Ident`
    /// wraps rustc's thread-local bridge.
    pub name: String,
    /// Original file stem (pre-normalization) and the path it was
    /// loaded from. For diagnostics.
    pub source_stem: String,
    pub source_path: PathBuf,
    /// Every top-level integer field from the JSON. Keys are the
    /// config.json field names verbatim (`num_hidden_layers`,
    /// `hidden_size`, etc.).
    pub bounds: BTreeMap<String, u64>,
    /// Every top-level float field from the JSON. Mirror of
    /// [`bounds`](Self::bounds) for non-integer scalars like
    /// `query_pre_attn_scalar` (Gemma2), `attn_logit_softcapping`,
    /// `rms_norm_eps`. Downstream callers read these by name; this
    /// module doesn't know which ones are used where.
    pub scalars: BTreeMap<String, f64>,
    /// Parsed `quantization_config` subobject, if present in the
    /// JSON. `None` for plain dense models. Consumers resolve
    /// per-weight storage format via
    /// [`crate::quantization::storage_format_for_weight`].
    pub quantization: Option<QuantizationConfig>,
    /// HF's `tie_word_embeddings` flag. When `true`, `lm_head`
    /// shares its weight buffer with `embed_tokens` and has no
    /// on-disk `lm_head.*` tensors — the codegen FieldLoad plan
    /// falls back to [`crate::codegen::FieldLoad::LinearTiedToEmbedding`]
    /// and the quant resolver keeps `lm_head` dense even under an
    /// AWQ config whose `modules_to_not_convert` doesn't list it.
    pub tie_word_embeddings: bool,
    /// HF `architectures: [..]` strings from this model's
    /// `config.json`. The macro unions these across every compiled
    /// model in an arch directory to produce the `hf_arches` list
    /// baked into the arch's
    /// [`ferrite_forward::FerriteArchRegistration`] registration,
    /// driving the runtime `try_load(..., arch_hint)` dispatch.
    pub architectures: Vec<String>,
    /// Extra JSON files that contributed to this variant's final
    /// config — the quantization preset and per-size override
    /// files that the loader deep-merged onto the dense base.
    /// Empty for dense variants; populated for synthesized
    /// `<size>-<preset>` variants. The `#[forward]` macro adds
    /// each to its `include_str!` tracking list so cargo rebuilds
    /// on any overlay/override edit, not just on dense-base edits.
    pub extra_tracked_paths: Vec<PathBuf>,
    /// Parsed `rope_scaling` subobject, if present. Drives the
    /// `RotaryCache` constructor picked in the emitted `Weights::load`.
    /// `None` = no scaling (standard `new_from_stream` with plain
    /// base freqs).
    pub rope_scaling: Option<RopeScaling>,
    /// Deterministic content hash of the raw `rope_scaling` JSON
    /// subobject, baked into this variant's `fingerprint_matches`.
    /// Discriminates checkpoints that share `type` +
    /// `max_position_embeddings` but differ in `short_factor` /
    /// `long_factor` — e.g. Phi-3.5-mini vs Phi-3-mini-128k,
    /// Phi-4-mini-instruct vs Phi-4-mini-reasoning. `None` when the
    /// config has no `rope_scaling`.
    pub rope_scaling_hash: Option<u64>,
    /// Qwen2-VL / Qwen2.5-VL `rope_scaling.mrope_section`: rotary-pair
    /// counts for the (T, H, W) axes of the multimodal RoPE
    /// generalization. Sums to `head_dim/2`. Drives the per-variant
    /// `const MROPE_SECTION: Option<[u32; 3]> = Some([..]);` override
    /// emitted on `impl CanonicalParams`. `None` for every text-only
    /// arch (the rope kernel takes the legacy 1D-positions path).
    /// See `~/.claude/plans/distributed-mapping-map.md` Phase A/B.
    pub mrope_section: Option<[u32; 3]>,
    /// On-disk safetensors prefix layout for a vision tower. `None`
    /// for text-only configs and for vision configs that omit the
    /// `vision_safetensors_layout` JSON field — codegen falls back to
    /// the Qwen-default layout (`visual` / `blocks` / no subtrees).
    /// Per-arch fields: `default_root` is the unindexed-weight root,
    /// `layered_subpath` is appended for indexed weights, and
    /// `subtrees` overrides the prefix for DSL paths whose first
    /// segment matches a key (e.g. Gemma3's `mm.*` → on-disk
    /// `multi_modal_projector.*`, sibling of `vision_tower`).
    pub vision_layout: Option<VisionSafetensorsLayout>,
    /// Vision d_model fingerprint: which on-disk weight name + dim
    /// the per-variant `try_load_mm` reads to discriminate among
    /// variants of the same arch. `None` falls back to Qwen's
    /// `visual.merger.mlp.2.weight` dim 0. Required only for
    /// non-Qwen-default MM arches.
    pub vision_d_model_fingerprint: Option<VisionDModelFingerprint>,
    /// Vision patch-embed flatten target: which on-disk weight to
    /// flatten in-place on the CPU side before `LinearLayer::load`
    /// reads it. `key` is the on-disk safetensors name; `leading_dim`
    /// is the axis count to keep — every dim after it gets multiplied
    /// into one flat row (handles 4D `[E, C, P, P]` SigLIP and 5D
    /// `[E, C, T, P, P]` Qwen uniformly). `None` falls back to Qwen's
    /// `visual.patch_embed.proj.weight` leading_dim 0 (today's
    /// hardcoded 5D behavior).
    pub vision_patch_embed_flatten: Option<VisionPatchEmbedFlatten>,
    /// Safetensors key of a learned positional-embedding table that the
    /// vision wrapper interpolates host-side per forward
    /// (`fast_pos_embed_interpolate`) — Qwen3.5-VL's
    /// `vision_tower.pos_embed.weight` `[num_grid², embed_dim]`. `None`
    /// for towers without one (most), so the wrapper leaves `pos_embeds`
    /// unset and the DSL never emits `LoadPosEmbeds`. Distinct from the
    /// SigLIP `pos_embed(position_ids, weight)` row-gather (Gemma3-MM),
    /// which is a DSL op over a `vision_num_positions`-driven extern.
    pub vision_pos_embed_key: Option<String>,
    /// Decoder-side safetensors prefix to prepend to every text-decoder
    /// safetensors key. `None` for text-only and Qwen-style VL arches
    /// where the text decoder ships at top-level (`model.layers.<L>.<...>`,
    /// `lm_head.weight`, `model.embed_tokens.weight`). `Some("language_model")`
    /// for Gemma3-MM-style multimodal arches where HF nests the text
    /// decoder under `language_model.<...>` alongside `vision_tower.<...>`
    /// and `multi_modal_projector.<...>`. Read from the per-variant
    /// `decoder_safetensors_prefix` JSON field; threaded through
    /// `Program::decoder_safetensors_prefix` and consumed by
    /// [`crate::codegen::safetensors_prefix`] under `Prelude::Decoder`.
    pub decoder_safetensors_prefix: Option<String>,
    /// DSL-leaf → on-disk-leaf renames for arches whose layer classes
    /// share an on-disk weight name with DIFFERENT shapes (Gemma4:
    /// global layers' `self_attn.q_proj` is [8192, 3840] vs sliding
    /// [4096, 3840] — the DSL uses distinct names like `q_proj_global`
    /// so each gets its own manifest shape, and this map folds them
    /// back to the shared disk leaf). JSON: `weight_leaf_renames:
    /// {"self_attn.q_proj_global": "self_attn.q_proj", ...}`. Sorted
    /// for determinism; empty for every other arch.
    pub weight_leaf_renames: Vec<(String, String)>,
    /// HF `torch_dtype` string, lowercased. Read by codegen as the
    /// FALLBACK for the rotary cache compute dtype when the embed
    /// tensor isn't visible in the weights table at load time. Today's
    /// primary path reads `embed_tokens.weight`'s on-disk dtype at
    /// runtime — see `emit_weights_struct`'s `rotary_prelude`. AWQ
    /// variants frequently override the base dtype (Qwen2.5 base
    /// ships bf16; `*-Instruct-AWQ` ships f16) while keeping the
    /// manifest `torch_dtype` literal unchanged, so the runtime read
    /// is load-bearing for AWQ correctness. `None` when the JSON
    /// omits the field.
    pub torch_dtype: Option<String>,
}

/// On-disk safetensors layout for a vision tower. Drives
/// [`crate::codegen::safetensors_prefix`] for `Prelude::Vision`
/// programs. Field semantics in [`ModelParams::vision_layout`].
#[derive(Clone, Debug)]
pub struct VisionSafetensorsLayout {
    /// Disk root for unindexed weights (e.g. `visual` for Qwen2-VL,
    /// `vision_tower.vision_model` for Gemma3-MM SigLIP).
    pub default_root: String,
    /// Suffix appended after `default_root` for indexed (per-block)
    /// weights — e.g. `blocks` (Qwen2-VL: `visual.blocks.{l}.*`),
    /// `encoder.layers` (Gemma3-MM: `vision_tower.vision_model.encoder.layers.{l}.*`).
    pub layered_subpath: String,
    /// First-DSL-segment → disk-prefix overrides for sibling
    /// subtrees. Lookups treat the matching subtree as unindexed —
    /// the override fully replaces the `<default_root>(.<layered_subpath>.{l})?`
    /// prefix. Example: `{"mm" → "multi_modal_projector"}` routes
    /// Gemma3-MM's `mm.mm_soft_emb_norm` to disk
    /// `multi_modal_projector.mm_soft_emb_norm`.
    pub subtrees: BTreeMap<String, String>,
}

impl VisionSafetensorsLayout {
    /// Today's hardcoded Qwen2-VL / Qwen2.5-VL convention. Used
    /// when a vision config omits `vision_safetensors_layout`.
    pub fn qwen_default() -> Self {
        Self {
            default_root: "visual".to_string(),
            layered_subpath: "blocks".to_string(),
            subtrees: BTreeMap::new(),
        }
    }

    /// Qwen3.5-VL on-disk convention: the wrapper checkpoint roots
    /// the tower at `vision_tower.*` (not `visual.*`).
    pub fn qwen3_5_default() -> Self {
        Self {
            default_root: "vision_tower".to_string(),
            layered_subpath: "blocks".to_string(),
            subtrees: BTreeMap::new(),
        }
    }

    /// Gemma3-MM SigLIP convention: tower under
    /// `vision_tower.vision_model.encoder.layers.{l}.*`, with the
    /// `mm.*` DSL subtree routed to the sibling
    /// `multi_modal_projector.*`.
    pub fn gemma3_default() -> Self {
        Self {
            default_root: "vision_tower.vision_model".to_string(),
            layered_subpath: "encoder.layers".to_string(),
            subtrees: BTreeMap::from([("mm".to_string(), "multi_modal_projector".to_string())]),
        }
    }
}

/// d_model fingerprint key + dim — see [`ModelParams::vision_d_model_fingerprint`].
#[derive(Clone, Debug)]
pub struct VisionDModelFingerprint {
    pub key: String,
    pub dim: usize,
}

impl VisionDModelFingerprint {
    pub fn qwen_default() -> Self {
        Self {
            key: "visual.merger.mlp.2.weight".to_string(),
            dim: 0,
        }
    }

    /// Qwen3.5-VL: merger renamed `merger.linear_fc2`, rooted at
    /// `vision_tower.*`.
    pub fn qwen3_5_default() -> Self {
        Self {
            key: "vision_tower.merger.linear_fc2.weight".to_string(),
            dim: 0,
        }
    }

    /// Gemma3-MM: the SigLIP→text projector matrix is
    /// `[vision_embed_dim, d_model]`, so d_model is dim 1.
    pub fn gemma3_default() -> Self {
        Self {
            key: "multi_modal_projector.mm_input_projection_weight".to_string(),
            dim: 1,
        }
    }
}

/// Patch-embed flatten target — see [`ModelParams::vision_patch_embed_flatten`].
#[derive(Clone, Debug)]
pub struct VisionPatchEmbedFlatten {
    pub key: String,
    pub leading_dim: usize,
    /// `true` when the on-disk conv weight is channels-LAST
    /// (`[out, kt, kh, kw, in]`, MLX convention) and must be permuted to
    /// `[out, in, kt, kh, kw]` before flattening so it pairs with the
    /// channels-FIRST `[in, kt, kh, kw]` patch packing. `false` (the
    /// default) does a plain reshape — correct for channels-first
    /// (torch/SigLIP `[out, in, p, p]`) checkpoints.
    pub channels_last: bool,
}

impl VisionPatchEmbedFlatten {
    pub fn qwen_default() -> Self {
        Self {
            key: "visual.patch_embed.proj.weight".to_string(),
            leading_dim: 0,
            channels_last: false,
        }
    }

    /// Qwen3.5-VL: 5D conv weight at `vision_tower.*`; MLX-converted
    /// checkpoints ship channels-LAST (`try_load_mm` sniffs the
    /// actual layout at load — see vision_glue's channels-first
    /// detection — so this flag only marks "may need the permute").
    pub fn qwen3_5_default() -> Self {
        Self {
            key: "vision_tower.patch_embed.proj.weight".to_string(),
            leading_dim: 0,
            channels_last: true,
        }
    }

    /// Gemma3-MM SigLIP: 4D `[E, C, P, P]` torch conv weight,
    /// channels-first.
    pub fn gemma3_default() -> Self {
        Self {
            key: "vision_tower.vision_model.embeddings.patch_embedding.weight".to_string(),
            leading_dim: 0,
            channels_last: false,
        }
    }
}

/// Arch-agnostic rope-scaling flavor parsed from `config.json`.
/// Integer-valued `original_max_position_embeddings` is kept as
/// `u64` so both `bounds` readers and the rotary kernels (which
/// want `usize`) can consume it without ambiguity.
#[derive(Clone, Debug)]
pub enum RopeScaling {
    /// `rope_scaling.type == "llama3"` — frequency-bucketed scaling
    /// used by Llama-3.x. All fields come from `rope_scaling.*`.
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position_embeddings: u64,
    },
    /// `rope_scaling.type == "longrope"` or `"su"` — Phi-3 LongRoPE
    /// (su-scaling). Factor vectors have length `rotary_dim/2`.
    /// `short_mscale` / `long_mscale` default to the Phi-3 paper
    /// formula when omitted from the config — Python vLLM's
    /// `Phi3LongRoPEScaledRotaryEmbedding.__init__` materializes
    /// the default from `scaling_factor = sqrt(1 + ln(max_pos/orig_max) / ln(orig_max))`
    /// when the caller passes `None`.
    LongRope {
        short_factor: Vec<f64>,
        long_factor: Vec<f64>,
        original_max_position_embeddings: u64,
        short_mscale: f64,
        long_mscale: f64,
    },
    /// `rope_scaling.type == "yarn"` — DeepSeek-V2 YaRN NTK-by-parts
    /// interpolation with mscale correction.
    Yarn {
        factor: f64,
        beta_fast: f64,
        beta_slow: f64,
        mscale: f64,
        mscale_all_dim: f64,
        original_max_position_embeddings: u64,
    },
}

/// Errors produced while loading configs.
#[derive(Debug)]
pub enum ConfigError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    NotADirectory(PathBuf),
    BadStem {
        path: PathBuf,
        reason: &'static str,
    },
    Quantization {
        path: PathBuf,
        source: crate::quantization::ParseError,
    },
    BadQuantizations {
        path: PathBuf,
        reason: &'static str,
    },
    /// A `#[vision_forward]` config (verbatim HF VL-wrapper
    /// checkpoint) is missing a field the per-family `vision_*`
    /// bound derivation needs, or its arch family is unknown to
    /// [`VisionFamily`].
    VisionDerivation {
        path: PathBuf,
        reason: String,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "reading {}: {source}", path.display()),
            Self::Json { path, source } => write!(f, "parsing {}: {source}", path.display()),
            Self::NotADirectory(p) => write!(f, "not a directory: {}", p.display()),
            Self::BadStem { path, reason } => {
                write!(f, "bad file stem for {}: {reason}", path.display())
            }
            Self::Quantization { path, source } => {
                write!(f, "quantization_config in {}: {source}", path.display())
            }
            Self::BadQuantizations { path, reason } => {
                write!(f, "{}: {reason}", path.display())
            }
            Self::VisionDerivation { path, reason } => {
                write!(f, "vision config {}: {reason}", path.display())
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// Load every `*.json` file in `dir` as a `ModelParams`, then
/// synthesize additional `<size>-<preset>` variants for every
/// `quantizations.json`-declared preset × dense base. Dense bases,
/// preset overlays, and optional `<size>-<preset>.overrides.json`
/// files are deep-merged at load time.
///
/// Files skipped from the dense-base scan:
/// - `weights.json` — per-arch shape manifest, loaded separately.
/// - `quantizations.json` — preset declaration list.
/// - `*.overrides.json` — per-(size, preset) drift overrides
///   applied during synthesis.
///
/// Overlay presets are looked up under `<dir>/../quantizations/`
/// (sibling of the arch directory). Each preset's JSON fragment
/// deep-merges onto the base — typically just
/// `{quantization_config: {...}}`, but any top-level field is
/// allowed.
///
/// Results are sorted alphabetically by file stem for build
/// determinism; synthesized variants appear after their base.
pub fn load_dir(dir: &Path) -> Result<Vec<ModelParams>, ConfigError> {
    load_dir_mode(dir, ConfigMode::Decoder)
}

/// [`load_dir`] for a `#[vision_forward]` crate's configs/ dir.
/// Same file discovery / overlay synthesis / filtering; per-file
/// `ModelParams` construction goes through the vision path
/// (per-family `vision_*` bound derivation from the nested HF
/// `vision_config` block) instead of the flat decoder harvest.
pub fn load_dir_vision(dir: &Path) -> Result<Vec<ModelParams>, ConfigError> {
    load_dir_mode(dir, ConfigMode::Vision)
}

/// Which macro is consuming the configs — `#[forward]` (Decoder) or
/// `#[vision_forward]` (Vision). The configs/ files themselves are
/// VERBATIM HF checkpoint configs either way; the mode selects which
/// view of the checkpoint the crate compiles:
///
/// - **Decoder** harvests the flat top-level fields (after
///   `normalize_hf_config` hoists `text_config` / `rope_parameters`)
///   — the text decoder's identity.
/// - **Vision** derives the `vision_*` bound set + `d_model` +
///   `vision_norm_eps` from the nested `vision_config` block,
///   arch-family-keyed, and deliberately harvests NOTHING else from
///   the top level (a VL-wrapper's text fields like
///   `intermediate_size` / `head_dim` would otherwise leak into the
///   vision expansion's `W::` consts — e.g. shadowing Qwen2.5-VL's
///   `vision_intermediate_size_padded` SwiGLU split-point).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ConfigMode {
    Decoder,
    Vision,
}

fn load_dir_mode(dir: &Path, mode: ConfigMode) -> Result<Vec<ModelParams>, ConfigError> {
    if !dir.is_dir() {
        return Err(ConfigError::NotADirectory(dir.to_path_buf()));
    }

    // Optional dev-iteration filter. `FERRITE_MODELS=stem1,stem2,...`
    // restricts the macro to those exact model stems (matching base
    // file stems and synthesized `<base>-<preset>` variants); unset
    // means no filter — every model in the directory compiles. The
    // per-arch crate's build.rs declares
    // `cargo:rerun-if-env-changed=FERRITE_MODELS` so cargo's
    // incremental cache invalidates when this changes.
    let enabled: Option<std::collections::HashSet<String>> =
        std::env::var("FERRITE_MODELS").ok().map(|s| {
            s.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect()
        });

    let all_json: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|source| ConfigError::Io {
            path: dir.to_path_buf(),
            source,
        })?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();

    let is_overrides = |p: &Path| {
        p.file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.ends_with(".overrides.json"))
            .unwrap_or(false)
    };

    let mut base_paths: Vec<PathBuf> = all_json
        .iter()
        .filter(|p| {
            let name = p.file_name().and_then(|s| s.to_str());
            !matches!(name, Some("weights.json") | Some("quantizations.json")) && !is_overrides(p)
        })
        .cloned()
        .collect();
    base_paths.sort_by(|a, b| {
        a.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .cmp(b.file_stem().and_then(|s| s.to_str()).unwrap_or(""))
    });

    let mut out: Vec<ModelParams> = Vec::new();
    // Remember each base's raw JSON so we can deep-merge overlays
    // onto it without re-reading and without copying the ModelParams
    // structure (synthesized variants have different bounds/quant
    // and need to re-parse from the merged raw JSON).
    let mut base_raw: Vec<(PathBuf, String, serde_json::Value)> = Vec::new();
    for p in &base_paths {
        let (raw, json) = read_json_file(p)?;
        let stem = stem_of(p)?;
        let model = model_params_from_json_mode(&json, &stem, p, Vec::new(), mode)?;
        base_raw.push((p.clone(), stem, json));
        out.push(model);
        let _ = raw;
    }

    // Overlay synthesis. Each arch's `quantizations.json` is a flat
    // list of preset names; every listed preset fans out across
    // every dense base in the arch to produce one compiled variant
    // per `(size, preset)` pair. Per-HF-repo drift (e.g.
    // TinyLlama-GPTQ's vocab_size=32003 vs the dense base's 32000)
    // lands as `<size>-<preset>.overrides.json` files that deep-
    // merge last.
    //
    // Universal fan-out is only compile-affordable because codegen
    // does cross-variant forward-fn deduplication — variants with
    // identical Impl-set signatures share one emitted body. Without
    // that dedup, N sizes × M presets quickly blow up release-build
    // LLVM work.
    // Under `--features metal` we skip CUDA-only quant variants
    // (awq-gemm / gptq-* / bnb-nf4-dq / fp8-* / ct-int4-sym / ggml) —
    // their on-disk layouts route through `MarlinFusedGateUpSiluMul` /
    // `Bnb4Linear` / `Fp8Linear` / `GgmlLinear` impls that the metal
    // impl pool has no claimants for, so the solver explodes with
    // `UnclaimedTile` on the first quant tile. The dense base
    // variants still flow through.
    //
    // MLX-affine presets (`mlx-affine-b<bits>-g<gs>`) are the
    // exception: their codegen materializes a Dense `LinearLayer` at
    // load time via `LinearLayer::load_affine_dequant_as_dense`, so
    // the forward path remains the existing bf16 Gemm path that the
    // metal impl pool already claims. The preset-filter inside the
    // overlay loop below keeps CUDA-only presets out while still
    // synthesizing the affine variant.
    let quantizations_path = dir.join("quantizations.json");
    if quantizations_path.exists() {
        let (_, qjson) = read_json_file(&quantizations_path)?;
        // Each entry is either a bare preset name (`"fp8-..."`,
        // `"ggml"`) or a single-key object carrying registration data
        // for that preset's runtime side. Currently used by `ggml`
        // to forward arch-specific GGUF spec (qk_permute, gguf_arch
        // override, tensor_renames, metadata reads) into the
        // `ferrite_gguf::register!` call the macro emits. The preset
        // overlay loop here just needs the preset NAMES; the
        // structured fields are picked up later by `load_gguf_spec`.
        let entries = qjson
            .get("quantizations")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ConfigError::BadQuantizations {
                path: quantizations_path.clone(),
                reason: "expected `{\"quantizations\": [<entry>, ...]}` array",
            })?;
        let presets: Vec<String> = entries
            .iter()
            .filter_map(|v| {
                if let Some(s) = v.as_str() {
                    Some(s.to_string())
                } else if let Some(obj) = v.as_object()
                    && obj.len() == 1
                {
                    obj.keys().next().cloned()
                } else {
                    None
                }
            })
            .collect();

        // Presets live in the shared `ferrite-quantizations` crate.
        // Walk up from `dir` (which is `<crate>/configs/`) to find
        // the workspace root (first ancestor with a `Cargo.toml`
        // containing `[workspace]`), then descend to
        // `crates/ferrite-quantizations/presets/`.
        let preset_root = {
            let mut cur: Option<&Path> = Some(dir);
            let mut found: Option<PathBuf> = None;
            while let Some(d) = cur {
                let cargo_toml = d.join("Cargo.toml");
                if cargo_toml.exists()
                    && let Ok(s) = fs::read_to_string(&cargo_toml)
                    && s.contains("[workspace]")
                {
                    found = Some(
                        d.join("crates")
                            .join("ferrite-quantizations")
                            .join("presets"),
                    );
                    break;
                }
                cur = d.parent();
            }
            found.ok_or_else(|| ConfigError::NotADirectory(dir.to_path_buf()))?
        };

        for preset_name in &presets {
            // Metal: `mlx-affine-*` variants reach the solver as Dense
            // (codegen dequantizes at load time), and `nvfp4` reaches it
            // as a genuine metal quant claimed by `MetalNvfp4QmmImpl`
            // (forward-time E2M1 dequant-on-read qmv/qmm_t). Every other
            // preset routes through CUDA-only Impls (Marlin / Bnb4 / Fp8
            // / Ggml) that the metal impl pool has no claimants for, so
            // the solver would explode with `UnclaimedTile`.
            if cfg!(feature = "metal")
                && !preset_name.starts_with("mlx-affine-")
                && preset_name != "nvfp4"
            {
                continue;
            }
            // CUDA: the mirror — `mlx-affine-*` weights are an Apple
            // checkpoint format with no CUDA Impl in the pool (the
            // Affine quant flow lives entirely in the metal kernels).
            // `nvfp4` is currently metal-only too (E2M1 dequant qmv/qmm_t
            // shaders + `MetalNvfp4QmmImpl`); a CUDA NVFP4 path is future
            // work. Skip both so the cuda solver doesn't fail with
            // `UnclaimedTile` on `Embed` / `Gemm` for those storage tags.
            if cfg!(feature = "cuda")
                && (preset_name.starts_with("mlx-affine-") || preset_name == "nvfp4")
            {
                continue;
            }
            let preset_path = preset_root.join(format!("{preset_name}.json"));
            let (_, preset_json) = read_json_file(&preset_path)?;

            for (base_path, base_stem, base_json) in &base_raw {
                let variant_stem = format!("{base_stem}-{preset_name}");
                let override_path = dir.join(format!("{variant_stem}.overrides.json"));

                let mut merged = base_json.clone();
                deep_merge(&mut merged, &preset_json);
                let mut extra_tracked = vec![preset_path.clone()];
                if override_path.exists() {
                    let (_, override_json) = read_json_file(&override_path)?;
                    deep_merge(&mut merged, &override_json);
                    extra_tracked.push(override_path);
                }

                let variant = model_params_from_json_mode(
                    &merged,
                    &variant_stem,
                    base_path,
                    extra_tracked,
                    mode,
                )?;
                out.push(variant);
            }
        }
    }

    // Drop any explicitly-quantized base configs under
    // `--features metal` *except* MLX-affine, which routes through the
    // existing Dense Impl pool at solve time (codegen materializes a
    // Dense `LinearLayer` at load via
    // `LinearLayer::load_affine_dequant_as_dense`). Other
    // checked-in `<size>-<preset>.json` base configs
    // (e.g. `qwen3-0.6b-bnb-4bit.json`) still reach the solver as
    // quantized and the metal impl pool has no claimants — the
    // solver would explode on the first quantized tile.
    if cfg!(feature = "metal") {
        out.retain(|m| {
            m.quantization.is_none()
                || matches!(
                    m.quantization.as_ref().map(|qc| &qc.method),
                    Some(crate::quantization::QuantMethod::Affine { .. })
                        | Some(crate::quantization::QuantMethod::Nvfp4 { .. })
                )
        });
    }

    // Keep the final list sorted by stem so emitted
    // arch-dispatcher arm order stays deterministic across
    // `quantizations.json` edits.
    out.sort_by(|a, b| a.source_stem.cmp(&b.source_stem));

    if let Some(set) = &enabled {
        // Empty result is OK — when FERRITE_MODELS filters every
        // model out of one arch (e.g. user is iterating on llama
        // and all arches are enabled), this arch's dispatcher emits
        // an empty Weights enum and no inventory registration. The
        // downstream emit_arch_dispatcher already handles empty
        // arms by producing nothing.
        let available: Vec<String> = out.iter().map(|m| m.source_stem.clone()).collect();
        // A user-supplied stem matches either exactly (a base or a
        // synthesized `<base>-<preset>` variant) OR as a base prefix —
        // `FERRITE_MODELS=llama-3.2-3b` keeps the dense base AND every
        // overlay variant (`-ggml`, `-awq-gemm`, …). Without the prefix
        // pass, requesting a base by its bare stem silently dropped
        // every quant variant for that base and arches like `-ggml`
        // never registered.
        out.retain(|m| {
            set.iter()
                .any(|s| m.source_stem == *s || m.source_stem.starts_with(&format!("{s}-")))
        });

        // Typo detection: warn (don't fail) when the filter was
        // non-empty AND this arch had models AND none matched. The
        // most common gotcha is dot-vs-dash on stems like
        // `llama-3.2-3b` — surface the right spelling by checking
        // whether the requested stem matches any available stem
        // after dot→dash normalization.
        if !available.is_empty() && out.is_empty() {
            let normalized: std::collections::HashMap<String, &String> =
                available.iter().map(|s| (s.replace('.', "-"), s)).collect();
            let suggestions: Vec<&String> = set
                .iter()
                .filter_map(|req| {
                    normalized
                        .get(&req.replace('.', "-"))
                        .copied()
                        .filter(|orig| *orig != req)
                })
                .collect();
            let mut req_sorted: Vec<&String> = set.iter().collect();
            req_sorted.sort();
            eprintln!(
                "ferrite · warn: FERRITE_MODELS={:?} matched 0 of {} models in {}",
                req_sorted,
                available.len(),
                dir.display(),
            );
            if !suggestions.is_empty() {
                let mut sug_sorted: Vec<&&String> = suggestions.iter().collect();
                sug_sorted.sort();
                eprintln!(
                    "           did you mean: {}",
                    sug_sorted
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                );
            }
        }
    }

    Ok(out)
}

/// Load a single config.json as a dense-base ModelParams.
pub fn load_file(path: &Path) -> Result<ModelParams, ConfigError> {
    let (_, json) = read_json_file(path)?;
    let stem = stem_of(path)?;
    model_params_from_json(&json, &stem, path, Vec::new())
}

/// Read + parse a JSON file, returning the raw string (for
/// diagnostics) and the parsed `Value`. Centralized so
/// `ConfigError::{Io, Json}` always carry the right path.
fn read_json_file(path: &Path) -> Result<(String, serde_json::Value), ConfigError> {
    let contents = fs::read_to_string(path).map_err(|source| ConfigError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let json: serde_json::Value =
        serde_json::from_str(&contents).map_err(|source| ConfigError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    Ok((contents, json))
}

/// Per-arch GGUF registration data harvested from `quantizations.json`.
/// Populated only when the arch's quantizations list contains a `"ggml"`
/// entry (string or object form). Forwarded into the
/// `ferrite_gguf::register!` call the macro emits.
#[derive(Debug, Clone)]
pub struct GgufSpec {
    /// GGUF `general.architecture` override. Defaults to the `arch`
    /// ident on the `#[forward]` block; set explicitly when the arch
    /// reports a non-matching tag (Mistral GGUFs report `"llama"`,
    /// DeepSeek-V2 `"deepseek2"`, etc.).
    pub gguf_arch: Option<String>,
    pub qk_permute: bool,
    pub llama3_rope_scaling_inference: bool,
    /// Per-suffix tensor renames: `(gguf_suffix, hf_suffix)` pairs.
    pub tensor_renames: Vec<(String, String)>,
    /// `(gguf_key_template, extra_key)` u32 reads. The template may
    /// contain `{arch}` which `apply_metadata` substitutes.
    pub metadata_u32: Vec<(String, String)>,
    pub metadata_f32: Vec<(String, String)>,
    /// Constants always inserted into `HfModelConfig.extra` (split by
    /// numeric type so the macro can emit the right `GgufDefault`
    /// variant).
    pub metadata_defaults_u32: Vec<(String, u32)>,
    pub metadata_defaults_f32: Vec<(String, f32)>,
    /// Subtracted from every rmsnorm weight at GGUF load time. Lets
    /// archs (Gemma2/3) whose llama.cpp converter pre-bakes a
    /// constant into the stored weight recover the canonical "raw w"
    /// shape so the runtime kernel's `(w + offset)` fold isn't
    /// double-applied. Default 0.0.
    pub norm_weight_offset: f32,
    /// Whether this arch is the canonical owner of the gguf tag —
    /// the only crate that emits the inventory `ferrite_gguf::register!`
    /// call. Default `true`. Set `false` on non-canonical claimants
    /// (e.g. Mistral for `"llama"`, deepseek-v3-flat for `"deepseek2"`)
    /// so a single deterministic spec covers each gguf_arch. The arch
    /// still gets a `-ggml` overlay variant + `gguf_archs` entry in
    /// its `FerriteArchRegistration` so the dispatcher tries it.
    pub register_spec: bool,
}

impl Default for GgufSpec {
    fn default() -> Self {
        Self {
            gguf_arch: None,
            qk_permute: false,
            llama3_rope_scaling_inference: false,
            tensor_renames: Vec::new(),
            metadata_u32: Vec::new(),
            metadata_f32: Vec::new(),
            metadata_defaults_u32: Vec::new(),
            metadata_defaults_f32: Vec::new(),
            norm_weight_offset: 0.0,
            register_spec: true,
        }
    }
}

/// Read the arch's `quantizations.json` and return its GGUF spec —
/// `Some` when the list contains a `"ggml"` entry (string or object
/// form), `None` otherwise.
///
/// Schema for the structured form (all fields optional):
///
/// ```json
/// {"ggml": {
///   "qk_permute": true,
///   "gguf_arch": "llama",
///   "tensor_renames": {
///     "attn_qkv.weight": "self_attn.qkv_proj.weight"
///   },
///   "metadata_u32": {"{arch}.attention.sliding_window": "sliding_window"},
///   "metadata_f32": {"{arch}.attention.scale": "attention_multiplier"},
///   "metadata_defaults": {"sliding_window_pattern": 6},
///   "llama3_rope_scaling_inference": true
/// }}
/// ```
pub fn load_gguf_spec(dir: &Path) -> Result<Option<GgufSpec>, ConfigError> {
    let path = dir.join("quantizations.json");
    if !path.exists() {
        return Ok(None);
    }
    let (_, qjson) = read_json_file(&path)?;
    let arr = match qjson.get("quantizations").and_then(|v| v.as_array()) {
        Some(a) => a,
        None => return Ok(None),
    };
    let mut data: Option<&serde_json::Value> = None;
    let mut found_bare = false;
    for entry in arr {
        if entry.as_str() == Some("ggml") {
            found_bare = true;
        }
        if let Some(obj) = entry.as_object()
            && let Some(d) = obj.get("ggml")
        {
            data = Some(d);
            break;
        }
    }
    if !found_bare && data.is_none() {
        return Ok(None);
    }
    let mut spec = GgufSpec::default();
    let Some(data) = data else {
        return Ok(Some(spec));
    };
    let obj = data
        .as_object()
        .ok_or_else(|| ConfigError::BadQuantizations {
            path: path.clone(),
            reason: "`ggml` entry must be a string or `{\"ggml\": {<fields>}}` object",
        })?;
    let bad = || ConfigError::BadQuantizations {
        path: path.clone(),
        reason: "ggml field has wrong shape",
    };
    if let Some(v) = obj.get("gguf_arch").and_then(|v| v.as_str()) {
        spec.gguf_arch = Some(v.to_string());
    }
    if let Some(v) = obj.get("qk_permute").and_then(|v| v.as_bool()) {
        spec.qk_permute = v;
    }
    if let Some(v) = obj
        .get("llama3_rope_scaling_inference")
        .and_then(|v| v.as_bool())
    {
        spec.llama3_rope_scaling_inference = v;
    }
    if let Some(v) = obj.get("norm_weight_offset").and_then(|v| v.as_f64()) {
        spec.norm_weight_offset = v as f32;
    }
    if let Some(v) = obj.get("register_spec").and_then(|v| v.as_bool()) {
        spec.register_spec = v;
    }
    if let Some(v) = obj.get("tensor_renames") {
        let map = v.as_object().ok_or_else(bad)?;
        for (k, val) in map {
            spec.tensor_renames
                .push((k.clone(), val.as_str().ok_or_else(bad)?.to_string()));
        }
    }
    if let Some(v) = obj.get("metadata_u32") {
        let map = v.as_object().ok_or_else(bad)?;
        for (k, val) in map {
            spec.metadata_u32
                .push((k.clone(), val.as_str().ok_or_else(bad)?.to_string()));
        }
    }
    if let Some(v) = obj.get("metadata_f32") {
        let map = v.as_object().ok_or_else(bad)?;
        for (k, val) in map {
            spec.metadata_f32
                .push((k.clone(), val.as_str().ok_or_else(bad)?.to_string()));
        }
    }
    if let Some(v) = obj.get("metadata_defaults") {
        let map = v.as_object().ok_or_else(bad)?;
        for (k, val) in map {
            // Pick the variant by JSON type — integers go to u32,
            // numbers to f32. Bools/strings are not currently used.
            if let Some(n) = val.as_u64() {
                spec.metadata_defaults_u32
                    .push((k.clone(), n.try_into().map_err(|_| bad())?));
            } else if let Some(f) = val.as_f64() {
                spec.metadata_defaults_f32.push((k.clone(), f as f32));
            } else {
                return Err(bad());
            }
        }
    }
    Ok(Some(spec))
}

/// Extract a `Path::file_stem` as a String, erroring if it's
/// missing or non-UTF-8 (neither should happen on real disks;
/// guards against the `*.json.bak` / `~` editor-swap case).
fn stem_of(path: &Path) -> Result<String, ConfigError> {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .ok_or_else(|| ConfigError::BadStem {
            path: path.to_path_buf(),
            reason: "file has no stem or non-UTF-8 stem",
        })
}

/// Build a `ModelParams` from an already-parsed JSON value (possibly
/// the result of deep-merging a dense base with a quantization
/// preset + per-size overrides). `source_stem` is the variant name
/// that ends up on the generated Weights enum variant + inventory
/// registration. `source_path` points at the dense base for
/// synthesized variants; `extra_tracked_paths` carries the preset +
/// override files so the `#[forward]` macro can `include_str!` them
/// for cargo change detection.
/// Normalize an HF `config.json` for macro consumption. VL-wrapper
/// checkpoints (Qwen3.5 / Qwen3.5-MoE / …) nest the text decoder's
/// fields under `text_config`, and newer transformers nest rotary
/// params under `rope_parameters` — while the per-field readers
/// (`extract_bounds`, `extract_scalars`, `extract_mrope_section`, …)
/// all consume the flat single-decoder view. Hoist both levels.
///
/// An explicit top-level field always wins (hoisting never
/// overwrites), and `vision_config` deliberately stays nested — the
/// vision glue owns that subtree. The `configs/` files themselves are
/// VERBATIM copies of the HF checkpoint configs (fetched by
/// `probe-weights`); this normalization is the macro's job, never an
/// edit to the json.
fn normalize_hf_config(json: &serde_json::Value) -> serde_json::Value {
    let mut out = json.clone();
    let Some(obj) = out.as_object_mut() else {
        return out;
    };
    if let Some(text) = obj.get("text_config").cloned()
        && let Some(text_obj) = text.as_object()
    {
        for (k, v) in text_obj {
            obj.entry(k.clone()).or_insert(v.clone());
        }
    }
    if let Some(rope) = obj.get("rope_parameters").cloned()
        && let Some(rope_obj) = rope.as_object()
    {
        for (k, v) in rope_obj {
            obj.entry(k.clone()).or_insert(v.clone());
        }
    }
    // ModernBERT spells its dual rotary bases `global_rope_theta` /
    // `local_rope_theta`; the per-field readers (and the emitted
    // `rotary` / `rotary_local` cache ctors) consume the
    // gemma-convention `rope_theta` / `rope_local_base_freq` names.
    // Alias, don't rename — the config stays verbatim and an
    // explicit standard-name field still wins.
    if obj.get("model_type").and_then(|v| v.as_str()) == Some("modernbert") {
        if let Some(g) = obj.get("global_rope_theta").cloned() {
            obj.entry("rope_theta".to_string()).or_insert(g);
        }
        if let Some(l) = obj.get("local_rope_theta").cloned() {
            obj.entry("rope_local_base_freq".to_string()).or_insert(l);
        }
    }
    out
}

fn model_params_from_json_mode(
    json: &serde_json::Value,
    source_stem: &str,
    source_path: &Path,
    extra_tracked_paths: Vec<PathBuf>,
    mode: ConfigMode,
) -> Result<ModelParams, ConfigError> {
    match mode {
        ConfigMode::Decoder => {
            model_params_from_json(json, source_stem, source_path, extra_tracked_paths)
        }
        ConfigMode::Vision => {
            vision_params_from_json(json, source_stem, source_path, extra_tracked_paths)
        }
    }
}

/// Shared field parsers for the decoder / vision `ModelParams`
/// heads. Both construction paths call these on their respective
/// json view (decoder: normalized/hoisted; vision: raw) so a field's
/// extraction logic lives once.
fn parse_name(source_stem: &str, source_path: &Path) -> Result<String, ConfigError> {
    stem_to_ident(source_stem).map_err(|reason| ConfigError::BadStem {
        path: source_path.to_path_buf(),
        reason,
    })
}

fn parse_architectures(json: &serde_json::Value) -> Vec<String> {
    json.get("architectures")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn parse_quantization(
    json: &serde_json::Value,
    source_path: &Path,
) -> Result<Option<crate::quantization::QuantizationConfig>, ConfigError> {
    crate::quantization::QuantizationConfig::parse(json).map_err(|e| ConfigError::Quantization {
        path: source_path.to_path_buf(),
        source: e,
    })
}

/// `None` (field absent) lets `apply_arch_semantic_defaults` supply
/// the family's modeling-code default (HF's own default is TRUE;
/// gemma2/gemma3 checkpoints rely on it); explicit json wins.
fn parse_tie_word_embeddings(json: &serde_json::Value) -> Option<bool> {
    json.get("tie_word_embeddings").and_then(|v| v.as_bool())
}

/// Top-level `torch_dtype` → `text_config.torch_dtype` →
/// `text_config.dtype` (newer transformers serialization: Qwen3.5
/// wrappers carry the compute dtype only as `text_config.dtype`,
/// which the hoist surfaces as `dtype`, never `torch_dtype`).
fn parse_torch_dtype(json: &serde_json::Value) -> Option<String> {
    json.get("torch_dtype")
        .or_else(|| {
            json.get("text_config")
                .and_then(|t| t.get("torch_dtype").or_else(|| t.get("dtype")))
        })
        .and_then(|v| v.as_str())
        .map(|s| s.to_ascii_lowercase())
}

fn parse_decoder_safetensors_prefix(json: &serde_json::Value) -> Option<String> {
    json.get("decoder_safetensors_prefix")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

fn model_params_from_json(
    json: &serde_json::Value,
    source_stem: &str,
    source_path: &Path,
    extra_tracked_paths: Vec<PathBuf>,
) -> Result<ModelParams, ConfigError> {
    let json = &normalize_hf_config(json);
    let name = parse_name(source_stem, source_path)?;
    let mut bounds = extract_bounds(json);
    derive_implicit_bounds(&mut bounds);
    let scalars = extract_scalars(json);
    let quantization = parse_quantization(json, source_path)?;
    let mut tie_word_embeddings = parse_tie_word_embeddings(json);
    let architectures = parse_architectures(json);
    let rope_scaling = extract_rope_scaling(json);
    let rope_scaling_hash = json.get("rope_scaling").map(hash_json_value);
    let mrope_section = extract_mrope_section(json);
    let vision_layout = extract_vision_layout(json);
    let vision_d_model_fingerprint = extract_vision_d_model_fingerprint(json);
    let vision_patch_embed_flatten = extract_vision_patch_embed_flatten(json);
    let vision_pos_embed_key = json
        .get("vision_pos_embed_key")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    // Gemma4: DSL-name → on-disk leaf renames (global/sliding classes
    // share disk leaves with different shapes). Sorted for stable
    // longest-suffix matching in `codegen::safetensors_prefix`.
    let weight_leaf_renames: Vec<(String, String)> = json
        .get("weight_leaf_renames")
        .and_then(|v| v.as_object())
        .map(|m| {
            let mut v: Vec<(String, String)> = m
                .iter()
                .map(|(k, val)| {
                    (
                        k.clone(),
                        val.as_str()
                            .expect("weight_leaf_renames values must be strings")
                            .to_string(),
                    )
                })
                .collect();
            v.sort();
            v
        })
        .unwrap_or_default();
    let mut decoder_safetensors_prefix = parse_decoder_safetensors_prefix(json);
    let torch_dtype = parse_torch_dtype(json);

    apply_arch_semantic_defaults(
        &architectures,
        &mut bounds,
        &mut decoder_safetensors_prefix,
        &mut tie_word_embeddings,
    );

    Ok(ModelParams {
        name,
        source_stem: source_stem.to_string(),
        source_path: source_path.to_path_buf(),
        bounds,
        scalars,
        quantization,
        tie_word_embeddings: tie_word_embeddings.unwrap_or(false),
        architectures,
        extra_tracked_paths,
        rope_scaling,
        rope_scaling_hash,
        mrope_section,
        vision_layout,
        vision_d_model_fingerprint,
        vision_patch_embed_flatten,
        vision_pos_embed_key,
        decoder_safetensors_prefix,
        weight_leaf_renames,
        torch_dtype,
    })
}

/// VL arch families known to the vision-bound derivation. Keyed on
/// `architectures[0]` — the same string the runtime's
/// `FerriteMmRegistration` dispatch matches on. Each family maps the
/// HF `vision_config` block's (renamed-per-family) keys onto the
/// flat `vision_*` bound set the `#[vision_forward]` pipeline
/// consumes (vision_glue / codegen / shape.rs / weights.json shape
/// exprs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum VisionFamily {
    /// `Qwen2VLForConditionalGeneration` — vision_config keys
    /// `embed_dim` / `depth` / `num_heads` / `mlp_ratio` / `in_chans`.
    Qwen2Vl,
    /// `Qwen2_5_VLForConditionalGeneration` — `hidden_size` (= embed
    /// dim) / `intermediate_size` / `out_hidden_size` / `window_size`.
    Qwen2_5Vl,
    /// `Qwen3_5*` / `Qwen3_6*` wrappers — `hidden_size` /
    /// `intermediate_size` / `out_hidden_size` / `in_channels` /
    /// `num_position_embeddings`.
    Qwen3_5Vl,
    /// `Gemma3ForConditionalGeneration` — SigLIP vision_config:
    /// `hidden_size` / `intermediate_size` / `num_attention_heads` /
    /// `num_hidden_layers` / `num_channels` / `image_size` /
    /// `layer_norm_eps`; learned pos-embed, no rope, no merge.
    Gemma3,
}

/// Arch-string family predicates — the single source of truth for
/// the prefix spellings, shared by [`VisionFamily::detect`] and
/// [`apply_arch_semantic_defaults`]. Keeping these in one place is
/// load-bearing: the vision path runs BOTH detect and the semantic
/// defaults over the same `architectures` array, and if the two
/// drifted a family could derive its vision geometry yet miss its
/// modeling defaults (rms_norm_zero_centered, decoder prefix, tie).
fn is_qwen3_5_family_arch(a: &str) -> bool {
    a.starts_with("Qwen3_5") || a.starts_with("Qwen3_6")
}
fn is_gemma3_family_arch(a: &str) -> bool {
    a.starts_with("Gemma3")
}
fn is_gemma2_family_arch(a: &str) -> bool {
    a.starts_with("Gemma2")
}

/// Per-family data the vision derivation consumes — one compiler-
/// forced site per family. Everything that previously lived in
/// scattered `match family { .. , _ => None }` chains (whose
/// wildcard arms silently handed a NEW family the Qwen defaults)
/// is a field here instead: adding a [`VisionFamily`] variant now
/// fails to compile until its spec names every key spelling and
/// every structural sidecar.
struct VisionFamilySpec {
    /// `vision_config` key spellings for the shared tower geometry.
    /// (Qwen2-VL: `embed_dim`/`num_heads`/`in_chans`; Qwen2.5-VL:
    /// embed dim is `hidden_size`; Qwen3.5: chans is `in_channels`;
    /// Gemma3/SigLIP: transformers-standard names.)
    embed_key: &'static str,
    depth_key: &'static str,
    heads_key: &'static str,
    chans_key: &'static str,
    /// Rotary towers (every Qwen VL) rope half the head dim;
    /// SigLIP uses a learned absolute pos-embed and no rope at all
    /// (`vision_rope_half_dim = 0`).
    rope: bool,
    /// `vision_config` key carrying the block-norm eps. `None` =
    /// the family's modeling code hardcodes 1e-6 (transformers
    /// Qwen2VL/Qwen2_5_VL/Qwen3_5 vision blocks, mlx-vlm likewise).
    norm_eps_key: Option<&'static str>,
    /// Structural weight-layout sidecars. `None` = the Qwen2-VL
    /// convention via `qwen_default()` at the use sites
    /// (vision_glue / codegen).
    layout: Option<VisionSafetensorsLayout>,
    fingerprint: Option<VisionDModelFingerprint>,
    patch_embed_flatten: Option<VisionPatchEmbedFlatten>,
    /// Learned pos-embed table interpolated host-side per forward
    /// (Qwen3.5-VL only); see [`ModelParams::vision_pos_embed_key`].
    pos_embed_key: Option<&'static str>,
}

impl VisionFamily {
    fn detect(architectures: &[String]) -> Option<Self> {
        match architectures.first().map(String::as_str) {
            Some("Qwen2VLForConditionalGeneration") => Some(Self::Qwen2Vl),
            Some("Qwen2_5_VLForConditionalGeneration") => Some(Self::Qwen2_5Vl),
            Some(a) if is_qwen3_5_family_arch(a) => Some(Self::Qwen3_5Vl),
            Some("Gemma3ForConditionalGeneration") => Some(Self::Gemma3),
            _ => None,
        }
    }

    fn spec(self) -> VisionFamilySpec {
        match self {
            Self::Qwen2Vl => VisionFamilySpec {
                embed_key: "embed_dim",
                depth_key: "depth",
                heads_key: "num_heads",
                chans_key: "in_chans",
                rope: true,
                norm_eps_key: None,
                layout: None,
                fingerprint: None,
                patch_embed_flatten: None,
                pos_embed_key: None,
            },
            Self::Qwen2_5Vl => VisionFamilySpec {
                embed_key: "hidden_size",
                depth_key: "depth",
                heads_key: "num_heads",
                chans_key: "in_chans",
                rope: true,
                norm_eps_key: None,
                layout: None,
                fingerprint: None,
                patch_embed_flatten: None,
                pos_embed_key: None,
            },
            Self::Qwen3_5Vl => VisionFamilySpec {
                embed_key: "hidden_size",
                depth_key: "depth",
                heads_key: "num_heads",
                chans_key: "in_channels",
                rope: true,
                norm_eps_key: None,
                layout: Some(VisionSafetensorsLayout::qwen3_5_default()),
                fingerprint: Some(VisionDModelFingerprint::qwen3_5_default()),
                patch_embed_flatten: Some(VisionPatchEmbedFlatten::qwen3_5_default()),
                pos_embed_key: Some("vision_tower.pos_embed.weight"),
            },
            Self::Gemma3 => VisionFamilySpec {
                embed_key: "hidden_size",
                depth_key: "num_hidden_layers",
                heads_key: "num_attention_heads",
                chans_key: "num_channels",
                rope: false,
                norm_eps_key: Some("layer_norm_eps"),
                layout: Some(VisionSafetensorsLayout::gemma3_default()),
                fingerprint: Some(VisionDModelFingerprint::gemma3_default()),
                patch_embed_flatten: Some(VisionPatchEmbedFlatten::gemma3_default()),
                pos_embed_key: None,
            },
        }
    }
}

/// Build a `#[vision_forward]` variant's `ModelParams` from a
/// VERBATIM HF VL-wrapper config.json.
///
/// Everything the vision pipeline needs is DERIVED here from the
/// nested `vision_config` block (formulas) + arch-keyed structural
/// defaults (on-disk weight layout), so the configs/ files stay
/// byte-verbatim checkpoint copies. Flat top-level `vision_*` /
/// `d_model` keys always win when present — that is the
/// `.overrides.json` surface, and it keeps legacy flat configs
/// loading identically during migration.
///
/// Deliberately NOT harvested (unlike the decoder path):
/// - top-level / `text_config` text-decoder fields — they would leak
///   into the vision expansion's `W::` consts (see [`ConfigMode`]);
/// - `rope_scaling` / `mrope_section` — text-side rotary identity;
///   the tower's 2D rope is the DSL's `vision_rope` op and the mm
///   registration carries no rope fingerprint.
fn vision_params_from_json(
    json: &serde_json::Value,
    source_stem: &str,
    source_path: &Path,
    extra_tracked_paths: Vec<PathBuf>,
) -> Result<ModelParams, ConfigError> {
    let name = parse_name(source_stem, source_path)?;
    let architectures = parse_architectures(json);
    let family = VisionFamily::detect(&architectures).ok_or_else(|| {
        ConfigError::VisionDerivation {
            path: source_path.to_path_buf(),
            reason: format!(
                "unknown VL arch family {architectures:?} — teach \
                 VisionFamily::detect + VisionFamily::spec + \
                 derive_vision_bounds the new family",
            ),
        }
    })?;
    let spec = family.spec();

    // Flat harvest restricted to the vision namespace: `d_model` +
    // `vision_*` integers, minus the wrapper's `vision_*_token_id`
    // text-tokenizer ids (those are decoder-side splice identity,
    // not tower geometry).
    let mut bounds: BTreeMap<String, u64> = json
        .as_object()
        .map(|obj| {
            obj.iter()
                .filter(|(k, _)| {
                    (*k == "d_model" || k.starts_with("vision_")) && !k.ends_with("_token_id")
                })
                .filter_map(|(k, v)| v.as_u64().map(|n| (k.clone(), n)))
                .collect()
        })
        .unwrap_or_default();
    derive_vision_bounds(json, family, &spec, &mut bounds, source_path)?;

    // Mirror of the decoder path's int/float double-counting: every
    // derived integer bound is also visible as a scalar, plus the
    // one real float the vision glue reads (`vision_norm_eps`).
    let mut scalars: BTreeMap<String, f64> =
        bounds.iter().map(|(k, v)| (k.clone(), *v as f64)).collect();
    let norm_eps = json
        .get("vision_norm_eps")
        .and_then(|v| v.as_f64())
        .or_else(|| {
            spec.norm_eps_key.and_then(|key| {
                json.get("vision_config")
                    .and_then(|vc| vc.get(key))
                    .and_then(|v| v.as_f64())
            })
        })
        .unwrap_or(1e-6);
    scalars.insert("vision_norm_eps".to_string(), norm_eps);

    // `None` lets `apply_arch_semantic_defaults` (below) supply the
    // family modeling default, same as the decoder path. Inert
    // vision-side either way — lm_head tying is text-decoder
    // identity.
    let mut tie_word_embeddings = parse_tie_word_embeddings(json);

    let quantization = parse_quantization(json, source_path)?;

    // Structural weight-layout sidecars: explicit JSON fields win
    // (the `.overrides.json` surface), else the family spec implies
    // them (`None` in the spec = the Qwen2-VL `qwen_default()`s at
    // the use sites).
    let vision_layout = extract_vision_layout(json).or(spec.layout);
    let vision_d_model_fingerprint =
        extract_vision_d_model_fingerprint(json).or(spec.fingerprint);
    let vision_patch_embed_flatten =
        extract_vision_patch_embed_flatten(json).or(spec.patch_embed_flatten);
    let vision_pos_embed_key = json
        .get("vision_pos_embed_key")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .or_else(|| spec.pos_embed_key.map(str::to_string));

    let torch_dtype = parse_torch_dtype(json);

    // Same top-level `weight_leaf_renames` hook as the text-only parse
    // path (Gemma4's hybrid classes share disk leaves under a VL
    // wrapper arch — dropping this here would silently break loads).
    let weight_leaf_renames: Vec<(String, String)> = json
        .get("weight_leaf_renames")
        .and_then(|v| v.as_object())
        .map(|m| {
            let mut v: Vec<(String, String)> = m
                .iter()
                .map(|(k, val)| {
                    (
                        k.clone(),
                        val.as_str()
                            .expect("weight_leaf_renames values must be strings")
                            .to_string(),
                    )
                })
                .collect();
            v.sort();
            v
        })
        .unwrap_or_default();

    let mut decoder_safetensors_prefix = parse_decoder_safetensors_prefix(json);
    apply_arch_semantic_defaults(
        &architectures,
        &mut bounds,
        &mut decoder_safetensors_prefix,
        &mut tie_word_embeddings,
    );
    let tie_word_embeddings = tie_word_embeddings.unwrap_or(false);

    Ok(ModelParams {
        name,
        source_stem: source_stem.to_string(),
        source_path: source_path.to_path_buf(),
        bounds,
        scalars,
        quantization,
        tie_word_embeddings,
        architectures,
        extra_tracked_paths,
        rope_scaling: None,
        rope_scaling_hash: None,
        mrope_section: None,
        vision_layout,
        vision_d_model_fingerprint,
        vision_patch_embed_flatten,
        vision_pos_embed_key,
        decoder_safetensors_prefix,
        weight_leaf_renames,
        torch_dtype,
    })
}

/// Fill the flat `vision_*` + `d_model` bound set from the HF
/// `vision_config` block (and `text_config.hidden_size` /
/// `mm_tokens_per_image` where the family needs them). Bounds
/// already present (flat-key overrides) always win; only absent
/// keys are derived. Every inserted key is load-bearing: vision_glue
/// panics, codegen bakes silent zeros, or shape inference fails on
/// a missing one (or it is named by a weights.json shape expr / a
/// DSL body).
fn derive_vision_bounds(
    json: &serde_json::Value,
    family: VisionFamily,
    spec: &VisionFamilySpec,
    bounds: &mut BTreeMap<String, u64>,
    source_path: &Path,
) -> Result<(), ConfigError> {
    let err = |reason: String| ConfigError::VisionDerivation {
        path: source_path.to_path_buf(),
        reason,
    };
    let vc = json.get("vision_config").and_then(|v| v.as_object());
    let vcu = |k: &str| vc.and_then(|o| o.get(k)).and_then(|v| v.as_u64());
    let vcf = |k: &str| vc.and_then(|o| o.get(k)).and_then(|v| v.as_f64());

    // Per-family vision_config key spellings come from the spec —
    // see [`VisionFamilySpec`].
    let (embed_key, depth_key, heads_key, chans_key) = (
        spec.embed_key,
        spec.depth_key,
        spec.heads_key,
        spec.chans_key,
    );
    let got = |bounds: &BTreeMap<String, u64>, k: &str| bounds.get(k).copied();

    let embed = got(bounds, "vision_embed_dim")
        .or_else(|| vcu(embed_key))
        .ok_or_else(|| err(format!("cannot derive vision_embed_dim (vision_config.{embed_key})")))?;
    let depth = got(bounds, "vision_depth")
        .or_else(|| vcu(depth_key))
        .ok_or_else(|| err(format!("cannot derive vision_depth (vision_config.{depth_key})")))?;
    let heads = got(bounds, "vision_num_heads")
        .or_else(|| vcu(heads_key))
        .ok_or_else(|| err(format!("cannot derive vision_num_heads (vision_config.{heads_key})")))?;
    let chans = got(bounds, "vision_in_chans")
        .or_else(|| vcu(chans_key))
        .ok_or_else(|| err(format!("cannot derive vision_in_chans (vision_config.{chans_key})")))?;
    let patch = got(bounds, "vision_patch_size")
        .or_else(|| vcu("patch_size"))
        .ok_or_else(|| err("cannot derive vision_patch_size".to_string()))?;
    // SigLIP has no temporal axis and no spatial merge — the keys
    // are simply absent from its vision_config; 1 is the identity
    // for both.
    let temporal = got(bounds, "vision_temporal_patch_size")
        .or_else(|| vcu("temporal_patch_size"))
        .unwrap_or(1);
    let merge = got(bounds, "vision_spatial_merge_size")
        .or_else(|| vcu("spatial_merge_size"))
        .unwrap_or(1);
    if heads == 0 || !embed.is_multiple_of(heads) {
        return Err(err(format!(
            "vision_embed_dim={embed} not divisible by vision_num_heads={heads}"
        )));
    }
    let head_dim = got(bounds, "vision_head_dim").unwrap_or(embed / heads);

    bounds.insert("vision_embed_dim".to_string(), embed);
    bounds.insert("vision_depth".to_string(), depth);
    bounds.insert("vision_num_heads".to_string(), heads);
    bounds.insert("vision_head_dim".to_string(), head_dim);
    bounds.insert("vision_in_chans".to_string(), chans);
    bounds.insert("vision_patch_size".to_string(), patch);
    bounds.insert("vision_temporal_patch_size".to_string(), temporal);
    bounds.insert("vision_spatial_merge_size".to_string(), merge);
    // patch_embed GEMM K: one patch's flattened pixel count.
    bounds
        .entry("vision_in_features".to_string())
        .or_insert(chans * temporal * patch * patch);
    bounds
        .entry("vision_merge_factor".to_string())
        .or_insert(merge * merge);
    bounds
        .entry("vision_merge_hidden".to_string())
        .or_insert(embed * merge * merge);
    // Rotary towers rope half the head dim; non-rope towers
    // (SigLIP's learned absolute pos-embed) carry 0.
    bounds
        .entry("vision_rope_half_dim".to_string())
        .or_insert(if spec.rope { head_dim / 2 } else { 0 });

    match family {
        VisionFamily::Qwen2Vl => {
            // Qwen2-VL spells MLP width as a ratio (int or float in
            // the wild).
            let mlp = got(bounds, "vision_mlp_hidden")
                .or_else(|| vcf("mlp_ratio").map(|r| (embed as f64 * r) as u64))
                .ok_or_else(|| err("cannot derive vision_mlp_hidden (vision_config.mlp_ratio)".to_string()))?;
            bounds.insert("vision_mlp_hidden".to_string(), mlp);
        }
        VisionFamily::Qwen2_5Vl => {
            let inter = got(bounds, "vision_intermediate_size")
                .or_else(|| vcu("intermediate_size"))
                .ok_or_else(|| {
                    err("cannot derive vision_intermediate_size".to_string())
                })?;
            bounds.insert("vision_intermediate_size".to_string(), inter);
            // cuBLAS bf16 GEMM rejects K not divisible by 8; the
            // manifest's __pad_to_mult8__ entries zero-pad gate/up/
            // down to this width at load (no-op when already %8==0,
            // e.g. the 72B's 3456).
            bounds
                .entry("vision_intermediate_size_padded".to_string())
                .or_insert(inter.div_ceil(8) * 8);
            let win = got(bounds, "vision_window_size")
                .or_else(|| vcu("window_size"))
                .ok_or_else(|| err("cannot derive vision_window_size".to_string()))?;
            bounds.insert("vision_window_size".to_string(), win);
        }
        VisionFamily::Qwen3_5Vl => {
            let mlp = got(bounds, "vision_mlp_hidden")
                .or_else(|| vcu("intermediate_size"))
                .ok_or_else(|| err("cannot derive vision_mlp_hidden".to_string()))?;
            bounds.insert("vision_mlp_hidden".to_string(), mlp);
            // (`num_position_embeddings` stays un-derived: the
            // learned pos-embed table's row count is read from the
            // tensor itself at load — vision_glue's `sqrt(rows)`.)
        }
        VisionFamily::Gemma3 => {
            let mlp = got(bounds, "vision_mlp_hidden")
                .or_else(|| vcu("intermediate_size"))
                .ok_or_else(|| err("cannot derive vision_mlp_hidden".to_string()))?;
            bounds.insert("vision_mlp_hidden".to_string(), mlp);
            // `image_size` is a pure intermediate (only grid_side /
            // num_positions are consumed downstream) — not inserted.
            let image = got(bounds, "vision_image_size")
                .or_else(|| vcu("image_size"))
                .ok_or_else(|| err("cannot derive vision_image_size".to_string()))?;
            if patch == 0 || !image.is_multiple_of(patch) {
                return Err(err(format!(
                    "vision_image_size={image} not divisible by vision_patch_size={patch}"
                )));
            }
            let grid = got(bounds, "vision_patch_grid_side").unwrap_or(image / patch);
            bounds.insert("vision_patch_grid_side".to_string(), grid);
            let num_positions = got(bounds, "vision_num_positions").unwrap_or(grid * grid);
            bounds.insert("vision_num_positions".to_string(), num_positions);
            // Pooling: SigLIP's grid² tokens collapse to the
            // wrapper's `mm_tokens_per_image` via a square
            // avg-pool; kernel side = sqrt(grid² / tokens). The
            // pooled-token count itself is an intermediate — only
            // pool_factor / pool_kernel are consumed downstream.
            let pooled = got(bounds, "vision_pooled_tokens")
                .or_else(|| json.get("mm_tokens_per_image").and_then(|v| v.as_u64()))
                .ok_or_else(|| {
                    err("cannot derive vision_pooled_tokens (mm_tokens_per_image)".to_string())
                })?;
            if pooled == 0 || !num_positions.is_multiple_of(pooled) {
                return Err(err(format!(
                    "vision_num_positions={num_positions} not divisible by \
                     vision_pooled_tokens={pooled}"
                )));
            }
            let pool_factor = got(bounds, "vision_pool_factor").unwrap_or(num_positions / pooled);
            bounds.insert("vision_pool_factor".to_string(), pool_factor);
            let pool_kernel = got(bounds, "vision_pool_kernel").unwrap_or_else(|| {
                (pool_factor as f64).sqrt().round() as u64
            });
            if pool_kernel * pool_kernel != pool_factor {
                return Err(err(format!(
                    "vision_pool_factor={pool_factor} is not a perfect square \
                     (pool kernel must be square)"
                )));
            }
            bounds.insert("vision_pool_kernel".to_string(), pool_kernel);
        }
    }

    // The text decoder's hidden size = the merger/projector output
    // width. Qwen2.5/3.5 carry it in vision_config directly
    // (`out_hidden_size`); Qwen2-VL's vision_config spells it
    // `hidden_size` (its embed dim is `embed_dim`); Gemma3's
    // projector targets `text_config.hidden_size`.
    let d_model = got(bounds, "d_model")
        .or_else(|| match family {
            VisionFamily::Qwen2_5Vl | VisionFamily::Qwen3_5Vl => vcu("out_hidden_size"),
            VisionFamily::Qwen2Vl => {
                vcu("hidden_size").or_else(|| json.get("hidden_size").and_then(|v| v.as_u64()))
            }
            VisionFamily::Gemma3 => json
                .get("text_config")
                .and_then(|t| t.get("hidden_size"))
                .and_then(|v| v.as_u64()),
        })
        .ok_or_else(|| err("cannot derive d_model (text hidden / merger out width)".to_string()))?;
    bounds.insert("d_model".to_string(), d_model);

    Ok(())
}

/// Parse `vision_safetensors_layout`, if present. Missing or
/// malformed → `None`; codegen falls back to
/// [`VisionSafetensorsLayout::qwen_default`] in that case.
fn extract_vision_layout(json: &serde_json::Value) -> Option<VisionSafetensorsLayout> {
    let obj = json.get("vision_safetensors_layout")?.as_object()?;
    let default_root = obj.get("default_root")?.as_str()?.to_string();
    let layered_subpath = obj.get("layered_subpath")?.as_str()?.to_string();
    let mut subtrees = BTreeMap::new();
    if let Some(s) = obj.get("subtrees").and_then(|v| v.as_object()) {
        for (k, v) in s {
            if let Some(disk) = v.as_str() {
                subtrees.insert(k.clone(), disk.to_string());
            }
        }
    }
    Some(VisionSafetensorsLayout {
        default_root,
        layered_subpath,
        subtrees,
    })
}

fn extract_vision_d_model_fingerprint(json: &serde_json::Value) -> Option<VisionDModelFingerprint> {
    let obj = json.get("vision_d_model_fingerprint")?.as_object()?;
    let key = obj.get("key")?.as_str()?.to_string();
    let dim = obj.get("dim")?.as_u64()? as usize;
    Some(VisionDModelFingerprint { key, dim })
}

fn extract_vision_patch_embed_flatten(json: &serde_json::Value) -> Option<VisionPatchEmbedFlatten> {
    let obj = json.get("vision_patch_embed_flatten")?.as_object()?;
    let key = obj.get("key")?.as_str()?.to_string();
    let leading_dim = obj.get("leading_dim")?.as_u64()? as usize;
    let channels_last = obj
        .get("channels_last")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Some(VisionPatchEmbedFlatten {
        key,
        leading_dim,
        channels_last,
    })
}

/// Extract `mrope_section` as a fixed `[u32; 3]`, from a TOP-LEVEL
/// `mrope_section` first, else `rope_scaling.mrope_section`. Returns
/// `None` when absent (every text-only arch) or malformed (<3 entries).
/// Qwen2-VL/2.5-VL ship it under `rope_scaling` in the HF config (so the
/// checkpoint carries it and the rope_scaling fingerprint still matches).
/// Qwen3.5-VL DOESN'T carry it in the checkpoint (it's a model-code
/// default `[11,11,10]`) — so its ferrite config sets it TOP-LEVEL,
/// keeping `rope_scaling` absent (== the checkpoint) so the
/// `HfFingerprint` rope_scaling_hash disambiguation still matches.
fn extract_mrope_section(json: &serde_json::Value) -> Option<[u32; 3]> {
    let arr = json
        .get("mrope_section")
        .or_else(|| {
            json.get("rope_scaling")
                .and_then(|rs| rs.get("mrope_section"))
        })?
        .as_array()?;
    if arr.len() < 3 {
        return None;
    }
    let a = arr[0].as_u64()? as u32;
    let b = arr[1].as_u64()? as u32;
    let c = arr[2].as_u64()? as u32;
    Some([a, b, c])
}

/// Macro-side mirror of `ferrite_forward::hash_json_value` — the
/// macro can't depend on `ferrite-forward` (cycle), so the same
/// function body is duplicated here. Changes MUST stay in lockstep:
/// the manifest hash baked at compile time only matches the
/// runtime hash if both hashers agree bit-for-bit.
fn hash_json_value(v: &serde_json::Value) -> u64 {
    use std::hash::{Hash, Hasher};
    fn recurse<H: Hasher>(v: &serde_json::Value, h: &mut H) {
        match v {
            serde_json::Value::Null => 0u8.hash(h),
            serde_json::Value::Bool(b) => {
                1u8.hash(h);
                b.hash(h);
            }
            serde_json::Value::Number(n) => {
                2u8.hash(h);
                let f = n.as_f64().unwrap_or(0.0);
                f.to_bits().hash(h);
            }
            serde_json::Value::String(s) => {
                3u8.hash(h);
                s.hash(h);
            }
            serde_json::Value::Array(arr) => {
                4u8.hash(h);
                arr.len().hash(h);
                for v in arr {
                    recurse(v, h);
                }
            }
            serde_json::Value::Object(obj) => {
                5u8.hash(h);
                let mut keys: Vec<&String> = obj.keys().collect();
                keys.sort();
                keys.len().hash(h);
                for k in keys {
                    k.hash(h);
                    recurse(&obj[k], h);
                }
            }
        }
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    recurse(v, &mut h);
    h.finish()
}

/// Parse the `rope_scaling` subobject into a [`RopeScaling`]. Returns
/// `None` when the key is absent or the `type`/`rope_type` is neither
/// `llama3`, `longrope`, nor `su`. Integer-ish fields are extracted
/// via `as_u64` / `as_f64` so HF configs that write `4096` or `4096.0`
/// both parse. `attention_factor` falls back to the Phi-3 paper
/// formula when the config omits it.
fn extract_rope_scaling(json: &serde_json::Value) -> Option<RopeScaling> {
    let rs = json.get("rope_scaling")?;
    let rope_type = rs
        .get("rope_type")
        .or_else(|| rs.get("type"))
        .and_then(|v| v.as_str())?;
    match rope_type {
        "llama3" => {
            let factor = rs.get("factor").and_then(|v| v.as_f64())?;
            let low_freq_factor = rs
                .get("low_freq_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            let high_freq_factor = rs
                .get("high_freq_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(4.0);
            let original_max_position_embeddings = rs
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(8192);
            Some(RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            })
        }
        "longrope" | "su" => {
            let parse_factors = |key: &str| -> Option<Vec<f64>> {
                rs.get(key)?
                    .as_array()?
                    .iter()
                    .map(|v| v.as_f64())
                    .collect()
            };
            let short_factor = parse_factors("short_factor")?;
            let long_factor = parse_factors("long_factor")?;
            let original_max_position_embeddings = rs
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .or_else(|| {
                    json.get("original_max_position_embeddings")
                        .and_then(|v| v.as_u64())
                })
                .unwrap_or(4096);
            let max_pos = json
                .get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(4096);
            // Python vLLM's `Phi3LongRoPEScaledRotaryEmbedding.__init__`:
            //   scale = max_pos / orig_max
            //   scaling_factor = 1.0 if scale <= 1.0 else sqrt(1 + log(scale)/log(orig_max))
            //   short_mscale = short_mscale or scaling_factor
            //   long_mscale  = long_mscale  or scaling_factor
            let scaling_factor = {
                let scale = max_pos as f64 / original_max_position_embeddings as f64;
                if scale <= 1.0 {
                    1.0
                } else {
                    (1.0 + scale.ln() / (original_max_position_embeddings as f64).ln()).sqrt()
                }
            };
            let short_mscale = rs
                .get("short_mscale")
                .and_then(|v| v.as_f64())
                .unwrap_or(scaling_factor);
            let long_mscale = rs
                .get("long_mscale")
                .and_then(|v| v.as_f64())
                .unwrap_or(scaling_factor);
            Some(RopeScaling::LongRope {
                short_factor,
                long_factor,
                original_max_position_embeddings,
                short_mscale,
                long_mscale,
            })
        }
        "yarn" => {
            let factor = rs.get("factor").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let beta_fast = rs.get("beta_fast").and_then(|v| v.as_f64()).unwrap_or(32.0);
            let beta_slow = rs.get("beta_slow").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let mscale = rs.get("mscale").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let mscale_all_dim = rs
                .get("mscale_all_dim")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let original_max_position_embeddings = rs
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(4096);
            Some(RopeScaling::Yarn {
                factor,
                beta_fast,
                beta_slow,
                mscale,
                mscale_all_dim,
                original_max_position_embeddings,
            })
        }
        _ => None,
    }
}

/// Recursive deep-merge of JSON objects. When both sides agree on
/// a key whose value is an object, merge keys recursively; else
/// the overlay wins at that leaf. Used to stack a dense base with
/// a quantization preset (+ optional per-size overrides) into one
/// final variant JSON.
fn deep_merge(base: &mut serde_json::Value, overlay: &serde_json::Value) {
    use serde_json::Value;
    match (base, overlay) {
        (Value::Object(base_map), Value::Object(overlay_map)) => {
            for (k, v) in overlay_map {
                let entry = base_map.entry(k.clone()).or_insert(Value::Null);
                deep_merge(entry, v);
            }
        }
        (slot, overlay_val) => {
            *slot = overlay_val.clone();
        }
    }
}

/// Every top-level integer field becomes a bound. Anything else
/// (strings, bools, floats, nested objects) is ignored — the DSL
/// only quantifies over integers.
fn extract_bounds(json: &serde_json::Value) -> BTreeMap<String, u64> {
    json.as_object()
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| {
                    // Integers → as-is. Booleans → 0/1 (HF stores
                    // model config booleans like `norm_topk_prob`,
                    // `tie_word_embeddings`, `use_qk_norm` here; the
                    // bounds map is the only u64 table downstream
                    // consumers read).
                    v.as_u64()
                        .map(|n| (k.clone(), n))
                        .or_else(|| v.as_bool().map(|b| (k.clone(), b as u64)))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Every top-level number field (int OR float) becomes a scalar.
/// Integer fields also go to `bounds` via [`extract_bounds`] — they
/// double-count into both tables because some HF configs write
/// scale-type fields as integer literals (e.g. Gemma3's
/// `query_pre_attn_scalar: 256`, `rope_theta: 1000000`), while
/// other configs write them as floats (Gemma2 uses `256.0`,
/// `10000.0`). Readers of physics-scale values (softmax scales,
/// rope thetas, norm epsilons) look in `scalars`; readers of
/// shape-determining values look in `bounds`. Both views must
/// agree on integer-valued scale fields.
fn extract_scalars(json: &serde_json::Value) -> BTreeMap<String, f64> {
    json.as_object()
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_f64().map(|n| (k.clone(), n)))
                .collect()
        })
        .unwrap_or_default()
}

/// Apply HF's implicit config.json defaults to the bounds map.
///
/// HF configs are allowed to omit certain fields that have well-
/// defined defaults. Without these, shape inference on an older
/// config (like `llama-2-13b/config.json`, which omits `head_dim`)
/// would leave `num_heads * head_dim` unclosed. The defaults are
/// universal across every HF transformer — they live here, in the
/// config loader, rather than getting baked into per-arch shape
/// anchoring.
///
/// Defaults applied:
/// - **`head_dim`** ← `hidden_size / num_attention_heads`. Llama-2
///   and Llama-3 (pre-3.2) omit the field; Llama-3.2+, Qwen2.5+,
///   Gemma2 list it explicitly. Both forms are valid HF JSON.
/// - **`num_key_value_heads`** ← `num_attention_heads`. Configs
///   predating grouped-query attention assume MHA and don't list
///   the field.
///
/// Explicit values in the JSON always win — we only fill absent
/// keys.
fn derive_implicit_bounds(bounds: &mut BTreeMap<String, u64>) {
    // MLA (DeepSeek family) head_dim — MUST run before the generic
    // hidden/heads rule: MLA's per-head Q/K width is
    // `qk_nope_head_dim + qk_rope_head_dim` (V2-Lite: 128+64=192),
    // NOT hidden/heads (2048/16=128). The verbatim HF configs carry
    // the two summands but no `head_dim`.
    if !bounds.contains_key("head_dim")
        && let (Some(&nope), Some(&rope)) = (
            bounds.get("qk_nope_head_dim"),
            bounds.get("qk_rope_head_dim"),
        )
    {
        bounds.insert("head_dim".to_string(), nope + rope);
    }
    if !bounds.contains_key("head_dim")
        && let (Some(&hidden), Some(&heads)) =
            (bounds.get("hidden_size"), bounds.get("num_attention_heads"))
        && heads != 0
        && hidden.is_multiple_of(heads)
    {
        bounds.insert("head_dim".to_string(), hidden / heads);
    }
    // MLA projection out-dims referenced by the deepseek crates'
    // weights.json shape exprs — pure arithmetic over the verbatim
    // checkpoint fields. `q_proj_out` is q_proj's (or q_b_proj's,
    // under Q-LoRA) output width; `kv_a_proj_out` is the compressed
    // KV + rope-K width; `kv_lora_out` is kv_b_proj's decompressed
    // output; `attn_out` is o_proj's input (heads · v_head_dim,
    // consumed by the non-flat deepseek-v3 manifest). Explicit
    // values (synthetic test configs) always win.
    if let (Some(&heads), Some(&nope), Some(&rope), Some(&vhd)) = (
        bounds.get("num_attention_heads"),
        bounds.get("qk_nope_head_dim"),
        bounds.get("qk_rope_head_dim"),
        bounds.get("v_head_dim"),
    ) {
        bounds
            .entry("q_proj_out".to_string())
            .or_insert(heads * (nope + rope));
        bounds
            .entry("kv_lora_out".to_string())
            .or_insert(heads * (nope + vhd));
        bounds
            .entry("attn_out".to_string())
            .or_insert(heads * vhd);
        if let Some(&kv_lora_rank) = bounds.get("kv_lora_rank") {
            bounds
                .entry("kv_a_proj_out".to_string())
                .or_insert(kv_lora_rank + rope);
        }
    }
    if !bounds.contains_key("num_key_value_heads")
        && let Some(&heads) = bounds.get("num_attention_heads")
    {
        bounds.insert("num_key_value_heads".to_string(), heads);
    }
    if !bounds.contains_key("sliding_window_global_remainder")
        && let Some(&p) = bounds.get("sliding_window_pattern")
        && p > 0
    {
        bounds.insert("sliding_window_global_remainder".to_string(), p - 1);
    }
    // All-MoE decoders (Qwen3.5-MoE) have no dense MLP, so their HF
    // configs omit `intermediate_size` — but the bound is a universal
    // DISPATCH_FIELDS constant and the MLP-fusion seams
    // (SynthGateUpSiluMul, FusedGateUpSiluMul) size their SwiGLU from
    // it. For such configs the model's only non-routed SwiGLU is the
    // shared expert, so derive its width (0 when there is no shared
    // expert either). Dense / hybrid models ship `intermediate_size`
    // explicitly — explicit values always win.
    if !bounds.contains_key("intermediate_size") && bounds.contains_key("moe_intermediate_size") {
        let shared = bounds
            .get("shared_expert_intermediate_size")
            .copied()
            .unwrap_or(0);
        bounds.insert("intermediate_size".to_string(), shared);
    }
    // Geometry dims referenced by weights.json shape exprs — pure
    // arithmetic over checkpoint fields, derived here so the configs/
    // files stay verbatim HF copies. Explicit values always win.
    if !bounds.contains_key("attn_q_dim")
        && let (Some(&heads), Some(&hd)) =
            (bounds.get("num_attention_heads"), bounds.get("head_dim"))
    {
        bounds.insert("attn_q_dim".to_string(), heads * hd);
    }
    if !bounds.contains_key("q_gate_dim")
        && let Some(&q) = bounds.get("attn_q_dim")
    {
        // `attn_output_gate` (Qwen3.5 family) doubles q_proj's output:
        // per head `[query | gate]`.
        let gate = bounds.get("attn_output_gate").copied().unwrap_or(0) != 0;
        bounds.insert("q_gate_dim".to_string(), if gate { 2 * q } else { q });
    }
    if !bounds.contains_key("kv_dim")
        && let (Some(&kvh), Some(&hd)) = (bounds.get("num_key_value_heads"), bounds.get("head_dim"))
    {
        bounds.insert("kv_dim".to_string(), kvh * hd);
    }
    if !bounds.contains_key("gdn_value_dim")
        && let (Some(&vh), Some(&vd)) = (
            bounds.get("linear_num_value_heads"),
            bounds.get("linear_value_head_dim"),
        )
    {
        bounds.insert("gdn_value_dim".to_string(), vh * vd);
    }
    // Gated-DeltaNet conv channel count: q and k at key width plus v
    // at value width (`in_proj_qkv`'s output / `conv1d`'s channels).
    if !bounds.contains_key("gdn_conv_dim")
        && let (Some(&kh), Some(&kd), Some(&vdim)) = (
            bounds.get("linear_num_key_heads"),
            bounds.get("linear_key_head_dim"),
            bounds.get("gdn_value_dim"),
        )
    {
        bounds.insert("gdn_conv_dim".to_string(), 2 * kh * kd + vdim);
    }
}

/// Modeling-code semantics that HF checkpoint configs do NOT carry —
/// defaults the transformers / mlx-lm modeling source hardcodes per
/// architecture family. The `configs/` files stay verbatim HF copies,
/// so these arch-keyed defaults live here. Explicit config fields
/// (synthesized test configs, `.overrides.json` overlays) always win.
fn apply_arch_semantic_defaults(
    architectures: &[String],
    bounds: &mut BTreeMap<String, u64>,
    decoder_safetensors_prefix: &mut Option<String>,
    tie_word_embeddings: &mut Option<bool>,
) {
    // Family membership via the shared predicates (the same
    // spellings `VisionFamily::detect` keys on — see the predicate
    // fns for why drift between the two would be a silent bug).
    let qwen3_5_family = architectures.iter().any(|a| is_qwen3_5_family_arch(a));
    let gemma3_family = architectures.iter().any(|a| is_gemma3_family_arch(a));
    let gemma2_family = architectures.iter().any(|a| is_gemma2_family_arch(a));
    // Gemma2 alternates sliding/global attention every other layer
    // (`layer_is_sliding[i] = i % 2 == 0`). The checkpoint configs
    // don't carry a cadence field (newer transformers serializes an
    // expanded `layer_types` list instead); the DSL's
    // `layer % sliding_window_pattern` predicate needs the
    // compressed form.
    if gemma2_family {
        bounds
            .entry("sliding_window_pattern".to_string())
            .or_insert(2);
        // `derive_implicit_bounds` (which fills the remainder from an
        // EXPLICIT config pattern) has already run by the time this
        // default lands, so supply the matching remainder here too —
        // keeps the "pattern present ⇒ remainder present" invariant
        // for any DSL that predicates on it (gemma3-style
        // `layer % pattern == remainder`; gemma2's own DSL uses the
        // literal `== 0`).
        bounds
            .entry("sliding_window_global_remainder".to_string())
            .or_insert(1);
    }
    // Qwen3.5 / Qwen3.6 `*RMSNorm` stores zero-centered gains
    // (`x * (1 + w)`); mlx-lm's sanitize adds +1 on load, ferrite
    // keeps the on-disk form and offsets in-kernel via
    // NORM_WEIGHT_OFFSET.
    if qwen3_5_family {
        bounds
            .entry("rms_norm_zero_centered".to_string())
            .or_insert(1);
    }
    // Qwen3-family sparse-MoE routers renormalize the top-k weights;
    // newer configs omit `norm_topk_prob` and the modeling code
    // defaults it to TRUE (transformers Qwen3Next / Qwen3.5-MoE,
    // mlx-lm ModelArgs, vLLM's `getattr(config, "norm_topk_prob",
    // True)`). Qwen2-MoE's modeling default is false — its arch
    // prefix doesn't match. Qwen3-MoE ships the field explicitly, so
    // this only fires where the checkpoint config is silent.
    if bounds.contains_key("num_experts") && architectures.iter().any(|a| a.starts_with("Qwen3")) {
        bounds.entry("norm_topk_prob".to_string()).or_insert(1);
    }
    // VL-wrapper checkpoints nest the text decoder's weights under
    // `language_model.*` / `model.language_model.*` on disk (the two
    // orderings are alias-bridged at load by GpuWeights). The wrapper
    // arch implies the prefix; text-only `*ForCausalLM` repos leave
    // it unset. Qwen2-VL / Qwen2.5-VL wrappers ship the text decoder
    // at top-level `model.*` despite the wrapper arch — their
    // families deliberately stay off this rule.
    if decoder_safetensors_prefix.is_none()
        && (qwen3_5_family || gemma3_family)
        && architectures
            .iter()
            .any(|a| a.ends_with("ForConditionalGeneration"))
    {
        *decoder_safetensors_prefix = Some("language_model".to_string());
    }
    // Gemma ties embed_tokens ⟷ lm_head; HF's modeling default for
    // `tie_word_embeddings` is TRUE and the google/unsloth gemma2 /
    // gemma3 checkpoint configs rely on it (no field in the json, no
    // `lm_head.*` on disk). Ferrite's parse default is false, which
    // would route lm_head to a dense disk load and fail with
    // `weight not found: lm_head.weight`. Explicit json values win —
    // a genuinely untied repack (e.g. mlx-community/gemma-3-1b-it-4bit
    // ships a real quantized lm_head) declares
    // `"tie_word_embeddings": false` via its `.overrides.json`.
    if tie_word_embeddings.is_none() && (gemma3_family || gemma2_family) {
        *tie_word_embeddings = Some(true);
    }
    // Per-class attention geometry defaults: uniform-geometry arches
    // never declare global_* keys, but the shape sigs anchor the
    // `attention()` (global-class) tile to `global_head_dim` /
    // `num_global_key_value_heads` unconditionally — default them to
    // the base values so every existing arch resolves identically.
    if !bounds.contains_key("global_head_dim")
        && let Some(&hd) = bounds.get("head_dim")
    {
        bounds.insert("global_head_dim".to_string(), hd);
    }
    if !bounds.contains_key("num_global_key_value_heads")
        && let Some(&kv) = bounds.get("num_key_value_heads")
    {
        bounds.insert("num_global_key_value_heads".to_string(), kv);
    }
}

/// Normalize a file stem into a valid Rust identifier:
///   - replace runs of non-alphanumeric chars with `_`
///   - prepend `m_` if the result starts with a digit.
fn stem_to_ident(stem: &str) -> Result<String, &'static str> {
    if stem.is_empty() {
        return Err("empty stem");
    }
    let mut out = String::with_capacity(stem.len() + 2);
    let mut prev_underscore = false;
    for c in stem.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    // Trim leading/trailing underscores and collapse.
    let trimmed = out.trim_matches('_').to_string();
    let final_ = if trimmed.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        format!("m_{trimmed}")
    } else {
        trimmed
    };
    if final_.is_empty() {
        return Err("stem normalizes to empty identifier");
    }
    Ok(final_)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute path to `crates/ferrite-model-<arch>/configs/` from
    /// this crate's manifest dir, for tests that load real configs.
    fn repo_model_archs(arch: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(format!("ferrite-model-{arch}"))
            .join("configs")
    }

    #[test]
    fn stem_normalization() {
        assert_eq!(
            stem_to_ident("llama-3.2-1b").unwrap().to_string(),
            "llama_3_2_1b"
        );
        assert_eq!(stem_to_ident("qwen2-7b").unwrap().to_string(), "qwen2_7b");
        assert_eq!(
            stem_to_ident("qwen2.5-0.5b").unwrap().to_string(),
            "qwen2_5_0_5b"
        );
        // Leading digit → "m_" prefix.
        assert_eq!(stem_to_ident("3b-model").unwrap().to_string(), "m_3b_model");
        // Double-dashes collapse.
        assert_eq!(stem_to_ident("a--b").unwrap().to_string(), "a_b");
        assert!(stem_to_ident("").is_err());
    }

    #[test]
    fn load_real_llama_configs() {
        let dir = repo_model_archs("llama");
        let configs = load_dir(&dir).expect("load llama configs");

        // 12 dense bases (9 Llama + 2 smollm2 + 1 tinyllama) ×
        // (1 dense + 8 quant presets from `quantizations.json`:
        //  awq-gemm, bnb-nf4-dq, gptq-sym, gptq-sym-desc_act,
        //  ct-int4-sym, fp8-dynamic-per-tensor, fp8-static-per-tensor,
        //  ggml) = 108. Individual sizes don't always have real HF
        //  repos in every preset, but the compiler emits variants for
        //  all of them so fingerprint dispatch stays open-set at
        //  runtime.
        assert_eq!(
            configs.len(),
            108,
            "expected 108 Llama variants (12 bases × 9 variants)"
        );

        // Ground-truth check on llama-3.2-1b. Published values:
        //   num_hidden_layers = 16
        //   hidden_size       = 2048
        //   intermediate_size = 8192
        //   num_attention_heads = 32
        //   num_key_value_heads = 8
        //   head_dim            = 64
        //   vocab_size          = 128256
        let cfg = configs
            .iter()
            .find(|c| c.source_stem == "llama-3.2-1b")
            .expect("llama-3.2-1b present");
        assert_eq!(cfg.name.to_string(), "llama_3_2_1b");
        assert_eq!(cfg.bounds.get("num_hidden_layers"), Some(&16));
        assert_eq!(cfg.bounds.get("hidden_size"), Some(&2048));
        assert_eq!(cfg.bounds.get("intermediate_size"), Some(&8192));
        assert_eq!(cfg.bounds.get("num_attention_heads"), Some(&32));
        assert_eq!(cfg.bounds.get("num_key_value_heads"), Some(&8));
        assert_eq!(cfg.bounds.get("head_dim"), Some(&64));
        assert_eq!(cfg.bounds.get("vocab_size"), Some(&128256));
        // source_path preserved for later rebuild tracking and
        // diagnostics.
        assert!(cfg.source_path.ends_with("llama-3.2-1b.json"));
    }

    #[test]
    fn load_real_qwen2_configs() {
        let dir = repo_model_archs("qwen2");
        let configs = load_dir(&dir).expect("load qwen2 configs");
        // 13 dense (11 text + qwen2-vl-2b + qwen2.5-vl-3b text
        // decoders) × (1 dense + 7 presets: awq-gemm, bnb-nf4-dq,
        // ct-int4-sym, gptq-sym, fp8-dynamic-per-tensor,
        // fp8-static-per-tensor, ggml) = 104.
        assert_eq!(
            configs.len(),
            104,
            "expected 104 Qwen2 variants (13 bases × 8 variants)"
        );

        // Ground-truth check on Qwen2-0.5B:
        //   num_hidden_layers    = 24
        //   hidden_size          = 896
        //   intermediate_size    = 4864
        //   num_attention_heads  = 14
        //   num_key_value_heads  = 2
        //   vocab_size           = 151936
        let cfg = configs
            .iter()
            .find(|c| c.source_stem == "qwen2-0.5b")
            .expect("qwen2-0.5b present");
        assert_eq!(cfg.name.to_string(), "qwen2_0_5b");
        assert_eq!(cfg.bounds.get("num_hidden_layers"), Some(&24));
        assert_eq!(cfg.bounds.get("hidden_size"), Some(&896));
        assert_eq!(cfg.bounds.get("intermediate_size"), Some(&4864));
        assert_eq!(cfg.bounds.get("num_attention_heads"), Some(&14));
        assert_eq!(cfg.bounds.get("num_key_value_heads"), Some(&2));
        assert_eq!(cfg.bounds.get("vocab_size"), Some(&151936));
    }

    #[test]
    fn load_real_qwen2_vl_vision_configs() {
        // Vision configs (G.5.b) carry only `d_model` + `vision_*`
        // bounds — no decoder-only keys (`hidden_size`,
        // `num_attention_heads`, etc.). This test pins three things:
        //
        //   1. `load_dir` parses each variant cleanly without
        //      requiring the decoder fields,
        //   2. `derive_implicit_bounds` does NOT spuriously synthesize
        //      `head_dim` / `num_key_value_heads` (its source keys are
        //      absent → every branch short-circuits),
        //   3. every `vision_*` bound the codegen reads
        //      (`vision_num_heads`, `vision_head_dim`, plus the shape-
        //      anchoring `vision_in_features` / `vision_rope_half_dim`)
        //      lands in `bounds`.
        let dir = repo_model_archs("qwen2-vl");
        let configs = load_dir_vision(&dir).expect("load qwen2-vl vision configs");
        assert_eq!(
            configs.len(),
            3,
            "expected 3 Qwen2-VL vision variants (2b/7b/72b)"
        );

        // Per-variant `d_model` (= text-decoder hidden), vision tower
        // is shape-identical otherwise.
        let d_model_for = |stem: &str| -> u64 {
            *configs
                .iter()
                .find(|c| c.source_stem == stem)
                .unwrap_or_else(|| panic!("{stem} present"))
                .bounds
                .get("d_model")
                .expect("d_model")
        };
        assert_eq!(d_model_for("qwen2-vl-2b-instruct"), 1536);
        assert_eq!(d_model_for("qwen2-vl-7b-instruct"), 3584);
        assert_eq!(d_model_for("qwen2-vl-72b-instruct"), 8192);

        let cfg = configs
            .iter()
            .find(|c| c.source_stem == "qwen2-vl-2b-instruct")
            .expect("qwen2-vl-2b-instruct present");

        // Bounds the codegen reads via `emit_canonical_params_impl`
        // and `extern_shape`.
        assert_eq!(cfg.bounds.get("vision_num_heads"), Some(&16));
        assert_eq!(cfg.bounds.get("vision_head_dim"), Some(&80));
        assert_eq!(cfg.bounds.get("vision_in_features"), Some(&1176));
        assert_eq!(cfg.bounds.get("vision_rope_half_dim"), Some(&40));
        // Bounds the manifest formulas anchor on.
        assert_eq!(cfg.bounds.get("vision_embed_dim"), Some(&1280));
        assert_eq!(cfg.bounds.get("vision_mlp_hidden"), Some(&5120));
        assert_eq!(cfg.bounds.get("vision_merge_hidden"), Some(&5120));
        assert_eq!(cfg.bounds.get("vision_depth"), Some(&32));
        assert_eq!(cfg.bounds.get("vision_spatial_merge_size"), Some(&2));
        assert_eq!(cfg.bounds.get("vision_merge_factor"), Some(&4));
        assert_eq!(cfg.bounds.get("vision_patch_size"), Some(&14));
        assert_eq!(cfg.bounds.get("vision_temporal_patch_size"), Some(&2));
        assert_eq!(cfg.bounds.get("vision_in_chans"), Some(&3));

        // Decoder-only keys are absent — `derive_implicit_bounds`
        // must not have synthesized them.
        assert!(!cfg.bounds.contains_key("hidden_size"));
        assert!(!cfg.bounds.contains_key("num_attention_heads"));
        assert!(!cfg.bounds.contains_key("num_key_value_heads"));
        assert!(!cfg.bounds.contains_key("head_dim"));
        assert!(!cfg.bounds.contains_key("num_hidden_layers"));
        assert!(!cfg.bounds.contains_key("vocab_size"));

        // Float scalars land in `scalars`. Vision norm eps for the
        // pre/post-attn LayerNorms; auto-extracted by
        // `extract_scalars` from any f64 field.
        assert_eq!(cfg.scalars.get("vision_norm_eps"), Some(&1e-6));

        // HF arch claim string + tie flag. The 2B checkpoint TIES
        // embed/lm_head (`tie_word_embeddings: true` in the verbatim
        // HF config — the old invented config wrongly said false);
        // inert vision-side (no lm_head in the tower), but the value
        // must mirror the checkpoint.
        assert_eq!(
            cfg.architectures,
            vec!["Qwen2VLForConditionalGeneration".to_string()]
        );
        assert!(cfg.tie_word_embeddings);
    }

    #[test]
    fn qwen2_vl_vision_weights_manifest_anchors_on_vision_bounds() {
        // The vision weights.json declares per-tensor shapes anchored
        // on `vision_*` bounds (and `d_model` on `merger.mlp.2`).
        // Pin a few representative shapes so a future edit that
        // accidentally swaps in decoder-side bounds (e.g.
        // `hidden_size`) trips this guard at unit-test time, before
        // it reaches a #[vision_forward] expansion.
        use crate::weights_manifest::load_or_empty;
        let dir = repo_model_archs("qwen2-vl");
        let manifest = load_or_empty(&dir).expect("load qwen2-vl weights.json");

        // G.5.f flipped `attn.qkv` from a top-level shape entry to a
        // `__packed_splits__` mapping → `[attn.q, attn.k, attn.v]`. The
        // body writes three separate gemms (text-side qwen2 pattern),
        // so the post-split keys are what land in `entries`.
        let splits = manifest
            .packed_splits
            .get("attn.qkv")
            .expect("attn.qkv in __packed_splits__");
        assert_eq!(
            splits,
            &vec!["attn.q".to_string(), "attn.k".into(), "attn.v".into()]
        );
        let q = manifest.lookup(&["attn", "q"]).expect("attn.q in manifest");
        // `[vision_embed_dim, vision_embed_dim]` — K, N order.
        assert_eq!(q.len(), 2);

        let merger_proj = manifest
            .lookup(&["merger", "mlp_2"])
            .expect("merger.mlp_2 in manifest");
        // `[vision_merge_hidden, d_model]` in K, N order.
        assert_eq!(merger_proj.len(), 2);

        let patch_embed = manifest
            .lookup(&["patch_embed", "proj"])
            .expect("patch_embed.proj in manifest");
        // `[vision_in_features, vision_embed_dim]` in K, N order.
        assert_eq!(patch_embed.len(), 2);

        let norm1 = manifest.lookup(&["norm1"]).expect("norm1 in manifest");
        assert_eq!(norm1.len(), 1);
    }

    #[test]
    fn bounds_are_sorted_by_stem_for_determinism() {
        let dir = repo_model_archs("llama");
        let configs = load_dir(&dir).unwrap();
        let stems: Vec<&str> = configs.iter().map(|c| c.source_stem.as_str()).collect();
        let mut sorted = stems.clone();
        sorted.sort();
        assert_eq!(stems, sorted, "configs should be alphabetically sorted");
    }

    #[test]
    fn missing_dir_errors_cleanly() {
        let result = load_dir(Path::new("/nonexistent/path/to/configs"));
        assert!(matches!(result, Err(ConfigError::NotADirectory(_))));
    }

    #[test]
    fn float_fields_land_in_scalars_integer_fields_do_not() {
        let tmp = std::env::temp_dir().join("ferrite_forward_scalars_test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::write(
            tmp.join("m.json"),
            r#"{
                "num_hidden_layers": 16,
                "hidden_size": 2048,
                "rms_norm_eps": 0.000001,
                "query_pre_attn_scalar": 256.0,
                "attn_logit_softcapping": 50.0
            }"#,
        )
        .unwrap();
        let cfg = load_file(&tmp.join("m.json")).unwrap();

        // Integers go to bounds AND to scalars (as f64). Floats go to
        // scalars only. This double-entry for integers is required so
        // readers of scale-type fields find values regardless of how
        // the upstream HF config formatted them.
        assert_eq!(cfg.bounds.get("num_hidden_layers"), Some(&16));
        assert_eq!(cfg.bounds.get("hidden_size"), Some(&2048));
        assert_eq!(cfg.scalars.get("num_hidden_layers"), Some(&16.0));
        assert_eq!(cfg.scalars.get("hidden_size"), Some(&2048.0));

        assert_eq!(cfg.scalars.get("rms_norm_eps"), Some(&0.000001));
        assert_eq!(cfg.scalars.get("query_pre_attn_scalar"), Some(&256.0));
        assert_eq!(cfg.scalars.get("attn_logit_softcapping"), Some(&50.0));

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn llama_configs_have_no_gemma_scalars() {
        // Regression guard: if someone adds `query_pre_attn_scalar`
        // to a Llama config, that would silently change the
        // hardcoded softmax scale in `AttentionViaCacheImpl`. The
        // Llama path has no such field today; lock that in.
        let dir = repo_model_archs("llama");
        let configs = load_dir(&dir).expect("load llama configs");
        for cfg in &configs {
            assert!(
                !cfg.scalars.contains_key("query_pre_attn_scalar"),
                "{} unexpectedly has query_pre_attn_scalar",
                cfg.source_stem
            );
            assert!(
                !cfg.scalars.contains_key("attn_logit_softcapping"),
                "{} unexpectedly has attn_logit_softcapping",
                cfg.source_stem
            );
        }
    }

    #[test]
    fn non_json_files_are_ignored() {
        // Create a temp dir with one json and one txt file.
        let tmp = std::env::temp_dir().join("ferrite_forward_phase3_test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join("model.json"), r#"{"num_hidden_layers": 4}"#).unwrap();
        fs::write(tmp.join("README.txt"), "ignore me").unwrap();

        let configs = load_dir(&tmp).unwrap();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].source_stem, "model");

        fs::remove_dir_all(&tmp).ok();
    }
}
