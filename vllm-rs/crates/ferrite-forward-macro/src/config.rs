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
        let model = model_params_from_json(&json, &stem, p, Vec::new())?;
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
    let quantizations_path = dir.join("quantizations.json");
    if quantizations_path.exists() {
        let (_, qjson) = read_json_file(&quantizations_path)?;
        let presets: Vec<String> = qjson
            .get("quantizations")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ConfigError::BadQuantizations {
                path: quantizations_path.clone(),
                reason: "expected `{\"quantizations\": [\"<preset>\", ...]}` string array",
            })?
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
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

                let variant =
                    model_params_from_json(&merged, &variant_stem, base_path, extra_tracked)?;
                out.push(variant);
            }
        }
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
fn model_params_from_json(
    json: &serde_json::Value,
    source_stem: &str,
    source_path: &Path,
    extra_tracked_paths: Vec<PathBuf>,
) -> Result<ModelParams, ConfigError> {
    let name = stem_to_ident(source_stem).map_err(|reason| ConfigError::BadStem {
        path: source_path.to_path_buf(),
        reason,
    })?;
    let mut bounds = extract_bounds(json);
    derive_implicit_bounds(&mut bounds);
    let scalars = extract_scalars(json);
    let quantization = crate::quantization::QuantizationConfig::parse(json).map_err(|e| {
        ConfigError::Quantization {
            path: source_path.to_path_buf(),
            source: e,
        }
    })?;
    let tie_word_embeddings = json
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let architectures: Vec<String> = json
        .get("architectures")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let rope_scaling = extract_rope_scaling(json);
    let rope_scaling_hash = json.get("rope_scaling").map(hash_json_value);

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
        rope_scaling,
        rope_scaling_hash,
    })
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
                .filter_map(|(k, v)| v.as_u64().map(|n| (k.clone(), n)))
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
    if !bounds.contains_key("head_dim")
        && let (Some(&hidden), Some(&heads)) =
            (bounds.get("hidden_size"), bounds.get("num_attention_heads"))
        && heads != 0
        && hidden.is_multiple_of(heads)
    {
        bounds.insert("head_dim".to_string(), hidden / heads);
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
        // (1 dense + 7 quant presets from `quantizations.json`:
        //  awq-gemm, bnb-nf4-dq, gptq-sym, gptq-sym-desc_act,
        //  ct-int4-sym, fp8-dynamic-per-tensor, fp8-static-per-tensor)
        //  = 96. Individual sizes don't always have real HF repos in
        //  every preset, but the compiler emits variants for all of
        //  them so fingerprint dispatch stays open-set at runtime.
        assert_eq!(
            configs.len(),
            96,
            "expected 96 Llama variants (12 bases × 8 variants)"
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
        // 11 dense × (1 dense + 6 presets: awq-gemm, bnb-nf4-dq,
        // ct-int4-sym, gptq-sym, fp8-dynamic-per-tensor,
        // fp8-static-per-tensor) = 77.
        assert_eq!(
            configs.len(),
            77,
            "expected 77 Qwen2 variants (11 bases × 7 variants)"
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
