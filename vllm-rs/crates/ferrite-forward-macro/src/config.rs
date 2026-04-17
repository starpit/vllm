// SPDX-License-Identifier: Apache-2.0
//! Phase 3: read `config.json` files from an `model_architectures/<arch>/`
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
        }
    }
}

impl std::error::Error for ConfigError {}

/// Load every `*.json` file in `dir` as a `ModelParams`. Results
/// are sorted alphabetically by file stem for build determinism.
pub fn load_dir(dir: &Path) -> Result<Vec<ModelParams>, ConfigError> {
    if !dir.is_dir() {
        return Err(ConfigError::NotADirectory(dir.to_path_buf()));
    }
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|source| ConfigError::Io {
            path: dir.to_path_buf(),
            source,
        })?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        // `weights.json` is the per-arch shape manifest loaded by
        // `weights_manifest::load_or_empty`, not a model config.
        // Filter it out of the per-model scan.
        .filter(|p| p.file_name().and_then(|s| s.to_str()) != Some("weights.json"))
        .collect();
    // Sort by file stem, not full PathBuf. `PathBuf::cmp` compares
    // byte-by-byte including the extension, which puts
    // `llama-3.2-1b-awq.json` before `llama-3.2-1b.json` (hyphen
    // 0x2D < dot 0x2E). Stem-sort keeps the natural
    // `llama-3.2-1b, llama-3.2-1b-awq` ordering and keeps every
    // existing config's position unchanged.
    paths.sort_by(|a, b| {
        a.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .cmp(b.file_stem().and_then(|s| s.to_str()).unwrap_or(""))
    });
    paths.iter().map(|p| load_file(p)).collect()
}

/// Load a single config.json.
pub fn load_file(path: &Path) -> Result<ModelParams, ConfigError> {
    let contents = fs::read_to_string(path).map_err(|source| ConfigError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let json: serde_json::Value =
        serde_json::from_str(&contents).map_err(|source| ConfigError::Json {
            path: path.to_path_buf(),
            source,
        })?;

    let source_stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| ConfigError::BadStem {
            path: path.to_path_buf(),
            reason: "file has no stem or non-UTF-8 stem",
        })?
        .to_string();
    let name = stem_to_ident(&source_stem).map_err(|reason| ConfigError::BadStem {
        path: path.to_path_buf(),
        reason,
    })?;
    let mut bounds = extract_bounds(&json);
    derive_implicit_bounds(&mut bounds);
    let scalars = extract_scalars(&json);
    let quantization = crate::quantization::QuantizationConfig::parse(&json).map_err(|e| {
        ConfigError::Quantization {
            path: path.to_path_buf(),
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

    Ok(ModelParams {
        name,
        source_stem,
        source_path: path.to_path_buf(),
        bounds,
        scalars,
        quantization,
        tie_word_embeddings,
        architectures,
    })
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

/// Every top-level *non-integer* number field becomes a scalar.
/// (Integer fields go to `bounds` via [`extract_bounds`]; `as_u64`
/// is checked first so an integer like `42` doesn't double-count
/// into `scalars` as `42.0`.)
fn extract_scalars(json: &serde_json::Value) -> BTreeMap<String, f64> {
    json.as_object()
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| {
                    if v.as_u64().is_some() {
                        None
                    } else {
                        v.as_f64().map(|n| (k.clone(), n))
                    }
                })
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

    /// Absolute path to `<repo>/model_architectures` from this crate's manifest dir.
    fn repo_model_archs() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("model_architectures")
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
        let dir = repo_model_archs().join("llama");
        let configs = load_dir(&dir).expect("load llama configs");

        // 9 Llama configs (405B gated, skipped) + smollm2-135m +
        // smollm2-360m (second size gives `probe-weights` cross-size
        // disambiguation) + llama-3.2-1b-awq (AWQ end-to-end slice) +
        // tinyllama-1.1b-gptq-desc-act (first desc_act=true GPTQ).
        assert_eq!(configs.len(), 13, "expected 13 Llama configs");

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
        let dir = repo_model_archs().join("qwen2");
        let configs = load_dir(&dir).expect("load qwen2 configs");
        assert_eq!(configs.len(), 13, "expected 13 Qwen2 configs");

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
        let dir = repo_model_archs().join("llama");
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

        // Integers go to bounds, not scalars.
        assert_eq!(cfg.bounds.get("num_hidden_layers"), Some(&16));
        assert_eq!(cfg.bounds.get("hidden_size"), Some(&2048));
        assert!(!cfg.scalars.contains_key("num_hidden_layers"));
        assert!(!cfg.scalars.contains_key("hidden_size"));

        // Non-integer numbers go to scalars.
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
        let dir = repo_model_archs().join("llama");
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
