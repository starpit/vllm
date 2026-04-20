// SPDX-License-Identifier: Apache-2.0
//! Per-architecture weights.json manifest.
//!
//! `model_architectures/<arch>/weights.json` declares the shape of
//! every weight tensor the forward pass consumes, expressed in bound
//! names from the arch's config.json (e.g. `"hidden_size"`,
//! `"head_dim * num_attention_heads"`). The file is produced by the
//! `probe-weights` bin in the `ferrite-forward` crate by probing a
//! representative checkpoint's safetensors headers.
//!
//! Shape inference uses this manifest to anchor weight shapes that
//! dataflow can't pin by itself (the canonical example: Qwen3/Gemma3
//! per-head `q_norm` of shape `[head_dim]`, where a naïve op-signature
//! inference would resolve to `[heads * head_dim]` via the upstream
//! gemm). Unlike the old `weight_conventions.rs` table, the manifest
//! is per-arch data, not baked-in global knowledge.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::shape::{Dim, Shape, canonical_mul};

/// Parsed `weights.json`. Keys are dotted weight paths (e.g.
/// `"self_attn.q_proj"`), values are symbolic shapes (`Vec<Dim>`).
///
/// An optional `__packed_splits__` top-level key declares checkpoints
/// that ship fused tensors on disk (e.g. Phi-3's `self_attn.qkv_proj.weight`,
/// `mlp.gate_up_proj.weight`) instead of the per-slice logical tensors
/// the DSL references. Each entry maps the packed prefix (unindexed —
/// the codegen fans over `model.layers.{L}.…`) to an ordered list of
/// target paths; each target must itself appear in `entries`, and its
/// first-dim shape formula is the per-slice row count. At load time the
/// codegen-emitted prelude calls
/// [`GpuWeights::synthesize_packed_row_split_sizes`] per (layer, packed
/// prefix), splitting the mmap into virtual siblings before any
/// `Linear::load` runs.
#[derive(Debug, Default, Clone)]
pub struct WeightsManifest {
    pub entries: BTreeMap<String, Shape>,
    pub packed_splits: BTreeMap<String, Vec<String>>,
}

impl WeightsManifest {
    /// Look up a weight path (as a slice of segments from
    /// `WeightTable::path`) in the manifest, joining segments with
    /// `.` to match the file's key format.
    pub fn lookup<S: AsRef<str>>(&self, path_segments: &[S]) -> Option<&Shape> {
        let dotted: Vec<&str> = path_segments.iter().map(|s| s.as_ref()).collect();
        self.entries.get(&dotted.join("."))
    }

    /// Empty manifest — used when an arch directory has no
    /// `weights.json` yet (arches predating the probe workflow).
    /// Shape inference falls back to its dataflow-only behavior.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Test helper: build a manifest from a literal list of
    /// (dotted-path, shape-formula-strings). Parses each formula
    /// via the same path as [`load_file`].
    #[cfg(test)]
    pub fn from_literals(entries: &[(&str, &[&str])]) -> Self {
        let mut out = BTreeMap::new();
        for (path, dims) in entries {
            let shape: Shape = dims.iter().map(|d| parse_dim(d).unwrap()).collect();
            out.insert((*path).to_string(), shape);
        }
        Self {
            entries: out,
            packed_splits: BTreeMap::new(),
        }
    }

    /// The convention set the original `weight_conventions.rs` table
    /// encoded — the "universal HF dense-attention + SwiGLU decoder"
    /// shapes every Llama/Qwen2/Gemma2/Granite test body exercises.
    /// Exposed for test call sites; real macro runs load a per-arch
    /// `weights.json` off disk instead.
    #[cfg(test)]
    pub fn llama_test_conventions() -> Self {
        Self::from_literals(&[
            ("embed_tokens", &["vocab_size", "hidden_size"]),
            ("lm_head", &["hidden_size", "vocab_size"]),
            (
                "self_attn.q_proj",
                &["hidden_size", "head_dim * num_attention_heads"],
            ),
            (
                "self_attn.k_proj",
                &["hidden_size", "head_dim * num_key_value_heads"],
            ),
            (
                "self_attn.v_proj",
                &["hidden_size", "head_dim * num_key_value_heads"],
            ),
            (
                "self_attn.o_proj",
                &["head_dim * num_attention_heads", "hidden_size"],
            ),
            ("input_layernorm", &["hidden_size"]),
            ("post_attention_layernorm", &["hidden_size"]),
            ("norm", &["hidden_size"]),
            ("mlp.gate_proj", &["hidden_size", "intermediate_size"]),
            ("mlp.up_proj", &["hidden_size", "intermediate_size"]),
            ("mlp.down_proj", &["intermediate_size", "hidden_size"]),
        ])
    }
}

/// Errors produced when loading a `weights.json`.
#[derive(Debug)]
pub enum ManifestError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    /// A shape value in the JSON wasn't an array of strings.
    BadShape {
        path: PathBuf,
        weight: String,
        reason: String,
    },
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "reading {}: {source}", path.display()),
            Self::Json { path, source } => write!(f, "parsing {}: {source}", path.display()),
            Self::BadShape {
                path,
                weight,
                reason,
            } => write!(
                f,
                "{}: bad shape for `{}`: {reason}",
                path.display(),
                weight
            ),
        }
    }
}

impl std::error::Error for ManifestError {}

/// Load the `weights.json` next to the arch's config files. If the
/// file doesn't exist, returns an empty manifest — caller decides
/// whether that's OK or an error.
pub fn load_or_empty(arch_dir: &Path) -> Result<WeightsManifest, ManifestError> {
    let path = arch_dir.join("weights.json");
    if !path.exists() {
        return Ok(WeightsManifest::empty());
    }
    load_file(&path)
}

/// Load and parse an explicit path. Strict: file must exist and
/// parse cleanly.
pub fn load_file(path: &Path) -> Result<WeightsManifest, ManifestError> {
    let bytes = fs::read(path).map_err(|source| ManifestError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let raw: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|source| ManifestError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    let obj = raw.as_object().ok_or_else(|| ManifestError::BadShape {
        path: path.to_path_buf(),
        weight: "<top-level>".into(),
        reason: "expected a JSON object".into(),
    })?;
    let mut entries: BTreeMap<String, Shape> = BTreeMap::new();
    let mut packed_splits: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (k, v) in obj {
        if k == "__packed_splits__" {
            let map = v.as_object().ok_or_else(|| ManifestError::BadShape {
                path: path.to_path_buf(),
                weight: k.clone(),
                reason: "`__packed_splits__` must be an object".into(),
            })?;
            for (packed, targets_v) in map {
                let arr = targets_v
                    .as_array()
                    .ok_or_else(|| ManifestError::BadShape {
                        path: path.to_path_buf(),
                        weight: packed.clone(),
                        reason: "packed-split value must be an array of target paths".into(),
                    })?;
                let mut targets = Vec::with_capacity(arr.len());
                for (i, item) in arr.iter().enumerate() {
                    let s = item.as_str().ok_or_else(|| ManifestError::BadShape {
                        path: path.to_path_buf(),
                        weight: packed.clone(),
                        reason: format!("target {i} is not a string"),
                    })?;
                    targets.push(s.to_string());
                }
                packed_splits.insert(packed.clone(), targets);
            }
            continue;
        }
        let shape = parse_shape(v).map_err(|reason| ManifestError::BadShape {
            path: path.to_path_buf(),
            weight: k.clone(),
            reason,
        })?;
        entries.insert(k.clone(), shape);
    }
    Ok(WeightsManifest {
        entries,
        packed_splits,
    })
}

/// Parse a JSON shape value — must be an array of strings, each
/// string either a bound name (`"hidden_size"`) or a `*`-separated
/// product (`"head_dim * num_attention_heads"`).
fn parse_shape(v: &serde_json::Value) -> Result<Shape, String> {
    let arr = v
        .as_array()
        .ok_or_else(|| "expected an array of shape strings".to_string())?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let s = item
            .as_str()
            .ok_or_else(|| format!("dim {i} is not a string"))?;
        out.push(parse_dim(s)?);
    }
    Ok(out)
}

fn parse_dim(s: &str) -> Result<Dim, String> {
    let factors: Vec<Dim> = s
        .split('*')
        .map(str::trim)
        .map(|f| {
            if f.is_empty() {
                return Err("empty factor in product".to_string());
            }
            if let Ok(n) = f.parse::<u64>() {
                Ok(Dim::Lit(n))
            } else {
                Ok(Dim::Bound(f.to_string()))
            }
        })
        .collect::<Result<_, _>>()?;
    match factors.len() {
        0 => Err("empty shape string".to_string()),
        1 => Ok(factors.into_iter().next().unwrap()),
        _ => Ok(canonical_mul(factors)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Dim;

    #[test]
    fn parses_single_bound() {
        assert_eq!(
            parse_dim("hidden_size").unwrap(),
            Dim::Bound("hidden_size".into())
        );
    }

    #[test]
    fn parses_product_of_two_bounds_canonically() {
        // Product form is canonicalised — bound names sorted
        // alphabetically so `head_dim * num_attention_heads` produces
        // the same `Dim` as `num_attention_heads * head_dim`.
        let a = parse_dim("head_dim * num_attention_heads").unwrap();
        let b = parse_dim("num_attention_heads * head_dim").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn parses_integer_literal_factor() {
        let d = parse_dim("2 * head_dim").unwrap();
        match d {
            Dim::Mul(fs) => {
                assert!(fs.iter().any(|f| matches!(f, Dim::Lit(2))));
                assert!(
                    fs.iter()
                        .any(|f| matches!(f, Dim::Bound(b) if b == "head_dim"))
                );
            }
            _ => panic!("expected Mul"),
        }
    }

    #[test]
    fn lookup_joins_segments_with_dots() {
        let mut m = WeightsManifest::empty();
        m.entries.insert(
            "self_attn.q_proj".into(),
            vec![Dim::Bound("hidden_size".into())],
        );
        let got = m.lookup(&["self_attn", "q_proj"]);
        assert!(got.is_some());
    }

    #[test]
    fn missing_file_returns_empty_manifest() {
        let tmp = std::env::temp_dir().join("ferrite_weights_manifest_test_nonexistent");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let m = load_or_empty(&tmp).unwrap();
        assert!(m.entries.is_empty());
    }
}
