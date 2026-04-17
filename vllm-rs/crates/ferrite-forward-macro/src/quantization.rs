// SPDX-License-Identifier: Apache-2.0
//! Parse `quantization_config` from HF `config.json` and resolve the
//! storage format of every weight the DSL references.
//!
//! Storage format is a property of the bits on disk — it's fixed by
//! the upstream HF repo and known at compile time from the model's
//! `config.json`. Compute kernels (marlin, cutlass_scaled_mm, ...)
//! are a separate decision the solver makes at solve time, over the
//! set of Impls whose `matches()` accept the source weights' storage
//! format. This module only handles the first half — what the
//! weights ARE.
//!
//! Today's coverage: `Dense` (no `quantization_config` present),
//! `Awq` (AutoAWQ's `quant_method: "awq"` shape), and `Gptq`
//! (GPTQ-for-LLMs / AutoGPTQ's `quant_method: "gptq"` shape). Both
//! honor `modules_to_not_convert`. FP8 / BnB / FP8-block land here
//! as they're wired up; each added variant must ship together with
//! its parser + per-format FieldLoad arm + at least one solver-
//! accepting Impl that emits the matching kernel call.

#![allow(dead_code)]

use crate::classified::{OpKind, Program, WeightId};
use crate::config::ModelParams;
use crate::fuf::{Fuf, FufInput};

/// How a single weight's bits are laid out on disk. Attached to
/// every `WeightId` via [`storage_format_for_weight`] once per
/// compiled model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StorageFormat {
    /// Raw bf16/fp16 matmul weights. No scales, no packing. The
    /// path every dense model takes today.
    Dense,
    /// AutoAWQ INT4 weights. Carries the config bits the loader
    /// and the kernel both need: `bits` (currently always 4 for
    /// AWQ-in-the-wild), `group_size` (scales per group-of-K
    /// rows), `zero_point` (has per-group zero points — false for
    /// symmetric), and the packing `version` (`Gemm` = AutoAWQ's
    /// default, `Gemv` = small-batch variant, `Marlin` = already
    /// repacked to marlin's tiled layout on disk).
    Awq {
        bits: u32,
        group_size: u32,
        zero_point: bool,
        version: AwqVersion,
    },
    /// GPTQ-for-LLMs / AutoGPTQ INT4 weights. `bits` (4 today for
    /// the ferrite path), `group_size` (scales per group-of-K rows,
    /// -1 → per-channel collapses to one group), `desc_act` (when
    /// `true`, the on-disk weights carry a `.g_idx` tensor encoding
    /// the activation-order permutation — the loader reads it and
    /// hands sort_indices to the Marlin repack kernel), `sym`
    /// (symmetric vs asymmetric quantization — AutoGPTQ's default
    /// is symmetric, in which case zero points are baked into the
    /// GPTQ uint4b8 scalar type and the `.qzeros` tensor is
    /// discarded at load).
    Gptq {
        bits: u32,
        group_size: u32,
        desc_act: bool,
        sym: bool,
    },
}

/// AutoAWQ's on-disk weight packing. The loader's repack behavior
/// depends on this — `Gemm`/`Gemv` need a runtime repack to
/// marlin's layout before a marlin kernel can consume them;
/// `Marlin` means the repack was done upstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AwqVersion {
    Gemm,
    Gemv,
    Marlin,
}

/// Top-level `quantization_config` section of HF's `config.json`,
/// when present.
#[derive(Clone, Debug)]
pub struct QuantizationConfig {
    pub method: QuantMethod,
    /// Suffix-match against a weight's dotted path — each entry
    /// that matches keeps that weight in `StorageFormat::Dense`.
    /// HF's canonical use: `["lm_head"]` to leave the tied or
    /// untied output projection unquantized.
    pub modules_to_not_convert: Vec<String>,
}

/// The parsed `quant_method` discriminator. One variant per HF-
/// supported method we handle; unknown methods produce
/// [`ParseError::UnsupportedMethod`] so new formats don't silently
/// degrade to `Dense`.
#[derive(Clone, Debug)]
pub enum QuantMethod {
    Awq {
        bits: u32,
        group_size: u32,
        zero_point: bool,
        version: AwqVersion,
    },
    Gptq {
        bits: u32,
        group_size: u32,
        desc_act: bool,
        sym: bool,
    },
}

/// Errors from [`QuantizationConfig::parse`]. All variants preserve
/// enough context to tell the user which field of which config was
/// malformed — surfaces as a `compile_error!` via
/// [`syn::Error`] in the macro drive.
#[derive(Debug)]
pub enum ParseError {
    /// `quantization_config` was present but not a JSON object.
    NotAnObject,
    /// `quant_method` missing or not a string.
    MissingMethod,
    /// `quant_method` value isn't one we understand. Lists the
    /// method we saw — "gptq", "fp8", "bitsandbytes", etc. — so
    /// the user can see what needs to be wired up next.
    UnsupportedMethod(String),
    /// A required numeric field (`bits`, `group_size`) was missing
    /// or out of range.
    BadField {
        field: &'static str,
        reason: &'static str,
    },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAnObject => write!(f, "`quantization_config` is not an object"),
            Self::MissingMethod => {
                write!(
                    f,
                    "`quantization_config.quant_method` missing or not a string"
                )
            }
            Self::UnsupportedMethod(m) => write!(
                f,
                "`quantization_config.quant_method = \"{m}\"` not yet supported by ferrite-forward",
            ),
            Self::BadField { field, reason } => {
                write!(f, "`quantization_config.{field}`: {reason}")
            }
        }
    }
}

impl std::error::Error for ParseError {}

impl QuantizationConfig {
    /// Parse the `quantization_config` subobject. Returns `Ok(None)`
    /// when the key is absent (the model is plain dense); `Ok(Some)`
    /// on a recognized method; `Err` on a present-but-malformed or
    /// not-yet-supported config.
    pub fn parse(root: &serde_json::Value) -> Result<Option<Self>, ParseError> {
        let Some(qc) = root.get("quantization_config") else {
            return Ok(None);
        };
        let obj = qc.as_object().ok_or(ParseError::NotAnObject)?;

        let method_str = obj
            .get("quant_method")
            .and_then(|v| v.as_str())
            .ok_or(ParseError::MissingMethod)?;

        let method = match method_str {
            "awq" => parse_awq(obj)?,
            "gptq" => parse_gptq(obj)?,
            other => return Err(ParseError::UnsupportedMethod(other.to_string())),
        };

        let modules_to_not_convert = obj
            .get("modules_to_not_convert")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Some(Self {
            method,
            modules_to_not_convert,
        }))
    }
}

fn parse_awq(obj: &serde_json::Map<String, serde_json::Value>) -> Result<QuantMethod, ParseError> {
    let bits = obj
        .get("bits")
        .and_then(|v| v.as_u64())
        .ok_or(ParseError::BadField {
            field: "bits",
            reason: "missing or not a u64",
        })? as u32;
    if bits != 4 {
        return Err(ParseError::BadField {
            field: "bits",
            reason: "AWQ ferrite path only handles 4-bit today",
        });
    }
    let group_size = obj
        .get("group_size")
        .and_then(|v| v.as_u64())
        .ok_or(ParseError::BadField {
            field: "group_size",
            reason: "missing or not a u64",
        })? as u32;
    if group_size == 0 {
        return Err(ParseError::BadField {
            field: "group_size",
            reason: "must be > 0",
        });
    }
    // `zero_point` defaults to `true` in AutoAWQ when unspecified.
    let zero_point = obj
        .get("zero_point")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let version = match obj
        .get("version")
        .and_then(|v| v.as_str())
        .unwrap_or("gemm")
    {
        "gemm" => AwqVersion::Gemm,
        "gemv" => AwqVersion::Gemv,
        "marlin" => AwqVersion::Marlin,
        other => {
            return Err(ParseError::BadField {
                field: "version",
                reason: match other {
                    "gemm" | "gemv" | "marlin" => unreachable!(),
                    _ => "unrecognized AWQ version (want gemm / gemv / marlin)",
                },
            });
        }
    };
    Ok(QuantMethod::Awq {
        bits,
        group_size,
        zero_point,
        version,
    })
}

fn parse_gptq(obj: &serde_json::Map<String, serde_json::Value>) -> Result<QuantMethod, ParseError> {
    let bits = obj
        .get("bits")
        .and_then(|v| v.as_u64())
        .ok_or(ParseError::BadField {
            field: "bits",
            reason: "missing or not a u64",
        })? as u32;
    if bits != 4 {
        return Err(ParseError::BadField {
            field: "bits",
            reason: "GPTQ ferrite path only handles 4-bit today",
        });
    }
    // `group_size` in AutoGPTQ is i64: positive integer for grouped, -1
    // for per-channel. The Marlin kernel collapses per-channel to a
    // single group internally; accept and pass through as 0 so the
    // loader routes it through `scale_perm_single`.
    let group_size_i64 =
        obj.get("group_size")
            .and_then(|v| v.as_i64())
            .ok_or(ParseError::BadField {
                field: "group_size",
                reason: "missing or not an i64",
            })?;
    let group_size: u32 = match group_size_i64 {
        -1 => 0,
        n if n > 0 => n as u32,
        _ => {
            return Err(ParseError::BadField {
                field: "group_size",
                reason: "must be -1 (per-channel) or > 0",
            });
        }
    };
    let desc_act = obj
        .get("desc_act")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    // AutoGPTQ defaults `sym` to `true`. When symmetric, zero points are
    // baked into Marlin's `uint4b8` scalar type and the on-disk `.qzeros`
    // tensor is consumed and discarded at load.
    let sym = obj.get("sym").and_then(|v| v.as_bool()).unwrap_or(true);
    Ok(QuantMethod::Gptq {
        bits,
        group_size,
        desc_act,
        sym,
    })
}

/// Resolve the storage format of a single weight by matching its
/// dotted path against the model's `quantization_config`.
///
/// Rules, in order (each returns `Dense` on success):
/// 1. Model has no `quantization_config` → every weight is `Dense`.
/// 2. The weight's dotted path ends with any entry in
///    `modules_to_not_convert` — HF's canonical exclusion list.
/// 3. `tie_word_embeddings: true` AND the weight is `lm_head` —
///    tied models have no `lm_head.*` safetensors to quantize; the
///    Gemm at the lm_head tile shares the embedding buffer via
///    [`crate::codegen::FieldLoad::LinearTiedToEmbedding`] and must
///    stay dense.
/// 4. The weight is `lm_head` AND the quant method is GPTQ —
///    AutoGPTQ's convention is that `lm_head` is never quantized
///    (safetensors ship it as plain `lm_head.weight` dense). This
///    applies whether or not `modules_to_not_convert` was set
///    explicitly; AutoGPTQ simply skips lm_head unconditionally.
/// 5. The weight is never consumed by a `Gemm` tile in the FUF.
///    Quant metadata only applies to matmul weights
///    (qweight/scales/qzeros/g_idx triples). `Embedding`, `RmsNorm`,
///    biases, etc. stay `Dense`.
///
/// Otherwise the method's parameters (bits/group_size/…) are carried
/// through into `StorageFormat::Awq{..}` or `StorageFormat::Gptq{..}`.
pub fn storage_format_for_weight(
    program: &Program,
    fuf: &Fuf,
    id: WeightId,
    model: &ModelParams,
) -> StorageFormat {
    let Some(ref qc) = model.quantization else {
        return StorageFormat::Dense;
    };

    let dotted: String = program.weights.path(id).join(".");
    for excl in &qc.modules_to_not_convert {
        if dotted.ends_with(excl) {
            return StorageFormat::Dense;
        }
    }

    // Tied lm_head: no on-disk `lm_head.*`; the codegen FieldLoad
    // shares the embedding buffer as a dense LinearLayer.
    if dotted == "lm_head" && model.tie_word_embeddings {
        return StorageFormat::Dense;
    }

    // AutoGPTQ convention: `lm_head` is never quantized, even when
    // untied and not listed in `modules_to_not_convert`. Safetensors
    // ship it as dense `lm_head.weight`. Without this rule, the
    // compiler emits a `load_gptq` on `lm_head` that can't find
    // `.qweight` and the model fails to load.
    if dotted == "lm_head" && matches!(qc.method, QuantMethod::Gptq { .. }) {
        return StorageFormat::Dense;
    }

    // AWQ/GPTQ metadata applies to matmul weights only — the
    // qweight/scales/[qzeros|g_idx] triple produces a 4-bit packed
    // weight that the marlin GEMM kernel consumes. Weights that
    // never reach a Gemm tile (Embedding, RmsNorm, biases) have no
    // quantized representation on disk and must stay Dense.
    let mut reached_by_gemm = false;
    for node in &fuf.nodes {
        if node.op != OpKind::Gemm {
            continue;
        }
        for input in &node.inputs {
            if let FufInput::Weight { id: wid, .. } = input
                && *wid == id
            {
                reached_by_gemm = true;
                break;
            }
        }
        if reached_by_gemm {
            break;
        }
    }
    if !reached_by_gemm {
        return StorageFormat::Dense;
    }

    match qc.method {
        QuantMethod::Awq {
            bits,
            group_size,
            zero_point,
            version,
        } => StorageFormat::Awq {
            bits,
            group_size,
            zero_point,
            version,
        },
        QuantMethod::Gptq {
            bits,
            group_size,
            desc_act,
            sym,
        } => StorageFormat::Gptq {
            bits,
            group_size,
            desc_act,
            sym,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(s: &str) -> serde_json::Value {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn absent_config_returns_none() {
        let v = json(r#"{"hidden_size": 2048}"#);
        assert!(QuantizationConfig::parse(&v).unwrap().is_none());
    }

    #[test]
    fn parses_awq_gemm_with_modules_to_not_convert() {
        let v = json(
            r#"{
                "quantization_config": {
                    "quant_method": "awq",
                    "bits": 4,
                    "group_size": 128,
                    "zero_point": true,
                    "version": "gemm",
                    "modules_to_not_convert": ["lm_head"]
                }
            }"#,
        );
        let qc = QuantizationConfig::parse(&v).unwrap().expect("some");
        assert!(matches!(
            qc.method,
            QuantMethod::Awq {
                bits: 4,
                group_size: 128,
                zero_point: true,
                version: AwqVersion::Gemm,
            }
        ));
        assert_eq!(qc.modules_to_not_convert, vec!["lm_head".to_string()]);
    }

    #[test]
    fn parses_awq_marlin_packed() {
        let v = json(
            r#"{
                "quantization_config": {
                    "quant_method": "awq",
                    "bits": 4,
                    "group_size": 128,
                    "version": "marlin"
                }
            }"#,
        );
        let qc = QuantizationConfig::parse(&v).unwrap().unwrap();
        assert!(matches!(
            qc.method,
            QuantMethod::Awq {
                version: AwqVersion::Marlin,
                ..
            }
        ));
        // Defaults: zero_point is `true` in AutoAWQ when omitted.
        assert!(matches!(
            qc.method,
            QuantMethod::Awq {
                zero_point: true,
                ..
            }
        ));
    }

    #[test]
    fn rejects_unknown_method() {
        let v = json(
            r#"{"quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic"}}"#,
        );
        let err = QuantizationConfig::parse(&v).unwrap_err();
        assert!(matches!(err, ParseError::UnsupportedMethod(ref s) if s == "fp8"));
    }

    #[test]
    fn parses_gptq_defaults() {
        // `desc_act` and `sym` default per AutoGPTQ: `sym = true`,
        // `desc_act = false`.
        let v = json(
            r#"{
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 128
                }
            }"#,
        );
        let qc = QuantizationConfig::parse(&v).unwrap().expect("some");
        assert!(matches!(
            qc.method,
            QuantMethod::Gptq {
                bits: 4,
                group_size: 128,
                desc_act: false,
                sym: true,
            }
        ));
    }

    #[test]
    fn parses_gptq_desc_act_asym() {
        let v = json(
            r#"{
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 128,
                    "desc_act": true,
                    "sym": false
                }
            }"#,
        );
        let qc = QuantizationConfig::parse(&v).unwrap().unwrap();
        assert!(matches!(
            qc.method,
            QuantMethod::Gptq {
                desc_act: true,
                sym: false,
                ..
            }
        ));
    }

    #[test]
    fn parses_gptq_per_channel_group_size_neg_one() {
        // AutoGPTQ uses `group_size: -1` to mean per-channel; we
        // collapse that to `group_size: 0` so the loader picks
        // `scale_perm_single` on the Marlin side.
        let v = json(
            r#"{
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": -1
                }
            }"#,
        );
        let qc = QuantizationConfig::parse(&v).unwrap().unwrap();
        assert!(matches!(qc.method, QuantMethod::Gptq { group_size: 0, .. }));
    }

    #[test]
    fn rejects_gptq_bad_group_size() {
        let v = json(
            r#"{"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 0}}"#,
        );
        assert!(matches!(
            QuantizationConfig::parse(&v),
            Err(ParseError::BadField {
                field: "group_size",
                ..
            })
        ));
    }

    #[test]
    fn rejects_non_4bit_gptq() {
        let v = json(
            r#"{"quantization_config": {"quant_method": "gptq", "bits": 8, "group_size": 128}}"#,
        );
        assert!(matches!(
            QuantizationConfig::parse(&v),
            Err(ParseError::BadField { field: "bits", .. })
        ));
    }

    #[test]
    fn rejects_non_4bit_awq() {
        let v = json(
            r#"{"quantization_config": {"quant_method": "awq", "bits": 8, "group_size": 128}}"#,
        );
        assert!(matches!(
            QuantizationConfig::parse(&v),
            Err(ParseError::BadField { field: "bits", .. })
        ));
    }

    #[test]
    fn rejects_bad_version() {
        let v = json(
            r#"{"quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128, "version": "bogus"}}"#,
        );
        assert!(matches!(
            QuantizationConfig::parse(&v),
            Err(ParseError::BadField {
                field: "version",
                ..
            })
        ));
    }
}
