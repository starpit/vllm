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
    ///
    /// `layout` selects between AutoGPTQ's native on-disk layout
    /// (`.qweight [K/8, N]`, `.scales [num_groups, N]`) and
    /// compressed-tensors' repack (`.weight_packed [N, K/8]`,
    /// `.weight_scale [N, num_groups]`). Same uint4b8 bits either
    /// way; the difference is tensor names + axis order, which the
    /// loader transposes back to GPTQ-native before repack.
    Gptq {
        bits: u32,
        group_size: u32,
        desc_act: bool,
        sym: bool,
        layout: GptqLayout,
    },
    /// BitsAndBytes 4-bit weights — `.weight` is U8 packed nibbles
    /// with a sibling `.absmax` / `.quant_state.bitsandbytes__nf4` /
    /// `.nested_absmax` / `.nested_quant_map` metadata. `quant_type`
    /// picks the NF4 / FP4 LUT; `blocksize` is the default scale
    /// granularity (the actual blocksize is read from the per-tensor
    /// quant_state JSON at load time — this is the fallback when
    /// it's absent).
    Bnb4 {
        quant_type: BnbQuantType,
        blocksize: u32,
    },
    /// FP8 (E4M3) quantized weights — `.weight` is `float8_e4m3fn`
    /// `[N, K]` with a sibling `.weight_scale` (per-tensor `[1]`,
    /// per-channel `[N]` or `[N, 1]`, or per-block `[ceil(N/bn),
    /// ceil(K/bk)]` for blockwise). `scheme` selects how activation
    /// quantization happens at forward time; `block_size` picks
    /// per-tensor (`None`) vs. blockwise scales. Matches Python
    /// vLLM's `Fp8Config` / `Fp8LinearMethod`.
    Fp8 {
        scheme: Fp8ActivationScheme,
        /// `Some([bn, bk])` for blockwise (DeepSeek `[128, 128]`);
        /// `None` for per-tensor / per-channel scales.
        block_size: Option<[u32; 2]>,
    },
    /// GGML/GGUF block-quantized weights. The on-disk per-tensor
    /// dtype (Q4_0, Q4_K, Q8_0, Q6_K, …) is heterogeneous within a
    /// single GGUF — different linears can ship with different
    /// quants — and is stored in the `.gguf` file's per-tensor
    /// header rather than `config.json`. So this `StorageFormat`
    /// carries no fields: at compile time we only know "GGUF
    /// quantized, runtime-dispatched"; the actual per-weight dtype
    /// is read from the GGUF file at load time and stored in the
    /// `GgmlStorage` the `GpuWeights` hands out via
    /// `take_quantized_linear`. The runtime `GgmlLinear` kernel
    /// dispatcher already handles dtype variation.
    Ggml,
    /// MLX-native affine INT4 quantization. Carries `bits`
    /// (currently always 4 for `mlx-community/*-4bit` checkpoints)
    /// and `group_size` (32, 64, or 128 — uniformly 64 across the
    /// canonical Llama/Qwen/Gemma/Mistral/Mixtral/Phi/DeepSeek 4bit
    /// repos sampled in P0). Per-output-row scales + biases are
    /// stored as `<prefix>.scales` / `<prefix>.biases` (`F16`
    /// dtype) alongside the packed weight `<prefix>.weight`
    /// (`U32` dtype, `[N, K/8]` shape for bits=4). The kernel
    /// reads scales/biases as `T_scale = half` and casts to float
    /// in registers — see `INT4_PARITY_PROBES.md` §7.
    Affine { bits: u32, group_size: u32 },
    /// NVIDIA ModelOpt NVFP4 weights. `.weight` is `uint8` `[N, K/2]`
    /// (two packed E2M1 codes per byte: low nibble = even element,
    /// high nibble = odd; each nibble is bit3 = sign, bits0‑2 =
    /// magnitude index into `[0,0.5,1,1.5,2,3,4,6]`). `.weight_scale`
    /// is `float8_e4m3` `[N, K/group_size]` per-block scale (stored
    /// linear on disk — the CUTLASS swizzle is a post-load step we
    /// don't apply for dequant-on-read). `.weight_scale_2` is an
    /// `f32` per-tensor global scale. The Metal load folds
    /// `f32(weight_scale)/weight_scale_2` into a single per-group F16
    /// scale (`Nvfp4Linear`), and the `qmv`/`qmm_t` kernels dequant on
    /// read — weight-only (activations stay bf16/f16). `group_size` is
    /// 16 for every ModelOpt NVFP4 checkpoint. Matches Python vLLM's
    /// `dequantize_to_dtype` (`nvfp4_emulation_utils.py`).
    Nvfp4 { group_size: u32 },
}

/// FP8 activation quantization scheme. Matches
/// `vllm_cuda::quant::Fp8ActivationScheme` one-for-one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8ActivationScheme {
    /// Per-token activation scale computed at forward time
    /// (`scaled_fp8_quant_dynamic`). Default on SM89+.
    Dynamic,
    /// Pre-calibrated per-tensor `input_scale` baked into the
    /// checkpoint (`scaled_fp8_quant_static`).
    Static,
}

/// GPTQ on-disk layout — decides which tensor names + axis order the
/// loader reads, and which fingerprint variant the compiled code
/// sniffs at runtime.
///
/// Both layouts hold the same INT4 packing (uint4b8 for symmetric).
/// `Qweight` is AutoGPTQ's native format. `WeightPacked` is Neural
/// Magic / RedHatAI's compressed-tensors repack — same bits, shapes
/// transposed to `[N, K/8]` / `[N, num_groups]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GptqLayout {
    /// AutoGPTQ native: `.qweight` at `[K/8, N]`, `.scales` at
    /// `[num_groups, N]`, optional `.qzeros` (consumed+discarded for
    /// symmetric), optional `.g_idx` (for `desc_act`).
    Qweight,
    /// compressed-tensors: `.weight_packed` at `[N, K/8]`,
    /// `.weight_scale` at `[N, num_groups]`, transposed before
    /// `gptq_repack_into` so the downstream kernel call is identical.
    /// CT doesn't emit `.g_idx` (it doesn't use activation ordering).
    WeightPacked,
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

/// BitsAndBytes 4-bit code table. NF4 = 16 quantiles of N(0,1)
/// rescaled to [-1, 1]; FP4 = E2M1 float values. Picked at macro-
/// expansion time from `bnb_4bit_quant_type` and folded into the
/// `Weights::load` prelude's `upload_bnb_code` call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BnbQuantType {
    NF4,
    FP4,
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
        layout: GptqLayout,
    },
    Bnb4 {
        quant_type: BnbQuantType,
        blocksize: u32,
    },
    Fp8 {
        scheme: Fp8ActivationScheme,
        block_size: Option<[u32; 2]>,
    },
    /// GGML/GGUF block-quantized weights. No knobs at compile time —
    /// per-tensor dtype is read from the GGUF file at load time.
    Ggml,
    /// MLX-native affine INT4 quantization. MLX-format `config.json`
    /// has no `quant_method` field — the parser detects this method
    /// via the absence of `quant_method` plus presence of `bits` +
    /// `group_size` directly under `quantization_config` (or the
    /// alternative top-level `quantization` key MLX also writes).
    ///
    /// `quantize_embed` distinguishes two `mlx_lm.convert` conventions
    /// for `tie_word_embeddings: false` checkpoints:
    ///   * `false` (default, `mlx-affine-b<bits>-g<gs>` preset) —
    ///     older convert behavior: `embed_tokens` is kept dense F16
    ///     `[vocab, hidden]`, only `lm_head` + transformer linears
    ///     are quantized. Verified against
    ///     `mlx-community/Meta-Llama-3-8B-Instruct-4bit`.
    ///   * `true` (`mlx-affine-b<bits>-g<gs>-qembed` preset) —
    ///     newer convert behavior: `embed_tokens` is also quantized
    ///     to the full affine triple `[vocab, hidden / pack_factor]`
    ///     U32 + `.scales`/`.biases` F16. Verified against
    ///     `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`.
    ///
    /// When `tie_word_embeddings: true`, the embed always ships
    /// quantized (because it's the same tensor as the quantized
    /// lm_head) and this flag has no effect.
    Affine {
        bits: u32,
        group_size: u32,
        quantize_embed: bool,
    },
    /// NVIDIA ModelOpt NVFP4. Detected via `quant_method: "modelopt"`
    /// (or no `quant_method` in a standalone `hf_quant_config.json`)
    /// with a `quant_algo` containing `"NVFP4"`. `group_size` is read
    /// from the config (default 16). Weight-only dequant-on-read on
    /// Metal; embeddings + `lm_head` stay dense (ModelOpt keeps them
    /// high precision and usually lists `lm_head` in `exclude_modules`).
    Nvfp4 { group_size: u32 },
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
        // MLX-format checkpoints write the same payload under both
        // `quantization_config` (HF-canonical) and `quantization`
        // (MLX-only) — accept either. HF-transformers checkpoints
        // only write `quantization_config`.
        let qc_value = root
            .get("quantization_config")
            .or_else(|| root.get("quantization"));
        let Some(qc) = qc_value else {
            return Ok(None);
        };
        let obj = qc.as_object().ok_or(ParseError::NotAnObject)?;

        // HF-canonical configs carry `quant_method`; MLX-affine
        // checkpoints have no `quant_method`, just `{bits,
        // group_size}` at the root of the section. Distinguish on
        // presence of `quant_method` first; fall through to MLX-
        // affine only when neither path matches.
        let method = if let Some(method_str) = obj.get("quant_method").and_then(|v| v.as_str()) {
            match method_str {
                "awq" => parse_awq(obj)?,
                "gptq" => parse_gptq(obj)?,
                "compressed-tensors" => parse_compressed_tensors(obj)?,
                "bitsandbytes" => parse_bitsandbytes(obj)?,
                "fp8" => parse_fp8(obj)?,
                // ModelOpt NVFP4 (and its `modelopt_fp4` / bare `nvfp4`
                // aliases). The `quant_algo` discriminator distinguishes
                // NVFP4 from ModelOpt FP8 — `parse_nvfp4` rejects the
                // latter so it doesn't silently degrade.
                "modelopt" | "modelopt_fp4" | "nvfp4" => parse_nvfp4(obj)?,
                // GGML/GGUF block-quantized — no compile-time knobs.
                "ggml" | "gguf" => QuantMethod::Ggml,
                other => return Err(ParseError::UnsupportedMethod(other.to_string())),
            }
        } else if obj.contains_key("quant_algo")
            || obj
                .get("quantization")
                .and_then(|q| q.get("quant_algo"))
                .is_some()
        {
            // ModelOpt ships a standalone `hf_quant_config.json` whose
            // `quantization` subobject carries `quant_algo` with no
            // `quant_method` field. Route it to the NVFP4 parser.
            parse_nvfp4(obj)?
        } else if obj.contains_key("bits") && obj.contains_key("group_size") {
            parse_affine_no_method(obj)?
        } else {
            return Err(ParseError::MissingMethod);
        };

        // AWQ/GPTQ carry `modules_to_not_convert`; compressed-tensors
        // carries the same list under `ignore`; ModelOpt NVFP4 uses
        // `exclude_modules`, possibly nested under `quantization`. Accept
        // whichever the upstream repo shipped.
        let modules_to_not_convert = obj
            .get("modules_to_not_convert")
            .or_else(|| obj.get("ignore"))
            .or_else(|| obj.get("exclude_modules"))
            .or_else(|| {
                obj.get("quantization")
                    .and_then(|q| q.get("exclude_modules"))
            })
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

/// Parse the MLX-native affine quantization payload — `{bits,
/// group_size}` directly under `quantization_config` (or
/// `quantization`), with no `quant_method` field. MLX writes this
/// shape on every `mlx-community/*-4bit` repo on HF.
fn parse_affine_no_method(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<QuantMethod, ParseError> {
    let bits = obj
        .get("bits")
        .and_then(|v| v.as_u64())
        .ok_or(ParseError::BadField {
            field: "bits",
            reason: "missing or not a u64",
        })? as u32;
    let group_size = obj
        .get("group_size")
        .and_then(|v| v.as_u64())
        .ok_or(ParseError::BadField {
            field: "group_size",
            reason: "missing or not a u64",
        })? as u32;
    if bits != 4 {
        return Err(ParseError::BadField {
            field: "bits",
            reason: "MLX-affine ferrite path only handles 4-bit today",
        });
    }
    if !matches!(group_size, 32 | 64 | 128) {
        return Err(ParseError::BadField {
            field: "group_size",
            reason: "MLX-affine supports group_size ∈ {32, 64, 128}",
        });
    }
    // Optional discriminator emitted by the
    // `mlx-affine-b<bits>-g<gs>-qembed` preset. Default false — the
    // original preset shape (embed_tokens dense F16 on untied
    // checkpoints, e.g. Llama-3-8B-4bit) keeps the prior default.
    let quantize_embed = obj
        .get("quant_embed")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Ok(QuantMethod::Affine {
        bits,
        group_size,
        quantize_embed,
    })
}

/// Parse a ModelOpt NVFP4 `quantization_config` / `hf_quant_config.json`
/// payload. The discriminating field is `quant_algo`, which lives either
/// directly under the section or nested under a `quantization` subobject
/// (ModelOpt's standalone-file shape). We require it to contain `"NVFP4"`;
/// a ModelOpt FP8 checkpoint (`quant_algo: "FP8"`) is a different path and
/// is rejected here rather than silently mishandled. `group_size` defaults
/// to 16 — the value every ModelOpt NVFP4 checkpoint ships.
fn parse_nvfp4(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<QuantMethod, ParseError> {
    // `quant_algo` may be top-level or nested under `quantization`.
    let nested = obj.get("quantization").and_then(|q| q.as_object());
    let lookup = |key: &str| -> Option<&serde_json::Value> {
        obj.get(key).or_else(|| nested.and_then(|n| n.get(key)))
    };

    let quant_algo = lookup("quant_algo")
        .and_then(|v| v.as_str())
        .ok_or(ParseError::BadField {
            field: "quant_algo",
            reason: "missing or not a string",
        })?;
    if !quant_algo.to_uppercase().contains("NVFP4") {
        return Err(ParseError::BadField {
            field: "quant_algo",
            reason: "ferrite NVFP4 path only handles `NVFP4` quant_algo (ModelOpt FP8 unsupported)",
        });
    }

    let group_size = lookup("group_size").and_then(|v| v.as_u64()).unwrap_or(16) as u32;
    if group_size == 0 {
        return Err(ParseError::BadField {
            field: "group_size",
            reason: "must be > 0",
        });
    }

    Ok(QuantMethod::Nvfp4 { group_size })
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
        layout: GptqLayout::Qweight,
    })
}

/// Parse `quantization_config.quant_method == "compressed-tensors"`.
///
/// Neural Magic / RedHatAI's compressed-tensors format is a union of
/// several storage formats selected by the tuple
/// `(config_groups[*].weights.type, num_bits)`. Today ferrite handles
/// only the INT4 group (`"int"` / `4`), which maps directly to GPTQ's
/// uint4b8 packing on disk — just with `.weight_packed` /
/// `.weight_scale` tensor names and `[N, K/8]` / `[N, num_groups]`
/// shapes that the loader transposes back. FP8 groups (`"float"` /
/// `8`) land in a later slice via their own `QuantMethod` variant.
///
/// Activation ordering (`desc_act`) is never used by compressed-
/// tensors, so we pin it to `false`. Zero-point handling piggybacks
/// on `symmetric` exactly like true GPTQ.
fn parse_compressed_tensors(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<QuantMethod, ParseError> {
    let config_groups =
        obj.get("config_groups")
            .and_then(|v| v.as_object())
            .ok_or(ParseError::BadField {
                field: "config_groups",
                reason: "missing or not an object",
            })?;
    let group = config_groups.values().next().ok_or(ParseError::BadField {
        field: "config_groups",
        reason: "empty — no quantization group declared",
    })?;
    let weights = group.get("weights").ok_or(ParseError::BadField {
        field: "config_groups.*.weights",
        reason: "missing weights spec",
    })?;

    let weight_type = weights.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let weight_bits =
        weights
            .get("num_bits")
            .and_then(|v| v.as_u64())
            .ok_or(ParseError::BadField {
                field: "config_groups.*.weights.num_bits",
                reason: "missing or not a u64",
            })? as u32;

    // INT4 → GPTQ-compatible uint4b8 packing (compressed-tensors
    // repack layout). FP8 + other formats land in later slices.
    if weight_type == "int" && weight_bits == 4 {
        let group_size = weights
            .get("group_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as u32;
        let sym = weights
            .get("symmetric")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        // `actorder` carries compressed-tensors' version of desc_act.
        // "null" / absent = no activation reordering (TinyLlama-W4A16-e2e);
        // "group" / "weight" = activation-ordered, ships a non-identity
        // `.weight_g_idx` tensor (RedHatAI/Qwen2.5-*-quantized.w4a16 and
        // other desc_act=true-via-AutoGPTQ-then-repacked checkpoints).
        // Without this the loader reads the permuted weight but skips
        // the permutation → garbage output.
        let desc_act = matches!(
            weights.get("actorder").and_then(|v| v.as_str()),
            Some("group") | Some("weight")
        );
        return Ok(QuantMethod::Gptq {
            bits: 4,
            group_size,
            desc_act,
            sym,
            layout: GptqLayout::WeightPacked,
        });
    }

    // FP8 float-quantized → same on-disk layout as `quant_method: "fp8"`.
    // Neural Magic / RedHatAI FP8 checkpoints. The activation scheme is
    // derived from `input_activations.dynamic`: `true` (or absent, the
    // common case for per-token quantization) maps to `Dynamic`, `false`
    // to `Static`. Blockwise scales carry a `weights.block_structure`
    // `[bn, bk]` array (e.g., `[128, 128]` for DeepSeek-style blockwise).
    if weight_type == "float" && weight_bits == 8 {
        let scheme = match group
            .get("input_activations")
            .and_then(|v| v.get("dynamic"))
            .and_then(|v| v.as_bool())
        {
            Some(false) => Fp8ActivationScheme::Static,
            Some(true) | None => Fp8ActivationScheme::Dynamic,
        };
        let block_size = match weights.get("block_structure") {
            None | Some(serde_json::Value::Null) => None,
            Some(v) => {
                let arr = v.as_array().ok_or(ParseError::BadField {
                    field: "config_groups.*.weights.block_structure",
                    reason: "must be a two-element array",
                })?;
                if arr.len() != 2 {
                    return Err(ParseError::BadField {
                        field: "config_groups.*.weights.block_structure",
                        reason: "must have exactly two elements",
                    });
                }
                let bn = arr[0].as_u64().ok_or(ParseError::BadField {
                    field: "config_groups.*.weights.block_structure[0]",
                    reason: "not a u64",
                })? as u32;
                let bk = arr[1].as_u64().ok_or(ParseError::BadField {
                    field: "config_groups.*.weights.block_structure[1]",
                    reason: "not a u64",
                })? as u32;
                if bn == 0 || bk == 0 {
                    return Err(ParseError::BadField {
                        field: "config_groups.*.weights.block_structure",
                        reason: "block dimensions must be > 0",
                    });
                }
                Some([bn, bk])
            }
        };
        return Ok(QuantMethod::Fp8 { scheme, block_size });
    }

    Err(ParseError::UnsupportedMethod(format!(
        "compressed-tensors weights type={weight_type:?} num_bits={weight_bits} \
         (ferrite handles INT4 and FP8 today; other types land in follow-on slices)",
    )))
}

/// Parse `quantization_config.quant_method == "bitsandbytes"`.
///
/// bitsandbytes ships 4-bit weights as U8-packed nibbles with a
/// sibling absmax tensor (double-quantized via a 256-entry
/// lookup table) + a per-tensor JSON `quant_state.bitsandbytes__nf4`
/// blob carrying the block size. Today ferrite handles only the
/// 4-bit path — `load_in_8bit` hits `UnsupportedMethod` so the bnb
/// 8-bit slice stays unimplemented rather than silently downgrading.
fn parse_bitsandbytes(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<QuantMethod, ParseError> {
    let load_4bit = obj
        .get("load_in_4bit")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !load_4bit {
        return Err(ParseError::UnsupportedMethod(
            "bitsandbytes with load_in_4bit=false (8-bit not yet supported)".to_string(),
        ));
    }
    let qt = obj
        .get("bnb_4bit_quant_type")
        .and_then(|v| v.as_str())
        .unwrap_or("nf4");
    let quant_type = match qt {
        "fp4" => BnbQuantType::FP4,
        "nf4" => BnbQuantType::NF4,
        other => {
            return Err(ParseError::BadField {
                field: "bnb_4bit_quant_type",
                reason: match other {
                    "nf4" | "fp4" => unreachable!(),
                    _ => "unrecognized bnb quant_type (want nf4 / fp4)",
                },
            });
        }
    };
    // bitsandbytes' blocksize default is 64. Per-tensor quant_state
    // JSON on disk overrides this at load time; the bound here is
    // only the compile-time fallback.
    Ok(QuantMethod::Bnb4 {
        quant_type,
        blocksize: 64,
    })
}

/// Parse `quantization_config.quant_method == "fp8"`.
///
/// Matches Python vLLM's `Fp8Config.from_config()` / HF `neuralmagic` /
/// `RedHatAI` FP8 checkpoint schemas. Fields:
/// - `activation_scheme`: `"dynamic"` (default, per-token at runtime) or
///   `"static"` (pre-calibrated per-tensor `input_scale` in the
///   safetensors).
/// - `weight_block_size`: optional `[bn, bk]` two-element array. Present
///   on DeepSeek-style block-quantized checkpoints (typically
///   `[128, 128]`); absent on per-tensor / per-channel checkpoints.
///
/// Other Python `Fp8Config` fields (`is_checkpoint_fp8_serialized`,
/// `ignored_layers`) are load-time concerns: the fingerprint + the
/// `modules_to_not_convert` / `ignore` list handle them. This parser
/// only pulls the shape-bearing knobs the solver and codegen need.
fn parse_fp8(obj: &serde_json::Map<String, serde_json::Value>) -> Result<QuantMethod, ParseError> {
    let scheme = match obj
        .get("activation_scheme")
        .and_then(|v| v.as_str())
        .unwrap_or("dynamic")
    {
        "dynamic" => Fp8ActivationScheme::Dynamic,
        "static" => Fp8ActivationScheme::Static,
        other => {
            return Err(ParseError::BadField {
                field: "activation_scheme",
                reason: match other {
                    "dynamic" | "static" => unreachable!(),
                    _ => "unrecognized FP8 activation_scheme (want dynamic / static)",
                },
            });
        }
    };
    let block_size = match obj.get("weight_block_size") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => {
            let arr = v.as_array().ok_or(ParseError::BadField {
                field: "weight_block_size",
                reason: "must be a two-element array",
            })?;
            if arr.len() != 2 {
                return Err(ParseError::BadField {
                    field: "weight_block_size",
                    reason: "must have exactly two elements",
                });
            }
            let bn = arr[0].as_u64().ok_or(ParseError::BadField {
                field: "weight_block_size[0]",
                reason: "not a u64",
            })? as u32;
            let bk = arr[1].as_u64().ok_or(ParseError::BadField {
                field: "weight_block_size[1]",
                reason: "not a u64",
            })? as u32;
            if bn == 0 || bk == 0 {
                return Err(ParseError::BadField {
                    field: "weight_block_size",
                    reason: "block dimensions must be > 0",
                });
            }
            Some([bn, bk])
        }
    };
    Ok(QuantMethod::Fp8 { scheme, block_size })
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
        if std::env::var("FERRITE_GGUF_BUILD_TRACE").is_ok() && model.source_stem.contains("ggml") {
            eprintln!(
                "[ggml-build] storage_format_for_weight model={} weight_id={id:?} → Dense (no quantization config)",
                model.source_stem
            );
        }
        return StorageFormat::Dense;
    };
    if std::env::var("FERRITE_GGUF_BUILD_TRACE").is_ok() && model.source_stem.contains("ggml") {
        eprintln!(
            "[ggml-build] storage_format_for_weight model={} weight_id={id:?} method={:?}",
            model.source_stem, qc.method
        );
    }

    let dotted: String = program.weights.path(id).join(".");
    for excl in &qc.modules_to_not_convert {
        if dotted.ends_with(excl) {
            return StorageFormat::Dense;
        }
    }

    // Tied lm_head: no on-disk `lm_head.*`; the codegen FieldLoad
    // shares the embedding buffer.
    //
    // For AWQ/GPTQ/Bnb/Fp8/Ggml the shared buffer is dense (those
    // formats keep the embedding in fp), so the lm_head Gemm sees
    // a Dense tensor.
    //
    // For MLX-affine the on-disk embed_tokens is itself quantized
    // (`model.embed_tokens.{weight,scales,biases}` is the affine
    // triple — confirmed across every mlx-community 4bit repo
    // sampled in P0). Pre-P6, the load path CPU-dequantized the
    // embedding (`Embedding::load_affine_dequant` → BF16 Dense),
    // so the tied lm_head saw a Dense tensor and routed through
    // MetalGemmImpl. P6 lifts the embedding to forward-time gather
    // + dequant (`Instruction::AffineEmbed`), which means the
    // embed buffer triple stays quantized at runtime — the tied
    // lm_head must now also see Affine storage so the solver
    // picks `MetalAffineQmmImpl` and the codegen
    // `LinearTiedToEmbedding { affine: Some((gs, bits)) }` arm
    // emits `LinearLayer::AffineQuant(...)` sharing the embed's
    // packed buffers.
    if dotted == "lm_head" && model.tie_word_embeddings {
        if let QuantMethod::Affine {
            bits, group_size, ..
        } = qc.method
        {
            return StorageFormat::Affine { bits, group_size };
        }
        return StorageFormat::Dense;
    }

    // MLX-affine convention for `tie_word_embeddings: false`:
    //   * Older `mlx_lm.convert` keeps `embed_tokens` dense F16
    //     `[vocab, hidden]` and only quantizes `lm_head` + the
    //     transformer linears. Verified against
    //     `mlx-community/Meta-Llama-3-8B-Instruct-4bit`.
    //   * Newer convert runs ALSO quantize the embed —
    //     `model.embed_tokens.{weight,scales,biases}` is the full
    //     affine triple. Verified against
    //     `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`.
    //
    // The choice is per-checkpoint and the macro can't see disk at
    // expansion time, so it's encoded in the preset:
    //   * `mlx-affine-b<bits>-g<gs>`        → quantize_embed=false (dense)
    //   * `mlx-affine-b<bits>-g<gs>-qembed` → quantize_embed=true  (Affine)
    //
    // The tied path keeps the shared embed buffer Affine-packed (its
    // dotted-name lm_head rule already returns Affine above), so the
    // rule below only fires for untied checkpoints; tied checkpoints
    // ignore `quantize_embed` entirely.
    if dotted == "embed_tokens"
        && !model.tie_word_embeddings
        && let QuantMethod::Affine {
            bits,
            group_size,
            quantize_embed,
        } = qc.method
    {
        return if quantize_embed {
            StorageFormat::Affine { bits, group_size }
        } else {
            StorageFormat::Dense
        };
    }

    // AutoGPTQ convention: `lm_head` is never quantized, even when
    // untied and not listed in `modules_to_not_convert`. Safetensors
    // ship it as dense `lm_head.weight`. Without this rule, the
    // compiler emits a `load_gptq` on `lm_head` that can't find
    // `.qweight` and the model fails to load. Covers both AutoGPTQ
    // native and compressed-tensors INT4 (both fall through the
    // `QuantMethod::Gptq` arm regardless of on-disk layout).
    if dotted == "lm_head" && matches!(qc.method, QuantMethod::Gptq { .. }) {
        return StorageFormat::Dense;
    }

    // bitsandbytes convention: `lm_head` is likewise never
    // quantized. unsloth / official bitsandbytes checkpoints ship
    // `lm_head.weight` as plain bf16/fp16 alongside all the
    // 4-bit-packed decoder layers.
    if dotted == "lm_head" && matches!(qc.method, QuantMethod::Bnb4 { .. }) {
        return StorageFormat::Dense;
    }

    // FP8 convention: `lm_head` is never quantized. neuralmagic /
    // RedHatAI FP8 checkpoints ship `lm_head.weight` as BF16/F16
    // alongside the FP8 decoder layers. Same rationale as GPTQ —
    // the output projection is tied or kept high-precision for
    // sampling quality.
    if dotted == "lm_head" && matches!(qc.method, QuantMethod::Fp8 { .. }) {
        return StorageFormat::Dense;
    }

    // ModelOpt NVFP4 convention: `lm_head` (and `embed_tokens`) stay
    // high precision — ModelOpt keeps them dense and usually lists
    // `lm_head` in `exclude_modules`. Belt-and-suspenders rule in case a
    // checkpoint omits it from the list. (embed_tokens needs no rule: it
    // only feeds an Embed op, which isn't a matmul consumer for NVFP4, so
    // it already falls through to Dense below.)
    if dotted == "lm_head" && matches!(qc.method, QuantMethod::Nvfp4 { .. }) {
        return StorageFormat::Dense;
    }

    // GGUF convention: the on-disk loader (`GgufGpuWeights::load`)
    // dequantizes `output.weight` (lm_head) and ships it via
    // `take_gguf_dense`. The codegen FieldLoad arm for GGUF lm_head
    // uses the dense path, so the storage format must agree.
    if dotted == "lm_head" && matches!(qc.method, QuantMethod::Ggml) {
        return StorageFormat::Dense;
    }

    // AWQ/GPTQ metadata applies to matmul weights only — the
    // qweight/scales/[qzeros|g_idx] triple produces a 4-bit packed
    // weight that the marlin GEMM kernel consumes. Weights that
    // never reach a Gemm tile (Embedding, RmsNorm, biases) have no
    // quantized representation on disk and must stay Dense.
    //
    // `OpKind::Moe` carries one logical `moe[layer]` weight whose
    // underlying experts are matmul-quantizable in V3/Kimi K2 FP8
    // checkpoints, so it counts as reaching a matmul.
    //
    // MLX-affine additionally quantizes the embedding table on the
    // disk (`model.embed_tokens.{weight,scales,biases}` triple —
    // verified across every mlx-community 4bit repo in P0 except
    // the Gemma-3-MM vision multimodal). For Affine, an Embedding
    // op consumer is also a quantized consumer.
    let mut reached_by_matmul = false;
    let affine_method = matches!(qc.method, QuantMethod::Affine { .. });
    for node in &fuf.nodes {
        let consumer = node.op == OpKind::Gemm
            || node.op == OpKind::Moe
            || (affine_method && node.op == OpKind::Embed);
        if !consumer {
            continue;
        }
        for input in &node.inputs {
            if let FufInput::Weight { id: wid, .. } = input
                && *wid == id
            {
                reached_by_matmul = true;
                break;
            }
        }
        if reached_by_matmul {
            break;
        }
    }
    if !reached_by_matmul {
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
            layout,
        } => StorageFormat::Gptq {
            bits,
            group_size,
            desc_act,
            sym,
            layout,
        },
        QuantMethod::Bnb4 {
            quant_type,
            blocksize,
        } => StorageFormat::Bnb4 {
            quant_type,
            blocksize,
        },
        QuantMethod::Fp8 { scheme, block_size } => StorageFormat::Fp8 { scheme, block_size },
        QuantMethod::Ggml => StorageFormat::Ggml,
        QuantMethod::Affine {
            bits,
            group_size,
            quantize_embed: _,
        } => StorageFormat::Affine { bits, group_size },
        QuantMethod::Nvfp4 { group_size } => StorageFormat::Nvfp4 { group_size },
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
        let v = json(r#"{"quantization_config": {"quant_method": "smoothquant"}}"#);
        let err = QuantizationConfig::parse(&v).unwrap_err();
        assert!(matches!(err, ParseError::UnsupportedMethod(ref s) if s == "smoothquant"));
    }

    #[test]
    fn parses_fp8_dynamic_per_tensor() {
        // Default activation_scheme is "dynamic"; absent
        // `weight_block_size` → per-tensor.
        let v = json(
            r#"{
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic"
                }
            }"#,
        );
        let qc = QuantizationConfig::parse(&v).unwrap().expect("some");
        assert!(matches!(
            qc.method,
            QuantMethod::Fp8 {
                scheme: Fp8ActivationScheme::Dynamic,
                block_size: None,
            }
        ));
    }

    #[test]
    fn parses_fp8_static_per_tensor() {
        let v = json(
            r#"{
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "static",
                    "ignored_layers": ["lm_head"]
                }
            }"#,
        );
        let qc = QuantizationConfig::parse(&v).unwrap().unwrap();
        assert!(matches!(
            qc.method,
            QuantMethod::Fp8 {
                scheme: Fp8ActivationScheme::Static,
                block_size: None,
            }
        ));
    }

    #[test]
    fn parses_fp8_block_128x128() {
        // DeepSeek-V3 style blockwise FP8.
        let v = json(
            r#"{
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                    "weight_block_size": [128, 128]
                }
            }"#,
        );
        let qc = QuantizationConfig::parse(&v).unwrap().unwrap();
        assert!(matches!(
            qc.method,
            QuantMethod::Fp8 {
                scheme: Fp8ActivationScheme::Dynamic,
                block_size: Some([128, 128]),
            }
        ));
    }

    #[test]
    fn rejects_fp8_bad_block_size() {
        let v =
            json(r#"{"quantization_config": {"quant_method": "fp8", "weight_block_size": [128]}}"#);
        assert!(matches!(
            QuantizationConfig::parse(&v),
            Err(ParseError::BadField {
                field: "weight_block_size",
                ..
            })
        ));
    }

    #[test]
    fn rejects_fp8_unknown_activation_scheme() {
        let v = json(
            r#"{"quantization_config": {"quant_method": "fp8", "activation_scheme": "channelwise"}}"#,
        );
        assert!(matches!(
            QuantizationConfig::parse(&v),
            Err(ParseError::BadField {
                field: "activation_scheme",
                ..
            })
        ));
    }

    #[test]
    fn parses_gptq_defaults() {
        // `desc_act` and `sym` default per AutoGPTQ: `sym = true`,
        // `desc_act = false`. `layout` is AutoGPTQ's native `.qweight`
        // for any `quant_method: "gptq"`.
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
                layout: GptqLayout::Qweight,
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
                layout: GptqLayout::Qweight,
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
    fn parses_compressed_tensors_int4() {
        // Neural Magic / RedHatAI compressed-tensors INT4: honored
        // as GPTQ with `layout: WeightPacked`. `ignore` is CT's
        // `modules_to_not_convert` spelling.
        let v = json(
            r#"{
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "ignore": ["lm_head"],
                    "config_groups": {
                        "group_0": {
                            "weights": {
                                "type": "int",
                                "num_bits": 4,
                                "group_size": 128,
                                "symmetric": true
                            }
                        }
                    }
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
                layout: GptqLayout::WeightPacked,
            }
        ));
        assert_eq!(qc.modules_to_not_convert, vec!["lm_head".to_string()]);
    }

    #[test]
    fn rejects_compressed_tensors_unsupported_type() {
        // INT8 and other yet-unsupported CT variants reject so they
        // don't silently fall into an unrelated path.
        let v = json(
            r#"{
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "g": {
                            "weights": {"type": "int", "num_bits": 8}
                        }
                    }
                }
            }"#,
        );
        assert!(matches!(
            QuantizationConfig::parse(&v),
            Err(ParseError::UnsupportedMethod(_))
        ));
    }

    #[test]
    fn parses_compressed_tensors_fp8_dynamic_per_tensor() {
        // RedHatAI / Neural Magic FP8-dynamic shape: `type: "float"`,
        // `num_bits: 8`, `input_activations.dynamic: true`.
        let v = json(
            r#"{
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "format": "float-quantized",
                    "ignore": ["lm_head"],
                    "config_groups": {
                        "group_0": {
                            "weights": {
                                "type": "float",
                                "num_bits": 8,
                                "symmetric": true,
                                "strategy": "channel"
                            },
                            "input_activations": {
                                "type": "float",
                                "num_bits": 8,
                                "dynamic": true,
                                "strategy": "token"
                            },
                            "targets": ["Linear"]
                        }
                    }
                }
            }"#,
        );
        let qc = QuantizationConfig::parse(&v).unwrap().expect("some");
        assert!(matches!(
            qc.method,
            QuantMethod::Fp8 {
                scheme: Fp8ActivationScheme::Dynamic,
                block_size: None,
            }
        ));
        assert_eq!(qc.modules_to_not_convert, vec!["lm_head".to_string()]);
    }

    #[test]
    fn parses_compressed_tensors_fp8_static() {
        let v = json(
            r#"{
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "g": {
                            "weights": {"type": "float", "num_bits": 8},
                            "input_activations": {"type": "float", "num_bits": 8, "dynamic": false}
                        }
                    }
                }
            }"#,
        );
        let qc = QuantizationConfig::parse(&v).unwrap().expect("some");
        assert!(matches!(
            qc.method,
            QuantMethod::Fp8 {
                scheme: Fp8ActivationScheme::Static,
                block_size: None,
            }
        ));
    }

    #[test]
    fn parses_compressed_tensors_fp8_blockwise() {
        let v = json(
            r#"{
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "g": {
                            "weights": {
                                "type": "float",
                                "num_bits": 8,
                                "strategy": "block",
                                "block_structure": [128, 128]
                            }
                        }
                    }
                }
            }"#,
        );
        let qc = QuantizationConfig::parse(&v).unwrap().expect("some");
        assert!(matches!(
            qc.method,
            QuantMethod::Fp8 {
                scheme: Fp8ActivationScheme::Dynamic,
                block_size: Some([128, 128]),
            }
        ));
    }

    #[test]
    fn parses_bitsandbytes_nf4() {
        let v = json(
            r#"{
                "quantization_config": {
                    "quant_method": "bitsandbytes",
                    "load_in_4bit": true,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": true,
                    "llm_int8_skip_modules": ["lm_head"]
                }
            }"#,
        );
        let qc = QuantizationConfig::parse(&v).unwrap().expect("some");
        assert!(matches!(
            qc.method,
            QuantMethod::Bnb4 {
                quant_type: BnbQuantType::NF4,
                blocksize: 64,
            }
        ));
    }

    #[test]
    fn parses_bitsandbytes_fp4() {
        let v = json(
            r#"{
                "quantization_config": {
                    "quant_method": "bitsandbytes",
                    "load_in_4bit": true,
                    "bnb_4bit_quant_type": "fp4"
                }
            }"#,
        );
        let qc = QuantizationConfig::parse(&v).unwrap().unwrap();
        assert!(matches!(
            qc.method,
            QuantMethod::Bnb4 {
                quant_type: BnbQuantType::FP4,
                ..
            }
        ));
    }

    #[test]
    fn rejects_bitsandbytes_8bit() {
        let v = json(
            r#"{
                "quantization_config": {
                    "quant_method": "bitsandbytes",
                    "load_in_4bit": false
                }
            }"#,
        );
        assert!(matches!(
            QuantizationConfig::parse(&v),
            Err(ParseError::UnsupportedMethod(_))
        ));
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
